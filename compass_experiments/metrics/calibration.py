from collections.abc import Mapping
from numbers import Real
from typing import Dict, List, Tuple

import numpy as np

CONFIDENCE_SCORING_VERSION = 2

# Token strings the API might return for each label
_LABEL_TOKENS: Dict[str, List[str]] = {
    "A": ["A", " A", "a", " a"],
    "B": ["B", " B", "b", " b"],
    "C": ["C", " C", "c", " c"],
    "D": ["D", " D", "d", " d"],
}


def serialize_top_logprobs(top_logprobs) -> List[dict]:
    """Keep the first-token candidates needed to recompute confidence offline."""
    records = []
    for entry in top_logprobs or []:
        token = entry.get("token") if isinstance(entry, Mapping) else entry.token
        logprob = entry.get("logprob") if isinstance(entry, Mapping) else entry.logprob
        if (not isinstance(token, str) or isinstance(logprob, bool)
                or not isinstance(logprob, Real) or not np.isfinite(logprob) or logprob > 0):
            raise ValueError("Token log probabilities must be finite, nonpositive numbers")
        records.append({"token": token, "logprob": float(logprob)})
    return records


def extract_choice_probs(
    top_logprobs,
    valid_choices: List[str],
) -> Tuple[Dict[str, float], float]:
    """Aggregate observed token spellings, then normalize across answer labels.

    Confidence is conditional on observed answer-token mass, not the full token
    vocabulary. Coverage counts labels with an observed spelling; even coverage
    of 1 does not mean every spelling was returned in the API's top-k list.
    Missing labels retain the -10 fallback and must be excluded from ECE.
    Accept SDK token objects or the serialized dictionaries saved in checkpoints.
    """
    if not valid_choices or len(set(valid_choices)) != len(valid_choices):
        raise ValueError("Choice labels must be nonempty and unique")
    tokens = {}
    for entry in serialize_top_logprobs(top_logprobs):
        token, logprob = entry["token"], entry["logprob"]
        if token in tokens and tokens[token] != logprob:
            raise ValueError(f"Conflicting probabilities for duplicate token {token!r}")
        tokens[token] = logprob  # A repeated candidate is not extra probability mass.

    choice_logs = {}
    for choice in valid_choices:
        spellings = _LABEL_TOKENS.get(choice, [])
        values = [tokens[token] for token in sorted(tokens) if token in spellings]
        if values:
            choice_logs[choice] = float(np.logaddexp.reduce(values))
    coverage = len(choice_logs) / len(valid_choices)
    log_values = np.array([choice_logs.get(choice, -10.0) for choice in valid_choices])
    log_values -= log_values.max()
    probabilities = np.exp(log_values)
    probabilities /= probabilities.sum()
    return dict(zip(valid_choices, map(float, probabilities))), coverage


def compute_ece(confidences: List[float], correct_flags: List[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    if len(confidences) != len(correct_flags):
        raise ValueError("Each confidence must have a corresponding correctness label")
    if type(n_bins) is not int or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    if any(not np.isfinite(p) or not 0.0 <= p <= 1.0 for p in confidences):
        raise ValueError("Confidences must be finite values between 0 and 1")
    if not confidences:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n    = len(confidences)
    ece  = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Include 1.0 in the last bin; shared boundaries belong to the next bin.
        mask        = [lo <= p and (p < hi or i == n_bins - 1) for p in confidences]
        bucket_conf = [p for p, m in zip(confidences,   mask) if m]
        bucket_corr = [c for c, m in zip(correct_flags, mask) if m]
        if not bucket_conf:
            continue
        ece += (len(bucket_conf) / n) * abs(
            sum(bucket_conf) / len(bucket_conf)
            - sum(bucket_corr) / len(bucket_corr)
        )
    return float(ece)
