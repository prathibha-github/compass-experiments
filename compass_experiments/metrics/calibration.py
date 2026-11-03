from typing import Dict, List, Optional, Tuple

import numpy as np

# Token strings the API might return for each label
_LABEL_TOKENS: Dict[str, List[str]] = {
    "A": ["A", " A", "a", " a"],
    "B": ["B", " B", "b", " b"],
    "C": ["C", " C", "c", " c"],
    "D": ["D", " D", "d", " d"],
}


def extract_choice_probs(
    top_logprobs,
    valid_choices: List[str],
) -> Tuple[Dict[str, float], float]:
    """
    Convert top_logprobs from OpenAI (at token-0) into probabilities for each
    valid choice.

    Returns
    -------
    probs    : dict mapping each choice label to its normalised probability
    coverage : fraction of valid choice labels actually found in top_logprobs
               (< 1.0 means some choices were filled with -10 fallback log-prob,
               making the resulting probabilities unreliable for ECE)
    """
    if not top_logprobs:
        uniform = 1.0 / len(valid_choices)
        return {c: uniform for c in valid_choices}, 0.0

    lp_map: Dict[str, float] = {}
    for lp in top_logprobs:
        tok = lp.token
        for choice in valid_choices:
            if choice not in lp_map and tok in _LABEL_TOKENS.get(choice, []):
                lp_map[choice] = lp.logprob

    found    = set(lp_map)
    coverage = len(found) / len(valid_choices)

    # Fill any unseen choice with a low log-prob so the softmax still sums to 1,
    # but callers should check coverage < 1.0 before trusting ECE.
    for c in valid_choices:
        if c not in lp_map:
            lp_map[c] = -10.0

    log_vals = np.array([lp_map[c] for c in valid_choices], dtype=float)
    log_vals -= log_vals.max()
    probs    = np.exp(log_vals)
    probs   /= probs.sum()
    return {c: float(p) for c, p in zip(valid_choices, probs)}, coverage


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
