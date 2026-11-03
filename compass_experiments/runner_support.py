import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from compass_experiments.metrics.accuracy import multiple_choice_accuracy
from compass_experiments.metrics.calibration import compute_ece
from compass_experiments.metrics.robustness import compute_robustness_score
from compass_experiments.metrics.tic_suite import TicSuite


TIC_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class ScenarioResults:
    name: str
    accuracy: float
    ece: Optional[float]
    ece_coverage: float
    robustness_score: Optional[float]
    n_completed: int
    n_requested: int
    checkpoint_path: str
    ambig_unknown_rate: Optional[float] = None
    disambig_accuracy: Optional[float] = None


def load_scenario_checkpoint(path: str, logger: logging.Logger) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec["id"] in done:
                    raise ValueError(f"Duplicate scenario identity at {path}:{line_number}")
                done[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid scenario checkpoint at {path}:{line_number}") from exc
    if done:
        logger.info("    Resumed %d checkpointed instances from %s", len(done), path)
    return done


def aggregate_scenario_results(
    name: str,
    records: List[dict],
    n_requested: int,
    checkpoint_path: str,
    logger: logging.Logger,
) -> ScenarioResults:
    if not records:
        return ScenarioResults(
            name=name,
            accuracy=0.0,
            ece=None,
            ece_coverage=0.0,
            robustness_score=None,
            n_completed=0,
            n_requested=n_requested,
            checkpoint_path=checkpoint_path,
        )

    predictions = [r["prediction"] for r in records]
    references = [r["correct"] for r in records]
    confidences = [r["confidence"] for r in records]
    correct_flags = [r["is_correct"] for r in records]
    coverages = [r.get("coverage", 1.0) for r in records]
    perturbed_ok = [
        r["perturbed_is_correct"]
        for r in records
        if r.get("perturbed_is_correct") is not None
    ]

    accuracy = multiple_choice_accuracy(predictions, references)
    full_cov_pairs = [
        (c, f)
        for c, f, cov in zip(confidences, correct_flags, coverages)
        if cov >= 1.0
    ]
    ece_coverage = len(full_cov_pairs) / len(records) if records else 0.0
    if full_cov_pairs:
        ece_confs, ece_flags = zip(*full_cov_pairs)
        ece = compute_ece(list(ece_confs), list(ece_flags))
    else:
        ece = None
        logger.warning(
            "%s: no instances had full logprob coverage — ECE cannot be computed", name
        )
    if 0 < ece_coverage < 1.0:
        n_low = len(records) - len(full_cov_pairs)
        logger.warning(
            "%s: %d/%d instances had incomplete logprob coverage and were excluded from ECE",
            name,
            n_low,
            len(records),
        )

    rob_score = None
    if perturbed_ok:
        perturbed_acc = sum(perturbed_ok) / len(perturbed_ok)
        rob_score = compute_robustness_score(accuracy, perturbed_acc)

    sr = ScenarioResults(
        name=name,
        accuracy=round(accuracy, 4),
        ece=round(ece, 4) if ece is not None else None,
        ece_coverage=round(ece_coverage, 4),
        robustness_score=round(rob_score, 4) if rob_score is not None else None,
        n_completed=len(records),
        n_requested=n_requested,
        checkpoint_path=checkpoint_path,
    )

    ambig = [
        r for r in records
        if r.get("metadata", {}).get("context_condition") == "ambig"
    ]
    disambig = [
        r for r in records
        if r.get("metadata", {}).get("context_condition") == "disambig"
    ]
    if ambig:
        sr.ambig_unknown_rate = round(sum(r["is_correct"] for r in ambig) / len(ambig), 4)
    if disambig:
        sr.disambig_accuracy = round(sum(r["is_correct"] for r in disambig) / len(disambig), 4)

    return sr


def extract_judge_disagreements(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    detectors = record.get("detectors", {})
    if not isinstance(detectors, dict):
        return rows

    for detector_name, detector in detectors.items():
        if not isinstance(detector, dict):
            continue
        metadata = detector.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if "secondary_score" not in metadata:
            continue

        rows.append(
            {
                "condition": record.get("condition", "unknown"),
                "task_type": record.get("task_type", "general"),
                "prompt_id": record.get("prompt_id"),
                "sample_idx": record.get("sample_idx"),
                "detector": detector_name,
                "primary_score": float(detector.get("score", 0.0)),
                "secondary_score": float(metadata.get("secondary_score", 0.0)),
                "score_delta": float(metadata.get("score_delta", 0.0)),
                "agreement": bool(metadata.get("agreement", False)),
            }
        )
    return rows


def summarize_judge_disagreements(
    rows: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    def _init_bucket() -> Dict[str, float]:
        return {"n": 0.0, "agree": 0.0, "delta_sum": 0.0, "high_disagreement": 0.0}

    def _update(bucket: Dict[str, float], row: Dict[str, Any]) -> None:
        bucket["n"] += 1.0
        if row.get("agreement"):
            bucket["agree"] += 1.0
        delta = float(row.get("score_delta", 0.0))
        bucket["delta_sum"] += delta
        if delta >= threshold:
            bucket["high_disagreement"] += 1.0

    def _finalize(bucket: Dict[str, float]) -> Dict[str, float]:
        n = bucket["n"]
        if n <= 0:
            return {
                "n": 0,
                "agreement_pct": 0.0,
                "mean_abs_score_delta": 0.0,
                "high_disagreement_pct": 0.0,
            }
        return {
            "n": int(n),
            "agreement_pct": round(100.0 * bucket["agree"] / n, 1),
            "mean_abs_score_delta": round(bucket["delta_sum"] / n, 4),
            "high_disagreement_pct": round(100.0 * bucket["high_disagreement"] / n, 1),
        }

    overall = _init_bucket()
    by_condition: DefaultDict[str, Dict[str, float]] = defaultdict(_init_bucket)
    by_task_type: DefaultDict[str, Dict[str, float]] = defaultdict(_init_bucket)

    for row in rows:
        _update(overall, row)
        _update(by_condition[row.get("condition", "unknown")], row)
        _update(by_task_type[row.get("task_type", "general")], row)

    top_disagreements = sorted(
        rows,
        key=lambda row: row.get("score_delta", 0.0),
        reverse=True,
    )[:10]

    return {
        "threshold": threshold,
        "overall": _finalize(overall),
        "conditions": {
            key: _finalize(bucket)
            for key, bucket in sorted(by_condition.items())
        },
        "task_types": {
            key: _finalize(bucket)
            for key, bucket in sorted(by_task_type.items())
        },
        "top_disagreements": top_disagreements,
    }


def record_task_type(rec: dict, suite: TicSuite) -> str:
    task_type = rec.get("task_type")
    if task_type:
        return task_type

    prompt_id = rec.get("prompt_id")
    if prompt_id:
        for prompt in suite.prompts:
            if prompt.id == prompt_id:
                return prompt.task_type

    prompt_idx = rec.get("prompt_idx")
    if isinstance(prompt_idx, int) and 0 <= prompt_idx < len(suite.prompts):
        return suite.prompts[prompt_idx].task_type

    return "general"


def load_tic_checkpoint(
    path: str,
    suite: TicSuite,
    logger: logging.Logger,
) -> Tuple[Dict[tuple, dict], Dict[str, List[str]]]:
    done: Dict[tuple, dict] = {}
    condition_outputs: Dict[str, List[str]] = {c.name: [] for c in suite.conditions}
    valid_conditions = set(condition_outputs)

    if not os.path.exists(path):
        return done, condition_outputs

    skipped = 0
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                logger.warning("Skipping corrupt tic checkpoint line %s:%d", path, line_no)
                continue

            version = rec.get("schema_version", 0)
            if version not in (0, TIC_CHECKPOINT_SCHEMA_VERSION):
                skipped += 1
                logger.warning(
                    "Skipping unsupported tic checkpoint schema v%s at %s:%d",
                    version,
                    path,
                    line_no,
                )
                continue

            try:
                condition = rec["condition"]
                key = (condition, rec["prompt_idx"], rec["sample_idx"])
                completion = rec["completion"]
            except KeyError as exc:
                skipped += 1
                logger.warning(
                    "Skipping malformed tic checkpoint line %s:%d missing %s",
                    path,
                    line_no,
                    exc,
                )
                continue

            if condition not in valid_conditions:
                skipped += 1
                logger.warning(
                    "Skipping tic checkpoint line %s:%d with unknown condition %r",
                    path,
                    line_no,
                    condition,
                )
                continue

            if version == TIC_CHECKPOINT_SCHEMA_VERSION and rec.get("suite") != suite.name:
                skipped += 1
                logger.warning(
                    "Skipping tic checkpoint line %s:%d for suite %r while loading %r",
                    path,
                    line_no,
                    rec.get("suite"),
                    suite.name,
                )
                continue

            if key in done:
                raise ValueError(f"Duplicate style checkpoint identity {key} at {path}:{line_no}")
            done[key] = rec
            condition_outputs[condition].append(completion)

    if done:
        logger.info("    Resumed %d checkpointed %s records from %s", len(done), suite.name, path)
    if skipped:
        logger.warning("    Skipped %d incompatible %s checkpoint records", skipped, suite.name)
    return done, condition_outputs
