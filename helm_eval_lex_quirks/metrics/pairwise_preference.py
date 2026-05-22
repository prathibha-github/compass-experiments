"""Pairwise preference analysis for tic-suite outputs."""
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


PAIRWISE_METHODS = {"count", "bradley_terry"}
PAIRWISE_SEGMENTS = {"none", "task_type", "condition"}


def _record_key(record: dict) -> Tuple[Any, Any, Any]:
    return (
        record.get("condition"),
        record.get("prompt_id", record.get("prompt_idx")),
        record.get("sample_idx"),
    )


def _record_task_type(record: dict) -> Optional[str]:
    return record.get("task_type")


def _parse_created_at(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _record_score(record: dict, detector_name: Optional[str] = None) -> Optional[float]:
    detectors = record.get("detectors", {})
    if detector_name:
        if not isinstance(detectors, dict):
            return None
        detector = detectors.get(detector_name)
        if not isinstance(detector, dict) or "count" not in detector:
            return None
        return float(detector.get("count", 0.0))

    if isinstance(detectors, dict) and detectors:
        first = next(iter(detectors.values()))
        if isinstance(first, dict) and "count" in first:
            return float(
                sum(float(detector.get("count", 0.0)) for detector in detectors.values())
            )
    if "linkedin_score" in record:
        return float(record.get("linkedin_score", 0.0))
    if "emoji_count" in record or "phrase_hits" in record:
        return float(record.get("emoji_count", 0.0) + record.get("phrase_hits", 0.0))
    return 0.0


def _load_records(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []

    records: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_suite_records(results_by_model: Dict[str, Any], suite_name: str) -> Dict[str, Dict]:
    suite_records: Dict[str, Dict] = {}
    for model_name, result in results_by_model.items():
        suite_result = getattr(result, "tic_results", {}).get(suite_name)
        if not suite_result:
            continue
        path = suite_result.get("checkpoint_path")
        if not path:
            continue
        records = _load_records(path)
        suite_records[model_name] = {
            _record_key(record): record for record in records
        }
    return suite_records


def _init_bucket(models: Iterable[str]) -> Dict[str, dict]:
    bucket: Dict[str, dict] = {}
    for model in models:
        bucket[model] = {
            "score": 0.0,
            "matches": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "ties": 0.0,
        }
    return bucket


def _round_metric(value: float) -> float:
    return round(value, 3)


def _win_summary(left_wins: float, right_wins: float, ties: float) -> dict:
    matches = left_wins + right_wins + ties
    return {
        "matches": _round_metric(matches),
        "wins": _round_metric(left_wins),
        "losses": _round_metric(right_wins),
        "ties": _round_metric(ties),
        "win_rate": round(100 * left_wins / matches, 1) if matches else 0.0,
        "preference_score": _round_metric(left_wins + 0.5 * ties),
    }


def _rank_bucket(bucket: Dict[str, dict]) -> List[dict]:
    ranking = []
    for model, stats in bucket.items():
        if stats["matches"] <= 0:
            continue
        ranking.append(
            {
                "model": model,
                "preference_score": round(stats["score"], 3),
                "matches": _round_metric(stats["matches"]),
                "wins": _round_metric(stats["wins"]),
                "losses": _round_metric(stats["losses"]),
                "ties": _round_metric(stats["ties"]),
            }
        )
    ranking.sort(key=lambda row: (row["preference_score"], row["matches"]), reverse=True)
    return ranking


def _comparison_weight(
    left_record: dict,
    right_record: dict,
    halflife_days: Optional[int],
    now: datetime,
) -> float:
    if not halflife_days:
        return 1.0

    left_ts = _parse_created_at(left_record.get("created_at"))
    right_ts = _parse_created_at(right_record.get("created_at"))
    valid = [ts for ts in (left_ts, right_ts) if ts is not None]
    if not valid:
        return 1.0

    ages = []
    for ts in valid:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
        ages.append(age_days)
    mean_age_days = sum(ages) / len(ages)
    return 0.5 ** (mean_age_days / float(halflife_days))


def _segment_key(left_record: dict, right_record: dict, segment_by: str) -> str:
    if segment_by == "none":
        return "overall"
    if segment_by == "task_type":
        return _record_task_type(left_record) or _record_task_type(right_record) or "general"
    return (
        left_record.get("condition")
        or right_record.get("condition")
        or "unknown"
    )


def _build_bt_ranking(
    models: List[str],
    comparisons: Dict[Tuple[str, str], float],
    wins: Dict[Tuple[str, str], float],
    bucket: Optional[Dict[str, dict]] = None,
    max_iters: int = 200,
    tol: float = 1e-8,
) -> List[dict]:
    if len(models) < 2:
        return []

    strengths = {model: 1.0 for model in models}
    eps = 1e-12

    for _ in range(max_iters):
        updated: Dict[str, float] = {}
        max_delta = 0.0
        for model in models:
            win_total = sum(wins.get((model, other), 0.0) for other in models if other != model)
            denom = 0.0
            for other in models:
                if other == model:
                    continue
                n_ij = comparisons.get((model, other), 0.0)
                if n_ij <= 0:
                    continue
                denom += n_ij / max(strengths[model] + strengths[other], eps)
            new_strength = max(win_total / denom, eps) if denom > 0 else eps
            updated[model] = new_strength
            max_delta = max(max_delta, abs(new_strength - strengths[model]))

        # Normalize to keep values numerically stable.
        norm = sum(updated.values()) / len(updated)
        if norm > 0:
            for model in updated:
                updated[model] /= norm

        strengths = updated
        if max_delta < tol:
            break

    ranking: List[dict] = []
    for model in models:
        total_matches = sum(comparisons.get((model, other), 0.0) for other in models if other != model)
        if total_matches <= 0:
            continue
        expected = 0.0
        expected_weight = 0.0
        for other in models:
            if other == model:
                continue
            n_ij = comparisons.get((model, other), 0.0)
            if n_ij <= 0:
                continue
            p_win = strengths[model] / max(strengths[model] + strengths[other], eps)
            expected += p_win * n_ij
            expected_weight += n_ij
        expected_win_rate = (expected / expected_weight) if expected_weight > 0 else 0.0
        ranking.append(
            {
                "model": model,
                "preference_score": round(expected_win_rate, 4),
                "bt_score": round(math.log(max(strengths[model], eps)), 4),
                "matches": _round_metric(
                    bucket.get(model, {}).get("matches", total_matches)
                    if bucket else total_matches
                ),
                "wins": _round_metric(bucket.get(model, {}).get("wins", 0.0)) if bucket else 0.0,
                "losses": _round_metric(bucket.get(model, {}).get("losses", 0.0)) if bucket else 0.0,
                "ties": _round_metric(bucket.get(model, {}).get("ties", 0.0)) if bucket else 0.0,
            }
        )

    ranking.sort(key=lambda row: (row["preference_score"], row.get("bt_score", 0.0)), reverse=True)
    return ranking


def analyze_pairwise_preference(
    results_by_model: Dict[str, Any],
    detector_name: Optional[str] = None,
    method: str = "count",
    segment_by: str = "task_type",
    time_decay_halflife_days: Optional[int] = None,
) -> Dict[str, dict]:
    """
    Compare models head-to-head on shared tic-suite checkpoint records.

    Lower detector score wins. By default, the score is the sum of detector
    counts for the record, with suite-specific fallbacks for older LinkedIn
    checkpoints. If detector_name is set, only that detector count is used.
    """
    if method not in PAIRWISE_METHODS:
        raise ValueError(f"Unknown pairwise method: {method}")
    if segment_by not in PAIRWISE_SEGMENTS:
        raise ValueError(f"Unknown pairwise segmentation: {segment_by}")

    suite_names = set()
    for result in results_by_model.values():
        suite_names.update(getattr(result, "tic_results", {}).keys())

    pairwise_results: Dict[str, dict] = {}
    model_names = list(results_by_model.keys())
    now = datetime.now(timezone.utc)

    for suite_name in sorted(suite_names):
        suite_records = _load_suite_records(results_by_model, suite_name)
        participating_models = [m for m in model_names if m in suite_records]
        if len(participating_models) < 2:
            continue

        overall_bucket = _init_bucket(participating_models)
        segment_buckets: DefaultDict[str, Dict[str, dict]] = defaultdict(
            lambda: _init_bucket(participating_models)
        )

        comparisons: Dict[Tuple[str, str], float] = defaultdict(float)
        wins: Dict[Tuple[str, str], float] = defaultdict(float)
        segment_comparisons: DefaultDict[str, Dict[Tuple[str, str], float]] = defaultdict(
            lambda: defaultdict(float)
        )
        segment_wins: DefaultDict[str, Dict[Tuple[str, str], float]] = defaultdict(
            lambda: defaultdict(float)
        )

        pairwise_pairs: Dict[str, dict] = {}
        segmented_pairs: DefaultDict[str, Dict[str, dict]] = defaultdict(dict)

        for left_model, right_model in combinations(participating_models, 2):
            left_records = suite_records[left_model]
            right_records = suite_records[right_model]
            shared_keys = sorted(set(left_records) & set(right_records))
            if not shared_keys:
                continue

            left_wins = right_wins = ties = 0.0
            per_segment: DefaultDict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

            for key in shared_keys:
                left_record = left_records[key]
                right_record = right_records[key]
                left_score = _record_score(left_record, detector_name)
                right_score = _record_score(right_record, detector_name)
                if left_score is None or right_score is None:
                    continue

                weight = _comparison_weight(
                    left_record,
                    right_record,
                    time_decay_halflife_days,
                    now,
                )
                segment = _segment_key(left_record, right_record, segment_by)

                if left_score < right_score:
                    left_wins += weight
                    per_segment[segment][0] += weight
                    wins[(left_model, right_model)] += weight
                    segment_wins[segment][(left_model, right_model)] += weight
                elif right_score < left_score:
                    right_wins += weight
                    per_segment[segment][1] += weight
                    wins[(right_model, left_model)] += weight
                    segment_wins[segment][(right_model, left_model)] += weight
                else:
                    ties += weight
                    per_segment[segment][2] += weight
                    wins[(left_model, right_model)] += 0.5 * weight
                    wins[(right_model, left_model)] += 0.5 * weight
                    segment_wins[segment][(left_model, right_model)] += 0.5 * weight
                    segment_wins[segment][(right_model, left_model)] += 0.5 * weight

                comparisons[(left_model, right_model)] += weight
                comparisons[(right_model, left_model)] += weight
                segment_comparisons[segment][(left_model, right_model)] += weight
                segment_comparisons[segment][(right_model, left_model)] += weight

            matches = left_wins + right_wins + ties
            if not matches:
                continue

            pair_key = f"{left_model}::{right_model}"
            pairwise_pairs[pair_key] = _win_summary(left_wins, right_wins, ties)

            overall_bucket[left_model]["wins"] += left_wins
            overall_bucket[left_model]["losses"] += right_wins
            overall_bucket[left_model]["ties"] += ties
            overall_bucket[left_model]["matches"] += matches
            overall_bucket[left_model]["score"] += left_wins + 0.5 * ties

            overall_bucket[right_model]["wins"] += right_wins
            overall_bucket[right_model]["losses"] += left_wins
            overall_bucket[right_model]["ties"] += ties
            overall_bucket[right_model]["matches"] += matches
            overall_bucket[right_model]["score"] += right_wins + 0.5 * ties

            for segment, counts in per_segment.items():
                left_segment_wins, right_segment_wins, segment_ties = counts
                segment_matches = sum(counts)
                if segment_matches <= 0:
                    continue
                segmented_pairs[segment][pair_key] = _win_summary(
                    left_segment_wins,
                    right_segment_wins,
                    segment_ties,
                )
                segment_bucket = segment_buckets[segment]
                segment_bucket[left_model]["wins"] += left_segment_wins
                segment_bucket[left_model]["losses"] += right_segment_wins
                segment_bucket[left_model]["ties"] += segment_ties
                segment_bucket[left_model]["matches"] += segment_matches
                segment_bucket[left_model]["score"] += left_segment_wins + 0.5 * segment_ties

                segment_bucket[right_model]["wins"] += right_segment_wins
                segment_bucket[right_model]["losses"] += left_segment_wins
                segment_bucket[right_model]["ties"] += segment_ties
                segment_bucket[right_model]["matches"] += segment_matches
                segment_bucket[right_model]["score"] += right_segment_wins + 0.5 * segment_ties

        if not pairwise_pairs:
            continue

        active_models = [
            model for model in participating_models
            if overall_bucket[model]["matches"] > 0
        ]
        if method == "bradley_terry":
            overall_ranking = _build_bt_ranking(
                active_models,
                comparisons,
                wins,
                bucket=overall_bucket,
            )
        else:
            overall_ranking = _rank_bucket(overall_bucket)

        segment_output: Dict[str, dict] = {}
        if segment_by != "none":
            for segment, segment_pairs in segmented_pairs.items():
                if method == "bradley_terry":
                    ranking = _build_bt_ranking(
                        active_models,
                        segment_comparisons[segment],
                        segment_wins[segment],
                        bucket=segment_buckets[segment],
                    )
                else:
                    ranking = _rank_bucket(segment_buckets[segment])
                if not ranking:
                    continue
                segment_output[segment] = {
                    "pairs": segment_pairs,
                    "ranking": ranking,
                }

        pairwise_results[suite_name] = {
            "models": active_models,
            "scoring": {
                "mode": "single_detector" if detector_name else "sum_detector_counts",
                "detector": detector_name,
                "pairwise_method": method,
                "segment_by": segment_by,
                "time_decay_halflife_days": time_decay_halflife_days,
            },
            "overall": {
                "pairs": pairwise_pairs,
                "ranking": overall_ranking,
            },
            "segments": segment_output,
            # Backward-compatible alias used in existing report/tests.
            "task_types": segment_output if segment_by == "task_type" else {},
        }

    return pairwise_results
