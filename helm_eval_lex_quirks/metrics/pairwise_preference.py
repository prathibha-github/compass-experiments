"""Pairwise preference analysis for tic-suite outputs."""
import json
import os
from collections import defaultdict
from itertools import combinations
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


def _record_key(record: dict) -> Tuple[Any, Any, Any]:
    return (
        record.get("condition"),
        record.get("prompt_id", record.get("prompt_idx")),
        record.get("sample_idx"),
    )


def _record_task_type(record: dict) -> Optional[str]:
    return record.get("task_type")


def _record_score(record: dict) -> float:
    detectors = record.get("detectors", {})
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
            "matches": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
    return bucket


def _win_summary(left_wins: int, right_wins: int, ties: int) -> dict:
    matches = left_wins + right_wins + ties
    return {
        "matches": matches,
        "wins": left_wins,
        "losses": right_wins,
        "ties": ties,
        "win_rate": round(100 * left_wins / matches, 1) if matches else 0.0,
        "preference_score": round(left_wins + 0.5 * ties, 2),
    }


def _rank_bucket(bucket: Dict[str, dict]) -> List[dict]:
    ranking = []
    for model, stats in bucket.items():
        ranking.append(
            {
                "model": model,
                "preference_score": round(stats["score"], 2),
                "matches": stats["matches"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
            }
        )
    ranking.sort(key=lambda row: (row["preference_score"], row["matches"]), reverse=True)
    return ranking


def analyze_pairwise_preference(results_by_model: Dict[str, Any]) -> Dict[str, dict]:
    """
    Compare models head-to-head on shared tic-suite checkpoint records.

    Lower detector score wins. The score is the sum of detector counts for the
    record, with suite-specific fallbacks for older LinkedIn checkpoints.
    """
    suite_names = set()
    for result in results_by_model.values():
        suite_names.update(getattr(result, "tic_results", {}).keys())

    pairwise_results: Dict[str, dict] = {}
    model_names = list(results_by_model.keys())

    for suite_name in sorted(suite_names):
        suite_records = _load_suite_records(results_by_model, suite_name)
        participating_models = [m for m in model_names if m in suite_records]
        if len(participating_models) < 2:
            continue

        overall_bucket = _init_bucket(participating_models)
        task_buckets: DefaultDict[str, Dict[str, dict]] = defaultdict(
            lambda: _init_bucket(participating_models)
        )
        pairwise_pairs: Dict[str, dict] = {}
        task_pairwise_pairs: DefaultDict[str, Dict[str, dict]] = defaultdict(dict)

        for left_model, right_model in combinations(participating_models, 2):
            left_records = suite_records[left_model]
            right_records = suite_records[right_model]
            shared_keys = sorted(set(left_records) & set(right_records))
            if not shared_keys:
                continue

            left_wins = right_wins = ties = 0
            per_task: DefaultDict[str, List[int]] = defaultdict(lambda: [0, 0, 0])

            for key in shared_keys:
                left_record = left_records[key]
                right_record = right_records[key]
                task_type = _record_task_type(left_record) or _record_task_type(right_record) or "general"
                left_score = _record_score(left_record)
                right_score = _record_score(right_record)

                if left_score < right_score:
                    left_wins += 1
                    per_task[task_type][0] += 1
                elif right_score < left_score:
                    right_wins += 1
                    per_task[task_type][1] += 1
                else:
                    ties += 1
                    per_task[task_type][2] += 1

            pair_key = f"{left_model}::{right_model}"
            pairwise_pairs[pair_key] = _win_summary(left_wins, right_wins, ties)

            overall_bucket[left_model]["wins"] += left_wins
            overall_bucket[left_model]["losses"] += right_wins
            overall_bucket[left_model]["ties"] += ties
            overall_bucket[left_model]["matches"] += left_wins + right_wins + ties
            overall_bucket[left_model]["score"] += left_wins + 0.5 * ties

            overall_bucket[right_model]["wins"] += right_wins
            overall_bucket[right_model]["losses"] += left_wins
            overall_bucket[right_model]["ties"] += ties
            overall_bucket[right_model]["matches"] += left_wins + right_wins + ties
            overall_bucket[right_model]["score"] += right_wins + 0.5 * ties

            for task_type, counts in per_task.items():
                left_task_wins, right_task_wins, task_ties = counts
                task_pairwise_pairs[task_type][pair_key] = _win_summary(
                    left_task_wins, right_task_wins, task_ties
                )
                task_bucket = task_buckets[task_type]
                task_bucket[left_model]["wins"] += left_task_wins
                task_bucket[left_model]["losses"] += right_task_wins
                task_bucket[left_model]["ties"] += task_ties
                task_bucket[left_model]["matches"] += sum(counts)
                task_bucket[left_model]["score"] += left_task_wins + 0.5 * task_ties

                task_bucket[right_model]["wins"] += right_task_wins
                task_bucket[right_model]["losses"] += left_task_wins
                task_bucket[right_model]["ties"] += task_ties
                task_bucket[right_model]["matches"] += sum(counts)
                task_bucket[right_model]["score"] += right_task_wins + 0.5 * task_ties

        if not pairwise_pairs:
            continue

        pairwise_results[suite_name] = {
            "models": participating_models,
            "overall": {
                "pairs": pairwise_pairs,
                "ranking": _rank_bucket(overall_bucket),
            },
            "task_types": {
                task_type: {
                    "pairs": pairwise_pairs_for_task,
                    "ranking": _rank_bucket(task_buckets[task_type]),
                }
                for task_type, pairwise_pairs_for_task in task_pairwise_pairs.items()
            },
        }

    return pairwise_results
