"""Persona-transfer analysis over tic-suite summaries."""
from typing import Dict, List

from helm_eval_lex_quirks.metrics.tic_suite import TicSuite


def _detector_transfer_score(pct_delta: float, mean_delta: float) -> float:
    """Combine prevalence and intensity deltas into a sortable score."""
    return round(max(0.0, pct_delta) + 10 * max(0.0, mean_delta), 2)


def analyze_persona_transfer(
    suite_results: Dict[str, dict],
    suites_by_name: Dict[str, TicSuite],
) -> Dict[str, dict]:
    """
    Compare persona/style conditions against each suite baseline.

    This measures whether a persona condition increases tic prevalence on the
    same neutral task prompts. It does not claim persistent cross-session model
    state; public API calls are stateless unless the caller supplies history.
    """
    transfer_results: Dict[str, dict] = {}

    for suite_name, result in suite_results.items():
        suite = suites_by_name.get(suite_name)
        analysis = result.get("analysis", {})
        if not suite or not analysis:
            continue

        baseline = suite.baseline_condition
        if baseline not in analysis:
            baseline = next(iter(analysis))
        baseline_stats = analysis[baseline]

        persona_conditions = {}
        for condition, stats in analysis.items():
            if condition == baseline:
                continue

            detector_rows: List[dict] = []
            for detector_name, detector_stats in stats.get("detectors", {}).items():
                base_detector = baseline_stats.get("detectors", {}).get(detector_name, {})
                base_pct = float(base_detector.get("pct_hit", 0.0))
                persona_pct = float(detector_stats.get("pct_hit", 0.0))
                base_mean = float(base_detector.get("mean_count", 0.0))
                persona_mean = float(detector_stats.get("mean_count", 0.0))
                pct_delta = round(persona_pct - base_pct, 1)
                mean_delta = round(persona_mean - base_mean, 2)
                mean_lift = round((persona_mean + 0.1) / (base_mean + 0.1), 2)

                detector_rows.append(
                    {
                        "detector": detector_name,
                        "baseline_pct_hit": base_pct,
                        "persona_pct_hit": persona_pct,
                        "pct_hit_delta": pct_delta,
                        "baseline_mean_count": base_mean,
                        "persona_mean_count": persona_mean,
                        "mean_count_delta": mean_delta,
                        "mean_count_lift": mean_lift,
                        "transfer_score": _detector_transfer_score(pct_delta, mean_delta),
                    }
                )

            detector_rows.sort(key=lambda row: row["transfer_score"], reverse=True)
            persona_conditions[condition] = {
                "n_outputs": stats.get("n_outputs", 0),
                "top_detectors": detector_rows[:10],
                "transfer_score": round(
                    sum(row["transfer_score"] for row in detector_rows), 2
                ),
            }

        transfer_results[suite_name] = {
            "baseline_condition": baseline,
            "persona_conditions": persona_conditions,
        }

    return transfer_results
