import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import yaml

from compass_experiments.metrics.builtin_tic_suites import STYLE_SUITE_NOTES
from compass_experiments.runner import EvalResults


def _ci_text(stats: dict, prefix: str = "pct_hit") -> str:
    low = stats.get(f"{prefix}_ci_low")
    high = stats.get(f"{prefix}_ci_high")
    if low is None or high is None:
        return ""
    return f"[{low:.1f},{high:.1f}]"


def _prompt_text(stats):
    summary = stats.get("prompt_statistics", {}).get("all")
    if summary is None:
        return "prompts=unknown prompt-ci=N/A"
    interval = summary["interval"]
    ci = f"[{100*interval[0]:.1f},{100*interval[1]:.1f}]" if interval else "N/A"
    return f"prompts={summary['distinct_prompts']} prompt-ci={ci}"


def _prompt_uncertain(row):
    summary = row.get("prompt_statistics", {}).get("all")
    if summary is None or summary["distinct_prompts"] < 20 or summary["interval"] is None:
        return True
    low, high = summary["interval"]
    # A degenerate bootstrap is not evidence that unseen prompts have no risk.
    return high - low >= .4 or low == high


def _sample_flag(stats: dict) -> str:
    return " low-n" if stats.get("low_sample") else ""


def _top_detector(stats: dict) -> Tuple[str, Dict[str, Any]]:
    detectors = stats.get("detectors", {})
    if not detectors:
        return ("", {})
    return max(
        detectors.items(),
        key=lambda item: (item[1].get("pct_hit", 0.0), item[1].get("mean_count", 0.0)),
    )


def _pairwise_results(results_by_model: Dict[str, EvalResults]) -> Dict[str, dict]:
    for res in results_by_model.values():
        if res.pairwise_preference_results:
            return res.pairwise_preference_results
    return {}


def _summary_dict(res: EvalResults) -> Dict[str, Any]:
    """Aggregate metrics only — per-instance records live in checkpoint JSONL files."""
    out = {
        "model":          res.model,
        "total_cost_usd": round(res.total_cost_usd, 6),
        "token_usage":    res.token_usage,
        "scenarios":      {},
    }
    for name, sr in res.scenario_results.items():
        entry: dict = {
            "status":           sr.status,
            "error":            sr.error,
            "accuracy":         sr.accuracy,
            "ece":              sr.ece,
            "ece_coverage":     sr.ece_coverage,
            "robustness_score": sr.robustness_score,
            "n_completed":      sr.n_completed,
            "n_invalid":        sr.n_invalid,
            "n_requested":      sr.n_requested,
            "checkpoint_jsonl": sr.checkpoint_path,
        }
        if sr.ambig_unknown_rate is not None:
            entry["ambig_unknown_rate"] = sr.ambig_unknown_rate
        if sr.disambig_accuracy is not None:
            entry["disambig_accuracy"] = sr.disambig_accuracy
        out["scenarios"][name] = entry

    if res.quirk_results:
        out["quirk_detection"] = res.quirk_results[:15]
    if res.linkedin_results:
        out["linkedin_style"] = res.linkedin_results
    if res.tic_results:
        out["tic_suites"] = res.tic_results
    if res.persona_transfer_results:
        out["persona_transfer"] = res.persona_transfer_results
    return out


def _append_tic_suite_sections(lines: List[str], results_by_model: Dict[str, EvalResults], sep: str) -> None:
    for m, res in results_by_model.items():
        for suite_name, suite_results in res.tic_results.items():
            if suite_name == "linkedin_style":
                continue
            lines += [
                "",
                sep,
                f"  TIC SUITE DETECTION — {suite_name} — {m}",
                "  Detector prevalence and mean counts across system prompt conditions",
                "-" * 96,
            ]
            note = STYLE_SUITE_NOTES.get(suite_name)
            if note:
                lines.append(f"  Note: {note}")
            analysis = suite_results.get("analysis", {})
            for condition, stats in analysis.items():
                lines.append(
                    f"  {condition:<20}  n={stats['n_outputs']:>3}"
                    f"{_sample_flag(stats)}"
                )
                for detector_name, detector_stats in stats.get("detectors", {}).items():
                    lines.append(
                        f"    {detector_name:<18}  "
                        f"hit%={detector_stats['pct_hit']:>5.1f}  "
                        f"response-ci={_ci_text(detector_stats):<13}  "
                        f"{_prompt_text(detector_stats)}  "
                        f"mean={detector_stats['mean_count']:>4.2f}"
                    )
            checkpoint_path = suite_results.get("checkpoint_path")
            if checkpoint_path:
                lines.append(f"  Checkpoint: {checkpoint_path}")
            judge = suite_results.get("judge")
            if judge:
                tok = judge.get("token_usage", {})
                lines.append(
                    f"  Judge: {judge.get('model')}  "
                    f"${judge.get('cost_usd', 0.0):.4f}  "
                    f"({tok.get('input', 0):,} in + {tok.get('output', 0):,} out tokens)"
                )
            judge_secondary = suite_results.get("judge_secondary")
            if judge_secondary:
                tok = judge_secondary.get("token_usage", {})
                lines.append(
                    f"  Judge secondary: {judge_secondary.get('model')}  "
                    f"${judge_secondary.get('cost_usd', 0.0):.4f}  "
                    f"({tok.get('input', 0):,} in + {tok.get('output', 0):,} out tokens)"
                )
            judge_agreement = suite_results.get("judge_agreement")
            if judge_agreement:
                overall = judge_agreement.get("overall", {})
                threshold = judge_agreement.get("threshold", 0.0)
                lines.append(
                    "  Judge agreement: "
                    f"n={overall.get('n', 0)}  "
                    f"agree={overall.get('agreement_pct', 0.0):.1f}%  "
                    f"mean|Δ|={overall.get('mean_abs_score_delta', 0.0):.3f}  "
                    f"high(Δ>={threshold:.2f})={overall.get('high_disagreement_pct', 0.0):.1f}%"
                )
                lines.append("  Judge disagreement by condition:")
                for condition, stats in judge_agreement.get("conditions", {}).items():
                    lines.append(
                        f"    {condition:<18}  n={stats.get('n', 0):>3}  "
                        f"agree={stats.get('agreement_pct', 0.0):>5.1f}%  "
                        f"mean|Δ|={stats.get('mean_abs_score_delta', 0.0):>5.3f}"
                    )
                top_disagreements = judge_agreement.get("top_disagreements", [])
                if top_disagreements:
                    lines.append("  Top judge disagreements:")
                    for row in top_disagreements[:5]:
                        lines.append(
                            f"    {row.get('detector', ''):<18}  "
                            f"{row.get('condition', ''):<14}  "
                            f"{row.get('task_type', ''):<16}  "
                            f"Δ={row.get('score_delta', 0.0):>5.3f}"
                        )

            task_segments = suite_results.get("task_segments", {})
            if task_segments:
                lines.append("  Task segments:")
                for task_type, task_analysis in task_segments.items():
                    lines.append(f"    {task_type}")
                    for condition, stats in task_analysis.items():
                        detector_name, detector_stats = _top_detector(stats)
                        if not detector_name:
                            continue
                        lines.append(
                            f"      {condition:<18}  n={stats['n_outputs']:>3}"
                            f"{_sample_flag(stats)}  "
                            f"top={detector_name:<18}  "
                            f"hit%={detector_stats['pct_hit']:>5.1f}  "
                            f"response-ci={_ci_text(detector_stats):<13} "
                            f"{_prompt_text(detector_stats)}"
                        )


def _append_persona_transfer_sections(lines: List[str], results_by_model: Dict[str, EvalResults], sep: str) -> None:
    for m, res in results_by_model.items():
        if not res.persona_transfer_results:
            continue
        lines += [
            "",
            sep,
            f"  PERSONA TRANSFER — {m}",
            "  Detector deltas for persona/style conditions vs each suite baseline",
            "  Scope: stateless prompt-condition deltas.",
            "-" * 96,
        ]
        for suite_name, suite_results in res.persona_transfer_results.items():
            baseline = suite_results.get("baseline_condition", "baseline")
            lines.append(f"  Suite: {suite_name}  baseline={baseline}")
            for condition, stats in suite_results.get("persona_conditions", {}).items():
                lines.append(
                    f"    {condition:<18}  n={stats.get('n_outputs', 0):>3}  "
                    f"score={stats.get('transfer_score', 0):>6.2f}"
                )
                for row in stats.get("top_detectors", [])[:5]:
                    if row["transfer_score"] <= 0:
                        continue
                    lines.append(
                        f"      {row['detector']:<18}  "
                        f"hit_delta={row['pct_hit_delta']:>6.1f}pp  "
                        f"mean_delta={row['mean_count_delta']:>5.2f}  "
                        f"lift={row['mean_count_lift']:>5.2f}"
                    )


def _append_pairwise_section(lines: List[str], results_by_model: Dict[str, EvalResults], sep: str) -> None:
    pairwise = _pairwise_results(results_by_model)
    if not pairwise:
        return

    lines += [
        "",
        sep,
        "  PAIRWISE PREFERENCE",
        _pairwise_scoring_text(pairwise),
        "-" * 96,
    ]
    for suite_name, suite_results in pairwise.items():
        lines.append(f"  Suite: {suite_name}")
        scoring = suite_results.get("scoring", {})
        method = scoring.get("pairwise_method", "count")
        segment_by = scoring.get("segment_by", "task_type")
        halflife = scoring.get("time_decay_halflife_days")
        lines.append(f"    method={method}  segment_by={segment_by}")
        if halflife:
            lines.append(f"    time_decay_halflife_days={halflife}")
        for row in suite_results.get("overall", {}).get("ranking", []):
            score = row.get("preference_score", row.get("bt_score", 0.0))
            lines.append(
                f"    {row['model']:<30}  "
                f"score={score:>6.3f}  "
                f"matches={row['matches']:>4}  "
                f"W/L/T={row.get('wins', 0)}/{row.get('losses', 0)}/{row.get('ties', 0)}"
            )
        segments = suite_results.get("segments", {})
        if not segments:
            segments = suite_results.get("task_types", {})
        if segments:
            label = "Task types" if segment_by == "task_type" else f"Segments ({segment_by})"
            lines.append(f"    {label}:")
            for task_type, task_results in segments.items():
                ranking = task_results.get("ranking", [])
                if not ranking:
                    continue
                leader = ranking[0]
                leader_score = leader.get("preference_score", leader.get("bt_score", 0.0))
                lines.append(
                    f"      {task_type:<24}  "
                    f"leader={leader['model']:<24}  "
                    f"score={leader_score:>6.3f}"
                )


def _pairwise_scoring_text(pairwise: Dict[str, dict]) -> str:
    methods = {
        suite_results.get("scoring", {}).get("pairwise_method", "count")
        for suite_results in pairwise.values()
    }
    is_bt = methods == {"bradley_terry"}
    detectors = {
        suite_results.get("scoring", {}).get("detector")
        for suite_results in pairwise.values()
        if suite_results.get("scoring", {}).get("detector")
    }
    if len(detectors) == 1:
        detector = next(iter(detectors))
        if is_bt:
            return (
                f"  Bradley-Terry: lower {detector} detector count preferred on "
                "shared style-suite records"
            )
        return f"  Lower {detector} detector count wins on shared style-suite records"
    if is_bt:
        return "  Bradley-Terry: lower detector-count total preferred on shared style-suite records"
    return "  Lower detector-count total wins on shared style-suite records"


def _append_accuracy_section(
    lines: List[str],
    results_by_model: Dict[str, EvalResults],
    models: List[str],
    scenario_names: List[str],
) -> None:
    col = 20
    lines.append(f"{'Scenario':<{col}}" + "  ".join(
        f"{'Acc':>7}  {'ECE(cov)':>10}  {'Rob':>7}   [{m}]" for m in models
    ))
    lines.append("-" * 96)

    for sname in scenario_names:
        row = f"{sname:<{col}}"
        for m in models:
            sr = results_by_model[m].scenario_results.get(sname)
            if sr and sr.accuracy is not None:
                ece_str = (
                    f"{sr.ece:.4f}({sr.ece_coverage:.0%})"
                    if sr.ece is not None else "     N/A    "
                )
                rob_str = f"{sr.robustness_score:.4f}" if sr.robustness_score is not None else "   N/A"
                n_str = (
                    f" [{sr.n_completed}/{sr.n_requested}]"
                    if sr.n_completed < sr.n_requested else ""
                )
                if sr.n_invalid:
                    n_str += f" [invalid={sr.n_invalid}]"
                row += f"  {sr.accuracy:>7.4f}  {ece_str:>10}  {rob_str:>7}{n_str}"
            else:
                row += f"  {'N/A':>7}  {'N/A':>10}  {'N/A':>7}"
            if sr and sr.status != "complete":
                row += f" [{sr.status}: {sr.n_completed}/{sr.n_requested}; {sr.error}]"
        lines.append(row)

    lines.append("-" * 96)
    for m, res in results_by_model.items():
        accs = [sr.accuracy for sr in res.scenario_results.values() if sr.status == "complete" and sr.accuracy is not None]
        if accs and all(sr.status == "complete" for sr in res.scenario_results.values()):
            lines.append(f"{'Mean':<{col}}  {sum(accs)/len(accs):>7.4f}   [{m}]")


def _append_bbq_sections(lines: List[str], results_by_model: Dict[str, EvalResults]) -> None:
    for m, res in results_by_model.items():
        sr = res.scenario_results.get("bbq")
        if sr and sr.ambig_unknown_rate is not None:
            lines += [
                "",
                f"  BBQ Bias Metrics — {m}",
                f"    Ambiguous unknown-choice rate : {sr.ambig_unknown_rate:.4f}"
                "  (higher = less biased; random ≈ 0.33)",
            ]
            if sr.disambig_accuracy is not None:
                lines.append(f"    Disambiguated accuracy        : {sr.disambig_accuracy:.4f}")


def _append_linkedin_cross_model_section(
    lines: List[str],
    results_by_model: Dict[str, EvalResults],
    models: List[str],
    sep: str,
) -> None:
    if any(res.linkedin_results for res in results_by_model.values()):
        lines += ["", sep, "  LINKEDIN STYLE — CROSS-MODEL COMPARISON", ""]
        conditions = ("neutral", "professional", "mildly_positive", "linkedin_style")
        lines.append("  (neutral=baseline, professional=instruction-following, mildly_positive=soft-constraint, linkedin_style=control)")
        lines += [""]

        header = f"{'Condition':<20}  " + "  ".join(f"{m:<14}" for m in models)
        lines.append(header)
        lines.append("-" * (20 + 2 + 16 * len(models)))

        for cond in conditions:
            row = f"  {cond:<18}  "
            cond_data = []
            for m in models:
                res = results_by_model[m]
                if res.linkedin_results:
                    analysis = res.linkedin_results.get("analysis", {})
                    stats = analysis.get(cond)
                    if stats:
                        cond_data.append(f"{stats['pct_with_emoji']:>5.1f}% n={stats['n_outputs']:>2}")
                    else:
                        cond_data.append("N/A")
                else:
                    cond_data.append("N/A")
            row += "  ".join(f"{data:<14}" for data in cond_data)
            lines.append(row)


def _append_cost_section(lines: List[str], results_by_model: Dict[str, EvalResults], sep: str) -> None:
    lines += ["", sep, "  COST SUMMARY", "-" * 96]
    for m, res in results_by_model.items():
        tok = res.token_usage
        lines.append(
            f"  {m:<30}  ${res.total_cost_usd:.4f}  "
            f"({tok.get('input',0):,} in + {tok.get('output',0):,} out tokens)"
        )


def _append_checkpoint_section(lines: List[str], results_by_model: Dict[str, EvalResults], sep: str) -> None:
    lines += ["", sep, "  PER-INSTANCE RECORDS (JSONL checkpoints)", "-" * 96]
    for m, res in results_by_model.items():
        for sname, sr in res.scenario_results.items():
            lines.append(f"  {m}/{sname:<18}  {sr.checkpoint_path}")


def _append_quirk_sections(lines: List[str], results_by_model: Dict[str, EvalResults], sep: str) -> None:
    for m, res in results_by_model.items():
        if not res.quirk_results:
            continue
        lines += [
            "",
            sep,
            f"  QUIRK DETECTION — {m}",
            "  Lexical tics whose frequency is anomalously high in one system-prompt condition",
            f"  {'Word':<22}  {'Lift':>6}  {'Dominant Condition':<22}  N/P/Pr/F",
            "-" * 96,
        ]
        for q in res.quirk_results[:15]:
            cts = q["counts"]
            cnt = "/".join(str(cts.get(c, 0)) for c in ("neutral", "playful", "professional", "friendly"))
            lines.append(f"  {q['word']:<22}  {q['lift']:>6.2f}  {q['dominant_condition']:<22}  {cnt}")


def _append_linkedin_style_sections(lines: List[str], results_by_model: Dict[str, EvalResults], sep: str) -> None:
    for m, res in results_by_model.items():
        if not res.linkedin_results:
            continue
        lines += [
            "",
            sep,
            f"  LINKEDIN STYLE DETECTION — {m}",
            "  Emoji prevalence and LinkedIn-speak tics across system prompt conditions",
            "-" * 96,
        ]
        analysis = res.linkedin_results.get("analysis", {})
        for cond, stats in analysis.items():
            lines.append(
                f"  {cond:<20}  n={stats['n_outputs']:>3}  "
                f"any%={stats['pct_with_emoji']:>5.1f}  "
                f"response-ci={_ci_text(stats, 'pct_with_emoji'):<13}  "
                f"{_prompt_text(stats.get('detectors', {}).get('emoji', {}))}  "
                f"excessive%={stats['pct_excessive_emoji']:>5.1f}  "
                f"mean_emoji={stats['mean_emoji_count']:>4.2f}  "
                f"phrases={stats['mean_phrase_hits']:>4.2f}"
                f"{_sample_flag(stats)}"
            )
        checkpoint_path = res.linkedin_results.get("checkpoint_path")
        if checkpoint_path:
            lines.append(f"  Checkpoint: {checkpoint_path}")


def _append_model_result_sections(
    lines: List[str],
    results_by_model: Dict[str, EvalResults],
    models: List[str],
    scenario_names: List[str],
    sep: str,
) -> None:
    _append_accuracy_section(lines, results_by_model, models, scenario_names)
    _append_bbq_sections(lines, results_by_model)
    _append_linkedin_cross_model_section(lines, results_by_model, models, sep)
    _append_cost_section(lines, results_by_model, sep)
    _append_checkpoint_section(lines, results_by_model, sep)
    _append_quirk_sections(lines, results_by_model, sep)
    _append_linkedin_style_sections(lines, results_by_model, sep)


def format_table(results_by_model: Dict[str, EvalResults]) -> str:
    models = list(results_by_model.keys())
    snames = list(dict.fromkeys(name for res in results_by_model.values() for name in res.scenario_results))
    sep    = "=" * 96
    lines  = [sep, "  COMPASS EXPERIMENT RESULTS", f"  {datetime.now():%Y-%m-%d %H:%M:%S}", sep, ""]

    _append_model_result_sections(lines, results_by_model, models, snames, sep)
    _append_tic_suite_sections(lines, results_by_model, sep)
    _append_persona_transfer_sections(lines, results_by_model, sep)
    _append_pairwise_section(lines, results_by_model, sep)

    lines.append(sep)
    return "\n".join(lines)


def save(results_by_model: Dict[str, EvalResults], output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {m: _summary_dict(r) for m, r in results_by_model.items()}
    pairwise = _pairwise_results(results_by_model)
    if pairwise:
        summary["pairwise_preference"] = pairwise
    json_path = os.path.join(output_dir, f"results_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    txt_path = os.path.join(output_dir, f"summary_{ts}.txt")
    with open(txt_path, "w") as f:
        f.write(format_table(results_by_model))

    return json_path, txt_path


def _load_thresholds(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _is_low_sample(stats: Dict[str, Any], det_stats: Dict[str, Any]) -> bool:
    # Condition-level low_sample is the primary signal; detector-level fallback
    # supports checkpoint variants that store it only under detectors.
    return bool(stats.get("low_sample") or det_stats.get("low_sample"))


def _iter_detector_rows(results_by_model: Dict[str, EvalResults]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model, res in results_by_model.items():
        for suite_name, suite_results in res.tic_results.items():
            analysis = suite_results.get("analysis", {})
            for condition, stats in analysis.items():
                for detector, det_stats in stats.get("detectors", {}).items():
                    rows.append(
                        {
                            "model": model,
                            "suite": suite_name,
                            "segment": "condition",
                            "task_type": None,
                            "condition": condition,
                            "detector": detector,
                            "n_outputs": int(stats.get("n_outputs", 0)),
                            "low_sample": _is_low_sample(stats, det_stats),
                            "pct_hit": float(det_stats.get("pct_hit", 0.0)),
                            "mean_count": float(det_stats.get("mean_count", 0.0)),
                            "prompt_statistics": det_stats.get("prompt_statistics", {}),
                            "ci_low": det_stats.get("pct_hit_ci_low"),
                            "ci_high": det_stats.get("pct_hit_ci_high"),
                        }
                    )
            task_segments = suite_results.get("task_segments", {})
            for task_type, task_analysis in task_segments.items():
                for condition, stats in task_analysis.items():
                    for detector, det_stats in stats.get("detectors", {}).items():
                        rows.append(
                            {
                                "model": model,
                                "suite": suite_name,
                                "segment": "task",
                                "task_type": task_type,
                                "condition": condition,
                                "detector": detector,
                                "n_outputs": int(stats.get("n_outputs", 0)),
                                "low_sample": _is_low_sample(stats, det_stats),
                                "pct_hit": float(det_stats.get("pct_hit", 0.0)),
                                "mean_count": float(det_stats.get("mean_count", 0.0)),
                                "prompt_statistics": det_stats.get("prompt_statistics", {}),
                            "ci_low": det_stats.get("pct_hit_ci_low"),
                                "ci_high": det_stats.get("pct_hit_ci_high"),
                            }
                        )
    return rows


def _threshold_rule(thresholds: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    suites = thresholds.get("suites", {})
    suite_cfg = suites.get(row["suite"], {}) if isinstance(suites, dict) else {}
    detectors_cfg = suite_cfg.get("detectors", {}) if isinstance(suite_cfg, dict) else {}
    rule = detectors_cfg.get(row["detector"], {}) if isinstance(detectors_cfg, dict) else {}
    return rule if isinstance(rule, dict) else {}


def _find_threshold_crossings(
    rows: List[Dict[str, Any]],
    thresholds: Dict[str, Any],
) -> List[Dict[str, Any]]:
    crossings: List[Dict[str, Any]] = []
    for row in rows:
        rule = _threshold_rule(thresholds, row)
        if not rule:
            continue
        max_pct_hit = rule.get("max_pct_hit")
        max_mean_count = rule.get("max_mean_count")
        if isinstance(max_pct_hit, (int, float)) and row["pct_hit"] > float(max_pct_hit):
            crossings.append(
                {
                    **row,
                    "metric": "pct_hit",
                    "threshold": float(max_pct_hit),
                    "value": row["pct_hit"],
                    "exceed_by": row["pct_hit"] - float(max_pct_hit),
                }
            )
        if isinstance(max_mean_count, (int, float)) and row["mean_count"] > float(max_mean_count):
            crossings.append(
                {
                    **row,
                    "metric": "mean_count",
                    "threshold": float(max_mean_count),
                    "value": row["mean_count"],
                    "exceed_by": row["mean_count"] - float(max_mean_count),
                }
            )
    return crossings


def _ci_width(row: Dict[str, Any]) -> float:
    low = row.get("ci_low")
    high = row.get("ci_high")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return max(float(high) - float(low), 0.0)
    return 0.0


def _deep_dive_rows(
    rows: List[Dict[str, Any]],
    crossings: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    crossing_index = {
        (
            row["model"],
            row["suite"],
            row["segment"],
            row.get("task_type"),
            row["condition"],
            row["detector"],
        ): row
        for row in crossings
    }
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        key = (
            row["model"],
            row["suite"],
            row["segment"],
            row.get("task_type"),
            row["condition"],
            row["detector"],
        )
        crossing = crossing_index.get(key)
        severity = float(crossing["exceed_by"]) if crossing else 0.0
        # Priority score = severity + uncertainty where:
        # - severity is threshold exceedance magnitude (0 if no crossing),
        # - uncertainty is 1.0 for low-n plus CI width scaled to [0,1].
        # This intentionally includes uncertain non-crossing rows.
        uncertainty = (1.0 if row.get("low_sample") else 0.0) + _ci_width(row) / 100.0
        score = severity + uncertainty
        if score <= 0:
            continue
        scored.append((score, {**row, "priority_score": round(score, 4), "crossing": crossing}))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _score, row in scored[:top_k]]


def _safety_recommendations(
    crossings: List[Dict[str, Any]],
    uncertain_rows: List[Dict[str, Any]],
) -> List[str]:
    recs: List[str] = []
    if crossings:
        recs.append(
            f"Threshold crossings detected in {len(crossings)} detector slices; audit top exceedances before promotion."
        )
    else:
        recs.append("No configured threshold crossings detected in this run.")
    if uncertain_rows:
        recs.append(
            f"{len(uncertain_rows)} uncertain slices flagged; inspect distinct prompt coverage and interval limits before expanding the sample."
        )
    else:
        recs.append("No uncertainty flags triggered under current prompt-coverage/CI heuristics.")
    recs.append("Re-run across multiple dates and model versions before drawing deployment conclusions.")
    return recs


def save_safety_case(
    results_by_model: Dict[str, EvalResults],
    output_dir: str,
    thresholds_path: str = "configs/thresholds.yaml",
    deep_dive_top_k: int = 0,
) -> str:
    """
    Generate a markdown behavioral evaluation report for tic-suite outputs.

    The report includes threshold crossings (from thresholds YAML), uncertainty
    flags, deep-dive candidates, and recommendations. Returns the path to the
    generated markdown file, or an empty string when no tic-suite results exist.
    """
    if not any(res.tic_results for res in results_by_model.values()):
        return ""

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"safety_case_{ts}.md")

    thresholds = _load_thresholds(thresholds_path)
    rows = _iter_detector_rows(results_by_model)
    crossings = _find_threshold_crossings(rows, thresholds)
    uncertain_rows = [row for row in rows if row.get("low_sample") or _prompt_uncertain(row)]
    deep_dive = _deep_dive_rows(rows, crossings, deep_dive_top_k)
    recommendations = _safety_recommendations(crossings, uncertain_rows)

    models = ", ".join(results_by_model.keys())
    suites = sorted(
        {
            suite_name
            for res in results_by_model.values()
            for suite_name in res.tic_results.keys()
        }
    )

    lines = [
        "# Behavioral Evaluation Report",
        "",
        "Configured thresholds are screening rules, not evidence of deployment safety.",
        "Response-level Wilson intervals assume independent observations. Prompt-cluster intervals describe variation across the observed prompts; they do not establish coverage of unseen tasks.",
        "",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- Models: {models}",
        f"- Tic suites: {', '.join(suites)}",
        f"- Threshold config: {thresholds_path if thresholds else 'not found/empty'}",
        "",
        "## Threshold Crossings",
    ]
    if crossings:
        lines.append("| Model | Suite | Segment | Condition | Detector | Metric | Value | Threshold | Exceed By |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
        # Cap table size for readability in long multi-model runs.
        for row in sorted(crossings, key=lambda r: r["exceed_by"], reverse=True)[:50]:
            segment = row["segment"] if row["segment"] == "condition" else f"task:{row.get('task_type')}"
            lines.append(
                f"| {row['model']} | {row['suite']} | {segment} | {row['condition']} | "
                f"{row['detector']} | {row['metric']} | {row['value']:.2f} | "
                f"{row['threshold']:.2f} | {row['exceed_by']:.2f} |"
            )
    else:
        lines.append("No threshold crossings detected.")

    lines += ["", "## Uncertainty Flags", ""]
    lines.append(f"- Flagged slices: {len(uncertain_rows)}")
    lines.append("- Rule: low response count, fewer than 20 distinct prompts, unknown prompt coverage, or an unavailable, degenerate, or wide prompt interval (>=40 percentage points). The count threshold is a screening heuristic.")

    for row in uncertain_rows:
        lines.append(f"- {row['model']} / {row['suite']} / {row['condition']} / {row['detector']} / {row['task_type'] or 'all tasks'}: {_prompt_text(row)}; responses={row['n_outputs']}")

    if deep_dive_top_k > 0:
        lines += ["", f"## Deep Dive Candidates (Top {deep_dive_top_k})", ""]
        if deep_dive:
            lines.append("| Priority | Model | Suite | Segment | Condition | Detector | hit% | mean | low-n |")
            lines.append("|---:|---|---|---|---|---|---:|---:|---|")
            for row in deep_dive:
                segment = row["segment"] if row["segment"] == "condition" else f"task:{row.get('task_type')}"
                lines.append(
                    f"| {row['priority_score']:.3f} | {row['model']} | {row['suite']} | "
                    f"{segment} | {row['condition']} | {row['detector']} | "
                    f"{row['pct_hit']:.1f} | {row['mean_count']:.2f} | {row['low_sample']} |"
                )
        else:
            lines.append("No deep-dive candidates under current heuristics.")

    lines += ["", "## Recommendations", ""]
    for rec in recommendations:
        lines.append(f"- {rec}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path
