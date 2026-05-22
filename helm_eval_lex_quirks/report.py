import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from helm_eval_lex_quirks.metrics.builtin_tic_suites import STYLE_SUITE_NOTES
from helm_eval_lex_quirks.runner import EvalResults


def _ci_text(stats: dict, prefix: str = "pct_hit") -> str:
    low = stats.get(f"{prefix}_ci_low")
    high = stats.get(f"{prefix}_ci_high")
    if low is None or high is None:
        return ""
    return f"[{low:.1f},{high:.1f}]"


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
            "accuracy":         sr.accuracy,
            "ece":              sr.ece,
            "ece_coverage":     sr.ece_coverage,
            "robustness_score": sr.robustness_score,
            "n_completed":      sr.n_completed,
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
                        f"ci={_ci_text(detector_stats):<13}  "
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
                            f"ci={_ci_text(detector_stats):<13}"
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
        for row in suite_results.get("overall", {}).get("ranking", []):
            lines.append(
                f"    {row['model']:<30}  "
                f"score={row['preference_score']:>6.2f}  "
                f"matches={row['matches']:>4}  "
                f"W/L/T={row['wins']}/{row['losses']}/{row['ties']}"
            )
        task_types = suite_results.get("task_types", {})
        if task_types:
            lines.append("    Task types:")
            for task_type, task_results in task_types.items():
                ranking = task_results.get("ranking", [])
                if not ranking:
                    continue
                leader = ranking[0]
                lines.append(
                    f"      {task_type:<24}  "
                    f"leader={leader['model']:<24}  "
                    f"score={leader['preference_score']:>6.2f}"
                )


def _pairwise_scoring_text(pairwise: Dict[str, dict]) -> str:
    detectors = {
        suite_results.get("scoring", {}).get("detector")
        for suite_results in pairwise.values()
        if suite_results.get("scoring", {}).get("detector")
    }
    if len(detectors) == 1:
        detector = next(iter(detectors))
        return f"  Lower {detector} detector count wins on shared style-suite records"
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
            if sr:
                ece_str = (
                    f"{sr.ece:.4f}({sr.ece_coverage:.0%})"
                    if sr.ece is not None else "     N/A    "
                )
                rob_str = f"{sr.robustness_score:.4f}" if sr.robustness_score is not None else "   N/A"
                n_str = (
                    f" [{sr.n_completed}/{sr.n_requested}]"
                    if sr.n_completed < sr.n_requested else ""
                )
                row += f"  {sr.accuracy:>7.4f}  {ece_str:>10}  {rob_str:>7}{n_str}"
            else:
                row += f"  {'N/A':>7}  {'N/A':>10}  {'N/A':>7}"
        lines.append(row)

    lines.append("-" * 96)
    for m, res in results_by_model.items():
        accs = [sr.accuracy for sr in res.scenario_results.values()]
        if accs:
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
                f"ci={_ci_text(stats, 'pct_with_emoji'):<13}  "
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
    snames = list(next(iter(results_by_model.values())).scenario_results.keys())
    sep    = "=" * 96
    lines  = [sep, "  HELM EVAL RESULTS", f"  {datetime.now():%Y-%m-%d %H:%M:%S}", sep, ""]

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
