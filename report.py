import json
import os
from datetime import datetime
from typing import Dict

from helm_eval.runner import EvalResults


def _summary_dict(res: EvalResults) -> dict:
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
    return out


def format_table(results_by_model: Dict[str, EvalResults]) -> str:
    models = list(results_by_model.keys())
    snames = list(next(iter(results_by_model.values())).scenario_results.keys())
    sep    = "=" * 96
    lines  = [sep, "  HELM EVAL RESULTS", f"  {datetime.now():%Y-%m-%d %H:%M:%S}", sep, ""]

    # ── accuracy / ECE / robustness ───────────────────────────────────────────
    col = 20
    lines.append(f"{'Scenario':<{col}}" + "  ".join(
        f"{'Acc':>7}  {'ECE(cov)':>10}  {'Rob':>7}   [{m}]" for m in models
    ))
    lines.append("-" * 96)

    for sname in snames:
        row = f"{sname:<{col}}"
        for m in models:
            sr = results_by_model[m].scenario_results.get(sname)
            if sr:
                ece_str = (
                    f"{sr.ece:.4f}({sr.ece_coverage:.0%})"
                    if sr.ece is not None else "     N/A    "
                )
                rob_str = f"{sr.robustness_score:.4f}" if sr.robustness_score is not None else "   N/A"
                n_str   = (
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

    # ── BBQ bias ──────────────────────────────────────────────────────────────
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

    # ── cost ──────────────────────────────────────────────────────────────────
    lines += ["", sep, "  COST SUMMARY", "-" * 96]
    for m, res in results_by_model.items():
        tok = res.token_usage
        lines.append(
            f"  {m:<30}  ${res.total_cost_usd:.4f}  "
            f"({tok.get('input',0):,} in + {tok.get('output',0):,} out tokens)"
        )

    # ── per-instance record paths ─────────────────────────────────────────────
    lines += ["", sep, "  PER-INSTANCE RECORDS (JSONL checkpoints)", "-" * 96]
    for m, res in results_by_model.items():
        for sname, sr in res.scenario_results.items():
            lines.append(f"  {m}/{sname:<18}  {sr.checkpoint_path}")

    # ── quirk detection ───────────────────────────────────────────────────────
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

    lines.append(sep)
    return "\n".join(lines)


def save(results_by_model: Dict[str, EvalResults], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {m: _summary_dict(r) for m, r in results_by_model.items()}
    json_path = os.path.join(output_dir, f"results_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    txt_path = os.path.join(output_dir, f"summary_{ts}.txt")
    with open(txt_path, "w") as f:
        f.write(format_table(results_by_model))

    return json_path, txt_path
