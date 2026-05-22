"""
HELM-style evaluation harness.

Usage examples
--------------
# Dry run — estimate cost only
python -m helm_eval_lex_quirks.main --dry-run

# Full eval on gpt-4o-mini with defaults (100 samples/scenario)
python -m helm_eval_lex_quirks.main --api-key sk-...

# Cheaper quick smoke-test (20 samples, no robustness)
python -m helm_eval_lex_quirks.main --api-key sk-... --n-samples 20 --no-robustness

# Evaluate multiple OpenAI and Claude models
python -m helm_eval_lex_quirks.main --api-key sk-... --anthropic-api-key sk-ant-... \
  --models gpt-4o-mini claude-haiku-4-5-20251001

# Run LinkedIn style detection
python -m helm_eval_lex_quirks.main --api-key sk-... --linkedin-style

# Run generic tic/style suites only
python -m helm_eval_lex_quirks.main --api-key sk-... --style-only --style-suite lexical punctuation tone

# Run persona-transfer analysis over selected style suites
python -m helm_eval_lex_quirks.main --api-key sk-... --style-only --style-suite all --persona-transfer

# Compare models head-to-head on style-suite checkpoints
python -m helm_eval_lex_quirks.main --api-key sk-... --style-only --style-suite all --pairwise-preference --models gpt-4o-mini gpt-4.1-nano

# Compare models on one detector only
python -m helm_eval_lex_quirks.main --api-key sk-... --style-only --style-suite rest_cai_validation --pairwise-detector rest_suggestion --models claude-haiku-4-5-20251001 claude-sonnet-4-6

# Compare models with Bradley-Terry and time decay
python -m helm_eval_lex_quirks.main --api-key sk-... --style-only --style-suite all \
  --pairwise-preference --pairwise-method bradley_terry --pairwise-segment-by condition \
  --pairwise-time-decay-halflife-days 90

# Run subjective tone judges
python -m helm_eval_lex_quirks.main --api-key sk-... --anthropic-api-key sk-ant-... --style-only --style-suite llm_judge_tone --judge-model gpt-4o-mini --models claude-haiku-4-5-20251001

# Run self-harm safety handling audit (LLM judge required)
python -m helm_eval_lex_quirks.main --api-key sk-... --anthropic-api-key sk-ant-... --style-only --style-suite self_harm_safety --judge-model gpt-4o-mini --models gpt-4o-mini claude-haiku-4-5-20251001

# Run dual-judge disagreement audit for subjective suites
python -m helm_eval_lex_quirks.main --api-key sk-... --anthropic-api-key sk-ant-... \
  --style-only --style-suite llm_judge_tone --judge-model gpt-4o-mini \
  --judge-model-secondary claude-haiku-4-5-20251001 \
  --judge-disagreement-threshold 0.4 --models claude-sonnet-4-6

# Emit a safety-case markdown report with deep-dive candidates
python -m helm_eval_lex_quirks.main --style-only --style-suite all \
  --deep-dive-top-k 10 --safety-thresholds configs/thresholds.yaml
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from helm_eval_lex_quirks.config import DEFAULT_SCENARIO_CONFIGS, EvalConfig, model_provider
from helm_eval_lex_quirks.metrics.builtin_tic_suites import STYLE_SUITE_REGISTRY, expand_suite_names
from helm_eval_lex_quirks.metrics.pairwise_preference import analyze_pairwise_preference
from helm_eval_lex_quirks.metrics.tic_suite import suite_uses_llm_judge
from helm_eval_lex_quirks.runner import SCENARIO_REGISTRY, EvalResults, EvalRunner
from helm_eval_lex_quirks import report as rpt


def _load_results_for_pairwise(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("pairwise input JSON must be a top-level object.")

    results = {}
    for model_name, payload in data.items():
        if not isinstance(payload, dict):
            continue
        tic_results = payload.get("tic_suites", {})
        if not tic_results:
            continue
        results[model_name] = EvalResults(
            model=model_name,
            tic_results=tic_results,
        )
    return results


def _run_pairwise_from_results(args: argparse.Namespace) -> None:
    results = _load_results_for_pairwise(args.pairwise_from_results)
    if len(results) < 2:
        raise ValueError("Need at least two models with tic suite data in the results JSON.")

    pairwise = analyze_pairwise_preference(
        results,
        detector_name=args.pairwise_detector,
        method=args.pairwise_method,
        segment_by=args.pairwise_segment_by,
        time_decay_halflife_days=args.pairwise_time_decay_halflife_days,
    )
    if not pairwise:
        raise ValueError("No pairwise-comparable shared records were found in referenced checkpoints.")

    for result in results.values():
        result.pairwise_preference_results = pairwise

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"pairwise_from_results_{ts}.json")
    txt_path = os.path.join(args.output_dir, f"pairwise_from_results_{ts}.txt")
    with open(json_path, "w") as f:
        json.dump(pairwise, f, indent=2)
    table = rpt.format_table(results)
    with open(txt_path, "w") as f:
        f.write(table)

    print()
    print(table)
    print(f"\nPairwise JSON : {json_path}")
    print(f"Pairwise text : {txt_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="HELM-style eval harness for OpenAI models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-key",   default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--anthropic-api-key", default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key for Claude models (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--models",    nargs="+", default=["gpt-4o-mini"],
                        help="Model names (gpt-4o-mini, gpt-4.1-nano, claude-haiku-4-5-20251001, claude-sonnet-4-6, etc.)")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIO_CONFIGS),
                        choices=list(SCENARIO_REGISTRY))
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override the per-scenario sample count")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-robustness",  action="store_true",
                        help="Skip robustness (typo-perturbation) evaluation")
    parser.add_argument("--quirks", action="store_true",
                        help="Enable goblin-style lexical quirk detection (disabled by default)")
    parser.add_argument("--no-quirks", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--linkedin-style", action="store_true",
                        help="Run LinkedIn style (emoji + phrase) detection")
    parser.add_argument("--linkedin-only", action="store_true",
                        help="Run only LinkedIn style detection, skip benchmark scenarios")
    parser.add_argument("--linkedin-samples", type=int, default=4,
                        help="Repetitions per prompt for LinkedIn style detection")
    parser.add_argument("--style-suite", nargs="+", default=[],
                        choices=list(STYLE_SUITE_REGISTRY) + ["all"],
                        help="Run tic/style suite(s): lexical, punctuation, formatting, tone, llm_judge_tone, self_harm_safety, linkedin, or all")
    parser.add_argument("--style-only", action="store_true",
                        help="Run only selected generic style suites, skip benchmark scenarios")
    parser.add_argument("--style-samples", type=int, default=4,
                        help="Repetitions per prompt for generic tic/style suites")
    parser.add_argument(
        "--style-samples-increment",
        action="store_true",
        help=(
            "Append this many new style samples per prompt/condition on each run "
            "instead of targeting fixed sample_idx range [0..style_samples-1]"
        ),
    )
    parser.add_argument("--persona-transfer", action="store_true",
                        help="Compare persona/style conditions against each style suite baseline")
    parser.add_argument("--pairwise-preference", action="store_true",
                        help="Compare models head-to-head on shared style-suite records")
    parser.add_argument("--pairwise-detector",
                        help="Score pairwise comparisons using one detector count; implies --pairwise-preference")
    parser.add_argument(
        "--pairwise-from-results",
        help="Offline mode: recompute pairwise from an existing results_*.json file",
    )
    parser.add_argument(
        "--pairwise-method",
        choices=("count", "bradley_terry"),
        default="count",
        help="Pairwise ranking method",
    )
    parser.add_argument(
        "--pairwise-segment-by",
        choices=("none", "task_type", "condition"),
        default="task_type",
        help="Segmentation key for pairwise sub-rankings",
    )
    parser.add_argument(
        "--pairwise-time-decay-halflife-days",
        type=int,
        default=None,
        help="Optional half-life for recency weighting in pairwise scoring",
    )
    parser.add_argument(
        "--run-tag",
        help="Optional tag persisted on checkpoint records for run tracking",
    )
    parser.add_argument("--judge-model",
                        help="Model used for suites with LLM judge detectors")
    parser.add_argument(
        "--judge-model-secondary",
        help="Optional second judge model for agreement/disagreement auditing",
    )
    parser.add_argument(
        "--judge-disagreement-threshold",
        type=float,
        default=0.4,
        help="Absolute score-delta threshold for flagging high judge disagreement",
    )
    parser.add_argument(
        "--safety-thresholds",
        default="configs/thresholds.yaml",
        help="Path to YAML threshold config used for safety-case report generation",
    )
    parser.add_argument(
        "--deep-dive-top-k",
        type=int,
        default=0,
        help="Number of high-risk/uncertain detector slices to include in safety-case deep dive",
    )
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--request-interval", type=float, default=2.0,
                        help="Seconds between API calls. Raise to 4-6 for free-tier accounts.")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Estimate cost and exit without calling the API")
    args = parser.parse_args(argv)

    # Handle --linkedin-only: skip all benchmark scenarios
    if args.linkedin_only:
        args.linkedin_style = True
        args.no_robustness = True
        args.scenarios = []

    try:
        style_suites = expand_suite_names(args.style_suite)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    if (args.persona_transfer or args.pairwise_preference or args.pairwise_detector) and not style_suites and not args.linkedin_style:
        style_suites = expand_suite_names(["all"])
    if args.style_only:
        args.no_robustness = True
        args.scenarios = []
        if not style_suites:
            print("ERROR: --style-only requires at least one --style-suite.", file=sys.stderr)
            sys.exit(1)
    uses_llm_judge = any(
        suite_uses_llm_judge(STYLE_SUITE_REGISTRY[suite_key])
        for suite_key in style_suites
    )
    if uses_llm_judge and not args.judge_model:
        print("ERROR: selected style suites with LLM judges require --judge-model.", file=sys.stderr)
        sys.exit(1)
    if args.judge_model_secondary and args.judge_model_secondary == args.judge_model:
        print("ERROR: --judge-model-secondary must differ from --judge-model.", file=sys.stderr)
        sys.exit(1)
    if args.judge_model_secondary and not uses_llm_judge:
        logging.warning(
            "--judge-model-secondary was provided but no LLM-judge style suite is enabled; "
            "secondary judge will not be used."
        )
    if args.pairwise_time_decay_halflife_days is not None and args.pairwise_time_decay_halflife_days <= 0:
        print("ERROR: --pairwise-time-decay-halflife-days must be > 0.", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.judge_disagreement_threshold <= 1.0):
        print("ERROR: --judge-disagreement-threshold must be in [0, 1].", file=sys.stderr)
        sys.exit(1)
    if args.deep_dive_top_k < 0:
        print("ERROR: --deep-dive-top-k must be >= 0.", file=sys.stderr)
        sys.exit(1)
    if args.pairwise_from_results:
        try:
            _run_pairwise_from_results(args)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        return
    model_names = list(args.models)
    if args.judge_model:
        model_names.append(args.judge_model)
    if args.judge_model_secondary:
        model_names.append(args.judge_model_secondary)
    providers = {model_provider(model) for model in model_names}
    if "openai" in providers and not args.api_key:
        print("ERROR: OpenAI models require --api-key or OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)
    if "anthropic" in providers and not args.anthropic_api_key and not args.dry_run:
        print("ERROR: Claude models require --anthropic-api-key or ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)

    cfg = EvalConfig(
        api_key=args.api_key,
        anthropic_api_key=args.anthropic_api_key,
        models=args.models,
        scenarios=args.scenarios,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        run_robustness=not args.no_robustness,
        run_quirk_detection=args.quirks,
        run_linkedin_style=args.linkedin_style,
        linkedin_only=args.linkedin_only,
        linkedin_samples=args.linkedin_samples,
        style_suites=style_suites,
        style_only=args.style_only,
        style_samples=args.style_samples,
        style_samples_increment=args.style_samples_increment,
        run_persona_transfer=args.persona_transfer,
        run_pairwise_preference=args.pairwise_preference or bool(args.pairwise_detector),
        pairwise_detector=args.pairwise_detector,
        pairwise_method=args.pairwise_method,
        pairwise_segment_by=args.pairwise_segment_by,
        pairwise_time_decay_halflife_days=args.pairwise_time_decay_halflife_days,
        run_tag=args.run_tag,
        judge_model=args.judge_model,
        judge_model_secondary=args.judge_model_secondary,
        judge_disagreement_threshold=args.judge_disagreement_threshold,
        safety_thresholds_path=args.safety_thresholds,
        deep_dive_top_k=args.deep_dive_top_k,
        seed=args.seed,
        dry_run=args.dry_run,
        request_interval=args.request_interval,
    )

    runner   = EvalRunner(cfg)
    estimates = runner.estimate_cost()

    print("\nEstimated cost (before running):")
    for model, cost in estimates.items():
        extras = []
        if cfg.run_robustness:
            extras.append("robustness")
        if cfg.run_quirk_detection:
            extras.append("quirk-detection")
        if cfg.run_linkedin_style and "linkedin" not in cfg.style_suites:
            extras.append("linkedin-style")
        if cfg.style_suites:
            extras.append("style-suites=" + ",".join(cfg.style_suites))
        if cfg.run_persona_transfer:
            extras.append("persona-transfer")
        if cfg.run_pairwise_preference:
            pairwise_label = "pairwise-preference"
            if cfg.pairwise_detector:
                pairwise_label += f"={cfg.pairwise_detector}"
            pairwise_label += (
                f",method={cfg.pairwise_method},segment={cfg.pairwise_segment_by}"
            )
            if cfg.pairwise_time_decay_halflife_days:
                pairwise_label += f",halflife_days={cfg.pairwise_time_decay_halflife_days}"
            extras.append(pairwise_label)
        if cfg.run_tag:
            extras.append(f"run-tag={cfg.run_tag}")
        if cfg.judge_model:
            extras.append(f"judge-model={cfg.judge_model}")
        if cfg.judge_model_secondary:
            extras.append(f"judge-model-secondary={cfg.judge_model_secondary}")
        if cfg.deep_dive_top_k > 0:
            extras.append(f"deep-dive-top-k={cfg.deep_dive_top_k}")
        extra_str = f"  [{', '.join(extras)}]" if extras else ""
        print(f"  {model:<20}  ~${cost:.4f}{extra_str}")

    if args.dry_run:
        print("\nDry run — remove --dry-run to execute.")
        return

    print()
    results = runner.run()

    json_path, txt_path = rpt.save(results, cfg.output_dir)
    safety_path = rpt.save_safety_case(
        results,
        cfg.output_dir,
        thresholds_path=cfg.safety_thresholds_path,
        deep_dive_top_k=cfg.deep_dive_top_k,
    )
    print()
    print(rpt.format_table(results))
    print(f"\nFull JSON : {json_path}")
    print(f"Summary   : {txt_path}")
    if safety_path:
        print(f"Safety Case: {safety_path}")


if __name__ == "__main__":
    main()
