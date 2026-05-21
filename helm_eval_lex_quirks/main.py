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
"""
import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from helm_eval_lex_quirks.config import DEFAULT_SCENARIO_CONFIGS, EvalConfig, model_provider
from helm_eval_lex_quirks.metrics.builtin_tic_suites import STYLE_SUITE_REGISTRY, expand_suite_names
from helm_eval_lex_quirks.runner import SCENARIO_REGISTRY, EvalRunner
from helm_eval_lex_quirks import report as rpt


def main(argv=None):
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
                        help="Run generic tic/style suite(s): lexical, punctuation, formatting, tone, linkedin, or all")
    parser.add_argument("--style-only", action="store_true",
                        help="Run only selected generic style suites, skip benchmark scenarios")
    parser.add_argument("--style-samples", type=int, default=4,
                        help="Repetitions per prompt for generic tic/style suites")
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
    if args.style_only:
        args.no_robustness = True
        args.scenarios = []
        if not style_suites:
            print("ERROR: --style-only requires at least one --style-suite.", file=sys.stderr)
            sys.exit(1)
    providers = {model_provider(model) for model in args.models}
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
        extra_str = f"  [{', '.join(extras)}]" if extras else ""
        print(f"  {model:<20}  ~${cost:.4f}{extra_str}")

    if args.dry_run:
        print("\nDry run — remove --dry-run to execute.")
        return

    print()
    results = runner.run()

    json_path, txt_path = rpt.save(results, cfg.output_dir)
    print()
    print(rpt.format_table(results))
    print(f"\nFull JSON : {json_path}")
    print(f"Summary   : {txt_path}")


if __name__ == "__main__":
    main()
