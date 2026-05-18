"""
HELM-style evaluation harness.

Usage examples
--------------
# Dry run — estimate cost only
python -m helm_eval.main --dry-run

# Full eval on gpt-4o-mini with defaults (100 samples/scenario)
python -m helm_eval.main --api-key sk-...

# Cheaper quick smoke-test (20 samples, no robustness)
python -m helm_eval.main --api-key sk-... --n-samples 20 --no-robustness --no-quirks

# Evaluate multiple models
python -m helm_eval.main --api-key sk-... --models gpt-4o-mini gpt-4o
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

from helm_eval.config import DEFAULT_SCENARIO_CONFIGS, EvalConfig
from helm_eval.runner import SCENARIO_REGISTRY, EvalRunner
from helm_eval import report as rpt


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="HELM-style eval harness for OpenAI models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-key",   default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--models",    nargs="+", default=["gpt-4o-mini"],
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIO_CONFIGS),
                        choices=list(SCENARIO_REGISTRY))
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override the per-scenario sample count")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-robustness",  action="store_true",
                        help="Skip robustness (typo-perturbation) evaluation")
    parser.add_argument("--no-quirks", action="store_true",
                        help="Skip goblin-style lexical quirk detection")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--request-interval", type=float, default=2.0,
                        help="Seconds between API calls. Raise to 4-6 for free-tier accounts.")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Estimate cost and exit without calling the API")
    args = parser.parse_args(argv)

    if not args.api_key:
        print("ERROR: supply --api-key or set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    cfg = EvalConfig(
        api_key=args.api_key,
        models=args.models,
        scenarios=args.scenarios,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        run_robustness=not args.no_robustness,
        run_quirk_detection=not args.no_quirks,
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
