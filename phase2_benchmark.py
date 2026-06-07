#!/usr/bin/env python3
"""
Constitutional Compliance Benchmark Runner.

Evaluates models across multiple evaluation suites using LLM-based and heuristic judges.
Supports local (Ollama) and cloud-based (OpenAI/Anthropic) models for evaluation,
with separate configurable judge model for scoring.

Usage:
    python phase2_benchmark.py --models llama3.1 mistral phi
    python phase2_benchmark.py --models gpt-4o-mini claude-haiku --judge-model gpt-4o
    python phase2_benchmark.py --suites task_focus truthfulness --samples 5 --resume
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from compass import (
    evaluate_suite_completions,
    generate_suite_completions,
)
from helm_eval_lex_quirks.metrics.builtin_tic_suites import (
    STYLE_SUITE_REGISTRY,
)
from helm_eval_lex_quirks.metrics.tic_suite import (
    LLMJudgeDetector,
    wilson_interval,
)
from helm_eval_lex_quirks.models.anthropic_client import AnthropicClient
from helm_eval_lex_quirks.models.gemini_client import GeminiClient
from helm_eval_lex_quirks.models.ollama_client import OllamaClient
from helm_eval_lex_quirks.models.openai_client import OpenAIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_client(model: str, disable_safety_filters: bool = False) -> Tuple[object, str]:
    """
    Create appropriate client for model.

    Returns:
        (client, client_type) where client_type is 'ollama', 'openai', 'anthropic', or 'gemini'
    """
    # Local models via Ollama
    ollama_models = {"llama3.1", "mistral", "phi", "llama3", "llama3.2"}
    if any(model.startswith(m) for m in ollama_models):
        model_name = f"{model}:latest" if ":" not in model else model
        return OllamaClient(model=model_name), "ollama"

    # Cloud models
    if "gemini" in model.lower():
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        return GeminiClient(
            api_key=api_key, model=model,
            disable_safety_filters=disable_safety_filters,
        ), "gemini"
    elif model.startswith("gpt-5"):
        from helm_eval_lex_quirks.models.openai_responses_client import OpenAIResponsesClient
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIResponsesClient(api_key=api_key, model=model), "openai"
    elif "gpt" in model.lower():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIClient(api_key=api_key, model=model), "openai"
    elif "claude" in model.lower():
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return AnthropicClient(api_key=api_key, model=model), "anthropic"
    else:
        raise ValueError(f"Unsupported model: {model}")


def load_checkpoint(checkpoint_file: Path) -> Set[Tuple[str, str, str, str, str, int]]:
    """
    Load completed evaluation identities from checkpoint JSONL.

    Returns a set of (model, suite, detector, prompt_id, condition, sample_idx) tuples
    that have already been evaluated.

    Checkpoints written before sample_idx existed are treated as sample_idx=0 for
    backward compatibility.
    """
    completed = set()
    if not checkpoint_file.exists():
        return completed

    legacy_records = 0
    with open(checkpoint_file) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                sample_idx = result.get("sample_idx", 0)
                missing_sample_idx = "sample_idx" not in result
                if not isinstance(sample_idx, int):
                    # Legacy/manual checkpoints may serialize this as a string.
                    sample_idx = int(sample_idx)
                if missing_sample_idx:
                    legacy_records += 1
                key = (
                    result["model"],
                    result["suite"],
                    result["detector"],
                    result["prompt_id"],
                    result["condition"],
                    sample_idx,
                )
                completed.add(key)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON at checkpoint line {line_num}")
                continue
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    f"Skipping invalid checkpoint record at line {line_num}"
                )
                continue

    if legacy_records:
        logger.warning(
            "Loaded %d legacy checkpoint records without sample_idx; "
            "defaulted them to sample_idx=0.",
            legacy_records,
        )

    return completed


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the most recently modified benchmark checkpoint in output_dir."""
    checkpoints = list(output_dir.glob("benchmark.jsonl"))
    if not checkpoints:
        return None
    # Return the first (and should be only) benchmark.jsonl
    return checkpoints[0]


def _dry_run_plan(
    models: List[str],
    suites: List[str],
    samples_per_condition: int,
    judge_model: str,
    temperature: float,
) -> None:
    """Log the run plan and API-call exposure without calling any model.

    Reports generation and judge call counts per suite plus a rough cost
    estimate, so a real run's spend is visible before it starts.
    """
    from helm_eval_lex_quirks.config import model_config

    logger.info("DRY RUN — no API calls will be made")
    logger.info(
        f"  Models: {', '.join(models)} | judge: {judge_model} | temperature: {temperature}"
    )

    total_gen = 0
    total_judge = 0
    for suite_name in suites:
        if suite_name not in STYLE_SUITE_REGISTRY:
            logger.error(f"  Unknown suite: {suite_name}")
            continue
        suite = STYLE_SUITE_REGISTRY[suite_name]
        n_prompts = len(suite.prompts)
        n_conditions = len(suite.conditions)
        n_judge_detectors = sum(
            isinstance(d, LLMJudgeDetector) for d in suite.detectors
        )
        gen_calls = n_prompts * n_conditions * samples_per_condition * len(models)
        judge_calls = gen_calls * n_judge_detectors
        total_gen += gen_calls
        total_judge += judge_calls
        logger.info(
            f"  [{suite_name}] {n_prompts} prompts x {n_conditions} conditions x "
            f"{samples_per_condition} samples x {len(models)} models = {gen_calls} generations; "
            f"{n_judge_detectors} judge detector(s) -> {judge_calls} judge calls"
        )
        if n_judge_detectors and not judge_model:
            logger.warning(
                f"  [{suite_name}] has judge detectors but no --judge-model set"
            )

    # Rough cost estimate; assumes ~100 in / ~200 out per generation and
    # ~450 in / ~50 out per judge call. Order of magnitude, not a quote.
    gen_per_model = total_gen // max(len(models), 1)
    est = 0.0
    for model in models:
        mc = model_config(model)
        est += gen_per_model * (
            100 * mc.input_cost_per_million + 200 * mc.output_cost_per_million
        ) / 1_000_000
    if judge_model and total_judge:
        jc = model_config(judge_model)
        est += total_judge * (
            450 * jc.input_cost_per_million + 50 * jc.output_cost_per_million
        ) / 1_000_000

    logger.info(
        f"  TOTAL API calls: {total_gen} generations + {total_judge} judge = "
        f"{total_gen + total_judge}"
    )
    logger.info(f"  Rough cost estimate: ~${est:.2f} (order of magnitude)")
    logger.info(
        "  Set the relevant API keys (e.g. ANTHROPIC_API_KEY) before the real run."
    )


def run_benchmark(
    models: List[str],
    judge_model: str,
    suites: List[str],
    samples_per_condition: int = 20,
    output_dir: str = "results/benchmark",
    resume: bool = False,
    temperature: float = 0.0,
    auto_resume: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Run the two-phase suite benchmark.

    Generation and scoring are decoupled: each suite's completions are generated
    once per (model, prompt, condition, sample) and persisted, then scored with
    the suite's detectors. Re-running scoring (for example with a different judge)
    does not regenerate. LLM-judge detectors are prompt-aware: the judge sees the
    user request, so it can tell a volunteered suggestion from a solicited answer.

    Args:
        models: Models under test.
        judge_model: Judge model for LLM-judge detectors (ignored by heuristic-only suites).
        suites: Suite names to run.
        samples_per_condition: Completions per (prompt, condition) cell.
        output_dir: Root directory; each suite writes under output_dir/<suite>.
        resume: Continue from persisted generations instead of regenerating.
        temperature: Sampling temperature for the model under test. Keep 0.0 for
            deterministic suites; raise above 0.0 for suites whose per-condition
            samples are meant to be independent draws (e.g. rest_cai_validation).
            The judge stays deterministic.
        auto_resume: Alias for resume kept for CLI compatibility; either flag
            preserves existing completions rather than discarding them.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    keep_generations = resume or auto_resume

    logger.info("Benchmark Runner")
    logger.info(f"  Models: {', '.join(models)}")
    logger.info(f"  Suites: {', '.join(suites)}")
    logger.info(
        f"  Samples/condition: {samples_per_condition}  temperature: {temperature}"
    )
    logger.info(f"  Mode: {'RESUME generations' if keep_generations else 'NEW RUN'}")

    if dry_run:
        _dry_run_plan(models, suites, samples_per_condition, judge_model, temperature)
        return

    eval_paths: Dict[str, Path] = {}
    for suite_name in suites:
        if suite_name not in STYLE_SUITE_REGISTRY:
            logger.error(f"Unknown suite: {suite_name}")
            continue

        suite = STYLE_SUITE_REGISTRY[suite_name]
        suite_dir = output_path / suite_name

        def client_factory(model: str, _suite_name: str = suite_name):
            client, _ = create_client(
                model, disable_safety_filters=(_suite_name == "self_harm_safety")
            )
            return client

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Suite: {suite_name} ({len(suite.detectors)} detectors)")
        logger.info(f"{'=' * 70}")

        generations_path = generate_suite_completions(
            suite,
            models=models,
            samples=samples_per_condition,
            output_dir=suite_dir,
            temperature=temperature,
            resume=keep_generations,
            client_factory=client_factory,
        )

        needs_judge = any(
            isinstance(detector, LLMJudgeDetector) for detector in suite.detectors
        )
        eval_paths[suite_name] = evaluate_suite_completions(
            suite,
            generations_path,
            suite_dir,
            judge_model=judge_model if needs_judge else None,
            resume=True,
            client_factory=client_factory,
        )

    summary_file = output_path / f"benchmark_summary_{timestamp}.txt"
    _write_summary(eval_paths, summary_file)
    logger.info(f"✓ Benchmark complete; summary: {summary_file}")


def _write_summary(eval_paths: Dict[str, Path], summary_file: Path) -> None:
    """Aggregate per-(model, condition, detector) hit rates with Wilson CIs.

    Breaking the summary down by model is what makes the cross-model comparison
    (e.g. opus vs sonnet vs haiku) readable directly from one table.
    """
    from collections import defaultdict

    agg: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "total": 0})
    for path in eval_paths.values():
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (row["model"], row["suite"], row["condition"], row["detector"])
                agg[key]["hits"] += int(bool(row.get("hit")))
                agg[key]["total"] += 1

    with open(summary_file, "w") as f:
        f.write("Constitutional Compliance Benchmark Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("Model | Suite | Condition | Detector | Hit Rate | 95% CI (Wilson) | n\n")
        f.write("-" * 110 + "\n")
        for key in sorted(agg):
            model, suite_name, condition, detector = key
            stats = agg[key]
            total = stats["total"]
            pct = 100 * stats["hits"] / total if total else 0.0
            ci = wilson_interval(stats["hits"], total)
            f.write(
                f"{model} | {suite_name} | {condition} | {detector} | "
                f"{pct:5.1f}% | [{ci['low']:5.1f}%, {ci['high']:5.1f}%] | {total:3d}\n"
            )
        f.write("\n" + "=" * 90 + "\n")
        f.write(f"Total evaluations: {sum(s['total'] for s in agg.values())}\n")

    logger.info(f"✓ Summary saved: {summary_file}")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Constitutional Compliance Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate local models (no API keys required)
  python phase2_benchmark.py --models llama3.1 mistral phi

  # Evaluate cloud models with LLM judge
  python phase2_benchmark.py --models gpt-4o-mini claude-haiku --judge-model gpt-4o

  # Custom configuration
  python phase2_benchmark.py \\
    --models llama3.1 gpt-4o-mini claude-haiku \\
    --judge-model gpt-4o \\
    --suites task_focus truthfulness \\
    --samples 10

  # Resume from checkpoint
  python phase2_benchmark.py --models llama3.1 --resume
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.1", "mistral", "phi"],
        help="Models to evaluate (default: llama3.1 mistral phi)",
    )

    parser.add_argument(
        "--suites",
        nargs="+",
        default=["task_focus", "truthfulness"],
        help="Suites to run (default: task_focus truthfulness)",
    )

    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Judge model for LLM-based detectors (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Samples per condition (default: 20)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature for the model under test (default: 0.0, "
            "deterministic). Raise above 0.0 for suites whose per-condition samples "
            "should be independent draws (e.g. rest_cai_validation); at 0.0 repeated "
            "samples are near-identical and confidence intervals understate uncertainty."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="results/benchmark",
        help="Output directory (default: results/benchmark)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (load latest results)",
    )

    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "Continue from an existing non-empty checkpoint in --output-dir instead "
            "of discarding it. Use this to safely re-invoke the same command without "
            "re-paying for completed evaluations. No effect if --resume is given or "
            "no checkpoint exists."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan and API-call exposure, then exit without calling any model.",
    )

    args = parser.parse_args(argv)

    run_benchmark(
        models=args.models,
        judge_model=args.judge_model,
        suites=args.suites,
        samples_per_condition=args.samples,
        output_dir=args.output_dir,
        resume=args.resume,
        temperature=args.temperature,
        auto_resume=args.auto_resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
