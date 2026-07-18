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
import inspect
import json
import logging
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from compass import (
    evaluate_suite_completions,
    generate_suite_completions,
)
from compass.evaluation import load_suite_generations
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
from helm_eval_lex_quirks.models.openai_responses_client import OpenAIResponsesClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SUITE_TOKEN_BUDGETS = {
    "default": 1000,
    "gemini": 2000,
}


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


def _default_suite_max_tokens_for_model(model: str) -> int:
    prefixes = sorted(
        (prefix for prefix in DEFAULT_SUITE_TOKEN_BUDGETS if prefix != "default"),
        key=len,
        reverse=True,
    )
    for prefix in prefixes:
        if model.startswith(prefix):
            return int(DEFAULT_SUITE_TOKEN_BUDGETS[prefix])
    return int(DEFAULT_SUITE_TOKEN_BUDGETS["default"])


def _parse_max_tokens_by_model(items: Optional[List[str]]) -> Dict[str, int]:
    budgets: Dict[str, int] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid --max-tokens-by-model entry {item!r}; expected MODEL=TOKENS."
            )
        model, raw_budget = item.split("=", 1)
        model = model.strip()
        if not model:
            raise ValueError("Model name cannot be empty in --max-tokens-by-model.")
        try:
            budget = int(raw_budget)
        except ValueError as exc:
            raise ValueError(
                f"Invalid token budget {raw_budget!r} for model {model!r}."
            ) from exc
        if budget <= 0:
            raise ValueError(
                f"Invalid token budget {budget!r} for model {model!r}; must be > 0."
            )
        budgets[model] = budget
    return budgets


def _effective_max_tokens_by_model(
    models: List[str],
    max_tokens_by_model: Optional[Dict[str, int]] = None,
    allow_mixed_token_budgets: bool = True,
) -> Dict[str, int]:
    if max_tokens_by_model is None:
        budgets = {
            model: _default_suite_max_tokens_for_model(model) for model in models
        }
    else:
        budgets = {}
        for model in models:
            if model not in max_tokens_by_model:
                raise ValueError(
                    f"Missing max token budget for model {model!r} in custom map."
                )
            budgets[model] = int(max_tokens_by_model[model])
    distinct = sorted(set(budgets.values()))
    if len(distinct) > 1 and not allow_mixed_token_budgets:
        raise ValueError(
            "Mixed max token budgets detected across models: "
            f"{budgets}. Re-run without --disallow-mixed-token-budgets to override."
        )
    return budgets


def _dry_run_plan(
    models: List[str],
    suites: List[str],
    samples_per_condition: int,
    judge_model: str,
    temperature: float,
    max_tokens_by_model: Optional[Dict[str, int]] = None,
    allow_mixed_token_budgets: bool = True,
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
    effective_budgets = _effective_max_tokens_by_model(
        models,
        max_tokens_by_model,
        allow_mixed_token_budgets=allow_mixed_token_budgets,
    )
    logger.info(f"  Generation max tokens by model: {effective_budgets}")

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

    # Rough cost estimate; assumes ~100 input tokens per generation plus the
    # configured generation budget. GPT-5 Responses calls use hidden reasoning
    # headroom, so estimate their output exposure with the same 10x multiplier
    # used by the model wrapper.
    # ~450 in / ~50 out per judge call. Order of magnitude, not a quote.
    gen_per_model = total_gen // max(len(models), 1)
    est = 0.0
    for model in models:
        mc = model_config(model)
        output_budget = effective_budgets[model]
        if model.startswith("gpt-5"):
            output_budget *= 10
        est += gen_per_model * (
            100 * mc.input_cost_per_million
            + output_budget * mc.output_cost_per_million
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


def _generate_suite_completions_compat(
    suite,
    *,
    models: List[str],
    samples: int,
    output_dir: Path,
    temperature: float,
    resume: bool,
    max_tokens_by_model: Dict[str, int],
    allow_mixed_token_budgets: bool,
    client_factory,
) -> Path:
    """Call Compass suite generation across old and new suite_runner APIs."""
    generation_params = inspect.signature(generate_suite_completions).parameters
    if "max_tokens_by_model" in generation_params:
        return generate_suite_completions(
            suite,
            models=models,
            samples=samples,
            output_dir=output_dir,
            temperature=temperature,
            resume=resume,
            max_tokens_by_model=max_tokens_by_model,
            allow_mixed_token_budgets=allow_mixed_token_budgets,
            client_factory=client_factory,
        )

    generations_path: Optional[Path] = None
    for idx, model in enumerate(models):
        model_suite = replace(suite, max_tokens=max_tokens_by_model[model])
        generations_path = generate_suite_completions(
            model_suite,
            models=[model],
            samples=samples,
            output_dir=output_dir,
            temperature=temperature,
            resume=resume or idx > 0,
            client_factory=client_factory,
        )
    if generations_path is None:
        raise ValueError("At least one model is required.")
    return generations_path


def run_benchmark(
    models: List[str],
    judge_model: str,
    suites: List[str],
    samples_per_condition: int = 20,
    output_dir: str = "results/benchmark",
    resume: bool = False,
    temperature: float = 0.0,
    auto_resume: bool = True,
    max_tokens_by_model: Optional[Dict[str, int]] = None,
    allow_mixed_token_budgets: bool = True,
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
        auto_resume: Preserve existing completions by default so interrupted or
            repeated runs do not discard paid generations. Disable only for an
            intentionally fresh run.
        max_tokens_by_model: Optional generation token budget per model. If not
            provided, Compass suite defaults are used.
        allow_mixed_token_budgets: Allow different generation token budgets
            across models.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    keep_generations = resume or auto_resume

    logger.info("Benchmark Runner")
    logger.info(f"  Models: {', '.join(models)}")
    logger.info(f"  Suites: {', '.join(suites)}")
    logger.info(
        f"  Samples/condition: {samples_per_condition}  temperature: {temperature}"
    )
    effective_budgets = _effective_max_tokens_by_model(
        models,
        max_tokens_by_model,
        allow_mixed_token_budgets=allow_mixed_token_budgets,
    )
    logger.info(f"  Generation max tokens by model: {effective_budgets}")
    logger.info(f"  Mode: {'RESUME generations' if keep_generations else 'NEW RUN'}")

    if dry_run:
        _dry_run_plan(
            models,
            suites,
            samples_per_condition,
            judge_model,
            temperature,
            max_tokens_by_model=effective_budgets,
            allow_mixed_token_budgets=allow_mixed_token_budgets,
        )
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

        needs_judge = any(
            isinstance(detector, LLMJudgeDetector) for detector in suite.detectors
        )
        judge_tag = judge_model if needs_judge else "heuristic"
        evaluations_path = suite_dir / f"suite_evaluations_{judge_tag}.jsonl"
        checkpoint_path = suite_dir / "suite_generations.jsonl"
        existing_generations = len(load_suite_generations(checkpoint_path))

        generations_path = _generate_suite_completions_compat(
            suite,
            models=models,
            samples=samples_per_condition,
            output_dir=suite_dir,
            temperature=temperature,
            resume=keep_generations,
            max_tokens_by_model=effective_budgets,
            allow_mixed_token_budgets=allow_mixed_token_budgets,
            client_factory=client_factory,
        )

        resume_evaluations = keep_generations and existing_generations > 0
        if resume_evaluations and evaluations_path.exists():
            generations_mtime = generations_path.stat().st_mtime
            evaluations_mtime = evaluations_path.stat().st_mtime
            if generations_mtime > evaluations_mtime:
                resume_evaluations = False
                logger.info(
                    "Suite generations are newer than evaluations; stale evaluations "
                    "will be reset"
                )
        elif keep_generations and not resume_evaluations:
            logger.info("No existing suite generations found; starting fresh")

        eval_paths[suite_name] = evaluate_suite_completions(
            suite,
            generations_path,
            suite_dir,
            judge_model=judge_model if needs_judge else None,
            resume=resume_evaluations,
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

  # Force a fresh run, discarding existing suite generations/evaluations
  python phase2_benchmark.py --models llama3.1 --no-auto-resume
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
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Uniform generation token budget for every model. If omitted, "
            "suite benchmark defaults are used."
        ),
    )

    parser.add_argument(
        "--max-tokens-by-model",
        nargs="*",
        default=None,
        help=(
            "Per-model generation token budgets, e.g. "
            "claude-opus-4-6=200 gpt-5-mini=200."
        ),
    )

    parser.add_argument(
        "--disallow-mixed-token-budgets",
        dest="allow_mixed_token_budgets",
        action="store_false",
        default=True,
        help="Fail if effective generation token budgets differ across models.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (load latest results)",
    )

    parser.add_argument(
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        default=True,
        help=(
            "Continue from existing suite generations/evaluations in --output-dir "
            "instead of discarding them (default)."
        ),
    )

    parser.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help=(
            "Force a fresh run by discarding existing suite generations and "
            "evaluations for the selected suites."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan and API-call exposure, then exit without calling any model.",
    )

    args = parser.parse_args(argv)
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    try:
        custom_budgets = _parse_max_tokens_by_model(args.max_tokens_by_model)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_tokens is not None and custom_budgets:
        parser.error("--max-tokens and --max-tokens-by-model cannot be combined")
    max_tokens_by_model = None
    if args.max_tokens is not None:
        max_tokens_by_model = {model: args.max_tokens for model in args.models}
    elif custom_budgets:
        max_tokens_by_model = custom_budgets

    run_benchmark(
        models=args.models,
        judge_model=args.judge_model,
        suites=args.suites,
        samples_per_condition=args.samples,
        output_dir=args.output_dir,
        resume=args.resume,
        temperature=args.temperature,
        auto_resume=args.auto_resume,
        max_tokens_by_model=max_tokens_by_model,
        allow_mixed_token_budgets=args.allow_mixed_token_budgets,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
