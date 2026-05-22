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
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from helm_eval_lex_quirks.metrics.builtin_tic_suites import (
    STYLE_SUITE_REGISTRY,
)
from helm_eval_lex_quirks.metrics.tic_suite import (
    LLMJudgeDetector,
    LLM_JUDGE_SYSTEM_PROMPT,
)
from helm_eval_lex_quirks.models.anthropic_client import AnthropicClient
from helm_eval_lex_quirks.models.ollama_client import OllamaClient
from helm_eval_lex_quirks.models.openai_client import OpenAIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single evaluation result."""

    model: str
    suite: str
    detector: str
    prompt_id: str
    condition: str
    score: float
    hit: bool
    confidence: Optional[float] = None
    rationale: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)


def create_client(model: str) -> Tuple[object, str]:
    """
    Create appropriate client for model.

    Returns:
        (client, client_type) where client_type is 'ollama', 'openai', or 'anthropic'
    """
    # Local models via Ollama
    ollama_models = {"llama3.1", "mistral", "phi", "llama3", "llama3.2"}
    if any(model.startswith(m) for m in ollama_models):
        model_name = f"{model}:latest" if ":" not in model else model
        return OllamaClient(model=model_name), "ollama"

    # Cloud models
    if "gpt" in model.lower():
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


def load_checkpoint(checkpoint_file: Path) -> Dict[Tuple, bool]:
    """
    Load completed evaluation identities from checkpoint JSONL.

    Returns a set of (model, suite, detector, prompt_id, condition, sample_idx) tuples
    that have already been evaluated.
    """
    completed = set()
    if not checkpoint_file.exists():
        return completed

    with open(checkpoint_file) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                # Use full identity tuple including sample_idx
                # sample_idx is implicit from order, but include all identifying fields
                key = (
                    result["model"],
                    result["suite"],
                    result["detector"],
                    result["prompt_id"],
                    result["condition"],
                )
                completed.add(key)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON at checkpoint line {line_num}")
                continue

    return completed


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the most recently modified benchmark checkpoint in output_dir."""
    checkpoints = list(output_dir.glob("benchmark.jsonl"))
    if not checkpoints:
        return None
    # Return the first (and should be only) benchmark.jsonl
    return checkpoints[0]


def run_benchmark(
    models: List[str],
    judge_model: str,
    suites: List[str],
    samples_per_condition: int = 5,
    output_dir: str = "results/benchmark",
    resume: bool = False,
) -> None:
    """
    Run benchmark evaluation.

    Args:
        models: List of model names to evaluate
        judge_model: Model to use for LLM-based rubrics
        suites: List of suite names to run
        samples_per_condition: Samples per condition per suite
        output_dir: Where to save results
        resume: Whether to load and continue from latest checkpoint
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use fixed checkpoint filename for resume capability
    checkpoint_file = output_path / "benchmark.jsonl"

    logger.info("Benchmark Runner")
    logger.info(f"  Models to evaluate: {', '.join(models)}")
    logger.info(f"  Judge model: {judge_model}")
    logger.info(f"  Suites: {', '.join(suites)}")
    logger.info(f"  Checkpoint: {checkpoint_file}")
    logger.info(f"  Mode: {'RESUME' if resume else 'NEW RUN'}")

    # Load judge client once (shared across all evaluations)
    logger.info(f"\nInitializing judge client: {judge_model}")
    try:
        judge_client, judge_type = create_client(judge_model)
        logger.info(f"  Judge type: {judge_type}")
    except Exception as e:
        logger.error(f"Failed to create judge client: {e}")
        sys.exit(1)

    # Load checkpoint if resuming
    completed = load_checkpoint(checkpoint_file) if resume else set()
    logger.info(f"Checkpoint has {len(completed)} completed evaluations")

    results_written = 0

    for suite_name in suites:
        if suite_name not in STYLE_SUITE_REGISTRY:
            logger.error(f"Unknown suite: {suite_name}")
            continue

        suite = STYLE_SUITE_REGISTRY[suite_name]
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Suite: {suite_name} ({len(suite.detectors)} detectors)")
        logger.info(f"{'=' * 70}")

        for model in models:
            logger.info(f"\n  Model: {model}")

            try:
                client, client_type = create_client(model)
                logger.info(f"    Client type: {client_type}")
            except Exception as e:
                logger.error(f"    Failed to create client: {e}")
                continue

            # Iterate through prompts and conditions
            for prompt in suite.prompts:
                for condition in suite.conditions:
                    for sample_idx in range(samples_per_condition):
                        # Check if all detectors for this sample are already completed
                        sample_key = (model, suite_name, prompt.id, condition.name)
                        all_detectors_done = all(
                            (model, suite_name, d.name, prompt.id, condition.name) in completed
                            for d in suite.detectors
                        )

                        if all_detectors_done:
                            logger.debug(
                                f"    Skipping (all detectors completed): "
                                f"{prompt.id} [{condition.name}] sample {sample_idx + 1}"
                            )
                            continue

                        try:
                            # Generate completion
                            response = client.complete(
                                prompt=prompt.text,
                                max_tokens=256,
                                temperature=0.0,
                                system=condition.system_prompt,
                            )

                            # Run detectors
                            for detector in suite.detectors:
                                try:
                                    if isinstance(detector, LLMJudgeDetector):
                                        # Create judge function with strict system prompt
                                        def judge_fn(prompt_text: str, max_tokens: int) -> str:
                                            judge_response = judge_client.complete(
                                                prompt=prompt_text,
                                                max_tokens=max_tokens,
                                                temperature=0.0,
                                                system=LLM_JUDGE_SYSTEM_PROMPT,
                                            )
                                            return judge_response.completion

                                        detector_result = detector.detect_with_judge(
                                            text=response.completion,
                                            judge_fn=judge_fn,
                                        )
                                    else:
                                        # Regular heuristic detectors
                                        detector_result = detector.detect(
                                            text=response.completion
                                        )

                                    # Extract metadata
                                    confidence = detector_result.metadata.get("confidence")
                                    rationale = detector_result.metadata.get("rationale", "")

                                    # Create result
                                    bench_result = BenchmarkResult(
                                        model=model,
                                        suite=suite_name,
                                        detector=detector.name,
                                        prompt_id=prompt.id,
                                        condition=condition.name,
                                        score=detector_result.score,
                                        hit=detector_result.hit,
                                        confidence=confidence,
                                        rationale=rationale,
                                        input_tokens=response.input_tokens,
                                        output_tokens=response.output_tokens,
                                    )

                                    # Write to checkpoint immediately
                                    with open(checkpoint_file, "a") as f:
                                        f.write(json.dumps(bench_result.to_dict()) + "\n")
                                    results_written += 1

                                except Exception as e:
                                    logger.error(
                                        f"    Detector {detector.name} failed for "
                                        f"{prompt.id} [{condition.name}]: {e}"
                                    )
                                    continue

                            logger.info(
                                f"    [{suite_name:20}] {prompt.id:15} "
                                f"[{condition.name:12}] sample {sample_idx + 1}/{samples_per_condition}"
                            )

                        except Exception as e:
                            logger.error(
                                f"    Failed to evaluate {prompt.id} "
                                f"[{condition.name}] sample {sample_idx + 1}: {e}"
                            )
                            continue

    logger.info(f"\n{'=' * 70}")
    logger.info(f"✓ Benchmark complete")
    logger.info(f"  Results written: {results_written}")
    logger.info(f"  Checkpoint: {checkpoint_file}")

    # Generate summary
    _generate_summary(checkpoint_file, output_path / f"benchmark_summary_{timestamp}.txt")


def _generate_summary(checkpoint_file: Path, summary_file: Path) -> None:
    """Generate human-readable summary from checkpoint."""
    from collections import defaultdict

    summary = defaultdict(lambda: {"hits": 0, "total": 0, "scores": []})

    with open(checkpoint_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                key = f"{result['model']} | {result['suite']} | {result['detector']}"
                summary[key]["hits"] += int(result["hit"])
                summary[key]["total"] += 1
                summary[key]["scores"].append(result["score"])
            except json.JSONDecodeError:
                continue

    with open(summary_file, "w") as f:
        f.write("Constitutional Compliance Benchmark Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Checkpoint: {checkpoint_file}\n\n")

        f.write("Model | Suite | Detector | Hit Rate | Mean Score | Samples\n")
        f.write("-" * 90 + "\n")

        for key in sorted(summary.keys()):
            stats = summary[key]
            hit_rate = 100 * stats["hits"] / stats["total"] if stats["total"] > 0 else 0
            mean_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            f.write(
                f"{key} | {hit_rate:5.1f}% | {mean_score:6.3f} | {stats['total']:3d}\n"
            )

        f.write("\n" + "=" * 90 + "\n")
        f.write(f"Total evaluations: {sum(s['total'] for s in summary.values())}\n")

    logger.info(f"✓ Summary saved: {summary_file}")


def main():
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
        default=5,
        help="Samples per condition (default: 5)",
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

    args = parser.parse_args()

    run_benchmark(
        models=args.models,
        judge_model=args.judge_model,
        suites=args.suites,
        samples_per_condition=args.samples,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
