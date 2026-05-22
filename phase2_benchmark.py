#!/usr/bin/env python3
"""
Phase 2 Constitutional Compliance Benchmark Runner.

Evaluates models across 5 constitutional rubrics using local (Ollama) and
cloud-based (OpenAI/Anthropic) judges.

Usage:
    python phase2_benchmark.py --models llama3.1 mistral phi --output-dir results/phase2
    python phase2_benchmark.py --models gpt-4o-mini claude-haiku --judge-model gpt-4o --samples 10
"""
import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from helm_eval_lex_quirks.config import ModelConfig, model_config
from helm_eval_lex_quirks.metrics.builtin_tic_suites import (
    STYLE_SUITE_REGISTRY,
)
from helm_eval_lex_quirks.metrics.tic_suite import LLMJudgeDetector
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
    tokens_used: Dict = None

    def to_dict(self):
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        if self.tokens_used is None:
            d["tokens_used"] = {}
        return d


def create_client(model: str, judge_model: str) -> Tuple[object, str]:
    """
    Create appropriate client for model.

    Returns:
        (client, client_type) where client_type is 'ollama', 'openai', or 'anthropic'
    """
    # Local models via Ollama
    ollama_models = {"llama3.1", "mistral", "phi", "llama3", "llama3.2"}
    if any(model.startswith(m) for m in ollama_models):
        return OllamaClient(model=f"{model}:latest" if ":" not in model else model), "ollama"

    # Cloud models
    try:
        cfg = model_config(model)
    except ValueError:
        raise ValueError(f"Unknown model: {model}")

    if "gpt" in model.lower():
        return OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model=model), "openai"
    elif "claude" in model.lower():
        return (
            AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"), model=model),
            "anthropic",
        )
    else:
        raise ValueError(f"Unsupported model: {model}")


def run_benchmark(
    models: List[str],
    judge_model: str,
    suites: List[str],
    samples_per_condition: int = 5,
    output_dir: str = "results/phase2",
) -> None:
    """
    Run Phase 2 benchmark.

    Args:
        models: List of model names to benchmark
        judge_model: Judge model for LLM-based rubrics
        suites: List of suite names to run
        samples_per_condition: Samples per condition per suite
        output_dir: Where to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Phase 2 Benchmark Runner")
    logger.info(f"  Models: {', '.join(models)}")
    logger.info(f"  Judge: {judge_model}")
    logger.info(f"  Suites: {', '.join(suites)}")
    logger.info(f"  Output: {output_path.absolute()}")

    results: List[BenchmarkResult] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
                client, client_type = create_client(model, judge_model)
                logger.info(f"    Client type: {client_type}")
            except Exception as e:
                logger.error(f"    Failed to create client: {e}")
                continue

            # Iterate through prompts and conditions
            for prompt in suite.prompts:
                for condition in suite.conditions:
                    for sample_idx in range(samples_per_condition):
                        try:
                            # Build prompt with system condition
                            system_prompt = condition.system_prompt
                            user_prompt = prompt.text

                            # Generate completion
                            response = client.complete(
                                prompt=user_prompt,
                                max_tokens=256,
                                temperature=0.0,
                                system=system_prompt,
                            )

                            # Run detectors
                            for detector in suite.detectors:
                                # LLMJudgeDetector requires a judge function
                                if isinstance(detector, LLMJudgeDetector):
                                    # Create judge function for this detector
                                    def judge_fn(prompt: str, max_tokens: int) -> str:
                                        judge_response = client.complete(
                                            prompt=prompt,
                                            max_tokens=max_tokens,
                                            temperature=0.0,
                                        )
                                        return judge_response.completion

                                    detector_result = detector.detect_with_judge(
                                        text=response.completion,
                                        judge_fn=judge_fn,
                                    )
                                else:
                                    # Regular detectors just need the text
                                    detector_result = detector.detect(
                                        text=response.completion
                                    )

                                # Convert detector output to BenchmarkResult
                                confidence = detector_result.metadata.get("confidence")
                                rationale = detector_result.metadata.get("rationale", "")

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
                                    tokens_used=response.total_tokens
                                    if hasattr(response, "total_tokens")
                                    else {},
                                )
                                results.append(bench_result)

                            logger.info(
                                f"    [{suite_name:15}] {prompt.id:15} "
                                f"[{condition.name:12}] sample {sample_idx + 1}/{samples_per_condition}"
                            )

                        except Exception as e:
                            logger.error(
                                f"    Error evaluating {prompt.id} "
                                f"[{condition.name}] sample {sample_idx + 1}: {e}"
                            )
                            continue

    # Save results
    results_file = output_path / f"phase2_results_{timestamp}.jsonl"
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + "\n")
    logger.info(f"\n✓ Results saved: {results_file}")

    # Generate summary
    summary_file = output_path / f"phase2_summary_{timestamp}.txt"
    _generate_summary(results, summary_file)
    logger.info(f"✓ Summary saved: {summary_file}")


def _generate_summary(results: List[BenchmarkResult], output_file: Path) -> None:
    """Generate human-readable summary."""
    from collections import defaultdict

    summary = defaultdict(lambda: {"hits": 0, "total": 0, "scores": []})

    for result in results:
        key = f"{result.model} | {result.suite} | {result.detector}"
        summary[key]["hits"] += int(result.hit)
        summary[key]["total"] += 1
        summary[key]["scores"].append(result.score)

    with open(output_file, "w") as f:
        f.write("Phase 2 Constitutional Compliance Benchmark Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Results: {len(results)} evaluations\n\n")

        f.write("Model | Suite | Detector | Hit Rate | Mean Score | Samples\n")
        f.write("-" * 80 + "\n")

        for key in sorted(summary.keys()):
            stats = summary[key]
            hit_rate = 100 * stats["hits"] / stats["total"] if stats["total"] > 0 else 0
            mean_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            f.write(
                f"{key} | {hit_rate:.1f}% | {mean_score:.3f} | {stats['total']}\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Total evaluations: {len(results)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Constitutional Compliance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with local Ollama models
  python phase2_benchmark.py --models llama3.1 mistral phi

  # Run with cloud models
  python phase2_benchmark.py --models gpt-4o-mini claude-haiku --judge-model gpt-4o

  # Custom configuration
  python phase2_benchmark.py \\
    --models llama3.1 gpt-4o-mini claude-haiku \\
    --suites task_focus truthfulness llm_judge_tone \\
    --samples 10 \\
    --output-dir results/phase2_v2
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.1", "mistral", "phi"],
        help="Models to benchmark (default: llama3.1 mistral phi)",
    )

    parser.add_argument(
        "--suites",
        nargs="+",
        default=["task_focus", "truthfulness", "llm_judge_tone"],
        help="Suites to run (default: task_focus truthfulness llm_judge_tone)",
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
        default="results/phase2",
        help="Output directory (default: results/phase2)",
    )

    args = parser.parse_args()

    run_benchmark(
        models=args.models,
        judge_model=args.judge_model,
        suites=args.suites,
        samples_per_condition=args.samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
