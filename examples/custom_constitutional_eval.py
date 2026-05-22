#!/usr/bin/env python3
"""Example script for constitutional evaluation workflows."""

from compass import (
    JudgeConfig,
    LLMJudge,
    EvaluationCache,
    RubricLibrary,
    MultiModelComparator,
)
from compass.reproducibility import reproducibility_report
from pathlib import Path


def evaluate_single_response():
    """Evaluate a single model response against a rubric."""
    print("=" * 70)
    print("Single response evaluation")
    print("=" * 70)

    # Sycophancy judge.
    config = JudgeConfig(
        rubric=RubricLibrary.sycophancy,
        judge_model="gpt-4o-mini",
    )

    # Cache results for repeat runs.
    cache = EvaluationCache(cache_dir=".cache/judges")
    judge = LLMJudge(config, cache)

    # Sample completion.
    completion = """
    That's a great idea! I think it's perfect just as-is.
    You clearly know what you're doing, and I wouldn't change a thing.
    """

    result = judge.evaluate(completion)

    print(f"\nCompletion: {completion[:50]}...")
    print(f"Score: {result.score:.2f}")
    print(f"Hit (sycophantic): {result.hit}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Rationale: {result.rationale}")
    print(f"Cache hit: {result.metadata['from_cache']}")
    print(f"Tokens used: {result.tokens_used}")


def evaluate_multiple_completions():
    """Evaluate multiple completions and compare them."""
    print("\n" + "=" * 70)
    print("Multiple completions")
    print("=" * 70)

    cache = EvaluationCache(cache_dir=".cache/judges")

    # Two rubrics.
    sycophancy_config = JudgeConfig(
        rubric=RubricLibrary.sycophancy,
        judge_model="gpt-4o-mini",
    )
    truthfulness_config = JudgeConfig(
        rubric=RubricLibrary.truthfulness,
        judge_model="gpt-4o-mini",
    )

    sycophancy_judge = LLMJudge(sycophancy_config, cache)
    truthfulness_judge = LLMJudge(truthfulness_config, cache)

    # Example completions.
    completions = {
        "claude-haiku (cautious)": """
            I'm not entirely certain, but based on my training data,
            the capital of Australia is Canberra. However, I recommend
            verifying this with a current source.
        """,
        "gpt-4o-mini (confident)": """
            The capital of Australia is Canberra. It's located in the
            Australian Capital Territory.
        """,
        "llama3.1 (hedging)": """
            Australia's capital is generally recognized as Canberra,
            though it could be worth double-checking authoritative sources.
        """,
    }

    results = {}
    for model, completion in completions.items():
        syco_result = sycophancy_judge.evaluate(completion)
        truth_result = truthfulness_judge.evaluate(completion)

        results[model] = {
            "sycophancy_score": syco_result.score,
            "truthfulness_score": truth_result.score,
        }

        print(f"\n{model}:")
        print(f"  Completion: {completion.strip()[:60]}...")
        print(f"  Sycophancy: {syco_result.score:.2f}")
        print(f"  Truthfulness: {truth_result.score:.2f}")


def compare_models_on_suite():
    """Compare multiple models on a tic suite using pairwise preference."""
    print("\n" + "=" * 70)
    print("Suite inspection")
    print("=" * 70)

    from helm_eval_lex_quirks.metrics.builtin_tic_suites import (
        TASK_FOCUS_SUITE,
        STYLE_SUITE_REGISTRY,
    )
    from helm_eval_lex_quirks.metrics.tic_suite import LLMJudgeDetector

    print("\nAvailable evaluation suites:")
    for key in sorted(STYLE_SUITE_REGISTRY.keys()):
        suite = STYLE_SUITE_REGISTRY[key]
        detector_count = len(suite.detectors)
        prompt_count = len(suite.prompts)
        print(f"  {key}: {prompt_count} prompts, {detector_count} detectors")

    suite = TASK_FOCUS_SUITE
    print("\nTask Focus suite:")
    print(f"  Prompts: {[p.id for p in suite.prompts]}")
    print(f"  Conditions: {[c.name for c in suite.conditions]}")
    print(f"  Detectors: {[d.name for d in suite.detectors]}")
    print(f"  Detector types: {[type(d).__name__ for d in suite.detectors]}")

    for detector in suite.detectors:
        if isinstance(detector, LLMJudgeDetector):
            print(f"\n  Detector '{detector.name}' rubric:")
            print(f"    {detector.rubric[:100]}...")


def reproducibility_example():
    """Demonstrate reproducibility and caching."""
    print("\n" + "=" * 70)
    print("Reproducibility and caching")
    print("=" * 70)

    cache = EvaluationCache(cache_dir=".cache/judges")
    config = JudgeConfig(
        rubric=RubricLibrary.sycophancy,
        judge_model="gpt-4o-mini",
    )

    judge = LLMJudge(config, cache)

    # Same text twice; second call should hit cache.
    text = "Your code looks perfect! I wouldn't change anything."

    print("\nFirst evaluation:")
    result1 = judge.evaluate(text)
    print(f"  Score: {result1.score:.2f}")
    print(f"  From cache: {result1.metadata['from_cache']}")
    print(f"  Cost: ${result1.cost_usd:.4f}")

    print("\nSecond evaluation (same text):")
    result2 = judge.evaluate(text)
    print(f"  Score: {result2.score:.2f}")
    print(f"  From cache: {result2.metadata['from_cache']}")
    print(f"  Cost: ${result2.cost_usd:.4f}")

    print("\nReproducibility metadata:")
    metadata = judge.metadata()
    print(f"  Total evaluations: {metadata.total_evaluations}")
    print(f"  Cache hits: {metadata.cache_hits}")
    print(f"  Total cost: ${metadata.total_cost_usd:.2f}")
    print(f"  Judge model: {metadata.judge_model}")

    report_path = Path("reproducibility_report.txt")
    report = reproducibility_report(judge.metadata())
    print(f"\nFull report saved to {report_path}:")
    print(report[:200] + "...")


def run_phase2_example():
    """Show how to use phase2_benchmark with custom configurations."""
    print("\n" + "=" * 70)
    print("Phase 2 benchmark")
    print("=" * 70)

    print("""
To run the Constitutional Compliance Benchmark:

1. With local models (Ollama):
   python phase2_benchmark.py \\
     --models llama3.1 mistral phi \\
     --samples 5 \\
     --output-dir results/local_benchmark

2. With cloud models:
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   python phase2_benchmark.py \\
     --models gpt-4o-mini claude-haiku \\
     --judge-model gpt-4o \\
     --samples 10 \\
     --output-dir results/cloud_benchmark

3. Resume from checkpoint:
   python phase2_benchmark.py \\
     --models llama3.1 \\
     --resume \\
     --output-dir results/local_benchmark

Output files:
  - benchmark.jsonl: JSONL checkpoint with all evaluations
  - benchmark_summary_TIMESTAMP.txt: Human-readable statistics

Each evaluation includes:
  - model, suite, detector: Identification
  - score, hit: Detection results
  - confidence, rationale: Judge reasoning
  - input_tokens, output_tokens: Cost tracking
""")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                  Compass Constitutional Evaluation Examples              ║
║                                                                          ║
║  This script demonstrates using compass for evaluating LLM outputs      ║
║  against constitutional principles.                                     ║
╚════════════════════════════════════════════════════════════════════════╝
""")

    try:
        # Requires API keys or local models.
        # evaluate_single_response()
        # evaluate_multiple_completions()
        compare_models_on_suite()
        run_phase2_example()

        print("\n" + "=" * 70)
        print("For more examples, see:")
        print("  - PHASE2_BENCHMARK.md: Full benchmark documentation")
        print("  - tests/test_*.py: Unit tests showing API usage")
        print("  - compass/__init__.py: Public API reference")
        print("=" * 70)

    except ImportError as e:
        print(f"\nNote: Some examples require API keys or local models: {e}")
        print("See comments in the script for how to set up.")
