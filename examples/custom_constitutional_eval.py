#!/usr/bin/env python3
"""Example: custom constitutional evaluation helpers."""
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_compass():
    """
    Import Compass modules lazily so this example script still runs in repos
    where Compass is not installed.
    """
    try:
        from compass import (
            JudgeConfig,
            LLMJudge,
            EvaluationCache,
            RubricLibrary,
        )
        from compass.reproducibility import reproducibility_report
        return (
            JudgeConfig,
            LLMJudge,
            EvaluationCache,
            RubricLibrary,
            reproducibility_report,
        )
    except ImportError as exc:
        raise ImportError(
            "Compass is not installed. Install the Compass package to run "
            "examples 1, 2, and 4."
        ) from exc


def evaluate_single_response():
    """Evaluate a single model response against a rubric."""
    JudgeConfig, LLMJudge, EvaluationCache, RubricLibrary, _ = _load_compass()
    print("=" * 70)
    print("Example 1: Single Response Evaluation")
    print("=" * 70)

    # Create a judge for sycophancy
    config = JudgeConfig(
        rubric=RubricLibrary.sycophancy,
        judge_model="gpt-4o-mini",
    )

    # Set up caching for reproducibility
    cache = EvaluationCache(cache_dir=".cache/judges")
    judge = LLMJudge(config, cache)

    # Evaluate a completion
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
    JudgeConfig, LLMJudge, EvaluationCache, RubricLibrary, _ = _load_compass()
    print("\n" + "=" * 70)
    print("Example 2: Multiple Completions with Comparison")
    print("=" * 70)

    cache = EvaluationCache(cache_dir=".cache/judges")

    # Create judges for different rubrics
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

    # Test completions from different models
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
    print("Example 3: Multi-Model Comparison")
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

    # Show the task_focus suite structure
    suite = TASK_FOCUS_SUITE
    print(f"\nTask Focus Suite structure:")
    print(f"  Prompts: {[p.id for p in suite.prompts]}")
    print(f"  Conditions: {[c.name for c in suite.conditions]}")
    print(f"  Detectors: {[d.name for d in suite.detectors]}")
    print(f"  Detector types: {[type(d).__name__ for d in suite.detectors]}")

    # Show that detectors are LLM-based for this suite
    for detector in suite.detectors:
        if isinstance(detector, LLMJudgeDetector):
            print(f"\n  Detector '{detector.name}' rubric:")
            print(f"    {detector.rubric[:100]}...")


def reproducibility_example():
    """Demonstrate reproducibility and caching."""
    (
        JudgeConfig,
        LLMJudge,
        EvaluationCache,
        RubricLibrary,
        reproducibility_report,
    ) = _load_compass()
    print("\n" + "=" * 70)
    print("Example 4: Reproducibility and Caching")
    print("=" * 70)

    cache = EvaluationCache(cache_dir=".cache/judges")
    config = JudgeConfig(
        rubric=RubricLibrary.sycophancy,
        judge_model="gpt-4o-mini",
    )

    judge = LLMJudge(config, cache)

    # Evaluate the same text twice - second should be from cache
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

    # Get reproducibility report
    print("\nReproducibility metadata:")
    metadata = judge.metadata()
    print(f"  Total evaluations: {metadata.total_evaluations}")
    print(f"  Cache hits: {metadata.cache_hits}")
    print(f"  Total cost: ${metadata.total_cost_usd:.2f}")
    print(f"  Judge model: {metadata.judge_model}")

    # Generate reproducibility report
    report_path = Path("reproducibility_report.txt")
    report = reproducibility_report(judge.metadata())
    print(f"\nFull report saved to {report_path}:")
    print(report[:200] + "...")


def run_phase2_example():
    """Show how to use phase2_benchmark with custom configurations."""
    print("\n" + "=" * 70)
    print("Example 5: Running Phase 2 Benchmark")
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

    # These run without Compass.
    compare_models_on_suite()
    run_phase2_example()

    print("\n" + "=" * 70)
    print("For more examples, see:")
    print("  - PHASE2_BENCHMARK.md: Full benchmark documentation")
    print("  - tests/test_*.py: Unit tests showing API usage")
    print(
        "  - examples/custom_constitutional_eval.py: "
        "Compass-backed examples 1, 2, and 4 require Compass installed"
    )
    print("=" * 70)
