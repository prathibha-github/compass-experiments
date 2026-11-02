#!/usr/bin/env python3
"""Repo-local example for inspecting suites and benchmark commands."""
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compass_experiments.metrics.builtin_tic_suites import STYLE_SUITE_REGISTRY


def print_suite_inventory() -> None:
    print("Registered suites")
    print("=" * 70)
    for key in sorted(STYLE_SUITE_REGISTRY):
        suite = STYLE_SUITE_REGISTRY[key]
        detector_names = ", ".join(detector.name for detector in suite.detectors)
        print(
            f"{key:22} prompts={len(suite.prompts):2d} "
            f"conditions={len(suite.conditions):2d} detectors={detector_names}"
        )


def print_commands() -> None:
    print("\nUseful commands")
    print("=" * 70)
    print("Style suites:")
    print(
        "  python -m compass_experiments.main "
        "--style-only --style-suite task_focus truthfulness "
        "--style-samples 3 --judge-model gpt-4o-mini --models gpt-4o-mini"
    )
    print("\nPairwise from an existing results file:")
    print(
        "  python -m compass_experiments.main "
        "--pairwise-from-results results/results_YYYYMMDD_HHMMSS.json "
        "--pairwise-method bradley_terry --pairwise-segment-by condition"
    )
    print("\nPhase 2 benchmark:")
    print(
        "  python phase2_benchmark.py "
        "--models llama3.1 mistral phi --judge-model gpt-4o-mini --samples 3"
    )


def print_notes() -> None:
    print("\nNotes")
    print("=" * 70)
    print("- This repo orchestrates runs and reporting.")
    print("- Shared clients, rubrics, and detector primitives live in compass.")
    print("- If you want a reusable new suite, define it in compass first.")


if __name__ == "__main__":
    print_suite_inventory()
    print_commands()
    print_notes()
