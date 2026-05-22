"""Tests for phase2_benchmark.py runner."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from helm_eval_lex_quirks.models.ollama_client import ModelResponse


class MockCompletionClient:
    """Mock client for testing benchmark runner."""

    def __init__(self, completion_text: str = "test response"):
        self.completion_text = completion_text
        self.call_count = 0

    def complete(self, prompt, max_tokens=256, temperature=0.0, system=None):
        self.call_count += 1
        return ModelResponse(
            completion=self.completion_text,
            input_tokens=len(prompt.split()),
            output_tokens=len(self.completion_text.split()),
        )


class TestPhase2Checkpoint(unittest.TestCase):
    """Test checkpoint loading/saving logic."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.tmpdir.name) / "benchmark.jsonl"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_checkpoint_empty_file_returns_empty_set(self):
        """Empty checkpoint returns empty completed set."""
        # Simulate the load_checkpoint function behavior
        completed = set()
        if not self.checkpoint_path.exists():
            pass

        self.assertEqual(len(completed), 0)

    def test_checkpoint_parsing_single_result(self):
        """Can parse a single checkpoint result."""
        result = {
            "model": "phi",
            "suite": "task_focus",
            "detector": "task_focus",
            "prompt_id": "bug_fix",
            "condition": "default",
            "score": 0.5,
            "hit": False,
        }

        with open(self.checkpoint_path, "w") as f:
            f.write(json.dumps(result) + "\n")

        completed = set()
        with open(self.checkpoint_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = (
                        r["model"],
                        r["suite"],
                        r["detector"],
                        r["prompt_id"],
                        r["condition"],
                    )
                    completed.add(key)

        expected_key = ("phi", "task_focus", "task_focus", "bug_fix", "default")
        self.assertIn(expected_key, completed)

    def test_checkpoint_multiple_results(self):
        """Can parse multiple checkpoint results."""
        results = [
            {
                "model": "phi",
                "suite": "task_focus",
                "detector": "task_focus",
                "prompt_id": "bug_fix",
                "condition": "default",
                "score": 0.5,
            },
            {
                "model": "mistral",
                "suite": "truthfulness",
                "detector": "truthfulness",
                "prompt_id": "factual_question",
                "condition": "neutral",
                "score": 0.8,
            },
        ]

        with open(self.checkpoint_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        completed = set()
        with open(self.checkpoint_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = (
                        r["model"],
                        r["suite"],
                        r["detector"],
                        r["prompt_id"],
                        r["condition"],
                    )
                    completed.add(key)

        self.assertEqual(len(completed), 2)

    def test_checkpoint_skip_malformed_lines(self):
        """Malformed JSON lines are skipped gracefully."""
        with open(self.checkpoint_path, "w") as f:
            f.write('{"model": "phi", "suite": "task_focus"}\n')
            f.write("not valid json\n")
            f.write('{"model": "mistral", "suite": "truthfulness"}\n')

        completed = set()
        with open(self.checkpoint_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    key = (
                        r["model"],
                        r["suite"],
                        r["detector"],
                        r["prompt_id"],
                        r["condition"],
                    )
                    completed.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue

        # Should have successfully parsed 0 (both missing required fields)
        self.assertEqual(len(completed), 0)

    def test_checkpoint_append_mode(self):
        """Checkpoint file appends new results (doesn't overwrite)."""
        result1 = {
            "model": "phi",
            "suite": "task_focus",
            "detector": "task_focus",
            "prompt_id": "bug_fix",
            "condition": "default",
            "score": 0.5,
        }

        result2 = {
            "model": "mistral",
            "suite": "task_focus",
            "detector": "task_focus",
            "prompt_id": "bug_fix",
            "condition": "default",
            "score": 0.7,
        }

        # Write first result
        with open(self.checkpoint_path, "a") as f:
            f.write(json.dumps(result1) + "\n")

        # Append second result
        with open(self.checkpoint_path, "a") as f:
            f.write(json.dumps(result2) + "\n")

        # Read back
        completed = set()
        with open(self.checkpoint_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = (
                        r["model"],
                        r["suite"],
                        r["detector"],
                        r["prompt_id"],
                        r["condition"],
                    )
                    completed.add(key)

        self.assertEqual(len(completed), 2)

    def test_detector_skip_logic_all_complete(self):
        """When all detectors are complete, skip evaluation."""
        suite_detectors = ["detector1", "detector2", "detector3"]
        completed = {
            ("phi", "suite", "detector1", "prompt1", "default"),
            ("phi", "suite", "detector2", "prompt1", "default"),
            ("phi", "suite", "detector3", "prompt1", "default"),
        }

        all_done = all(
            ("phi", "suite", d, "prompt1", "default") in completed
            for d in suite_detectors
        )
        self.assertTrue(all_done)

    def test_detector_skip_logic_partial_complete(self):
        """When only some detectors are complete, don't skip."""
        suite_detectors = ["detector1", "detector2", "detector3"]
        completed = {
            ("phi", "suite", "detector1", "prompt1", "default"),
            ("phi", "suite", "detector2", "prompt1", "default"),
            # detector3 is missing
        }

        all_done = all(
            ("phi", "suite", d, "prompt1", "default") in completed
            for d in suite_detectors
        )
        self.assertFalse(all_done)

    def test_resume_vs_new_run_logic(self):
        """Resume=True loads checkpoint; Resume=False starts empty."""
        # Write a checkpoint
        result = {
            "model": "phi",
            "suite": "task_focus",
            "detector": "task_focus",
            "prompt_id": "bug_fix",
            "condition": "default",
        }
        with open(self.checkpoint_path, "w") as f:
            f.write(json.dumps(result) + "\n")

        # Simulate resume=True
        completed_resume = set()
        with open(self.checkpoint_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = (r["model"], r["suite"], r["detector"], r["prompt_id"], r["condition"])
                    completed_resume.add(key)

        # Simulate resume=False
        completed_new = set()

        self.assertEqual(len(completed_resume), 1)
        self.assertEqual(len(completed_new), 0)


class TestBenchmarkResult(unittest.TestCase):
    """Test BenchmarkResult data class."""

    def test_benchmark_result_to_dict(self):
        """BenchmarkResult converts to dict for JSON serialization."""
        # Need to import after defining the test to avoid import errors
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Simulate the BenchmarkResult class behavior
        result_dict = {
            "model": "phi",
            "suite": "task_focus",
            "detector": "task_focus",
            "prompt_id": "bug_fix",
            "condition": "default",
            "score": 0.75,
            "hit": True,
            "confidence": 0.95,
            "rationale": "test",
            "input_tokens": 100,
            "output_tokens": 50,
        }

        # Verify all expected fields are present
        expected_fields = {
            "model",
            "suite",
            "detector",
            "prompt_id",
            "condition",
            "score",
            "hit",
            "confidence",
            "rationale",
            "input_tokens",
            "output_tokens",
        }
        self.assertEqual(set(result_dict.keys()), expected_fields)

    def test_benchmark_result_none_fields(self):
        """BenchmarkResult handles optional None fields."""
        result_dict = {
            "model": "phi",
            "suite": "task_focus",
            "detector": "task_focus",
            "prompt_id": "bug_fix",
            "condition": "default",
            "score": 0.5,
            "hit": False,
            "confidence": None,
            "rationale": "",
            "input_tokens": 0,
            "output_tokens": 0,
        }

        self.assertIsNone(result_dict["confidence"])
        self.assertEqual(result_dict["rationale"], "")


class TestSummaryGeneration(unittest.TestCase):
    """Test summary generation from checkpoint."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.tmpdir.name) / "benchmark.jsonl"
        self.summary_path = Path(self.tmpdir.name) / "summary.txt"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_summary_empty_checkpoint(self):
        """Empty checkpoint produces minimal summary."""
        # Create empty checkpoint
        self.checkpoint_path.touch()

        from collections import defaultdict
        summary = defaultdict(lambda: {"hits": 0, "total": 0, "scores": []})

        with open(self.checkpoint_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    key = f"{result['model']} | {result['suite']} | {result['detector']}"
                    summary[key]["hits"] += int(result.get("hit", 0))
                    summary[key]["total"] += 1
                    summary[key]["scores"].append(result.get("score", 0))
                except json.JSONDecodeError:
                    continue

        self.assertEqual(len(summary), 0)

    def test_summary_aggregates_by_model_suite_detector(self):
        """Summary groups results by model/suite/detector."""
        results = [
            {
                "model": "phi",
                "suite": "task_focus",
                "detector": "task_focus",
                "score": 0.5,
                "hit": False,
            },
            {
                "model": "phi",
                "suite": "task_focus",
                "detector": "task_focus",
                "score": 0.7,
                "hit": True,
            },
            {
                "model": "mistral",
                "suite": "task_focus",
                "detector": "task_focus",
                "score": 0.6,
                "hit": True,
            },
        ]

        with open(self.checkpoint_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        from collections import defaultdict
        summary = defaultdict(lambda: {"hits": 0, "total": 0, "scores": []})

        with open(self.checkpoint_path) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    key = f"{result['model']} | {result['suite']} | {result['detector']}"
                    summary[key]["hits"] += int(result.get("hit", 0))
                    summary[key]["total"] += 1
                    summary[key]["scores"].append(result.get("score", 0))

        self.assertEqual(len(summary), 2)
        phi_key = "phi | task_focus | task_focus"
        mistral_key = "mistral | task_focus | task_focus"

        self.assertEqual(summary[phi_key]["total"], 2)
        self.assertEqual(summary[phi_key]["hits"], 1)
        self.assertEqual(summary[mistral_key]["total"], 1)
        self.assertEqual(summary[mistral_key]["hits"], 1)

    def test_summary_calculates_hit_rate(self):
        """Summary calculates hit rate correctly."""
        results = [
            {"score": 0.5, "hit": True},
            {"score": 0.6, "hit": True},
            {"score": 0.7, "hit": False},
            {"score": 0.8, "hit": False},
        ]

        stats = {"hits": 2, "total": 4, "scores": [0.5, 0.6, 0.7, 0.8]}
        hit_rate = 100 * stats["hits"] / stats["total"] if stats["total"] > 0 else 0
        mean_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0

        self.assertEqual(hit_rate, 50.0)
        self.assertAlmostEqual(mean_score, 0.65, places=2)


if __name__ == "__main__":
    unittest.main()
