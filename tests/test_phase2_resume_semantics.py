"""Regression tests for phase2 benchmark checkpoint semantics."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase2_benchmark
from helm_eval_lex_quirks.metrics.tic_suite import (
    DetectorResult,
    StyleCondition,
    StylePrompt,
    TicSuite,
)
from helm_eval_lex_quirks.models.ollama_client import ModelResponse


class _StubDetector:
    name = "stub_detector"

    def detect(self, text: str) -> DetectorResult:
        return DetectorResult(
            name=self.name,
            count=1,
            score=1.0,
            hit=True,
            metadata={},
        )


class _CountingClient:
    def __init__(self, completion_text: str = "ok"):
        self.completion_text = completion_text
        self.call_count = 0

    def complete(self, prompt, max_tokens=256, temperature=0.0, system=None):
        self.call_count += 1
        return ModelResponse(
            completion=self.completion_text,
            input_tokens=1,
            output_tokens=1,
        )


class TestPhase2ResumeSemantics(unittest.TestCase):
    def _stub_suite(self) -> TicSuite:
        return TicSuite(
            name="stub_suite",
            prompts=[StylePrompt(id="prompt_1", text="hello")],
            conditions=[StyleCondition(name="neutral", system_prompt="Be direct.")],
            detectors=[_StubDetector()],
            baseline_condition="neutral",
        )

    def test_load_checkpoint_parses_sample_idx_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "benchmark.jsonl"
            with open(checkpoint_path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "model": "m",
                            "suite": "s",
                            "detector": "d",
                            "prompt_id": "p",
                            "condition": "c",
                            "sample_idx": 2,
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "model": "m",
                            "suite": "s",
                            "detector": "d",
                            "prompt_id": "p",
                            "condition": "c",
                        }
                    )
                    + "\n"
                )

            completed = phase2_benchmark.load_checkpoint(checkpoint_path)
            self.assertIn(("m", "s", "d", "p", "c", 2), completed)
            self.assertIn(("m", "s", "d", "p", "c", 0), completed)

    def test_resume_uses_sample_level_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "benchmark.jsonl"
            with open(checkpoint_path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "model": "model-a",
                            "suite": "stub_suite",
                            "detector": "stub_detector",
                            "prompt_id": "prompt_1",
                            "condition": "neutral",
                            "sample_idx": 0,
                            "score": 1.0,
                            "hit": True,
                        }
                    )
                    + "\n"
                )

            clients = {}

            def fake_create_client(model: str, **kwargs):
                clients.setdefault(model, _CountingClient())
                return clients[model], "stub"

            suite = self._stub_suite()
            with patch.object(
                phase2_benchmark, "STYLE_SUITE_REGISTRY", {"stub_suite": suite}
            ), patch.object(
                phase2_benchmark, "create_client", side_effect=fake_create_client
            ), patch.object(
                phase2_benchmark, "_generate_summary", return_value=None
            ):
                phase2_benchmark.run_benchmark(
                    models=["model-a"],
                    judge_model="judge-a",
                    suites=["stub_suite"],
                    samples_per_condition=2,
                    output_dir=tmpdir,
                    resume=True,
                )

            self.assertEqual(clients["model-a"].call_count, 1)
            with open(checkpoint_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            sample_idxs = sorted(row["sample_idx"] for row in rows)
            self.assertEqual(sample_idxs, [0, 1])

    def test_new_run_truncates_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "benchmark.jsonl"
            with open(checkpoint_path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "model": "legacy-model",
                            "suite": "legacy-suite",
                            "detector": "legacy-detector",
                            "prompt_id": "legacy",
                            "condition": "neutral",
                            "sample_idx": 99,
                            "score": 0.0,
                            "hit": False,
                        }
                    )
                    + "\n"
                )

            clients = {}

            def fake_create_client(model: str, **kwargs):
                clients.setdefault(model, _CountingClient())
                return clients[model], "stub"

            suite = self._stub_suite()
            with patch.object(
                phase2_benchmark, "STYLE_SUITE_REGISTRY", {"stub_suite": suite}
            ), patch.object(
                phase2_benchmark, "create_client", side_effect=fake_create_client
            ), patch.object(
                phase2_benchmark, "_generate_summary", return_value=None
            ):
                phase2_benchmark.run_benchmark(
                    models=["model-a"],
                    judge_model="judge-a",
                    suites=["stub_suite"],
                    samples_per_condition=1,
                    output_dir=tmpdir,
                    resume=False,
                )

            with open(checkpoint_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], "model-a")
            self.assertEqual(rows[0]["sample_idx"], 0)


if __name__ == "__main__":
    unittest.main()
