"""Regression tests for phase2 benchmark, now backed by the two-phase suite runner."""
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


def _stub_suite() -> TicSuite:
    return TicSuite(
        name="stub_suite",
        prompts=[StylePrompt(id="prompt_1", text="hello")],
        conditions=[StyleCondition(name="neutral", system_prompt="Be direct.")],
        detectors=[_StubDetector()],
        baseline_condition="neutral",
    )


class LoadCheckpointTests(unittest.TestCase):
    def test_load_checkpoint_parses_sample_idx_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "benchmark.jsonl"
            with open(checkpoint_path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "model": "m", "suite": "s", "detector": "d",
                            "prompt_id": "p", "condition": "c", "sample_idx": 2,
                        }
                    ) + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "model": "m", "suite": "s", "detector": "d",
                            "prompt_id": "p", "condition": "c",
                        }
                    ) + "\n"
                )

            completed = phase2_benchmark.load_checkpoint(checkpoint_path)
            self.assertIn(("m", "s", "d", "p", "c", 2), completed)
            self.assertIn(("m", "s", "d", "p", "c", 0), completed)


class TwoPhaseRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = self.tmp.name
        self.clients = {}

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_create_client(self, model: str, **kwargs):
        self.clients.setdefault(model, _CountingClient())
        return self.clients[model], "stub"

    def _run(self, **overrides):
        kwargs = dict(
            models=["model-a"],
            judge_model="judge-a",
            suites=["stub_suite"],
            samples_per_condition=2,
            output_dir=self.out,
        )
        kwargs.update(overrides)
        with patch.object(
            phase2_benchmark, "STYLE_SUITE_REGISTRY", {"stub_suite": _stub_suite()}
        ), patch.object(
            phase2_benchmark, "create_client", side_effect=self._fake_create_client
        ):
            phase2_benchmark.run_benchmark(**kwargs)

    def test_two_phase_persists_generations_and_scores(self):
        self._run()
        suite_dir = Path(self.out) / "stub_suite"
        gens = [
            json.loads(l)
            for l in open(suite_dir / "suite_generations.jsonl")
            if l.strip()
        ]
        # 1 prompt x 1 condition x 2 samples
        self.assertEqual(len(gens), 2)
        self.assertTrue(all("completion" in g for g in gens))
        self.assertEqual(self.clients["model-a"].call_count, 2)

        # Heuristic-only suite -> evaluations tagged "heuristic", one row per sample.
        eval_rows = [
            json.loads(l)
            for l in open(suite_dir / "suite_evaluations_heuristic.jsonl")
            if l.strip()
        ]
        self.assertEqual(len(eval_rows), 2)
        self.assertTrue(all(r["hit"] for r in eval_rows))

        # A summary file is written at the output root.
        self.assertTrue(list(Path(self.out).glob("benchmark_summary_*.txt")))

    def test_resume_does_not_regenerate(self):
        self._run()
        before = self.clients["model-a"].call_count
        self._run(resume=True)
        self.assertEqual(self.clients["model-a"].call_count, before)


if __name__ == "__main__":
    unittest.main()
