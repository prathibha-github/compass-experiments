"""Smoke tests for public CLI entry points."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import phase2_benchmark
from helm_eval_lex_quirks import main as helm_main


class MainEntrypointTests(unittest.TestCase):
    def test_helm_main_dry_run_prints_estimate(self):
        fake_runner = MagicMock()
        fake_runner.estimate_cost.return_value = {"gpt-4o-mini": 0.0123}

        with patch("helm_eval_lex_quirks.main.EvalRunner", return_value=fake_runner):
            out = io.StringIO()
            with redirect_stdout(out):
                helm_main.main(["--api-key", "fake-key", "--dry-run"])

        fake_runner.run.assert_not_called()
        text = out.getvalue()
        self.assertIn("Estimated cost", text)
        self.assertIn("Dry run", text)

    def test_helm_main_pairwise_from_results_routes_to_offline_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/results.json"
            with open(path, "w") as f:
                json.dump({}, f)

            with patch("helm_eval_lex_quirks.main._run_pairwise_from_results") as patched:
                helm_main.main(["--pairwise-from-results", path])

        patched.assert_called_once()
        args = patched.call_args.args[0]
        self.assertEqual(args.pairwise_from_results, path)


class Phase2EntrypointTests(unittest.TestCase):
    def test_phase2_main_routes_args_to_run_benchmark(self):
        with patch("phase2_benchmark.run_benchmark") as patched:
            phase2_benchmark.main(
                [
                    "--models", "llama3.1", "mistral",
                    "--judge-model", "gpt-4o-mini",
                    "--suites", "task_focus", "truthfulness",
                    "--samples", "3",
                    "--output-dir", "tmp-results",
                    "--resume",
                ]
            )

        patched.assert_called_once_with(
            models=["llama3.1", "mistral"],
            judge_model="gpt-4o-mini",
            suites=["task_focus", "truthfulness"],
            samples_per_condition=3,
            output_dir="tmp-results",
            resume=True,
            temperature=0.0,
            auto_resume=True,
            dry_run=False,
        )

    def test_phase2_main_can_disable_auto_resume(self):
        with patch("phase2_benchmark.run_benchmark") as patched:
            phase2_benchmark.main(
                [
                    "--models", "llama3.1",
                    "--no-auto-resume",
                ]
            )

        patched.assert_called_once()
        self.assertFalse(patched.call_args.kwargs["auto_resume"])


if __name__ == "__main__":
    unittest.main()
