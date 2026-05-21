import tempfile
import unittest

from helm_eval_lex_quirks.config import EvalConfig
from helm_eval_lex_quirks.metrics.builtin_tic_suites import PUNCTUATION_TIC_SUITE
from helm_eval_lex_quirks.metrics.persona_transfer import analyze_persona_transfer
from helm_eval_lex_quirks.models.openai_client import ModelResponse
from helm_eval_lex_quirks.report import format_table
from helm_eval_lex_quirks.runner import EvalRunner


class PersonaTransferTests(unittest.TestCase):
    def test_analyze_persona_transfer_reports_positive_deltas(self):
        suite_results = {
            "punctuation_tics": {
                "analysis": {
                    "neutral": {
                        "n_outputs": 2,
                        "detectors": {
                            "emoji": {"pct_hit": 0.0, "mean_count": 0.0},
                        },
                    },
                    "enthusiastic": {
                        "n_outputs": 2,
                        "detectors": {
                            "emoji": {"pct_hit": 50.0, "mean_count": 1.0},
                        },
                    },
                }
            }
        }

        result = analyze_persona_transfer(
            suite_results,
            {"punctuation_tics": PUNCTUATION_TIC_SUITE},
        )

        row = result["punctuation_tics"]["persona_conditions"]["enthusiastic"]["top_detectors"][0]
        self.assertEqual(row["detector"], "emoji")
        self.assertEqual(row["pct_hit_delta"], 50.0)
        self.assertEqual(row["mean_count_delta"], 1.0)
        self.assertGreater(row["transfer_score"], 0)

    def test_runner_populates_persona_transfer_results(self):
        class DummyClient:
            total_tokens = {"input": 0, "output": 0}
            total_cost_usd = 0.0

            def complete(self, prompt, max_tokens=1, logprobs=False, system=None):
                if system and "enthusiastic" in system:
                    completion = "Useful update 🚀🚀🚀"
                else:
                    completion = "Useful update."
                return ModelResponse(completion=completion, logprobs=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EvalConfig(
                api_key="dummy",
                output_dir=tmpdir,
                scenarios=[],
                style_only=True,
                style_suites=["punctuation"],
                style_samples=1,
                run_persona_transfer=True,
            )

            result = EvalRunner(cfg)._run_model(DummyClient(), "dummy-model")

        self.assertIn("punctuation_tics", result.persona_transfer_results)
        self.assertIn(
            "enthusiastic",
            result.persona_transfer_results["punctuation_tics"]["persona_conditions"],
        )
        self.assertIn("PERSONA TRANSFER", format_table({"dummy-model": result}))


if __name__ == "__main__":
    unittest.main()
