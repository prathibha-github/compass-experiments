import json
import os
import tempfile
import unittest

from helm_eval_lex_quirks.report import save
from helm_eval_lex_quirks.runner import EvalResults, ScenarioResults


class ReportOutputTests(unittest.TestCase):
    def test_save_writes_summary_and_text_report(self):
        results = {
            "model-a": EvalResults(
                model="model-a",
                scenario_results={
                    "boolq": ScenarioResults(
                        name="boolq",
                        accuracy=0.75,
                        ece=0.12,
                        ece_coverage=1.0,
                        robustness_score=0.9,
                        n_completed=4,
                        n_requested=4,
                        checkpoint_path="results/checkpoints/model-a__boolq.jsonl",
                    )
                },
                tic_results={
                    "punctuation_tics": {
                        "checkpoint_path": "results/checkpoints/model-a__punctuation.jsonl",
                        "analysis": {
                            "neutral": {
                                "n_outputs": 2,
                                "detectors": {
                                    "emoji": {
                                        "pct_hit": 50.0,
                                        "pct_hit_ci_low": 10.0,
                                        "pct_hit_ci_high": 90.0,
                                        "mean_count": 0.5,
                                    }
                                },
                            }
                        },
                    }
                },
                pairwise_preference_results={
                    "punctuation_tics": {
                        "overall": {
                            "ranking": [
                                {
                                    "model": "model-a",
                                    "preference_score": 1.0,
                                    "matches": 1,
                                    "wins": 1,
                                    "losses": 0,
                                    "ties": 0,
                                }
                            ]
                        },
                        "task_types": {},
                    }
                },
                token_usage={"input": 12, "output": 3},
                total_cost_usd=0.001234,
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, txt_path = save(results, tmpdir)

            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(txt_path))

            with open(json_path) as f:
                summary = json.load(f)
            with open(txt_path) as f:
                report_text = f.read()

        self.assertEqual(summary["model-a"]["scenarios"]["boolq"]["accuracy"], 0.75)
        tic_checkpoint = summary["model-a"]["tic_suites"]["punctuation_tics"]["checkpoint_path"]
        self.assertEqual(tic_checkpoint, "results/checkpoints/model-a__punctuation.jsonl")
        self.assertIn("pairwise_preference", summary)
        self.assertIn("HELM EVAL RESULTS", report_text)
        self.assertIn("TIC SUITE DETECTION", report_text)
        self.assertIn("PAIRWISE PREFERENCE", report_text)


if __name__ == "__main__":
    unittest.main()
