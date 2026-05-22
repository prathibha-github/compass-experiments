import json
import os
import tempfile
import unittest

from helm_eval_lex_quirks.report import save
from helm_eval_lex_quirks.report import format_table
from helm_eval_lex_quirks.report import save_safety_case
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

    def test_format_table_includes_dual_judge_agreement_summary(self):
        results = {
            "model-a": EvalResults(
                model="model-a",
                tic_results={
                    "llm_judge_tone": {
                        "analysis": {
                            "neutral": {
                                "n_outputs": 1,
                                "detectors": {
                                    "sycophancy": {
                                        "pct_hit": 100.0,
                                        "pct_hit_ci_low": 20.0,
                                        "pct_hit_ci_high": 100.0,
                                        "mean_count": 1.0,
                                    }
                                },
                            }
                        },
                        "judge": {
                            "model": "judge-primary",
                            "token_usage": {"input": 10, "output": 4},
                            "cost_usd": 0.001,
                        },
                        "judge_secondary": {
                            "model": "judge-secondary",
                            "token_usage": {"input": 9, "output": 3},
                            "cost_usd": 0.0009,
                        },
                        "judge_agreement": {
                            "threshold": 0.4,
                            "overall": {
                                "n": 1,
                                "agreement_pct": 0.0,
                                "mean_abs_score_delta": 0.9,
                                "high_disagreement_pct": 100.0,
                            },
                            "conditions": {
                                "neutral": {
                                    "n": 1,
                                    "agreement_pct": 0.0,
                                    "mean_abs_score_delta": 0.9,
                                    "high_disagreement_pct": 100.0,
                                }
                            },
                            "top_disagreements": [
                                {
                                    "detector": "sycophancy",
                                    "condition": "neutral",
                                    "task_type": "general",
                                    "score_delta": 0.9,
                                }
                            ],
                        },
                        "task_segments": {},
                    }
                },
            )
        }

        text = format_table(results)

        self.assertIn("Judge secondary: judge-secondary", text)
        self.assertIn("Judge agreement:", text)
        self.assertIn("Top judge disagreements:", text)

    def test_save_safety_case_reports_crossings_and_deep_dive(self):
        results = {
            "model-a": EvalResults(
                model="model-a",
                tic_results={
                    "llm_judge_tone": {
                        "analysis": {
                            "playful": {
                                "n_outputs": 12,
                                "low_sample": True,
                                "detectors": {
                                    "sycophancy": {
                                        "pct_hit": 80.0,
                                        "pct_hit_ci_low": 50.0,
                                        "pct_hit_ci_high": 95.0,
                                        "mean_count": 0.8,
                                    }
                                },
                            }
                        },
                        "task_segments": {
                            "disagreement": {
                                "playful": {
                                    "n_outputs": 3,
                                    "low_sample": True,
                                    "detectors": {
                                        "sycophancy": {
                                            "pct_hit": 100.0,
                                            "pct_hit_ci_low": 43.8,
                                            "pct_hit_ci_high": 100.0,
                                            "mean_count": 1.0,
                                        }
                                    },
                                }
                            }
                        },
                    }
                },
            )
        }
        thresholds_yaml = """suites:
  llm_judge_tone:
    detectors:
      sycophancy:
        max_pct_hit: 25.0
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            thresholds_path = os.path.join(tmpdir, "thresholds.yaml")
            with open(thresholds_path, "w") as f:
                f.write(thresholds_yaml)
            path = save_safety_case(
                results,
                tmpdir,
                thresholds_path=thresholds_path,
                deep_dive_top_k=3,
            )
            with open(path) as f:
                text = f.read()

        self.assertTrue(path.endswith(".md"))
        self.assertIn("Safety Case Report", text)
        self.assertIn("Threshold Crossings", text)
        self.assertIn("sycophancy", text)
        self.assertIn("Deep Dive Candidates", text)
        self.assertIn("Recommendations", text)

    def test_save_safety_case_handles_missing_or_invalid_yaml(self):
        results = {
            "model-a": EvalResults(
                model="model-a",
                tic_results={
                    "llm_judge_tone": {
                        "analysis": {
                            "neutral": {
                                "n_outputs": 3,
                                "detectors": {
                                    "sycophancy": {
                                        "pct_hit": 10.0,
                                        "pct_hit_ci_low": 0.0,
                                        "pct_hit_ci_high": 56.2,
                                        "mean_count": 0.1,
                                    }
                                },
                            }
                        },
                        "task_segments": {},
                    }
                },
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, "missing.yaml")
            missing_report = save_safety_case(results, tmpdir, thresholds_path=missing_path)
            with open(missing_report) as f:
                missing_text = f.read()

            invalid_path = os.path.join(tmpdir, "invalid.yaml")
            with open(invalid_path, "w") as f:
                f.write("suites: [broken\n")
            invalid_report = save_safety_case(results, tmpdir, thresholds_path=invalid_path)
            with open(invalid_report) as f:
                invalid_text = f.read()

        self.assertIn("Threshold config: not found/empty", missing_text)
        self.assertIn("Threshold config: not found/empty", invalid_text)

    def test_save_safety_case_returns_empty_string_for_empty_results(self):
        self.assertEqual(save_safety_case({}, tempfile.gettempdir()), "")
        self.assertEqual(
            save_safety_case({"model-a": EvalResults(model="model-a")}, tempfile.gettempdir()),
            "",
        )

    def test_save_safety_case_truncates_crossings_table_to_50_rows(self):
        detectors = {
            f"det_{idx}": {
                "pct_hit": 100.0,
                "pct_hit_ci_low": 10.0,
                "pct_hit_ci_high": 95.0,
                "mean_count": 1.0,
            }
            for idx in range(60)
        }
        results = {
            "model-a": EvalResults(
                model="model-a",
                tic_results={
                    "llm_judge_tone": {
                        "analysis": {
                            "playful": {
                                "n_outputs": 12,
                                "low_sample": True,
                                "detectors": detectors,
                            }
                        },
                        "task_segments": {},
                    }
                },
            )
        }
        thresholds_yaml = "suites:\n  llm_judge_tone:\n    detectors:\n" + "".join(
            f"      det_{idx}:\n        max_pct_hit: 1.0\n" for idx in range(60)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            thresholds_path = os.path.join(tmpdir, "thresholds.yaml")
            with open(thresholds_path, "w") as f:
                f.write(thresholds_yaml)
            path = save_safety_case(
                results,
                tmpdir,
                thresholds_path=thresholds_path,
                deep_dive_top_k=0,
            )
            with open(path) as f:
                text = f.read()

        crossing_lines = [
            line for line in text.splitlines()
            if line.startswith("| model-a | llm_judge_tone |")
        ]
        self.assertEqual(len(crossing_lines), 50)

    def test_save_safety_case_includes_uncertain_non_crossing_in_deep_dive(self):
        results = {
            "model-a": EvalResults(
                model="model-a",
                tic_results={
                    "llm_judge_tone": {
                        "analysis": {
                            "neutral": {
                                "n_outputs": 3,
                                "low_sample": True,
                                "detectors": {
                                    "sycophancy": {
                                        "pct_hit": 0.0,
                                        "pct_hit_ci_low": 0.0,
                                        "pct_hit_ci_high": 56.2,
                                        "mean_count": 0.0,
                                    }
                                },
                            }
                        },
                        "task_segments": {},
                    }
                },
            )
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, "missing.yaml")
            path = save_safety_case(
                results,
                tmpdir,
                thresholds_path=missing_path,
                deep_dive_top_k=5,
            )
            with open(path) as f:
                text = f.read()

        self.assertIn("Deep Dive Candidates", text)
        self.assertIn("| model-a | llm_judge_tone | condition | neutral | sycophancy |", text)


if __name__ == "__main__":
    unittest.main()
