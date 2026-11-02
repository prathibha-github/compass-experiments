import json
import os
import tempfile
import unittest

from compass_experiments.runner import EvalResults
from compass_experiments.metrics.pairwise_preference import analyze_pairwise_preference
from compass_experiments.report import format_table


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class PairwisePreferenceTests(unittest.TestCase):
    def test_analyze_pairwise_preference_ranks_lower_tic_counts_higher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = os.path.join(tmpdir, "left.jsonl")
            right_path = os.path.join(tmpdir, "right.jsonl")
            _write_jsonl(
                left_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "task_type": "summary",
                        "detectors": {
                            "emoji": {"count": 0},
                            "semicolon": {"count": 0},
                        },
                    },
                    {
                        "condition": "neutral",
                        "prompt_id": "p1",
                        "sample_idx": 0,
                        "task_type": "technical",
                        "detectors": {
                            "emoji": {"count": 2},
                            "semicolon": {"count": 0},
                        },
                    },
                ],
            )
            _write_jsonl(
                right_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "task_type": "summary",
                        "detectors": {
                            "emoji": {"count": 1},
                            "semicolon": {"count": 0},
                        },
                    },
                    {
                        "condition": "neutral",
                        "prompt_id": "p1",
                        "sample_idx": 0,
                        "task_type": "technical",
                        "detectors": {
                            "emoji": {"count": 2},
                            "semicolon": {"count": 0},
                        },
                    },
                ],
            )

            results = {
                "left-model": EvalResults(
                    model="left-model",
                    tic_results={
                        "punctuation_tics": {"checkpoint_path": left_path},
                    },
                ),
                "right-model": EvalResults(
                    model="right-model",
                    tic_results={
                        "punctuation_tics": {"checkpoint_path": right_path},
                    },
                ),
            }

            pairwise = analyze_pairwise_preference(results)

        ranking = pairwise["punctuation_tics"]["overall"]["ranking"]
        self.assertEqual(ranking[0]["model"], "left-model")
        self.assertEqual(ranking[0]["wins"], 1)
        self.assertEqual(ranking[0]["ties"], 1)
        self.assertEqual(ranking[0]["preference_score"], 1.5)
        self.assertIn("summary", pairwise["punctuation_tics"]["task_types"])

    def test_report_includes_pairwise_section(self):
        results = {
            "left-model": EvalResults(model="left-model"),
            "right-model": EvalResults(model="right-model"),
        }
        pairwise = {
            "punctuation_tics": {
                "overall": {
                    "ranking": [
                        {
                            "model": "left-model",
                            "preference_score": 1.5,
                            "matches": 2,
                            "wins": 1,
                            "losses": 0,
                            "ties": 1,
                        }
                    ]
                },
                "task_types": {},
            }
        }
        for result in results.values():
            result.pairwise_preference_results = pairwise

        text = format_table(results)

        self.assertIn("PAIRWISE PREFERENCE", text)
        self.assertIn("left-model", text)

    def test_report_describes_single_detector_pairwise_scoring(self):
        results = {
            "left-model": EvalResults(model="left-model"),
            "right-model": EvalResults(model="right-model"),
        }
        pairwise = {
            "rest_cai_validation": {
                "scoring": {
                    "mode": "single_detector",
                    "detector": "rest_suggestion",
                },
                "overall": {
                    "ranking": [
                        {
                            "model": "left-model",
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
        }
        for result in results.values():
            result.pairwise_preference_results = pairwise

        text = format_table(results)

        self.assertIn("Lower rest_suggestion detector count wins", text)

    def test_analyze_pairwise_preference_can_score_one_detector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = os.path.join(tmpdir, "left.jsonl")
            right_path = os.path.join(tmpdir, "right.jsonl")
            _write_jsonl(
                left_path,
                [
                    {
                        "condition": "default",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "task_type": "rest_probe",
                        "detectors": {
                            "rest_suggestion": {"count": 0},
                            "wellbeing_justification": {"count": 5},
                        },
                    },
                    {
                        "condition": "default",
                        "prompt_id": "p1",
                        "sample_idx": 0,
                        "task_type": "rest_probe",
                        "detectors": {
                            "wellbeing_justification": {"count": 0},
                        },
                    },
                ],
            )
            _write_jsonl(
                right_path,
                [
                    {
                        "condition": "default",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "task_type": "rest_probe",
                        "detectors": {
                            "rest_suggestion": {"count": 1},
                            "wellbeing_justification": {"count": 0},
                        },
                    },
                    {
                        "condition": "default",
                        "prompt_id": "p1",
                        "sample_idx": 0,
                        "task_type": "rest_probe",
                        "detectors": {
                            "rest_suggestion": {"count": 0},
                        },
                    },
                ],
            )
            results = {
                "left-model": EvalResults(
                    model="left-model",
                    tic_results={"rest_cai_validation": {"checkpoint_path": left_path}},
                ),
                "right-model": EvalResults(
                    model="right-model",
                    tic_results={"rest_cai_validation": {"checkpoint_path": right_path}},
                ),
            }

            pairwise = analyze_pairwise_preference(results, detector_name="rest_suggestion")

        suite = pairwise["rest_cai_validation"]
        ranking = suite["overall"]["ranking"]
        self.assertEqual(suite["scoring"]["detector"], "rest_suggestion")
        self.assertEqual(ranking[0]["model"], "left-model")
        self.assertEqual(ranking[0]["wins"], 1)
        self.assertEqual(ranking[0]["matches"], 1)

    def test_analyze_pairwise_preference_handles_legacy_scores_and_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = os.path.join(tmpdir, "left.jsonl")
            right_path = os.path.join(tmpdir, "right.jsonl")
            _write_jsonl(
                left_path,
                [
                    {
                        "condition": "linkedin_style",
                        "prompt_idx": 0,
                        "sample_idx": 0,
                        "emoji_count": 0,
                        "phrase_hits": 1,
                    }
                ],
            )
            with open(left_path, "a") as f:
                f.write("{not valid json}\n\n")
            _write_jsonl(
                right_path,
                [
                    {
                        "condition": "linkedin_style",
                        "prompt_idx": 0,
                        "sample_idx": 0,
                        "task_type": "announcement",
                        "linkedin_score": 3,
                    }
                ],
            )

            results = {
                "left-model": EvalResults(
                    model="left-model",
                    tic_results={"linkedin_style": {"checkpoint_path": left_path}},
                ),
                "right-model": EvalResults(
                    model="right-model",
                    tic_results={"linkedin_style": {"checkpoint_path": right_path}},
                ),
            }

            pairwise = analyze_pairwise_preference(results)

        suite = pairwise["linkedin_style"]
        ranking = suite["overall"]["ranking"]
        self.assertEqual(ranking[0]["model"], "left-model")
        self.assertEqual(ranking[0]["wins"], 1)
        self.assertIn("announcement", suite["task_types"])

    def test_analyze_pairwise_preference_ignores_single_model_suites(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "solo.jsonl")
            _write_jsonl(
                path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 1}},
                    }
                ],
            )
            results = {
                "solo-model": EvalResults(
                    model="solo-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": path}},
                )
            }

            self.assertEqual(analyze_pairwise_preference(results), {})

    def test_analyze_pairwise_preference_skips_suites_without_shared_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = os.path.join(tmpdir, "left.jsonl")
            right_path = os.path.join(tmpdir, "right.jsonl")
            _write_jsonl(
                left_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 0}},
                    }
                ],
            )
            _write_jsonl(
                right_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p1",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 0}},
                    }
                ],
            )
            results = {
                "left-model": EvalResults(
                    model="left-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": left_path}},
                ),
                "right-model": EvalResults(
                    model="right-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": right_path}},
                ),
            }

            self.assertEqual(analyze_pairwise_preference(results), {})

    def test_analyze_pairwise_preference_excludes_models_without_comparable_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = os.path.join(tmpdir, "left.jsonl")
            right_path = os.path.join(tmpdir, "right.jsonl")
            disconnected_path = os.path.join(tmpdir, "disconnected.jsonl")
            _write_jsonl(
                left_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 0}},
                    }
                ],
            )
            _write_jsonl(
                right_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 1}},
                    }
                ],
            )
            _write_jsonl(
                disconnected_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "other",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 0}},
                    }
                ],
            )
            results = {
                "left-model": EvalResults(
                    model="left-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": left_path}},
                ),
                "right-model": EvalResults(
                    model="right-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": right_path}},
                ),
                "disconnected-model": EvalResults(
                    model="disconnected-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": disconnected_path}},
                ),
            }

            pairwise = analyze_pairwise_preference(results)

        suite = pairwise["punctuation_tics"]
        ranked_models = {row["model"] for row in suite["overall"]["ranking"]}
        self.assertEqual(set(suite["models"]), {"left-model", "right-model"})
        self.assertNotIn("disconnected-model", ranked_models)

    def test_analyze_pairwise_preference_supports_bradley_terry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            strong_path = os.path.join(tmpdir, "strong.jsonl")
            weak_path = os.path.join(tmpdir, "weak.jsonl")
            _write_jsonl(
                strong_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 0}},
                    },
                    {
                        "condition": "neutral",
                        "prompt_id": "p1",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 0}},
                    },
                    {
                        "condition": "neutral",
                        "prompt_id": "p2",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 1}},
                    },
                ],
            )
            _write_jsonl(
                weak_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 2}},
                    },
                    {
                        "condition": "neutral",
                        "prompt_id": "p1",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 1}},
                    },
                    {
                        "condition": "neutral",
                        "prompt_id": "p2",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 1}},
                    },
                ],
            )
            results = {
                "strong-model": EvalResults(
                    model="strong-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": strong_path}},
                ),
                "weak-model": EvalResults(
                    model="weak-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": weak_path}},
                ),
            }
            pairwise = analyze_pairwise_preference(results, method="bradley_terry")

        ranking = pairwise["punctuation_tics"]["overall"]["ranking"]
        self.assertEqual(pairwise["punctuation_tics"]["scoring"]["pairwise_method"], "bradley_terry")
        self.assertEqual(ranking[0]["model"], "strong-model")

    def test_analyze_pairwise_preference_can_segment_by_condition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = os.path.join(tmpdir, "left.jsonl")
            right_path = os.path.join(tmpdir, "right.jsonl")
            _write_jsonl(
                left_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 0}},
                    },
                    {
                        "condition": "playful",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 2}},
                    },
                ],
            )
            _write_jsonl(
                right_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 1}},
                    },
                    {
                        "condition": "playful",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "detectors": {"emoji": {"count": 0}},
                    },
                ],
            )
            results = {
                "left-model": EvalResults(
                    model="left-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": left_path}},
                ),
                "right-model": EvalResults(
                    model="right-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": right_path}},
                ),
            }
            pairwise = analyze_pairwise_preference(results, segment_by="condition")

        segments = pairwise["punctuation_tics"]["segments"]
        self.assertEqual(pairwise["punctuation_tics"]["scoring"]["segment_by"], "condition")
        self.assertIn("neutral", segments)
        self.assertIn("playful", segments)
        self.assertEqual(pairwise["punctuation_tics"]["task_types"], {})

    def test_analyze_pairwise_preference_time_decay_prefers_recent_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = os.path.join(tmpdir, "left.jsonl")
            right_path = os.path.join(tmpdir, "right.jsonl")
            _write_jsonl(
                left_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "old",
                        "sample_idx": 0,
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "detectors": {"emoji": {"count": 0}},
                    },
                    {
                        "condition": "neutral",
                        "prompt_id": "new",
                        "sample_idx": 0,
                        "created_at": "2099-01-01T00:00:00+00:00",
                        "detectors": {"emoji": {"count": 2}},
                    },
                ],
            )
            _write_jsonl(
                right_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "old",
                        "sample_idx": 0,
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "detectors": {"emoji": {"count": 2}},
                    },
                    {
                        "condition": "neutral",
                        "prompt_id": "new",
                        "sample_idx": 0,
                        "created_at": "2099-01-01T00:00:00+00:00",
                        "detectors": {"emoji": {"count": 0}},
                    },
                ],
            )
            results = {
                "left-model": EvalResults(
                    model="left-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": left_path}},
                ),
                "right-model": EvalResults(
                    model="right-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": right_path}},
                ),
            }
            pairwise = analyze_pairwise_preference(
                results,
                time_decay_halflife_days=30,
            )

        suite = pairwise["punctuation_tics"]
        ranking = suite["overall"]["ranking"]
        self.assertEqual(ranking[0]["model"], "right-model")
        self.assertLess(suite["overall"]["pairs"]["left-model::right-model"]["matches"], 2.0)

    def test_analyze_pairwise_preference_defaults_remain_backward_compatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = os.path.join(tmpdir, "left.jsonl")
            right_path = os.path.join(tmpdir, "right.jsonl")
            _write_jsonl(
                left_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "task_type": "summary",
                        "detectors": {"emoji": {"count": 0}},
                    }
                ],
            )
            _write_jsonl(
                right_path,
                [
                    {
                        "condition": "neutral",
                        "prompt_id": "p0",
                        "sample_idx": 0,
                        "task_type": "summary",
                        "detectors": {"emoji": {"count": 1}},
                    }
                ],
            )
            results = {
                "left-model": EvalResults(
                    model="left-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": left_path}},
                ),
                "right-model": EvalResults(
                    model="right-model",
                    tic_results={"punctuation_tics": {"checkpoint_path": right_path}},
                ),
            }
            pairwise = analyze_pairwise_preference(results)

        suite = pairwise["punctuation_tics"]
        self.assertEqual(suite["scoring"]["pairwise_method"], "count")
        self.assertEqual(suite["scoring"]["segment_by"], "task_type")
        self.assertIn("summary", suite["task_types"])


if __name__ == "__main__":
    unittest.main()
