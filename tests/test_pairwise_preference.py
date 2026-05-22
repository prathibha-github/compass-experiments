import json
import os
import tempfile
import unittest

from helm_eval_lex_quirks.runner import EvalResults
from helm_eval_lex_quirks.metrics.pairwise_preference import analyze_pairwise_preference
from helm_eval_lex_quirks.report import format_table


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


if __name__ == "__main__":
    unittest.main()
