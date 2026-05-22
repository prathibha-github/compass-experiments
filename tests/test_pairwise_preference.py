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


if __name__ == "__main__":
    unittest.main()
