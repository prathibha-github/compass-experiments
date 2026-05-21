import json
import os
import tempfile
import unittest

from helm_eval_lex_quirks.metrics.linkedin_style import LINKEDIN_SUITE
from helm_eval_lex_quirks.runner import EvalRunner, TIC_CHECKPOINT_SCHEMA_VERSION


class TicCheckpointTests(unittest.TestCase):
    def test_load_tic_checkpoint_accepts_mixed_legacy_and_current_records(self):
        legacy_record = {
            "model": "dummy",
            "condition": "neutral",
            "prompt_idx": 0,
            "sample_idx": 0,
            "prompt": "Write an update.",
            "completion": "Legacy completion",
            "emoji_count": 0,
            "phrase_hits": 0,
            "linkedin_score": 0,
        }
        current_record = {
            "schema_version": TIC_CHECKPOINT_SCHEMA_VERSION,
            "model": "dummy",
            "suite": LINKEDIN_SUITE.name,
            "condition": "professional",
            "prompt_id": "announcement_0",
            "prompt_idx": 0,
            "sample_idx": 0,
            "prompt": "Write an update.",
            "completion": "Current completion",
            "detectors": {},
        }
        wrong_condition = {
            "schema_version": TIC_CHECKPOINT_SCHEMA_VERSION,
            "model": "dummy",
            "suite": LINKEDIN_SUITE.name,
            "condition": "unknown",
            "prompt_idx": 0,
            "sample_idx": 0,
            "completion": "Skip me",
        }
        wrong_suite = {
            "schema_version": TIC_CHECKPOINT_SCHEMA_VERSION,
            "model": "dummy",
            "suite": "other_suite",
            "condition": "neutral",
            "prompt_idx": 1,
            "sample_idx": 0,
            "completion": "Skip me too",
        }
        unsupported_schema = {
            "schema_version": 999,
            "model": "dummy",
            "suite": LINKEDIN_SUITE.name,
            "condition": "neutral",
            "prompt_idx": 2,
            "sample_idx": 0,
            "completion": "Skip unsupported",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.jsonl")
            with open(path, "w") as f:
                for record in (
                    legacy_record,
                    current_record,
                    wrong_condition,
                    wrong_suite,
                    unsupported_schema,
                    {"condition": "neutral"},
                ):
                    f.write(json.dumps(record) + "\n")
                f.write("{not valid json}\n")

            with self.assertLogs("helm_eval_lex_quirks.runner", level="WARNING") as logs:
                done, condition_outputs = EvalRunner._load_tic_checkpoint(path, LINKEDIN_SUITE)

        self.assertEqual(set(done), {("neutral", 0, 0), ("professional", 0, 0)})
        self.assertEqual(condition_outputs["neutral"], ["Legacy completion"])
        self.assertEqual(condition_outputs["professional"], ["Current completion"])
        self.assertEqual(condition_outputs["linkedin_style"], [])
        self.assertTrue(any("Skipped 5 incompatible" in msg for msg in logs.output))


if __name__ == "__main__":
    unittest.main()
