import json
from types import SimpleNamespace

import tempfile
import unittest
from pathlib import Path

from calibrate_detectors import export_tasks, report_labels
from compass.detectors import StyleCondition, StylePrompt, TicSuite, RegexDetector
from compass.evaluation.suite_runner import generate_suite_completions, evaluate_suite_completions


def artifacts(tmp_path):
    suite = TicSuite("x", [StylePrompt("p", "Help me")], [StyleCondition("c", "Be helpful")],
                     [RegexDetector("rest", "break")], "c")
    class Client:
        def complete(self, **kwargs):
            return SimpleNamespace(completion="Take a break. " + "context " * 100)
    generations = generate_suite_completions(suite, ["model"], 1, tmp_path, client_factory=lambda _: Client())
    evaluations = evaluate_suite_completions(suite, generations, tmp_path)
    return generations, evaluations


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def test_blind_export_preserves_full_context_and_missing_labels(self):
        tasks, keys = export_tasks(*artifacts(self.path), "rest")
        assert "predicted_hit" not in tasks[0] and "model" not in tasks[0]
        assert len(tasks[0]["completion"]) > 200
        assert tasks[0]["system_prompt"] == "Be helpful"
        report = report_labels(keys, tasks)
        assert report["agreement"] is None and report["unlabeled"] == 1
        tasks[0]["human_hit"] = False
        assert report_labels(keys, tasks)["false_positive"] == 1
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report_labels(keys, tasks * 2)


    def test_calibration_rejects_stale_completions(self):
        gen, evaluations = artifacts(self.path)
        row = json.loads(gen.read_text())
        row["completion"] = "a different response"
        gen.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            export_tasks(gen, evaluations, "rest")
