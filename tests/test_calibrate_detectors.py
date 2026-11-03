import json
from types import SimpleNamespace

import tempfile
import unittest
from pathlib import Path

from calibrate_detectors import export_tasks as _export_tasks, report_labels

def export_tasks(*args, **kwargs):
    return _export_tasks(*args, annotation_target="unsolicited rest suggestion",
                         instructions="Label true if the response suggests rest without being asked.",
                         instruction_version="1", **kwargs)
from compass.detectors import StyleCondition, StylePrompt, TicSuite, RegexDetector
from compass.evaluation.suite_runner import generate_suite_completions, evaluate_suite_completions


def artifacts(tmp_path, samples=1, legacy=False, extra_detector=False):
    suite = TicSuite("x", [StylePrompt("p", "Help me")], [StyleCondition("c", "Be helpful")],
                     [RegexDetector("rest", "break")], "c")
    if extra_detector:
        suite.detectors.append(RegexDetector("other", "context"))
    class Client:
        def complete(self, **kwargs):
            return SimpleNamespace(completion="Take a break. " + "context " * 100)
    generations = generate_suite_completions(suite, ["model"], samples, tmp_path, client_factory=lambda _: Client())
    if legacy:
        rows = [json.loads(line) for line in generations.read_text().splitlines()]
        for row in rows:
            row.pop("generation_contract")
        generations.write_text("".join(json.dumps(row) + "\n" for row in rows))
        generations.with_suffix(".manifest.json").unlink()
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

    def test_label_task_and_instruction_provenance(self):
        import copy
        generations, evaluations = artifacts(self.path)
        tasks, keys = export_tasks(generations, evaluations, "rest")
        self.assertEqual(keys[0]["prompt_id"], "p")
        self.assertEqual(keys[0]["sample_idx"], 0)
        self.assertEqual(keys[0]["suite"], "x")
        tasks[0]["human_hit"] = True
        tasks[0]["notes"] = "The response suggests a break."
        report = report_labels(keys, tasks, annotator="reviewer-a")
        self.assertEqual(report["annotator"], "reviewer-a")
        for field in ["completion", "prompt", "instructions", "annotation_target", "instruction_version"]:
            changed = copy.deepcopy(tasks)
            changed[0][field] = "changed"
            with self.assertRaisesRegex(ValueError, "changed"):
                report_labels(keys, changed)
        new_tasks, _ = _export_tasks(generations, evaluations, "rest", annotation_target="any rest mention",
                                      instructions="Label any mention of rest.", instruction_version="2")
        self.assertNotEqual(tasks[0]["id"], new_tasks[0]["id"])


    def test_trimming_both_sources_cannot_hide_planned_samples(self):
        generations, evaluations = artifacts(self.path, samples=2)
        for path in (generations, evaluations):
            path.write_text(path.read_text().splitlines()[0] + "\n")
        before = {p.name: p.read_bytes() for p in self.path.iterdir() if p.is_file()}
        with self.assertRaisesRegex(ValueError, "Incomplete.*generations"):
            export_tasks(generations, evaluations, "rest")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.iterdir() if p.is_file()})

    def test_missing_duplicate_and_stale_manifests_are_rejected(self):
        for artifact in ["generation", "evaluation"]:
            for mutation in ["missing", "duplicate", "stale", "shape"]:
                with self.subTest(artifact=artifact, mutation=mutation), tempfile.TemporaryDirectory() as directory:
                    gen, evaluations = artifacts(Path(directory))
                    manifest_path = (gen if artifact=="generation" else evaluations).with_suffix(".manifest.json")
                    manifest = json.loads(manifest_path.read_text())
                    if mutation == "missing":
                        manifest_path.unlink()
                    else:
                        if mutation == "duplicate":
                            manifest["expected"] *= 2
                        elif mutation == "stale":
                            manifest["expected"][0]["contract"] = "obsolete"
                        else:
                            manifest["expected"][0]["identity"].insert(0, "extra")
                        manifest_path.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        export_tasks(gen, evaluations, "rest")

    def test_incomplete_selected_detector_rejected_but_other_detector_can_be_incomplete(self):
        gen, evaluations = artifacts(self.path, samples=2, extra_detector=True)
        rows = [json.loads(line) for line in evaluations.read_text().splitlines()]
        selected = [row for row in rows if row["detector"] == "rest"]
        evaluations.write_text("".join(json.dumps(row) + "\n" for row in selected))
        tasks, keys = export_tasks(gen, evaluations, "rest")
        self.assertEqual(len(tasks), 2)
        for task in tasks:
            task["human_hit"] = True
        report = report_labels(keys, tasks)
        self.assertTrue(report["source_coverage_verified"])
        self.assertEqual(report["status"], "complete")
        evaluations.write_text(json.dumps(selected[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            export_tasks(gen, evaluations, "rest")

    def test_private_key_cannot_silently_shrink(self):
        tasks, keys = export_tasks(*artifacts(self.path, samples=2), "rest")
        tasks[0]["human_hit"] = True
        with self.assertRaisesRegex(ValueError, "population is incomplete"):
            report_labels(keys[:1], [next(task for task in tasks if task["id"] == keys[0]["id"])])

    def test_legacy_sources_and_old_exports_remain_unverified(self):
        tasks, keys = export_tasks(*artifacts(self.path, legacy=True), "rest")
        tasks[0]["human_hit"] = True
        for old_export in [False, True]:
            if old_export:
                keys[0].pop("export_population")
            report = report_labels(keys, tasks)
            self.assertEqual(report["label_status"], "complete")
            self.assertEqual(report["status"], "source_coverage_unverified")
            self.assertFalse(report["source_coverage_verified"])

    def test_cli_rejects_partial_sources_before_creating_export(self):
        from unittest.mock import patch
        from calibrate_detectors import main
        gen, evaluations = artifacts(self.path, samples=2)
        for path in [gen, evaluations]:
            path.write_text(path.read_text().splitlines()[0] + "\n")
        instructions = self.path / "instructions.txt"
        instructions.write_text("Label recommendations to rest.")
        output = self.path / "export"
        argv = ["calibrate_detectors.py", "export", "--generations", str(gen), "--evaluations", str(evaluations),
                "--detector", "rest", "--target", "rest advice", "--instructions", str(instructions),
                "--instruction-version", "1", "--output-dir", str(output)]
        with patch("sys.argv", argv), self.assertRaises(ValueError):
            main()
        self.assertFalse(output.exists())
