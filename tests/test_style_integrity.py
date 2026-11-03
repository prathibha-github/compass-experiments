import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from compass_experiments.config import EvalConfig
from compass_experiments.metrics.tic_suite import LLMJudgeDetector, StyleCondition, StylePrompt, TicSuite
from compass_experiments.runner import EvalRunner


class Client:
    total_tokens = {}
    total_cost_usd = 0

    def __init__(self, judge=False, fail=False):
        self.judge, self.fail, self.calls = judge, fail, 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        if self.fail and self.calls == 1:
            if self.judge:
                return SimpleNamespace(completion="invalid judgment")
            raise RuntimeError("generation failed")
        return SimpleNamespace(completion='{"score":1}' if self.judge else f"response {self.calls}")


class StyleIntegrityTests(unittest.TestCase):
    def suite(self):
        return TicSuite("x", [StylePrompt("p", "Help")], [StyleCondition("c", "Helpful")],
                        [LLMJudgeDetector("d", "Score")], "c")

    def test_retries_preserve_population_and_generated_responses(self):
        for increment in (False, True):
            for failing_phase in ("generation", "judge"):
                with self.subTest(increment=increment, phase=failing_phase), tempfile.TemporaryDirectory() as directory:
                    subject = Client(fail=failing_phase == "generation")
                    judge = Client(judge=True, fail=failing_phase == "judge")
                    runner = EvalRunner(EvalConfig(api_key="dummy", output_dir=directory,
                                                  judge_model="judge", style_samples_increment=increment))
                    with patch("compass_experiments.runner.create_client", return_value=judge):
                        with self.assertRaisesRegex(RuntimeError, "failed samples"):
                            runner._run_tic_suite(subject, "model", self.suite(), samples=2)
                        result = runner._run_tic_suite(subject, "model", self.suite(), samples=2)
                    rows = [json.loads(line) for line in Path(result["checkpoint_path"]).read_text().splitlines()]
                    self.assertEqual({row["sample_idx"] for row in rows}, {0, 1})
                    self.assertEqual(subject.calls, 3 if failing_phase == "generation" else 2)
                    self.assertEqual(result["analysis"]["c"]["n_outputs"], 2)

    def test_changed_prompt_or_judge_rejects_resume_before_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, judge = Client(), Client(judge=True)
            config = EvalConfig(api_key="dummy", output_dir=directory, judge_model="judge")
            runner, suite = EvalRunner(config), self.suite()
            with patch("compass_experiments.runner.create_client", return_value=judge):
                runner._run_tic_suite(subject, "model", suite, samples=1)
                changed = replace(suite, prompts=[StylePrompt("p", "Changed")])
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    runner._run_tic_suite(subject, "model", changed, samples=1)
                config.judge_model = "different-judge"
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    runner._run_tic_suite(subject, "model", suite, samples=1)
            self.assertEqual(subject.calls, 1)
            self.assertEqual(judge.calls, 1)

    def test_primary_and_secondary_judges_receive_saved_request_on_retry(self):
        class RecordingJudge(Client):
            def __init__(self, fail=False):
                super().__init__(judge=True, fail=fail)
                self.prompts = []

            def complete(self, prompt, **kwargs):
                self.prompts.append(prompt)
                return super().complete(prompt, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            subject = Client()
            primary, secondary = RecordingJudge(fail=True), RecordingJudge()
            config = EvalConfig(api_key='dummy', output_dir=directory,
                                judge_model='primary', judge_model_secondary='secondary')
            runner = EvalRunner(config)
            def factory(config, model):
                return primary if model == 'primary' else secondary
            with patch('compass_experiments.runner.create_client', side_effect=factory):
                with self.assertRaises(RuntimeError):
                    runner._run_tic_suite(subject, 'model', self.suite(), samples=1)
                result = runner._run_tic_suite(subject, 'model', self.suite(), samples=1)
            self.assertEqual(subject.calls, 1)
            for prompt in primary.prompts + secondary.prompts:
                self.assertIn('User request:\nHelp', prompt)
                self.assertNotIn('Generation system instructions:', prompt)
            row = json.loads(Path(result['checkpoint_path']).read_text())
            self.assertEqual(row['detectors']['d']['metadata']['judge_context']['user_request'], 'Help')

    def test_changed_shared_judge_prompt_version_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, judge = Client(), Client(judge=True)
            runner = EvalRunner(EvalConfig(api_key='dummy', output_dir=directory, judge_model='judge'))
            with patch('compass_experiments.runner.create_client', return_value=judge):
                runner._run_tic_suite(subject, 'model', self.suite(), samples=1)
                with patch('compass_experiments.runner.JUDGE_PROMPT_VERSION', 'next-version'):
                    with self.assertRaisesRegex(ValueError, 'inputs changed'):
                        runner._run_tic_suite(subject, 'model', self.suite(), samples=1)
            self.assertEqual(subject.calls, 1)
            self.assertEqual(judge.calls, 1)

    def test_duplicate_artifacts_are_rejected_without_calls_or_writes(self):
        for suffix in ['', '.generations.jsonl', '.requests.jsonl']:
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                subject, judge = Client(), Client(judge=True)
                runner = EvalRunner(EvalConfig(api_key='dummy', output_dir=directory, judge_model='judge'))
                with patch('compass_experiments.runner.create_client', return_value=judge):
                    result = runner._run_tic_suite(subject, 'model', self.suite(), samples=1)
                    path = Path(result['checkpoint_path'] + suffix)
                    path.write_text(path.read_text()*2)
                    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
                    with self.assertRaisesRegex(ValueError, 'Duplicate'):
                        runner._run_tic_suite(subject, 'model', self.suite(), samples=1)
                self.assertEqual(subject.calls, 1)
                self.assertEqual(judge.calls, 1)
                self.assertEqual({p.name: p.read_bytes() for p in path.parent.iterdir()}, before)
