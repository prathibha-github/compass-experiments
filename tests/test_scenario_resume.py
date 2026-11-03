from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import tempfile
import unittest
from pathlib import Path

from compass_experiments.runner import EvalRunner
from compass_experiments.scenarios.base import Instance
from tests._test_base import make_eval_config


class Scenario:
    name = 'test'

    def format_prompt(self, inst, few_shot):
        return str([(item.input, item.correct) for item in few_shot]) + inst.input


def instance():
    return Instance('id', 'Question', ['A', 'B'], ['yes', 'no'], 'A')


def run(tmp_path, client, instances=None, few_shot=None, robustness=False, seed=42):
    return EvalRunner(make_eval_config(output_dir=str(tmp_path), seed=seed))._eval_scenario(
        client, Scenario(), instances if instances is not None else [instance()],
        few_shot or [], robustness, str(tmp_path/'scenario.jsonl'), model_name='test-model',
    )


def client():
    return Mock(complete=Mock(return_value=SimpleNamespace(completion='A', logprobs=[])))


class ScenarioResumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_resume_makes_no_calls(self):
        fake = client()
        run(self.tmp_path, fake)
        run(self.tmp_path, fake)
        self.assertEqual(fake.complete.call_count, 1)

    def test_changed_inputs_fail_before_calls_or_writes(self):
        cases = {
            'prompt': {'instances': [replace(instance(), input='Changed')]},
            'answer': {'instances': [replace(instance(), correct='B')]},
            'few_shot': {'few_shot': [instance()]},
            'robustness': {'robustness': True},
            'seed': {'seed': 7},
            'subset': {'instances': []},
        }
        for change, args in cases.items():
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                tmp_path = Path(directory)
                run(tmp_path, client())
                before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
                fake = client()
                with self.assertRaisesRegex(ValueError, 'inputs changed'):
                    run(tmp_path, fake, **args)
                fake.complete.assert_not_called()
                self.assertEqual({p.name: p.read_bytes() for p in tmp_path.iterdir()}, before)

    def test_invalid_or_legacy_checkpoint_is_preserved(self):
        for contents in ['{bad json', '{"id":"id"}\n', '{"id":"id"}\n{"id":"id"}\n']:
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                tmp_path = Path(directory)
                path = tmp_path/'scenario.jsonl'
                path.write_text(contents)
                fake = client()
                with self.assertRaises(ValueError):
                    run(tmp_path, fake)
                fake.complete.assert_not_called()
                self.assertEqual(path.read_text(), contents)

    def test_duplicate_input_ids_rejected_before_calls(self):
        fake = client()
        with self.assertRaisesRegex(ValueError, 'must be unique'):
            run(self.tmp_path, fake, instances=[instance(), instance()])
        fake.complete.assert_not_called()

    def test_failed_instances_remain_in_plan_and_resume(self):
        fake = client()
        fake.complete.side_effect = [fake.complete.return_value, RuntimeError('offline')]
        instances = [instance(), replace(instance(), id='second')]
        with self.assertRaisesRegex(RuntimeError, 'resume to retry'):
            run(self.tmp_path, fake, instances=instances)
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            run(self.tmp_path, client())  # Cannot silently drop the failed instance.
        fake = client()
        result = run(self.tmp_path, fake, instances=instances)
        self.assertEqual(fake.complete.call_count, 1)
        self.assertEqual(result.n_completed, 2)
        self.assertEqual(result.n_requested, 2)

    def test_request_settings_are_sent_to_both_calls(self):
        fake = client()
        run(self.tmp_path, fake, robustness=True)
        self.assertEqual(fake.complete.call_count, 2)
        for call in fake.complete.call_args_list:
            self.assertEqual(call.kwargs['temperature'], 0.0)
            self.assertEqual(call.kwargs['system'], 'You are a helpful assistant.')
            self.assertEqual(call.kwargs['max_tokens'], 1)
        self.assertIs(fake.complete.call_args_list[0].kwargs['logprobs'], True)
        self.assertIs(fake.complete.call_args_list[1].kwargs['logprobs'], False)
