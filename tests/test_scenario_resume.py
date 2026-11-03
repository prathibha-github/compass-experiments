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

    def test_invalid_answers_are_incorrect_and_not_calibrated(self):
        for reply in ['I cannot answer.', 'Banana', '', 'Cannot tell']:
            with self.subTest(reply=reply), tempfile.TemporaryDirectory() as directory:
                fake = client()
                fake.complete.return_value.completion = reply
                result = run(Path(directory), fake, robustness=True)
                self.assertEqual(result.accuracy, 0.0)
                self.assertEqual(result.n_completed, 1)
                self.assertEqual(result.n_invalid, 1)
                self.assertIsNone(result.ece)
                import json
                row = json.loads((Path(directory)/'scenario.jsonl').read_text())
                self.assertIsNone(row['prediction'])
                self.assertFalse(row['perturbed_is_correct'])
                self.assertIsNone(row['confidence'])

    def test_parser_requires_a_label_boundary(self):
        from compass_experiments.metrics.accuracy import extract_mc_answer
        for reply in ['A', ' b ', 'C. Explanation', 'D) Reason', 'A: answer', 'A\r\nExplanation', 'B\u00a0Explanation']:
            self.assertEqual(extract_mc_answer(reply, list('ABCD')), reply.strip()[0].upper())
        for reply in ['Banana', 'Actually no', 'CANNOT', 'Definitely', 'E']:
            self.assertIsNone(extract_mc_answer(reply, list('ABCD')))

    def test_parser_version_invalidates_previous_scores(self):
        from unittest.mock import patch
        with patch('compass_experiments.runner.ANSWER_PARSER_VERSION', 1):
            run(self.tmp_path, client())
        fake = client()
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            run(self.tmp_path, fake)
        fake.complete.assert_not_called()

    def test_confidence_version_invalidates_previous_scores_before_calls_or_writes(self):
        from unittest.mock import patch
        with patch('compass_experiments.runner.CONFIDENCE_SCORING_VERSION', 1):
            run(self.tmp_path, client())
        before = {p.name: p.read_bytes() for p in self.tmp_path.iterdir()}
        fake = client()
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            run(self.tmp_path, fake)
        fake.complete.assert_not_called()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.tmp_path.iterdir()})

    def test_saved_token_candidates_reproduce_confidence_without_model_calls(self):
        import json
        import math
        from compass_experiments.metrics.calibration import CONFIDENCE_SCORING_VERSION, extract_choice_probs
        fake = client()
        fake.complete.return_value.logprobs = [SimpleNamespace(token=token, logprob=math.log(probability))
            for token, probability in [('B', .3), ('A', .2), (' A', .2), ('C', .2), ('D', .1)]]
        question = replace(instance(), choices=list('ABCD'), choice_texts=['one', 'two', 'three', 'four'])
        result = run(self.tmp_path, fake, instances=[question])
        self.assertAlmostEqual(result.ece, .6)
        row = json.loads((self.tmp_path/'scenario.jsonl').read_text())
        self.assertEqual(row['confidence_scoring_version'], CONFIDENCE_SCORING_VERSION)
        self.assertEqual(len(row['top_logprobs']), 5)
        probabilities, coverage = extract_choice_probs(row['top_logprobs'], row['choices'])
        self.assertEqual(probabilities, row['choice_probabilities'])
        self.assertEqual(coverage, row['coverage'])
        self.assertEqual(probabilities[row['prediction']], row['confidence'])
        run(self.tmp_path, fake, instances=[question])
        self.assertEqual(fake.complete.call_count, 1)
