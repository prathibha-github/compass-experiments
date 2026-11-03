import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from compass_experiments.main import main
from compass_experiments.runner import EvalRunner
from compass_experiments.report import save, format_table
from compass_experiments.scenarios.bbq import BBQScenario
from tests._test_base import make_eval_config


class ScenarioPopulationTests(unittest.TestCase):
    def test_bbq_few_shot_does_not_overlap_test(self):
        rows = [dict(context_condition=kind, context=f'{kind} {i}', question='Q',
                     ans0='A', ans1='B', ans2='Unknown', label=0)
                for kind in ('ambig', 'disambig') for i in range(10)]
        with patch('compass_experiments.scenarios.bbq.load_dataset', return_value={'test': rows}):
            for shots in (0, 2):
                instances, examples = BBQScenario().load(5, shots)
                self.assertEqual(len(instances), 5)
                self.assertEqual(len(examples), shots)
                self.assertFalse({i.input for i in instances} & {i.input for i in examples})
                self.assertEqual(sum(i.metadata['context_condition'] == 'ambig' for i in instances), 3)

    def test_failed_load_is_saved_and_cli_exits_unsuccessfully(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_eval_config(scenarios=['bbq'], style_only=False, n_samples=5,
                                      output_dir=directory)
            client = SimpleNamespace(total_tokens={}, total_cost_usd=0)
            runner = EvalRunner(config)
            with patch.object(BBQScenario, 'load', side_effect=RuntimeError('offline')):
                results = {'model': runner._run_model(client, 'model')}
            fake_runner = Mock()
            fake_runner.estimate_cost.return_value = {}
            fake_runner.run.return_value = results
            with patch('compass_experiments.main.EvalRunner', return_value=fake_runner), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, 'incomplete'):
                    main(['--api-key', 'fake', '--output-dir', directory])
            path, _ = save(results, directory)
            with open(path) as stream:
                row = json.load(stream)['model']['scenarios']['bbq']
            self.assertEqual((row['status'], row['n_requested'], row['n_completed']), ('failed', 5, 0))
            self.assertIsNone(row['accuracy'])
            text = format_table(results)
            self.assertIn('offline', text)
            self.assertNotIn('Mean ', text)

    def test_mmlu_balances_subjects_and_uses_matching_demonstrations(self):
        from collections import Counter
        from compass_experiments.scenarios.mmlu import MMLUScenario, SUBJECTS
        def dataset(_name, subject, **kwargs):
            self.assertIn(subject, {"anatomy", "high_school_us_history", "college_mathematics", "professional_law"})
            return {split: [dict(question=f'{subject} {split} {i}', choices=['a','b','c','d'], answer=0)
                            for i in range(8)] for split in ('test','dev')}
        with patch('compass_experiments.scenarios.mmlu.load_dataset', side_effect=dataset):
            scenario = MMLUScenario()
            instances, examples = scenario.load(5, 3)
            counts = Counter(i.metadata['subject'] for i in instances)
            self.assertEqual(sorted(counts.values()), [1,1,1,2])
            with patch('compass_experiments.scenarios.mmlu.SUBJECTS', list(reversed(SUBJECTS))):
                self.assertEqual((instances, examples), scenario.load(5, 3))
            for instance in instances:
                prompt = scenario.format_prompt(instance, examples)
                self.assertEqual(prompt.count('Answer: A'), 3)
                for subject in SUBJECTS:
                    self.assertEqual(f'{subject} dev' in prompt, subject == instance.metadata['subject'])

    def test_short_dataset_preserves_requested_count(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = EvalRunner(make_eval_config(scenarios=['bbq'], style_only=False,
                n_samples=5, output_dir=directory))
            client = SimpleNamespace(total_tokens={}, total_cost_usd=0)
            with patch.object(BBQScenario, 'load', return_value=([], [])):
                result = runner._run_model(client, 'model')
            row = result.scenario_results['bbq']
            self.assertEqual((row.status, row.n_requested, row.n_completed), ('incomplete', 5, 0))
