import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from compass_experiments.runner import EvalRunner, EvalResults
from compass_experiments.runner_support import attach_prompt_statistics
from compass_experiments.report import save_safety_case, format_table
from compass_experiments.metrics.tic_suite import TicSuite, StylePrompt, StyleCondition, EmojiDetector
from tests._test_base import make_eval_config


class PromptReportingTests(unittest.TestCase):
    def test_repetition_does_not_create_prompt_coverage_and_resume_keeps_statistics(self):
        suite = TicSuite('single_prompt', [StylePrompt('p', 'task')],
                         [StyleCondition('c', 'condition')], [EmojiDetector('emoji')], 'c')
        client = SimpleNamespace(complete=lambda *a, **kw: SimpleNamespace(completion='No emoji.'),
                                 total_cost_usd=0, total_tokens={})
        with tempfile.TemporaryDirectory() as directory:
            runner = EvalRunner(make_eval_config(output_dir=directory))
            result = runner._run_tic_suite(client, 'model', suite, samples=100)
            repeated = runner._run_tic_suite(client, 'model', suite, samples=100)
            self.assertEqual(result['analysis'], repeated['analysis'])
            stats = result['analysis']['c']['detectors']['emoji']['prompt_statistics']['all']
            self.assertEqual(stats['responses'], 100)
            self.assertEqual(stats['distinct_prompts'], 1)
            self.assertIsNone(stats['interval'])
            model = EvalResults('model', tic_results={'single_prompt': result})
            path = save_safety_case({'model': model}, directory)
            report = Path(path).read_text()
            self.assertIn('prompts=1 prompt-ci=N/A', report)
            self.assertNotIn('No uncertainty flags', report)
            self.assertIn('prompts=1 prompt-ci=N/A', format_table({'model': model}))

    def test_total_cost_includes_both_judges_once(self):
        suite = SimpleNamespace(name='s')
        runner = EvalRunner(make_eval_config())
        client = SimpleNamespace(total_cost_usd=1, total_tokens={'input':10, 'output':1})
        result = {'judge': {'cost_usd':10, 'token_usage':{'input':100, 'output':10}},
                  'judge_secondary': {'cost_usd':2, 'token_usage':{'input':20, 'output':2}}}
        with patch.object(runner, '_enabled_tic_suites', return_value=[('s',suite,1)]), \
             patch.object(runner, '_run_tic_suite', return_value=result):
            totals = runner._run_model(client, 'model')
        self.assertEqual(totals.total_cost_usd, 13)
        self.assertEqual(totals.token_usage, {'input':130, 'output':13})

    def test_distinct_prompts_and_mixed_hits_have_cluster_interval(self):
        analysis={'c':{'detectors':{'d':{}}}}
        records=[dict(condition='c',prompt_id=p,sample_idx=i,detectors={'d':{'hit':hit}})
                 for p,hit in [('a',False),('b',True)] for i in range(3)]
        attach_prompt_statistics(analysis, records)
        stats=analysis['c']['detectors']['d']['prompt_statistics']
        self.assertEqual(stats['all']['distinct_prompts'], 2)
        self.assertEqual(stats['all']['hit_rate'], .5)
        self.assertEqual(stats['all']['interval'], [0,1])
        self.assertEqual(stats['quality_unknown'], 6)
