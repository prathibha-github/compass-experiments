import json
from pathlib import Path
import tempfile
import unittest

from archived_labeling import archived_tasks
from calibrate_detectors import report_labels
from label_responses import render_page


class ArchivedLabelingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'suite_generations.jsonl'
        self.rows = [dict(suite_record_type='suite_generation', suite='rest_offtask',
                         model='hidden-model', condition=condition, prompt_id='p', sample_idx=sample,
                         prompt_text='Help me with the task.', completion='Take a break.' if sample else '',
                         hit_token_cap=bool(sample), is_fragment=bool(sample), quality_flagged=bool(sample))
                     for condition in ('default', 'hidden-condition') for sample in range(3)]
        self.write(self.rows)

    def write(self, rows):
        self.path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def export(self, **kwargs):
        return archived_tasks([self.path], instructions='Label rest advice.', **kwargs)

    def test_stable_sampling_covers_cells_and_records_design(self):
        tasks, keys = self.export()
        self.write(list(reversed(self.rows)))
        other, _ = self.export()
        self.assertEqual(tasks, other)
        self.assertEqual(len(tasks), 2)
        self.assertEqual({key['condition'] for key in keys}, {'default', 'hidden-condition'})
        self.assertTrue(all(key['inclusion_probability'] == 1/3 for key in keys))
        self.assertEqual(keys[0]['sampling']['available_count'], 6)
        html = render_page(tasks)
        for private in ('hidden-model', 'hidden-condition', str(self.path), 'predicted_hit'):
            self.assertNotIn(private, html)
        self.assertTrue(all(task['system_prompt'] is None for task in tasks))

    def test_preserves_empty_and_flagged_responses_and_rescores_locally(self):
        tasks, keys = self.export(samples_per_cell=10)
        self.assertEqual(len(tasks), 6)
        self.assertEqual(sum(task['completion'] == '' for task in tasks), 2)
        self.assertEqual(sum(task['generation_quality']['hit_token_cap'] for task in tasks), 4)
        self.assertEqual(sum(key['predicted_hit'] for key in keys), 4)
        self.assertTrue(all(key['prediction_origin'] == 'rescored_current_heuristic' for key in keys))
        self.assertIn('implementation', keys[0]['detector_definition']['definition'])
        report = report_labels(keys, [dict(task, human_hit=False) for task in tasks])
        self.assertFalse(report['source_coverage_verified'])
        self.assertIn('not a population-weighted', report['sampling_note'])
        tasks[0]['generation_quality']['is_fragment'] = None
        with self.assertRaisesRegex(ValueError, 'changed'):
            report_labels(keys, tasks)

    def test_rejects_invalid_rows_even_if_not_selected(self):
        for mutation in (dict(sample_idx=True), dict(sample_idx=-1), dict(prompt_text=None),
                         dict(completion=None), dict(suite_record_type='suite_evaluation'),
                         dict(hit_token_cap='yes'), dict(generation_contract='modern')):
            with self.subTest(mutation=mutation):
                self.write(self.rows + [dict(self.rows[0], sample_idx=999) | mutation])
                with self.assertRaises(ValueError):
                    self.export()

    def test_empty_duplicate_conflicting_and_modern_sources_rejected(self):
        for rows in ([], self.rows + self.rows[:1],
                     self.rows + [dict(self.rows[0], sample_idx=9, prompt_text='Different prompt')]):
            self.write(rows)
            with self.assertRaises(ValueError):
                self.export()
        self.write(self.rows)
        self.path.with_suffix('.manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'validated'):
            self.export()

    def test_pending_request_edits_not_reported(self):
        tasks, keys = self.export()
        tasks[0]['request_draft'] = dict(label=False, notes='')
        with self.assertRaisesRegex(ValueError, 'pending'):
            report_labels(keys, tasks)


if __name__ == '__main__':
    unittest.main()
