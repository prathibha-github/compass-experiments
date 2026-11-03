import json
from pathlib import Path
import tempfile
import unittest

from calibrate_detectors import report_labels
from label_responses import legacy_tasks, render_page, write_bundle


class LabelPageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'source.jsonl'
        self.row=dict(model='secret-model',suite='s',condition='secret-condition',prompt_id='p',sample_idx=0,
                      prompt='Help me finish this task.',completion='Take a short break.',
                      detectors={'rest_suggestion':{'hit':True}})
        self.instructions='Label recommendations. Mentions alone do not count.'
        self.write([self.row])

    def write(self,rows):
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def tasks(self):
        return legacy_tasks([self.path],detector='rest_suggestion',instructions=self.instructions)

    def test_export_is_blinded_but_keeps_private_provenance(self):
        tasks,keys=self.tasks()
        html=render_page(tasks)
        for hidden in ('secret-model','secret-condition','predicted_hit',str(self.path.resolve())):
            self.assertNotIn(hidden,html)
        self.assertIn(self.row['completion'],html)
        self.assertIsNone(tasks[0]['system_prompt'])
        self.assertEqual(keys[0]['source_path'],str(self.path.resolve()))
        labels=[dict(tasks[0],human_hit=True,advice_requested=False,review_status='labeled',annotator='person')]
        report=report_labels(keys,labels,annotator='person')
        self.assertFalse(report['source_coverage_verified'])
        self.assertEqual(report['status'],'source_coverage_unverified')
        self.assertEqual(report['label_status'],'complete')
        self.assertEqual(report['agreement'],1.0)

    def test_unsafe_response_is_embedded_as_data(self):
        response='</script><script>alert("bad")</script> __APP__ __CORE__ __DATA__ &'
        self.write([dict(self.row,completion=response)])
        tasks,_=self.tasks()
        html=render_page(tasks)
        self.assertNotIn('</script><script>alert',html)
        payload=html.split('<script type="application/json" id="task-data">')[1].split('</script>')[0]
        self.assertEqual(json.loads(payload)['tasks'][0]['completion'],response)

    def test_duplicates_and_contracted_sources_fail_before_export(self):
        self.write([self.row,self.row])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.tasks()
        self.write([dict(self.row,run_contract='modern')])
        with self.assertRaisesRegex(ValueError,'validated calibration export'):
            self.tasks()

    def test_instructions_change_identity_and_cannot_overwrite_export(self):
        tasks,keys=self.tasks()
        other,_=legacy_tasks([self.path],detector='rest_suggestion',instructions=self.instructions+' Different.')
        self.assertNotEqual(tasks[0]['id'],other[0]['id'])
        out=Path(self.tmp.name)/'labels'
        page=write_bundle(out,tasks,keys)
        before=page.read_bytes()
        with self.assertRaises(FileExistsError):
            write_bundle(out,other)
        self.assertEqual(page.read_bytes(),before)

    def test_wrong_annotation_target_is_rejected(self):
        tasks,_=self.tasks()
        tasks[0]['annotation_target']='unsolicited rest advice'
        with self.assertRaisesRegex(ValueError,'target'):
            render_page(tasks)

    def test_embedded_annotators_cannot_be_misattributed(self):
        tasks,keys=self.tasks()
        labels=[dict(tasks[0],human_hit=True,annotator='Alice')]
        with self.assertRaisesRegex(ValueError,'annotator'):
            report_labels(keys,labels,annotator='Bob')
        with self.assertRaisesRegex(ValueError,'annotator'):
            report_labels(keys,labels)
        self.assertEqual(report_labels(keys,labels,annotator='Alice')['annotator'],'Alice')

    def test_empty_response_is_preserved(self):
        self.write([dict(self.row,completion='')])
        tasks,_=self.tasks()
        self.assertEqual(tasks[0]['completion'],'')
        self.assertIn('Empty response',render_page(tasks))
