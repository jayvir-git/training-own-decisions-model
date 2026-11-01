"""Meaningful protocol, split-isolation, weighting and recovery tests; no model calls."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import codex_labeler_v2 as labeler
import label_agreement_v2 as runner


class V2Tests(unittest.TestCase):
    def test_three_class_schema_and_independent_flag(self):
        value = {k: v[0] for k, v in labeler.ENUMS.items()}
        value.update(reason='Relevant transferable research.', min_years=2)
        labeler.validate(value)
        for change in [{'verdict': 'needs-information'}, {'needs_verification': True},
                       {'min_years': True}, {'min_years': -1}, {'reason': 'word ' * 61}]:
            with self.assertRaises(ValueError):
                labeler.validate({**value, **change})

    def test_isolation_and_medium(self):
        args = labeler.command(Path('empty'), Path('schema'))
        for setting in ['model_reasoning_effort="medium"', 'features.shell_tool=false',
                        'features.unified_exec=false', 'features.hooks=false', 'features.memories=false',
                        'plugins._default.enabled=false', 'project_doc_max_bytes=0',
                        'forced_login_method="chatgpt"', 'mcp_servers={}']:
            self.assertIn(setting, args)
        for flag in ['--no-daemon', '--ephemeral', '--ignore-user-config', '--ignore-rules', '--strict-config']:
            self.assertIn(flag, args)
        self.assertEqual(args[args.index('-m')+1], 'gpt-6-astra')
        self.assertEqual(args[args.index('-s')+1], 'read-only')

    def test_ni_exclusion_and_abstentions(self):
        m = runner.metrics([('strong','strong'), ('reject','reject'), ('reject',None),
                            ('needs-information','reject')])
        self.assertEqual(m['n'], 3)
        self.assertAlmostEqual(m['accuracy'], 2/3)
        self.assertAlmostEqual(m['kappa'], .5)
        self.assertEqual(m['out_of_class_or_ties']['reject'], 1)
        self.assertEqual(m['classes']['reject']['recall'], .5)
        self.assertIsNone(runner.majority(['strong','stretch','reject']))

    def test_reweighting_recovers_population_not_oversampled_error_rate(self):
        # Census of 2 hard cases and a 1/4 sample of 8 easy cases.
        pairs = [('stretch','reject'), ('stretch','stretch'), ('reject','reject'), ('reject','reject')]
        self.assertEqual(runner.metrics(pairs)['accuracy'], .75)
        self.assertEqual(runner.metrics(pairs, [1,1,4,4])['accuracy'], .9)

    def test_tuning_cannot_retrieve_test_entries(self):
        entries = [{'split':'train'}, {'split':'val'}, {'split':'test','secret':'unseen'}]
        with patch.object(runner, 'read_json', return_value={'entries': entries}):
            self.assertEqual(runner.entries_for(), entries[:2])
            self.assertEqual(runner.entries_for(final=True), entries)

    def test_blind_payload_allowlist(self):
        p = {'title':'Engineer','description':'a'*13000, 'posting_id':'SECRET',
             'split':'SECRET', 'receipt':{'verdict':'SECRET'}, 'reason':'SECRET'}
        prompt = labeler.prompt_for('Facts', p)
        self.assertNotIn('SECRET', prompt)
        payload = json.loads(prompt.split('<posting>\n')[1].split('\n</posting>')[0])
        self.assertEqual(set(payload), {'employer','title','location','posted','description'})
        self.assertEqual(len(payload['description']), 12000)

    def test_duplicate_success_rejected_and_failed_attempt_resumable(self):
        row = {'posting_id':'p','run':1,'status':'ok'}
        self.assertEqual(len(runner.successful([{**row,'status':'failure'},row])), 1)
        with self.assertRaises(RuntimeError):
            runner.successful([row,row])

    def test_verification_counts_all_receipts(self):
        m = runner.binary_metrics([(True,True),(True,False),(False,True),(False,False)])
        self.assertEqual(m['precision'], .5)
        self.assertEqual(m['recall'], .5)

    def test_cohort_ni_exclusion_vote_split_and_incomplete(self):
        entries = [{'posting_id':str(i), 'split':'train', 'receipt':{'verdict':v}}
                   for i,v in enumerate(['reject','strong','needs-information','stretch'])]
        done = {}
        for pid,votes in [('0',['reject']*3),('1',['strong','stretch','reject']),
                          ('2',['stretch']*3),('3',['stretch'])]:
            for r,v in enumerate(votes,1):
                done[pid,r]={'output':{'verdict':v,'needs_verification':'yes' if pid=='2' else 'no'}}
        c=runner.cohort(entries,done)
        self.assertEqual(c['complete_postings'],3)
        self.assertEqual(c['majority']['n'],2)
        self.assertEqual(c['majority']['accuracy'],.5)
        self.assertEqual(c['unanimous_fraction_all'],2/3)
        self.assertEqual(c['unanimous_fraction_scored'],.5)
        self.assertEqual(c['verification']['tp'],1)
        self.assertEqual(c['verification']['n'],3)

    def test_interim_report_before_first_completed_posting(self):
        c=runner.cohort([], {})
        choice={'version':'initial','initial_accuracy':.8,'revision_accuracy':None,
                'projected_full_hours_at_4':1.2}
        result={**c,'choice':choice,'v1_same_cohort':c,'updated_at':'now','complete':False,
                'successful_calls':1,'full_elapsed_wall_seconds':7,
                'rule_conflicts':{'status':'pending','confirmed':[],'candidates':[]}}
        tune={'successful':259,'target':259,'raw':{'accuracy':.7},'reweighted':{'accuracy':.8},
              'strata':{},'timing':{'mean_seconds':7,'median_seconds':6,'p95_seconds':10}}
        with tempfile.TemporaryDirectory(dir=runner.OUT if runner.OUT.exists() else None) as tmp, \
             patch.object(runner,'BASE',Path(tmp)), patch.object(runner,'read_json',return_value=tune):
            runner.report(result,[],{}, {})
            text=(Path(tmp)/'AGREEMENT-v2.md').read_text(encoding='utf-8')
            for required in ['INCOMPLETE','Verification flag','Rule-conflict audit','Three-class majority',
                             'Self-consistency','Ten deterministic disagreements']:
                self.assertIn(required,text)

    def test_worker_bounded_concurrency_resume_and_context_guard(self):
        entries = [{'posting_id':f'p{i}'} for i in range(9)]
        posts = {e['posting_id']:{'description':'Posting'} for e in entries}
        active = peak = 0
        calls = []
        guard = threading.Lock()

        def fake_classify(pid, run, phase, version, post, context, attempt):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
                calls.append((pid,run))
            time.sleep(.01)
            with guard:
                active -= 1
            return {'posting_id':pid,'run':run,'phase':phase,'version':version,'attempt':attempt,
                    'status':'ok','seconds':.01,'rate_limited':False,
                    'prompt_sha256':runner.hashlib.sha256(labeler.prompt_for(context,post).encode()).hexdigest()}

        with tempfile.TemporaryDirectory(dir=runner.OUT if runner.OUT.exists() else None) as tmp:
            root = Path(tmp)
            (root / 'context-initial.md').write_text('Facts',encoding='utf-8')
            with patch.object(runner,'OUT',root), patch.object(runner,'prepare_tuning',return_value={'entries':entries}), \
                 patch.object(runner,'posting_inputs',return_value=posts), \
                 patch.object(runner,'protocol',side_effect=lambda v,e: {'context':runner.digest(runner.context_path(v))}), \
                 patch.object(runner,'classify',side_effect=fake_classify), \
                 patch.object(runner,'summarize_tuning',return_value={}), patch('builtins.print'):
                runner.run_phase('tune','initial')
                self.assertEqual(peak,4)
                self.assertEqual(len(calls),9)
                runner.run_phase('tune','initial')
                self.assertEqual(len(calls),9)
                self.assertEqual(len(runner.jsonl(root/'runs/tuning-initial.jsonl')),9)
                (root/'context-initial.md').write_text('Changed',encoding='utf-8')
                with self.assertRaises(RuntimeError):
                    runner.run_phase('tune','initial')


if __name__ == '__main__':
    unittest.main()
