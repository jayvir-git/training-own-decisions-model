"""Protocol/metric regression checks; no model calls or source-tree access."""
import json
from pathlib import Path
import unittest

from codex_labeler import ENUMS, command, prompt_for, validate
from label_agreement import clean_postings, metrics, successful


class AgreementTests(unittest.TestCase):
    def test_appended_reviewer_notes_are_removed_without_mutating_posting(self):
        original = {'p': {'description': 'Official qualifications\n\nNative screening notes:\nSECRET verdict and reason'}}
        cleaned, changes = clean_postings(original)
        self.assertEqual(cleaned['p']['description'], 'Official qualifications')
        self.assertIn('SECRET', original['p']['description'])
        self.assertEqual(len(changes), 1)
        untouched, changes = clean_postings({'p': {'description': 'Perform background screening tasks.'}})
        self.assertEqual(changes, [])
        self.assertEqual(untouched['p']['description'], 'Perform background screening tasks.')

    def test_blinding_allowlist_and_truncation(self):
        posting = {'employer': 'Acme', 'title': 'Developer', 'locations': ['NJ'],
                   'description': 'x' * 13000, 'verdict': 'SECRET_LABEL',
                   'reason': 'SECRET_REASON', 'posting_id': 'SECRET_ID',
                   'receipt': {'reason': 'SECRET_RECEIPT'}, 'source_ids': ['SECRET_SOURCE']}
        prompt = prompt_for('FACTS_ONLY', posting)
        self.assertNotIn('SECRET', prompt)
        payload = json.loads(prompt.split('<posting>\n')[1].split('\n</posting>')[0])
        self.assertEqual(set(payload), {'employer', 'title', 'location', 'posted', 'description'})
        self.assertEqual(len(payload['description']), 12000)
        self.assertIsNone(payload['posted'])

    def test_schema_validation(self):
        output = {k: v[0] for k, v in ENUMS.items()}
        output.update(reason='Supported fundamentals.', min_years=None)
        self.assertEqual(validate(output), output)
        for changes in [{'min_years': True}, {'min_years': -1}, {'min_years': 2.5},
                        {'reason': 'word ' * 61}, {'verdict': 'apply'}, {'extra': 'bad'}]:
            with self.assertRaises(ValueError):
                validate({**output, **changes})

    def test_kappa_confusion_and_ties(self):
        result = metrics([('strong', 'strong'), ('strong', 'reject'),
                          ('reject', 'reject'), ('reject', None)])
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['always_reject_accuracy'], .5)
        self.assertAlmostEqual(result['kappa'], .2)
        self.assertEqual(result['confusion'][0][2], 1)
        self.assertEqual(result['classes']['reject']['precision'], .5)
        self.assertEqual(result['classes']['reject']['recall'], .5)
        self.assertEqual(result['ties_by_class']['reject'], 1)
        self.assertIsNone(metrics([])['accuracy'])

    def test_resumption_ignores_failure_but_rejects_duplicate_success(self):
        failed = {'posting_id': 'p', 'run': 1, 'status': 'failure'}
        ok = {**failed, 'status': 'ok'}
        self.assertEqual(set(successful([failed, ok])), {('p', 1)})
        with self.assertRaises(RuntimeError):
            successful([ok, ok])

    def test_isolated_cli_settings(self):
        argv = command(Path('empty'), Path('schema.json'))
        for flag in ['--no-daemon', '--ignore-user-config', '--ignore-rules', '--ephemeral', '--strict-config']:
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index('-m') + 1], 'gpt-6-astra')
        self.assertEqual(argv[argv.index('-s') + 1], 'read-only')
        for setting in ['features.shell_tool=false', 'features.unified_exec=false',
                        'features.hooks=false', 'features.memories=false',
                        'plugins._default.enabled=false', 'project_doc_max_bytes=0',
                        'forced_login_method="chatgpt"']:
            self.assertIn(setting, argv)


if __name__ == '__main__':
    unittest.main()
