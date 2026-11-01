"""Quota/sampling/transport regression checks. No model calls."""
import json
from pathlib import Path
import tempfile
import sys
import threading
import time
import unittest
from unittest.mock import patch

import codex_labeler_v3 as transport
import label_agreement_v3 as runner


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=runner.ROOT)
        self.out = Path(self.tmp.name)
        self.patch = patch.object(runner, 'OUT', self.out)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_budget_reserves_concurrent_calls_and_caps(self):
        b = runner.Budget()
        b.data['input_tokens'] = runner.TOTAL - transport.RESERVATION
        self.assertIsNone(b.reserve('one'))
        self.assertEqual(b.reserve('two'), 'total_budget_cap')
        b.settle({'call_id': 'one', 'day': runner.day(), 'usage': {'input_tokens': 100}})
        self.assertEqual(b.data['input_tokens'], runner.TOTAL - transport.RESERVATION + 100)
        self.assertEqual(b.reserve('two'), 'total_budget_cap')
        b.settle({'call_id': 'one', 'day': runner.day(), 'usage': {'input_tokens': 100}})
        self.assertEqual(b.data['input_tokens'], runner.TOTAL - transport.RESERVATION + 100)

    def test_daily_and_unknown_usage_fail_closed(self):
        b = runner.Budget()
        b.data['days'][runner.day()] = runner.DAILY - transport.RESERVATION
        self.assertIsNone(b.reserve('one'))
        b.settle({'call_id': 'one', 'day': runner.day(), 'usage': None})
        self.assertIn('one', b.data['reservations'])
        self.assertEqual(b.reserve('two'), 'daily_cap')
        restored = runner.Budget().data
        self.assertEqual(restored['reservations'], b.data['reservations'])
        self.assertEqual(restored['input_tokens'], b.data['input_tokens'])

    def test_limit_detection(self):
        for text in ['Usage limit reached', 'rate_limit', 'HTTP 429', 'rate-limit exceeded']:
            self.assertTrue(transport.is_limit(text))
        self.assertFalse(transport.is_limit('13429 tokens'))

    def test_batch_validation(self):
        value = {'verdict': 'strong', 'needs_verification': 'no', 'sponsorship_blocked': 'unknown',
                 'clearance_or_citizenship_required': 'no', 'level': 'entry-I', 'track': 'software',
                 'reason': 'Supported evidence.', 'min_years': 0}
        items = [{'posting_id': pid, **value} for pid in ('a', 'b')]
        self.assertEqual(transport.validate_batch(items, ['a', 'b']), items)
        with self.assertRaises(ValueError):
            transport.validate_batch([items[0], items[0]], ['a', 'b'])
        with self.assertRaises(ValueError):
            transport.validate_batch(items[:1], ['a', 'b'])

    def test_soft_entry(self):
        def eligible(title, description):
            return runner.soft_entry({'title': title, 'description': description})
        self.assertTrue(eligible('Software Engineer', 'Requires 0-2 years experience.'))
        self.assertTrue(eligible('Software Engineer', 'Python and a bachelors degree.'))
        self.assertFalse(eligible('Software Engineer II', 'No experience required.'))
        self.assertFalse(eligible('Lead Software Engineer', '1 year experience.'))
        self.assertFalse(eligible('Software Engineer', 'At least 3-5 years of experience.'))

    def test_single_and_batch_blinding(self):
        p = {'title': 'Engineer', 'description': 'x'*15000, 'receipt': 'SECRET', 'split': 'test', 'score': 999}
        prompt, schema = runner.request(['a'], {'a': p}, 'context', False)
        self.assertNotIn('SECRET', prompt)
        self.assertNotIn('999', prompt)
        self.assertEqual(prompt.count('x'), 12001)  # context includes x
        prompt, schema = runner.request(['a', 'b'], {'a': p, 'b': p}, 'context', True)
        self.assertNotIn('SECRET', prompt)
        self.assertEqual(schema['maxItems'], 2)

    def test_streamed_limit_kills_active_sibling(self):
        fake = self.out / 'fake_cli.py'
        fake.write_text('import sys,time\np=sys.stdin.read()\n'
                        'if p=="limit":\n time.sleep(.2)\n print(\'{"type":"error","message":"HTTP 429 rate limit"}\',flush=True)\n'
                        'time.sleep(20)\n', encoding='utf-8')
        results = []
        transport.STOP.clear()
        transport.STOP_REASON = None
        with patch.object(transport, 'OUT', self.out), patch.object(transport, 'command', return_value=[sys.executable, str(fake)]):
            threads = [threading.Thread(target=lambda p=p: results.append(transport.invoke(p, self.out/p, transport.base.SCHEMA, ['a']))) for p in ('sibling', 'limit')]
            started = time.monotonic()
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            self.assertTrue(all(not t.is_alive() for t in threads))
            self.assertLess(time.monotonic() - started, 5)
            self.assertEqual(transport.STOP_REASON, 'usage_or_rate_limit')
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r['usage'] is None for r in results))
            self.assertFalse(transport.ACTIVE)
        transport.STOP.clear()
        transport.STOP_REASON = None


if __name__ == '__main__':
    unittest.main()
