"""Offline tests for label_opencode: fake OpenCode server, no network, no model calls."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import codex_labeler_v2 as base
import label_opencode as lo

GOOD = {'verdict': 'strong', 'needs_verification': 'no', 'sponsorship_blocked': 'unknown',
        'clearance_or_citizenship_required': 'no', 'level': 'entry-I', 'track': 'software',
        'reason': 'Entry-level US software role.', 'min_years': 0}
METER_OK = {k: {'percent': 1, 'limited': False, 'resetsAt': 1} for k in ('rolling', 'weekly', 'monthly')}


class FakeServer:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def generate(self, model, prompt):
        self.sent.append((model, prompt))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def meter(self):
        return METER_OK


def ok(value=GOOD):
    return 200, {'data': {'text': json.dumps(value)}}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=lo.ROOT)
        self.patch = patch.object(lo, 'OUT', Path(self.tmp.name))
        self.patch.start()
        lo.GLOBAL_STOP.clear()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def run_one(self, replies, model='glm-5.3-flash', ids=('p1', 'p2'), ledger=None):
        srv = FakeServer(replies)
        posts = {pid: {'employer': 'E', 'title': 'SWE I', 'locations': ['US'], 'posted': '2026-10-01',
                       'description': 'Entry level.'} for pid in ids}
        state = lo.run_model(srv, model, 'bakeoff', [[p] for p in ids], posts, 'CTX', ledger or lo.Ledger(), workers=1)
        return state, srv


class TransportTests(Base):
    def test_pinned_provider_and_variant(self):
        sent = {}

        class S(lo.Server):
            def __init__(self):
                pass

            def request(self, method, path, body=None, timeout=None):
                sent.update(body=body, path=path)
                return 200, {}
        S().generate('minimax-m3', 'hi')
        self.assertEqual(sent['path'], '/api/experimental/generate/text')
        self.assertEqual(sent['body']['model'], {'id': 'minimax-m3', 'providerID': 'opencode-go', 'variant': 'none'})
        S().generate('kimi-k2.7-code', 'hi')
        self.assertNotIn('variant', sent['body']['model'])
        with self.assertRaises(AssertionError):
            S().generate('beam-free', 'hi')

    def test_compose_keeps_v3_prompt_and_appends_schema(self):
        full = lo.compose('PROMPT', base.SCHEMA)
        self.assertTrue(full.startswith(base.INSTRUCTIONS + '\n\nPROMPT'))
        self.assertTrue(full.endswith(json.dumps(base.SCHEMA)))
        self.assertEqual(lo.compose('P', {'type': 'array'}), base.INSTRUCTIONS + '\n\nP')

    def test_extract_variants(self):
        self.assertEqual(lo.extract('```json\n{"a": 1}\n```')[0], {'a': 1})
        self.assertEqual(lo.extract('<think>x</think>Here: {"a": 1} done'), ({'a': 1}, True))
        self.assertEqual(lo.extract('ok [{"a":1}]', batch=True)[0], [{'a': 1}])
        with self.assertRaises(ValueError):
            lo.extract('no json')

    def test_error_classification(self):
        self.assertEqual(lo.classify_error(502, {'_tag': 'ProviderRequestError', 'status': 429, 'message': 'x'})[0], 'limit')
        self.assertEqual(lo.classify_error(502, {'_tag': 'ProviderRequestError', 'message': 'Usage limit reached'})[0], 'limit')
        self.assertEqual(lo.classify_error(503, {'_tag': 'UnavailableError', 'message': 'Generation credentials are unavailable'})[0], 'auth')
        self.assertEqual(lo.classify_error(401, {'_tag': 'UnauthorizedError', 'message': 'Authentication required'})[0], 'server_auth')
        self.assertEqual(lo.classify_error(404, {'_tag': 'ModelNotFoundError', 'message': 'Model not found'})[0], 'bad_request')
        self.assertEqual(lo.classify_error(502, {'_tag': 'ProviderRequestError', 'status': 500, 'message': 'upstream'})[0], 'transport')

    def test_peak_hours(self):
        tue = dt.datetime(2026, 10, 27, tzinfo=dt.timezone.utc)
        self.assertTrue(lo.peak('deepseek-v4.1-flash', tue.replace(hour=2)))
        self.assertTrue(lo.peak('deepseek-v4.1-flash', tue.replace(hour=9, minute=59)))
        self.assertFalse(lo.peak('deepseek-v4.1-flash', tue.replace(hour=5)))
        self.assertFalse(lo.peak('deepseek-v4.1-flash', tue.replace(hour=15)))
        self.assertFalse(lo.peak('deepseek-v4.1-flash', dt.datetime(2026, 10, 31, 2, tzinfo=dt.timezone.utc)))
        self.assertFalse(lo.peak('glm-5.3-flash', tue.replace(hour=2)))

    def test_meter_problem(self):
        self.assertIsNone(lo.meter_problem(METER_OK))
        self.assertIsNone(lo.meter_problem(None) and None)
        self.assertEqual(lo.meter_problem(None), 'go_meter_unavailable')
        bad = json.loads(json.dumps(METER_OK))
        bad['weekly']['limited'] = True
        self.assertEqual(lo.meter_problem(bad), 'go_meter_weekly_limited')
        bad = json.loads(json.dumps(METER_OK))
        bad['rolling']['percent'] = 15
        self.assertTrue(lo.meter_problem(bad).startswith('go_meter_rolling'))


class RunnerTests(Base):
    def test_ok_resumable(self):
        state, srv = self.run_one([ok(), ok()])
        self.assertEqual(state, 'complete')
        self.assertEqual(len(lo.outputs('bakeoff', 'glm-5.3-flash')[0]), 2)
        state, srv = self.run_one([])  # nothing left; no calls made
        self.assertEqual((state, srv.sent), ('complete', []))

    def test_parse_failure_retried_once_then_final(self):
        bad = (200, {'data': {'text': 'not json'}})
        state, srv = self.run_one([bad, ok(), bad, bad])
        self.assertEqual(len(srv.sent), 4)
        labels, parse = lo.outputs('bakeoff', 'glm-5.3-flash')
        self.assertEqual(set(labels), {'p1'})
        self.assertEqual(parse[('p2',)], 2)
        state, srv = self.run_one([])  # p2 exhausted its retry; not re-sent
        self.assertEqual(srv.sent, [])

    def test_invalid_schema_is_parse_failure(self):
        state, srv = self.run_one([ok({**GOOD, 'verdict': 'maybe'}), ok(), ok()], ids=('p1', 'p2'))
        rows = lo.jsonl(lo.OUT / 'calls.jsonl')
        self.assertEqual([r['status'] for r in rows], ['parse_failure', 'ok', 'ok'])

    def test_limit_stops_model(self):
        state, srv = self.run_one([(502, {'_tag': 'ProviderRequestError', 'status': 429, 'message': 'rate limit'}), ok()])
        self.assertEqual(state, 'limit')
        self.assertEqual(len(srv.sent), 1)

    def test_auth_stops_everything(self):
        state, srv = self.run_one([(503, {'_tag': 'UnavailableError', 'message': 'Generation credentials are unavailable'})])
        self.assertEqual(state, 'auth')
        self.assertTrue(lo.GLOBAL_STOP.is_set())

    def test_consecutive_transport_failures(self):
        state, srv = self.run_one([TimeoutError('t')] * 3 + [ok()] * 2, ids=('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(state, 'consecutive_failures')
        self.assertEqual(len(srv.sent), 3)
        rows = lo.jsonl(lo.OUT / 'calls.jsonl')
        self.assertTrue(all(r['charged_usd'] > 0 for r in rows))  # unknown usage charged (fail closed)

    def test_budget_cap(self):
        ledger = lo.Ledger()
        ledger.rows.append({'model': 'gpt-6-luna', 'charged_usd': 0.4 * 60 - 1e-6,
                            'finished_at': lo.now()})
        state, srv = self.run_one([ok()], model='gpt-6-luna', ledger=ledger)
        self.assertEqual(state, 'model_budget_cap_40pct')
        self.assertEqual(srv.sent, [])

    def test_five_hour_cap(self):
        ledger = lo.Ledger()
        ledger.rows.append({'model': 'gpt-6-luna', 'charged_usd': 0.1 * 60, 'finished_at': lo.now()})
        self.assertEqual(ledger.reserve('k', 'gpt-6-luna', 0.01), 'five_hour_cap_10pct')
        ledger.rows[0]['finished_at'] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=6)).isoformat()
        self.assertIsNone(ledger.reserve('k', 'gpt-6-luna', 0.01))

    def test_deepseek_peak_refused(self):
        with patch.object(lo, 'peak', return_value=True):
            state, srv = self.run_one([ok()], model='deepseek-v4.1-flash')
        self.assertEqual((state, srv.sent), ('deepseek_peak_hours', []))

    def test_batch_validation(self):
        items = [{'posting_id': p, **GOOD} for p in ('a', 'b')]
        self.assertEqual(lo.validate_batch(items, ['a', 'b']), items)
        with self.assertRaises(ValueError):
            lo.validate_batch([items[0], items[0]], ['a', 'b'])

    def test_cost_estimate(self):
        u = lo.estimate_usage('glm-5.3-flash', 'x' * 35000, 'y' * 350)
        self.assertEqual((u['input_tokens'], u['visible_output_tokens'], u['output_tokens']), (10000, 100, 1600))
        self.assertAlmostEqual(lo.cost('glm-5.3-flash', u), (10000 * .15 + 1600 * .50) / 1e6)
        self.assertAlmostEqual(lo.cost('deepseek-v4.1-flash', u, True), 2 * lo.cost('deepseek-v4.1-flash', u))


@unittest.skipUnless((lo.ROOT / 'data/labeling/evaluation.json').exists(),
                     'needs the private labelling data under data/labeling/')
class ComparabilityTests(unittest.TestCase):
    def test_prompts_match_v3_call_log(self):
        """The v4 prompt for astra-labelled postings is byte-identical to what v3 sent."""
        astra = lo.astra_labels()
        entries, posts = lo.evaluation()
        ctx = lo.context()
        ids = sorted(astra)[:25]
        for pid in ids:
            prompt, schema = lo.request([pid], posts, ctx, False)
            self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), astra[pid]['_prompt_sha256'], pid)
            self.assertEqual(schema, base.SCHEMA)


if __name__ == '__main__':
    unittest.main()
