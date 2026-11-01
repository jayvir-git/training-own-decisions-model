"""Offline tests for bulk_final_v4 merge rules and codex_tiebreak_v4 (Codex is mocked; never started)."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bulk_final_v4 as bf
import codex_tiebreak_v4 as tb
import codex_labeler_v3 as transport
import label_opencode as lo

LAB = {'needs_verification': 'no', 'sponsorship_blocked': 'unknown', 'clearance_or_citizenship_required': 'no',
       'level': 'entry-I', 'track': 'software', 'reason': 'Test reason.', 'min_years': 0}


def lab(v, pid=None):
    return {**({'posting_id': pid} if pid else {}), 'verdict': v, **LAB}


PRIVATE_DATA = Path(__file__).resolve().parents[1] / 'data/labeling/v3/bulk-sample-1500.json'


@unittest.skipUnless(PRIVATE_DATA.exists(), 'needs the private labelling data under data/labeling/')
class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=lo.ROOT)
        out = Path(self.tmp.name)
        self.patches = [patch.object(lo, 'OUT', out), patch.object(tb, 'CALLS', out / 'tiebreak-calls.jsonl'),
                        patch.object(tb, 'DISPATCH', out / 'tiebreak-dispatch.jsonl'),
                        patch.object(tb, 'BUDGET_PATH', out / 'tiebreak-budget.json')]
        for p in self.patches:
            p.start()
        manifest = lo.read_json(lo.V3 / 'bulk-sample-1500.json')
        self.ids = [e['posting_id'] for e in manifest['entries']]
        a, b, c, d, e, f = self.ids[:6]
        # luna: a strong, b stretch, c stretch, d strong, e stretch(glm parse fail), f reject; others reject
        self.luna = {pid: lab('reject', pid) for pid in self.ids}
        self.luna.update({a: lab('strong', a), b: lab('stretch', b), c: lab('stretch', c), d: lab('strong', d), e: lab('stretch', e)})
        self.glm = {a: lab('strong', a), b: lab('stretch', b), c: lab('reject', c), d: lab('stretch', d)}
        self.glm_parse = Counter({(e,): 2})

        def fake_outputs(phase, model):
            if phase == 'bulk':
                return self.luna, Counter()
            if phase == 'bulk-glm':
                return self.glm, self.glm_parse
            return {}, Counter()
        self.patches.append(patch.object(lo, 'outputs', side_effect=fake_outputs))
        self.patches[-1].start()
        transport.STOP.clear()
        transport.STOP_REASON = None

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()
        transport.STOP.clear()
        transport.STOP_REASON = None


class MergeTests(Fixture):
    def test_rules_and_queue(self):
        a, b, c, d, e, f = self.ids[:6]
        counts = bf.build()['counts']
        rows = {r['posting_id']: r for r in lo.jsonl(lo.OUT / 'bulk-final.jsonl')}
        self.assertEqual(len(rows), 1500)
        self.assertEqual((rows[a]['verdict'], rows[a]['label_source']), ('strong', 'luna+glm agree'))
        self.assertEqual((rows[b]['verdict'], rows[b]['label_source']), ('stretch', 'luna+glm agree'))
        self.assertEqual((rows[c]['status'], rows[c]['pending_reason']), ('pending_tiebreak', 'glm_disagrees'))
        self.assertEqual(rows[c]['labels']['glm']['verdict'], 'reject')
        self.assertEqual(rows[e]['pending_reason'], 'glm_parse_failed')
        self.assertEqual((rows[f]['verdict'], rows[f]['label_source']), ('reject', 'luna (reject, not re-checked)'))
        queue = lo.read_json(lo.OUT / 'tiebreak-queue.json')
        self.assertEqual(queue['ids'][0], d)  # strong-involving disagreement first
        self.assertEqual(set(queue['ids']), {c, d, e})
        self.assertEqual(counts['pairs_luna_to_glm'], {'stretch->reject': 1, 'stretch->stretch': 1, 'strong->stretch': 1, 'strong->strong': 1})
        self.assertEqual((counts['agree'], counts['disagree'], counts['glm_missing']), (2, 2, 1))
        self.assertEqual(counts['raw_totals']['pending_tiebreak'], 3)
        self.assertAlmostEqual(sum(counts['weighted_share'].values()), 1.0)


class TiebreakTests(Fixture):
    def fake_invoke(self, replies):
        calls = []

        def invoke(prompt, artifact, schema, ids, batch=False):
            calls.append(ids[0])
            reply = replies.pop(0)
            if reply.get('status') == 'failure' and 'limit' in reply.get('error', ''):
                transport.stop('usage_or_rate_limit')
            return reply
        return invoke, calls

    def run_bulk(self, replies, limit=1_000_000):
        bf.build()
        _, posts = lo.bulk_inputs()
        invoke, calls = self.fake_invoke(replies)
        data = tb.budget()
        with patch.object(transport, 'invoke', side_effect=invoke):
            state = tb.run_target('bulk', limit, posts, lo.context(), data, False)
        return state, calls, data

    def ok(self, v, tokens=13000):
        return {'status': 'ok', 'usage': {'input_tokens': tokens, 'cached_input_tokens': 0, 'output_tokens': 100}, 'output': lab(v)}

    def test_resolves_and_rebuilds(self):
        a, b, c, d, e, f = self.ids[:6]
        state, calls, data = self.run_bulk([self.ok('stretch'), self.ok('reject'), self.ok('stretch')])
        self.assertEqual(state, 'complete')
        self.assertEqual(calls, [d, c, e])
        self.assertEqual(data['bulk']['input_tokens'], 39000)
        bf.build()
        rows = {r['posting_id']: r for r in lo.jsonl(lo.OUT / 'bulk-final.jsonl')}
        self.assertEqual((rows[d]['verdict'], rows[d]['label_source']), ('stretch', 'astra tiebreak'))
        self.assertEqual(rows[c]['verdict'], 'reject')
        self.assertEqual(lo.read_json(lo.OUT / 'tiebreak-queue.json')['count'], 0)
        state, calls, _ = self.run_bulk([])  # resumable: nothing re-sent
        self.assertEqual((state, calls), ('complete', []))

    def test_stops_on_first_usage_limit(self):
        state, calls, data = self.run_bulk([{'status': 'failure', 'usage': None, 'error': 'usage limit'}, self.ok('strong')])
        self.assertEqual(state, 'usage_or_rate_limit')
        self.assertEqual(len(calls), 1)
        self.assertEqual(data['bulk']['input_tokens'], tb.RESERVE)  # unknown usage charged

    def test_hard_budget(self):
        state, calls, _ = self.run_bulk([self.ok('stretch', 13000)] * 3, limit=tb.RESERVE + 12999)
        self.assertEqual((state, len(calls)), ('token_budget_reached', 1))

    def test_orphan_dispatch_charged_once(self):
        lo.append({'call_id': 'x', 'target': 'bulk', 'ids': ['p']}, 'tiebreak-dispatch.jsonl')
        data = tb.budget()
        tb.settle_orphans(data)
        tb.settle_orphans(data)
        self.assertEqual(data['bulk']['input_tokens'], tb.RESERVE)


if __name__ == '__main__':
    unittest.main()
