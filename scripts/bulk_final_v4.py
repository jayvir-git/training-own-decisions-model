"""Merge v4 bulk labels into bulk-final.jsonl with per-row provenance.

Rules (user-approved 2026-10-28):
- Luna `reject` stays as-is                              -> label_source "luna (reject, not re-checked)".
- Luna strong/stretch re-labelled by glm-5.3 (phase bulk-glm):
    same verdict                                         -> label_source "luna+glm agree".
    different verdict, or no valid glm label             -> status "pending_tiebreak", both labels kept.
- A pending row with a Codex gpt-6-astra tie-break label (tiebreak-labels.jsonl)
                                                         -> astra's verdict, label_source "astra tiebreak".

  python -B scripts/bulk_final_v4.py      # rebuild bulk-final.jsonl, tiebreak-queue.json, summary
"""
from __future__ import annotations

from collections import Counter
import json

import label_opencode as lo
from label_agreement import jsonl, now, read_json

CLASSES = ['strong', 'stretch', 'reject']
GLM = 'glm-5.3'


def astra_tiebreaks(target='bulk'):
    """Latest ok Codex tie-break label per posting (written by codex_tiebreak_v4.py)."""
    result = {}
    for row in jsonl(lo.OUT / 'tiebreak-calls.jsonl'):
        if row.get('target') == target and row.get('status') == 'ok':
            result[row['ids'][0]] = row['output']
    return result


def queue_priority(row):
    # Disagreements touching `strong` first (rarest, most consequential), then stretch<->reject,
    # then missing glm labels; entry-title stratum before soft/uniform; posting id for determinism.
    pair = (row['labels']['luna']['verdict'], (row['labels']['glm'] or {}).get('verdict'))
    rank = 0 if 'strong' in pair else (1 if pair[1] is not None else 2)
    stratum = {'entry_title': 0, 'soft_entry': 1, 'uniform': 2}.get(row['stratum'], 3)
    return rank, stratum, row['posting_id']


def build():
    manifest = read_json(lo.V3 / 'bulk-sample-1500.json')
    luna, _ = lo.outputs('bulk', 'gpt-6-luna')
    glm, glm_parse = lo.outputs('bulk-glm', GLM)
    astra = astra_tiebreaks('bulk')
    glm_tried = {pid for pid in glm} | {ids[0] for ids in glm_parse}
    rows = []
    for e in manifest['entries']:
        pid = e['posting_id']
        l = luna[pid]
        g = glm.get(pid)
        base = {'posting_id': pid, **{k: e[k] for k in ('stratum', 'entry_signal', 'soft_entry_signal', 'inclusion_probability', 'weight')}}
        labels = {'luna': l, 'glm': g, 'astra': astra.get(pid)}
        if l['verdict'] == 'reject':
            row = {**base, 'status': 'final', 'verdict': 'reject', 'label_source': 'luna (reject, not re-checked)', 'final_label': l}
        elif g is not None and g['verdict'] == l['verdict']:
            row = {**base, 'status': 'final', 'verdict': l['verdict'], 'label_source': 'luna+glm agree', 'final_label': l}
        elif pid in astra:
            row = {**base, 'status': 'final', 'verdict': astra[pid]['verdict'], 'label_source': 'astra tiebreak', 'final_label': astra[pid]}
        else:
            why = 'glm_disagrees' if g is not None else ('glm_parse_failed' if pid in glm_tried else 'glm_not_run')
            row = {**base, 'status': 'pending_tiebreak', 'pending_reason': why, 'verdict': None,
                   'label_source': None, 'final_label': None}
        row['labels'] = labels
        row['labelers'] = {'luna': 'opencode-go/gpt-6-luna (low)', 'glm': f'opencode-go/{GLM} (low)',
                           'astra': 'codex gpt-6-astra (medium, v3 setup)'}
        rows.append(row)

    out = lo.OUT / 'bulk-final.jsonl'
    tmp = out.with_suffix('.jsonl.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    tmp.replace(out)
    pending = sorted((r for r in rows if r['status'] == 'pending_tiebreak'), key=queue_priority)
    queue = {'generated_at': now(), 'count': len(pending), 'order': 'strong-involving disagreements, stretch<->reject, missing glm; entry_title first',
             'ids': [r['posting_id'] for r in pending],
             'items': [{'posting_id': r['posting_id'], 'luna': r['labels']['luna']['verdict'],
                        'glm': (r['labels']['glm'] or {}).get('verdict'), 'reason': r['pending_reason'],
                        'stratum': r['stratum']} for r in pending]}
    lo.save(lo.OUT / 'tiebreak-queue.json', queue)
    summary = {'generated_at': now(), 'counts': counts(rows)}
    lo.save(lo.OUT / 'bulk-final-summary.json', summary)
    return summary


def counts(rows):
    rechecked = [r for r in rows if r['labels']['luna']['verdict'] in ('strong', 'stretch')]
    both = [r for r in rechecked if r['labels']['glm'] is not None]
    pairs = Counter(f"{r['labels']['luna']['verdict']}->{r['labels']['glm']['verdict']}" for r in both)
    status = Counter(r['status'] for r in rows)
    raw = Counter(r['verdict'] or 'pending_tiebreak' for r in rows)
    total_w = sum(r['weight'] for r in rows)
    weighted = {k: sum(r['weight'] for r in rows if (r['verdict'] or 'pending_tiebreak') == k) / total_w
                for k in [*CLASSES, 'pending_tiebreak']}
    by_source = Counter(r['label_source'] or 'pending_tiebreak' for r in rows)
    return {'rows': len(rows), 'luna_positive_rechecked': len(rechecked), 'glm_labelled': len(both),
            'agree': sum(p.split('->')[0] == p.split('->')[1] for p in pairs.elements()),
            'disagree': sum(p.split('->')[0] != p.split('->')[1] for p in pairs.elements()),
            'pairs_luna_to_glm': dict(sorted(pairs.items())), 'glm_missing': len(rechecked) - len(both),
            'status': dict(status), 'by_label_source': dict(by_source),
            'raw_totals': {k: raw.get(k, 0) for k in [*CLASSES, 'pending_tiebreak']},
            'weighted_share': weighted}


if __name__ == '__main__':
    print(json.dumps(build()['counts'], indent=2))
