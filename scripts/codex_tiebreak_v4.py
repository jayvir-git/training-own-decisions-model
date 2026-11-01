"""Codex gpt-6-astra tie-break for v4 (identical v3 labeller setup). PREPARED, NOT YET RUN.

Targets (choose one or both; each has its own hard input-token budget):
  --bulk  ids in data/labeling/v4/tiebreak-queue.json (luna vs glm-5.3 disagreements), queue order
  --eval  the 112 evaluation postings labelled by gpt-6-luna in v4 (astra stopped at 590/702 in v3)

  python -B scripts/codex_tiebreak_v4.py --bulk --dry-run
  python -B scripts/codex_tiebreak_v4.py --bulk --bulk-budget 1000000
  python -B scripts/codex_tiebreak_v4.py --bulk --eval --bulk-budget 1000000 --eval-budget 1600000

Same transport as v3 (codex_labeler_v3.invoke: gpt-6-astra, medium effort, v2 INSTRUCTIONS
as developer instructions, --output-schema, isolated empty cwd, no tools) and the same v3 single
prompt (label_agreement_v3.request). Only its artifact/log directory is redirected to v4.
One sequential call per posting, no retries. Before each call RESERVE input tokens must fit
in the target's remaining budget (v3 max observed 15,776). Calls with unknown usage are charged
RESERVE (fail closed). Stops on the first usage/rate limit or any failure. Resumable: postings
with a recorded ok/parse_failure result are skipped. Afterwards bulk-final.jsonl is rebuilt
(pending tie-breaks -> astra's verdict) and evaluation results are written for the eval target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import codex_labeler_v2 as base
import codex_labeler_v3 as transport
import label_opencode as lo
from label_agreement import jsonl, now, read_json

RESERVE = 20_000
DEFAULT_BUDGET = {'bulk': 1_000_000, 'eval': 1_600_000}
BUDGET_PATH = lo.OUT / 'tiebreak-budget.json'
CALLS = lo.OUT / 'tiebreak-calls.jsonl'
DISPATCH = lo.OUT / 'tiebreak-dispatch.jsonl'


def targets_ids(target):
    if target == 'bulk':
        return read_json(lo.OUT / 'tiebreak-queue.json')['ids']
    luna, _ = lo.outputs('evaluation', 'gpt-6-luna')
    astra = lo.astra_labels()
    return sorted(pid for pid in luna if pid not in astra)


def budget():
    data = read_json(BUDGET_PATH) if BUDGET_PATH.exists() else {}
    for t in DEFAULT_BUDGET:
        data.setdefault(t, {'input_tokens': 0, 'calls': 0, 'charged_unknown': 0})
    return data


def settle_orphans(data):
    """A dispatch without a recorded result had unknown spend: charge RESERVE once, allow retry."""
    recorded = {r['call_id'] for r in jsonl(CALLS)}
    charged = set(data.get('orphans_charged', []))
    for d in jsonl(DISPATCH):
        if d['call_id'] not in recorded and d['call_id'] not in charged:
            data[d['target']]['input_tokens'] += RESERVE
            data[d['target']]['charged_unknown'] += 1
            charged.add(d['call_id'])
    data['orphans_charged'] = sorted(charged)


def done_ids(target):
    return {r['ids'][0] for r in jsonl(CALLS) if r.get('target') == target and r['status'] in ('ok', 'parse_failure')}


def run_target(target, limit, posts, ctx, data, dry_run, skip=()):
    ids = [p for p in targets_ids(target) if p not in done_ids(target) and p not in skip]
    used = data[target]['input_tokens']
    plan = {'target': target, 'remaining_postings': len(ids), 'budget': limit, 'used': used,
            'guaranteed_calls_at_20k_worst_case': max(0, (limit - used) // RESERVE) if limit > used else 0,
            'expected_tokens_at_v3_mean_13347': 13347 * len(ids)}
    if dry_run:
        print(json.dumps(plan, indent=2))
        return 'dry_run'
    for pid in ids:
        if data[target]['input_tokens'] + RESERVE > limit:
            return 'token_budget_reached'
        if transport.STOP.is_set():
            return transport.STOP_REASON or 'stopped'
        prompt, schema = lo.request([pid], posts, ctx, False)
        key = f'tiebreak-{target}-{time.time_ns()}'
        artifact = lo.OUT / 'tiebreak-calls' / key
        meta = {'call_id': key, 'target': target, 'ids': [pid], 'started_at': now(),
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'model': 'gpt-6-astra', 'effort': 'medium', 'artifact': str(artifact.relative_to(lo.ROOT))}
        lo.append(meta, 'tiebreak-dispatch.jsonl')
        try:
            result = transport.invoke(prompt, artifact, schema, [pid], False)
        except Exception as exc:  # never retry; unknown usage is charged below
            transport.stop('transport_exception')
            result = {'status': 'failure', 'usage': None, 'error': repr(exc)[:500]}
        row = {**meta, **result, 'finished_at': now()}
        lo.append(row, 'tiebreak-calls.jsonl')
        usage = result.get('usage')
        data[target]['input_tokens'] += usage['input_tokens'] if usage else RESERVE
        data[target]['charged_unknown'] += 0 if usage else 1
        data[target]['calls'] += 1
        data['updated_at'] = now()
        lo.save(BUDGET_PATH, data)
        if result['status'] == 'failure':
            return transport.STOP_REASON or 'call_failure'
    return 'complete'


def eval_report():
    """Write astra tie-break labels for the 112 eval postings and luna-vs-astra agreement."""
    import label_agreement_v2 as v2
    entries, _ = lo.evaluation()
    index = {e['posting_id']: e for e in entries}
    luna, _ = lo.outputs('evaluation', 'gpt-6-luna')
    astra = {r['ids'][0]: r['output'] for r in jsonl(CALLS) if r.get('target') == 'eval' and r['status'] == 'ok'}
    both = sorted(set(luna) & set(astra))
    result = {'astra_labelled': len(astra), 'of': len(targets_ids('eval')),
              'luna_vs_astra': v2.metrics([(astra[p]['verdict'], luna[p]['verdict']) for p in both]),
              'astra_vs_receipt': v2.metrics([(index[p]['receipt']['verdict'], astra[p]['verdict']) for p in both])}
    lo.save(lo.OUT / 'evaluation-astra-tiebreak.json', {'summary': result, 'labels': astra})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bulk', action='store_true', help='tie-break the bulk queue (tiebreak-queue.json)')
    parser.add_argument('--eval', action='store_true', help='re-label the 112 luna-labelled evaluation postings')
    parser.add_argument('--bulk-budget', type=int, default=DEFAULT_BUDGET['bulk'])
    parser.add_argument('--eval-budget', type=int, default=DEFAULT_BUDGET['eval'])
    parser.add_argument('--codex', help='override codex.exe path (default: codex_labeler_v2.CODEX)')
    parser.add_argument('--dry-run', action='store_true', help='print the plan; never starts Codex')
    parser.add_argument('--call-timeout', type=int, default=180, help='seconds before a silent Codex call is killed (v3 default 180)')
    parser.add_argument('--skip', action='append', default=[], help='posting id to leave unresolved (e.g. one that repeatedly hangs Codex); repeatable')
    args = parser.parse_args()
    if not (args.bulk or args.eval):
        parser.error('choose --bulk and/or --eval')
    if args.bulk_budget > DEFAULT_BUDGET['bulk'] or args.eval_budget > DEFAULT_BUDGET['eval']:
        parser.error('budgets may be lowered but not raised above 1.0M (bulk) / 1.6M (eval)')
    transport.CALL_TIMEOUT = args.call_timeout
    if args.codex:
        base.CODEX = Path(args.codex)
    if not args.dry_run and not Path(base.CODEX).exists():
        raise SystemExit(f'Codex binary not found: {base.CODEX} (pass --codex)')
    # Redirect only the v3 transport's artifact/log/empty-cwd directory into v4.
    transport.OUT = lo.OUT
    ctx = lo.context()
    with lo.lock():
        data = budget()
        settle_orphans(data)
        if not args.dry_run:
            lo.save(BUDGET_PATH, data)
        outcome = {}
        if args.bulk:
            _, posts = lo.bulk_inputs()
            outcome['bulk'] = run_target('bulk', args.bulk_budget, posts, ctx, data, args.dry_run, set(args.skip))
        if args.eval and not transport.STOP.is_set():
            _, posts = lo.evaluation()
            outcome['eval'] = run_target('eval', args.eval_budget, posts, ctx, data, args.dry_run)
        if args.dry_run:
            args_preview = base.command(lo.OUT / 'oc-cwd', lo.OUT / 'schema.json')  # builds a list only
            print('dry run: Codex was not started. Codex argv prefix:', json.dumps(args_preview[:14]))
            return
        import bulk_final_v4
        summary = bulk_final_v4.build()
        print(json.dumps({'outcome': outcome, 'budget': {t: data[t] for t in DEFAULT_BUDGET},
                          'bulk_final': summary['counts']}, indent=2))
        if args.eval:
            print(json.dumps(eval_report(), indent=2)[:3000])


if __name__ == '__main__':
    main()
