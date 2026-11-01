"""V2 blind evaluation. Standard library; all writes stay in this repository.

Tune only train/val, freeze a choice, then run three fresh votes on all 702.
No tuning vote is reused in the final experiment. V1 files are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import time

import codex_labeler_v2 as labeler
from label_agreement import CONTAMINATION, clean_postings, digest, jsonl, now, pct, read_json, successful, timing, write_json

REPO = labeler.REPO
BASE = REPO / 'data/labeling'
OUT = BASE / 'v2'
V1_RUNS = BASE / 'runs/agreement-2026-10-24.jsonl'
CLASSES = labeler.VERDICTS
SEED = 20261024


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def freeze(path, data):
    if path.exists():
        if read_json(path) != data:
            raise RuntimeError(f'Frozen experiment mismatch: {path}')
    else:
        with path.open('x', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return data


@contextmanager
def lock():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runner.lock').open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def majority(votes):
    counts = Counter(votes)
    top, n = counts.most_common(1)[0]
    return top if n > len(votes) / 2 else None


def metrics(pairs, weights=None):
    """NI references excluded; out-of-class predictions remain incorrect/abstain."""
    if weights is None:
        weights = [1] * len(pairs)
    scored = [(a, b, w) for (a, b), w in zip(pairs, weights) if a in CLASSES]
    counts, truth, pred = Counter(), Counter(), Counter()
    for a, b, w in scored:
        counts[a, b] += w
        truth[a] += w
        pred[b] += w
    total = sum(truth.values())
    correct = sum(counts[v, v] for v in CLASSES)
    accuracy = correct / total if total else None
    expected = sum(truth[v] * pred[v] for v in CLASSES) / total**2 if total else None
    return {'n': len(scored), 'weight': total, 'correct': correct, 'accuracy': accuracy,
            'kappa': (accuracy - expected) / (1 - expected) if total and expected < 1 else None,
            'always_reject_accuracy': truth['reject'] / total if total else None,
            'confusion': [[counts[a, b] for b in CLASSES] for a in CLASSES],
            'out_of_class_or_ties': {a: sum(w for (x, b), w in counts.items() if x == a and b not in CLASSES) for a in CLASSES},
            'classes': {v: {'support': truth[v], 'predicted': pred[v],
                            'precision': counts[v, v] / pred[v] if pred[v] else None,
                            'recall': counts[v, v] / truth[v] if truth[v] else None} for v in CLASSES}}


def binary_metrics(pairs):
    tp = sum(a and b for a, b in pairs)
    fp = sum(not a and b for a, b in pairs)
    fn = sum(a and not b for a, b in pairs)
    tn = sum(not a and not b for a, b in pairs)
    return {'n': len(pairs), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None}


def entries_for(final=False):
    # Filter immediately; tuning never uses test text, votes, reasons, or metrics.
    return [e for e in read_json(BASE / 'evaluation.json')['entries']
            if final or e['split'] in ('train', 'val')]


def v1_votes(ids):
    # Read the existing shared log, retain only explicitly allowed posting IDs.
    return successful([r for r in jsonl(V1_RUNS) if r['posting_id'] in ids])


def prepare_tuning():
    entries = entries_for()
    ids = {e['posting_id'] for e in entries}
    votes = v1_votes(ids)
    groups = {'disagreed': [], 'agreed': []}
    for e in entries:
        pid = e['posting_id']
        pred = majority([votes[pid, r]['output']['verdict'] for r in (1, 2, 3)])
        if pred == e['receipt']['verdict']:
            groups['agreed'].append(pid)
        elif e['receipt']['verdict'] in CLASSES:
            groups['disagreed'].append(pid)
    agreed_sample = random.Random(SEED).sample(sorted(groups['agreed']), 150)
    selected = sorted(groups['disagreed']) + agreed_sample
    index = {e['posting_id']: e for e in entries}
    strata = {}
    for name, pool in groups.items():
        sample = groups['disagreed'] if name == 'disagreed' else agreed_sample
        eligible_pool = [p for p in pool if index[p]['receipt']['verdict'] in CLASSES]
        eligible_sample = [p for p in sample if index[p]['receipt']['verdict'] in CLASSES]
        strata[name] = {'population': len(pool), 'sample': len(sample),
                        'verdict_population': len(eligible_pool), 'verdict_sample': len(eligible_sample),
                        'verdict_weight': len(eligible_pool) / len(eligible_sample)}
    data = {'seed': SEED, 'eligible_splits': ['train', 'val'], 'selection':
            'All v1-majority disagreements excluding NI receipts, plus 150 sampled v1 agreements (including NI if sampled).',
            'strata': strata, 'entries': [{**index[p], 'stratum': 'disagreed' if p in groups['disagreed'] else 'agreed'} for p in selected],
            'v1_manifest_sha256': digest(BASE / 'evaluation.json'), 'v1_log_sha256': digest(V1_RUNS)}
    return freeze(OUT / 'tuning-manifest.json', data)


def context_path(version):
    return OUT / f'context-{version}.md'


def protocol(version, entries):
    paths = [REPO / 'data/processed/postings.jsonl', BASE / 'evaluation.json',
             REPO / 'scripts/codex_labeler_v2.py', context_path(version)]
    data = {'model': 'gpt-6-astra', 'effort': 'medium', 'context': version,
            'hashes': {str(p.relative_to(REPO)): digest(p) for p in paths},
            'ids': [e['posting_id'] for e in entries], 'description_limit': 12000,
            'instructions': labeler.INSTRUCTIONS, 'schema': labeler.SCHEMA}
    return data


def posting_inputs(entries):
    ids = {e['posting_id'] for e in entries}
    raw = {p['posting_id']: p for p in jsonl(REPO / 'data/processed/postings.jsonl') if p['posting_id'] in ids}
    posts, _ = clean_postings(raw)
    for e in entries:
        text = posts[e['posting_id']]['description'][:12000]
        reason = e['receipt']['reason']
        if (len(reason) > 60 and reason in text) or re.search(
                CONTAMINATION, text):
            raise RuntimeError(f'Possible reviewer contamination: {e["posting_id"]}')
    return posts


def log_path(phase, version):
    return OUT / 'runs' / (f'tuning-{version}.jsonl' if phase == 'tune' else 'full.jsonl')


def summarize_tuning(version):
    manifest = prepare_tuning()
    entries = manifest['entries']
    rows = jsonl(log_path('tune', version))
    frozen_path = OUT / f'protocol-tuning-{version}.json'
    if frozen_path.exists():
        freeze(frozen_path, protocol(version, entries))
    elif rows:
        raise RuntimeError('Tuning results have no frozen protocol.')
    done = successful(rows)
    pairs, weights = [], []
    by_group = defaultdict(list)
    for e in entries:
        key = e['posting_id'], 1
        if key not in done:
            continue
        pair = e['receipt']['verdict'], done[key]['output']['verdict']
        pairs.append(pair)
        weights.append(manifest['strata'][e['stratum']]['verdict_weight'])
        by_group[e['stratum']].append(pair)
    sessions = [s for s in jsonl(OUT / 'sessions.jsonl') if s['phase'] == 'tune' and s['version'] == version and s['event'] == 'finished']
    elapsed = sum(s['seconds'] for s in sessions)
    n_attempted = sum(s['attempted'] for s in sessions)
    wall_projection = elapsed / n_attempted * 2106 / 3600 if n_attempted else None
    call_projection = timing(rows).get('mean_seconds', 0) * 2106 / 4 / 3600
    result = {'version': version, 'successful': len(done), 'target': len(entries),
              'complete': len(done) == len(entries), 'attempts': len(rows),
              'raw': metrics(pairs), 'reweighted': metrics(pairs, weights),
              'strata': manifest['strata'], 'by_stratum': {k: metrics(v) for k, v in by_group.items()},
              'timing': timing(rows), 'wall_seconds': elapsed,
              'projected_full_hours_at_4': max(call_projection, wall_projection or 0),
              'call_projection_hours': call_projection, 'wall_projection_hours': wall_projection}
    write_json(OUT / f'tuning-summary-{version}.json', result)
    return result


def classify(pid, run, phase, version, posting, context, attempt):
    artifact = OUT / 'calls' / f'{phase}-{version}-{pid}-r{run}-a{attempt}-{time.time_ns()}'
    result = labeler.invoke(labeler.prompt_for(context, posting), artifact)
    return {'posting_id': pid, 'run': run, 'phase': phase, 'version': version, 'attempt': attempt,
            'finished_at': now(), **result}


def run_phase(phase, version):
    if phase == 'tune':
        if (OUT / 'choice.json').exists():
            raise RuntimeError('Context already frozen for final evaluation; no more tuning allowed.')
        if version == 'revision':
            initial = summarize_tuning('initial')
            if not initial['complete'] or initial['reweighted']['accuracy'] >= .85:
                raise RuntimeError('Revision permitted only after complete initial tuning below 85%.')
            if not (BASE / 'CHANGES-v2.md').exists():
                raise RuntimeError('Document the revision first.')
        entries = prepare_tuning()['entries']
    else:
        choice = read_json(OUT / 'choice.json')
        version = choice['version']
        if digest(context_path(version)) != choice['context_sha256'] or digest(BASE / 'context-v2.md') != choice['context_sha256']:
            raise RuntimeError('Chosen context changed after freeze.')
        entries = entries_for(final=True)
        if len(entries) != 702:
            raise RuntimeError('Expected exactly 702 final postings.')
    posts = posting_inputs(entries)
    frozen = protocol(version, entries)
    freeze(OUT / (f'protocol-tuning-{version}.json' if phase == 'tune' else 'protocol-full.json'), frozen)
    context = context_path(version).read_text(encoding='utf-8')
    path = log_path(phase, version)
    rows = jsonl(path)
    done = successful(rows)
    for (pid, _), row in done.items():
        expected = hashlib.sha256(labeler.prompt_for(context, posts[pid]).encode()).hexdigest()
        if row['prompt_sha256'] != expected or row['version'] != version:
            raise RuntimeError('Cannot reuse a vote from different input/context.')
    ids = [e['posting_id'] for e in entries]
    random.Random(SEED + 2).shuffle(ids)
    runs = (1,) if phase == 'tune' else (1, 2, 3)
    attempts = Counter((r['posting_id'], r['run']) for r in rows)
    work = [(pid, r) for pid in ids for r in runs if (pid, r) not in done and attempts[pid, r] < 3]
    started, launched = time.perf_counter(), now()
    append(OUT / 'sessions.jsonl', {'event': 'started', 'phase': phase, 'version': version, 'at': launched, 'pid': os.getpid()})
    attempted, failures, concurrency = 0, 0, 4
    last_report = 0
    progress = {'phase': phase, 'version': version, 'state': 'running', 'pid': os.getpid(), 'started_at': launched,
                'target': len(entries) * len(runs), 'successful': len(done), 'concurrency': concurrency}
    write_json(OUT / 'progress.json', {**progress, 'updated_at': now()})
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {}
        while work or pending:
            while work and len(pending) < concurrency:
                pid, r = work.pop(0)
                attempts[pid, r] += 1
                future = pool.submit(classify, pid, r, phase, version, posts[pid], context, attempts[pid, r])
                pending[future] = (pid, r)
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                key = pending.pop(future)
                row = future.result()
                append(path, row)
                attempted += 1
                if row['status'] == 'ok':
                    done[key] = row
                    failures = 0
                else:
                    failures += 1
                    if row['attempt'] < 3:
                        work.append(key)
                print(json.dumps({k: row[k] for k in ('posting_id', 'run', 'status', 'seconds', 'rate_limited')}), flush=True)
                if row['rate_limited']:
                    concurrency = max(1, concurrency // 2)
                    time.sleep(30)
                if failures >= 12:
                    raise RuntimeError('12 consecutive failures; stop rather than repeat provider errors.')
            progress.update(successful=len(done), remaining=len(work) + len(pending), concurrency=concurrency,
                            elapsed_seconds=round(time.perf_counter() - started, 2), updated_at=now())
            write_json(OUT / 'progress.json', progress)
            if phase == 'full' and time.perf_counter() - last_report > 60:
                analyze_final(entries, posts)
                last_report = time.perf_counter()
    elapsed = time.perf_counter() - started
    append(OUT / 'sessions.jsonl', {'event': 'finished', 'phase': phase, 'version': version,
                                  'at': now(), 'seconds': elapsed, 'attempted': attempted, 'pid': os.getpid()})
    progress.update(state='complete' if len(done) == progress['target'] else 'incomplete_failures', updated_at=now(), elapsed_seconds=elapsed)
    write_json(OUT / 'progress.json', progress)
    result = summarize_tuning(version) if phase == 'tune' else analyze_final(entries, posts)
    print(json.dumps(result, indent=2), flush=True)


def choose():
    if (OUT / 'choice.json').exists():
        return read_json(OUT / 'choice.json')
    initial = summarize_tuning('initial')
    if not initial['complete']:
        raise RuntimeError('Complete initial tuning before choosing.')
    best = initial
    revision_path = log_path('tune', 'revision')
    revision = None
    if revision_path.exists():
        revision = summarize_tuning('revision')
        if initial['reweighted']['accuracy'] >= .85 or not revision['complete']:
            raise RuntimeError('Revision not permitted or incomplete.')
        if revision['reweighted']['accuracy'] > initial['reweighted']['accuracy']:
            best = revision
    version = best['version']
    (BASE / 'context-v2.md').write_bytes(context_path(version).read_bytes())
    data = {'chosen_at': now(), 'version': version, 'context_sha256': digest(context_path(version)),
            'initial_accuracy': initial['reweighted']['accuracy'],
            'revision_accuracy': revision['reweighted']['accuracy'] if revision else None,
            'revision_used': version == 'revision', 'revision_evaluated': revision is not None,
            'projected_full_hours_at_4': best['projected_full_hours_at_4'],
            'selection': 'Higher train/val natural-prevalence-reweighted single-run agreement; initial wins a tie.',
            'test_results_used_for_selection': False, 'full_votes': '2106 fresh calls; no tuning votes reused'}
    freeze(OUT / 'choice.json', data)
    return data


def cohort(entries, done):
    pairs, unanimous, split, verification, complete_ids = [], [], [], [], []
    by_split = defaultdict(list)
    unanimous_all = 0
    for e in entries:
        pid = e['posting_id']
        if any((pid, r) not in done for r in (1, 2, 3)):
            continue
        outputs = [done[pid, r]['output'] for r in (1, 2, 3)]
        votes = [o['verdict'] for o in outputs]
        pair = e['receipt']['verdict'], majority(votes)
        pairs.append(pair)
        is_unanimous = len(set(votes)) == 1
        unanimous_all += is_unanimous
        (unanimous if is_unanimous else split).append(pair)
        by_split[e['split']].append(pair)
        verification.append((e['receipt']['verdict'] == 'needs-information',
                             majority([o.get('needs_verification', 'yes' if o['verdict'] == 'needs-information' else 'no') for o in outputs]) == 'yes'))
        complete_ids.append(pid)
    n_eligible = metrics(pairs)['n']
    return {'complete_postings': len(pairs), 'complete_ids': complete_ids, 'majority': metrics(pairs),
            'by_split': {s: metrics(by_split[s]) for s in ('train', 'val', 'test')},
            'verification': binary_metrics(verification), 'unanimous': metrics(unanimous), 'split_votes': metrics(split),
            'unanimous_fraction_all': unanimous_all / len(pairs) if pairs else None,
            'unanimous_fraction_scored': metrics(unanimous)['n'] / n_eligible if n_eligible else None}


def analyze_final(entries=None, posts=None):
    if not (OUT / 'choice.json').exists():
        raise RuntimeError('Final analysis forbidden before context selection.')
    entries = entries if entries is not None else entries_for(final=True)
    posts = posts if posts is not None else posting_inputs(entries)
    rows = jsonl(log_path('full', ''))
    done = successful(rows)
    result = cohort(entries, done)
    ids = set(result['complete_ids'])
    v1 = cohort([e for e in entries if e['posting_id'] in ids], v1_votes(ids))
    sessions = [s for s in jsonl(OUT / 'sessions.jsonl') if s['phase'] == 'full']
    start = next((s['at'] for s in sessions if s['event'] == 'started'), None)
    latest = max((r['finished_at'] for r in rows), default=None)
    wall = (dt.datetime.fromisoformat(latest) - dt.datetime.fromisoformat(start)).total_seconds() if latest and start else None
    audit_path = OUT / 'rule-conflicts.json'
    audit = read_json(audit_path) if audit_path.exists() else {'status': 'pending', 'confirmed': [], 'candidates': []}
    result.update(updated_at=now(), successful_calls=len(done), attempted_calls=len(rows), expected_calls=2106,
                  complete=len(done) == 2106, timing=timing(rows), choice=read_json(OUT / 'choice.json'),
                  full_elapsed_wall_seconds=wall, v1_same_cohort=v1, rule_conflicts=audit)
    write_json(OUT / 'agreement-summary.json', result)
    report(result, entries, posts, done)
    return {k: result[k] for k in ('complete', 'successful_calls', 'complete_postings', 'majority', 'verification', 'full_elapsed_wall_seconds')}


def report(result, entries, posts, done):
    m = result['majority']
    c = result['choice']
    v1 = result['v1_same_cohort']
    tune = read_json(OUT / f'tuning-summary-{c["version"]}.json')
    lines = ['# Blind labeler agreement v2', '', f'Updated: {result["updated_at"]}', '',
             f'Status: {"COMPLETE" if result["complete"] else "INCOMPLETE / INTERIM — not final results"}. '
             f'{result["successful_calls"]}/2106 valid calls; {result["complete_postings"]}/702 complete postings.', '',
             '## Protocol and tuning', '',
             'gpt-6-astra, medium effort, bounded concurrency 4. Existing ChatGPT login; no API keys. '
             'Fresh empty repo-local cwd and ephemeral read-only Codex process per call, user config/rules ignored; '
             'tools, skills, hooks, memories, apps, plugins, subagents and project instructions disabled. '
             'Observed tool items invalidate output. Commands/events/stderr are under v2/calls/.', '',
             'Only candidate context and employer/title/location/posted/first 12,000 cleaned description characters enter prompts. '
             'Native screening notes are removed. Receipts, split, IDs and other votes never enter prompts. '
             'Missing live-site state cannot be reconstructed from posting text.', '',
             f'Tuning: {tune["successful"]}/{tune["target"]}, raw accuracy {pct(tune["raw"]["accuracy"])}; '
             f'natural-prevalence-reweighted estimate {pct(tune["reweighted"]["accuracy"])}. '
             f'Initial {pct(c["initial_accuracy"])}; revision {pct(c["revision_accuracy"])}; chosen {c["version"]}.', '',
             'All train/val v1-majority disagreements with non-NI receipts plus a seed-20261024 sample of 150 v1 agreements. '
             'NI receipts can occur in the agreed sample but are excluded from verdict scores. '
             'Each scored stratum is weighted by its full train/val non-NI population divided by its scored sample size. '
             'This is a stratified estimate, not accuracy on an untouched holdout. At most one revision was permitted below 85%; '
             'the choice was frozen before final/test analysis. All 2106 final calls are fresh, separate from tuning.', '',
             'Strata: `' + json.dumps(tune['strata']) + '`.', '',
             f'Mean tuning call {tune["timing"]["mean_seconds"]:.2f}s; median {tune["timing"]["median_seconds"]:.2f}s; '
             f'p95 {tune["timing"]["p95_seconds"]:.2f}s. Projected full duration {c["projected_full_hours_at_4"]:.2f}h '
             '(maximum of average-call/concurrency and observed throughput projections). '
             f'Actual elapsed wall time to latest completed call: {result["full_elapsed_wall_seconds"] / 3600:.2f}h.' if result['full_elapsed_wall_seconds'] is not None else 'Full timing pending.', '',
             '## Three-class majority agreement', '',
             f'Accuracy {m["correct"]}/{m["n"]} ({pct(m["accuracy"])}); kappa {m["kappa"]}; '
             f'always-reject baseline {pct(m["always_reject_accuracy"])} on the same non-NI subset.', '',
             'NI receipts are excluded, never relabeled. Three distinct votes are abstentions, counted incorrect in accuracy/recall '
             'and as a zero-reference category for kappa; the 3x3 matrix omits them. Incomplete postings are excluded.', '',
             '| Receipt \\ predicted | strong | stretch | reject |', '|---|---:|---:|---:|']
    for v, row in zip(CLASSES, m['confusion']):
        lines.append('| ' + v + ' | ' + ' | '.join(map(str, row)) + ' |')
    lines += ['', 'Abstentions: `' + json.dumps(m['out_of_class_or_ties']) + '`.', '',
              '| Class | Precision | Recall | Support |', '|---|---:|---:|---:|']
    for v, x in m['classes'].items():
        lines.append(f'| {v} | {pct(x["precision"])} | {pct(x["recall"])} | {x["support"]} |')
    lines += ['', '| Split | Non-NI complete | Accuracy | Kappa | Always reject |', '|---|---:|---:|---:|---:|']
    for s, x in result['by_split'].items():
        lines.append(f'| {s} | {x["n"]} | {pct(x["accuracy"])} | {x["kappa"]} | {pct(x["always_reject_accuracy"])} |')
    v = result['verification']
    lines += ['', '## Verification flag', '',
              f'Majority needs_verification=yes versus receipt needs-information: precision {pct(v["precision"])}; '
              f'recall {pct(v["recall"])}; TP={v["tp"]}, FP={v["fp"]}, FN={v["fn"]}, TN={v["tn"]}.', '',
              'This is a proxy comparison: receipt NI often records inaccessible pages or unconfirmed requisitions absent from the text. '
              'The new flag intentionally measures unresolved decisive text gates, so these are different constructs.', '',
              '## Self-consistency and v1 comparison', '',
              '| Metric (matched completed cohort) | v1 low | v2 medium |', '|---|---:|---:|']
    for title, a, b in [
        ('Three-class accuracy, NI receipts excluded', v1['majority']['accuracy'], m['accuracy']),
        ('Kappa, NI receipts excluded', v1['majority']['kappa'], m['kappa']),
        ('Always reject, same subset', v1['majority']['always_reject_accuracy'], m['always_reject_accuracy']),
        ('Unanimous fraction, all receipts', v1['unanimous_fraction_all'], result['unanimous_fraction_all']),
        ('Unanimous fraction, non-NI receipts', v1['unanimous_fraction_scored'], result['unanimous_fraction_scored']),
        ('Accuracy when unanimous, non-NI', v1['unanimous']['accuracy'], result['unanimous']['accuracy']),
        ('Accuracy when split, non-NI', v1['split_votes']['accuracy'], result['split_votes']['accuracy']),
        ('Verification proxy precision', v1['verification']['precision'], v['precision']),
        ('Verification proxy recall', v1['verification']['recall'], v['recall'])]:
        fmt = (lambda x: 'N/A' if x is None else f'{x:.4f}') if title.startswith('Kappa') else pct
        lines.append(f'| {title} | {fmt(a)} | {fmt(b)} |')
    lines += ['', f'Non-NI conditional sample sizes: v1 unanimous={v1["unanimous"]["n"]}, split={v1["split_votes"]["n"]}; '
              f'v2 unanimous={result["unanimous"]["n"]}, split={result["split_votes"]["n"]}.', '',
              'V1 needs-information predictions remain incorrect for non-NI verdict references; they are not remapped to stretch/reject. '
              'For the verification comparison only, v1 needs-information is the positive proxy. '
              'Original v1 four-class all-702 headline was 76.2%, kappa 0.5614, unanimity 93.9%; '
              'it has a different denominator and label space from the three-class table. '
              'Prompt, class space and effort changed together, so vote-spread differences cannot be attributed to effort alone.']
    u, s = result['unanimous']['accuracy'], result['split_votes']['accuracy']
    if u is not None and s is not None:
        lines += ['', f'V2 unanimous-minus-split accuracy gap: {u-s:.1%}; compare sample sizes before treating spread as a confidence signal.']
        old_fraction = v1['unanimous_fraction_scored']
        new_fraction = result['unanimous_fraction_scored']
        spread = 'more' if new_fraction < old_fraction else 'less' if new_fraction > old_fraction else 'the same amount of'
        separation = 'lower accuracy on split votes' if u > s else 'no lower accuracy on split votes'
        lines += ['', f'{"Final" if result["complete"] else "Interim"} confidence assessment: v2 has {spread} '
                  f'vote spread on the matched non-NI cohort and {separation}. '
                  f'The split-vote evidence is based on n={result["split_votes"]["n"]}; '
                  'this is descriptive evidence, not validation of vote spread as a calibrated confidence score '
                  'or a causal benefit from medium effort.']
    audit = result['rule_conflicts']
    lines += ['', '## Rule-conflict audit', '', f'Status: {audit["status"]}. Confirmed flagged receipts: {len(audit["confirmed"])}. '
              f'Additional unconfirmed candidates: {len(audit.get("candidates", []))}.', '',
              audit.get('method', 'Audit pending; zero is not a completed count.'), '',
              'Receipts are preserved. A conflict is with the new policy, not proof that the new model is correct.']
    for x in audit['confirmed'][:15]:
        lines.append(f'- `{x["posting_id"]}` ({x["split"]}, receipt {x["receipt_verdict"]}): {x["reason"]}')
    lines += ['', '## Ten deterministic disagreements', '',
              'First ten by SHA-256(seed + posting ID), among completed non-NI disagreements; not selected by content.']
    examples = []
    for e in entries:
        pid = e['posting_id']
        if e['receipt']['verdict'] not in CLASSES or any((pid, r) not in done for r in (1, 2, 3)):
            continue
        votes = [done[pid, r]['output'] for r in (1, 2, 3)]
        if majority([o['verdict'] for o in votes]) != e['receipt']['verdict']:
            examples.append((hashlib.sha256(f'{SEED}:{pid}'.encode()).hexdigest(), e, votes))
    for _, e, votes in sorted(examples)[:10]:
        pid = e['posting_id']
        lines += ['', f'### {posts[pid]["title"]} — {posts[pid]["employer"]}', '',
                  f'`{pid}`; {e["split"]}. Receipt **{e["receipt"]["verdict"]}**: {e["receipt"]["reason"]}', '']
        for i, v in enumerate(votes, 1):
            lines.append(f'- Run {i}: **{v["verdict"]}**, needs_verification={v["needs_verification"]}: {v["reason"]}')
    lines += ['', '## Safety, limitations and resumption', '',
              'Agreement measures replication of historical reviewer judgments. Policy dates, source text and candidate evidence may differ. '
              'Train/val were used in selection; final test is the held-out estimate, with only 70 postings before NI exclusion. '
              'Interim cohorts can be biased by completion order.', '',
              'No source file was modified or accessed in this task. '
              'No commit. V1 files remain unchanged.', '',
              '`python -B scripts/label_agreement_v2.py status`', '',
              '`powershell -NoProfile -File scripts/start_label_agreement_v2.ps1` resumes the detached full worker. '
              '`python -B scripts/label_agreement_v2.py analyze` refreshes this report. '
              'A single OS lock prevents concurrent writers; successful keys and prompt hashes are checked; '
              'logs append and fsync, while status/report snapshots refresh. Up to three attempts per key.']
    (BASE / 'AGREEMENT-v2.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'tune', 'choose', 'full', 'status', 'analyze'])
    parser.add_argument('--version', choices=['initial', 'revision'], default='initial')
    args = parser.parse_args()
    if args.action == 'status':
        for p in [OUT / 'progress.json', OUT / 'choice.json']:
            if p.exists():
                print(p.name + ': ' + json.dumps(read_json(p), indent=2))
        return
    with lock():
        if args.action == 'prepare':
            m = prepare_tuning()
            print(json.dumps({'target': len(m['entries']), 'strata': m['strata']}, indent=2))
        elif args.action == 'choose':
            print(json.dumps(choose(), indent=2))
        elif args.action == 'analyze':
            print(json.dumps(analyze_final(), indent=2))
        else:
            try:
                run_phase(args.action, args.version)
            except Exception as exc:
                write_json(OUT / 'progress.json', {'phase': args.action, 'state': 'stopped_error', 'pid': os.getpid(),
                                                 'updated_at': now(), 'error': str(exc)})
                raise


if __name__ == '__main__':
    main()
