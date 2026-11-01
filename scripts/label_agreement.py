"""Resumable blind pilot/full evaluation using existing-login Codex CLI.

python -B scripts/label_agreement.py pilot
python -B scripts/label_agreement.py full
python -B scripts/label_agreement.py status
python -B scripts/label_agreement.py analyze

No training or labels written back to the dataset. All outputs are gitignored.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

from codex_labeler import OUT, REPO, VERDICTS, CODEX, SCHEMA, INSTRUCTIONS, invoke, prompt_for

# Private profile markers (candidate name, own project names) that must never reach a labeller.
# '|'-separated in TRIAGE_PRIVATE_MARKERS so they stay out of the repository.
PRIVATE_MARKERS = '|'.join(re.escape(m) for m in os.environ.get('TRIAGE_PRIVATE_MARKERS', '').split('|') if m) or r'(?!x)x'
CONTAMINATION = (r'(?i)Native screening notes|' + PRIVATE_MARKERS +
                 r'|screening verdict|fit note|reasonable stretch|candidate evidence|reviewer verdict|fit assessment')

DATE = '2026-10-24'
RUNS = OUT / 'runs' / f'agreement-{DATE}.jsonl'
MANIFEST = OUT / 'evaluation.json'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def jsonl(path):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'{path}:{number}: corrupt append-only log; refusing silent loss') from exc
    return rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_postings(postings):
    """Strip collector-appended reviewer notes, independent of receipt class."""
    cleaned, changes = {}, []
    for pid, posting in postings.items():
        original = posting['description']
        match = re.search(r'(?im)^\s*Native screening notes:\s*$', original)
        description = original[:match.start()].rstrip() if match else original
        cleaned[pid] = {**posting, 'description': description}
        if description != original:
            changes.append({'posting_id': pid, 'removed_characters': len(original) - len(description),
                            'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
                            'cleaned_sha256': hashlib.sha256(description.encode()).hexdigest()})
    return cleaned, changes


def audit_blinding(manifest, postings, changes):
    ids = {e['posting_id'] for e in manifest['entries']}
    context = (OUT / 'context.md').read_text(encoding='utf-8')
    for entry in manifest['entries']:
        reason = entry['receipt']['reason']
        description = postings[entry['posting_id']]['description'][:12000]
        if len(reason) > 60 and reason in description:
            raise RuntimeError(f'Receipt reason remains in blind text: {entry["posting_id"]}')
        if re.search(CONTAMINATION, description):
            raise RuntimeError(f'Possible reviewer contamination requires inspection: {entry["posting_id"]}')
    # An amendment is safe only if every already-accepted model input is identical.
    for row in successful(jsonl(RUNS)).values():
        prompt_hash = hashlib.sha256(prompt_for(context, postings[row['posting_id']]).encode()).hexdigest()
        if row['prompt_sha256'] != prompt_hash:
            raise RuntimeError('Existing result saw different input; cannot reuse contaminated/mixed-protocol votes.')
    record = {'method': 'Remove suffix from anchored Native screening notes: heading before description truncation; independent of verdict.',
              'eligible_changes': sorted((c for c in changes if c['posting_id'] in ids), key=lambda x: x['posting_id']),
              'scope': 'Labeler inputs only. Raw and processed dataset files unchanged.',
              'prior_results': 'All prior successful prompt hashes verified identical; none of the four amended postings was in pilot or completed before correction.'}
    path = OUT / 'blinding-cleanup.json'
    if path.exists():
        if read_json(path) != record:
            raise RuntimeError('Blinding cleanup changed since frozen audit.')
    else:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2)


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def append(row):
    RUNS.parent.mkdir(exist_ok=True)
    with RUNS.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def writer_lock():
    # OS-owned lock released on crash; prevents two independent resume writers.
    lock_path = OUT / 'runner.lock'
    with lock_path.open('a+b') as handle:
        if lock_path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare():
    OUT.mkdir(exist_ok=True)
    paths = [REPO / 'data/processed' / n for n in ['postings.jsonl', 'labels_agent.jsonl', 'splits.json']]
    hashes = {p.name: digest(p) for p in paths}
    hashes['context.md'] = digest(OUT / 'context.md')
    hashes['labeler_protocol'] = hashlib.sha256((INSTRUCTIONS + json.dumps(SCHEMA, sort_keys=True) + 'gpt-6-astra/low/12000').encode()).hexdigest()
    hashes['codex_labeler.py'] = digest(REPO / 'scripts/codex_labeler.py')
    postings, cleaning = clean_postings({p['posting_id']: p for p in jsonl(paths[0])})
    if MANIFEST.exists():
        manifest = read_json(MANIFEST)
        if manifest['hashes'] != hashes:
            raise RuntimeError('Input/protocol changed; refusing to mix experiments in this run log.')
        audit_blinding(manifest, postings, cleaning)
        return manifest, postings
    latest = {}
    for row in sorted(jsonl(paths[1]), key=lambda r: (r['reviewed_at'], r['receipt_fingerprint'])):
        if row['posting_id']:
            latest[row['posting_id']] = row
    splits = read_json(paths[2])
    entries = []
    for split in ['train', 'val', 'test']:
        for pid in splits[split]:
            if pid in latest and postings[pid]['description'].strip():
                entries.append({'posting_id': pid, 'split': split, 'receipt': latest[pid]})
    eligible = {pid for pid in latest if pid in postings and postings[pid]['description'].strip()}
    assert len(entries) == len({e['posting_id'] for e in entries}) == len(eligible)
    assert {e['posting_id'] for e in entries} == eligible
    rng = random.Random(20261024)
    pilot = []
    strata = dict(zip(VERDICTS, [8, 8, 7, 7]))
    for verdict, count in strata.items():
        pool = sorted(e['posting_id'] for e in entries if e['split'] == 'train' and e['receipt']['verdict'] == verdict)
        pilot.extend(rng.sample(pool, count))
    rng.shuffle(pilot)
    manifest = {'created_at': now(), 'hashes': hashes, 'model': 'gpt-6-astra',
                'reasoning_effort': 'low', 'codex_binary': str(CODEX),
                'seed': 20261024, 'pilot_ids': pilot, 'pilot_strata': strata,
                'pilot_reused_as_full_run': 1, 'description_limit': 12000,
                'entries': entries, 'expected_calls': len(entries) * 3,
                'posted_field': 'Not retained by processed postings; null unless present. No dates inferred from receipt timestamps.',
                'description_truncated_count': sum(len(postings[e['posting_id']]['description']) > 12000 for e in entries)}
    with MANIFEST.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    (OUT / 'output-schema.json').write_text(json.dumps(SCHEMA, indent=2), encoding='utf-8')
    audit_blinding(manifest, postings, cleaning)
    return manifest, postings


def successful(rows):
    done = {}
    for row in rows:
        if row['status'] == 'ok':
            key = (row['posting_id'], row['run'])
            if key in done:
                raise RuntimeError(f'Duplicate successful key: {key}')
            done[key] = row
    return done


def classify(pid, run, phase, posting, context, attempt):
    artifact_dir = OUT / 'calls' / f'{pid}-r{run}-a{attempt}-{time.time_ns()}'
    result = invoke(prompt_for(context, posting), artifact_dir)
    return {'posting_id': pid, 'run': run, 'attempt': attempt, 'phase': phase,
            'finished_at': now(), **result}


def timing(rows):
    values = sorted(r['seconds'] for r in rows)
    if not values:
        return {}
    return {'calls': len(values), 'mean_seconds': statistics.mean(values),
            'median_seconds': statistics.median(values), 'min_seconds': values[0],
            'p95_seconds': values[math.ceil(.95 * len(values)) - 1], 'max_seconds': values[-1]}


def metrics(pairs):
    # A three-way tie is an abstention: incorrect in primary accuracy and recall.
    # Kappa uses that zero-reference fifth category; the displayed matrix remains 4x4.
    counts = Counter(pairs)
    n = len(pairs)
    truth = Counter(a for a, b in pairs)
    predicted = Counter(b for a, b in pairs)
    correct = sum(a == b for a, b in pairs)
    observed = correct / n if n else None
    expected = sum(truth[v] * predicted[v] for v in VERDICTS) / n**2 if n else None
    return {'n': n, 'correct': correct, 'accuracy': observed,
            'kappa': (observed - expected) / (1 - expected) if n and expected < 1 else None,
            'always_reject_accuracy': truth['reject'] / n if n else None,
            'confusion': [[counts[a, b] for b in VERDICTS] for a in VERDICTS],
            'ties_by_class': {v: counts[v, None] for v in VERDICTS},
            'classes': {v: {'support': truth[v], 'predicted': predicted[v],
                            'precision': counts[v, v] / predicted[v] if predicted[v] else None,
                            'recall': counts[v, v] / truth[v] if truth[v] else None} for v in VERDICTS}}


def pct(value):
    return 'N/A' if value is None else f'{value:.1%}'


def analyze(manifest=None, postings=None):
    if manifest is None:
        manifest, postings = prepare()
    rows = jsonl(RUNS)
    done = successful(rows)
    entries = {e['posting_id']: e for e in manifest['entries']}
    complete, unanimous, split_votes = [], [], []
    by_split = defaultdict(list)
    examples = []
    for pid, entry in entries.items():
        votes = [done[pid, r]['output'] for r in [1, 2, 3] if (pid, r) in done]
        if len(votes) != 3:
            continue
        counter = Counter(v['verdict'] for v in votes)
        top, count = counter.most_common(1)[0]
        majority = top if count >= 2 else None
        pair = (entry['receipt']['verdict'], majority)
        complete.append(pair)
        (unanimous if count == 3 else split_votes).append(pair)
        by_split[entry['split']].append(pair)
        if pair[0] != pair[1]:
            examples.append({'posting_id': pid, 'title': postings[pid]['title'],
                             'employer': postings[pid]['employer'], 'split': entry['split'],
                             'receipt': entry['receipt'], 'votes': votes, 'majority': majority})
    pilot_rows = [r for r in rows if r['phase'] == 'pilot']
    pilot_pairs = [(entries[pid]['receipt']['verdict'], done[pid, 1]['output']['verdict'])
                   for pid in manifest['pilot_ids'] if (pid, 1) in done]
    pilot = metrics(pilot_pairs)
    pilot.update(timing=timing(pilot_rows), failures=sum(r['status'] != 'ok' for r in pilot_rows),
                 parse_failures=sum(r['status'] == 'parse_failure' for r in pilot_rows),
                 completed=len(pilot_pairs), target=30)
    summary = {'updated_at': now(), 'expected_postings': len(entries),
               'expected_calls': manifest['expected_calls'], 'successful_calls': len(done),
               'attempted_calls': len(rows), 'failures': sum(r['status'] != 'ok' for r in rows),
               'complete': len(complete) == len(entries), 'pilot': pilot,
               'majority': metrics(complete), 'unanimous': metrics(unanimous),
               'split_votes': metrics(split_votes),
               'fraction_unanimous': len(unanimous) / len(complete) if complete else None,
               'by_split': {s: metrics(by_split[s]) for s in ['train', 'val', 'test']},
               'all_eligible_reject_baseline': sum(e['receipt']['verdict'] == 'reject' for e in entries.values()) / len(entries),
               'timing': timing(rows)}
    write_json(OUT / 'agreement-summary.json', summary)
    gate = read_json(OUT / 'pilot-gate.json') if (OUT / 'pilot-gate.json').exists() else None
    m = summary['majority']
    lines = ['# Blind labeler agreement', '', f'Updated: {summary["updated_at"]}.', '',
             f'Status: {"COMPLETE" if summary["complete"] else "INCOMPLETE / INTERIM — not final evaluation results"}. '
             f'{len(done)}/{manifest["expected_calls"]} valid independent calls; '
             f'{m["n"]}/{len(entries)} postings have all three votes.', '',
             '## Mechanism and blinding', '',
             'Local Codex CLI, existing ChatGPT login, gpt-6-astra, low reasoning effort; no API keys. '
             'Fresh ephemeral process/session and empty repo-local working directory per call; '
             'read-only sandbox, approval never, no daemon, ignore user config/rules, zero project instruction bytes, '
             'skills injection/discovery, shell, browser, apps, plugins, hooks, memories and subagents disabled. '
             'No tools permitted by prompt; any observed tool item invalidates the result. '
             'Exact argv retained under calls/*/command.json; stdout/stderr retained beside it.', '',
             'Only context.md plus employer/title/location/posted/first 12,000 description characters are sent. '
             'Receipt labels/reasons, split, posting ID and other runs are never in prompts. '
             'Evaluation receipts are loaded only by the parent runner. '
             f'Context size: {len((OUT / "context.md").read_text(encoding="utf-8")):,} characters. '
             f'{manifest["description_truncated_count"]} eligible descriptions exceed the limit. '
             + manifest['posted_field'], '',
             'Blinding audit removed appended "Native screening notes:" sections from four descriptions before truncation. '
             'Two contained exact receipt reasons; two contained other reviewer judgments. None was in the pilot or '
             'sent before cleanup. All 46 existing successful prompt hashes were verified unchanged when the worker '
             'was restarted. Raw/processed files and the append-only results log were preserved. '
             'Four in-flight CLI processes were administratively stopped during this correction and have no '
             'completed result rows; they are separate from returned model/schema failures. '
             'See blinding-cleanup.json and worker-interruptions.json for the audit.', '',
             '## Pilot', '',
             'Fixed seed 20261024, train only: 8 strong, 8 stretch, 7 reject, 7 needs-information. '
             'This near-balanced sample is diagnostic, not a natural-prevalence accuracy estimate. '
             'Pilot successes count as run 1 of those postings; runs 2/3 are fresh and never see run 1.', '',
             f'Valid: {pilot["completed"]}/30; failures: {pilot["failures"]}; parse/schema failures: {pilot["parse_failures"]}. '
             f'Agreement: {pilot["correct"]}/{pilot["n"]} ({pct(pilot["accuracy"])}); '
             f'kappa: {pilot["kappa"] if pilot["kappa"] is not None else "N/A"}; '
             f'always-reject: {pct(pilot["always_reject_accuracy"])}.', '',
             'Timing: ' + json.dumps(pilot['timing']) + '.', '',
             'Gate: ' + (json.dumps(gate) if gate else 'Pilot/gate not yet complete.') + '', '',
             '## Three-vote agreement', '',
             f'Majority-vote accuracy: {m["correct"]}/{m["n"]} ({pct(m["accuracy"])}). '
             f'Cohen kappa: {m["kappa"] if m["kappa"] is not None else "N/A"}. '
             f'Always-reject on completed cohort: {pct(m["always_reject_accuracy"])}; '
             f'on all eligible postings: {pct(summary["all_eligible_reject_baseline"])}.', '',
             'Three distinct votes have no majority: treated as abstentions/incorrect in primary accuracy and class recall. '
             'Kappa includes abstention as a fifth prediction category with zero reference prevalence. '
             'The 4x4 matrix excludes abstentions, which are reported separately. '
             'Missing/failed runs are excluded from completed-cohort metrics and shown in coverage above.', '',
             '| Receipt \\ predicted | strong | stretch | reject | needs-information |',
             '|---|---:|---:|---:|---:|']
    for v, values in zip(VERDICTS, m['confusion']):
        lines.append('| ' + v + ' | ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', 'No-majority ties by receipt class: ' + json.dumps(m['ties_by_class']) + '.', '',
              '| Class | Precision | Recall | Support |', '|---|---:|---:|---:|']
    for v, stats in m['classes'].items():
        lines.append(f'| {v} | {pct(stats["precision"])} | {pct(stats["recall"])} | {stats["support"]} |')
    lines += ['', '## Self-consistency', '',
              f'Unanimous fraction: {pct(summary["fraction_unanimous"])}. '
              f'Agreement when unanimous: {pct(summary["unanimous"]["accuracy"])} (n={len(unanimous)}). '
              f'Agreement when split: {pct(summary["split_votes"]["accuracy"])} (n={len(split_votes)}). '
              'Vote spread is only a candidate confidence signal; assess these conditional rates and their sample sizes. '
              'Independent sessions do not guarantee statistically independent model errors.', '',
              '## Agreement by split', '', '| Split | Complete postings | Majority accuracy | Kappa |', '|---|---:|---:|---:|']
    for s, stats in summary['by_split'].items():
        k = 'N/A' if stats['kappa'] is None else f'{stats["kappa"]:.4f}'
        lines.append(f'| {s} | {stats["n"]} | {pct(stats["accuracy"])} | {k} |')
    lines += ['', '## Ten example disagreements', '',
              'Deterministic first ten by frozen evaluation order; not cherry-picked for a preferred conclusion.']
    for ex in examples[:10]:
        lines += ['', f'### {ex["title"]} — {ex["employer"]}', '',
                  f'Posting: `{ex["posting_id"]}`; split: {ex["split"]}.', '',
                  f'Receipt **{ex["receipt"]["verdict"]}**: {ex["receipt"]["reason"]}', '']
        for i, vote in enumerate(ex['votes'], 1):
            lines.append(f'- Run {i}, **{vote["verdict"]}**: {vote["reason"]}')
    if not examples:
        lines += ['', 'Not available until disagreeing three-run postings complete.']
    lines += ['', '## Limits and safety', '',
              '- Agreement measures replication of existing agent judgments, not hiring truth or user preferences.',
              '- Latest receipt per posting uses the dataset builder ordering (reviewed_at, receipt_fingerprint). Current policy/context can postdate the receipt; some skill-only historical rejects may now be stretches.',
              '- Processed text is the longest selected description and may differ from reviewed text; navigation/form residue and missing fields remain possible. The blind labeler cannot reproduce live-site verification or private application-state knowledge absent from the posting.',
              '- Candidate evidence is compressed and not an exact historical snapshot. See CONTEXT-CUTS.md and context-build.json for sources, cuts, provenance and redaction.',
              '- Partial results can be biased by completion order. No full-run conclusion is warranted until complete.',
              '- Source files were opened only for read-only import. No source file was modified. No source scripts or private configuration were accessed. No commit was made.', '',
              '## Progress / resume', '',
              '`python -B scripts/label_agreement.py status`', '',
              '`python -B scripts/label_agreement.py full` resumes only missing posting_id/run keys after the passed pilot gate. '
              '`python -B scripts/label_agreement.py analyze` refreshes this report. '
              'The full worker refreshes the report periodically and on completion.']
    (OUT / 'AGREEMENT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    pilot_lines = ['# Pilot call timings', '',
                   'One call per fixed train-only sample. Full receipts are never sent to the labeler.', '',
                   '| Posting ID | Receipt | Labeler | Status | Wall seconds |',
                   '|---|---|---|---|---:|']
    for row in pilot_rows:
        pid = row['posting_id']
        pilot_lines.append(f'| {pid} | {entries[pid]["receipt"]["verdict"]} | '
                           f'{row.get("output", {}).get("verdict", "—")} | {row["status"]} | {row["seconds"]:.3f} |')
    (OUT / 'PILOT.md').write_text('\n'.join(pilot_lines) + '\n', encoding='utf-8')
    return summary


def run_phase(phase, manifest, postings):
    context = (OUT / 'context.md').read_text(encoding='utf-8')
    rows = jsonl(RUNS)
    done = successful(rows)
    if phase == 'pilot':
        work = [(pid, 1) for pid in manifest['pilot_ids']]
    else:
        gate = read_json(OUT / 'pilot-gate.json')
        if not gate['passed']:
            raise RuntimeError('Pilot stop criterion triggered; full run forbidden.')
        # Finish all three runs for each posting early so partial analyses have coverage.
        ids = [e['posting_id'] for e in manifest['entries']]
        random.Random(manifest['seed'] + 1).shuffle(ids)
        work = [(pid, run) for pid in ids for run in [1, 2, 3]]
    attempted_pilot = {(r['posting_id'], r['run']) for r in rows if r['phase'] == 'pilot'}
    work = [key for key in work if key not in done and not (phase == 'pilot' and key in attempted_pilot)]
    attempts = Counter((r['posting_id'], r['run']) for r in rows)
    concurrency = 4
    started = time.perf_counter()
    initial_count = len(work)
    consecutive_failed_batches = 0
    while work:
        batch, work = work[:concurrency], work[concurrency:]
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for pid, run in batch:
                attempts[pid, run] += 1
                futures.append(executor.submit(classify, pid, run, phase, postings[pid], context, attempts[pid, run]))
            for future in as_completed(futures):
                result = future.result()
                append(result)
                results.append(result)
                print(json.dumps({k: result[k] for k in ['posting_id', 'run', 'status', 'seconds', 'rate_limited']}), flush=True)
        limited = any(r['rate_limited'] for r in results)
        if limited:
            concurrency = max(1, concurrency // 2)
        failed_batch = all(r['status'] != 'ok' for r in results)
        consecutive_failed_batches = consecutive_failed_batches + 1 if failed_batch else 0
        if phase == 'full':
            for r in results:
                if r['status'] != 'ok' and r['attempt'] < 3:
                    work.append((r['posting_id'], r['run']))
        summary = analyze(manifest, postings)
        write_json(OUT / 'progress.json', {'phase': phase, 'state': 'running', 'pid': os.getpid(),
                   'updated_at': now(), 'successful_calls': summary['successful_calls'],
                   'expected_calls': manifest['expected_calls'], 'complete_postings': summary['majority']['n'],
                   'remaining_tasks_this_process': len(work), 'concurrency': concurrency,
                   'elapsed_seconds_this_process': round(time.perf_counter() - started, 1)})
        if consecutive_failed_batches >= 3:
            raise RuntimeError('Three consecutive batches failed; stopped to avoid repeated provider failures.')
        if limited:
            time.sleep(30)
    summary = analyze(manifest, postings)
    elapsed = time.perf_counter() - started
    if phase == 'pilot':
        pilot = summary['pilot']
        mean = pilot['timing'].get('mean_seconds', float('inf'))
        estimate = mean * manifest['expected_calls'] / 4 / 3600
        # Include observed startup/batch overhead when pilot ran in one uninterrupted process.
        wall_estimate = elapsed / 30 * manifest['expected_calls'] / 3600 if initial_count == 30 else None
        projected = max(estimate, wall_estimate or 0)
        parse_rate = pilot['parse_failures'] / 30
        gate = {'passed': parse_rate <= .10 and projected <= 6 and pilot['completed'] >= 27,
                'parse_failure_rate': parse_rate, 'projected_full_hours_at_4': projected,
                'mean_call_projection_hours': estimate, 'observed_wall_projection_hours': wall_estimate,
                'pilot_wall_seconds_this_process': elapsed,
                'criteria': 'Stop if parse/schema failure >10%, projected full run >6h, or <27 valid pilot outputs.',
                'created_at': now()}
        write_json(OUT / 'pilot-gate.json', gate)
        summary = analyze(manifest, postings)
    state = 'pilot_passed' if phase == 'pilot' and read_json(OUT / 'pilot-gate.json')['passed'] else 'pilot_stopped' if phase == 'pilot' else 'complete' if summary['complete'] else 'incomplete_failures'
    write_json(OUT / 'progress.json', {'phase': phase, 'state': state, 'pid': os.getpid(),
               'updated_at': now(), 'successful_calls': summary['successful_calls'],
               'expected_calls': manifest['expected_calls'], 'complete_postings': summary['majority']['n']})
    print(json.dumps({'state': state, 'pilot': summary['pilot'] if phase == 'pilot' else None,
                      'majority': summary['majority']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pilot', 'full', 'status', 'analyze'])
    args = parser.parse_args()
    if args.action == 'status':
        for name in ['progress.json', 'pilot-gate.json']:
            if (OUT / name).exists():
                print(name + ': ' + json.dumps(read_json(OUT / name), indent=2))
        if RUNS.exists():
            rows = jsonl(RUNS)
            print(json.dumps({'log_attempts': len(rows), 'valid_unique_keys': len(successful(rows))}))
        return
    with writer_lock():
        manifest, postings = prepare()
        if args.action == 'analyze':
            print(json.dumps(analyze(manifest, postings), indent=2))
        else:
            try:
                run_phase(args.action, manifest, postings)
            except Exception as exc:
                write_json(OUT / 'progress.json', {'phase': args.action, 'state': 'stopped_error',
                           'pid': os.getpid(), 'updated_at': now(), 'error': str(exc)})
                analyze(manifest, postings)
                raise


if __name__ == '__main__':
    main()
