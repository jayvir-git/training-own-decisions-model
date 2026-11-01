"""Frozen v3 evaluation, batch gate, and bulk pipeline. No automatic retries or timers."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import difflib
import hashlib
import json
import os
import random
import re
import time

import codex_labeler_v3 as transport
import label_agreement_v2 as v2
from label_agreement import CONTAMINATION, clean_postings, digest, jsonl, now, pct, read_json

ROOT = v2.REPO
BASE = v2.BASE
OUT = transport.OUT
SEED = 20261025
TOTAL = 18_000_000
DAILY = 8_000_000
RULE = "Years requirements only decide rejection; they never cap a posting below strong. A stated minimum of 0 (e.g. '0-2 years') is no years requirement. A stated minimum of 1-2 years never causes a reject; the posting can still be strong if the candidate's evidence clearly meets it, otherwise stretch when research + internships + projects plausibly add up. A stated minimum of 3+ years is a reject unless the posting says internships/projects/academic experience count. For ranges use the lower bound. 'Preferred' years are not a gate."


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def freeze_text(path, text):
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise RuntimeError(f'Frozen file differs: {path}')
    else:
        with path.open('x', encoding='utf-8', newline='\n') as f:
            f.write(text)


def day():
    # Windows local timezone is verified in preflight against America/New_York.
    return dt.datetime.now().astimezone().date().isoformat()


class Budget:
    def __init__(self):
        self.path = OUT / 'budget.json'
        self.data = read_json(self.path) if self.path.exists() else {
            'total_limit': TOTAL, 'daily_limit': DAILY, 'timezone': 'America/New_York',
            'input_tokens': 0, 'days': {}, 'reservations': {}, 'accounted_calls': []}
        assert self.data['total_limit'] == TOTAL and self.data['daily_limit'] == DAILY
        self.persist()

    def persist(self):
        self.data['updated_at'] = now()
        save(self.path, self.data)

    def reserve(self, key):
        d = day()
        reserved = sum(x['tokens'] for x in self.data['reservations'].values())
        daily_reserved = sum(x['tokens'] for x in self.data['reservations'].values() if d in x['days'])
        if self.data['input_tokens'] + reserved + transport.RESERVATION > TOTAL:
            return 'total_budget_cap'
        if self.data['days'].get(d, 0) + daily_reserved + transport.RESERVATION > DAILY:
            return 'daily_cap'
        # A call can cross midnight; reserve against both dates conservatively.
        tomorrow = (dt.date.fromisoformat(d) + dt.timedelta(days=1)).isoformat()
        self.data['reservations'][key] = {'tokens': transport.RESERVATION, 'days': [d, tomorrow]}
        self.persist()
        return None

    def settle(self, row):
        key = row['call_id']
        if key in self.data['accounted_calls']:
            return
        usage = row.get('usage')
        if usage is not None:
            n = usage['input_tokens']
            self.data['input_tokens'] += n
            d = row['day']
            self.data['days'][d] = self.data['days'].get(d, 0) + n
            self.data['reservations'].pop(key, None)
        self.data['accounted_calls'].append(key)
        self.persist()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    old = (BASE / 'v2/context-initial.md').read_text(encoding='utf-8')
    start = old.index('- Stated minimum experience <=2 years:')
    end = old.index('- Keep unknown sponsorship', start)
    # Preserve the degree-pathway guidance, which is separate from the changed rule.
    new = old[:start] + '- ' + RULE + '\n- Use the lowest required relevant minimum among explicitly\n  offered degree pathways the candidate meets. A missing tenure requirement is not invented.\n' + old[end:]
    new = new.replace('An applicable\n  <=2-year experience rule produces stretch. ', '')
    freeze_text(OUT / 'context-v3.md', new)
    diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                      fromfile='v2/context-initial.md', tofile='v3/context-v3.md'))
    freeze_text(OUT / 'CHANGES-v3.md', '# V3 approved years-rule change\n\nOnly the years-rule block and its duplicate forced-stretch sentence change. All other text, including historical v2 headings, is preserved. No prompt iteration after results.\n\n```diff\n' + diff + '```\n')
    entries = v2.entries_for(final=True)
    posts = v2.posting_inputs(entries)
    audit(entries, posts)
    # Freeze the full data/code protocol before launching any model call.
    paths = [OUT / 'context-v3.md', BASE / 'evaluation.json', ROOT / 'data/processed/postings.jsonl',
             ROOT / 'data/processed/splits.json', ROOT / 'scripts/codex_labeler_v3.py',
             ROOT / 'scripts/label_agreement_v3.py', ROOT / 'scripts/codex_labeler_v2.py']
    v2.freeze(OUT / 'protocol.json', {'seed': SEED, 'model': 'gpt-6-astra', 'effort': 'medium',
        'runs_per_posting': 1, 'concurrency': 4, 'total_limit': TOTAL, 'daily_limit': DAILY,
        'hashes': {str(p.relative_to(ROOT)): digest(p) for p in paths},
        'schema': transport.base.SCHEMA, 'instructions': transport.base.INSTRUCTIONS,
        'description_limits': {'single': 12000, 'batch': 6000},
        'batch_schema_enforcement': 'Bare JSON array, schema included in prompt and validated locally; CLI object-root structured-output flag omitted for batches.',
        'gate': {'verdict_agreement': .93, 'parse_failure': .05},
        'evaluation_ids': [e['posting_id'] for e in entries]})
    return entries, posts


def audit(entries, posts):
    old = read_json(BASE / 'v2/rule-conflicts.json')
    resolved, confirmed, candidates = [], [], []
    for x in old['confirmed']:
        if x['receipt_verdict'] == 'strong' and 'stretch rule' in x['reason']:
            resolved.append({**x, 'v3_resolution': 'Years no longer cap strong, including zero-lower-bound ranges. This specific v2 conflict is resolved.'})
        else:
            confirmed.append({**x, 'v3_review': 'Retained: v3 also disallows rejection solely for a 1-2-year minimum. Prior source adjudication preserved.'})
    for x in old['candidates']:
        reasons = [r for r in x['reasons_to_review'] if not r.startswith('Strong receipt with a possible <=2-year')]
        if reasons:
            candidates.append({**x, 'reasons_to_review': reasons})
    # Rerun the deterministic screen over every receipt with the obsolete strong cap removed.
    import audit_rules_v2 as rules
    known = {x['posting_id'] for x in confirmed + candidates}
    for e in entries:
        p = posts[e['posting_id']]
        r = e['receipt']
        reasons = []
        if r['verdict'] == 'reject' and re.search(rules.LOW_YEARS, r['reason']):
            reasons.append('Reject mentions 1-2 years: inspect independent gates; tenure alone cannot reject under v3.')
        if r['verdict'] in ('strong', 'stretch'):
            if re.search(rules.STUDENT_TITLE, p['title']):
                reasons.append('Inspect student-role exclusion.')
            loc = ' | '.join(p['locations'])
            if re.search(rules.FOREIGN, loc) and not re.search(rules.US, loc):
                reasons.append('Inspect non-US location exclusion.')
            if re.search(r'\b[3-9]\s*(?:\+|[-–]\s*\d+)?\s*(?:years?|yrs?)\b', r['reason'], re.I):
                reasons.append('Inspect mandatory lower bound and accepted academic/project/internship pathway.')
        if reasons and e['posting_id'] not in known:
            candidates.append({'posting_id': e['posting_id'], 'split': e['split'], 'receipt_verdict': r['verdict'], 'reasons_to_review': reasons})
    save(OUT / 'rule-conflicts.json', {'scope': len(entries), 'method': 'Rerun conservative deterministic screen under v3; carry forward prior human source adjudications for unchanged rules; remove forced-stretch-only conflicts. Candidates are unconfirmed, not label errors. No receipt changes or model-output-based adjudication.',
        'confirmed': confirmed, 'resolved_v2_conflicts': resolved, 'candidates': candidates})


def payload(p, limit):
    return {'employer': p.get('employer'), 'title': p.get('title'), 'location': p.get('locations', []),
            'posted': p.get('posted'), 'description': p.get('description', '')[:limit]}


def request(ids, posts, context, batch):
    if not batch:
        return transport.base.prompt_for(context, posts[ids[0]]), transport.base.SCHEMA
    item_schema = {**transport.base.SCHEMA,
                   'properties': {**transport.base.SCHEMA['properties'], 'posting_id': {'type': 'string', 'enum': ids}},
                   'required': [*transport.base.SCHEMA['required'], 'posting_id']}
    schema = {'type': 'array', 'items': item_schema, 'minItems': len(ids), 'maxItems': len(ids)}
    prompt = context + '\n\nIndependently classify each delimited posting. Return an array of exactly ' + str(len(ids)) + ' objects, each with its exact posting_id. Never transfer evidence between postings.\n'
    for pid in ids:
        prompt += '<posting id="' + pid + '">\n' + json.dumps(payload(posts[pid], 6000), ensure_ascii=False) + '\n</posting>\n'
    prompt += '\nRequired JSON output schema:\n' + json.dumps(schema) + '\n'
    return prompt, schema


def rows_for(phase):
    return [r for r in jsonl(OUT / 'calls.jsonl') if r['phase'] == phase]


def outputs(phase):
    result = {}
    for row in rows_for(phase):
        if row['status'] == 'ok':
            values = row['output'] if row['batch'] else [{'posting_id': row['ids'][0], **row['output']}]
            for value in values:
                result[value['posting_id']] = value
    return result


def status(state, phase, **extra):
    save(OUT / 'progress.json', {'state': state, 'phase': phase, 'pid': os.getpid(), 'updated_at': now(), **extra})


def run_phase(phase, groups, posts, context, budget, batch=False):
    existing = rows_for(phase)
    completed = {tuple(r['ids']) for r in existing if r['status'] in ('ok', 'parse_failure')}
    todo = [ids for ids in groups if tuple(ids) not in completed]
    pending = {}
    stop_reason = None
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        while todo or pending:
            while todo and len(pending) < 4 and not transport.STOP.is_set():
                ids = todo[0]
                prompt, schema = request(ids, posts, context, batch)
                if len((prompt + transport.base.INSTRUCTIONS + json.dumps(schema)).encode()) > transport.MAX_REQUEST_BYTES:
                    raise RuntimeError('Prompt outside reservation envelope')
                key = phase + '-' + str(time.time_ns())
                refusal = budget.reserve(key)
                if refusal:
                    # Pending reservations may become smaller after reconciliation.
                    if not pending:
                        stop_reason = refusal
                    break
                todo.pop(0)
                artifact = OUT / 'calls' / key
                meta = {'call_id': key, 'phase': phase, 'ids': ids, 'batch': batch,
                        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'started_at': now(),
                        'artifact': str(artifact.relative_to(ROOT))}
                v2.append(OUT / 'dispatch.jsonl', meta)
                pending[pool.submit(transport.invoke, prompt, artifact, schema, ids, batch)] = meta
            status('running', phase, remaining=len(todo), active=len(pending), successful=len(outputs(phase)))
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                meta = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    transport.stop('transport_exception')
                    result = {'status': 'failure', 'usage': None, 'error': str(exc)}
                row = {**meta, **result, 'finished_at': now(), 'day': day()}
                v2.append(OUT / 'calls.jsonl', row)
                budget.settle(row)
                if phase == 'bulk':
                    gate = read_json(OUT / 'gate.json')
                    export_bulk(read_json(OUT / f'bulk-sample-{gate["bulk_target"]}.json'), gate['batch_size'])
            if phase == 'evaluation' and time.monotonic() - last_report > 60:
                analyze()
                last_report = time.monotonic()
            if transport.STOP.is_set():
                stop_reason = transport.STOP_REASON
                if not pending:
                    break
        status(stop_reason or ('complete' if not todo else 'incomplete'), phase,
               remaining=len(todo), successful=len(outputs(phase)))
    return stop_reason is None and not todo


def analyze():
    entries = v2.entries_for(final=True)
    labels = outputs('evaluation')
    matched = [e for e in entries if e['posting_id'] in labels]
    def metric(es):
        return v2.metrics([(e['receipt']['verdict'], labels[e['posting_id']]['verdict']) for e in es])
    old = v2.successful(jsonl(BASE / 'v2/runs/full.jsonl'))
    overlap = [e for e in matched if all((e['posting_id'], r) in old for r in (1, 2, 3))]
    old_pairs = [(e['receipt']['verdict'], v2.majority([old[e['posting_id'], r]['output']['verdict'] for r in (1, 2, 3)])) for e in overlap]
    # Also expose a single-vote comparison for every posting with any completed v2 vote.
    any_overlap = [e for e in matched if any((e['posting_id'], r) in old for r in (1, 2, 3))]
    first_pairs = [(e['receipt']['verdict'], next(old[e['posting_id'], r]['output']['verdict'] for r in (1, 2, 3) if (e['posting_id'], r) in old)) for e in any_overlap]
    audit_result = read_json(OUT / 'rule-conflicts.json')
    summary = {'complete': len(matched) == len(entries), 'completed': len(matched), 'target': len(entries),
               'overall': metric(matched), 'by_split': {s: metric([e for e in matched if e['split'] == s]) for s in ('train', 'val', 'test')},
               'verification': v2.binary_metrics([(e['receipt']['verdict'] == 'needs-information', labels[e['posting_id']]['needs_verification'] == 'yes') for e in matched]),
               'comparison': {'n_postings': len(overlap), 'v2_majority': v2.metrics(old_pairs), 'v3_single': metric(overlap)},
               'comparison_any_v2': {'n_postings': len(any_overlap), 'v2_first_completed': v2.metrics(first_pairs), 'v3_single': metric(any_overlap)},
               'updated_at': now()}
    save(OUT / 'agreement-summary.json', summary)
    lines = ['# V3 receipt agreement', '', f'Status: {"COMPLETE" if summary["complete"] else "INTERIM / INCOMPLETE"}; {len(matched)}/{len(entries)} postings.', '',
             'Frozen user-approved context; one fresh gpt-6-astra medium call per posting. No tuning after results. NI receipts excluded from three-class scores. Incomplete subsets can be biased by completion order. Train/val were previously used in v2 selection; test was already inspected in v2, so this is not a newly untouched holdout.', '',
             '| Cohort | N non-NI | Accuracy | Kappa | Always reject | Strong recall |', '|---|---:|---:|---:|---:|---:|']
    for name, m in [('overall', summary['overall']), *summary['by_split'].items()]:
        lines.append(f'| {name} | {m["n"]} | {pct(m["accuracy"])} | {m["kappa"]} | {pct(m["always_reject_accuracy"])} | {pct(m["classes"]["strong"]["recall"])} |')
    lines += ['', 'Confusion rows = receipt strong/stretch/reject; columns = prediction strong/stretch/reject.', '', '| Receipt | strong | stretch | reject |', '|---|---:|---:|---:|']
    for label, row in zip(v2.CLASSES, summary['overall']['confusion']):
        lines.append('| ' + label + ' | ' + ' | '.join(map(str, row)) + ' |')
    lines += ['', '| Class | Precision | Recall | Support |', '|---|---:|---:|---:|']
    for label, m in summary['overall']['classes'].items():
        lines.append(f'| {label} | {pct(m["precision"])} | {pct(m["recall"])} | {m["support"]} |')
    lines += ['', '## Verification proxy', '', json.dumps(summary['verification']), '', 'NI often reflects missing live-site state, while needs_verification measures unresolved text gates; these are different constructs.', '', '## Matched v2 vs v3', '',
              f'{len(overlap)} postings with all three v2 votes and one v3 result. V2 majority vs v3 single; the number of votes differs.', '', '| Model | Accuracy | Kappa | Strong recall |', '|---|---:|---:|---:|']
    for name in ('v2_majority', 'v3_single'):
        m = summary['comparison'][name]
        lines.append(f'| {name} | {pct(m["accuracy"])} | {m["kappa"]} | {pct(m["classes"]["strong"]["recall"])} |')
    lines += ['', 'Additional first-completed-v2-vote comparison (all overlapping completed postings):', '', '```json', json.dumps(summary['comparison_any_v2'], indent=2), '```', '', '## Updated rule-conflict audit', '', audit_result['method'], '',
              f'{len(audit_result["confirmed"])} retained confirmed conflicts; {len(audit_result["resolved_v2_conflicts"])} forced-stretch conflicts resolved; {len(audit_result["candidates"])} unconfirmed candidates. Full lists: rule-conflicts.json.', '']
    for x in audit_result['confirmed']:
        lines.append(f'- `{x["posting_id"]}` ({x["split"]}, {x["receipt_verdict"]}): {x["reason"]}')
    lines += ['', '## Budget and resumption', '', 'budget.json sums input_tokens from turn.completed, including cached input. Unknown usage retains conservative reservations. Daily cap uses America/New_York; no automatic next-day resume. No source files or private configuration were accessed. No commit.', '',
              '`python -B scripts/label_agreement_v3.py status`', '', '`powershell -NoProfile -File scripts/start_label_agreement_v3.ps1`', '']
    (OUT / 'AGREEMENT-v3.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def pilot_groups(entries, size):
    ids = sorted(e['posting_id'] for e in entries if e['split'] == 'train' and e['posting_id'] in outputs('evaluation'))
    ids = random.Random(SEED).sample(ids, 100)
    random.Random(SEED + 1).shuffle(ids)
    v2.freeze(OUT / 'pilot-ids.json', ids)
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def pilot_report(size):
    phase = f'pilot{size}'
    rows = rows_for(phase)
    labels = outputs(phase)
    single = outputs('evaluation')
    ids = read_json(OUT / 'pilot-ids.json')
    fields = [*transport.base.ENUMS, 'min_years']
    agreement = {k: sum(pid in labels and labels[pid][k] == single[pid][k] for pid in ids) / len(ids) for k in fields}
    failures = sum(r['status'] == 'parse_failure' for r in rows)
    parse_rate = failures / len(rows) if rows else 1
    single_cost = sum(r['usage']['input_tokens'] for r in rows_for('evaluation') if r['status'] == 'ok' and r['ids'][0] in ids) / len(ids)
    batch_cost = sum(r.get('usage', {}).get('input_tokens', 0) for r in rows if r.get('usage')) / len(ids)
    result = {'size': size, 'postings': len(ids), 'valid_postings': len(labels), 'calls': len(rows),
              'parse_failures': failures, 'parse_failure_rate': parse_rate,
              'agreement_including_missing_as_disagreement': agreement,
              'input_tokens_per_posting': {'single': single_cost, 'batch': batch_cost},
              'passed': agreement['verdict'] >= .93 and parse_rate <= .05}
    save(OUT / f'pilot{size}-summary.json', result)
    (OUT / f'PILOT-{size}.md').write_text('# Batch pilot\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n\nSame 100 train postings, shuffled; batch descriptions limited to 6,000 characters. Missing labels count as disagreement. Reason is free text, not a typed-field agreement target.\n', encoding='utf-8')
    return result


def sample_bulk(count=1500):
    manifest_path = OUT / f'bulk-sample-{count}.json'
    splits = read_json(ROOT / 'data/processed/splits.json')
    raw = {p['posting_id']: p for p in jsonl(ROOT / 'data/processed/postings.jsonl') if p['posting_id'] in set(splits['unlabeled_train_pool'])}
    posts, _ = clean_postings(raw)
    for p in posts.values():
        if splits['employer_assignment'].get(p['normalized_employer']) in ('val', 'test'):
            raise RuntimeError('Held-out employer in unlabeled train pool')
    ids = sorted(posts)
    pattern = r'\b(new[ -]?grad(?:uate)?|junior|jr\.?|associate|entry(?:[ -]level|[ -]tier)?|engineer\s+i|developer\s+i)\b'
    entry = [pid for pid in ids if re.search(pattern, ' | '.join([posts[pid]['title'], *posts[pid].get('titles', [])]), re.I)]
    n = count // 2
    rng = random.Random(SEED + 2)
    entry_set = set(entry)
    soft = sorted(pid for pid in ids if pid not in entry_set and soft_entry(posts[pid]))
    if len(entry) > n:
        raise RuntimeError('Entry-title census exceeds enriched target; sampling policy needs review')
    soft_count = min(n - len(entry), len(soft))
    enriched = rng.sample(soft, soft_count)
    remaining = sorted(set(ids) - entry_set - set(enriched))
    uniform_count = count - len(entry) - soft_count
    uniform = rng.sample(remaining, uniform_count)
    soft_set = set(soft)
    uniform_probability = uniform_count / len(remaining)
    selected = []
    for stratum, sample in [('entry_title', entry), ('soft_entry', enriched), ('uniform', uniform)]:
        for pid in sample:
            if pid in entry_set:
                probability = 1.0
            elif pid in soft_set:
                first = soft_count / len(soft)
                probability = first + (1 - first) * uniform_probability
            else:
                probability = uniform_probability
            selected.append({'posting_id': pid, 'stratum': stratum, 'entry_signal': pid in entry_set,
                             'soft_entry_signal': pid in soft_set,
                             'inclusion_probability': probability, 'weight': 1 / probability})
    rng.shuffle(selected)
    manifest = {'seed': SEED + 2, 'pool_size': len(ids), 'entry_pool_size': len(entry), 'soft_entry_pool_size': len(soft), 'count': count,
                'stratum_counts': dict(Counter(x['stratum'] for x in selected)),
                'signal': 'Entry title census, then soft-entry regex over cleaned posting descriptions, then uniform from the remainder. No receipts or model labels. User-approved fallback.',
                'entry_regex': pattern, 'weight_definition': 'Inverse marginal inclusion probability: entry title census=1; soft=k/S+(1-k/S)*u/(N-E-k); other=u/(N-E-k). Selection stratum can be uniform even when soft eligible.', 'entries': selected}
    v2.freeze(manifest_path, manifest)
    return manifest, posts


def soft_entry(post):
    title = ' | '.join([post['title'], *post.get('titles', [])])
    if re.search(r'\b(senior|sr|staff|principal|lead|manager|director|head|ii|iii|iv|v|vi|[2-9])\b', title, re.I):
        return False
    text = post['description']
    years = list(re.finditer(r'\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:\+|(?:-|–|—|to)\s*\d{1,2})?\s*(?:years?|yrs?)\b', text, re.I))
    values = []
    words = dict(zip(['one','two','three','four','five','six','seven','eight','nine','ten'], range(1, 11)))
    for m in years:
        # Prefer false negatives over treating a high requirement as soft-entry.
        before = text[max(0, m.start()-45):m.start()].lower()
        after = text[m.end():m.end()+35].lower()
        if re.search(r'preferred\s*[:;-]?\s*$', before) or re.match(r'\s*(?:of experience\s*)?(?:is\s+)?preferred\b', after):
            continue
        value = m.group(1).lower()
        values.append(int(value) if value.isdigit() else words[value])
    if values:
        return max(values) <= 2
    # Unquantified years language is ambiguous, not proof of no requirement.
    return not re.search(r'\b(?:years?|yrs?)\b.{0,30}\bexperience\b', text, re.I)


def export_bulk(manifest, size):
    existing = {r['posting_id'] for r in jsonl(OUT / 'bulk-labels.jsonl')}
    labels = outputs('bulk')
    for e in manifest['entries']:
        pid = e['posting_id']
        if pid in labels and pid not in existing:
            v2.append(OUT / 'bulk-labels.jsonl', {**labels[pid], **e,
                'labeler': {'model': 'gpt-6-astra', 'effort': 'medium', 'context_sha256': digest(OUT / 'context-v3.md'),
                            'mode': 'single' if size == 1 else f'batch-{size}'}})


def run():
    if time.tzname[0] not in ('Eastern Standard Time', 'EST'):
        raise RuntimeError('Calendar-day cap requires America/New_York local timezone')
    entries, posts = prepare()
    context = (OUT / 'context-v3.md').read_text(encoding='utf-8')
    budget = Budget()
    for row in jsonl(OUT / 'calls.jsonl'):
        budget.settle(row)
    # Do not silently replay a call whose process crashed before durable completion.
    recorded = {r['call_id'] for r in jsonl(OUT / 'calls.jsonl')}
    orphaned = [r['call_id'] for r in jsonl(OUT / 'dispatch.jsonl') if r['call_id'] not in recorded]
    if orphaned:
        status('orphaned_calls_require_reconciliation', 'recovery', calls=orphaned)
        return
    ids = [e['posting_id'] for e in entries]
    random.Random(SEED).shuffle(ids)
    ok = run_phase('evaluation', [[pid] for pid in ids], posts, context, budget)
    summary = analyze()
    if not ok or not summary['complete']:
        return
    size = 1
    for candidate in (5, 3):
        if not run_phase(f'pilot{candidate}', pilot_groups(entries, candidate), posts, context, budget, batch=True):
            return
        result = pilot_report(candidate)
        if result['passed']:
            size = candidate
            break
    v2.freeze(OUT / 'gate.json', {'mode': 'single' if size == 1 else f'batch-{size}', 'batch_size': size,
                                'bulk_target': 500 if size == 1 else 1500})
    manifest, bulk_posts = sample_bulk(500 if size == 1 else 1500)
    bulk_ids = [e['posting_id'] for e in manifest['entries']]
    for pid in bulk_ids:
        if re.search(CONTAMINATION, bulk_posts[pid]['description'][:12000]):
            raise RuntimeError(f'Possible reviewer contamination in bulk input: {pid}')
    groups = [bulk_ids[i:i + size] for i in range(0, len(bulk_ids), size)]
    export_bulk(manifest, size)
    try:
        run_phase('bulk', groups, bulk_posts, context, budget, batch=size > 1)
    finally:
        export_bulk(manifest, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'analyze', 'status'])
    args = parser.parse_args()
    if args.action == 'status':
        for name in ('progress.json', 'budget.json', 'gate.json'):
            path = OUT / name
            if path.exists():
                data = read_json(path)
                data.pop('accounted_calls', None)
                if 'reservations' in data:
                    data['reserved_input_tokens'] = sum(v['tokens'] for v in data.pop('reservations').values())
                print(name + ': ' + json.dumps(data))
        return
    v2.OUT = OUT  # Redirect the reused OS lock only; no v2 write functions are invoked.
    with v2.lock():
        try:
            if args.action == 'prepare':
                prepare()
                Budget()
                analyze()
            elif args.action == 'analyze':
                analyze()
            else:
                run()
        except Exception as exc:
            transport.stop('runner_exception')
            status('stopped_error', args.action, error=str(exc))
            raise


if __name__ == '__main__':
    main()
