"""V4 labeller on the OpenCode Go subscription (paid `opencode-go/*` models only; never Codex).

Same frozen v3 context, instructions, payload and schema as label_agreement_v3, so labels
are comparable with the gpt-6-astra v3 run. Standard library only.

Transport: a private `opencode serve` on 127.0.0.1 (random per-run password, memory only)
and its stateless POST /api/experimental/generate/text. That route sends only
{model, prompt} to the provider: no agent system prompt, tools or AGENTS.md. OpenCode's
own stored Go sign-in is used; no key is read, printed or written by this script.
providerID is pinned to `opencode-go` (never `opencode`, which bills Zen credits).

The route returns text only (no usage, no structured output), so: system instructions are
prepended to the v3 prompt, the JSON schema is appended and validated locally (one retry),
tokens are estimated from characters plus a reasoning allowance, and the Go usage meter
(/api/experimental/usage/limits) is snapshotted and enforced alongside the estimate.

  python -B scripts/label_opencode.py check
  python -B scripts/label_opencode.py probe --models glm-5.3-flash
  python -B scripts/label_opencode.py bakeoff [--models ...]
  python -B scripts/label_opencode.py analyze-bakeoff
  python -B scripts/label_opencode.py run --model M --phase evaluation|bulk
  python -B scripts/label_opencode.py pilot --model M
  python -B scripts/label_opencode.py report

Budget: estimated spend <= 40% of each model's Go plan monthly allowance, <= 10% per
rolling 5 hours; also stop if the Go meter is limited or reaches rolling 15% / weekly 40% /
monthly 40%. Stops on usage/rate-limit errors and after 3 consecutive transport failures.
No sleeping or automatic resumption; rerunning skips completed postings.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import codex_labeler_v2 as base
import label_agreement_v2 as v2
from label_agreement import CONTAMINATION, clean_postings, digest, jsonl, now, pct, read_json

ROOT = base.REPO
V3 = ROOT / 'data/labeling/v3'
OUT = ROOT / 'data/labeling/v4'
PROVIDER = 'opencode-go'
SEED = 20261027
TIMEOUT = 240
CAP_SHARE = .40
WINDOW_SHARE = .10
WINDOW = dt.timedelta(hours=5)
METER_LIMITS = {'rolling': 15, 'weekly': 40, 'monthly': 40}
METER_EVERY = 20
MAX_CONSECUTIVE_FAILURES = 3
MAX_PARSE_STREAK = 8
CHARS_PER_TOKEN = 3.5  # conservative for English prose/JSON; overestimates tokens
CLASSES = base.VERDICTS

# model: (Go plan monthly $, Zen $/1M (input, output, cached), peak-priced, variant, reasoning-token allowance)
MODELS = {
    'deepseek-v4.1-flash': (160, (.30, 1.20, .006), True, 'low', 1500),
    'glm-5.3-flash': (180, (.15, .50, .03), False, 'low', 1500),
    'glm-5.3': (120, (1.40, 4.40, .26), False, 'low', 1500),
    'kimi-k2.7-code': (180, (.95, 4.00, .19), False, None, 1500),
    'kimi-k3': (60, (3.0, 15.0, .30), False, 'max', 4000),
    'minimax-m3': (180, (.30, 1.20, .06), False, 'none', 300),
    'claude-haiku-5-5': (60, (.10, .50, .01), False, 'low', 1500),
    'qwen3.8-max': (60, (2.0, 6.0, .25), False, 'low', 1500),
    'gpt-6-luna': (60, (.10, .50, .01), False, 'low', 1500),
    'grok-4.7': (60, (2.0, 6.0, .50), False, 'low', 1500),
    'gemini-4-argon': (240, (2.0, 10.0, .10), False, 'low', 1500),
}
BAKEOFF_MODELS = ['deepseek-v4.1-flash', 'glm-5.3-flash', 'glm-5.3', 'kimi-k2.7-code',
                  'minimax-m3', 'gpt-6-luna', 'claude-haiku-5-5', 'gemini-4-argon']
SCHEMA_SUFFIX = '\n\nReturn only one JSON object (no markdown fences, no prose) matching this JSON schema:\n'

WRITE = threading.RLock()
GLOBAL_STOP = threading.Event()


# ---------------------------------------------------------------- persistence

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def append(row, name='calls.jsonl'):
    with WRITE:
        v2.append(OUT / name, row)


@contextmanager
def lock():
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runner.lock').open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit('Another label_opencode runner holds data/labeling/v4/runner.lock')
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


# ---------------------------------------------------------------- private OpenCode server

def opencode_exe():
    if os.environ.get('OPENCODE_BIN'):
        return os.environ['OPENCODE_BIN']
    found = shutil.which('opencode')
    if not found:
        raise SystemExit('opencode CLI not found on PATH')
    exe = Path(found).parent / 'node_modules/@opencode/cli/bin/opencode.exe'
    return str(exe) if exe.exists() else found


class Server:
    """Private loopback `opencode serve`; uses OpenCode's stored Go sign-in."""

    def __init__(self):
        self.password = secrets.token_urlsafe(24)
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            self.port = s.getsockname()[1]
        self.url = f'http://127.0.0.1:{self.port}'
        cwd = OUT / 'oc-cwd'
        cwd.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in ('OPENCODE_PASSWORD', 'OPENCODE_SERVER_PASSWORD')}
        env['OPENCODE_SERVER_PASSWORD'] = self.password
        self.auth = 'Basic ' + base64.b64encode(f'opencode:{self.password}'.encode()).decode()
        self.proc = subprocess.Popen([opencode_exe(), 'serve', '--hostname', '127.0.0.1', '--port', str(self.port)],
                                     cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit('opencode serve exited during startup')
            try:
                status, _ = self.request('GET', '/api/info', timeout=5)
                if status == 200:
                    return
            except OSError:
                pass
            time.sleep(1)
        self.close()
        raise SystemExit('opencode serve did not become ready')

    def request(self, method, path, body=None, timeout=TIMEOUT):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={'Authorization': self.auth, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw, status = r.read().decode('utf-8', 'replace'), r.status
        except urllib.error.HTTPError as e:
            raw, status = e.read().decode('utf-8', 'replace'), e.code
        try:
            return status, json.loads(raw)
        except ValueError:
            return status, raw

    def generate(self, model, prompt):
        # Hard pin: only the Go subscription provider, only allowed paid models.
        assert model in MODELS
        spec = {'id': model, 'providerID': PROVIDER}
        if MODELS[model][3]:
            spec['variant'] = MODELS[model][3]
        return self.request('POST', '/api/experimental/generate/text', {'prompt': prompt, 'model': spec})

    def meter(self):
        # The first read after startup can transiently return null; retry before failing closed.
        for attempt in range(4):
            try:
                status, data = self.request('GET', '/api/experimental/usage/limits', timeout=30)
            except OSError:
                status, data = None, None
            if status == 200 and isinstance(data, dict) and (data.get('data') or {}).get('go'):
                return data['data']['go']
            time.sleep(3)
        return None

    def close(self):
        if self.proc.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(self.proc.pid), '/T', '/F'], capture_output=True)
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@contextmanager
def server():
    s = Server()
    try:
        yield s
    finally:
        s.close()


# ---------------------------------------------------------------- budget

def peak(model, at=None):
    """DeepSeek peak: 01-04 and 06-10 UTC Mon-Fri (Beijing 09-12, 14-18)."""
    at = at or dt.datetime.now(dt.timezone.utc)
    return MODELS[model][2] and at.weekday() < 5 and (1 <= at.hour < 4 or 6 <= at.hour < 10)


def cost(model, usage, peak_hour=False):
    pin, pout, pcache = MODELS[model][1]
    fresh = max(usage['input_tokens'] - usage.get('cached_tokens', 0), 0)
    value = (fresh * pin + usage.get('cached_tokens', 0) * pcache + usage['output_tokens'] * pout) / 1e6
    return value * (2 if peak_hour else 1)


def estimate_usage(model, prompt, text):
    visible = len(text or '') / CHARS_PER_TOKEN
    return {'input_tokens': round(len(prompt) / CHARS_PER_TOKEN), 'cached_tokens': 0,
            'visible_output_tokens': round(visible), 'reasoning_allowance': MODELS[model][4],
            'output_tokens': round(visible + MODELS[model][4]), 'estimated': True}


def reservation(model, prompt):
    return cost(model, {'input_tokens': len(prompt) / CHARS_PER_TOKEN, 'output_tokens': 4000 + MODELS[model][4]}, peak(model))


class Ledger:
    """Estimated spend per model from calls.jsonl plus in-flight reservations (fail closed)."""

    def __init__(self):
        self.guard = threading.Lock()
        self.rows = [r for r in jsonl(OUT / 'calls.jsonl') if r.get('charged_usd')]
        self.reserved = {}

    def spent(self, model, since=None):
        return sum(r['charged_usd'] for r in self.rows if r['model'] == model
                   and (since is None or dt.datetime.fromisoformat(r['finished_at']) >= since))

    def reserve(self, key, model, amount):
        monthly = MODELS[model][0]
        with self.guard:
            pending = sum(a for (m, a) in self.reserved.values() if m == model)
            since = dt.datetime.now(dt.timezone.utc) - WINDOW
            if self.spent(model) + pending + amount > CAP_SHARE * monthly:
                return 'model_budget_cap_40pct'
            if self.spent(model, since) + pending + amount > WINDOW_SHARE * monthly:
                return 'five_hour_cap_10pct'
            self.reserved[key] = (model, amount)
        return None

    def settle(self, key, row):
        with self.guard:
            self.reserved.pop(key, None)
            if row.get('charged_usd'):
                self.rows.append(row)


def meter_problem(go):
    if go is None:
        return 'go_meter_unavailable'
    for name, limit in METER_LIMITS.items():
        m = go.get(name) or {}
        if m.get('limited'):
            return f'go_meter_{name}_limited'
        if (m.get('percent') or 0) >= limit:
            return f'go_meter_{name}_{m.get("percent")}pct'
    return None


def snapshot(srv, model, phase, label):
    go = srv.meter()
    append({'at': now(), 'model': model, 'phase': phase, 'label': label, 'go': go}, 'meter.jsonl')
    return go


# ---------------------------------------------------------------- one call

LIMIT = re.compile(r'usage[ _-]*limit|rate[ _-]*limit|too many requests|quota|allowance|insufficient|exceeded|limit reached|\b429\b|\b402\b', re.I)
AUTH = re.compile(r'credentials are unavailable|unauthori[sz]ed|forbidden|sign[ -]?in|\b401\b|\b403\b', re.I)


def compose(prompt, schema):
    """System instructions first (no system role on this route), schema last unless inlined (batch)."""
    full = base.INSTRUCTIONS + '\n\n' + prompt
    if schema.get('type') == 'object':
        full += SCHEMA_SUFFIX + json.dumps(schema)
    return full


def extract(text, batch=False):
    t = re.sub(r'(?s)<think>.*?</think>', '', text or '').strip()
    t = re.sub(r'^```(?:json)?\s*|\s*```$', '', t).strip()
    try:
        return json.loads(t), False
    except ValueError:
        o, c = ('[', ']') if batch else ('{', '}')
        i, j = t.find(o), t.rfind(c)
        if i < 0 or j <= i:
            raise ValueError('No JSON found in model output')
        return json.loads(t[i:j + 1]), True


def validate_batch(value, ids):
    if not isinstance(value, list) or len(value) != len(ids):
        raise ValueError('Batch must be an array with exactly one object per ID')
    seen = set()
    for item in value:
        if not isinstance(item, dict) or item.get('posting_id') not in ids or item['posting_id'] in seen:
            raise ValueError('Missing, duplicate or unexpected posting_id')
        seen.add(item['posting_id'])
        base.validate({k: v for k, v in item.items() if k != 'posting_id'})
    return value


def classify_error(status, data):
    text = json.dumps(data) if not isinstance(data, str) else data
    tag = data.get('_tag', '') if isinstance(data, dict) else ''
    inner = data.get('status') if isinstance(data, dict) else None
    if status == 401 and not inner:
        return 'server_auth', text  # our private server rejected us
    if inner in (401, 403) or AUTH.search(text) and 'Provider' not in tag:
        return 'auth', text
    if inner in (402, 429) or status in (402, 429) or LIMIT.search(text):
        return 'limit', text
    if re.search(r'ModelNotFound|UnsupportedModel|InvalidRequest|VariantUnavailable|invalid_request|requires Global regions', tag + text):
        return 'bad_request', text
    return 'transport', text


def call(srv, model, prompt, schema, ids, batch):
    full = compose(prompt, schema)
    is_peak = peak(model)
    started, t0 = now(), time.monotonic()
    meta = {'started_at': started, 'provider': PROVIDER, 'variant': MODELS[model][3], 'peak': is_peak,
            'composed_sha256': hashlib.sha256(full.encode()).hexdigest(), 'composed_chars': len(full)}
    try:
        status, data = srv.generate(model, full)
    except Exception as exc:  # timeout / connection: usage unknown
        return {**meta, 'seconds': round(time.monotonic() - t0, 3), 'status': 'error', 'error': repr(exc)[:500],
                'usage_unknown': True}, 'transport'
    meta['seconds'] = round(time.monotonic() - t0, 3)
    meta['http_status'] = status
    if status != 200 or not isinstance(data, dict) or 'data' not in data:
        kind, text = classify_error(status, data)
        return {**meta, 'status': 'error', 'error': text[:1500], 'charged_usd': 0.0}, kind
    text = data['data'].get('text', '')
    usage = estimate_usage(model, full, text)
    row = {**meta, 'usage': usage, 'cost_usd': round(cost(model, usage, is_peak), 7)}
    row['charged_usd'] = row['cost_usd']
    try:
        value, lenient = extract(text, batch)
        value = validate_batch(value, ids) if batch else base.validate(value)
        return {**row, 'status': 'ok', 'output': value, 'lenient_extraction': lenient}, None
    except (ValueError, TypeError) as exc:
        return {**row, 'status': 'parse_failure', 'error': str(exc)[:300], 'raw_text': (text or '')[:3000]}, None


# ---------------------------------------------------------------- data

def context():
    return (V3 / 'context-v3.md').read_text(encoding='utf-8')


def evaluation():
    entries = v2.entries_for(final=True)
    return entries, v2.posting_inputs(entries)


def astra_labels():
    result = {}
    for row in jsonl(V3 / 'calls.jsonl'):
        if row['phase'] == 'evaluation' and row['status'] == 'ok' and not row['batch']:
            result[row['ids'][0]] = {**row['output'], '_prompt_sha256': row['prompt_sha256']}
    return result


def bulk_inputs():
    manifest = read_json(V3 / 'bulk-sample-1500.json')
    ids = {e['posting_id'] for e in manifest['entries']}
    raw = {p['posting_id']: p for p in jsonl(ROOT / 'data/processed/postings.jsonl') if p['posting_id'] in ids}
    posts, _ = clean_postings(raw)
    assert set(posts) == ids, 'bulk manifest ids missing from postings.jsonl'
    for pid, p in posts.items():
        if re.search(CONTAMINATION, p['description'][:12000]):
            raise RuntimeError(f'Possible reviewer contamination in bulk input: {pid}')
    return manifest, posts


def request(ids, posts, ctx, batch):
    import label_agreement_v3 as v3  # exact v3 single/batch prompt construction
    return v3.request(ids, posts, ctx, batch)


def bakeoff_ids(entries, labels):
    path = OUT / 'bakeoff-ids.json'
    if path.exists():
        return read_json(path)
    quotas = {'strong': 50, 'stretch': 50, 'reject': 50}
    rng = random.Random(SEED)
    cells, selected = {}, []
    for verdict, quota in quotas.items():
        pools = {s: sorted(e['posting_id'] for e in entries if e['split'] == s and e['receipt']['verdict'] == verdict
                           and e['posting_id'] in labels) for s in ('train', 'val', 'test')}
        total = sum(map(len, pools.values()))
        take = {s: min(len(p), max(5, round(quota * len(p) / total))) for s, p in pools.items() if s != 'train'}
        take['train'] = min(len(pools['train']), quota - sum(take.values()))
        for s, pool in pools.items():
            sample = rng.sample(pool, take[s])
            selected += sample
            cells[f'{verdict}/{s}'] = {'population': len(pool), 'sample': len(sample),
                                       'weight': len(pool) / len(sample) if sample else 0}
    index = {e['posting_id']: e for e in entries}
    weights = {pid: cells[f'{index[pid]["receipt"]["verdict"]}/{index[pid]["split"]}']['weight'] for pid in selected}
    rng.shuffle(selected)
    data = {'seed': SEED, 'population': 'Evaluation postings with a non-NI receipt and an ok gpt-6-astra v3 label',
            'quotas': quotas, 'cells': cells, 'ids': selected, 'weights': weights}
    v2.freeze(path, data)
    return data


# ---------------------------------------------------------------- runner

def outputs(phase, model):
    result, parse = {}, Counter()
    for row in jsonl(OUT / 'calls.jsonl'):
        if row.get('phase') != phase or row.get('model') != model:
            continue
        if row['status'] == 'ok':
            values = row['output'] if row.get('batch') else [{'posting_id': row['ids'][0], **row['output']}]
            for value in values:
                result[value['posting_id']] = value
        elif row['status'] == 'parse_failure':
            parse[tuple(row['ids'])] += 1
    return result, parse


def progress(model, phase, state, **extra):
    with WRITE:
        path = OUT / 'progress.json'
        data = read_json(path) if path.exists() else {}
        data[f'{model}/{phase}'] = {'state': state, 'updated_at': now(), **extra}
        save(path, data)


def run_model(srv, model, phase, groups, posts, ctx, ledger, batch=False, workers=4, allow_peak=False,
              expected_sha=None):
    if model not in MODELS:
        raise SystemExit(f'{model} is not an allowed paid Go model')
    done, parse = outputs(phase, model)
    todo = [g for g in groups if not all(pid in done for pid in g) and parse[tuple(g)] < 2]
    state = {'consecutive': 0, 'parse_streak': 0, 'stop': None, 'calls': 0}
    guard = threading.Lock()
    problem = meter_problem(snapshot(srv, model, phase, 'start'))
    if problem:
        progress(model, phase, problem, remaining=len(todo))
        return problem

    def one(ids):
        prompt, schema = request(ids, posts, ctx, batch)
        if expected_sha and not batch and expected_sha.get(ids[0]) not in (None, hashlib.sha256(prompt.encode()).hexdigest()):
            raise RuntimeError(f'Prompt drift vs v3 for {ids[0]}')
        attempts = parse[tuple(ids)]
        while attempts < 2:
            if GLOBAL_STOP.is_set() or state['stop']:
                return
            call_key = f'{phase}-{model}-{time.time_ns()}'
            held = reservation(model, compose(prompt, schema))
            refusal = ledger.reserve(call_key, model, held)
            if refusal:
                state['stop'] = state['stop'] or refusal
                return
            result, kind = call(srv, model, prompt, schema, ids, batch)
            row = {'call_id': call_key, 'phase': phase, 'model': model, 'ids': ids, 'batch': batch,
                   'attempt': attempts + 1, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                   **result, 'finished_at': now()}
            if row.get('usage_unknown'):
                row['charged_usd'] = held  # dropped connection: unknown spend, fail closed
            append(row)
            ledger.settle(call_key, row)
            with guard:
                state['calls'] += 1
                if result['status'] == 'error':
                    state['consecutive'] += 1
                    if kind in ('auth', 'server_auth'):
                        GLOBAL_STOP.set()
                        state['stop'] = kind
                    elif kind in ('limit', 'bad_request'):
                        state['stop'] = state['stop'] or kind
                    elif state['consecutive'] >= MAX_CONSECUTIVE_FAILURES:
                        state['stop'] = state['stop'] or 'consecutive_failures'
                    return
                state['consecutive'] = 0
                if result['status'] == 'ok':
                    state['parse_streak'] = 0
                    return
                state['parse_streak'] += 1
                if state['parse_streak'] >= MAX_PARSE_STREAK:
                    state['stop'] = state['stop'] or 'parse_failure_streak'
                    return
            attempts += 1

    pending, checked = set(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while todo or pending:
            while todo and len(pending) < workers and not state['stop'] and not GLOBAL_STOP.is_set():
                if peak(model) and not allow_peak:
                    state['stop'] = 'deepseek_peak_hours'
                    break
                pending.add(pool.submit(one, todo.pop(0)))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                f.result()
            if state['calls'] - checked >= METER_EVERY:
                checked = state['calls']
                problem = meter_problem(snapshot(srv, model, phase, 'periodic'))
                if problem:
                    state['stop'] = state['stop'] or problem
            progress(model, phase, 'running', remaining=len(todo), spent_usd_est=round(ledger.spent(model), 4))
    snapshot(srv, model, phase, 'end')
    final = state['stop'] or ('auth_error' if GLOBAL_STOP.is_set() else ('complete' if not todo else 'incomplete'))
    labelled, _ = outputs(phase, model)
    progress(model, phase, final, remaining=len(todo), labelled=len(labelled), spent_usd_est=round(ledger.spent(model), 4))
    return final


# ---------------------------------------------------------------- analysis

def phase_rows(phase, model, ids=None):
    return [r for r in jsonl(OUT / 'calls.jsonl') if r.get('phase') == phase and r.get('model') == model
            and (ids is None or set(r['ids']) & set(ids))]


def meter_deltas():
    """Go meter percent change across each model/phase run (start -> end)."""
    out = {}
    last = {}
    for snap in jsonl(OUT / 'meter.jsonl'):
        key = f'{snap["model"]}/{snap["phase"]}'
        if snap['label'] == 'start':
            last[key] = snap
        elif snap['label'] == 'end' and key in last and snap.get('go') and last[key].get('go'):
            a, b = last[key]['go'], snap['go']
            d = out.setdefault(key, {'rolling': 0.0, 'weekly': 0.0, 'monthly': 0.0})
            for name in d:
                # A window reset during the run makes the delta unreliable; skip it.
                if a[name]['resetsAt'] == b[name]['resetsAt']:
                    d[name] += b[name]['percent'] - a[name]['percent']
    return out


def bakeoff_summary(models=None):
    entries, _ = evaluation()
    index = {e['posting_id']: e for e in entries}
    astra = astra_labels()
    sel = read_json(OUT / 'bakeoff-ids.json')
    ids, weights = sel['ids'], sel['weights']
    w = [weights[p] for p in ids]
    receipt = [index[p]['receipt']['verdict'] for p in ids]
    astra_v = [astra[p]['verdict'] for p in ids]
    deltas = meter_deltas()
    result = {'n': len(ids), 'astra': {'receipt': v2.metrics(list(zip(receipt, astra_v))),
                                         'receipt_weighted': v2.metrics(list(zip(receipt, astra_v)), w)}, 'models': {}}
    for model in models or BAKEOFF_MODELS:
        labels, _ = outputs('bakeoff', model)
        rows = phase_rows('bakeoff', model, ids)
        if not rows:
            continue
        spent = sum(r.get('charged_usd', 0) for r in rows)
        attempted = {pid for r in rows for pid in r['ids'] if r['status'] in ('ok', 'parse_failure')}
        pred = [labels[p]['verdict'] if p in labels else 'missing' for p in ids]
        first = [r for r in rows if r.get('attempt') == 1 and r['status'] in ('ok', 'parse_failure')]
        answered = [r for r in rows if r['status'] in ('ok', 'parse_failure')]
        agree = lambda field: sum(p in labels and labels[p][field] == astra[p][field] for p in ids) / len(ids)
        result['models'][model] = {
            'labelled': len(labels), 'attempted': len(attempted),
            'vs_astra': v2.metrics(list(zip(astra_v, pred))),
            'vs_astra_weighted': v2.metrics(list(zip(astra_v, pred)), w),
            'vs_receipt': v2.metrics(list(zip(receipt, pred))),
            'vs_receipt_weighted': v2.metrics(list(zip(receipt, pred)), w),
            'field_agreement_vs_astra': {f: agree(f) for f in [*base.ENUMS, 'min_years'] if f != 'verdict'},
            'parse_failure_rate_final': sum(1 for p in attempted if p not in labels) / len(attempted) if attempted else None,
            'parse_failure_rate_first_attempt': sum(r['status'] == 'parse_failure' for r in first) / len(first) if first else None,
            'lenient_extractions': sum(bool(r.get('lenient_extraction')) for r in rows),
            'errors': sum(r['status'] == 'error' for r in rows),
            'spent_usd_est': spent, 'usd_per_1000_est': 1000 * spent / len(attempted) if attempted else None,
            'pct_monthly_budget_est': 100 * spent / MODELS[model][0],
            'go_meter_delta_pct': deltas.get(f'{model}/bakeoff'),
            'mean_visible_output_tokens': sum(r['usage']['visible_output_tokens'] for r in answered) / len(answered) if answered else None,
            'mean_seconds': sum(r.get('seconds') or 0 for r in rows) / len(rows),
        }
    save(OUT / 'bakeoff-summary.json', result)
    return result


def choose(summary):
    astra_acc = summary['astra']['receipt']['accuracy']
    ok = []
    for model, m in summary['models'].items():
        if m['attempted'] >= .95 * summary['n'] and m['vs_astra']['accuracy'] >= .88 \
                and m['vs_receipt']['accuracy'] >= astra_acc - .02:
            ok.append((m['usd_per_1000_est'], model))
    if ok:
        return min(ok)[1], True
    best = max(summary['models'].items(), key=lambda kv: kv[1]['vs_astra']['accuracy'] or 0)
    return best[0], False


def fmt(x):
    return 'n/a' if x is None else f'{x:.3f}'


def write_bakeoff(summary):
    a = summary['astra']
    winner, qualified = choose(summary) if summary['models'] else (None, False)
    lines = ['# V4 bake-off: OpenCode Go models vs gpt-6-astra v3', '',
             f'Stratified subset: {summary["n"]} evaluation postings with a non-NI receipt and an ok astra v3 label '
             '(50 strong / 50 stretch / 50 reject by receipt; train+val+test; `bakeoff-ids.json`). Same frozen v3 '
             'context, instructions, payload and schema; the v3 prompt SHA-256 is checked per posting against the v3 call log. '
             'Transport: private `opencode serve` -> `generate/text`, provider `opencode-go`, low-effort variant. '
             'One labelling per posting; one retry only after a parse/validation failure. Missing labels count as disagreement.', '',
             f'Reference: astra vs receipts on this subset = {pct(a["receipt"]["accuracy"])} (kappa {fmt(a["receipt"]["kappa"])}); '
             f'reweighted to the receipt-class x split population = {pct(a["receipt_weighted"]["accuracy"])}.', '',
             '| Model | Variant | Labelled | Agree astra | kappa astra | Agree receipt | Δ vs astra (receipt) | Receipt (weighted) | Parse fail final / 1st try | Est. $ / 1,000 | Est. % of Go budget | Go meter Δ (rolling/weekly/monthly %) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    order = sorted(summary['models'].items(), key=lambda kv: kv[1]['usd_per_1000_est'] or 1e9)
    for model, m in order:
        d = m['go_meter_delta_pct']
        meter = '/'.join(f'{d[k]:.2f}' for k in ('rolling', 'weekly', 'monthly')) if d else 'n/a'
        lines.append(f'| {model} | {MODELS[model][3] or "default"} | {m["labelled"]}/{summary["n"]} | {pct(m["vs_astra"]["accuracy"])} | {fmt(m["vs_astra"]["kappa"])} | '
                     f'{pct(m["vs_receipt"]["accuracy"])} | {100 * (m["vs_receipt"]["accuracy"] - a["receipt"]["accuracy"]):+.1f} pts | '
                     f'{pct(m["vs_receipt_weighted"]["accuracy"])} | {pct(m["parse_failure_rate_final"])} / {pct(m["parse_failure_rate_first_attempt"])} | '
                     f'${m["usd_per_1000_est"]:.2f} | {m["pct_monthly_budget_est"]:.2f}% | {meter} |')

    def pr(m):
        return ' | '.join(f'{pct(m["classes"][c]["precision"])} / {pct(m["classes"][c]["recall"])}' for c in CLASSES)
    lines += ['', '## Precision / recall vs receipts (strong | stretch | reject)', '',
              '| Model | strong P/R | stretch P/R | reject P/R |', '|---|---|---|---|', f'| gpt-6-astra (v3 reference) | {pr(a["receipt"])} |']
    lines += [f'| {model} | {pr(m["vs_receipt"])} |' for model, m in order]
    lines += ['', '## Precision / recall vs astra labels, and other fields', '',
              '| Model | strong P/R | stretch P/R | reject P/R | needs_verification agree | min_years agree | sponsorship agree |', '|---|---|---|---|---:|---:|---:|']
    for model, m in order:
        f = m['field_agreement_vs_astra']
        lines.append(f'| {model} | {pr(m["vs_astra"])} | {pct(f["needs_verification"])} | {pct(f["min_years"])} | {pct(f["sponsorship_blocked"])} |')
    lines += ['', '## Confusion vs astra (rows astra strong/stretch/reject; cols model)', '']
    for model, m in order:
        lines.append(f'- {model}: {m["vs_astra"]["confusion"]}; transport errors={m["errors"]}; lenient JSON extractions={m["lenient_extractions"]}; '
                     f'mean visible output ≈{m["mean_visible_output_tokens"] or 0:.0f} tok; mean {m["mean_seconds"]:.1f}s/call')
    lines += ['', '## Choice', '']
    if winner:
        why = ('cheapest (estimated $/1,000) model with ≥88% verdict agreement with astra and receipt agreement within 2 pts of astra'
               if qualified else 'NO model met both thresholds; this is the highest astra agreement')
        lines.append(f'Rule-based selection: **{winner}** — {why}.')
    lines += ['', '## Caveats', '',
              '- Cost is an estimate: OpenCode `generate/text` returns no token usage. Input = composed prompt chars / 3.5; output = visible chars / 3.5 '
              '+ a fixed per-model reasoning allowance (hidden reasoning is not observable); Zen list prices; DeepSeek peak 2×. '
              'The Go meter delta is the subscription\'s own accounting, but it is one account-wide meter (shared with any concurrent OpenCode use) '
              'with coarse resolution.',
              '- v3 astra received instructions as a developer message and schema-constrained decoding; here instructions are prepended to the '
              'same user prompt and the schema is appended and validated locally. The v3 prompt itself is byte-identical.',
              '- The subset is enriched for strong/stretch, so raw agreement is not population agreement (see weighted column). '
              'Receipts were partly used during earlier prompt design.', '']
    (OUT / 'BAKEOFF.md').write_text('\n'.join(lines), encoding='utf-8')
    return winner, qualified


def pilot_report(model):
    ids = read_json(OUT / 'pilot-ids.json')
    single = {**outputs('bakeoff', model)[0], **outputs('evaluation', model)[0]}
    labels, _ = outputs('pilot5', model)
    rows = phase_rows('pilot5', model)
    answered = [r for r in rows if r['status'] in ('ok', 'parse_failure')]
    agreement = {f: sum(p in labels and labels[p][f] == single[p][f] for p in ids) / len(ids) for f in [*base.ENUMS, 'min_years']}
    parse_rate = sum(r['status'] == 'parse_failure' for r in answered) / len(answered) if answered else 1
    single_rows = [r for r in phase_rows('bakeoff', model) + phase_rows('evaluation', model) if r['ids'][0] in ids]
    result = {'model': model, 'size': 5, 'postings': len(ids), 'valid': len(labels), 'calls': len(answered),
              'parse_failure_rate': parse_rate, 'agreement_with_single': agreement,
              'usd_per_1000_est': {'single': 1000 * sum(r.get('charged_usd', 0) for r in single_rows) / len(ids),
                                   'batch5': 1000 * sum(r.get('charged_usd', 0) for r in rows) / len(ids)},
              'passed': agreement['verdict'] >= .93 and parse_rate <= .05}
    save(OUT / 'pilot5-summary.json', result)
    return result


def report_numbers():
    """Facts for REPORT.md: spend per model/phase, meter, label counts."""
    rows = [r for r in jsonl(OUT / 'calls.jsonl') if r.get('model') in MODELS]
    spend = {}
    for r in rows:
        d = spend.setdefault(r['model'], {}).setdefault(r['phase'], {'calls': 0, 'ok': 0, 'parse_failure': 0, 'error': 0, 'usd_est': 0.0})
        d['calls'] += 1
        d[r['status']] = d.get(r['status'], 0) + 1
        d['usd_est'] += r.get('charged_usd', 0)
    totals = {m: {'usd_est': sum(x['usd_est'] for x in ph.values()),
                  'pct_of_monthly_budget': 100 * sum(x['usd_est'] for x in ph.values()) / MODELS[m][0],
                  'cap_usd': CAP_SHARE * MODELS[m][0]} for m, ph in spend.items()}
    entries, _ = evaluation()
    astra = astra_labels()
    gate = read_json(OUT / 'gate.json') if (OUT / 'gate.json').exists() else None
    final_eval = {}
    for e in entries:
        pid = e['posting_id']
        if pid in astra:
            final_eval[pid] = ('gpt-6-astra', astra[pid]['verdict'])
    for m in MODELS:
        for pid, v in outputs('evaluation', m)[0].items():
            final_eval.setdefault(pid, (m, v['verdict']))
    bulk = jsonl(OUT / 'bulk-labels.jsonl')
    meters = jsonl(OUT / 'meter.jsonl')
    data = {'spend': spend, 'totals': totals, 'meter_first': meters[0]['go'] if meters else None,
            'meter_last': meters[-1]['go'] if meters else None, 'meter_deltas': meter_deltas(),
            'evaluation': {'labelled': len(final_eval), 'of': len(entries),
                           'by_labeler': dict(Counter(x[0] for x in final_eval.values())),
                           'verdicts': dict(Counter(x[1] for x in final_eval.values()))},
            'bulk': {'labelled': len(bulk), 'of': 1500, 'verdicts': dict(Counter(r['verdict'] for r in bulk)),
                     'by_model': dict(Counter(r['labeler']['model'] for r in bulk)),
                     'by_stratum': {s: dict(Counter(r['verdict'] for r in bulk if r['stratum'] == s)) for s in ('entry_title', 'soft_entry', 'uniform')},
                     'weighted_verdict_share': {v: sum(r['weight'] for r in bulk if r['verdict'] == v) / sum(r['weight'] for r in bulk) for v in CLASSES} if bulk else None},
            'gate': gate}
    save(OUT / 'report-numbers.json', data)
    return data


# ---------------------------------------------------------------- commands

def cmd_check():
    ledger = Ledger()
    print(json.dumps({'deepseek_peak_now': peak('deepseek-v4.1-flash'),
                      'spent_usd_est': {m: round(ledger.spent(m), 4) for m in MODELS if ledger.spent(m)}}, indent=2))
    with server() as srv:
        print(json.dumps({'go_meter': srv.meter()}, indent=2))


def cmd_probe(models):
    """Synthetic fictional posting, no personal data: verifies routing, variants and JSON output."""
    synthetic = ('Policy: classify the fictional posting for a fictional new-grad software candidate with a US work '
                 'permit. Verdict strong if it is an entry-level US software role.\n\n<posting>\n'
                 + json.dumps({'employer': 'Example Corp', 'title': 'Software Engineer I', 'location': ['Remote, US'],
                               'posted': '2026-10-01', 'description': 'Entry-level role. 0-1 years experience. Python.'}) + '\n</posting>')
    ledger = Ledger()
    with server() as srv:
        snapshot(srv, 'all', 'probe', 'start')
        for model in models:
            result, kind = call(srv, model, synthetic, base.SCHEMA, ['probe'], False)
            row = {'call_id': f'probe-{model}-{time.time_ns()}', 'phase': 'probe', 'model': model, 'ids': ['probe'],
                   **result, 'finished_at': now()}
            append(row)
            ledger.settle(row['call_id'], row)
            print(f'{model:22} {result["status"]:14} {kind or "":12} {result.get("seconds")}s '
                  f'${result.get("charged_usd", 0):.5f} {(result.get("error") or "")[:200]}')
            if kind in ('auth', 'server_auth'):
                break
        print('go meter:', json.dumps(snapshot(srv, 'all', 'probe', 'end')))


def cmd_bakeoff(models, allow_peak):
    entries, posts = evaluation()
    astra = astra_labels()
    sel = bakeoff_ids(entries, astra)
    ctx = context()
    expected = {p: astra[p]['_prompt_sha256'] for p in sel['ids']}
    ledger = Ledger()
    groups = [[p] for p in sel['ids']]
    with server() as srv:
        # Sequential models so each Go meter delta is attributable to one model.
        for m in models:
            print(m, run_model(srv, m, 'bakeoff', groups, posts, ctx, ledger, False, 4, allow_peak, expected), flush=True)
            if GLOBAL_STOP.is_set():
                break
    summary = bakeoff_summary(models=[m for m in MODELS if phase_rows('bakeoff', m)])
    print('selection:', write_bakeoff(summary))


def cmd_run(model, phase, allow_peak, workers):
    ctx = context()
    ledger = Ledger()
    expected = None
    if phase == 'evaluation':
        entries, posts = evaluation()
        astra = astra_labels()
        groups = [[e['posting_id']] for e in entries if e['posting_id'] not in astra]
        # Same v3 shuffle-independent prompts; record their hashes for audit.
    else:
        manifest, posts = bulk_inputs()
        size = read_json(OUT / 'gate.json')['batch_size'] if (OUT / 'gate.json').exists() else 1
        ids = [e['posting_id'] for e in manifest['entries']]
        groups = [ids[i:i + size] for i in range(0, len(ids), size)]
    batch = len(groups[0]) > 1
    with server() as srv:
        state = run_model(srv, model, phase, groups, posts, ctx, ledger, batch, workers, allow_peak, expected)
    if phase == 'bulk':
        export_bulk(model, f'batch-{len(groups[0])}' if batch else 'single')
    print(model, phase, state)
    return state


def cmd_pilot(model, allow_peak):
    ctx = context()
    entries, posts = evaluation()
    single = {**outputs('bakeoff', model)[0], **outputs('evaluation', model)[0]}
    path = OUT / 'pilot-ids.json'
    if not path.exists():
        v2.freeze(path, random.Random(SEED + 1).sample(sorted(single), 100))
    ids = read_json(path)
    groups = [ids[i:i + 5] for i in range(0, len(ids), 5)]
    with server() as srv:
        state = run_model(srv, model, 'pilot5', groups, posts, ctx, Ledger(), True, 4, allow_peak)
    result = pilot_report(model)
    (OUT / 'PILOT-5.md').write_text(
        '# Batch-5 pilot vs single-call labels (same model)\n\n```json\n' + json.dumps(result, indent=2)
        + '\n```\n\nBatch prompt = exact v3 batch construction (descriptions capped at 6,000 chars, schema inlined, '
        'validated locally). Missing labels count as disagreement. Gate: verdict agreement ≥93% and parse failures ≤5%.\n',
        encoding='utf-8')
    if state == 'complete':
        v2.freeze(OUT / 'gate.json', {'model': model, 'mode': 'batch-5' if result['passed'] else 'single',
                                      'batch_size': 5 if result['passed'] else 1})
    print(state, json.dumps(result, indent=2))


def luna_bulk_positive_ids():
    """Bulk postings Luna labelled strong/stretch, in manifest order."""
    manifest = read_json(V3 / 'bulk-sample-1500.json')
    luna, _ = outputs('bulk', 'gpt-6-luna')
    return [e['posting_id'] for e in manifest['entries'] if luna.get(e['posting_id'], {}).get('verdict') in ('strong', 'stretch')]


def cmd_relabel_bulk(model, workers):
    """Second-opinion single-call labels (phase bulk-glm) on Luna's bulk strong/stretch postings."""
    ctx = context()
    _, posts = bulk_inputs()
    ids = luna_bulk_positive_ids()
    with server() as srv:
        state = run_model(srv, model, 'bulk-glm', [[p] for p in ids], posts, ctx, Ledger(), False, workers)
    labelled, parse = outputs('bulk-glm', model)
    print(model, 'bulk-glm', state, f'{len(labelled)}/{len(ids)} labelled')
    import bulk_final_v4
    print(json.dumps(bulk_final_v4.build()['counts'], indent=2))
    return state


def export_bulk(model, mode):
    manifest = read_json(V3 / 'bulk-sample-1500.json')
    path = OUT / 'bulk-labels.jsonl'
    existing = {r['posting_id'] for r in jsonl(path)}
    labels, _ = outputs('bulk', model)
    ctx_sha = digest(V3 / 'context-v3.md')
    for e in manifest['entries']:
        pid = e['posting_id']
        if pid in labels and pid not in existing:
            v2.append(path, {**labels[pid], **e, 'labeler': {
                'model': f'{PROVIDER}/{model}', 'variant': MODELS[model][3], 'transport': 'opencode generate/text',
                'context_sha256': ctx_sha, 'mode': mode}})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('check')
    p = sub.add_parser('probe')
    p.add_argument('--models', nargs='+', default=BAKEOFF_MODELS)
    p = sub.add_parser('bakeoff')
    p.add_argument('--models', nargs='+', default=BAKEOFF_MODELS)
    p.add_argument('--allow-peak', action='store_true')
    sub.add_parser('analyze-bakeoff')
    sub.add_parser('report')
    p = sub.add_parser('relabel-bulk')
    p.add_argument('--model', default='glm-5.3')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-rolling', type=float, default=METER_LIMITS['rolling'])
    p = sub.add_parser('run')
    p.add_argument('--model', required=True)
    p.add_argument('--phase', choices=['evaluation', 'bulk'], required=True)
    p.add_argument('--allow-peak', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-rolling', type=float, default=METER_LIMITS['rolling'],
                   help='Stop when the account-wide Go 5-hour meter reaches this percent')
    p = sub.add_parser('pilot')
    p.add_argument('--model', required=True)
    p.add_argument('--max-rolling', type=float, default=METER_LIMITS['rolling'])
    p.add_argument('--allow-peak', action='store_true')
    args = parser.parse_args()
    for m in getattr(args, 'models', None) or [getattr(args, 'model', None)]:
        if m and m not in MODELS:
            raise SystemExit(f'{m} is not an allowed paid Go model')
    if getattr(args, 'max_rolling', None) is not None:
        METER_LIMITS['rolling'] = min(args.max_rolling, 50)  # never more than half a 5-hour window
    with lock():
        if args.action == 'check':
            cmd_check()
        elif args.action == 'probe':
            cmd_probe(args.models)
        elif args.action == 'bakeoff':
            cmd_bakeoff(args.models, args.allow_peak)
        elif args.action == 'analyze-bakeoff':
            print(write_bakeoff(bakeoff_summary(models=[m for m in MODELS if phase_rows('bakeoff', m)])))
        elif args.action == 'run':
            cmd_run(args.model, args.phase, args.allow_peak, args.workers)
        elif args.action == 'pilot':
            cmd_pilot(args.model, args.allow_peak)
        elif args.action == 'relabel-bulk':
            cmd_relabel_bulk(args.model, args.workers)
        elif args.action == 'report':
            print(json.dumps(report_numbers(), indent=2))


if __name__ == '__main__':
    main()
