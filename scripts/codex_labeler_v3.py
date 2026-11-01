"""V3 isolated CLI transport. Stream errors, cancel siblings, never retry a call."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time

import codex_labeler_v2 as base

OUT = base.REPO / 'data/labeling/v3'
STOP = threading.Event()
GUARD = threading.RLock()
ACTIVE = set()
STOP_REASON = None
# Reserve substantially more than the maximum allowed serialized request bytes.
# Unknown/interrupted usage keeps the reservation instead of being counted as zero.
RESERVATION = 131072
MAX_REQUEST_BYTES = 96000



CALL_TIMEOUT = 180  # seconds; overridable by callers (e.g. codex_tiebreak_v4 --call-timeout)

def stop(reason):
    global STOP_REASON
    with GUARD:
        if not STOP.is_set():
            STOP_REASON = reason
        STOP.set()
        for process in list(ACTIVE):
            if process.poll() is None:
                process.kill()


def is_limit(text):
    return bool(re.search(r'usage[ _-]*limit|rate[ _-]*limit|\b429\b|too many requests', text, re.I))


def command(cwd, schema):
    # Reuse all existing isolation flags; redirect only repo-local artifacts.
    args = base.command(cwd, schema)
    args = [arg.replace(str(base.OUT), str(OUT)) for arg in args]
    # Codex structured-output transport requires an object root. Batch output is
    # the requested bare array, so validate that schema locally instead.
    if json.loads(Path(schema).read_text(encoding='utf-8')).get('type') == 'array':
        index = args.index('--output-schema')
        del args[index:index + 2]
    return args


def validate_batch(value, ids):
    if not isinstance(value, list) or len(value) != len(ids):
        raise ValueError('Batch must be an array with exactly one object per ID')
    seen = set()
    for item in value:
        if not isinstance(item, dict) or 'posting_id' not in item:
            raise ValueError('Missing posting_id')
        pid = item['posting_id']
        if pid not in ids or pid in seen:
            raise ValueError('Duplicate or unexpected posting_id')
        seen.add(pid)
        base.validate({k: v for k, v in item.items() if k != 'posting_id'})
    return value


def invoke(prompt, artifact, schema, ids, batch=False):
    artifact.mkdir(parents=True, exist_ok=False)
    schema_path = artifact / 'schema.json'
    schema_path.write_text(json.dumps(schema), encoding='utf-8')
    temp_root = OUT / 'empty-cwds'
    temp_root.mkdir(exist_ok=True)
    cwd = Path(tempfile.mkdtemp(prefix='call-', dir=temp_root))
    args = command(cwd, schema_path)
    (artifact / 'command.json').write_text(json.dumps(args), encoding='utf-8')
    if len((prompt + base.INSTRUCTIONS + json.dumps(schema)).encode()) > MAX_REQUEST_BYTES:
        raise ValueError('Request exceeds conservative reservation byte envelope')
    env = {k: v for k, v in os.environ.items()
           if not any(x in k.upper() for x in ['API_KEY', 'AUTH_TOKEN'])
           and not k.upper().startswith(('T3_', 'CODEX_THREAD', 'CODEX_INTERNAL'))}
    start = time.monotonic()
    with GUARD:
        if STOP.is_set():
            return {'status': 'cancelled', 'usage': None, 'error': STOP_REASON}
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                errors='replace', cwd=cwd, env=env,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        ACTIVE.add(proc)
    events, errors = [], []

    def drain(stream, filename, target, structured):
        with (artifact / filename).open('w', encoding='utf-8') as handle:
            for line in stream:
                handle.write(line)
                handle.flush()
                target.append(line)
                event = {}
                if structured:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        pass
                # Do not interpret a posting/answer mentioning HTTP 429 as a provider error.
                error_line = not structured or event.get('type') in ('error', 'turn.failed')
                if error_line and is_limit(line):
                    stop('usage_or_rate_limit')
                elif error_line and ('reconnecting' in line.lower() or event.get('type') in ('error', 'turn.failed')):
                    stop('provider_error_no_retry')
                elif structured and event.get('type') == 'item.started':
                    if event.get('item', {}).get('type') not in ('agent_message', 'reasoning'):
                        stop('unexpected_tool')
    readers = [threading.Thread(target=drain, args=(proc.stdout, 'events.jsonl', events, True)),
               threading.Thread(target=drain, args=(proc.stderr, 'stderr.txt', errors, False))]
    for reader in readers:
        reader.start()
    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
        proc.wait(timeout=CALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        stop('timeout_unknown_usage')
        proc.wait()
    except BrokenPipeError:
        proc.wait()
    finally:
        for reader in readers:
            reader.join()
        with GUARD:
            ACTIVE.discard(proc)
    parsed = []
    for line in events:
        try:
            parsed.append(json.loads(line))
        except ValueError:
            pass
    usages = [e['usage'] for e in parsed if e.get('type') == 'turn.completed']
    usage = {k: sum(u.get(k, 0) for u in usages) for k in ('input_tokens', 'cached_input_tokens', 'output_tokens')} if usages else None
    result = {'status': 'failure', 'usage': usage, 'exit_code': proc.returncode,
              'seconds': round(time.monotonic() - start, 3), 'error': ''.join(errors)[-2000:]}
    messages, tools = [], []
    for e in parsed:
        if e.get('type') == 'item.completed':
            item = e.get('item', {})
            if item.get('type') == 'agent_message':
                messages.append(item.get('text', ''))
            elif item.get('type') != 'reasoning':
                tools.append(item)
    if len(usages) > 1 or tools:
        stop('unexpected_multiple_turns_or_tools')
    if proc.returncode == 0 and messages and not tools and usage:
        try:
            value = json.loads(messages[-1])
            value = validate_batch(value, ids) if batch else base.validate(value)
            result.update(status='ok', output=value)
            result.pop('error', None)
        except (ValueError, TypeError) as exc:
            result.update(status='parse_failure', error=str(exc))
    if usage and usage['input_tokens'] > RESERVATION:
        stop('reservation_envelope_exceeded')
    if result['status'] == 'failure':
        stop(STOP_REASON or 'call_failure_unknown_usage')
    return result
