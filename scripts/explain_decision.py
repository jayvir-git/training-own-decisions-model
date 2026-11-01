"""Explain verdict probabilities by removing one in-state sentence at a time.

The public ``explain_posting(posting, decide)`` API is scorer-agnostic. The CLI uses
``GGUFClient`` to rebuild serving states and query llama.cpp without PyTorch:

  .venv/Scripts/python.exe -B scripts/gpu_lock.py -- \
    .venv/Scripts/python.exe -B scripts/explain_decision.py \
    --split data/training/val.jsonl --n 70

Deltas are ``base probability - probability with the sentence removed``. Thus a
positive reject delta means that the sentence pushes toward reject; a positive good
delta means that it pushes toward strong/stretch. Test splits are deliberately not
accepted by the CLI.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import build_decision_dataset as bdd  # noqa: E402

DEFAULT_BINARY = Path.home() / ".unsloth/llama.cpp/build/bin/Release/llama-server.exe"
DEFAULT_MODEL = REPO / "outputs/full-e3/export-gguf-q8/native/gguf/model-Q8_0.gguf"
DROPPED_QUESTIONS = ("needs_verification",)
# Punctuation, line breaks, and common explicit bullets are boundaries. This is
# intentionally a transparent splitter rather than an NLP sentence tokenizer.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|[\r\n]+|[ \t]*[•▪◦*][ \t]*")


def _sentence_spans(text):
    """Return stripped content spans, keeping removal positions unambiguous."""
    spans, start = [], 0
    for match in _BOUNDARY.finditer(text or ""):
        left, right = start, match.start()
        while left < right and (text[left].isspace()):
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = match.end()
    left, right = start, len(text or "")
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        spans.append((left, right))
    return spans


def split_sentences(text) -> list[str]:
    """Split description text on punctuation, newlines, and common bullets."""
    return [text[start:end] for start, end in _sentence_spans(text or "")]


def _normal(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _verdict_probabilities(result):
    probs = result["verdict"]
    if set(probs) != set(bdd.VERDICTS):
        raise ValueError("decide must return strong, stretch and reject verdict probabilities")
    values = {name: float(probs[name]) for name in bdd.VERDICTS}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError("verdict probabilities must be finite values between zero and one")
    return values


def _verdict(probs):
    return max(bdd.VERDICTS, key=probs.__getitem__)


def explain_posting(posting: dict, decide) -> dict:
    """Occlude each description sentence that survives the serving token budget.

    ``decide`` receives a new posting dictionary and must return a mapping with a
    ``verdict`` probability dictionary. Sentences omitted by ``make_state`` are
    reported with zero deltas and are not scored, avoiding false attributions to
    text the model never saw.
    """
    description = posting.get("description") or ""
    base = _verdict_probabilities(decide(posting))
    base_good = base["strong"] + base["stretch"]
    spans = _sentence_spans(description)

    # Import lazily so the lightweight public function needs only the stdlib and
    # the dataset builder; GGUFClient supplies the real tokenizer in production.
    tokenizer = _shared_text_tokenizer()
    count = lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
    kept_body, _ = bdd.truncate_posting(description, 950, count)
    normalized_kept = _normal(kept_body)
    sentences = []
    for index, (start, end) in enumerate(spans):
        text = description[start:end]
        in_state = bool(_normal(text)) and _normal(text) in normalized_kept
        removed_probs = base
        if in_state:
            occluded = dict(posting, description=description[:start] + description[end:])
            removed_probs = _verdict_probabilities(decide(occluded))
        removed_good = removed_probs["strong"] + removed_probs["stretch"]
        sentences.append({
            "index": index,
            "text": text,
            "in_state": in_state,
            "delta_reject": base["reject"] - removed_probs["reject"],
            "delta_good": base_good - removed_good,
            "verdict_if_removed": _verdict(removed_probs),
        })
    scored = [sentence for sentence in sentences if sentence["in_state"]]
    top_reject = sorted((s for s in scored if s["delta_reject"] > 0),
                        key=lambda s: s["delta_reject"], reverse=True)[:3]
    top_good = sorted((s for s in scored if s["delta_good"] > 0),
                      key=lambda s: s["delta_good"], reverse=True)[:3]
    return {"base": base, "sentences": sentences,
            "top_reject_drivers": top_reject, "top_good_drivers": top_good}


_TOKENIZER = None


def _shared_text_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(bdd.DEFAULT_TOKENIZER, local_files_only=True)
    return _TOKENIZER


class GGUFClient:
    """Context-managed local llama-server plus the exact posting -> state adapter."""

    def __init__(self, model=DEFAULT_MODEL, binary=DEFAULT_BINARY, profile=bdd.PROFILE,
                 log_path=REPO / "outputs/explanations/llama-server.log", startup_timeout=300):
        self.model = Path(model)
        self.binary = Path(binary)
        self.profile = Path(profile)
        self.log_path = Path(log_path)
        self.startup_timeout = startup_timeout
        self.process = None
        self.base_url = None
        self._log = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.tokenizer = None
        self.profile_text = None

    def __enter__(self):
        if not self.binary.is_file() or not self.model.is_file():
            raise FileNotFoundError("GGUF model or llama-server binary is missing")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        command = [str(self.binary), "-m", str(self.model), "--alias", "local-decision",
                   "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
                   "-c", "4096", "-b", "4096", "-ub", "4096", "-ngl", "99"]
        self._log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + self.startup_timeout
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama-server exited; see {self.log_path}")
                try:
                    self._request("/health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"llama-server did not become ready; see {self.log_path}")
                    time.sleep(.5)
            return self
        except Exception:
            self.stop()
            raise

    def __exit__(self, exc_type, exc, traceback):
        self.stop()

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        self.base_url = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def _request(self, path, body=None):
        if self.base_url is None:
            raise RuntimeError("GGUFClient must be entered before making requests")
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body, allow_nan=False).encode("utf-8") if body is not None else None,
            headers={"Content-Type": "application/json"})
        with self._opener.open(request, timeout=120) as response:
            return json.load(response)

    def state_for(self, posting):
        if self.tokenizer is None:
            self.tokenizer = _shared_text_tokenizer()
            self.profile_text = self.profile.read_text(encoding="utf-8")
        converted = {"title": posting["title"], "employer": posting["company"],
                     "locations": [posting["location"]], "description": posting["description"]}
        return bdd.make_state(
            self.profile_text, converted, 950,
            lambda text: len(self.tokenizer(text, add_special_tokens=False)["input_ids"]))

    def decide(self, posting):
        """Build a serving state, call /v1/systemone, and decode typed answers."""
        state, truncated = self.state_for(posting)
        response = self.decide_state(state)
        verdict = response["answers"]["verdict"]["probabilities"]
        return {"verdict": {name: float(verdict[name]) for name in bdd.VERDICTS},
                "state_truncated": truncated}

    def decide_state(self, state):
        questions = {name: question for name, question in bdd.QUESTIONS.items()
                     if name not in DROPPED_QUESTIONS}
        response = self._request("/v1/systemone", {
            "model": "local-decision", "state": state, "questions": questions})
        # Same decoding rules as export_model.native_prediction, including option checks.
        for name, question in questions.items():
            answer = response["answers"][name]
            if question["type"] == "noul":
                value = float(answer["noul"])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"invalid native probability for {name}")
            else:
                probabilities = answer["probabilities"]
                options = (sorted(question["criteria"]) if question["type"] == "choice"
                           else [str(i) for i in range(len(question["criteria"]))])
                if set(options) != set(probabilities):
                    raise ValueError(f"native probability options differ for {name}")
        return response


def _as_posting(row):
    if isinstance(row.get("posting"), dict):
        row = row["posting"]
    posting = {
        "title": row.get("title") or "",
        "company": row.get("company") or row.get("employer") or "",
        "location": row.get("location") or "; ".join(map(str, row.get("locations") or [])),
        "description": row.get("description") or "",
    }
    return posting if posting["description"].strip() else None


def load_postings(split_path, rows, source_paths):
    """Resolve state-only split rows by posting/source ID from authorized local JSONL."""
    direct = {}
    wanted_ids, wanted_sources = set(), set()
    for row in rows:
        posting = _as_posting(row)
        if posting:
            direct[row["posting_id"]] = posting
        else:
            wanted_ids.add(row["posting_id"])
            if row.get("source_id"):
                wanted_sources.add(row["source_id"])
    found = {}
    for path in source_paths:
        path = Path(path)
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                keys = {source.get("posting_id"), source.get("source_id")}
                if not (keys & wanted_ids or keys & wanted_sources):
                    continue
                posting = _as_posting(source)
                if posting:
                    for key in keys:
                        if key:
                            found[key] = posting
    result, missing = [], []
    for row in rows:
        posting = direct.get(row["posting_id"]) or found.get(row["posting_id"]) or found.get(row.get("source_id"))
        if posting is None:
            missing.append(row["posting_id"])
        else:
            result.append((row, posting))
    if missing:
        raise ValueError("could not resolve postings for: " + ", ".join(map(str, missing[:10])))
    return result


_CATEGORY_PATTERNS = {
    "eeo": re.compile(r"equal (employment )?opportunity|without regard to|affirmative action|reasonable accommodation|fair chance", re.I),
    "benefits": re.compile(r"401\(?k|paid time off|\bpto\b|dental|vision insurance|parental leave|benefits|perks|wellness|health insurance", re.I),
    "years": re.compile(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s*\(\d+\))?\s*\+?\s*(?:(?:-|–|—|\bto\b)\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*)?(?:years?|yrs?)\b|\byears? of (?:relevant |professional |related )?experience\b|\bminimum of \d+\b", re.I),
    "sponsorship": re.compile(r"sponsor|visa|h-?1b|work authoriz|unrestricted authoriz|e-?verify|green card", re.I),
    "clearance_citizenship": re.compile(r"clearance|citizenship|\bcitizen\b|permanent residen|security investigation|polygraph", re.I),
    "seniority": re.compile(r"\bsenior\b|\bstaff engineer\b|\bprincipal\b|\blead\b|\bmanager\b|\bnew grad\b|\bjunior\b|\bentry.level\b", re.I),
    "boilerplate": re.compile(r"privacy (policy|notice)|pay transparency|candidate privacy|terms of use|cookie|recruitment fraud|e-?verify participation", re.I),
    "company_description": re.compile(r"about (us|the company)|who we are|our (company|mission|story)|we are (?:a|an|the) (?:global|leading|premier|world)|join our (?:team|mission)", re.I),
}
GATE_CATEGORIES = {"years", "sponsorship", "clearance_citizenship"}
SUSPICIOUS_CATEGORIES = {"boilerplate", "company_description", "benefits", "eeo"}


def categorize(text):
    return [name for name, pattern in _CATEGORY_PATTERNS.items() if pattern.search(text)]


def summarize(results):
    rejects = [r for r in results if _verdict(r["base"]) == "reject"]
    top_rows, gate_rejects, suspicious = [], 0, []
    category_counts = Counter()
    for result in rejects:
        scored = [s for s in result["sentences"] if s["in_state"]]
        if not scored:
            continue
        top = max(scored, key=lambda s: s["delta_reject"])
        categories = categorize(top["text"]) or ["other"]
        top_rows.append((result, top, categories))
        category_counts.update(categories)
        if GATE_CATEGORIES.intersection(categories):
            gate_rejects += 1
        if SUSPICIOUS_CATEGORIES.intersection(categories):
            suspicious.append({"posting_id": result["posting_id"], "sentence_index": top["index"],
                               "categories": categories, "delta_reject": top["delta_reject"],
                               "text": top["text"]})
    all_sentences = [s for r in results for s in r["sentences"]]
    scored = [s for s in all_sentences if s["in_state"]]
    return {
        "postings": len(results),
        "base_verdicts": dict(Counter(_verdict(r["base"]) for r in results)),
        "sentences": len(all_sentences),
        "scored_sentences": len(scored),
        "out_of_state_sentences": len(all_sentences) - len(scored),
        "reject_postings_with_top_driver": len(top_rows),
        "rejects_with_positive_top_driver": sum(top["delta_reject"] > 0 for _, top, _ in top_rows),
        "rejects_without_positive_top_driver": sum(top["delta_reject"] <= 0 for _, top, _ in top_rows),
        "top_driver_category_counts": dict(sorted(category_counts.items())),
        "rejects_top_driver_gate_count": gate_rejects,
        "rejects_top_driver_gate_fraction": gate_rejects / len(top_rows) if top_rows else None,
        "suspicious_reject_top_driver_count": len(suspicious),
        "suspicious_reject_top_driver_fraction": len(suspicious) / len(top_rows) if top_rows else None,
        "suspicious_cases": suspicious,
    }


def _default_sources():
    return [REPO / "data/processed/postings.jsonl",
            REPO / "data/raw/agent-export-20261028-2111/postings.jsonl"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", type=Path, required=True, help="validation JSONL; test.jsonl is rejected")
    parser.add_argument("--n", type=int, default=20, help="number of postings from the start of the split")
    parser.add_argument("--start", type=int, default=0, help="zero-based row offset")
    parser.add_argument("--out", type=Path, help="default outputs/explanations/<split>.jsonl")
    parser.add_argument("--postings", type=Path, action="append", help="source posting JSONL; may be repeated")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args(argv)
    if args.split.name == "test.jsonl":
        parser.error("test splits are not supported; use validation data only")
    if args.n <= 0 or args.start < 0:
        parser.error("--n must be positive and --start nonnegative")
    out = args.out or REPO / "outputs/explanations" / f"{args.split.stem}.jsonl"
    out = out.resolve()
    if not out.is_relative_to(REPO / "outputs"):
        parser.error("output must be below outputs/")
    summary_path = out.with_suffix(".summary.json")
    rows = bdd.read_jsonl(args.split)[args.start:args.start + args.n]
    if not rows:
        parser.error("selected split range is empty")
    pairs = load_postings(args.split, rows, args.postings or _default_sources())
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    log_path = out.parent / f"{out.stem}-llama-server.log"
    with GGUFClient(args.model, args.binary, log_path=log_path) as client:
        for number, (row, posting) in enumerate(pairs, 1):
            state, _ = client.state_for(posting)
            if state != row.get("state"):
                raise ValueError(f"rebuilt state differs from split state for {row['posting_id']}")
            result = {"posting_id": row["posting_id"], **explain_posting(posting, client.decide)}
            results.append(result)
            with out.open("a" if number > 1 else "w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            if number == 1 or number % 10 == 0 or number == len(pairs):
                print(f"explained {number}/{len(pairs)} postings", file=sys.stderr, flush=True)
    summary = summarize(results)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "suspicious_cases"}, indent=2))


if __name__ == "__main__":
    main()
