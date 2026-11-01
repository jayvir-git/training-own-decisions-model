"""Counterfactual rule tests for the exported typed decision model.

The runner does not load PyTorch.  It starts the local decision-aware
llama.cpp server, builds each state through ``build_decision_dataset.make_state``
and sends the resulting state/questions to ``/v1/systemone``.

Outputs are intentionally kept below outputs/: the states contain the private
candidate profile and held-out posting text.
"""
from __future__ import annotations

import argparse
import copy
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
sys.path.insert(0, str(REPO / "scripts"))
import build_decision_dataset as bdd  # noqa: E402


YEAR_RE = re.compile(
    r"(?P<low>\d+)\s*(?:(?:-|–|—|to)\s*(?P<high>\d+)|\+|or\s+more)?\s*"
    r"(?:years?|yrs?)\b",
    re.I,
)
REQUIRED_CUE = re.compile(
    r"\b(?:require(?:s|d)?|minimum|at\s+least|must\s+have|need(?:s|ed)?|"
    r"qualifications?|required\s+years?)\b",
    re.I,
)
PREFERRED_CUE = re.compile(r"\bprefer(?:red|ably)?\b|nice\s+to\s+have|plus", re.I)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

EDIT_SENTENCES = {
    "years_up": "Requires 5+ years of professional software experience.",
    "preferred": "5+ years of relevant experience preferred, but not required.",
    "sponsorship": "We are unable to sponsor employment visas now or in the future.",
    "clearance": "Must be a U.S. citizen and able to obtain a security clearance.",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sentence_spans(text: str):
    """Yield (start, end, stripped sentence) without losing replacement offsets."""
    start = 0
    for match in SENTENCE_SPLIT.finditer(text):
        end = match.start()
        part = text[start:end]
        if part.strip():
            yield start, end, part.strip()
        start = match.end()
    part = text[start:]
    if part.strip():
        yield start, len(text), part.strip()


def required_years(text: str) -> dict | None:
    """Return the first non-preferred explicit required-years statement.

    The policy uses the lower bound.  This helper is deliberately conservative:
    it only treats a sentence as a gate when it contains a requirement cue.
    """
    for start, end, sentence in sentence_spans(text or ""):
        match = YEAR_RE.search(sentence)
        if not match or PREFERRED_CUE.search(sentence) or not REQUIRED_CUE.search(sentence):
            continue
        return {"low": int(match.group("low")), "sentence": sentence,
                "start": start, "end": end}
    return None


def prepend_sentence(posting: dict, sentence: str) -> dict:
    """Return a posting with a controlled sentence at the description head."""
    out = copy.deepcopy(posting)
    old = (out.get("description") or "").strip()
    out["description"] = sentence if not old else f"{sentence}\n\n{old}"
    return out


def replace_required_year_sentence(posting: dict, replacement: str) -> tuple[dict, bool]:
    """Replace one explicit required-years sentence, preserving the rest."""
    out = copy.deepcopy(posting)
    text = out.get("description") or ""
    found = required_years(text)
    if not found:
        return out, False
    out["description"] = text[:found["start"]] + replacement + text[found["end"]:]
    return out, True


def _benefit_reorder(text: str) -> str:
    spans = [(a, b, s) for a, b, s in sentence_spans(text)
             if re.search(r"benefit|health|dental|vision|401\s*\(?k\)?|paid time", s, re.I)]
    if len(spans) < 2:
        return text
    first, second = spans[0], spans[1]
    # Preserve any intervening sentence while swapping only the two benefit
    # sentences; this keeps the control local even when benefits are separated.
    return text[:first[0]] + text[second[0]:second[1]] + text[first[1]:second[0]] + text[first[0]:first[1]] + text[second[1]:]


def control_edit(posting: dict, index: int) -> tuple[dict, str]:
    """Apply one deterministic irrelevant control edit."""
    out = copy.deepcopy(posting)
    kind = ("company", "benefits", "eeo", "typo")[index % 4]
    if kind == "company":
        out["employer"] = "Northstar Labs"
        out["normalized_employer"] = "northstar labs"
        return out, kind
    if kind == "benefits":
        before = out.get("description") or ""
        after = _benefit_reorder(before)
        if after == before:
            after = "Benefits include health, dental, and vision coverage.\n\n" + before
        out["description"] = after
        return out, kind
    if kind == "eeo":
        out["description"] = "We are an equal opportunity employer.\n\n" + (out.get("description") or "").lstrip()
        return out, kind
    text = out.get("description") or ""
    match = re.search(r"\bsoftware\b", text, re.I)
    out["description"] = (text[:match.start()] + "softwrae" + text[match.end():]
                           if match else "Please review teh role details.\n\n" + text)
    return out, kind


def edit_posting(posting: dict, family: str, index: int = 0) -> tuple[dict, str]:
    """Pure family edit function used by the runner and unit tests."""
    if family == "years_up":
        return prepend_sentence(posting, EDIT_SENTENCES[family]), EDIT_SENTENCES[family]
    if family == "preferred":
        return prepend_sentence(posting, EDIT_SENTENCES[family]), EDIT_SENTENCES[family]
    if family == "sponsorship":
        return prepend_sentence(posting, EDIT_SENTENCES[family]), EDIT_SENTENCES[family]
    if family == "clearance":
        return prepend_sentence(posting, EDIT_SENTENCES[family]), EDIT_SENTENCES[family]
    if family == "years_down":
        edited, changed = replace_required_year_sentence(
            posting, "Requires 1-2 years of relevant experience.")
        return edited, "Requires 1-2 years of relevant experience." if changed else ""
    if family == "invariance":
        return control_edit(posting, index)
    raise ValueError(f"unknown family: {family}")


def score_case(family: str, base: dict, edited: dict) -> tuple[bool, str]:
    """Score one typed counterfactual; returns (pass, compact reason)."""
    bp, ep = base["pred"], edited["pred"]
    if family == "years_up":
        ok = ep.get("verdict") == "reject" and ep.get("min_years") == 3
        return ok, "verdict=reject and min_years=5+" if ok else "missing required years movement"
    if family == "years_down":
        ok = ep.get("min_years") == 1 and ep.get("verdict") != "reject"
        return ok, "min_years=1 and verdict remains actionable" if ok else "years-down response still rejects or has wrong bucket"
    if family == "preferred":
        ok = ep.get("verdict") == bp.get("verdict") and ep.get("min_years") == bp.get("min_years")
        return ok, "preferred edit invariant" if ok else "preferred years changed a typed answer"
    if family == "sponsorship":
        ok = ep.get("sponsorship_blocked") is True and ep.get("verdict") == "reject"
        return ok, "sponsorship=true and verdict=reject" if ok else "sponsorship gate not fully reflected"
    if family == "clearance":
        ok = ep.get("clearance_or_citizenship_required") is True and ep.get("verdict") == "reject"
        return ok, "clearance=true and verdict=reject" if ok else "clearance gate not fully reflected"
    if family == "invariance":
        ok = ep.get("verdict") == bp.get("verdict")
        return ok, "verdict invariant" if ok else "verdict flipped on irrelevant edit"
    raise ValueError(f"unknown family: {family}")


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = passes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _parse_embedded_posting(state: str, posting_id: str) -> dict:
    body = state.split("# Posting", 1)[1]
    header, _, description = body.partition("\n\n")
    fields = {}
    for line in header.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip().lower()] = value.strip()
    return {"posting_id": posting_id, "title": fields.get("title", ""),
            "employer": fields.get("employer", ""),
            "normalized_employer": fields.get("employer", "").lower(),
            "locations": [fields.get("location", "")], "description": description}


def load_cases(val_path: Path, val2_path: Path) -> list[dict]:
    """Load only the two declared held-out sources and their raw text."""
    rows = [("val", r) for r in read_jsonl(val_path)] + [("val2", r) for r in read_jsonl(val2_path)]
    needed = {r["posting_id"] for _, r in rows if not r["posting_id"].startswith("export:")}
    postings = {}
    processed = REPO / "data" / "processed" / "postings.jsonl"
    if needed:
        with processed.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                if p.get("posting_id") in needed:
                    postings[p["posting_id"]] = p
    cases = []
    for source, row in rows:
        posting = postings.get(row["posting_id"])
        if posting is None or not (posting.get("description") or "").strip():
            posting = _parse_embedded_posting(row["state"], row["posting_id"])
        if not (posting.get("description") or "").strip():
            continue
        cases.append({"source": source, "row": row, "posting": posting})
    return cases


def _prediction(questions: dict, response: dict) -> dict:
    answers = response["answers"]
    pred, probabilities = {}, {}
    for name, question in questions.items():
        answer = answers[name]
        if question["type"] == "noul":
            value = float(answer["noul"])
            pred[name] = value >= 0.5
            probabilities[name] = {"true": value, "false": 1 - value}
        else:
            probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
            choice = max(probs, key=probs.get)
            pred[name] = int(choice) if question["type"] == "score" else choice
            probabilities[name] = probs
    return {"pred": pred, "probabilities": probabilities}


class GGUFClient:
    def __init__(self, server: Path, model: Path, out: Path):
        self.server = server
        self.model = model
        self.out = out
        self.process = None
        self.base = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __enter__(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        log = (self.out / "llama-server.log").open("w", encoding="utf-8")
        cmd = [str(self.server), "-m", str(self.model), "--alias", "local-decision",
               "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
               "-c", "4096", "-b", "4096", "-ub", "4096", "-ngl", "99"]
        self.process = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 300
        try:
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError("llama-server exited; see outputs/rule-tests/llama-server.log")
                try:
                    self.request("/health")
                    return self
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("llama-server did not become ready within 300 seconds")
                    time.sleep(0.5)
        except Exception:
            self.__exit__(None, None, None)
            raise

    def request(self, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.base + path, data=data,
                                         headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=180) as response:
            return json.load(response)

    def predict(self, state: str, questions: dict) -> dict:
        return _prediction(questions, self.request(
            "/v1/systemone", {"model": "local-decision", "state": state, "questions": questions}))

    def __exit__(self, *_):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def _build_state(profile: str, posting: dict, tokenizer) -> tuple[str, bool]:
    return bdd.make_state(profile, posting, 950,
                          lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def _base_agreement(cases: list[dict]) -> dict:
    counts = {}
    for field in ("verdict", "sponsorship_blocked", "clearance_or_citizenship_required", "min_years"):
        compared = matches = 0
        for case in cases:
            gold = case["row"].get("gold", {})
            if field in gold and field in case["base"]["pred"]:
                compared += 1
                matches += case["base"]["pred"][field] == gold[field]
        counts[field] = {"matches": matches, "n": compared,
                         "agreement": matches / compared if compared else None}
    return counts


def _select(cases: list[dict], family: str, max_per: int) -> list[dict]:
    selected = []
    for case in cases:
        p = case["posting"].get("description") or ""
        pred = case["base"]["pred"]
        if family == "years_up" and pred.get("verdict") in {"strong", "stretch"}:
            selected.append(case)
        elif family == "years_down":
            req = required_years(p)
            if (req and req["low"] >= 3 and pred.get("verdict") == "reject"
                    and pred.get("sponsorship_blocked") is False
                    and pred.get("clearance_or_citizenship_required") is False
                    and pred.get("min_years", 0) >= 2):
                selected.append(case)
        elif family == "preferred":
            req = required_years(p)
            if pred.get("verdict") in {"strong", "stretch"} and (not req or req["low"] <= 2):
                selected.append(case)
        elif family == "sponsorship":
            if (pred.get("verdict") in {"strong", "stretch"}
                    and pred.get("sponsorship_blocked") is False):
                selected.append(case)
        elif family == "clearance":
            if (pred.get("verdict") in {"strong", "stretch"}
                    and pred.get("clearance_or_citizenship_required") is False):
                selected.append(case)
        elif family == "invariance":
            selected.append(case)
    return selected if not max_per else selected[:max_per]


def _write_report(summary: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = summary["base"]
    lines = [
        "# Counterfactual rule tests",
        "",
        "This report tests behavioural movement on held-out postings, not ordinary accuracy. Each edit is deterministic and states are rebuilt with the 950-token posting budget; inserted gate sentences were checked for presence after truncation.",
        "",
        "## Data and baseline",
        "",
        f"The run used **{summary['n_cases']}** unique postings: {summary['sources']['val']} from `data/training/val.jsonl` and {summary['sources']['val2']} from `data/experiments/export-v3/val2.jsonl`. No test split was read. The unedited verdict BASE agreement with the supplied gold labels was **{base['verdict']['matches']}/{base['verdict']['n']} ({base['verdict']['agreement']:.1%})**. Auxiliary agreement is available only for the 70 `val` rows: sponsorship {base['sponsorship_blocked']['matches']}/{base['sponsorship_blocked']['n']}, clearance {base['clearance_or_citizenship_required']['matches']}/{base['clearance_or_citizenship_required']['n']}, and min-years {base['min_years']['matches']}/{base['min_years']['n']}.",
        "",
        "The baseline is a starting-point agreement check, not evidence that the model follows the policy. The counterfactual pass rates below are the primary results.",
        "",
        "## Expectations and results",
        "",
        "| Family | Frozen-policy expectation | n | Pass rate (95% Wilson interval) |",
        "|---|---|---:|---:|",
    ]
    expectations = {
        "years_up": "5+ required years -> `reject`, min-years `5+`",
        "years_down": "3+ required years replaced by 1–2 -> min-years `1–2`, not reject on years alone",
        "preferred": "5+ preferred/not required leaves verdict and min-years unchanged",
        "sponsorship": "explicit no sponsorship -> sponsorship `true`, `reject`",
        "clearance": "citizenship/clearance sentence -> clearance `true`, `reject`",
        "invariance": "irrelevant edit leaves verdict unchanged",
    }
    for name, result in summary["families"].items():
        interval = result["interval"]
        rate = "n/a" if not result["n"] else f"{result['pass_rate']:.1%} ({interval[0]:.1%}–{interval[1]:.1%})"
        lines.append(f"| {name} | {expectations[name]} | {result['n']} | {rate} |")
    lines += ["", "## Interpretation and typical failures", "",
              f"The clearest learned behaviour is the two explicit hard-gate families: sponsorship and clearance both passed {summary['families']['sponsorship']['passes']}/{summary['families']['sponsorship']['n']} and {summary['families']['clearance']['passes']}/{summary['families']['clearance']['n']}. Years-up movement passed {summary['families']['years_up']['pass_rate']:.1%}, but the preferred-years control passed only {summary['families']['preferred']['pass_rate']:.1%}; the model is therefore sensitive to years language but does not reliably distinguish preferred from required. Years-down is a small, conservative proxy sample ({summary['families']['years_down']['n']} cases) and is correspondingly uncertain. The retained invariance controls were stable at {summary['families']['invariance']['pass_rate']:.1%}, with {summary['families']['invariance']['omitted']} controls excluded when the irrelevant edit did not survive state construction.", "",
              "A pass means the typed answer met the family-specific expectation; it does not prove causal understanding. A failure can be a missed gate, a wrong score bucket, or an over-sensitive verdict boundary. For years-down, selection is intentionally conservative: base prediction is reject, min-years is 3+, and the model has no predicted sponsorship/clearance gate; this is a proxy for ‘years-only’, not an independent proof of the human reason.", ""]
    for name, result in summary["families"].items():
        lines.append(f"### {name}")
        lines.append(f"{result['passes']} of {result['n']} cases passed. {result['omitted']} edited states were omitted because the controlled text did not survive state construction." if result["omitted"] else f"{result['passes']} of {result['n']} cases passed.")
        if result["failures"]:
            lines.append("Typical redacted failure excerpts:")
            for excerpt in result["failures"][:3]:
                lines.append(f"- {excerpt}")
        else:
            lines.append("No failure excerpt was recorded.")
        lines.append("")
    lines += ["## Limits", "", "These are paired model self-tests on held-out text, not new human labels. The exported Q8_0 model can react to wording, truncation, and correlated posting artifacts. The policy allows relevant alternative experience for a 3+ minimum; the years-up edit deliberately does not add such an allowance. Controls test only verdict invariance, so stable answers do not establish invariance of every probability.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> dict:
    out = (REPO / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    if not out.is_relative_to(REPO / "outputs"):
        raise ValueError("--out must be below outputs/")
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(REPO / args.val, REPO / args.val2)
    if not cases:
        raise RuntimeError("no held-out postings could be loaded")
    profile = (REPO / args.profile).read_text(encoding="utf-8")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(bdd.DEFAULT_TOKENIZER, local_files_only=True)
    questions = bdd.QUESTIONS
    for case in cases:
        case["state"], case["state_truncated"] = _build_state(profile, case["posting"], tokenizer)

    with GGUFClient((REPO / args.server).resolve(), (REPO / args.model).resolve(), out) as client:
        for i, case in enumerate(cases, 1):
            case["base"] = client.predict(case["state"], questions)
            if i % 25 == 0:
                print(f"scored base {i}/{len(cases)}", file=sys.stderr, flush=True)
        summary = {"schema": 1, "model": str((REPO / args.model).resolve()),
                   "n_cases": len(cases),
                   "sources": {"val": sum(c["source"] == "val" for c in cases),
                               "val2": sum(c["source"] == "val2" for c in cases)},
                   "base": _base_agreement(cases), "families": {}}
        records = []
        for family in ("years_up", "years_down", "preferred", "sponsorship", "clearance", "invariance"):
            selected = _select(cases, family, args.max_per_family)
            result = {"n": 0, "passes": 0, "pass_rate": None, "interval": [0.0, 0.0],
                      "omitted": 0, "failures": []}
            for index, case in enumerate(selected):
                edited_posting, marker = edit_posting(case["posting"], family, index)
                edited_state, truncated = _build_state(profile, edited_posting, tokenizer)
                # Gate edits carry their exact sentence as a marker. Control
                # edits use several kinds of change, so verify that the
                # rebuilt state differs from the unedited state instead of
                # looking for the descriptive kind label ("company", etc.).
                visible = ((edited_state != case["state"])
                           if family == "invariance" else (not marker or marker in edited_state))
                if not visible:
                    result["omitted"] += 1
                    continue
                edited = client.predict(edited_state, questions)
                passed, reason = score_case(family, case["base"], edited)
                result["n"] += 1
                result["passes"] += int(passed)
                record = {"family": family, "source": case["source"],
                          "posting_id": case["row"]["posting_id"], "edit": marker,
                          "base": case["base"], "edited": edited, "passed": passed,
                          "reason": reason, "state_truncated": truncated,
                          "edit_visible": visible}
                records.append(record)
                if not passed and len(result["failures"]) < 5:
                    if family == "invariance":
                        excerpt = f"[Company] {marker} edit: {reason}"
                    else:
                        excerpt = f"[Company] after `{marker}`: {reason}"
                    if excerpt not in result["failures"]:
                        result["failures"].append(excerpt)
            if result["n"]:
                result["pass_rate"] = result["passes"] / result["n"]
                result["interval"] = list(wilson_interval(result["passes"], result["n"]))
            summary["families"][family] = result
        with (out / "results.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(summary, REPO / "training" / "RULE-TESTS.md")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default=str(Path.home() / ".unsloth/llama.cpp/build/bin/Release/llama-server.exe"))
    parser.add_argument("--model", default="outputs/full-e3/export-gguf-q8/native/gguf/model-Q8_0.gguf")
    parser.add_argument("--out", default="outputs/rule-tests")
    parser.add_argument("--profile", default="data/training/state-profile.md")
    parser.add_argument("--val", default="data/training/val.jsonl")
    parser.add_argument("--val2", default="data/experiments/export-v3/val2.jsonl")
    parser.add_argument("--max-per-family", type=int, default=0,
                        help="optional deterministic cap; zero means all eligible cases")
    args = parser.parse_args(argv)
    if args.max_per_family < 0:
        parser.error("--max-per-family must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
