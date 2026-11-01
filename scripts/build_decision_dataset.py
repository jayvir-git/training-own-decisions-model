"""Build Unsloth decision-model rows ({state, questions, gold}) for the triage model.

One row per posting. Writes data/training/{train,val,test,bulk}.jsonl plus manifest.json.

Gold priority
  train: labeller labels (v3 gpt-6-astra > v4 gpt-6-luna evaluation), else the latest
         non-needs-information reviewer receipt (verdict only).
  bulk:  v4 bulk-final.jsonl when present, else bulk-labels.jsonl. Train only, never a
         val/test posting or employer.
  val/test: verdict = latest non-needs-information reviewer receipt where present, else
         labeller verdict; other questions from the labeller. The full labeller label is
         kept in `labeller_gold`.

Reads the labelling outputs; never writes to them. The candidate profile placed in each
state lives in data/training/state-profile.md (gitignored), not in this file.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PROCESSED = DATA / "processed"
OUT = DATA / "training"
PROFILE = OUT / "state-profile.md"
V3_CALLS = DATA / "labeling" / "v3" / "calls.jsonl"
V4_CALLS = DATA / "labeling" / "v4" / "calls.jsonl"
V4_BULK = DATA / "labeling" / "v4" / "bulk-labels.jsonl"
V4_BULK_FINAL = DATA / "labeling" / "v4" / "bulk-final.jsonl"
DEFAULT_TOKENIZER = "unsloth/Qwen3.5-0.8B"
POSTING_TOKEN_BUDGET = 1400

YEAR_BUCKETS = ["no minimum stated, or 0 years", "1-2 years", "3-4 years", "5 or more years"]
QUESTIONS = {
    "verdict": {
        "type": "choice",
        "instructions": "Should the candidate apply to this posting?",
        "criteria": {
            "strong": "required fundamentals and core work are directly supported by the candidate's evidence",
            "stretch": "credible adjacent evidence with bounded, learnable gaps, or a decisive gate the text leaves unresolved",
            "reject": "a hard gate fails (location, sponsorship, citizenship, clearance, student status, 3+ required years) or the work is unrelated or far deeper than the evidence",
        },
    },
    "needs_verification": {
        "type": "noul",
        "instructions": "Does the posting text alone leave a decisive gate unresolvable (for example an ambiguous sponsorship restriction or truncated essential qualifications), with no other gate already rejecting?",
    },
    "sponsorship_blocked": {
        "type": "noul",
        "instructions": "Does the posting explicitly refuse visa sponsorship or require permanent, unrestricted work authorization?",
    },
    "clearance_or_citizenship_required": {
        "type": "noul",
        "instructions": "Does the posting require US citizenship, permanent residency, or a security clearance?",
    },
    "min_years": {
        "type": "score",
        "instructions": "Lowest required years of relevant experience stated in the posting (lower bound of a range; preferred years do not count).",
        "criteria": YEAR_BUCKETS,
    },
}
LABEL_FIELDS = ("verdict", "needs_verification", "sponsorship_blocked",
                "clearance_or_citizenship_required", "min_years")
VERDICTS = ("strong", "stretch", "reject")

GATE = re.compile(
    r"sponsor|visa|h-?1b|citizen|green card|permanent resident|clearance|authoriz|e-verify|"
    r"\byears?\b|\byrs?\b|experience|degree|bachelor|master|ph\.?d|graduat|student|enrolled|"
    r"intern|co-?op|apprentice|remote|hybrid|on-?site|in-office|relocat|location|based in|"
    r"required|requirement|qualification|must|minimum|entry|junior|senior|new grad",
    re.I)
BOILERPLATE = re.compile(
    r"equal (employment )?opportunity|without regard to|reasonable accommodation|"
    r"401\(?k|paid time off|dental|vision insurance|parental leave|privacy (policy|notice)|"
    r"benefits include|we offer|perks|wellness|fair chance|pay transparency|e-?verify participation",
    re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_obj(value):
    return json.loads(value) if isinstance(value, str) else value


def years_bucket(value):
    if value is None:
        return 0
    value = float(value)
    return 0 if value <= 0 else 1 if value <= 2 else 2 if value <= 4 else 3


def label_to_gold(label):
    """Labeller output -> decision gold. Unknown/missing noul fields are False (not established)."""
    gold = {}
    if label.get("verdict") in VERDICTS:
        gold["verdict"] = label["verdict"]
    for name in ("needs_verification", "sponsorship_blocked", "clearance_or_citizenship_required"):
        if label.get(name) in {"yes", "no", "unknown"}:
            gold[name] = label[name] == "yes"
    if "min_years" in label:
        gold["min_years"] = years_bucket(label["min_years"])
    return gold


def labeller_labels():
    """posting_id -> (source, label). astra (v3) beats luna (v4 evaluation)."""
    out = {}
    for call in read_jsonl(V4_CALLS):
        if call.get("phase") == "evaluation" and call.get("status") == "ok":
            for pid in as_obj(call["ids"]):
                out[pid] = ("v4-luna-eval", as_obj(call["output"]))
    for call in read_jsonl(V3_CALLS):
        if call.get("status") == "ok":
            for pid in as_obj(call["ids"]):
                out[pid] = ("v3-astra-eval", as_obj(call["output"]))
    return out


def latest_receipts(receipts):
    """posting_id -> latest reviewer verdict that is not needs-information."""
    by_posting = defaultdict(list)
    for r in receipts:
        if r.get("verdict") in VERDICTS:
            by_posting[r["posting_id"]].append(r)
    return {pid: max(rs, key=lambda r: r.get("reviewed_at") or "")["verdict"]
            for pid, rs in by_posting.items()}


def flatten_bulk(row):
    """bulk-final rows carry the label in `final_label` (None while pending a tiebreak);
    bulk-labels rows are the label itself."""
    if "final_label" not in row:
        return row
    meta = {"weight": row.get("weight"), "stratum": row.get("stratum"),
            "bulk_status": row.get("status"), "bulk_label_source": row.get("label_source")}
    return {**(row["final_label"] or {}), **meta}


def bulk_labels():
    path = V4_BULK_FINAL if V4_BULK_FINAL.exists() else V4_BULK
    source = "v4-bulk-final" if path == V4_BULK_FINAL else "v4-luna-bulk"
    return source, {r["posting_id"]: flatten_bulk(r) for r in read_jsonl(path)}


def char_counter(text):
    return max(1, len(text) // 4)


def truncate_posting(text, budget, count=char_counter):
    """Fit text into `budget` tokens, keeping gate-bearing sentences over boilerplate.

    Keeps the opening sentences (title/role framing), then gate sentences, then other
    content, then boilerplate, emitting the survivors in original order with gaps marked.
    Returns (text, truncated).
    """
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if count(text) <= budget:
        return text, False
    parts = [p.strip() for p in SENTENCE.split(text) if p and p.strip()]
    costs = [count(p) + 1 for p in parts]
    head_budget = budget * 0.15
    keep, used = set(), 0
    for i, c in enumerate(costs):
        if used + c > head_budget:
            break
        keep.add(i)
        used += c

    def rank(i):
        if GATE.search(parts[i]) and not BOILERPLATE.search(parts[i]):
            return 0
        return 2 if BOILERPLATE.search(parts[i]) else 1

    for i in sorted(range(len(parts)), key=lambda i: (rank(i), i)):
        if i not in keep and used + costs[i] <= budget:
            keep.add(i)
            used += costs[i]
    out, last = [], -1
    for i in sorted(keep):
        if i != last + 1:
            out.append("[...]")
        out.append(parts[i])
        last = i
    if last != len(parts) - 1:
        out.append("[...]")
    return "\n".join(out), True


def posting_text(posting):
    locations = posting.get("locations") or []
    return (f"Title: {posting.get('title') or ''}\n"
            f"Employer: {posting.get('employer') or ''}\n"
            f"Location: {'; '.join(map(str, locations)) or 'not stated'}\n\n")


def make_state(profile, posting, budget, count):
    body, truncated = truncate_posting(posting.get("description") or "", budget, count)
    return f"{profile.strip()}\n\n# Posting\n{posting_text(posting)}{body}", truncated


def employer_keys(posting, normalize):
    keys = {posting.get("normalized_employer") or ""}
    keys |= {normalize(a) for a in posting.get("employer_aliases") or []}
    keys.add(normalize(posting.get("employer")))
    return {k for k in keys if k}


def build_rows(splits, postings, labeller, receipts, bulk, bulk_source, profile, budget, count, normalize):
    split_of = {pid: name for name in ("train", "val", "test") for pid in splits[name]}
    held_employers = {e for e, s in splits["employer_assignment"].items() if s in {"val", "test"}}
    rows = {"train": [], "val": [], "test": [], "bulk": []}
    stats = Counter()

    def row(pid, split, gold, label_source, extra):
        state, truncated = make_state(profile, postings[pid], budget, count)
        stats[f"{split}_truncated"] += truncated
        questions = {k: v for k, v in QUESTIONS.items() if k in gold}
        return {"posting_id": pid, "split": split, "employer": postings[pid].get("normalized_employer"),
                "state": state, "questions": questions, "gold": gold,
                "label_source": label_source, "state_truncated": truncated, **extra}

    for split in ("train", "val", "test"):
        for pid in splits[split]:
            if pid not in postings:
                stats[f"{split}_missing_text"] += 1
                continue
            lab = labeller.get(pid)
            lab_gold = label_to_gold(lab[1]) if lab else {}
            receipt = receipts.get(pid)
            extra = {"receipt_verdict": receipt, "labeller_source": lab[0] if lab else None,
                     "labeller_gold": lab_gold or None}
            if split == "train":
                if lab_gold.get("verdict"):
                    gold, source = lab_gold, lab[0]
                elif receipt:
                    gold, source = {"verdict": receipt}, "receipt"
                else:
                    stats["train_no_label"] += 1
                    continue
            else:
                gold = dict(lab_gold)
                if receipt:
                    gold["verdict"], source = receipt, "receipt"
                elif gold.get("verdict"):
                    source = lab[0]
                else:
                    stats[f"{split}_no_label"] += 1
                    continue
            rows[split].append(row(pid, split, gold, source, extra))

    taken = {r["posting_id"] for rs in rows.values() for r in rs}
    for pid, lab in bulk.items():
        if split_of.get(pid) in {"val", "test"}:
            stats["bulk_dropped_heldout_id"] += 1
        elif pid in taken:
            stats["bulk_dropped_already_labelled"] += 1
        elif pid not in postings:
            stats["bulk_missing_text"] += 1
        elif employer_keys(postings[pid], normalize) & held_employers:
            stats["bulk_dropped_heldout_employer"] += 1
        else:
            gold = label_to_gold(lab)
            if not gold.get("verdict"):
                stats["bulk_no_verdict"] += 1
                continue
            rows["bulk"].append(row(pid, "bulk", gold, bulk_source,
                                    {"weight": float(lab["weight"]) if lab.get("weight") is not None else None, "stratum": lab.get("stratum"),
                                     "bulk_label_source": lab.get("bulk_label_source"),
                                     "train_split_id": split_of.get(pid) == "train"}))
    return rows, stats


def check_disjoint(rows, splits, postings, normalize):
    held_ids = set(splits["val"]) | set(splits["test"])
    held_emp = {e for e, s in splits["employer_assignment"].items() if s in {"val", "test"}}
    for name in ("train", "bulk"):
        for r in rows[name]:
            assert r["posting_id"] not in held_ids, f"{name} row {r['posting_id']} is a held-out id"
            assert not employer_keys(postings[r["posting_id"]], normalize) & held_emp, \
                f"{name} row {r['posting_id']} shares a held-out employer"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="HF tokenizer for budgets; 'chars' for len/4")
    ap.add_argument("--budget", type=int, default=POSTING_TOKEN_BUDGET, help="posting tokens per state")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO / "scripts"))
    from build_dataset import normalize_employer  # same normalization the splits used

    if args.tokenizer == "chars":
        count = char_counter
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        count = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])

    splits = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))
    labeller = labeller_labels()
    receipts = latest_receipts(read_jsonl(PROCESSED / "labels_agent.jsonl"))
    bulk_source, bulk = bulk_labels()
    needed = set(splits["train"]) | set(splits["val"]) | set(splits["test"]) | set(bulk)
    postings = {}
    with open(PROCESSED / "postings.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["posting_id"] in needed and (p.get("description") or "").strip():
                postings[p["posting_id"]] = p
    profile = PROFILE.read_text(encoding="utf-8")

    rows, stats = build_rows(splits, postings, labeller, receipts, bulk, bulk_source,
                             profile, args.budget, count, normalize_employer)
    check_disjoint(rows, splits, postings, normalize_employer)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"tokenizer": args.tokenizer, "posting_token_budget": args.budget,
                "profile_tokens": count(profile), "bulk_source": bulk_source,
                "stats": dict(sorted(stats.items())), "splits": {}}
    for name, rs in rows.items():
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["splits"][name] = {
            "rows": len(rs),
            "truncated": sum(r["state_truncated"] for r in rs),
            "label_source": dict(Counter(r["label_source"] for r in rs)),
            "verdict": dict(Counter(r["gold"]["verdict"] for r in rs)),
        }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
