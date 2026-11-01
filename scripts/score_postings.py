"""Score exported postings with a saved decision adapter and compare with the source verdicts.

  .venv/Scripts/python.exe -B scripts/score_postings.py --export data/raw/agent-export-<stamp>/postings.jsonl

Each export row needs title / company / location / description (rows without a description
are skipped) and may carry verdict_normalized, source_id, url. Rows are matched to
data/processed/postings.jsonl by fingerprint or URL so the summary can separate postings the
model trained on from postings it has never seen. Writes scored.jsonl + summary.json next to
the export (both under gitignored data/).
"""
from __future__ import annotations

import train_decision as td  # imports unsloth before transformers
from unsloth import FastDecisionModel

import argparse
from collections import Counter
import json
from pathlib import Path

import build_decision_dataset as bdd

SPLITS = ("train", "val", "test", "bulk")


def norm_url(url):
    return (url or "").strip().lower().split("?")[0].split("#")[0].rstrip("/")


def processed_index():
    """fingerprint/url -> posting_id, and posting_id -> split it was used in."""
    keys = {}
    with open(bdd.PROCESSED / "postings.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            for fp in p.get("fingerprints") or []:
                keys["fp:" + fp] = p["posting_id"]
            for u in (p.get("urls") or []) + (p.get("canonical_urls") or []):
                keys["url:" + norm_url(u)] = p["posting_id"]
    used = {}
    for split in SPLITS:
        for r in td.read_jsonl(td.DATA / f"{split}.jsonl"):
            used.setdefault(r["posting_id"], split)
    return keys, used


def match(row, keys):
    for fp in [row.get("source_id"), row.get("fingerprint"), *(row.get("fingerprints") or [])]:
        if fp and "fp:" + str(fp) in keys:
            return keys["fp:" + str(fp)]
    u = norm_url(row.get("url"))
    return keys.get("url:" + u) if u else None


def as_posting(row):
    loc = row.get("location") or row.get("locations") or []
    return {"title": row.get("title"), "employer": row.get("company") or row.get("employer"),
            "locations": loc if isinstance(loc, list) else [loc], "description": row.get("description")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--adapter", default="outputs/full-e3/adapter")
    ap.add_argument("--budget", type=int, default=950, help="posting tokens per state (as trained)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    export = td.REPO / args.export
    out_dir = export.parent

    model, tokenizer = FastDecisionModel.from_pretrained(
        str(td.REPO / args.adapter), max_seq_length=2048, load_in_4bit=True)
    FastDecisionModel.for_inference(model)
    from transformers import AutoTokenizer  # the plain text tokenizer the dataset budgets used
    tok = AutoTokenizer.from_pretrained(bdd.DEFAULT_TOKENIZER)
    count = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
    profile = bdd.PROFILE.read_text(encoding="utf-8")
    keys, used = processed_index()

    rows = [r for r in td.read_jsonl(export) if (r.get("description") or "").strip()][: args.limit]
    scored = []
    with open(out_dir / "scored.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            state, truncated = bdd.make_state(profile, as_posting(r), args.budget, count)
            ans = FastDecisionModel.predict(model, tokenizer, state, bdd.QUESTIONS)
            pid = match(r, keys)
            s = {"source_id": r.get("source_id"), "url": r.get("url"), "company": r.get("company"),
                 "title": r.get("title"), "reviewed_at": r.get("reviewed_at"),
                 "codex_verdict": r.get("verdict_normalized"), "posting_id": pid,
                 "seen_in": used.get(pid) if pid else None, "state_truncated": truncated,
                 "model": {k: v["answer"] for k, v in ans.items()},
                 "confidence": {k: v["probabilities"] for k, v in ans.items()}}
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            scored.append(s)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)}", flush=True)

    def compare(group):
        pairs = [(s["codex_verdict"], s["model"]["verdict"]) for s in group if s["codex_verdict"] in td.VERDICTS]
        if not pairs:
            return {"n": 0}
        res = td.prf([g for g, _ in pairs], [p for _, p in pairs], td.VERDICTS)
        res["codex_strong_called_reject"] = sum(g == "strong" and p == "reject" for g, p in pairs)
        return res

    groups = {"all": scored,
              "never_seen": [s for s in scored if not s["seen_in"]],
              **{f"seen_{k}": [s for s in scored if s["seen_in"] == k] for k in SPLITS}}
    summary = {"adapter": args.adapter, "export": args.export, "scored": len(scored),
               "skipped_no_description": sum(1 for _ in td.read_jsonl(export)) - len(rows),
               "model_verdicts": dict(Counter(s["model"]["verdict"] for s in scored)),
               "codex_verdicts": dict(Counter(s["codex_verdict"] for s in scored)),
               "agreement_vs_codex": {k: compare(g) for k, g in groups.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "agreement_vs_codex"}, indent=2))
    for k, v in summary["agreement_vs_codex"].items():
        print(k, v.get("n"), v.get("accuracy"), v.get("macro_f1"), v.get("confusion"))


if __name__ == "__main__":
    main()
