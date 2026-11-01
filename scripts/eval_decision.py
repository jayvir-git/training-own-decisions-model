"""Score a saved decision adapter on val (reload check) and, with --test, once on test.jsonl.

  .venv/Scripts/python.exe scripts/eval_decision.py --adapter outputs/full-e3/adapter --test

Writes eval-<split>.json and eval-<split>-predictions.jsonl next to the adapter.
"""
from __future__ import annotations

import train_decision as td  # imports unsloth before transformers
from unsloth import FastDecisionModel

import argparse
from collections import Counter
import json


def score(model, tokenizer, rows):
    preds = []
    for r in rows:
        ans = FastDecisionModel.predict(model, tokenizer, r["state"], r["questions"])
        preds.append({"posting_id": r["posting_id"], "label_source": r["label_source"], "gold": r["gold"],
                      "labeller_gold": r.get("labeller_gold"),
                      "pred": {k: v["answer"] for k, v in ans.items()},
                      "probabilities": {k: v["probabilities"] for k, v in ans.items()}})

    def verdict_eval(gold_of):
        sub = [p for p in preds if gold_of(p)]
        gold = [gold_of(p) for p in sub]
        return {"model": td.prf(gold, [p["pred"]["verdict"] for p in sub], td.VERDICTS),
                "always_reject": td.prf(gold, ["reject"] * len(gold), td.VERDICTS)}

    nouls = {}
    for q in td.NOULS + ("min_years",):
        pairs = [(p["labeller_gold"][q], p["pred"][q]) for p in preds
                 if p.get("labeller_gold") and q in p["labeller_gold"]]
        majority = Counter(g for g, _ in pairs).most_common(1)[0][0] if pairs else None
        nouls[q] = {"n": len(pairs), "accuracy": round(sum(g == x for g, x in pairs) / len(pairs), 3) if pairs else None,
                    "majority_baseline": round(sum(g == majority for g, _ in pairs) / len(pairs), 3) if pairs else None,
                    "positives_gold": sum(g is True for g, _ in pairs), "positives_pred": sum(x is True for _, x in pairs)}
    return preds, {
        "verdict_vs_receipt": verdict_eval(lambda p: p["gold"]["verdict"] if p["label_source"] == "receipt" else None),
        "verdict_vs_gold": verdict_eval(lambda p: p["gold"]["verdict"]),
        "verdict_vs_labeller": verdict_eval(lambda p: (p.get("labeller_gold") or {}).get("verdict")),
        "other_questions_vs_labeller": nouls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--test", action="store_true", help="also score test.jsonl (do this once, on the final model)")
    args = ap.parse_args()
    adapter = td.REPO / args.adapter
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(adapter), max_seq_length=args.max_seq_length, load_in_4bit=True)
    FastDecisionModel.for_inference(model)
    for split in ["val"] + (["test"] if args.test else []):
        preds, metrics = score(model, tokenizer, td.read_jsonl(td.DATA / f"{split}.jsonl"))
        (adapter.parent / f"eval-{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with open(adapter.parent / f"eval-{split}-predictions.jsonl", "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        m = metrics["verdict_vs_receipt"]["model"]
        print(split, "receipt acc", m["accuracy"], "macro_f1", m["macro_f1"], flush=True)


if __name__ == "__main__":
    main()
