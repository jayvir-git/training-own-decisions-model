"""Low-confidence abstention for the verdict, from saved full-e3 predictions (no GPU).

  .venv/Scripts/python.exe -B scripts/analyze_abstention.py

Rule (fixed before looking at val2/test): confidence = top verdict probability. The cut-off
is the highest one on original val (receipt rows) that still answers >= 75% of postings;
below it the model says "uncertain". Val2 (175 never-trained export postings) is the
held-out report; the already-spent test set is reported for information only and was not
used to choose anything. Also checks whether "uncertain" falls more often on postings an
agent marked needs-information. Writes outputs/abstention/results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERDICTS = ("strong", "stretch", "reject")
MIN_COVERAGE = 0.75
FILES = {
    "val": "outputs/full-e3/eval-val-predictions.jsonl",
    "val2": "outputs/export-baseline/evaluation/val2-predictions.jsonl",
    "test": "outputs/full-e3/eval-test-predictions.jsonl",
    "needs_info_check": "outputs/fair-verification/full-e3/fair-predictions.jsonl",
}


def read(name):
    rows = [json.loads(line) for line in open(REPO / FILES[name], encoding="utf-8")]
    if name in ("val", "test"):  # headline splits use receipt labels only
        rows = [r for r in rows if r["label_source"] == "receipt"]
    for r in rows:
        probs = r["probabilities"]["verdict"]
        r["conf"] = max(probs.values())
        r["pred_verdict"] = max(probs, key=probs.get)
    return rows


def macro_f1(gold, pred):
    f1s = []
    for c in VERDICTS:
        tp = sum(g == c == p for g, p in zip(gold, pred))
        fp = sum(g != c == p for g, p in zip(gold, pred))
        fn = sum(g == c != p for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def summarize(rows, cut):
    answered = [r for r in rows if r["conf"] >= cut]
    held = [r for r in rows if r["conf"] < cut]
    acc = lambda rs: round(sum(r["gold"]["verdict"] == r["pred_verdict"] for r in rs) / len(rs), 3) if rs else None
    f1 = lambda rs: round(macro_f1([r["gold"]["verdict"] for r in rs], [r["pred_verdict"] for r in rs]), 3) if rs else None
    wrong = [r for r in rows if r["gold"]["verdict"] != r["pred_verdict"]]
    # A strong/stretch posting confidently thrown away is the costly mistake for a triage tool.
    lost = lambda rs: sum(r["gold"]["verdict"] in ("strong", "stretch") and r["pred_verdict"] == "reject" for r in rs)
    good = lambda r: r["gold"]["verdict"] in ("strong", "stretch")

    def triage(auto_reject):  # everything not auto-rejected goes to a person
        return {"auto_rejected": len(auto_reject), "reviewed": len(rows) - len(auto_reject),
                "good_total": sum(map(good, rows)), "good_auto_rejected": sum(map(good, auto_reject))}
    return {"n": len(rows), "coverage": round(len(answered) / len(rows), 3),
            "answered_predicted": {c: sum(r["pred_verdict"] == c for r in answered) for c in VERDICTS},
            "always_reject_accuracy_answered": acc([dict(r, pred_verdict="reject") for r in answered]),
            "triage_plain_model": triage([r for r in rows if r["pred_verdict"] == "reject"]),
            "triage_with_uncertain": triage([r for r in answered if r["pred_verdict"] == "reject"]),
            "accuracy_all": acc(rows), "macro_f1_all": f1(rows),
            "accuracy_answered": acc(answered), "macro_f1_answered": f1(answered),
            "accuracy_uncertain": acc(held),
            "errors_total": len(wrong), "errors_sent_to_uncertain": sum(r["conf"] < cut for r in wrong),
            "good_postings_rejected_all": lost(rows), "good_postings_rejected_answered": lost(answered)}


def curve(rows):
    out = []
    for cut in [round(0.34 + 0.02 * i, 2) for i in range(31)]:
        s = summarize(rows, cut)
        out.append({"cut": cut, "coverage": s["coverage"], "accuracy_answered": s["accuracy_answered"],
                    "macro_f1_answered": s["macro_f1_answered"]})
    return out


def auc(pos, neg):
    """P(a random positive is less confident than a random negative); ties count half."""
    pairs = [(p < n) + 0.5 * (p == n) for p in pos for n in neg]
    return round(sum(pairs) / len(pairs), 3)


def main():
    data = {k: read(k) for k in FILES}
    val = data["val"]
    cut = max(c for c in sorted({r["conf"] for r in val})
              if sum(r["conf"] >= c for r in val) / len(val) >= MIN_COVERAGE)
    check = data["needs_info_check"]
    pos = [r["conf"] for r in check if r["label_source"] == "export-weak-needs"]
    neg = [r["conf"] for r in check if r["label_source"] == "export-review"]
    result = {
        "rule": f"confidence = top verdict probability; cut-off = highest on val keeping coverage >= {MIN_COVERAGE}",
        "cut": round(cut, 4),
        "splits": {k: summarize(data[k], cut) for k in ("val", "val2", "test")},
        "curves": {k: curve(data[k]) for k in ("val", "val2")},
        "needs_info_check": {
            "n_needs_info": len(pos), "n_resolved": len(neg),
            "uncertain_rate_needs_info": round(sum(p < cut for p in pos) / len(pos), 3),
            "uncertain_rate_resolved": round(sum(n < cut for n in neg) / len(neg), 3),
            "roc_auc_low_confidence": auc(pos, neg)},
    }
    out = REPO / "outputs" / "abstention"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for k, s in result["splits"].items():
        print(k, {x: s[x] for x in ("coverage", "accuracy_answered", "always_reject_accuracy_answered",
                                    "answered_predicted", "triage_plain_model", "triage_with_uncertain")})
    print("cut", result["cut"], "needs_info_check", result["needs_info_check"])


if __name__ == "__main__":
    main()
