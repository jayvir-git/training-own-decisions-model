"""Train the triage decision model with Unsloth's decision-model API, then calibrate and
score it on val. Never reads test.jsonl.

  .venv/Scripts/python.exe scripts/train_decision.py --max-steps 60 --out outputs/smoke

Rows come from data/training/ (scripts/build_decision_dataset.py). Writes the adapter
(LoRA + head) and results.json / predictions.jsonl under --out.
"""
from __future__ import annotations

import unsloth  # noqa: F401  (must precede transformers)
from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import torch
from transformers import TrainerCallback, TrainingArguments

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "training"
VERDICTS = ("strong", "stretch", "reject")
NOULS = ("needs_verification", "sponsorship_blocked", "clearance_or_citizenship_required")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def decision_rows(rows):
    return [{"state": r["state"], "questions": r["questions"], "gold": r["gold"]} for r in rows]


def weighted_repeats(rows, seed):
    """Stratum weights as integer repeats (expected count = weight / mean weight)."""
    weights = [float(r.get("weight") or 1.0) for r in rows]
    mean = sum(weights) / len(weights)
    rng, out = random.Random(seed), []
    for r, w in zip(rows, weights):
        k = w / mean
        out += [r] * (math.floor(k) + (rng.random() < k - math.floor(k)))
    return out


def oversample(rows, spec):
    """Repeat rows by gold verdict, e.g. spec "strong=3,stretch=2" (unlisted verdicts stay x1)."""
    times = {k: int(v) for k, v in (part.split("=") for part in spec.split(",") if part)}
    return [r for r in rows for _ in range(times.get(r["gold"]["verdict"], 1))]


def prf(gold, pred, labels):
    per, f1s = {}, []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[c] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "support": tp + fn}
        f1s.append(f)
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0
    return {"n": len(gold), "accuracy": round(acc, 3), "macro_f1": round(sum(f1s) / len(f1s), 3),
            "per_class": per, "confusion": dict(Counter(f"{g}->{p}" for g, p in zip(gold, pred)))}


class StepTimer(TrainerCallback):
    def __init__(self):
        self.times, self.losses, self._t = [], [], None

    def on_step_begin(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self._t = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.times.append(time.perf_counter() - self._t)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.losses.append((state.global_step, round(float(logs["loss"]), 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--epochs", type=float, default=None, help="overrides --max-steps")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--with-bulk", action="store_true", help="add data/training/bulk.jsonl to train")
    ap.add_argument("--bulk-weighting", choices=["none", "repeat"], default="none")
    ap.add_argument("--oversample", default="", help='repeat train rows by verdict, e.g. "strong=3,stretch=2"')
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", default="outputs/smoke")
    ap.add_argument("--save-merged", action="store_true")
    ap.add_argument("--grad-ckpt", choices=["unsloth", "gpu"], default="unsloth",
                    help="unsloth = offload checkpoints to host RAM (needs free RAM); gpu = standard recompute on GPU")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(DATA / "train.jsonl")
    val_rows = read_jsonl(DATA / "val.jsonl")
    if args.with_bulk:
        bulk = read_jsonl(DATA / "bulk.jsonl")
        train_rows += weighted_repeats(bulk, args.seed) if args.bulk_weighting == "repeat" else bulk
    train_rows = oversample(train_rows, args.oversample)
    held = {r["posting_id"] for r in val_rows} | {r["posting_id"] for r in read_jsonl(DATA / "test.jsonl")}
    assert not held & {r["posting_id"] for r in train_rows}, "held-out posting in train"

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length, load_in_4bit=True)
    model = FastDecisionModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0,
        use_gradient_checkpointing=(True if args.grad_ckpt == "gpu" else "unsloth"), random_state=args.seed)
    load_s = time.perf_counter() - t0

    train_items, train_report = FastDecisionModel.build_dataset(decision_rows(train_rows), tokenizer, model)
    val_items, val_report = FastDecisionModel.build_dataset(decision_rows(val_rows), tokenizer, model)
    print("train build", train_report, "val build", val_report, flush=True)

    timer = StepTimer()
    bf16 = is_bfloat16_supported()
    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_items,
        head_learning_rate=args.head_lr,
        callbacks=[timer],
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=min(10, max(1, args.max_steps // 6)),
            max_steps=-1 if args.epochs else args.max_steps,
            num_train_epochs=args.epochs or 1,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            bf16=bf16,
            fp16=not bf16,
            logging_steps=5,
            save_strategy="no",
            output_dir=str(out / "trainer"),
            report_to="none",
            seed=args.seed,
            dataloader_num_workers=0,
        ),
    )
    stats = trainer.train()
    train_peak = torch.cuda.max_memory_reserved()

    calib = FastDecisionModel.calibrate(model, tokenizer, val_items)
    print("calibrate", calib, flush=True)
    FastDecisionModel.for_inference(model)
    preds = []
    t1 = time.perf_counter()
    for r in val_rows:
        ans = FastDecisionModel.predict(model, tokenizer, r["state"], r["questions"])
        preds.append({"posting_id": r["posting_id"], "label_source": r["label_source"], "gold": r["gold"],
                      "labeller_gold": r.get("labeller_gold"),
                      "pred": {k: v["answer"] for k, v in ans.items()},
                      "probabilities": {k: v["probabilities"] for k, v in ans.items()}})
    predict_s = (time.perf_counter() - t1) / len(val_rows)

    def verdict_eval(rows, gold_of):
        rows = [p for p in rows if gold_of(p)]
        gold = [gold_of(p) for p in rows]
        return {"model": prf(gold, [p["pred"]["verdict"] for p in rows], VERDICTS),
                "always_reject": prf(gold, ["reject"] * len(gold), VERDICTS)}

    nouls = {}
    for q in NOULS + ("min_years",):
        pairs = [(p["labeller_gold"][q], p["pred"][q]) for p in preds
                 if p.get("labeller_gold") and q in p["labeller_gold"]]
        majority = Counter(g for g, _ in pairs).most_common(1)[0][0] if pairs else None
        nouls[q] = {"n": len(pairs), "accuracy": round(sum(g == x for g, x in pairs) / len(pairs), 3) if pairs else None,
                    "majority_baseline": round(sum(g == majority for g, _ in pairs) / len(pairs), 3) if pairs else None,
                    "positives_gold": sum(g is True for g, _ in pairs), "positives_pred": sum(x is True for _, x in pairs)}

    steps = timer.times
    results = {
        "config": vars(args) | {"bf16": bf16, "train_rows": len(train_rows), "train_items": len(train_items)},
        "build_report": {"train": train_report, "val": val_report},
        "load_seconds": round(load_s, 1),
        "train_runtime_s": round(stats.metrics["train_runtime"], 1),
        "optimizer_steps": len(steps),
        "step_seconds_mean": round(sum(steps) / len(steps), 2),
        "step_seconds_median": round(sorted(steps)[len(steps) // 2], 2),
        "train_loss": round(stats.metrics["train_loss"], 4),
        "loss_curve": timer.losses,
        "peak_vram_reserved_gib_train": round(train_peak / 2**30, 2),
        "peak_vram_reserved_gib_all": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "peak_vram_allocated_gib_all": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "calibrate_val": calib,
        "predict_seconds_per_row": round(predict_s, 3),
        "verdict_vs_receipt": verdict_eval(preds, lambda p: p["gold"]["verdict"] if p["label_source"] == "receipt" else None),
        "verdict_vs_val_gold": verdict_eval(preds, lambda p: p["gold"]["verdict"]),
        "verdict_vs_labeller": verdict_eval(preds, lambda p: (p.get("labeller_gold") or {}).get("verdict")),
        "other_questions_vs_labeller": nouls,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    model.save_pretrained(str(out / "adapter"))
    if args.save_merged:
        model.save_pretrained_merged(str(out / "merged"))
    print(json.dumps({k: v for k, v in results.items() if k not in ("loss_curve",)}, indent=2))


if __name__ == "__main__":
    main()
