# Smoke test: Qwen3.5-0.8B decision model (2026-10-28)

Command: `.venv/Scripts/python.exe scripts/train_decision.py --max-steps 60 --out outputs/smoke`
Artifacts (gitignored): `outputs/smoke/{adapter/, results.json, predictions.jsonl, train.log}`.
Data used: `data/training/smoke-b1400/` (posting budget 1400 tokens; see "Truncation").
Test split was not loaded.

## Setup
- `FastDecisionModel.from_pretrained("unsloth/Qwen3.5-0.8B", max_seq_length=2048, load_in_4bit=True)`. A plain LM, so Unsloth attaches a fresh Clef JointSchemaHead (width 512).
- `get_peft_model(r=16, lora_alpha=16, lora_dropout=0, use_gradient_checkpointing="unsloth")`: 38.1M trainable / 699M.
- `DecisionTrainer`: batch 1 × grad-accum 8, 60 steps (480 examples ≈ 0.85 epoch of 562 train rows), LoRA LR 2e-4, head LR 1e-4, cosine, warmup 10, bf16.
- Train = 562 train-split postings with labeller gold (475 astra, 87 luna); 32 train postings have no label (20 of them are covered by bulk-final).
- `FastDecisionModel.calibrate` on val items, then `FastDecisionModel.predict` per val row.

## Results (val)
| metric | model | always-reject |
|---|---|---|
| verdict acc vs reviewer receipts (n=63) | 0.698 | 0.698 |
| verdict macro-F1 vs receipts | 0.274 | 0.274 |
| strong P/R | 0 / 0 (support 6) | 0 / 0 |
| stretch P/R | 0 / 0 (support 13) | 0 / 0 |
| reject P/R | 0.698 / 1.00 (support 44) | same |
| verdict acc vs all val gold (n=70, 7 labeller-filled) | 0.629 | 0.629 |
| verdict acc vs labeller (n=70) | 0.586 | 0.586 |
| needs_verification / sponsorship_blocked / clearance acc (vs labeller) | 0.957 / 0.957 / 0.786 | identical majority baselines (0 positives predicted) |
| min_years acc | 0.557 | 0.557 majority |
| calibrate (all decisions, cross-fitted) | acc 0.777, ECE 0.077, record acc 0.114 | |

**What this shows:** the model predicts the training prior for every question. Output probabilities are almost constant across postings (verdict P(reject) 0.60–0.62, SD 0.003), and P(reject) doesn't depend on the gold class. The LoRA did train (non-zero lora_B norms) and loss fell 0.88 → 0.49, but that drop is the head fitting class priors. Under one epoch with a fresh head isn't enough to learn anything input-dependent. **The pipeline works (load, build, train, calibrate, predict, save on native Windows); the 60-step model has no value yet.**

## Resources
- Peak VRAM: 2.05 GiB reserved during training, 2.30 GiB overall (torch counters). Fits 4 GB with ~1.7 GiB headroom.
- Model load + LoRA: 246 s (includes the ~80 s `import unsloth`; first-ever load also downloaded the model).
- Training: 1,345 s for 60 steps. Per optimizer step (8 examples): median 11.8 s, mean 22.4 s (long-tail steps; batch 1 means sequence length varies a lot).
- Predict: 0.70 s per posting (5 questions).

## Full-run time estimate (562 train + 1,437 bulk-final ≈ 2,000 rows)
Per example: 1.47 s at the median step, 2.80 s at the observed wall-clock rate.
- 2 epochs (~4,000 examples, 500 steps): **1.6–3.1 h**
- 3 epochs (~6,000 examples, 750 steps): **2.5–4.7 h**
Plus ~5 min load and ~1 min calibrate/predict. The 950-token budget below makes inputs shorter, which should land nearer the low end.

## Truncation
- Unsloth's Clef encoding of the 5 questions costs **~625 tokens per row** (inferred from its own truncation count). The builder first assumed ~300, so with the 1400-token posting budget Unsloth still cut the end of the state on 386/562 train and 59/70 val inputs. The smoke run trained on those tail-cut inputs.
- Rebuilt with `--budget 950` (current `data/training/`). The builder's gate-aware cut now handles truncation: it keeps the opening sentences, then sentences that mention sponsorship, citizenship, years, location, degree or student status, drops EEO/benefits boilerplate first, and marks gaps `[...]`. Builder truncation: train 410/562, val 61/70, test 50/70, bulk 1,011/1,437. Only ~1% (15 train, 1 val, 13 bulk) still slightly exceed what Unsloth keeps.
- Val is unusually truncation-heavy because 33 of its long postings are from one employer (Palantir); that's employer clustering, not a bug.
- Alternative: `max_seq_length=3072` with a ~1,900-token budget. VRAM headroom suggests it fits, but it is untested and roughly 1.5× slower per step.

## Recommended full-run config
```
.venv/Scripts/python.exe scripts/build_decision_dataset.py --budget 950   # after the GLM tiebreaks finish
.venv/Scripts/python.exe scripts/train_decision.py --with-bulk --epochs 3 --batch-size 1 --grad-accum 8 \
    --lr 2e-4 --head-lr 1e-4 --rank 16 --out outputs/full-e3
```
- Data: train (562) + bulk-final (1,437 finished rows now; 63 are `pending_tiebreak` and are skipped, so rebuild once they're done). Bulk is checked id- and employer-disjoint from val/test. It contains 20 train-split postings that had no other label, and none were dropped.
- Epochs 3: Unsloth recommends 3–4 with less data. ~3–4.5 h on this GPU.
- Use bulk **unweighted** (`--bulk-weighting none`). Stratum weights would push the mix further toward reject (bulk is already 87% reject), which is the prior the smoke model collapsed to. Keep the weights for population-level evaluation, not training.
- If the full run still collapses to the prior, try these in order: (1) oversample strong/stretch rows ~2× in train; (2) a head-only warm-up with `FastDecisionModel.freeze_backbone(model)` for ~1 epoch, then `unfreeze_backbone` (an Unsloth code comment calls this "Stage A of the from-LM recipe"; the docs give no settings); (3) Qwen3.5-2B (docs list 8 GB, so not on this GPU without WSL or cloud).
- Acceptance gate on val (receipts, n=63): beat always-reject on macro-F1 (0.274) with non-zero strong and stretch recall before touching test.
- Optional: `--save-merged` writes the 16-bit merged model for the Decision API (`save_pretrained_merged`). Not yet exercised on Windows.
