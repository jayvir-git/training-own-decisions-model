# Export training experiments

Status: complete; full-e3 existing test reused if reference won

## Protocol

Learning objective: typed-decision modelling, held-out macro-F1, per-class precision/recall and calibration. No job-search utility claim.
Unsloth 2026.10.9 / Qwen3.5-0.8B; unchanged five-question schema; rank 16, batch 1, accumulation 8, 3 epochs, backbone LR 2e-4, head LR 1e-4, GPU checkpointing. Both runs start from the same base model.
All GPU processes run through scripts/gpu_lock.py. Each training process has a 3.9-hour watchdog. Full-e3 and full-e3-os are read-only. No commits.

## Data and leakage controls

Final data: data/experiments/export-v3. v1/v2 are preparation-only drafts; neither was trained or used for model selection. Identity components join fingerprint, canonical URL (preserving job query parameters), posting ID and exact description hashes. All matching IDs and employer aliases are checked against original val/test assignments.
Val2 contains never-trained posting identities. Selection prefers employers unseen by full-e3, then employers with minimal old-training overlap to reach 175; all val2 and verification employers are excluded from BOTH new training sets. Thus full-e3 may have seen other postings at some val2 employers, but no val2 posting. This favors the reference baseline. Employer-disjoint calibration subsets are drawn from each training pool, independent of original val/val2. No test outcomes enter selection.
The requested non-text-reject reason regex excludes whole matched identity components, including matching old training rows and evaluation rows. Conservative identity joins can overexclude reused descriptions/URLs.
Conflict policy: latest export detailed review verdict wins over old labeller verdict, consistent with receipt-based evaluation; retain the other four existing targets and existing state. This mixes data expansion and label correction, so causal improvement cannot be attributed to row count alone.
Partial labels are supported by installed unsloth/models/decision.py, _clef_items lines 365-430: missing gold questions are skipped; kept questions alone are encoded. New verdict-only rows carry only that question. Missing labels are never fabricated as negatives. Context7 lacked version-specific partial-label documentation; installed source is authoritative.

| Dataset | Rows | Verdict counts |
|---|---:|---|
| exp1 | 2144 | {'reject': 1717, 'stretch': 342, 'strong': 85} |
| exp2 | 2168 | {'reject': 1715, 'stretch': 369, 'strong': 84} |
| val2 | 175 | {'reject': 132, 'stretch': 27, 'strong': 16} |
| verification | 20 | {'stretch': 20} |

Resolved verdict conflicts: 102; weak positive training rows before calibration: 45. Filter event counts: {'missing_text_or_employer': 185, 'held_original_rows': 195, 'non_text_reject_rows': 82}.

Weak needs_verification supervision maps needs_information to stretch + true only when reasons mention text requirements, qualifications, sponsorship/authorization or contradictions, excluding source availability and non-text status reasons. Existing explicit sponsorship/citizenship/3+year hard-gate labels veto these positives. This is a conservative lexical proxy, NOT audited ground truth. Remaining candidate-evidence or policy ambiguity may still be noise. No negative verification labels are inferred from export verdicts.
The verification holdout is positive-only; its recall measures proxy recovery, and precision on that subset alone is uninformative. Inspect original-val positive precision/recall and false positives as well. Original val has very few genuine positives. No class weighting in these first controlled runs.

## Commands

Run orchestration: `.venv/Scripts/python.exe -B scripts/run_export_experiments.py` (launch detached with PowerShell Start-Process; logs under outputs/export-experiments).
Each train command: `.venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B scripts/train_export_experiment.py --dataset data/experiments/export-v3/expN --epochs 3 --batch-size 1 --grad-accum 8 --lr 2e-4 --head-lr 1e-4 --rank 16 --grad-ckpt gpu --out outputs/<run>`.
Scoring uses evaluate_export_experiment.py --adapter <adapter> --out <distinct evaluation directory>. Only the final chosen new model gets --test; the test-started.json sentinel prevents accidental reruns. If full-e3 wins, its existing test result is reused, not rescored.

## Results

| Model | Split | N | Accuracy | Macro-F1 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| full-e3 | val | 62 | 0.806 | 0.691 | 0.529 | 0.289 |
| always reject | val | 62 | 0.694 | 0.273 | - | - |
| full-e3 | val2 | 175 | 0.737 | 0.609 | 0.648 | 0.365 |
| always reject | val2 | 175 | 0.754 | 0.287 | - | - |
| full-e3 | verification | 20 | 0.55 | 0.237 | 1.157 | 0.741 |
| always reject | verification | 20 | 0.0 | 0.0 | - | - |

### full-e3: per-class precision / recall

| Split | Class | Precision | Recall | Support |
|---|---|---:|---:|---:|
| val | strong | 0.6 | 0.5 | 6 |
| val | stretch | 0.667 | 0.615 | 13 |
| val | reject | 0.867 | 0.907 | 43 |
| val | needs_verification=true | 0.0 | 0.0 | 2 |
| val2 | strong | 0.5 | 0.562 | 16 |
| val2 | stretch | 0.356 | 0.593 | 27 |
| val2 | reject | 0.929 | 0.788 | 132 |
| val2 | needs_verification=true | 0.0 | 0.0 | 0 |
| verification | strong | 0.0 | 0.0 | 0 |
| verification | stretch | 1.0 | 0.55 | 20 |
| verification | reject | 0.0 | 0.0 | 0 |
| verification | needs_verification=true | 0.0 | 0.0 | 20 |

| Model | Split | N | Accuracy | Macro-F1 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| exp1 | val | 62 | 0.758 | 0.682 | 0.593 | 0.328 |
| always reject | val | 62 | 0.694 | 0.273 | - | - |
| exp1 | val2 | 175 | 0.731 | 0.603 | 0.583 | 0.337 |
| always reject | val2 | 175 | 0.754 | 0.287 | - | - |
| exp1 | verification | 20 | 0.55 | 0.237 | 0.970 | 0.635 |
| always reject | verification | 20 | 0.0 | 0.0 | - | - |

### exp1: per-class precision / recall

| Split | Class | Precision | Recall | Support |
|---|---|---:|---:|---:|
| val | strong | 1.0 | 0.5 | 6 |
| val | stretch | 0.471 | 0.615 | 13 |
| val | reject | 0.857 | 0.837 | 43 |
| val | needs_verification=true | 0.0 | 0.0 | 2 |
| val2 | strong | 0.571 | 0.5 | 16 |
| val2 | stretch | 0.333 | 0.593 | 27 |
| val2 | reject | 0.92 | 0.788 | 132 |
| val2 | needs_verification=true | 0.0 | 0.0 | 0 |
| verification | strong | 0.0 | 0.0 | 0 |
| verification | stretch | 1.0 | 0.55 | 20 |
| verification | reject | 0.0 | 0.0 | 0 |
| verification | needs_verification=true | 0.0 | 0.0 | 20 |

| Model | Split | N | Accuracy | Macro-F1 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| exp2 | val | 62 | 0.742 | 0.653 | 0.545 | 0.336 |
| always reject | val | 62 | 0.694 | 0.273 | - | - |
| exp2 | val2 | 175 | 0.634 | 0.476 | 0.747 | 0.448 |
| always reject | val2 | 175 | 0.754 | 0.287 | - | - |
| exp2 | verification | 20 | 0.35 | 0.173 | 0.863 | 0.597 |
| always reject | verification | 20 | 0.0 | 0.0 | - | - |

### exp2: per-class precision / recall

| Split | Class | Precision | Recall | Support |
|---|---|---:|---:|---:|
| val | strong | 0.75 | 0.5 | 6 |
| val | stretch | 0.444 | 0.615 | 13 |
| val | reject | 0.875 | 0.814 | 43 |
| val | needs_verification=true | 0.0 | 0.0 | 2 |
| val2 | strong | 0.286 | 0.375 | 16 |
| val2 | stretch | 0.261 | 0.444 | 27 |
| val2 | reject | 0.861 | 0.705 | 132 |
| val2 | needs_verification=true | 0.0 | 0.0 | 0 |
| verification | strong | 0.0 | 0.0 | 0 |
| verification | stretch | 1.0 | 0.35 | 20 |
| verification | reject | 0.0 | 0.0 | 0 |
| verification | needs_verification=true | 0.0 | 0.0 | 20 |

| Model | Split | N | Accuracy | Macro-F1 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|

## Selection and limitations

Predeclared verdict score = mean macro-F1 on original receipt val and val2. Experiment 1 competes against full-e3 on this score. Experiment 2 is preferred only if its score is within 0.02 of that winner and verification-holdout positive recall improves. Original-val false positives are reported, not hidden. Test is scored once after final selection; never used to choose. Export labels are agent judgements, not independent human truth; policy-era drift and tiny minority-class supports limit conclusions.
Calibration reports held-out multiclass NLL and Brier (sum over three classes), using training-employer holdout temperature calibration. Full-e3 was calibrated on original val, so its original-val calibration numbers are optimistic; val2 provides the independent comparison.
Existing regression suites: 21 tests in tests/ and 47 scripts/test_*.py tests passed; legacy resource warnings did not fail tests. No existing trainer or dataset builder was edited.

Selection: `{"winner": "full-e3", "adapter": "outputs/full-e3/adapter", "validation_scores": {"full-e3": 0.6499999999999999, "exp1": 0.6425000000000001, "exp2": 0.5645}, "test_used_for_selection": false}`

## Exp 3: verification threshold diagnosis (2026-10-29)

**Result: full-e3 already ranks positives; 0.5 hides that signal.** A threshold
of 0.075 meets the requested positive-F1 and verdict constraints under the
cross-fitted protocol below. Stop before A (decoupled training) or C (fourth
verdict): B can reuse saved predictions without loading the GPU. No training,
new adapter, test access, or writes to the four protected model directories.

### Fixed evaluation protocol and data

Before inspecting probabilities, fixed five folds, the threshold grid below,
maximum calibration positive-class F1, and higher-threshold tie breaking.
`scripts/analyze_verification_thresholds.py` writes the split manifest before
reading predictions. Source files are exactly
`data/experiments/export-v3/verification.jsonl` (20 weak positives) and
`data/experiments/export-v3/val.jsonl` (68 labelled original-val rows: 66 false,
2 true). Total 88, 22 positives (25%). Missing auxiliary labels would be omitted,
never converted to false. All 20 weak positives are evaluated; their forced
stretch verdicts are irrelevant here. No `test.jsonl` is read.

Stratify by source and boolean label, sort IDs by SHA-256 of `needs-v1:<ID>`,
then allocate round-robin to five folds. For each fold, select the threshold
using only the other four folds and score that held-out fold. Evaluation folds
have 17-19 rows; every row contributes exactly once to aggregate metrics.
This is cross-fitted threshold selection, not threshold maximization on the
scored rows. All five folds independently selected the same threshold for each
model, so the final operating points are unambiguous. Folds are posting-level,
not newly employer-disjoint. Existing dataset identity exclusions are inherited;
employer clustering can still make uncertainty estimates optimistic.

Predictions come from the existing `evaluate_export_experiment.py` outputs in
`outputs/{export-baseline,export-more-e3,export-needs-e3}/evaluation/`.
Posting IDs and gold are checked on join; duplicate IDs and invalid probabilities
fail. Verdict scores reuse that evaluator's receipt-only original val (n=62)
and val2 (n=175). The new auxiliary cohort uses all labelled original-val rows.
No additional model selection by test results occurs.

**Limitations:** full-e3's existing temperature calibration used original val;
thus these are held-out threshold decisions but not a pristine independent
probability-calibration benchmark. Historical validation reuse also remains.
Weak positives are agent-label proxies with a different source from negatives;
source effects and label noise may explain some ranking. Cross-fitting is not a
new external validation set. Reported probabilities are rounded saved scores;
PR-AUC below means noninterpolated average precision, with ties grouped.

### Results

| Model / policy | Threshold (all five folds) | Precision | Recall | Positive F1 | ROC-AUC | PR-AUC (AP) | Verdict val / val2 | Mean | Verdict constraint |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| Always no | n/a | 0 | 0 | 0 | 0.500 | 0.250 | n/a | n/a | n/a |
| Always yes (context) | 0 | 0.250 | 1.000 | 0.400 | 0.500 | 0.250 | n/a | n/a | n/a |
| **full-e3** | **0.075** | **0.640** | **0.727** | **0.681** | **0.837** | **0.619** | **0.691 / 0.609** | **0.650** | **Pass (>=0.630)** |
| exp1 | 0.010 | 0.316 | 0.818 | 0.456 | 0.673 | 0.379 | 0.682 / 0.603 | 0.643 | Pass |
| exp2 | 0.025 | 0.373 | 0.864 | 0.521 | 0.736 | 0.475 | 0.653 / 0.476 | 0.565 | Fail |

All three models have positive precision/recall/F1 = 0 at threshold 0.5.
Full-e3 cross-fit confusion: TP=16, FP=9, FN=6, TN=57. On weak positives alone,
recall is 14/20=0.700 (precision is not meaningful on that positive-only subset).
On original val: TP=2, FP=9, FN=0, TN=57; precision=0.182, recall=1.000,
F1=0.308, negative false-positive rate=9/66=0.136. The 0.640 combined precision
must not be projected onto the natural low-prevalence population.

### Threshold curves

Each cell is positive-class **precision / recall / F1** on the full 88-row
cohort. These curves are descriptive; threshold selection used only the
calibration portion in each fold, as specified above.

| Threshold | full-e3 | exp1 | exp2 |
|---:|---|---|---|
| 0 | .250 / 1.000 / .400 | .250 / 1.000 / .400 | .250 / 1.000 / .400 |
| .010 | .262 / 1.000 / .415 | .316 / .818 / .456 | .292 / .955 / .447 |
| .025 | .339 / .955 / .500 | .346 / .409 / .375 | .373 / .864 / .521 |
| .050 | .465 / .909 / .615 | .444 / .182 / .258 | .409 / .409 / .409 |
| .075 | .640 / .727 / .681 | .500 / .091 / .154 | .625 / .227 / .333 |
| .100 | .588 / .455 / .513 | .000 / .000 / .000 | .500 / .091 / .154 |
| .125 | .500 / .227 / .312 | .000 / .000 / .000 | .500 / .045 / .083 |
| .150 | .556 / .227 / .323 | .000 / .000 / .000 | 1.000 / .045 / .087 |
| .175 | .714 / .227 / .345 | .000 / .000 / .000 | .000 / .000 / .000 |
| .200 | .600 / .136 / .222 | .000 / .000 / .000 | .000 / .000 / .000 |
| .250 | 1.000 / .045 / .087 | .000 / .000 / .000 | .000 / .000 / .000 |
| .300 | .000 / .000 / .000 | .000 / .000 / .000 | .000 / .000 / .000 |
| .400 | .000 / .000 / .000 | .000 / .000 / .000 | .000 / .000 / .000 |
| .500 | .000 / .000 / .000 | .000 / .000 / .000 | .000 / .000 / .000 |
| 1.000 | .000 / .000 / .000 | .000 / .000 / .000 | .000 / .000 / .000 |

### Cheap training path audit

Context7 `/unslothai/unsloth` did not return version-specific decision training
documentation. Installed Unsloth 2026.10.9 is authoritative here:
`models/decision.py:1692` delegates `freeze_backbone` to
`models/decision_from_lm.py`, which freezes every encoder parameter (including
LoRA) and records names for `unfreeze_backbone`. The separate `model.head`
remains trainable. `DecisionTrainer.create_optimizer` includes only parameters
with `requires_grad`, and supports `head_learning_rate`. Thus loading the
full-e3 adapter, freezing the encoder, and training the head is a sound API path.
It still requires backbone forward passes; no measured minutes-scale timing is
claimed. The head is shared across questions, so freezing the backbone alone
does not guarantee unchanged verdicts. Missing gold also removes those questions
from training encoding (`_clef_items`), which must be considered when comparing
partial-label training with full-schema inference.

The installed trainer exposes smoothing, Brier, ordinal and KL weights, but no
per-question or positive-class loss-weight argument. Row oversampling is a
supported way to alter class exposure; custom loss weighting needs an explicit
implementation. Neither was needed after B succeeded. No fourth verdict was
trained or proposed for serving. Existing dynamic choice handling alone would
not validate a new schema's predictions or calibration.

### Commands, artifacts, and conclusion

CPU-only reproduction:

```powershell
.venv/Scripts/python.exe -B scripts/analyze_verification_thresholds.py
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
.venv/Scripts/python.exe -B -m unittest discover -s scripts -p 'test_*.py' -v
```

Outputs: `outputs/verification-thresholds/{split.json,results.json}` and three
per-row `*-decisions.jsonl` files. Regression checks: **35 tests in tests/ and
47 scripts/test_*.py tests pass**, including new deterministic split, disjointness,
missing-label, duplicate-ID, tied-score/AP and threshold-policy tests. No GPU
command or long detached job was needed. No commits were made.

Keep **outputs/full-e3/adapter** and the same five questions. Optional serving
flag `--needs-verification-threshold 0.075` applies the evaluated decision rule
only to the exact standard question, preserving all probabilities and other
answers, and exposes `decision_threshold` in the response. The default remains
unchanged. This is an operating threshold, not a newly calibrated probability.
CPU tests validate the policy; real-model HTTP was not rerun since the underlying
adapter and prediction path are unchanged.

The evidence rejects "learned nothing" and a head-capacity failure: the original
head has useful ranking. Rare positives and a mismatched operating point are the
immediate explanation for zero yes answers. Weak-label noise and too few clean
positives remain plausible quality limits, but these experiments do not isolate
them causally. Do not drop the question or promote needs_information to a fourth
verdict based on this failure. Retain it as an experimental signal; audit more
text-only positive labels and validate precision on a fresh representative set
before using it as an automatic gate. Broader historical needs_information
reasons should remain separate from the strict text-gate definition.

### Review note: the 0.075 threshold result is confounded (parent review, 2026-10-29)

The evaluation set above takes all positives from the agent export, as stretch-like postings. Its negatives come from original validation, which is about 70% rejects. A score that only separates "not a clear reject" from "reject" would therefore look good on that set.

Same-source check: full-e3 was scored on val2 (`outputs/export-baseline/evaluation/val2-predictions.jsonl`). Val2 holds 175 export postings that the agent resolved as strong, stretch or reject, not needs-information, so they are reasonable negatives.

| Threshold | Val2 postings flagged |
|---:|---:|
| 0.05 | 97 / 175 |
| **0.075** | **78 / 175 (45%)** |
| 0.10 | 60 / 175 |
| 0.20 | 10 / 175 |
| 0.50 | 0 / 175 |

At 0.075 the flags follow the verdict: 33 of 43 strong/stretch postings are flagged, against 45 of 132 rejects. The `needs_verification` probability mostly tracks "not a reject", not "a gate the text leaves unresolved". The 0.681 positive F1 above is not evidence that the question works. The `--needs-verification-threshold` flag stays opt-in and is not recommended. A fair evaluation needs negatives from the same source as the positives.

## Exp 4: fair verification evaluation

### Frozen protocol (2026-10-29)

The Exp 3 operating point is not evidence of gate detection. This experiment uses
20 weak positives from export-v3/verification.jsonl and all 175 resolved export
val2 postings as negatives. Positives lose their artificial stretch verdict.
The fair set therefore has 20 unresolved, 16 strong, 27 stretch and 132 reject
rows. No genuine verdict-matched positives exist: do not describe the forced
stretch label as an independent verdict. Report single-stratum false-positive
rates (ranking metrics undefined for single-class strata), plus positives versus
each negative verdict and positives versus the combined 43 strong/stretch rows.
The latter comparison directly challenges the non-reject shortcut.

Calibration is separate: the 47 new export rows in exp2/calibration.jsonl,
6 weak positives and 41 resolved negatives (6 stretch, 35 reject). Employers are
disjoint from fit and evaluation; the original export-v3 calibration employers
were shared across exp1/exp2. Old-source calibration rows are excluded because
full-e3 trained on some of them. There are no strong calibration negatives, a
material limitation. Thresholds maximize positive F1 on this calibration split
using the Exp 3 fixed grid, breaking ties toward higher thresholds. No threshold
is selected on fair evaluation. No new probability-temperature calibration is
applied; these are operating thresholds on saved-adapter probabilities.

Predeclared usability screen: fair precision >= 0.5 and recall >= 0.5,
ROC-AUC > 0.5 against the strong/stretch negatives, and mean verdict macro-F1
on original receipt val and val2 >= 0.630 (full-e3 reference approximately
0.650 minus 0.020). These are exploratory practical gates, not significance
claims. PR-AUC denotes tie-grouped, noninterpolated average precision.

### Variant A labels and leakage

Before scoring/training, set every export-resolved row to needs_verification=false:
a resolved review is the proxy that no decisive gate remains open. This includes
old-source rows whose verdict was replaced by export-v3 without updating their
label_source: the builder reconstructs export-v3 identity components from posting
IDs, fingerprints, canonical URLs and exact description hashes. Weak positives
have only needs_verification=true and no verdict target. Other old targets remain.
Nine rows (four fit, five fair-evaluation) have ambiguity-related reason wording; they are
recorded privately in the manifest. No lexical exception changes the resolved
label rule. This is an explicit weak-label assumption, not audited ground truth.

Fit has 1,825 unique rows: 39 weak positives and 14 remaining old positives.
The 460 export-resolved fit rows receive negative verification supervision.
Repeat each weak positive eight times (fixed before outcomes), giving 2,098
training rows and 326 positive exposures. Calibration and evaluation are never
oversampled. ID and employer exclusions are asserted against original held-out
split metadata and the val2/verification rows; calibration employers are also
excluded from fit. No test.jsonl is opened. Original exclusion provenance is
inherited from export-v3; initialization retains full-e3's historical exposure,
so this is not a claim of a newly pristine backbone holdout.

Installed Unsloth 2026.10.9 was inspected: freeze_backbone freezes encoder and
LoRA parameters; DecisionTrainer optimizes only requires_grad parameters. Assert
that only head parameters train. Start from outputs/full-e3/adapter, head LR
1e-4, three epochs, batch 1/accumulation 8, cosine schedule, 10 warmup steps,
weight decay 0.01, seed 3407, bf16 and standard GPU checkpointing. No new LoRA
is attached. Partial-label rows encode only supervised questions, whereas
inference uses all five questions; this remains a train/inference limitation.
Context7 lacked matching version-specific decision APIs and Transformers
5.17.0 documentation; installed Unsloth and Transformers source is the authority. Save to a new directory and reload before evaluation.

### Conditional Variant C protocol

Only after a completed A fails, C may start from the same full-e3 adapter with
the same head-only learning rates and repeats, but one epoch. This compute
budget was fixed while A was still training, before its evaluation: measured
A throughput projects about 2.4 hours, leaving room for a roughly 45-minute
C pilot within an approximately four-hour total GPU budget. This is a short
pilot, not a matched-epoch comparison or an exhaustive rejection of the fourth
class. Add needs_information as a fourth
verdict criterion and remove the separate needs_verification question. Weak
positive rows supervise only the new verdict; resolved rows keep their verdicts.
The 14 remaining old positive rows (all previously stretch) also map to the new
verdict, retaining their other auxiliary targets. Remove unresolved gates from
the stretch criterion so the two choice definitions do not overlap.
Use the fourth-choice probability for the verification evaluation and select its
threshold on the same calibration split. Verdict macro-F1 still averages the
original three reference classes, counting predicted needs_information as a
miss. Apply the calibrated gate threshold to C verdicts too: above threshold
return needs_information; below it choose the highest-probability original
class. This keeps the verdict constraint consistent with serving. Record raw
four-way argmax metrics separately as a diagnostic. This is a schema experiment, not evidence of independent clean labels.
Run via `scripts/run_fair_verification.py --only-c`, which asserts A has failed
and applies the same GPU lock, detached logging and 3.9-hour watchdog.

### Commands and execution

```powershell
.venv/Scripts/python.exe -B scripts/fair_verification.py
Start-Process -FilePath '.venv/Scripts/python.exe' -ArgumentList '-B scripts/run_fair_verification.py' -WindowStyle Hidden -RedirectStandardOutput 'outputs/fair-verification/runner.stdout.log' -RedirectStandardError 'outputs/fair-verification/runner.stderr.log'
.venv/Scripts/python.exe -B -m unittest discover -s tests -q
.venv/Scripts/python.exe -B -m unittest discover -s scripts -p 'test_*.py' -q
```

The sequential runner wraps every GPU worker with
`.venv/Scripts/python.exe -B scripts/gpu_lock.py --`, captures both streams,
records process status, detects nonzero exits and enforces a 3.9-hour process-tree
watchdog per run. Baselines score before Variant A trains. Private data, audit
reasons, predictions and adapters stay under gitignored data/ and outputs/.
Protected model directories are read-only. No commits are made.

Final validation: 48 tests in tests/ and 47 scripts/test_*.py tests pass;
`git diff --check` is clean. The optional C schema and threshold policy have
CPU regression coverage. Both trained adapters saved and reloaded successfully.

Exp2 reuses existing saved fair/val/val2 predictions and freshly scores the new
calibration subset. Fresh full-e3 and exp1 fair scores exactly reproduced saved
verification probabilities (maximum difference zero across all 195 rows).

### Results

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) | Verdict val | Verdict val2 | Mean | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full-e3 | 0.025 | 0.131 | 0.950 | 0.230 | 0.660 | 0.206 | 0.691 | 0.609 | 0.650 | no |
| exp1 | 0.010 | 0.133 | 0.800 | 0.229 | 0.603 | 0.124 | 0.682 | 0.603 | 0.643 | no |
| exp2 | 0.010 | 0.114 | 0.950 | 0.204 | 0.569 | 0.134 | 0.653 | 0.476 | 0.565 | no |
| variant-a | 0.010 | 0.110 | 0.450 | 0.176 | 0.589 | 0.129 | 0.668 | 0.601 | 0.634 | no |
| variant-c | 0.050 | 0.129 | 0.600 | 0.212 | 0.636 | 0.151 | 0.441 | 0.347 | 0.394 | no |

Always-yes fair baseline: precision 0.103, recall 1.000, F1 0.186, ROC-AUC 0.500 and AP 0.103. Always-no positive precision/recall/F1 are zero.

#### Matched and verdict-stratified checks

Each paired comparison uses the same 20 unresolved positives plus only the named negative verdict(s). These are diagnostic comparisons, not invented positive verdict labels. Thresholds remain those selected on calibration.

| Model | Negative cohort | P | R | F1 | ROC-AUC | AP | Negative FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| full-e3 | matched_nonreject | 0.317 | 0.950 | 0.475 | 0.416 | 0.342 | 0.953 |
| full-e3 | positive_vs_strong | 0.559 | 0.950 | 0.704 | 0.341 | 0.518 | 0.938 |
| full-e3 | positive_vs_stretch | 0.422 | 0.950 | 0.585 | 0.461 | 0.473 | 0.963 |
| full-e3 | positive_vs_reject | 0.183 | 0.950 | 0.306 | 0.739 | 0.297 | 0.644 |
| exp1 | matched_nonreject | 0.291 | 0.800 | 0.427 | 0.315 | 0.239 | 0.907 |
| exp1 | positive_vs_strong | 0.516 | 0.800 | 0.627 | 0.181 | 0.403 | 0.938 |
| exp1 | positive_vs_stretch | 0.400 | 0.800 | 0.533 | 0.394 | 0.381 | 0.889 |
| exp1 | positive_vs_reject | 0.198 | 0.800 | 0.317 | 0.696 | 0.216 | 0.492 |
| exp2 | matched_nonreject | 0.311 | 0.950 | 0.469 | 0.422 | 0.290 | 0.977 |
| exp2 | positive_vs_strong | 0.559 | 0.950 | 0.704 | 0.342 | 0.467 | 0.938 |
| exp2 | positive_vs_stretch | 0.413 | 0.950 | 0.576 | 0.469 | 0.456 | 1.000 |
| exp2 | positive_vs_reject | 0.153 | 0.950 | 0.264 | 0.617 | 0.220 | 0.795 |
| variant-a | matched_nonreject | 0.231 | 0.450 | 0.305 | 0.391 | 0.335 | 0.698 |
| variant-a | positive_vs_strong | 0.429 | 0.450 | 0.439 | 0.381 | 0.560 | 0.750 |
| variant-a | positive_vs_stretch | 0.333 | 0.450 | 0.383 | 0.396 | 0.427 | 0.667 |
| variant-a | positive_vs_reject | 0.173 | 0.450 | 0.250 | 0.653 | 0.182 | 0.326 |
| variant-c | matched_nonreject | 0.286 | 0.600 | 0.387 | 0.517 | 0.393 | 0.698 |
| variant-c | positive_vs_strong | 0.522 | 0.600 | 0.558 | 0.584 | 0.694 | 0.688 |
| variant-c | positive_vs_stretch | 0.387 | 0.600 | 0.471 | 0.478 | 0.468 | 0.704 |
| variant-c | positive_vs_reject | 0.190 | 0.600 | 0.289 | 0.675 | 0.200 | 0.386 |

Matched non-reject always-yes baseline: precision 0.317, recall 1.000, F1 0.482, ROC-AUC 0.500, AP 0.317. Every tested model has lower matched F1 at its calibration-selected threshold.
Within each resolved-only stratum there are zero positives: ROC-AUC/AP and positive recall are undefined; precision and F1 are zero by convention when false positives occur. The paired-table FPR is exactly that stratum's FPR. Within unresolved-only rows, only recall is informative.

#### Training and conclusion

variant-a: runtime 8476.9s; {'total': 9008, 'skipped': 0, 'reason': None, 'truncated': 31}.
variant-c: runtime 2749.2s; {'total': 7222, 'skipped': 0, 'reason': None, 'truncated': 1}.
C raw four-way argmax verdict macro-F1: val 0.622, val2 0.588, mean 0.605. The headline uses the calibration-selected threshold policy.
C used one epoch under the declared compute budget; it is not an exhaustive test of a fourth verdict. The current weak labels do not support a causal capacity claim.
No completed model passes the fair test. Do not recommend the Exp 3 threshold or claim that needs_verification works. Retain full-e3 for verdicts; redefine the auxiliary question around directly observable text ambiguity and obtain independently audited same-source positives/negatives, or drop it as a serving gate. These results do not establish a head-capacity limit: weak supervision, train/inference schema differences and calibration instability remain confounded.
Serving: keep outputs/full-e3/adapter; do not enable --needs-verification-threshold 0.075. No replacement adapter/threshold is recommended.

All evaluation rows were scored from reloaded adapters. Run commands, stdout/stderr, status and predictions are under outputs/fair-verification/<model>/. No test evaluation or protected-directory mutation was performed.

## Decision: needs_verification dropped (2026-10-29)

The question is dropped rather than relabelled. full-e3 is not retrained without it: every verdict comparison in this file was made with the five-question schema, and removing a question would invalidate that selection. The server strips the answer from posting responses, and the opt-in threshold flag and `scripts/verification_policy.py` were removed. Reviving the question would need hand-audited labels for "the posting text leaves a gate unresolved", with negatives drawn from the same source.


## Exp 5: low-confidence "uncertain" instead of a trained question (2026-10-29)

Script: `scripts/analyze_abstention.py` (saved full-e3 predictions, no GPU). Rule fixed before scoring val2/test: confidence = top verdict probability; cut-off = highest on original val (63 receipt rows) that still answers ≥ 75%. Result: cut-off 0.673. Test was already spent and is shown for information only.

| Split | Answered | Accuracy answered | Always-reject on the same rows | Auto-rejected (plain → with uncertain) | Good postings auto-rejected (plain → with uncertain) |
|---|---:|---:|---:|---:|---:|
| val (n=63) | 76% | 0.896 | 0.833 | 46 → 43 | 6 → 4 of 19 |
| val2 (n=175) | 53% | 0.914 | 0.925 | 112 → 90 | 8 → 6 of 43 |
| test (n=63) | 76% | 0.854 | 0.854 | 38 → 38 | 2 → 2 of 18 |

- The model is only ever confident about rejects: no `strong` prediction clears the cut-off. "Uncertain" therefore means "maybe worth a look", not "the posting is ambiguous".
- Higher accuracy on answered rows is not a gain: always-reject does as well or better on the same rows.
- Coverage falls from 76% on val to 53% on val2, so the cut-off does not transfer across sources.
- As triage it costs 22 extra reviews to save 2 good postings on val2, and changes nothing on test.
- It does not find needs-information postings: 70% of the 20 weak positives are uncertain against 47% of resolved postings; ROC-AUC 0.566.

Not adopted. Serving is unchanged.
