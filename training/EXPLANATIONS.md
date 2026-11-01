# Why did it say reject? Sentence-occlusion explanations

## Method

For each validation posting, the explainer scores the unmodified posting, removes one description sentence at a time, rebuilds the serving state, and scores it again. It uses the full-e3 Q8_0 GGUF through llama.cpp's `/v1/systemone`; no PyTorch model is loaded. State construction is the serving path: the fixed local profile, posting metadata, `build_decision_dataset.make_state(..., budget=950)`, the current typed questions, and the local Qwen tokenizer. `needs_verification` is neither requested nor reported.

The reported effects are signed so that positive values support the named outcome:

- `delta_reject = P(reject | original) - P(reject | sentence removed)`;
- `delta_good = P(strong or stretch | original) - P(strong or stretch | sentence removed)`.

Thus removing a red/reject-driving sentence lowers reject probability and produces a positive `delta_reject`. A positive `delta_good` means the sentence supports strong/stretch. `verdict_if_removed` is the argmax after occlusion. The reusable interface is `scripts/explain_decision.py::explain_posting(posting, decide)`; its scorer is injected, so the same method works with the GGUF client or another serving implementation.

The splitter breaks on punctuation, newlines, and common bullets. The explainer first applies the original 950-token selection to the unmodified description and only scores sentences present in that state. Omitted sentences are retained in output with `in_state: false` and zero deltas; zero there means **not observed by the model**, not evidence of irrelevance. The CLI also checks that every reconstructed base state exactly equals the state stored in the validation row.

Run, from the repository root (the lock is mandatory because llama-server uses the GPU):

```powershell
.venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B scripts/explain_decision.py --split data/training/val.jsonl --n 70
.venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B scripts/explain_decision.py --split data/experiments/export-v3/val2.jsonl --n 175
```

Private per-sentence results and logs are written under `outputs/explanations/`. The CLI refuses `test.jsonl`; this analysis uses only `data/training/val.jsonl` and `data/experiments/export-v3/val2.jsonl`.

## Validation results

The val2 run stopped at 139/175 when its process died (no error logged; likely host memory). The remaining 36 postings were scored with `--start 139 --n 36` and the two files concatenated into `outputs/explanations/val2-full.jsonl` (175 unique postings).

| | val (n=70) | val2, unseen employers (n=175) |
|---|---:|---:|
| Predicted reject | 48 | 115 |
| Top reject driver is a gate sentence (keyword audit) | 33 (69%) | 42 (37%) |
| Top driver flagged suspicious by keyword | 0 | 3 |
| Top driver moves P(reject) by < 0.05 (no single decisive sentence) | 15 | 44 |
| Removing the top driver changes the verdict | 20 | 32 |
| Median top-driver effect on P(reject) | 0.27 | 0.08 |

**The keyword audit undercounts real reasons on val2.** Of the 65 val2 rejects whose top driver matched no category, 32 move P(reject) by at least 0.05. Reading those 32 by hand (approximate, one reader):

- **About 8 are real gates the regex missed:** years written without spaces, non-US locations, export-control restrictions, and seniority requirements.
- **About 9 are plausible requirements outside the frozen gates:** GPA cut-offs (3.5+, 3.6+), graduation-date or co-op-only windows, and specific domain experience (biomedical research, a teaching certification).
- **About 15 look spurious:** section headers ("Required Qualifications", "Ideal Candidate Qualifications"), posting dates, one-word labels ("Sales", "Manufacturing", "Permanent"), company descriptions, and a city name. These are the clearest evidence of shortcuts: removing a header should not move a decision.

The other 33 have tiny top-driver effects. Those postings have several overlapping reasons, so removing any one sentence leaves the verdict saturated near reject. Occlusion cannot attribute those decisions. This saturation is also why the medians differ so much between splits.

The keyword audit is deliberately simple and categories can overlap. “Gate” below means years of experience, sponsorship/work authorization, or clearance/citizenship. Seniority is counted separately. Suspicious means the largest reject-attributed sentence matches boilerplate, company-description, benefits, or EEO patterns; it is a review queue, not proof that the model used an invalid rule.

## Individual examples

Posting excerpts, per-posting probabilities and the original explanation figure are withheld from the public release. The aggregate results above summarize the analysis without publishing the underlying postings.

## Limitations

- **Occlusion is not causal truth.** It measures this model's sensitivity to one text edit, not the reason a human should reject, the teacher's rationale, or a faithful account of internal computation.
- **Removal changes more than one fact.** Sentences can be grammatically or semantically correlated. Removing a requirement may leave a duplicate, remove needed context, or change how another sentence is interpreted. One-at-a-time effects need not add up to the total decision.
- **Truncation is part of the observation.** The budget selector prioritizes gate-like text. Text absent from the base state cannot be attributed by this method. After an occlusion, a previously omitted sentence can enter the rebuilt state, so the measured contrast can include replacement text as well as removal.
- **Sentence splitting is heuristic.** Bullets and newlines are treated as sentence boundaries; abbreviations, fragments, and malformed source text can produce awkward units. A very large effect from a formatting fragment should be inspected rather than interpreted literally.
- **Small probability movements are not necessarily meaningful.** Quantization, calibration, and borderline verdicts limit precision. A changed argmax is a useful flag, not a confidence interval.
- **Keyword categories are not adjudicated labels.** Written-out years and common gate phrases are covered, but the patterns can miss paraphrases or match non-gating uses. Suspicious matches require reading the private output; no claim of semantic proof is made.
- **Validation-only scope.** These 245 postings describe sensitivities of this checkpoint on the two approved validation populations. They do not establish general faithfulness, hiring correctness, or test-set performance.
