# Data: what is published and what you must supply

The model was trained on the author's own labelled job-search data. None of that data is in this
repository: posting text, reviewer receipts, teacher-label calls, the candidate profile, trained
weights and run outputs all live in the gitignored `data/` and `outputs/` folders.

## Published

- All code used for dataset construction, labelling, training, evaluation, export, serving,
  rule tests, explanations and the demo (`scripts/`, `demo/`).
- Unit tests (`tests/`, `scripts/test_*.py`) that run on synthetic inputs only.
- Write-ups with aggregate results (`README.md`, `training/*.md`) and figures (`docs/figures/`).
- A fictional candidate profile in the expected format: `examples/state-profile.example.md`.

## Not published

- The importers that copied the author's private job-search records into a raw snapshot.
  `scripts/build_dataset.py` starts from that snapshot; its expected layout is described below
  with neutral folder names.
- The labeller context (policy plus candidate facts), all labels, splits and run outputs.
- The fine-tuned adapter, merged model and GGUF file.

Without these, the published scores cannot be reproduced. The method can be repeated with your own
data by supplying files in the formats below.

## Pipeline inputs and outputs (all under `data/`)

| Stage | Reads | Writes |
|---|---|---|
| `build_dataset.py` | `raw/<YYYYMMDD-HHMM>/` snapshot with `manifest.json` | `processed/postings.jsonl`, `labels_agent.jsonl`, `labels_user.jsonl`, `splits.json`, `REPORT.md` |
| labelling scripts (`codex_labeler*`, `label_agreement*`, `label_opencode.py`, `codex_tiebreak_v4.py`, `bulk_final_v4.py`) | `processed/`, `labeling/context.md` | `labeling/v3/`, `labeling/v4/` |
| `build_decision_dataset.py` | `processed/`, `labeling/`, `training/state-profile.md` | `training/{train,val,test,bulk}.jsonl`, `training/manifest.json` |
| `train_decision.py`, `eval_decision.py` | `training/*.jsonl` | `outputs/<run>/` |
| `export_model.py`, `serve_decision.py`, `rule_tests.py`, `explain_decision.py`, `demo/app.py` | `outputs/<run>/`, `training/state-profile.md` | `outputs/` |
| export experiments (`prepare_export_experiments.py`, `fair_verification.py`, `score_postings.py`) | `raw/agent-export-<stamp>/postings.jsonl` (a second, independently reviewed export) | `experiments/`, `outputs/` |

### Raw snapshot layout expected by `build_dataset.py`

```
data/raw/<YYYYMMDD-HHMM>/
  manifest.json                       {"complete": true, "files": [{"dest_path", "sha256", "status", ...}]}
  agent_a/output/latest.json          {"jobs": [posting, ...]}
  agent_a/state/jobs.sqlite           table jobs, one posting per row, keyed by fingerprint
  agent_a/state/review-receipts.json  {fingerprint: {verdict, reason, reviewed_at, official_url, content_hash}}
  agent_a/state/opportunities.json    {"opportunities": [user decision rows]}
  agent_a/known_applications.json     {"applications": [user decision rows]}
  agent_b/finder/snapshots/**/*.json  one posting per file
  agent_b/finder/state/postings.json  {id: posting}
  agent_b/finder/out/sweep-*.csv      posting rows
  agent_b/tracker.md                  markdown table: id, employer, title, url, ..., applied, ..., status, ..., notes
  existing/desk-export-*.json         optional dashboard export
```

A posting row needs at least `employer`, `title`, `description` and ideally `url`, `job_id` and
`locations`. If your data has a different shape, it is simpler to write `processed/postings.jsonl`
and `processed/splits.json` yourself and start at `build_decision_dataset.py`.

### Training rows

Each line of `data/training/<split>.jsonl`:

```json
{"posting_id": "p-0001", "split": "train", "employer": "example co",
 "state": "<candidate profile>\n\n# Posting\nTitle: ...\nEmployer: ...\nLocation: ...\n\n<description>",
 "questions": {"verdict": {...}, "sponsorship_blocked": {...}, "...": {}},
 "gold": {"verdict": "stretch", "sponsorship_blocked": false,
          "clearance_or_citizenship_required": false, "min_years": 1}}
```

`questions` is a subset of `build_decision_dataset.QUESTIONS`; `min_years` is the bucket index
(0 = none/0, 1 = 1-2, 2 = 3-4, 3 = 5+). Questions missing from `gold` are skipped in training.

### Candidate profile

`data/training/state-profile.md` is prepended to every state. Start from
`examples/state-profile.example.md`; it is fictional and shows the two sections the model was
trained with (candidate facts and the frozen rules). The released results were trained with the
author's private profile, so changing the profile without retraining changes what the inputs mean.

### Private markers for the labelling guard

The labelling scripts refuse to send a posting whose text contains reviewer notes or private
profile markers. Set `TRIAGE_PRIVATE_MARKERS` to a `|`-separated list of names that must never
reach a labeller (for example your name and your own project names).
