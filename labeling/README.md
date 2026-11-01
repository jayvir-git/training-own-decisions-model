# Blind Codex labeler agreement

This experiment uses the existing Codex ChatGPT login, not API keys. It never
executes source-project scripts. Seven permitted private source files (screening policy
and candidate evidence) were copied, using read-only source opens, into a new
`data/raw/context-<timestamp>/` snapshot and condensed into `data/labeling/context.md`.
That importer and the context are private and not published (see docs/DATA.md). This
separate naming avoids confusing the dataset builder's complete-import discovery.
No commit is part of this workflow.

The frozen `data/labeling/evaluation.json` contains the latest receipt per eligible
posting, split membership, fixed pilot sample and input/protocol hashes. It is only
used by the parent runner. The labeler prompt allowlists employer, title, locations,
posted (null when absent), and the first 12,000 description characters, preceded by
the compact context. Every run starts a new ephemeral session in a new empty
directory under `data/labeling/empty-cwds/`. No receipt, prior vote, posting ID or
split is supplied to the model.

A blinding audit found four captured descriptions with appended `Native screening
notes:`. These suffixes are removed before truncation; the four postings had not
been sent and were not pilot samples. `blinding-cleanup.json` records input hashes
and cut lengths. Every previously accepted prompt hash is checked before resuming,
so existing votes are reused only when their exact input is unchanged. Raw and
processed datasets remain unchanged. Remaining receipt-reason/annotation markers
cause a hard stop rather than a contaminated model call.

Each call uses this command shape (stdin carries the prompt):

```text
<local-codex.exe> --no-daemon -a never exec
  --ignore-user-config --ignore-rules --strict-config --ephemeral
  --skip-git-repo-check -m gpt-6-astra -s read-only --json --color never
  -C <empty-directory> --output-schema <schema.json>
  -c model_reasoning_effort="low" -c model_provider="openai"
  -c forced_login_method="chatgpt" -c project_doc_max_bytes=0
  <tool/plugin/memory/skill/hook disabling overrides> -
```

`scripts/codex_labeler.py::command` defines every override. Exact argv is also
saved for every call in `data/labeling/calls/*/command.json`; JSONL events and stderr
are retained alongside it. Tools are disabled where supported and prohibited by
the prompt; any observed tool item invalidates a response. Read-only sandboxing
alone is not a filesystem-read isolation boundary. Environment API-key/auth-token
variables are removed from the child; authentication uses the existing login.
The CLI's `--ignore-user-config` flag explicitly preserves auth.

Run from the repo:

```powershell
python -B scripts/test_label_agreement.py
python -B scripts/label_agreement.py pilot
powershell -NoProfile -File scripts/start_label_agreement.ps1
python -B scripts/label_agreement.py status
```

The launcher starts a hidden, independent Windows process, with stdout/stderr logs
and a timestamped launch record under `data/labeling/`. It is not a scheduled task.
The computer must stay running. Check the recorded PID with `Get-Process -Id <pid>`.
The worker updates `progress.json`, `agreement-summary.json`, `PILOT.md` and
`AGREEMENT.md` after every bounded batch and when it finishes. `state=complete` and
2,106 valid keys confirm completion for the current 702-posting experiment.

`python -B scripts/label_agreement.py full` resumes missing successful keys after
the passed pilot gate. The append-only log is keyed by posting_id/run, with separate
attempt numbers for failures. An OS-owned writer lock prevents simultaneous
writers and releases on process exit/crash. Corrupt log lines, changed frozen
inputs/protocol or duplicate successful keys fail loudly. Pilot calls are one
attempt each; successful pilot responses are reused as full run 1. Full failures
have at most three attempts per invocation; rate-limit errors halve concurrency
from 4 toward 1 and pause for 30 seconds. Three entirely failed batches stop the
worker. No alternative model/provider is selected.

The pilot is 8 strong, 8 stretch, 7 reject and 7 needs-information train postings,
seed 20261024. Stop conditions: parse/schema failures >10%, projected full duration
>6 hours at concurrency 4 (including observed overhead), or fewer than 27 valid
responses. Runtime and schema failures are recorded separately.

Analysis requires three valid votes per posting. Three distinct votes abstain;
abstentions count as incorrect for primary accuracy and recall, and form a zero-
reference fifth prediction category in Cohen's kappa. The requested 4x4 confusion
matrix lists abstentions separately. Failures/incomplete postings affect coverage,
not the completed-cohort denominator. Current-context evidence and selected posting
text are not guaranteed to match historical reviewer inputs; report those limits.

`python -B scripts/label_agreement.py analyze` refreshes analysis when no worker is
running. All dataset, context and evaluation artifacts stay in gitignored `data/`.

CLI options were checked against the installed binary's help and the official
[noninteractive documentation](https://learn.chatgpt.com/docs/non-interactive-mode)
and [configuration schema](https://learn.chatgpt.com/docs/config-schema.json).

**Public copy.** One phrase of the labeller instructions in `scripts/codex_labeler.py` and
`codex_labeler_v2.py` was made generic for publication, and the private-marker list in the contamination guard moved
to the `TRIAGE_PRIVATE_MARKERS` environment variable. The recorded runs used the original
wording, so their saved prompt hashes will not match prompts built from this copy.
