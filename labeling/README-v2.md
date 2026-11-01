# Blind agreement v2

All outputs live under gitignored `data/labeling/`. No private source files are accessed. Do not run source import/build scripts for this experiment. No commits.

1. `python -B scripts/test_label_agreement_v2.py`
2. `python -B scripts/label_agreement_v2.py prepare` freezes train/val selection.
3. `powershell -NoProfile -File scripts/start_label_agreement_v2.ps1 -Phase tune`
4. Inspect `v2/tuning-summary-initial.json` after completion. A single documented revision may run with `-Phase tune -Version revision` only if the complete initial weighted score is below 85%. Its context must be saved as `v2/context-revision.md` before running. Do not inspect test results at this stage.
5. `python -B scripts/label_agreement_v2.py choose` picks the better allowed context and freezes `v2/choice.json`.
6. Report measured tuning timing and projected duration before `powershell -NoProfile -File scripts/start_label_agreement_v2.ps1` launches the final detached worker.
7. `python -B scripts/label_agreement_v2.py status` displays progress/PID. The worker refreshes `AGREEMENT-v2.md` every minute and at completion. `python -B scripts/label_agreement_v2.py analyze` refreshes it manually when no writer is active.

After selection, `python -B scripts/audit_rules_v2.py --final` builds the conservative rule-conflict audit from all 702 receipts. Without `--final`, it restricts itself to train/val. Human-verified findings are stored separately from unconfirmed text-pattern candidates; the latter are not silently counted as contradictory labels. The report clearly identifies the confirmed count as a lower bound. The full worker picks up the latest audit on its next report refresh.

Fresh final votes are in `v2/runs/full.jsonl`, initial and revision tuning in separate JSONL files. Existing successes are never re-run or overwritten. A process interruption may leave an unlogged in-flight call artifact; a new attempt uses a distinct artifact directory. Failed attempts remain logged. After three failed attempts for a key the worker reports incomplete coverage; it does not retry indefinitely.

The chosen context, schema, instructions, cleaned prompt hashes and input file hashes are frozen. Edits to the chosen context or labeler invalidate resume, rather than mixing protocols. The runner uses Python's standard library and retains the v1 isolation flags. See `CHANGES-v2.md` for exact scoring populations and the revision decision.
