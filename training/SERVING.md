# Local decision serving and export

The implementation uses the installed Unsloth 2026.10.9 `FastDecisionModel` API.
All models, predictions, logs and measurements remain in gitignored `outputs/`;
the training profile and validation inputs remain in gitignored `data/`.
No upload or push helper is called. Hugging Face offline mode and telemetry
opt-outs are enabled before model imports. Python connections outside loopback
are rejected. The base model must already be cached locally.

## Run the Decision API

### Dropped question (2026-10-29)

Posting requests return four answers: `needs_verification` is still computed by
the full-e3 head but removed from the response, because no variant passed the
fair evaluation in `EXPERIMENTS.md` (Exp 4). The earlier opt-in
`--needs-verification-threshold` flag was removed for the same reason. Explicit
`state` + `questions` requests are unaffected and return exactly what they ask for.

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B scripts/serve_decision.py --adapter outputs/full-e3/adapter
```

The server binds only to `127.0.0.1:8765`. Use `--port` to change the port.
It loads once and handles requests sequentially. **It holds the GPU lock for its
entire lifetime, including idle time.** Stop it with Ctrl+C to let queued training
jobs run. Direct invocation also re-enters through `scripts/gpu_lock.py`; lock
ownership is checked through process ancestors, including Windows venv launchers.

`--adapter` accepts any local adapter or merged model directory. `--precision auto`
uses 4-bit for adapters and 16-bit for merged checkpoints; an explicit `4bit` or
`16bit` overrides this. Requantizing a merged model is a separate configuration
and is not the 16-bit verification benchmark.

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
$request = @{
  posting = @{
    title = 'Junior Software Developer'
    company = 'Example Tools Ltd'
    location = 'Remote, United States'
    description = 'Build Python tools and SQL reports. Entry-level applicants welcome. No minimum years of experience stated.'
  }
} | ConvertTo-Json -Depth 10
Invoke-RestMethod http://127.0.0.1:8765/decide -Method Post -ContentType application/json -Body $request
```

Posting requests use `data/training/state-profile.md`, the current
`build_decision_dataset.QUESTIONS`, and `make_state(..., budget=950, count=...)`.
`--profile` and `--budget` are configurable. Token counting uses a separate plain
`AutoTokenizer` for `unsloth/Qwen3.5-0.8B`, not the multimodal processor returned by
the model loader. Answers and question names are not hardcoded into serving.

Alternatively send an explicit state and schema; this does not read the profile:

```json
{
  "state": "Example Tools Ltd offers a fully remote developer role.",
  "questions": {
    "remote": {"type": "noul", "instructions": "Is the role fully remote?"}
  }
}
```

Illustrative response (invented values, not a model measurement):

```json
{
  "answers": {
    "remote": {
      "type": "noul",
      "noul": 0.9,
      "answer": true,
      "probabilities": {"true": 0.9, "false": 0.1}
    }
  },
  "state_truncated": false,
  "latency_seconds": 0.5
}
```

`answers` preserves Unsloth's typed fields: choice answers are strings, noul
answers are booleans, and score answers are integer level indices. Each question
includes probabilities. This is a small local wrapper, not a complete hosted
JEV API implementation. HTTP errors use JSON: 400 invalid input, 404 route,
408 read timeout, 413 oversized/missing body, 500 inference failure.
Bodies are limited to 256 KiB. There is no authentication, TLS, batching or CORS.
The stdlib server is for a trusted local learning workflow, not public deployment.
`/health` is available after load and reports precision, load time and VRAM.

For scripting, save either request form under `data/` and use:

```powershell
.venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B scripts/serve_decision.py --adapter outputs/full-e3/adapter --once data/example-request.json --out outputs/full-e3/once-response.json
```

Omitting `--out` emits JSON to stdout; library diagnostics go to stderr. Keep any
redirected responses under `outputs/`. The CLI releases the lock when it exits.

## Export and verify

```powershell
.venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B scripts/export_model.py --adapter outputs/full-e3/adapter --format merged
.venv/Scripts/python.exe -B scripts/export_model.py --adapter outputs/full-e3/adapter --format gguf
```

The default output is the adapter parent's `export-merged` or `export-gguf`.
Use `--out outputs/<run>/export-<name>` for a fresh run. Nonempty destinations and
destinations overlapping the source adapter are rejected. The source is never
saved over. `--val` and `--reference` allow a later experiment to supply matching
validation inputs and reference predictions. Reference defaults to the adapter
parent's `eval-val-predictions.jsonl`.

Merged export uses `model.save_pretrained_merged(..., save_method='merged_16bit')`.
The output model directory contains the backbone safetensors, tokenizer/processor,
joint decision head and decision configuration with calibrated temperatures.
`save_pretrained_clef` on a LoRA model would save adapters only; it is not a merge.

The verifier measures an adapter reload, exports, then reloads the merged model
in **16-bit**. Each stage has a fresh subprocess under the same GPU lock, so old
weights and CUDA allocations are released before the next stage. All validation
rows are joined by unique `posting_id`; missing IDs, changed questions, changed
probability option sets and nonfinite probabilities fail rather than silently
shrinking the comparison. Agreement covers every typed answer, with per-question
and whole-row agreement. Maximum probability difference uses the probabilities
returned by `predict` (rounded by Unsloth), not unrounded logits.

The merger is pointed at the existing local Hugging Face snapshot in memory;
otherwise this Unsloth version attempts a Hub existence check even with offline
mode enabled. This changes no adapter files. Source file SHA-256 hashes are
recorded and checked again at the end. A full run also checks `/health`, both
`/decide` request forms and the public `--once` CLI, using fictional requests.

Files include `report.json`, per-stage logs, `adapter-check/metrics.json`,
`merged-check/metrics.json`, and both sets of `predictions.jsonl`. Load time starts
before heavy library imports and ends after CUDA synchronization. One warm-up
request is excluded; validation request latencies include synchronized inference.
VRAM uses PyTorch allocated/reserved/peak allocated counters, not total GPU use
by Windows or the CUDA driver. Per-request timings are retained in metrics JSON.

## Other consumers and GGUF limitations

The merged directory includes a vendored `joint_schema_model.py` with
`load_release_model(path, device='cuda', dtype=torch.bfloat16)` and `systemone`.
A non-Unsloth PyTorch consumer must load both backbone and joint head; loading
the backbone as a normal language model does not implement typed decisions.
The reference `systemone` helper applies plain softmax. It does **not** read
Unsloth's saved per-type or option-count calibration. A compatible consumer must
divide each question's logits by the effective temperature before softmax,
including any remaining `head_temperature` and option-count bucket overrides.
Do not apply `folded_temperature` a second time: it is already in the head weights.
Non-Unsloth inference has not been verified by this project.

Unsloth's GGUF exporter first needs merged weights, then a decision-aware
llama.cpp converter (b11443 or newer). It writes the joint head and decision
temperature metadata; Q8_0 is the recommended first comparison. Q8_0/F16/BF16
conversion does not require `llama-quantize`; K-quants do. A decision-capable
`llama-server` provides `/v1/systemone` for non-Unsloth serving and reload checks.

### GGUF result (2026-10-29)

Setup, outside this repo and without compiling: `git clone --depth 1 --branch b11982
https://github.com/ggml-org/llama.cpp ~/.unsloth/llama.cpp` for the converter, plus the
official `llama-b11982-bin-win-cuda-12.4-x64.zip` and matching `cudart` zip unpacked into
`~/.unsloth/llama.cpp/build/bin/Release/` (CUDA 12.4 because the driver supports 13.1).
Then the normal command:

```powershell
.venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B scripts/export_model.py --adapter outputs/full-e3/adapter --format gguf --out outputs/full-e3/export-gguf-q8
```

Output: `native/gguf/model-Q8_0.gguf` (841 MB). The converter also writes
`mmproj-Q8_0.gguf` (116 MB), Qwen3.5's vision projector, which text decisions do not use.
The first run failed at verification: one validation state tokenized to 2,052 tokens in
llama.cpp, above the server's 2,048 batch, so the server now starts with 4,096.

| 70 validation postings | Adapter, 4-bit | Merged, 16-bit | GGUF Q8_0, llama-server |
| --- | ---: | ---: | ---: |
| Verdict agreement with adapter | 70/70 | 65/70 | 65/70 |
| Verdict agreement with merged | - | - | **70/70** |
| Receipt verdict accuracy / macro-F1 (n=63) | 0.810 / 0.691 | 0.810 / 0.723 | 0.810 / 0.723 |
| Other four questions, agreement | 100% | 100% | 100% |
| Load time | 64 s | 59 s | **5.6 s** |
| Mean request latency | 0.39 s | 0.32 s | 0.35 s (HTTP) |
| Size on disk | 113 MB adapter + base | 1.7 GB | **841 MB** |

Q8_0 quantization costs nothing measurable here: it reproduces the merged model exactly
on verdicts. The 5 adapter differences come from merging, not from GGUF, because QLoRA
trained the adapter against a 4-bit base. GGUF VRAM was sampled as total device use
(1,223 MiB, includes the desktop) and is not comparable to the PyTorch counters.
Reports: `outputs/full-e3/export-gguf-q8/report.json` (adapter and merged) and
`gguf-check-2/metrics.json` (GGUF).

## Validation results

CPU contract tests: `python -B -m unittest discover -s tests -p test_serving.py -v`.
Tests exercise live loopback HTTP with a fake engine, request validation, expanded
choice sets, output containment, lock reuse and strict reference comparisons.
GPU export measurements are recorded in `outputs/full-e3/export-merged/report.json`.

Initial adapter reload, 2026-10-28, RTX 3050 Laptop 4 GB:

| Measurement | Adapter, 4-bit |
| --- | ---: |
| Validation rows / typed answers | 70 / 350 |
| Answer agreement with saved reference | 100% |
| Whole-row agreement | 100% |
| Maximum probability difference | 0.0 |
| Load time, including imports | 78.377 s |
| Mean / median request latency | 0.3875 / 0.3907 s |
| p95 request latency | 0.3982 s |
| Loaded allocated VRAM | 1243.5 MiB |
| Peak request allocated VRAM | 1388.3 MiB |
| Reserved VRAM after requests | 1560 MiB |

The first merge failed at the offline Hub existence check. Its failed report and
logs are retained under `outputs/full-e3/export-merged/`. The corrected full run
is queued under the shared GPU lock with destination
`outputs/full-e3/export-merged-local/`; queue diagnostics are in
`outputs/full-e3/export-merged-local-queue.log`. At this writing it is waiting
behind the experiment training run. Merged results and real-model HTTP/CLI
checks are pending, not verified. The CPU suite currently has 10 passing tests.
The baseline above was measured before those added HTTP/CLI smoke checks.
