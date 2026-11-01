# Local decision demo

A small, fully offline browser demo of the typed decision model: paste a job
posting, see typed answers with probability bars and latency, and a "why" view
that highlights which sentences pushed the verdict toward reject versus toward
a good fit. Built for a ~20-second screen recording; nothing is uploaded and
no external fonts, scripts or CDNs are used.

## Run

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B demo/app.py
```

Then open <http://127.0.0.1:8770> (port 8770 is the default; `--port` changes
it). Stop with Ctrl+C.

**GPU lock.** The demo runs the GGUF model on the shared 4 GB GPU, so it holds
the repository's GPU lock (`outputs/GPU.lock`) for its entire lifetime,
including idle time. Other GPU jobs (training, exports) queue behind it. Stop
the demo when you are done. `demo/app.py` re-enters through
`scripts/gpu_lock.py` if it was not launched under the wrapper, the same way
`scripts/serve_decision.py` does.

## What it holds

| File | Role |
|---|---|
| `demo/app.py` | Standard-library HTTP server on 127.0.0.1. Starts and owns a `llama-server` child (GGUF Q8_0 model, flags mirroring `scripts/export_model.py::measure_gguf`) on a private loopback port, and stops it on exit. Builds decision states exactly like `serve_decision.DecisionEngine.decide` (`build_decision_dataset.make_state`, private profile, 950-token budget, `QUESTIONS`, `AutoTokenizer` for counting). |
| `demo/index.html` | Single-file UI, inline CSS/JS. Left: posting form with three fictional built-in examples. Right: verdict chip and bars, yes/no gate chips, required-years buckets, latency, and the sentence-level "why" view. |
| `demo/screenshot.png` | Optional screenshot of the running UI, when captured. |

Endpoints (loopback only):

- `GET /` — the UI. `GET /health` — backend status. `GET /api/examples` — the built-in fictional postings.
- `POST /api/decide` `{title, company, location, description}` — typed answers
  with probabilities, `state_truncated`, `latency_ms`. `needs_verification` is
  computed by the head but dropped from every response (it failed fair
  evaluation and was removed from serving).
- `POST /api/explain` (same body) — sentence-level explanation from
  `scripts/explain_decision.py::explain_posting`. If that module is missing,
  the endpoint returns 501 and the UI shows the "why" panel as unavailable.
- Deep links: `http://127.0.0.1:8770#example=one-blocker&why=1` fills the
  form, runs the decision, and opens the "why" view — handy for starting a
  screen recording. Example ids: `entry-python`, `citizens-only`,
  `mixed-signals`, `one-blocker`.
- Use `one-blocker` for the "why" view. It is 82% reject, and removing only its visa
  sentence makes it strong (−0.67). `citizens-only` has three overlapping
  reasons, so no single sentence moves it; the page says so rather than
  listing near-zero "drivers" (effects under 0.02 are hidden).
- The examples were written by hand and checked against the live model, and
  one draft was rewritten after the model rated it differently. They show
  the interface; they are not evidence of accuracy.

## Privacy and limits

- The private candidate profile (`data/training/state-profile.md`; start from
  `examples/state-profile.example.md`) is read locally and embedded into model input
  states only. It is never sent to the browser, logged, or included in any
  response or error message.
- Trusted local use only: no authentication, TLS, CORS or multi-user support;
  single request at a time; bodies capped at 256 KiB, as in `serve_decision`.
- The three built-in example postings are fiction written for this demo. Real
  postings from `data/` are never used as examples.
- If the demo process is killed without Ctrl+C, a Windows job object still
  terminates the `llama-server` child, so no GPU process is left behind.

## Tests

CPU-only contract tests with a fake backend (no GPU, no llama-server):

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests
```

`tests/test_demo.py` covers request validation, JSON shapes, 501 behavior when
the explanation module is absent, and that the private profile never appears
in any response.
