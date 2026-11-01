"""Local demo server: a small browser UI over the GGUF decision model.

Run from the repository root (PowerShell):

  .venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B demo/app.py

Then open http://127.0.0.1:8770. Serves demo/index.html plus JSON endpoints
POST /api/decide and POST /api/explain (see demo/README.md). Owns a llama-server
child on a private loopback port and stops it on exit; the private candidate
profile is read locally and never sent to the browser. The shared GPU lock
(outputs/GPU.lock) is held for the server's whole lifetime: other GPU jobs
queue until this server is stopped with Ctrl+C.
"""
from __future__ import annotations

import argparse
import atexit
import ipaddress
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

MAX_BODY = 256 * 1024
DEFAULT_PORT = 8770
ALIAS = "local-decision"
LLAMA_SERVER = Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
GGUF_MODEL = REPO / "outputs" / "full-e3" / "export-gguf-q8" / "native" / "gguf" / "model-Q8_0.gguf"
# Still answered by the trained head, but never shown: the question failed fair
# evaluation and was dropped (README.md, Experiments). Mirrors serve_decision.
DROPPED_QUESTIONS = ("needs_verification",)

# Fictional postings written for this demo. Never sourced from data/.
EXAMPLES = [
    {
        "id": "entry-python",
        "label": "Entry-level Python",
        "expect": "strong",
        "posting": {
            "title": "Junior Python Developer",
            "company": "Northwind Analytics Ltd",
            "location": "Remote",
            "description": (
                "Northwind Analytics builds reporting tools for small retailers. We are hiring "
                "a junior Python developer to join a six-person product team. You will write "
                "Python services that clean customer data, build SQL reports, and fix bugs in "
                "existing dashboards. Requirements: a bachelor's degree in any subject or "
                "equivalent practical experience, familiarity with Python and SQL, and comfort "
                "with git. No minimum years of professional experience is required; recent "
                "graduates and career switchers are encouraged to apply. We sponsor work visas "
                "where needed and welcome applicants who need authorization support. The role "
                "is fully remote and asynchronous-friendly, with quarterly team meetups. "
                "Benefits include health cover, 30 days of paid leave, and a yearly learning "
                "budget. Our stack is Python 3.12, PostgreSQL, and pandas, with a little "
                "TypeScript on the front end. The interview process is a short take-home "
                "exercise followed by a conversational design chat."
            ),
        },
    },
    {
        "id": "citizens-only",
        "label": "5+ years, citizens only",
        "expect": "reject",
        "posting": {
            "title": "Senior Firmware Engineer",
            "company": "Ravencrest Defense Systems",
            "location": "Arlington, VA (on-site)",
            "description": (
                "Ravencrest Defense Systems supplies rugged electronics to US defense programs. "
                "We need a senior firmware engineer with 5+ years of experience writing "
                "safety-critical embedded C for microcontrollers. This position requires US "
                "citizenship due to export-control regulations. Applicants must be US citizens; "
                "no visa sponsorship is available under any circumstances and no exceptions are "
                "made for permanent residents. The successful candidate must be able to obtain "
                "and hold an active Department of Defense security clearance. Required: five or "
                "more years of embedded C development, RTOS experience, and familiarity with "
                "DO-178 processes. The role is fully on-site in Arlington, Virginia, five days "
                "per week. Relocation assistance is available. Compensation ranges from "
                "$150,000 to $180,000 depending on clearance level."
            ),
        },
    },
    {
        "id": "mixed-signals",
        "label": "Mid-level, mixed signals",
        "expect": "stretch",
        "posting": {
            "title": "Backend Engineer (Python)",
            "company": "Bluefrog Systems",
            "location": "Hybrid, Austin TX",
            "description": (
                "Bluefrog Systems makes scheduling software for clinics. We are looking for a "
                "backend engineer to own several Python services end to end. You should have "
                "1-2 years of professional experience building production APIs, solid Python "
                "skills, and some exposure to Kubernetes. Experience with GraphQL, Rust, or "
                "event-driven architecture is a plus but not required. The team works in a "
                "hybrid model with two days per week in our Austin office. We sponsor work "
                "visas and support relocation to the United States. A degree or equivalent "
                "experience is accepted. The role includes an on-call rotation of roughly one "
                "week in six."
            ),
        },
    },
    {
        "id": "one-blocker",
        "label": "Good fit, one blocker",
        "expect": "reject",
        "posting": {
            "title": "Graduate Software Engineer",
            "company": "Lanternfish Labs",
            "location": "Remote, United States",
            "description": (
                "Lanternfish Labs builds open-source developer tools for data teams. We are "
                "hiring a graduate software engineer to work on our Python SDK and CLI. You will "
                "write and review Python code, improve test coverage, and help users on GitHub. "
                "A bachelor's degree in computer science or a related field is expected; "
                "internships or personal projects count as experience. No minimum years of "
                "professional experience is required. We are unable to sponsor work visas for "
                "this role, now or in the future. The team is fully remote with flexible hours, "
                "and every engineer gets a yearly learning budget."
            ),
        },
    },
]


def validate_posting(body):
    """The demo accepts exactly {title, company, location, description}."""
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    if set(body) != {"title", "company", "location", "description"}:
        raise ValueError("expected exactly title, company, location and description")
    if any(not isinstance(body[k], str) for k in ("title", "company", "location", "description")):
        raise ValueError("title, company, location and description must be strings")
    if not body["description"].strip():
        raise ValueError("description must not be empty")


class StateBuilder:
    """Builds decision states exactly like serve_decision.DecisionEngine.decide:
    build_decision_dataset.make_state over the private profile, budget 950,
    QUESTIONS, and a plain AutoTokenizer for counting."""

    def __init__(self, profile=None, budget=950, count=None):
        self.profile_path = Path(profile) if profile else REPO / "data" / "training" / "state-profile.md"
        self.budget = budget
        self._count = count  # tests inject a cheap counter; None means the real tokenizer
        self._tokenizer = None
        self._profile_text = None

    def count(self, text):
        if self._count is not None:
            return self._count(text)
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            import build_decision_dataset as bdd
            self._tokenizer = AutoTokenizer.from_pretrained(bdd.DEFAULT_TOKENIZER, local_files_only=True)
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def state_for(self, posting):
        import build_decision_dataset as bdd
        if self._profile_text is None:
            self._profile_text = self.profile_path.read_text(encoding="utf-8")
        p = {"title": posting["title"], "employer": posting["company"],
             "locations": [posting["location"]], "description": posting["description"]}
        state, truncated = bdd.make_state(self._profile_text, p, self.budget, self.count)
        return state, bdd.QUESTIONS, truncated

    def warm(self):
        """Load the profile and tokenizer before serving so the first click is fast."""
        self.state_for({"title": "warm", "company": "warm", "location": "warm",
                        "description": "Warm-up posting for tokenizer loading."})


def decode_answers(questions, answers):
    """Decode llama.cpp's typed answers the way export_model.native_prediction does."""
    out = {}
    for name, question in questions.items():
        answer = answers[name]
        if question["type"] == "noul":
            value = min(max(float(answer["noul"]), 0.0), 1.0)
            out[name] = {"answer": value >= 0.5, "probabilities": {"true": value, "false": 1.0 - value}}
            continue
        probs = {k: float(v) for k, v in answer["probabilities"].items()}
        options = sorted(question["criteria"]) if question["type"] == "choice" \
            else [str(i) for i in range(len(question["criteria"]))]
        if set(options) != set(probs):
            raise ValueError(f"native probability options differ for {name}")
        chosen = max(options, key=probs.__getitem__)
        if question["type"] == "score":
            chosen = int(chosen)
        out[name] = {"answer": chosen, "probabilities": probs}
    return out


_job_handles = []  # kept open for our lifetime; closing them kills the child processes


def _kill_on_close_job(pid):
    """Best effort: put the child in a Windows job that dies when this process dies,
    so llama-server can never outlive the demo even on a hard kill."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        job = kernel32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError("SetInformationJobObject failed")
        process = kernel32.OpenProcess(0x0001 | 0x0100, False, pid)  # TERMINATE | SET_QUOTA
        if not process:
            raise OSError("OpenProcess failed")
        try:
            if not kernel32.AssignProcessToJobObject(job, process):
                raise OSError("AssignProcessToJobObject failed")
        finally:
            kernel32.CloseHandle(process)
        _job_handles.append(job)
        return job
    except Exception:  # best effort only: ctypes or protection errors must not break startup
        return None


class LlamaBackend:
    """Owns a llama-server child serving the GGUF decision model on a private
    loopback port. Flags mirror scripts/export_model.py::measure_gguf."""

    def __init__(self, model=GGUF_MODEL, binary=LLAMA_SERVER, host="127.0.0.1",
                 log=None, ready_timeout=300.0):
        self.process = None
        self.load_seconds = None
        model, binary = Path(model), Path(binary)
        missing = [str(p) for p in (model, binary) if not p.is_file()]
        if missing:
            raise RuntimeError("missing: " + ", ".join(missing))
        self.log = Path(log).resolve() if log else REPO / "outputs" / "demo" / "llama-server.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket() as sock:
            sock.bind((host, 0))
            self.port = sock.getsockname()[1]
        self.base = f"http://{host}:{self.port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        cmd = [str(binary), "-m", str(model), "--alias", ALIAS, "--host", host,
               "--port", str(self.port), "--parallel", "1", "-c", "4096", "-b", "4096",
               "-ub", "4096", "-ngl", "99"]  # llama.cpp tokenizes some states just past 2048
        started = time.perf_counter()
        with self.log.open("w", encoding="utf-8") as handle:
            self.process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
        _kill_on_close_job(self.process.pid)
        deadline = time.monotonic() + ready_timeout
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup; see {self.log}")
            try:
                with self.opener.open(self.base + "/health", timeout=10):
                    break
            except (urllib.error.URLError, TimeoutError, OSError):
                if time.monotonic() >= deadline:
                    self.stop()
                    raise RuntimeError(f"llama-server not ready within {ready_timeout:.0f}s; see {self.log}")
                time.sleep(0.5)
        self.load_seconds = time.perf_counter() - started

    def describe(self):
        return {"kind": "llama-server", "alias": ALIAS, "port": self.port, "log": str(self.log)}

    def decide(self, state, questions):
        request = urllib.request.Request(
            self.base + "/v1/systemone",
            data=json.dumps({"model": ALIAS, "state": state, "questions": questions}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=120) as response:
            return json.load(response)["answers"]

    def stop(self):
        process, self.process = self.process, None
        if not process or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class DemoApp:
    """Ties the state builder to a backend. The profile text only ever exists
    inside state strings passed to the local backend; it never enters a response."""

    def __init__(self, builder, backend):
        self.builder, self.backend = builder, backend

    def health(self):
        return {"status": "ok", "backend": self.backend.describe(),
                "load_seconds": self.backend.load_seconds}

    def decide_posting(self, posting):
        started = time.perf_counter()
        state, questions, truncated = self.builder.state_for(posting)
        answers = decode_answers(questions, self.backend.decide(state, questions))
        answers = {k: v for k, v in answers.items() if k not in DROPPED_QUESTIONS}
        return {"answers": answers, "state_truncated": truncated,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    def decide_probabilities(self, posting):
        """Callable handed to explain_decision.explain_posting."""
        state, questions, _ = self.builder.state_for(posting)
        answers = decode_answers(questions, self.backend.decide(state, questions))
        return {k: v["probabilities"] for k, v in answers.items() if k not in DROPPED_QUESTIONS}


def load_explainer():
    """Imported per request so a module missing at runtime yields 501, not a crash."""
    from explain_decision import explain_posting
    return explain_posting


class _BodySize(Exception):
    pass


def handler_for(app, explainer_loader=None):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(300)  # explanations re-run the model per sentence

        def log_message(self, *args):
            pass  # No posting text, states or answers in access logs.

        def send_json(self, status, body):
            raw = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                raw = (DEMO / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if self.path == "/health":
                self.send_json(200, app.health())
                return
            if self.path == "/api/examples":
                self.send_json(200, EXAMPLES)
                return
            self.send_json(404, {"error": "not found"})

        def read_body(self):
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("chunked requests are unsupported")
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n <= MAX_BODY:
                raise _BodySize()
            return json.loads(self.rfile.read(n))

        def do_POST(self):
            try:
                body = self.read_body()
            except _BodySize:
                self.send_json(413, {"error": f"body must be 1..{MAX_BODY} bytes"})
                return
            except (ValueError, UnicodeDecodeError):
                self.send_json(400, {"error": "invalid JSON body"})
                return
            if self.path not in ("/api/decide", "/api/explain"):
                self.send_json(404, {"error": "not found"})
                return
            try:
                validate_posting(body)
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
                return
            posting = {k: body[k] for k in ("title", "company", "location", "description")}
            if self.path == "/api/decide":
                try:
                    self.send_json(200, app.decide_posting(posting))
                except (socket.timeout, TimeoutError):
                    self.send_json(408, {"error": "request timeout"})
                except Exception:
                    self.send_json(500, {"error": "decision backend failed"})
                return
            try:
                explainer = (explainer_loader or load_explainer)()
            except ImportError:
                self.send_json(501, {"error": "sentence-level explanation is unavailable"})
                return
            try:
                self.send_json(200, explainer(posting, app.decide_probabilities))
            except (socket.timeout, TimeoutError):
                self.send_json(408, {"error": "request timeout"})
            except Exception:
                self.send_json(500, {"error": "explanation failed"})
    return Handler


def offline():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "UNSLOTH_DISABLE_STATISTICS"):
        os.environ[key] = "1"

    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            host = args[1][0] if event == "socket.connect" else args[0]
            if host == "localhost":
                return
            try:
                if ipaddress.ip_address(host).is_loopback:
                    return
            except ValueError:
                pass
            raise OSError("offline demo blocks non-loopback network access")

    sys.addaudithook(audit)


def under_gpu_lock():
    """Re-enter through the shared wrapper, or reuse an ancestor's lock
    (same approach as serve_decision.under_gpu_lock)."""
    import gpu_lock
    holder = gpu_lock.holder()
    import psutil
    ancestors = {p.pid for p in psutil.Process().parents()}
    if holder and holder["pid"] in ancestors and gpu_lock.alive(holder["pid"]):
        os.environ["DECISION_GPU_LOCK_PID"] = str(holder["pid"])
        return
    command = [sys.executable, "-B", str(REPO / "scripts/gpu_lock.py"), "--",
               sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, cwd=REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", type=Path, default=GGUF_MODEL, help="GGUF decision model")
    parser.add_argument("--binary", type=Path, default=LLAMA_SERVER, help="llama-server executable")
    parser.add_argument("--profile", type=Path, default=REPO / "data" / "training" / "state-profile.md")
    parser.add_argument("--budget", type=int, default=950, help="posting token budget")
    parser.add_argument("--llama-log", type=Path, default=REPO / "outputs" / "demo" / "llama-server.log")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.budget <= 0:
        parser.error("port and budget must be positive; port <= 65535")
    args.model, args.binary = args.model.resolve(), args.binary.resolve()
    args.profile, args.llama_log = args.profile.resolve(), args.llama_log.resolve()
    under_gpu_lock()
    offline()
    builder = StateBuilder(args.profile, args.budget)
    backend = LlamaBackend(args.model, args.binary, log=args.llama_log)
    atexit.register(backend.stop)
    builder.warm()
    app = DemoApp(builder, backend)
    with HTTPServer(("127.0.0.1", args.port), handler_for(app)) as server:
        print(f"demo ready at http://127.0.0.1:{args.port} (pid {os.getpid()}); "
              f"llama-server on 127.0.0.1:{backend.port}; GPU lock held until exit; "
              f"Ctrl+C to stop", file=sys.stderr, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
