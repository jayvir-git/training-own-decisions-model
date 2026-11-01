"""Local, offline Decision API. Heavy imports happen only after acquiring GPU.lock."""
from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO = Path(__file__).resolve().parents[1]
MAX_BODY = 256 * 1024
# Still in the trained head, but dropped from posting responses: no fair test beat chance (EXPERIMENTS.md Exp 4).
DROPPED_QUESTIONS = ("needs_verification",)


def local_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPO / "outputs") or path == REPO / "outputs":
        raise ValueError("output must be below the repository's outputs/ directory")
    return path


def offline():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "UNSLOTH_DISABLE_STATISTICS"):
        os.environ[key] = "1"
    # Also fail closed if a dependency ignores the Hugging Face offline flags.
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
            raise OSError("offline serving blocks non-loopback network access")
    sys.addaudithook(audit)


def under_gpu_lock():
    """Re-enter through the shared wrapper, or reuse its lock in export workers."""
    import gpu_lock
    h = gpu_lock.holder()
    # Windows venv python.exe is a redirector: the wrapper can be a grandparent.
    # psutil is already installed with the training runtime; no torch import here.
    import psutil
    ancestors = {p.pid for p in psutil.Process().parents()}
    if h and h["pid"] in ancestors and gpu_lock.alive(h["pid"]):
        os.environ["DECISION_GPU_LOCK_PID"] = str(h["pid"])
        return
    command = [sys.executable, "-B", str(REPO / "scripts/gpu_lock.py"), "--",
               sys.executable, "-B", str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, cwd=REPO))


def validate_request(body):
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    if "posting" in body:
        if set(body) != {"posting"} or not isinstance(body["posting"], dict):
            raise ValueError("use either posting or state/questions")
        p = body["posting"]
        if any(not isinstance(p.get(k), str) for k in ("title", "company", "location", "description")):
            raise ValueError("posting requires string title, company, location and description")
        if not p["description"].strip():
            raise ValueError("posting description must not be empty")
        return
    if set(body) != {"state", "questions"} or not isinstance(body.get("state"), str):
        raise ValueError("provide string state and questions object")
    questions = body["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    for name, q in questions.items():
        if not isinstance(name, str) or not name or not isinstance(q, dict) or q.get("type") not in ("choice", "noul", "score"):
            raise ValueError("each named question needs type choice, noul or score")
        if not isinstance(q.get("instructions"), str):
            raise ValueError("question instructions must be a string")
        criteria = q.get("criteria")
        if q["type"] == "noul" and criteria is not None and (not isinstance(criteria, dict)
                or any(k not in ("true", "false") or not isinstance(v, str) for k, v in criteria.items())):
            raise ValueError("noul criteria may describe true and false only")
        if q["type"] == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2
                or any(not k or not isinstance(v, str) for k, v in criteria.items())):
            raise ValueError("choice criteria must map at least two answer names to descriptions")
        if q["type"] == "score" and (not isinstance(criteria, list) or len(criteria) < 2
                or any(not isinstance(v, str) for v in criteria)):
            raise ValueError("score criteria must list at least two descriptions")


class DecisionEngine:
    def __init__(self, adapter, precision="auto", profile=None, budget=950):
        offline()
        started = time.perf_counter()
        from unsloth import FastDecisionModel
        import torch
        self.api, self.torch = FastDecisionModel, torch
        path = Path(adapter).resolve()
        if not path.is_dir():
            raise ValueError("model must be an existing local directory")
        self.precision = ("4bit" if (path / "adapter_config.json").exists() else "16bit") if precision == "auto" else precision
        torch.cuda.reset_peak_memory_stats()
        self.model, self.tokenizer = FastDecisionModel.from_pretrained(
            str(path), max_seq_length=2048, load_in_4bit=self.precision == "4bit",
            local_files_only=True)
        FastDecisionModel.for_inference(self.model)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.load_vram = self.vram()
        self.profile_path = Path(profile) if profile else REPO / "data/training/state-profile.md"
        self.budget = budget
        self.text_tokenizer = None

    def vram(self):
        cuda = self.torch.cuda
        return {"allocated_mib": cuda.memory_allocated() / 2**20,
                "reserved_mib": cuda.memory_reserved() / 2**20,
                "peak_allocated_mib": cuda.max_memory_allocated() / 2**20}

    def decide(self, body):
        validate_request(body)
        started = time.perf_counter()
        truncated = False
        if "posting" in body:
            import build_decision_dataset as bdd
            if self.text_tokenizer is None:
                from transformers import AutoTokenizer
                self.text_tokenizer = AutoTokenizer.from_pretrained(bdd.DEFAULT_TOKENIZER, local_files_only=True)
            p = body["posting"]
            posting = {"title": p["title"], "employer": p["company"],
                       "locations": [p["location"]], "description": p["description"]}
            state, truncated = bdd.make_state(self.profile_path.read_text(encoding="utf-8"), posting,
                self.budget, lambda s: len(self.text_tokenizer(s, add_special_tokens=False)["input_ids"]))
            questions = bdd.QUESTIONS
        else:
            state, questions = body["state"], body["questions"]
        self.torch.cuda.synchronize()
        with self.torch.inference_mode():
            answers = self.api.predict(self.model, self.tokenizer, state, questions)
        self.torch.cuda.synchronize()
        if "posting" in body:
            answers = {k: v for k, v in answers.items() if k not in DROPPED_QUESTIONS}
        return {"answers": answers, "state_truncated": truncated,
                "latency_seconds": time.perf_counter() - started}


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass  # No states, postings or model answers in access logs.

        def send_json(self, status, body):
            raw = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.send_json(200, {"status": "ok", "precision": engine.precision,
                "load_seconds": engine.load_seconds, "vram": engine.vram()}) if self.path == "/health" else self.send_json(404, {"error": "not found"})

        def do_POST(self):
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("chunked requests are unsupported")
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= MAX_BODY:
                    self.send_json(413, {"error": "body must be 1..262144 bytes"})
                    return
                raw = self.rfile.read(n)
                if self.path != "/decide":
                    self.send_json(404, {"error": "not found"})
                    return
                body = json.loads(raw)
                validate_request(body)
                result = engine.decide(body)
            except (ValueError, UnicodeDecodeError):
                self.send_json(400, {"error": "invalid JSON or decision request"})
                return
            except (socket.timeout, TimeoutError):
                self.send_json(408, {"error": "request timeout"})
                return
            except Exception:
                self.send_json(500, {"error": "inference failed"})
                return
            self.send_json(200, result)
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="outputs/full-e3/adapter", help="adapter or merged local directory")
    p.add_argument("--precision", choices=("auto", "4bit", "16bit"), default="auto")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--profile", type=Path)
    p.add_argument("--budget", type=int, default=950)
    p.add_argument("--once", type=Path)
    p.add_argument("--out", type=Path, help="once response file below outputs/ (otherwise stdout)")
    args = p.parse_args()
    if args.budget <= 0 or not 1 <= args.port <= 65535:
        p.error("budget and port must be positive; port <= 65535")
    out = local_output(args.out) if args.out else None
    body = json.loads(args.once.read_text(encoding="utf-8-sig")) if args.once else None
    if args.once:
        validate_request(body)
    under_gpu_lock()
    with contextlib.redirect_stdout(sys.stderr):
        engine = DecisionEngine(args.adapter, args.precision, args.profile, args.budget)
        result = engine.decide(body) if args.once else None
    if args.once:
        result["metrics"] = {"load_seconds": engine.load_seconds, "vram": engine.vram()}
        raw = json.dumps(result, indent=2, allow_nan=False)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(raw, encoding="utf-8")
        else:
            print(raw)
        return
    with HTTPServer(("127.0.0.1", args.port), handler_for(engine)) as server:
        print(f"Decision API ready at http://127.0.0.1:{args.port}; GPU lock held until exit", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
