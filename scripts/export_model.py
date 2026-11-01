"""Export locally, then measure fresh-process reloads against saved validation predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from serve_decision import DecisionEngine, REPO, handler_for, local_output, offline, under_gpu_lock
from http.server import HTTPServer


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fingerprints(adapter, val, reference):
    paths = [p for p in adapter.iterdir() if p.is_file()] + [val, reference]
    result = {}
    for path in paths:
        with path.open("rb") as f:
            result[str(path)] = hashlib.file_digest(f, "sha256").hexdigest()
    return result


def index_rows(rows):
    indexed = {r["posting_id"]: r for r in rows}
    if len(indexed) != len(rows) or not rows:
        raise ValueError("validation/reference must be nonempty with unique posting_id values")
    return indexed


def compare(reference, actual):
    ref, got = index_rows(reference), index_rows(actual)
    if ref.keys() != got.keys():
        raise ValueError("validation and reference posting IDs differ")
    matches = total = exact_rows = 0
    maximum = 0.0
    per_question = {}
    for pid, r in ref.items():
        a = got[pid]
        if not r["pred"] or r["pred"].keys() != a["pred"].keys():
            raise ValueError("question names differ from the saved reference")
        row_ok = True
        for name, answer in r["pred"].items():
            equal = type(answer) is type(a["pred"][name]) and answer == a["pred"][name]
            row_ok &= equal
            matches += equal
            total += 1
            counts = per_question.setdefault(name, {"matches": 0, "n": 0})
            counts["matches"] += equal
            counts["n"] += 1
            rp, ap = r["probabilities"][name], a["probabilities"][name]
            if not rp or rp.keys() != ap.keys():
                raise ValueError("probability option names differ from the saved reference")
            for option in rp:
                x, y = float(rp[option]), float(ap[option])
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
                    raise ValueError("invalid probability")
                maximum = max(maximum, abs(x - y))
        exact_rows += row_ok
    return {"rows": len(ref), "answers": total, "answer_agreement": matches / total,
            "exact_row_agreement": exact_rows / len(ref), "max_probability_difference": maximum,
            "per_question": {k: {**v, "agreement": v["matches"] / v["n"]} for k, v in per_question.items()}}


def gguf_prerequisites():
    folder = Path(os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR") or
                  os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or Path.home() / ".unsloth/llama.cpp")
    required = [folder / "convert_hf_to_gguf.py", folder / "conversion/clef.py",
                folder / "gguf-py/gguf/constants.py"]
    missing = [str(p) for p in required if not p.is_file()]
    if not missing:
        constants = required[-1].read_text(encoding="utf-8")
        if "{arch}.decision.temperature.{name}" not in constants or "CLEF" not in constants:
            missing.append("llama.cpp b11443+ decision metadata support")
    server = shutil.which("llama-server") or next((str(p) for p in
        (folder / "build/bin/Release/llama-server.exe", folder / "build/bin/llama-server.exe",
         folder / "llama-server.exe") if p.is_file()), None)
    if not server:
        missing.append("llama-server with /v1/systemone support (b11443+) for reload verification")
    return folder, server, missing


def native_prediction(row, response):
    """Decode llama.cpp's typed endpoint without assuming today's question names."""
    answers = response["answers"]
    pred, probabilities = {}, {}
    for name, question in row["questions"].items():
        answer = answers[name]
        if question["type"] == "noul":
            value = float(answer["noul"])
            probs = {"true": value, "false": 1 - value}
            chosen = value >= .5
        else:
            probs = answer["probabilities"]
            options = sorted(question["criteria"]) if question["type"] == "choice" else [str(i) for i in range(len(question["criteria"]))]
            if set(options) != set(probs):
                raise ValueError("native probability options differ")
            chosen = max(options, key=probs.__getitem__)
            if question["type"] == "score":
                chosen = int(chosen)
        pred[name], probabilities[name] = chosen, probs
    return {"posting_id": row["posting_id"], "pred": pred, "probabilities": probabilities}


def measure_gguf(args, out):
    _, binary, missing = gguf_prerequisites()
    if missing:
        raise RuntimeError("; ".join(missing))
    rows, reference = read_rows(args.val), read_rows(args.reference)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{port}"
    def request(path, body=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=120) as response:
            return json.load(response)
    start = time.perf_counter()
    cmd = [binary, "-m", str(args.adapter), "--alias", "local-decision", "--host", "127.0.0.1",
           "--port", str(port), "--parallel", "1", "-c", "4096", "-b", "4096", "-ub", "4096", "-ngl", "99"]  # llama.cpp tokenizes some states just past 2048
    with (out / "llama-server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError("llama-server exited; see llama-server.log")
                try:
                    request("/health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("llama-server did not become ready within 300 seconds")
                    time.sleep(.5)
            load_seconds = time.perf_counter() - start
            def decide(row):
                return request("/v1/systemone", {"model": "local-decision", "state": row["state"], "questions": row["questions"]})
            decide(rows[0])
            predictions, times, vram = [], [], []
            for row in rows:
                start = time.perf_counter()
                answer = decide(row)
                times.append(time.perf_counter() - start)
                predictions.append(native_prediction(row, answer))
                sample = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                if sample.returncode == 0:
                    vram.append(float(sample.stdout.splitlines()[0]))
            with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
                for row in predictions:
                    f.write(json.dumps(row) + "\n")
            return {"verification": compare(reference, predictions), "load_seconds": load_seconds,
                "precision": "Q8_0", "latency_seconds": {"n": len(times), "mean": statistics.mean(times),
                    "median": statistics.median(times), "p95": sorted(times)[math.ceil(.95 * len(times)) - 1]},
                "per_request_seconds": times, "sampled_device_vram_mib": max(vram) if vram else None,
                "vram_note": "Total device memory sampled after each request; includes driver/desktop, not a peak allocation counter."}
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def measure(engine, rows, reference, out):
    predictions, latencies = [], []
    # One warm-up, excluded from latency distribution; load time includes library import.
    engine.decide({k: rows[0][k] for k in ("state", "questions")})
    engine.torch.cuda.reset_peak_memory_stats()
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            result = engine.decide({k: row[k] for k in ("state", "questions")})
            answers = result["answers"]
            prediction = {"posting_id": row["posting_id"],
                "pred": {k: v["answer"] for k, v in answers.items()},
                "probabilities": {k: v["probabilities"] for k, v in answers.items()}}
            f.write(json.dumps(prediction) + "\n")
            predictions.append(prediction)
            latencies.append(result["latency_seconds"])
    ordered = sorted(latencies)
    return {"verification": compare(reference, predictions), "load_seconds": engine.load_seconds,
            "precision": engine.precision, "load_vram": engine.load_vram, "request_vram": engine.vram(),
            "latency_seconds": {"n": len(latencies), "mean": statistics.mean(latencies),
                "median": statistics.median(latencies), "p95": ordered[math.ceil(.95 * len(ordered)) - 1],
                "min": min(latencies), "max": max(latencies)}, "per_request_seconds": latencies}


def smoke_http(engine, out):
    """Exercise real HTTP transport and both request forms after benchmarking."""
    server = HTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = {}
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with opener.open(base + "/health", timeout=10) as response:
            results["health"] = json.load(response)
        requests = {
            "state": {"state": "Example Tools Ltd offers a fully remote developer role.",
                "questions": {"remote": {"type": "noul", "instructions": "Is the role fully remote?"}}},
            "posting": {"posting": {"title": "Junior Software Developer", "company": "Example Tools Ltd",
                "location": "Remote, United States", "description": "Build Python tools and SQL reports. Entry-level applicants welcome. No minimum years of experience stated."}}}
        for name, body in requests.items():
            req = urllib.request.Request(base + "/decide", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=180) as response:
                results[name] = json.load(response)
        (out / "http-smoke.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def worker(args):
    offline()
    out = local_output(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.worker == "merge":
        engine = DecisionEngine(args.adapter, "4bit")
        # The installed merger probes the Hub for a repo ID even in offline mode.
        # Resolve the exact cached base first and change only the in-memory source.
        from huggingface_hub import snapshot_download
        config = engine.model.decision_config
        base = config["base_model"]
        source = str(Path(base).resolve()) if Path(base).is_dir() else snapshot_download(
            base, revision=config.get("base_revision") or getattr(engine.model.encoder.config, "_commit_hash", None),
            local_files_only=True)
        engine.model.encoder.config._name_or_path = source
        start = time.perf_counter()
        engine.model.save_pretrained_merged(str(out / "model"), engine.tokenizer, save_method="merged_16bit")
        result = {"status": "exported", "export_seconds": time.perf_counter() - start,
                  "local_base_source": source}
    elif args.worker == "convert":
        folder, _, missing = gguf_prerequisites()
        if missing:
            raise RuntimeError("; ".join(missing))
        os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] = str(folder)
        from unsloth.models.decision_gguf import export_decision_gguf
        result = export_decision_gguf(str(args.adapter), quantization_method="q8_0",
            output_dir=str(out / "gguf"), print_output=True)
    elif args.worker == "measure-gguf":
        result = measure_gguf(args, out)
    else:
        rows, reference = read_rows(args.val), read_rows(args.reference)
        index_rows(rows)
        index_rows(reference)
        engine = DecisionEngine(args.adapter, args.precision)
        result = measure(engine, rows, reference, out)
        smoke_http(engine, out)
    (out / ("export.json" if args.worker in ("merge", "convert") else "metrics.json")).write_text(
        json.dumps(result, indent=2), encoding="utf-8")


def run_worker(args, kind, adapter, out, precision="auto"):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--adapter", str(adapter),
        "--format", args.format, "--out", str(out), "--worker", kind,
        "--precision", precision, "--val", str(args.val), "--reference", str(args.reference)]
    with (out / f"{kind}.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, check=True, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--format", required=True, choices=("merged", "gguf"))
    p.add_argument("--out", type=Path)
    p.add_argument("--val", type=Path, default=REPO / "data/training/val.jsonl")
    p.add_argument("--reference", type=Path, help="defaults to adapter parent's eval-val-predictions.jsonl")
    p.add_argument("--worker", choices=("merge", "measure", "convert", "measure-gguf"), help=argparse.SUPPRESS)
    p.add_argument("--precision", choices=("auto", "4bit", "16bit"), default="auto", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.adapter = args.adapter.resolve()
    args.reference = (args.reference or args.adapter.parent / "eval-val-predictions.jsonl").resolve()
    args.val = args.val.resolve()
    out = local_output(args.out or args.adapter.parent / f"export-{args.format}")
    args.out = out
    if out == args.adapter or out.is_relative_to(args.adapter) or args.adapter.is_relative_to(out):
        p.error("export directory must not overlap the source adapter")
    if args.worker:
        under_gpu_lock()
        worker(args)
        return
    if out.exists() and any(out.iterdir()):
        p.error("output already exists and is nonempty; choose a new --out")
    # Fail before any model load, installation, download or conversion.
    if args.format == "gguf":
        folder, server, missing = gguf_prerequisites()
        if missing:
            out.mkdir(parents=True, exist_ok=True)
            report = {"status": "unavailable", "missing": missing,
                      "converter_directory": str(folder), "minimum_llama_cpp": "b11443",
                      "note": "No tools downloaded or installed; no GGUF produced."}
            (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            raise SystemExit(2)
    rows, reference = read_rows(args.val), read_rows(args.reference)
    if index_rows(rows).keys() != index_rows(reference).keys():
        p.error("validation and reference IDs differ")
    under_gpu_lock()
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "adapter_path": str(args.adapter), "format": args.format,
              "reference": str(args.reference), "validation": str(args.val),
              "source_sha256": fingerprints(args.adapter, args.val, args.reference)}
    try:
        print("Measuring adapter reload", flush=True)
        run_worker(args, "measure", args.adapter, out / "adapter-check", "4bit")
        report["adapter"] = json.loads((out / "adapter-check/metrics.json").read_text())
        print("Exporting merged 16-bit weights", flush=True)
        run_worker(args, "merge", args.adapter, out)
        print("Measuring merged 16-bit reload", flush=True)
        run_worker(args, "measure", out / "model", out / "merged-check", "16bit")
        report["merged"] = json.loads((out / "merged-check/metrics.json").read_text())
        if args.format == "gguf":
            run_worker(args, "convert", out / "model", out / "native")
            run_worker(args, "measure-gguf", out / "native/gguf/model-Q8_0.gguf", out / "gguf-check")
            report["gguf"] = json.loads((out / "gguf-check/metrics.json").read_text())
        # Public CLI smoke check in another fresh process, still under this lock.
        request_file = out / "example-request.json"
        request_file.write_text(json.dumps({"state": "Example Tools Ltd offers a fully remote developer role.",
            "questions": {"remote": {"type": "noul", "instructions": "Is the role fully remote?"}}}), encoding="utf-8")
        with (out / "once.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-B", str(REPO / "scripts/serve_decision.py"),
                "--adapter", str(out / "model"), "--once", str(request_file),
                "--out", str(out / "once-response.json")], check=True, stdout=log, stderr=subprocess.STDOUT)
        if fingerprints(args.adapter, args.val, args.reference) != report["source_sha256"]:
            raise RuntimeError("source adapter, validation or reference changed during export")
        report["matches_reference_exactly"] = all(
            report[k]["verification"]["answer_agreement"] == 1.0
            and report[k]["verification"]["max_probability_difference"] == 0.0
            for k in ("adapter", "merged", "gguf") if k in report)
        report["status"] = "verified" if report["matches_reference_exactly"] else "verified_with_differences"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
