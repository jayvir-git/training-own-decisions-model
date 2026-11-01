"""One GPU job at a time (4 GB RTX 3050). Wrap any GPU-using command:

  .venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B scripts/train_decision.py ...
  .venv/Scripts/python.exe -B scripts/gpu_lock.py --status

Waits until outputs/GPU.lock is free (a lock whose PID is dead is treated as stale), holds it while
the command runs, then releases it. The lock file records owner PID, command and start time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

LOCK = Path(__file__).resolve().parents[1] / "outputs" / "GPU.lock"


def alive(pid):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def holder():
    try:
        return json.loads(LOCK.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def acquire(cmd, poll=30):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    announced = False
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "cmd": cmd, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
            return
        except FileExistsError:
            h = holder()
            if h and not alive(h["pid"]):
                LOCK.unlink(missing_ok=True)
                continue
            if not announced:
                print(f"gpu_lock: waiting for {h}", file=sys.stderr, flush=True)
                announced = True
            time.sleep(poll)


def main():
    if sys.argv[1:] == ["--status"]:
        print(json.dumps(holder()))
        return 0
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    cmd = sys.argv[sys.argv.index("--") + 1:]
    if cmd and Path(cmd[0]).exists():  # Windows CreateProcess won't resolve ./relative/exe paths
        cmd[0] = str(Path(cmd[0]).resolve())
    acquire(" ".join(cmd))
    try:
        return subprocess.call(cmd)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
