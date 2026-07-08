"""Start the backend briefly and verify core API endpoints."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "logs" / "smoke_uvicorn.log"
PORT = os.environ.get("SMOKE_PORT", "8010")
BASE_URL = f"http://127.0.0.1:{PORT}"


def fetch(path: str) -> str:
    with urllib.request.urlopen(BASE_URL + path, timeout=5) as response:
        return response.read().decode("utf-8")


def wait_ready() -> None:
    deadline = time.monotonic() + 15
    last_error = None
    while time.monotonic() < deadline:
        try:
            fetch("/")
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.4)
    raise RuntimeError(f"Backend did not become ready: {last_error}")


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PERSON_TRACKING_VIDEO", "data/sample.mp4")
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        PORT,
    ]
    with LOG_PATH.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready()
            for path in ["/", "/status", "/statistics", "/events?limit=5"]:
                print(path, fetch(path))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
