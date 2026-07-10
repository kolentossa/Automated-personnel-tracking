#!/usr/bin/env python3
"""Download and verify Rockchip model-zoo RetinaFace Mobile320."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "RetinaFace_mobile320.onnx"
MODEL_URL = (
    "https://ftrg.zbox.filez.com/v2/delivery/data/"
    "95f00b0fc900458ba134f8b180b3f7a1/examples/RetinaFace/RetinaFace_mobile320.onnx"
)
EXPECTED_SHA256 = "1061ac88e7c833cf058c3e8eb50367dc5e3daadcc14967b5dede8ac4409b86fa"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the verified RetinaFace Mobile320 ONNX model.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()

    if output.is_file() and _sha256(output) == EXPECTED_SHA256:
        print(f"model: {output}")
        print(f"sha256: {EXPECTED_SHA256}")
        print("status: already verified")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=180) as response, temporary.open("wb") as destination:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)
        digest = _sha256(temporary)
        if digest != EXPECTED_SHA256:
            print(f"error: sha256 mismatch: {digest}", file=sys.stderr)
            return 2
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"model: {output}")
    print(f"sha256: {EXPECTED_SHA256}")
    print("status: downloaded and verified")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
