#!/usr/bin/env python3
"""Download the pinned, licensed DAMO cigarette checkpoint and verify it."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = (
    "https://www.modelscope.cn/iic/cv_tinynas_object-detection_damoyolo_cigarette.git"
)
REVISION = "v1.1.0"
COMMIT = "b757d03ab58d9c30fe1e59b6af6d19431381d5ce"
FILENAME = "damoyolo_tinynasL25_S_cigarette.pt"
EXPECTED_SIZE = 130_945_475
EXPECTED_SHA256 = "daae5418929e166b92a9551c7d9686bd670cf8a7a6f0850d8d722cc3aa00079f"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / FILENAME,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        _verify(output)
        print(f"already_verified={output}")
        return 0

    with tempfile.TemporaryDirectory(prefix="behavior-model-") as directory:
        checkout = Path(directory) / "source"
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                REVISION,
                REPOSITORY,
                str(checkout),
            ]
        )
        actual_commit = _capture(["git", "-C", str(checkout), "rev-parse", "HEAD"])
        if actual_commit != COMMIT:
            raise RuntimeError(
                f"Source revision mismatch: expected {COMMIT}, got {actual_commit}"
            )
        _run(["git", "-C", str(checkout), "lfs", "pull", "--include", FILENAME])
        source = checkout / FILENAME
        _verify(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
    _verify(output)
    print(f"output={output}")
    print(f"revision={REVISION}")
    print(f"commit={COMMIT}")
    print(f"sha256={EXPECTED_SHA256}")
    print("license=Apache-2.0")
    return 0


def _verify(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(
            f"Checkpoint size mismatch: expected {EXPECTED_SIZE}, got {path.stat().st_size}"
        )
    actual = _sha256(path)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual}"
        )


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _capture(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
