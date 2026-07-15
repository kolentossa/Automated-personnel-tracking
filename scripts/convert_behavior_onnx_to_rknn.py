#!/usr/bin/env python3
"""Convert a supplied YOLO behavior ONNX model to RKNN on x86 Linux."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", default=Path("models/behavior_yolov8n.rknn"), type=Path)
    parser.add_argument("--dataset", type=Path, help="Calibration image list required for INT8")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.onnx.is_file():
        parser.error(f"ONNX model does not exist: {args.onnx}")
    if not args.no_quantize and (args.dataset is None or not args.dataset.is_file()):
        parser.error("INT8 conversion requires --dataset; use --no-quantize only for validation")
    if args.output.exists() and not args.force:
        parser.error(f"Output exists: {args.output}; pass --force to replace it")

    try:
        from rknn.api import RKNN
    except Exception as exc:
        raise SystemExit("Install Rockchip rknn-toolkit2 in an x86 Linux virtual environment") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        _require_zero(rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=args.target,
        ), "config")
        _require_zero(rknn.load_onnx(model=str(args.onnx)), "load_onnx")
        _require_zero(rknn.build(
            do_quantization=not args.no_quantize,
            dataset=str(args.dataset) if args.dataset else None,
        ), "build")
        _require_zero(rknn.export_rknn(str(args.output)), "export_rknn")
    finally:
        rknn.release()
    print(f"output={args.output}")
    print(f"sha256={_sha256(args.output)}")
    return 0


def _require_zero(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"RKNN {operation} failed with code {result}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
