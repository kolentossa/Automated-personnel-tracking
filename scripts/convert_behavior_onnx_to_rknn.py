#!/usr/bin/env python3
"""Convert a static behavior ONNX model to RKNN Toolkit2 format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", default=Path("models/behavior_yolov8n.rknn"), type=Path)
    parser.add_argument("--dataset", type=Path, help="Calibration image list required for INT8")
    parser.add_argument("--model-family", choices=("damoyolo", "yolov8"), default="yolov8")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.onnx.is_file():
        parser.error(f"ONNX model does not exist: {args.onnx}")
    if args.input_size < 32:
        parser.error("--input-size must be at least 32")
    if not args.no_quantize and (args.dataset is None or not args.dataset.is_file()):
        parser.error("INT8 conversion requires --dataset; use --no-quantize only for validation")
    if args.output.exists() and not args.force:
        parser.error(f"Output exists: {args.output}; pass --force to replace it")

    try:
        from rknn.api import RKNN
    except Exception as exc:
        raise SystemExit("Install a Rockchip rknn-toolkit2 wheel compatible with this Python environment") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preprocess = _preprocess_for(args.model_family)
    config_options = {
        "mean_values": [preprocess["mean"]],
        "std_values": [preprocess["std"]],
        "target_platform": args.target,
        "optimization_level": 3,
        "quantized_dtype": "asymmetric_quantized-8",
    }
    if args.model_family == "damoyolo":
        # Toolkit2 2.3.0 otherwise tries to broadcast the DFL integral
        # constant (17, 1) over every prediction while fusing MatMul + Mul.
        config_options["disable_rules"] = ["fuse_mul_into_matmul"]
    rknn = RKNN(verbose=True)
    try:
        _require_zero(rknn.config(**config_options), "config")
        load_options = {}
        if args.model_family == "damoyolo":
            load_options = {
                "inputs": ["images"],
                "input_size_list": [[1, 3, args.input_size, args.input_size]],
                "outputs": ["scores", "boxes"],
            }
        _require_zero(rknn.load_onnx(model=str(args.onnx), **load_options), "load_onnx")
        _require_zero(rknn.build(
            do_quantization=not args.no_quantize,
            dataset=str(args.dataset) if args.dataset else None,
        ), "build")
        _require_zero(rknn.export_rknn(str(args.output)), "export_rknn")
    finally:
        rknn.release()
    print(f"output={args.output}")
    print(f"sha256={_sha256(args.output)}")
    print(json.dumps({
        "model_family": args.model_family,
        "target_platform": args.target,
        "input_shape": [1, 3, args.input_size, args.input_size],
        "quantization": "none" if args.no_quantize else "asymmetric_quantized-8",
        "preprocess": preprocess,
    }, sort_keys=True))
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


def _preprocess_for(model_family: str) -> dict:
    if model_family == "damoyolo":
        return {
            "color": "RGB",
            "layout": "NHWC runtime input; Toolkit converts to NCHW model input",
            "resize": "direct stretch",
            "range": "0-255",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        }
    return {
        "color": "RGB",
        "layout": "NHWC runtime input; Toolkit converts to NCHW model input",
        "resize": "letterbox with value 114",
        "range": "0-1",
        "mean": [0.0, 0.0, 0.0],
        "std": [255.0, 255.0, 255.0],
    }


if __name__ == "__main__":
    raise SystemExit(main())
