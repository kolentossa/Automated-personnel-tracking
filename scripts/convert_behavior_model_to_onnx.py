#!/usr/bin/env python3
"""Export the pinned DAMO cigarette checkpoint to a static two-output ONNX."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--damo-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for path in (args.checkpoint, args.config, args.damo_repo):
        if not path.exists():
            parser.error(f"Required path does not exist: {path}")
    if args.output.exists() and not args.force:
        parser.error(f"Output exists: {args.output}; pass --force to replace it")

    sys.path.insert(0, str(args.damo_repo.resolve()))
    import torch
    from torch import nn

    from damo.base_models.core.ops import RepConv, SiLU
    from damo.config.base import parse_config
    from damo.detectors.detector import build_local_model
    from damo.utils.model_utils import replace_module

    device = torch.device("cpu")
    config = parse_config(str(args.config.resolve()))
    structure = Path(config.model.backbone.structure_file)
    if not structure.is_absolute():
        config.model.backbone.structure_file = str(
            (args.config.resolve().parent / structure).resolve()
        )
    model = build_local_model(config, device)
    try:
        checkpoint = torch.load(
            str(args.checkpoint.resolve()), map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(str(args.checkpoint.resolve()), map_location=device)
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)
    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()
    model.head.nms = False

    dummy = torch.full(
        (1, 3, args.input_size, args.input_size), 114.0, dtype=torch.float32
    )
    with torch.no_grad():
        outputs = model(dummy)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise RuntimeError(
            f"Expected DAMO scores/boxes outputs, got {type(outputs).__name__}"
        )
    expected = ((1, 8400, 2), (1, 8400, 4))
    shapes = tuple(tuple(int(value) for value in output.shape) for output in outputs)
    if shapes != expected:
        raise RuntimeError(
            f"Unexpected DAMO output shapes: expected {expected}, got {shapes}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["images"],
        output_names=["scores", "boxes"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    _validate_onnx(args.output, args.input_size)
    print(f"output={args.output.resolve()}")
    print(f"sha256={_sha256(args.output)}")
    print("input=images:[1,3,640,640]:float32:RGB:0-255")
    print("outputs=scores:[1,8400,2],boxes:[1,8400,4]")
    return 0


def _validate_onnx(path: Path, input_size: int) -> None:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if [item.name for item in model.graph.input] != ["images"]:
        raise RuntimeError("ONNX must have exactly one input named images")
    if [item.name for item in model.graph.output] != ["scores", "boxes"]:
        raise RuntimeError("ONNX must expose scores and boxes outputs in that order")
    dimensions = [
        value.dim_value for value in model.graph.input[0].type.tensor_type.shape.dim
    ]
    if dimensions != [1, 3, input_size, input_size]:
        raise RuntimeError(f"ONNX input shape mismatch: {dimensions}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
