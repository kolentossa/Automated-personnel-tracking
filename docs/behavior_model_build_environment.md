# Behavior Model RKNN Build Environment

## Target Board

| Item | Verified value |
| --- | --- |
| Board | RK3588 LubanCat, `aarch64` |
| OS | Debian GNU/Linux 11 (bullseye) |
| Kernel | `5.10.209-rk3588` |
| Runtime Python | 3.9.2 |
| `rknn-toolkit-lite2` | 2.3.0 |
| `librknnrt` | 2.3.0, build `c949ad889d`, 2024-11-07 |
| NPU driver | 0.9.8 |
| Camera | `/dev/video11`, 1280x720 NV12 through GStreamer |

The existing person/phone `yolov8n.rknn` was built with Toolkit 1.6.0 and is
known to run on Runtime 2.3.0. The new behavior model was built with Toolkit
2.3.0 to match the installed Runtime exactly. The Runtime and driver were not
upgraded.

## Isolated Conversion Environment

Rockchip publishes a Toolkit2 2.3.0 CPython 3.9 AArch64 wheel. Because the
board already provided a compatible native conversion host, conversion was
performed in a project-local virtual environment instead of changing system
Python or requiring a separate x86 machine:

```text
/home/cat/projects/person-tracking/.venv/codex-cache/rknn-build-py39-v2
```

Key packages:

```text
Python 3.9.2
rknn-toolkit2 2.3.0
torch 2.2.0
onnx 1.14.1
onnxruntime 1.16.0
numpy 1.24.4
OpenCV 4.5.1
```

The official Rockchip repository was pinned at tag 2.3.0, commit
`a8dd54d41e92c95b4f95780ed0534362b2c98b92`. No `sudo`, system package
installation, system service, or files outside the project were modified.

## PyTorch to Static ONNX

The verified DAMO checkpoint was exported with:

```text
opset: 12
input: images [1,3,640,640], static batch and shape
outputs: scores [1,8400,2], boxes [1,8400,4]
simplify: false
constant folding: true
color: RGB
resize: direct stretch to 640x640 (DAMO keep_ratio=false)
range: float32 0-255
layout: NCHW
mean/std: [0,0,0] / [1,1,1]
boxes: decoded xyxy in input coordinates
NMS: outside the model, class-aware
```

The source card describes one cigarette class, while the checkpoint head has
two score channels. Channel 0 was verified as cigarette. Channel 1 is retained
in the static graph, named `__unused__` in configuration, and never selected
by postprocessing. The application rejects a different output count, shape,
or class-map width instead of returning an empty result.

## Calibration

`scripts/prepare_behavior_calibration.py` selected and decoded exactly 300
images using seed 3588:

```text
220 CigDet v1 training positives
 73 COCO128 negatives
  7 curated negatives
```

The calibration list SHA-256 is
`fc3ffeeab155c7ec9fccc32e069c0e90f18a3920965f48cb50561d1fbfd0cfda`.
It includes small cigarettes, upper bodies, hand-to-mouth scenes, phones,
food/drink, steam/fog, lighter, and complex backgrounds. Images remain in the
ignored project cache.

## Conversion

The first FP build exposed a Toolkit2 2.3.0 optimizer defect in
`fuse_mul_into_matmul`: the DAMO DFL integral attempted to broadcast `(17,1)`
against `(33600,1)`. The converter's documented diagnostic recommended
disabling that one rule. `scripts/convert_behavior_onnx_to_rknn.py` therefore
sets `disable_rules=['fuse_mul_into_matmul']` only for `damoyolo`.

Reproducible commands:

```bash
python scripts/convert_behavior_onnx_to_rknn.py \
  --onnx damo_cigarette_640_static_opset12.onnx \
  --output damo_cigarette_640_fp.rknn \
  --model-family damoyolo --input-size 640 --target rk3588 \
  --no-quantize

python scripts/convert_behavior_onnx_to_rknn.py \
  --onnx damo_cigarette_640_static_opset12.onnx \
  --output behavior_damoyolo_cigarette_int8.rknn \
  --model-family damoyolo --input-size 640 --target rk3588 \
  --dataset damo_cigarette_300.txt
```

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| Original PT | 130,945,475 | `daae5418929e166b92a9551c7d9686bd670cf8a7a6f0850d8d722cc3aa00079f` |
| Static ONNX | 62,693,340 | `9b40116db87d1fc27ae637e3ab4c4f027d0af10be2c2f944f12e864e1f7d784e` |
| FP RKNN | 36,963,033 | `7de5d050797c714d6252d0ee6079d142760db808a3b2f413a2eba66d89e0ad08` |
| INT8 RKNN | 18,883,072 | `d04c43a3a695c9985fbd03db1e0a2956763374fd686d949b8cd96cabdc7c5941` |

## Consistency Results

| Comparison | Result |
| --- | --- |
| PyTorch to ONNX, 23 images | 29/29 boxes matched; mean IoU 0.9999993; max confidence delta 0.00000057 |
| ONNX AP50, 111 CigDet tests | 0.806662 |
| FP RKNN | AP50 0.808908; mean IoU to ONNX 0.994482; NPU mean 59.432 ms |
| INT8 RKNN | AP50 0.776251; 3.04-point drop; mean IoU 0.913890; mean confidence delta 0.013495; NPU mean 26.400 ms, P95 26.811 ms |
| Small target recall | ONNX 0.784314; INT8 0.764706 |

INT8 passed the five-point AP50 drop limit and retained substantially lower
latency, so it is the deployed artifact. Run the strict comparison again with:

```bash
python scripts/validate_behavior_model.py \
  --onnx damo_cigarette_640_static_opset12.onnx \
  --rknn fp=damo_cigarette_640_fp.rknn \
  --rknn int8=behavior_damoyolo_cigarette_int8.rknn \
  --images /path/to/CigDet_dataset/test \
  --output data/damo-consistency.json
```

The deployed binary lives at
`models/behavior_damoyolo_cigarette_int8.rknn`, owned by `cat:cat`, mode 0644.
It is intentionally ignored by Git.
