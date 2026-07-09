# RKNN YOLOv8n NPU Deployment Run Log

Date: 2026-07-09

## Phase 0: Current Service Baseline

Project directory:

```text
/home/cat/projects/person-tracking
```

Git status before this run:

```text
## optimize-rknn-yolo-npu
```

Current branch:

```text
optimize-rknn-yolo-npu
```

Recent commits:

```text
1d6ea9c optimize: add rknn yolo npu pipeline
f89190e fix: use semantic person detector for live camera
1fa7024 fix: use gstreamer pipeline for rk3588 camera
61a023f deploy: add rk3588 camera web tracking MVP
2a7aff0 MVP local person tracking demo
```

Port 8000 listener:

```text
LISTEN 0 0 0.0.0.0:8000 0.0.0.0:* users:(("python",pid=24744,fd=6))
```

Port 8000 health summary:

```json
{
  "status": "ok",
  "running": true,
  "camera_status": "online",
  "source": "/dev/video11",
  "fps": 8.59,
  "latency_ms": 119.6,
  "error": "",
  "available_camera": {
    "device": "/dev/video11",
    "backend": "gstreamer",
    "readable": true,
    "width": 1280,
    "height": 720,
    "fps": 120.0
  }
}
```

Port 8000 stats summary:

```json
{
  "camera_status": "online",
  "source": "/dev/video11",
  "model": "models/MobileNetSSD_deploy.caffemodel",
  "detector": "opencv-dnn-mobilenetssd-caffe",
  "fps": 8.59,
  "latency_ms": 119.6,
  "privacy_mode": true,
  "face_mosaic_enabled": true
}
```

Conclusion: port 8000 is preserved and still serves the current stable MobileNetSSD camera pipeline. It was not stopped, restarted, or replaced.

## Phase 1: Ignore Rules

Required model ignore rules were checked. The project already ignored most large model formats. This run adds the missing:

```text
models/*.pth
models/*.param
```

Model files must remain untracked.

## Phase 2: RKNN Detector And Config

Existing files checked:

```text
app/detectors/rknn_yolo.py
scripts/benchmark_pipeline.py
config.yaml
```

The active RKNN config points to:

```yaml
detector:
  type: rknn-yolo
  model_path: models/yolov8n.rknn
  model_family: yolov8
  input_size: 640
  confidence_threshold: 0.35
  nms_threshold: 0.45
  class_filter: ["person"]
  fallback_to_cpu: false
```

The camera config remains compatible with the working LubanCat camera path:

```yaml
camera:
  camera_device: /dev/video11
  capture_backend: gstreamer
```

## Phase 3: RKNN Model Search

Target model path:

```text
/home/cat/projects/person-tracking/models/yolov8n.rknn
```

Current project `models/` directory:

```text
MobileNetSSD_deploy.caffemodel
MobileNetSSD_deploy.prototxt
person_detector_v0_pretrained.onnx
README.md
yolov5n_fp32.onnx
yolov5n.onnx
```

Result:

```text
models/yolov8n.rknn is missing.
```

The wider `/home/cat` search found many unrelated `.rknn` files, including YOLOv5s demos and a YOLOv8 segmentation model, but no trusted YOLOv8n person-detection RKNN suitable to copy as `models/yolov8n.rknn`.

Decision: do not start port 8001, do not benchmark fake no-op results, and do not convert RKNN on the RK3588 ARM64 board. Generate the Windows + WSL2 export guide instead.

## Phase 4: Check Script Status

Added:

```text
scripts/check_rknn_model.py
```

Actual current result until the model is copied:

```text
error: RKNN model does not exist: /home/cat/projects/person-tracking/models/yolov8n.rknn
check_exit_code=2
```

Static checks:

```text
python -m py_compile app/*.py app/detectors/*.py scripts/benchmark_pipeline.py scripts/check_rknn_model.py
rknn_detector_import=ok
detector_type=rknn-yolo
detector_model_path=models/yolov8n.rknn
fallback_to_cpu=False
target_fps=30
```

Benchmark guard check:

```text
detector: no-op-person-detector
npu_enabled: false
warning: Configured RKNN model not found: /home/cat/projects/person-tracking/models/yolov8n.rknn Motion fallback is disabled to avoid marking non-person moving objects.
error: refusing to benchmark placeholder detector
benchmark_exit_code=2
```

No 8001 Web service was started because the RKNN model is missing.

## Blocker

The deployment is blocked by the missing model file:

```text
models/yolov8n.rknn
```

After copying that file to the RK3588, continue with:

```bash
cd /home/cat/projects/person-tracking
source .venv/bin/activate
python scripts/check_rknn_model.py
```
