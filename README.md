# Automated Personnel Tracking System

A local-first personnel tracking demo for the RK3588 LubanCat-5 v2. The system reads prerecorded video, detects anonymous person-like objects, tracks them across frames, counts virtual-line crossings, and exposes a local FastAPI service plus a responsive dashboard.

The first demo does not require a USB camera. It generates a synthetic `data/sample.mp4` and runs the same camera, detection, tracking, counting, backend, and frontend flow that future camera sources will use.

## Current Production Deployment

The optimized camera service is deployed on the RK3588 at:

```text
http://192.168.1.213:8001
```

It reads `/dev/video11`, uses YOLOv8n RKNN on the NPU, tracks anonymous person
IDs, performs line-crossing counts, and applies RetinaFace plus optical-flow
face mosaic before Web streaming. The legacy service on port `8000` has been
retired and should remain stopped.

Current measured operating point:

```text
camera: /dev/video11
fps: approximately 30
rolling application latency: approximately 12-16 ms
detector: rknn-yolov8n
npu_enabled: true
face_detector: retinaface-mobile320-onnx
```

The service currently runs as a project-local background process. No systemd
unit or boot-time service has been installed, so it must be started again after
the board reboots.

## Architecture

```text
Video Source
    |
    v
Camera Interface Layer
    |
    v
Person Detection
    |
    v
Multi Object Tracking
    |
    v
Entry/Exit Logic
    |
    v
Backend API
    |
    v
Local Web Dashboard
```

Main modules:

- `vision/camera/`: `VideoFileCamera`, `WebCamera`, and `RTSPCamera` behind one `Camera` interface.
- `vision/detection/`: detector interface, OpenCV DNN YOLO support, and a no-model motion detector for the generated demo.
- `vision/tracking/`: lightweight ByteTrack-style tracker with stable temporary IDs.
- `vision/counting/`: virtual-line entry and exit counting.
- `backend/`: FastAPI routes, event store, and pipeline lifecycle.
- `frontend/`: static local dashboard served by the backend at `/dashboard/`.

## Installation

Run everything as the normal `cat` user from inside the project directory. Do not use `sudo`.

```bash
cd ~/projects/person-tracking
./scripts/setup.sh
```

The setup script creates `.venv/` and installs Python packages inside that environment. It uses `--system-site-packages` so the existing board-provided OpenCV package can be used without installing or modifying system Python packages. Verify the interpreter path:

```bash
source .venv/bin/activate
which python
```

Expected path:

```text
/home/cat/projects/person-tracking/.venv/bin/python
```

Download the Rockchip model-zoo RetinaFace model into the ignored `models/`
directory. The downloader verifies the pinned SHA-256 before installing it:

```bash
.venv/bin/python scripts/download_face_model.py
```

## Legacy Synthetic Demo

```bash
cd ~/projects/person-tracking
./scripts/run_demo.sh
```

This older demo generates `data/sample.mp4` and starts the legacy backend on
`0.0.0.0:8000`. It is retained for offline development and is not the current
RK3588 camera deployment.

Open locally or from the LAN:

```text
API:       http://127.0.0.1:8000/
Dashboard: http://127.0.0.1:8000/dashboard/
```

From another device on the same LAN, replace `127.0.0.1` with the RK3588 board IP address.

## API

```text
GET /            service status
GET /status      current people, camera status, fps, frame count
GET /events      recent anonymous ENTER/EXIT events
GET /statistics  entered, exited, and current people totals
```

Example root response:

```json
{
  "status": "running",
  "device": "RK3588",
  "service": "person tracking"
}
```

## Camera Replacement

Downstream detection, tracking, counting, backend, and frontend code depend only on the `Camera` interface.

Current demo:

```python
from vision.camera import VideoFileCamera
camera = VideoFileCamera("data/sample.mp4", loop=True)
```

Future USB camera:

```python
from vision.camera import WebCamera
camera = WebCamera(device_index=0)
```

Future RTSP stream:

```python
from vision.camera import RTSPCamera
camera = RTSPCamera("rtsp://user:pass@camera/stream")
```

Only the camera construction changes.

## YOLO Model Use

The demo runs without a model by using `MotionPersonDetector`. For model-backed person detection, export a lightweight YOLO model to ONNX on a workstation and copy it to:

```text
models/yolov8n.onnx
```

Then start the demo with:

```bash
PERSON_TRACKING_MODEL=models/yolov8n.onnx ./scripts/run_demo.sh data/sample.mp4
```

Training and fine tuning should stay on a workstation. The RK3588 should only run inference.

## Privacy Design

This project is designed for local, anonymous counting.

- All video processing runs locally on the RK3588.
- No images, video frames, events, or metadata are uploaded to cloud services.
- Raw camera video is not recorded by the backend.
- The generated sample video is synthetic and contains no real people.
- The system does not perform face recognition or biometric identification.
- Tracking IDs are temporary anonymous IDs, not real identities.
- Stored events contain only timestamp, event type, and temporary tracking ID.
- Frames are processed transiently in memory and discarded after each pipeline step.

Anonymous event example:

```json
{
  "timestamp": "2026-07-07T10:30:00+00:00",
  "event_type": "ENTER",
  "tracking_id": 7
}
```

## Future RK3588 NPU Optimization

Recommended path:

1. Train or fine tune on a workstation.
2. Export a lightweight YOLO nano or small model to ONNX.
3. Validate accuracy and latency on representative videos.
4. Convert the validated model to RKNN on the supported toolchain.
5. Store optimized model files under `models/`.
6. Add an RKNN detector implementation behind the existing `Detector` interface.

No backend, frontend, tracker, counter, or camera code should need to change for NPU inference.

## Tests

```bash
cd ~/projects/person-tracking
./scripts/run_tests.sh
```

The tests cover virtual-line counting and stable track IDs.
## RK3588 Camera Web Tracking MVP

This repository now includes a deployment-oriented local Web app under `app/`.
It is intended to run on the LubanCat RK3588 as the normal `cat` user and does
not require training a new model.

Run it from the project root:

```bash
cd ~/projects/person-tracking
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Or use the helper script:

```bash
cd ~/projects/person-tracking
./scripts/run_dev.sh
```

For the current project-local background deployment:

```bash
mkdir -p logs
nohup ./scripts/run_dev.sh > logs/uvicorn-8001-production.log 2>&1 &
```

Open the dashboard from another LAN device by replacing the address with the
board IP:

```text
http://RK3588_IP:8001
```

Implemented routes:

```text
GET  /                  Web dashboard
GET  /video             MJPEG stream
GET  /api/stats         JSON statistics
GET  /api/health        Service and camera health
GET  /api/config/counting
                          Current counting-line and enter-direction config
POST /api/config/counting
                          Update counting-line and enter-direction config
POST /api/reset-stats   Reset occupancy and crossing counters
```

The Web sidebar shows `current_occupancy`, `total_entered`, `total_exited`,
`active_tracks`, `fps`, `latency_ms`, and `camera_status`. The video stream only
draws detection boxes, temporary track IDs, confidence, and the crossing line.
Count totals are kept outside the video image.

Configuration lives in `config.yaml`. Important fields:

```yaml
camera:
  source_type: camera
  camera_device: /dev/video-camera0
  auto_detect: true

detection:
  model_path: ""
  confidence_threshold: 0.35
  input_size: 640

privacy:
  face_mosaic_enabled: true
  face_detector: retinaface-onnx
  face_model_path: models/RetinaFace_mobile320.onnx
  face_input_size: 320
  face_confidence_threshold: 0.6
  face_detector_threads: 1
  face_detect_every_n_frames: 5
  face_result_max_age_ms: 1000
  face_mosaic_padding: 0.25
  face_tracking_enabled: true
  face_tracking_min_points: 4
  face_tracking_win_size: 31
  face_tracking_max_level: 3
  face_tracking_max_motion_px: 96
  head_fallback_enabled: true
```

The optimized RK3588 profile uses `performance.detect_every_n_frames: 3`,
pins the process to big CPU cores `4-7`, reports a 30-frame rolling latency,
and streams 800x450 JPEG frames at quality 74. Person inference runs at about
10 Hz while tracking, privacy mosaic, video encoding, and Web streaming remain
at the camera frame rate.

`camera.source_type` can later be changed to `video`, using `camera.video_file`,
to reuse the same detection/tracking/counting pipeline with a prerecorded file.
The default camera implementation uses OpenCV `VideoCapture` with the V4L2
backend and automatically probes `/dev/video*`. If a sensor is present but
OpenCV cannot read that board-specific RKISP node, `/api/health` and the Web
page report the error instead of crashing.

Detection uses the configured local detector path. The deployment path is
`detector.type: rknn-yolo` with a YOLO RKNN model under `models/`. The legacy
`detection` section remains as an optional CPU fallback configuration for
MobileNetSSD or ONNX models. Motion fallback is disabled by default so moving
non-person objects are not counted as people.

Privacy behavior:

- Frames are processed locally on the RK3588.
- Raw unmasked camera frames are not saved by the Web app.
- Mosaic anonymisation is applied before JPEG frames are sent to the browser.
- Face recognition, identity recognition, ReID, embeddings, and face databases
  are not used.
- `RetinaFace_mobile320.onnx` performs real face detection asynchronously so
  face inference cannot queue video frames or add directly to stream latency.
- Sparse Lucas-Kanade optical flow propagates each detected face box on every
  video frame. RetinaFace refreshes the boxes every five frames to correct
  drift, and invalid optical-flow tracks immediately fall back to head mosaic.
- `/api/health` and `/api/stats` expose `face_detector`,
  `face_detector_available`, `faces_detected`, `face_detection_ms`,
  `face_tracking_ms`, `face_tracked_boxes`, and `face_privacy_mode` for runtime
  verification.
- A person whose face is not detected still receives a conservative head-box
  mosaic when `head_fallback_enabled` is true.

Verify the detector against a known face image with:

```bash
.venv/bin/python scripts/test_face_privacy.py --image data/retinaface_test.jpg
```

The test fails unless RetinaFace returns a real face box, all changed pixels
stay inside that box, and optical flow follows a 96-pixel synthetic movement
without losing the mosaic. The test image is intentionally ignored by Git.

### RK3588 IMX415 camera capture note

On this LubanCat-5 V2 board the IMX415 sensor is enabled and `v4l2-ctl` can
stream from `/dev/video11`, but OpenCV direct V4L2 capture does not open the
RKISP multiplanar node. The working path is OpenCV with the GStreamer backend:

```text
v4l2src device=/dev/video11 ! video/x-raw,format=NV12,width=1280,height=720 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false
```

`config.yaml` therefore defaults to:

```yaml
camera:
  camera_device: /dev/video11
  capture_backend: gstreamer
  width: 1280
  height: 720
```

This keeps the Python processing pipeline unchanged while avoiding the OpenCV
V4L2 issue. If the camera connector or overlay changes, update
`camera.camera_device` and test with `/api/health`.

### Web counting configuration

The deployed dashboard can adjust the counting line and the ENTER direction
without editing code or logging in with SSH.

Open the Web UI:

```text
http://RK3588_IP:8001
```

Use the `Counting Configuration` panel:

1. Click the video once to set the line start point.
2. Click the video again to set the line end point.
3. Choose `Left → Right` or `Right → Left` as the ENTER direction.
4. Click `Save Configuration`.

Direction changes are applied immediately by the Web UI. The saved values are
persisted in `config.yaml`, so the same counting line and direction are kept
after the FastAPI service restarts.

The persisted config shape is:

```yaml
counting:
  line:
    x1: 640
    y1: 0
    x2: 640
    y2: 720
  direction:
    mode: left_to_right
  cooldown_frames: 20
```

API examples:

```bash
curl http://127.0.0.1:8001/api/config/counting

curl -X POST http://127.0.0.1:8001/api/config/counting \
  -H 'Content-Type: application/json' \
  -d '{"line":{"x1":640,"y1":0,"x2":640,"y2":720},"direction":"left_to_right"}'
```

### Person detector selection

The motion detector is only a fallback for synthetic demos. It detects moving
foreground blobs and will mark non-person moving objects, so it should not be
used for the live camera deployment. The RK3588 optimized path defaults to
RKNN YOLO on the NPU and keeps only COCO class `person`:

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

The CPU fallback remains available when `detector.fallback_to_cpu` is set to
`true`. It uses OpenCV DNN MobileNetSSD Caffe and keeps only VOC class
`person`:

```yaml
detection:
  model_path: models/MobileNetSSD_deploy.caffemodel
  model_config_path: models/MobileNetSSD_deploy.prototxt
  confidence_threshold: 0.45
  input_size: 300
```

The `.caffemodel` and `.prototxt` files stay under `models/` on the board and
are ignored by Git. This avoids committing large model artifacts while keeping
the runtime fully local.

### RKNN YOLO NPU deployment

Convert the model on an x86 Linux workstation, then copy only the final `.rknn`
file to the RK3588 board. Do not commit model files to Git.

Example workstation flow:

```bash
python3 -m venv .venv-rknn-convert
. .venv-rknn-convert/bin/activate
python -m pip install ultralytics onnx rknn-toolkit2
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12 simplify=True
```

Use the Rockchip RKNN Toolkit 2 conversion script for the board target
`rk3588`, using representative calibration images if the model is quantized.
The converted file should be named:

```text
models/yolov8n.rknn
```

Copy the converted model to the board:

```bash
scp models/yolov8n.rknn cat@RK3588_IP:/home/cat/projects/person-tracking/models/
```

Run the Web service on the RK3588:

```bash
cd ~/projects/person-tracking
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Then open:

```text
http://RK3588_IP:8001
```

Check NPU status and latency:

```bash
curl http://127.0.0.1:8001/api/health
python scripts/benchmark_pipeline.py --frames 300
python scripts/benchmark_pipeline.py --video data/privacy_motion_test.mp4 --frames 300
```

Expected optimized health fields after `models/yolov8n.rknn` is present:

```json
{
  "source": "/dev/video11",
  "detector": "rknn-yolov8n",
  "npu_enabled": true,
  "face_detector_available": true,
  "fps": 30.0,
  "inference_ms": 8.0,
  "total_latency_ms": 14.0
}
```
