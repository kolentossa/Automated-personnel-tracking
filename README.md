# Automated Personnel Tracking System

A local-first personnel tracking demo for the RK3588 LubanCat-5 v2. The system reads prerecorded video, detects anonymous person-like objects, tracks them across frames, counts virtual-line crossings, and exposes a local FastAPI service plus a responsive dashboard.

The first demo does not require a USB camera. It generates a synthetic `data/sample.mp4` and runs the same camera, detection, tracking, counting, backend, and frontend flow that future camera sources will use.

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

## Running The Demo

```bash
cd ~/projects/person-tracking
./scripts/run_demo.sh
```

The script generates `data/sample.mp4`, starts the backend on `0.0.0.0:8000`, and launches the processing pipeline.

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
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or use the helper script:

```bash
cd ~/projects/person-tracking
./scripts/run_dev.sh
```

Open the dashboard from another LAN device by replacing the address with the
board IP:

```text
http://RK3588_IP:8000
```

Implemented routes:

```text
GET  /                  Web dashboard
GET  /video             MJPEG stream
GET  /api/stats         JSON statistics
GET  /api/health        Service and camera health
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
```

`camera.source_type` can later be changed to `video`, using `camera.video_file`,
to reuse the same detection/tracking/counting pipeline with a prerecorded file.
The default camera implementation uses OpenCV `VideoCapture` with the V4L2
backend and automatically probes `/dev/video*`. If a sensor is present but
OpenCV cannot read that board-specific RKISP node, `/api/health` and the Web
page report the error instead of crashing.

Detection uses the existing local demo detector path. If `detection.model_path`
points to a local ONNX model, OpenCV DNN YOLO is used and only the person class
is kept. If no model is configured or the file is missing, the app falls back to
the existing no-model motion person detector. RKNN/NPU inference is intentionally
left behind the detector interface for the next deployment step.

Privacy behavior:

- Frames are processed locally on the RK3588.
- Raw unmasked camera frames are not saved by the Web app.
- Mosaic anonymisation is applied before JPEG frames are sent to the browser.
- Face recognition, identity recognition, ReID, embeddings, and face databases
  are not used.
- If a Haar cascade is unavailable or no face is found, the top portion of each
  person box is mosaiced as a head fallback.
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
