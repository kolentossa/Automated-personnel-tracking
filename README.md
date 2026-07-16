# Automated Personnel Tracking System

A local-first personnel and behavior tracking service for the RK3588
LubanCat-5 v2. It reads a camera or prerecorded video, detects people, phones,
and cigarettes on the NPU, tracks anonymous IDs, counts line crossings, and
serves a FastAPI dashboard on the local network.

## Current Production Deployment

The optimized camera service is deployed on the RK3588 at:

```text
http://192.168.1.213:8001
```

It reads `/dev/video11`, runs YOLOv8n person/phone detection, a YOLO11n phone
context model, and an INT8 DAMO-YOLO cigarette detector on the RK3588 NPU,
tracks anonymous person IDs, and performs line-crossing plus temporal behavior
analysis. This branch deliberately disables face detection and all mosaic:
the Web stream and event evidence are unredacted. Use it only on a controlled
LAN and never expose port 8001 to the public Internet.
The legacy service on port `8000` has been retired and should remain stopped.

Current measured operating point:

```text
camera: /dev/video11
steady Web FPS: approximately 27-29
rolling application latency: approximately 25-38 ms
detector: rknn-yolov8n
npu_enabled: true
phone_context_detector: rknn-yolo11n
behavior_detector: rknn-damoyolo-cigarette-int8
smoking_detection_available: true
face_detection_enabled: false
mosaic_enabled: false
privacy_mode: no_mosaic
```

The final behavior-enabled 300-frame sequential benchmark measured 26.21 FPS,
37.64 ms mean latency, 28.85 ms P50, and 97.98 ms P95. See
`docs/rk3588_behavior_benchmark.md` for stage timings and measurement
definitions.

The final 30-minute supervised Web stability run measured 28.118 FPS average,
39.9 ms P95 application latency, zero API/camera/process failures, and stable
memory. RSS changed by -20.528 MB; the fitted 6.886 MB/hour slope stayed well
below the 64 MB/hour limit. Low-frequency GC and glibc trim maintenance keep
native buffers bounded while the three RKNN model contexts share the NPU.

The service runs under a project-local restart supervisor. No systemd unit or
boot-time service is installed, so run
`./scripts/manage_behavior_service.sh start` after a board reboot.

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

Install the three SHA-verified RKNN models in the ignored `models/` directory
before starting the production service:

```bash
sha256sum models/yolov8n.rknn \
  models/yolo11n.rknn \
  models/behavior_damoyolo_cigarette_int8.rknn
```

The expected hashes and reproducible behavior/YOLO11 conversion details are in
`models/behavior_model_manifest.json`. No face model is required.

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

## Security And No-Mosaic Mode

This branch is local-only but is not visually anonymized.

- All inference runs on the RK3588; no cloud API is used.
- Face recognition, biometric identification, ReID, embeddings, and identity
  databases are not implemented.
- Tracking IDs are temporary anonymous IDs, not real identities.
- Face detection is disabled and no RetinaFace model is loaded.
- Mosaic, blur, pixelation, and redaction are not called in the runtime path.
- MJPEG frames, event snapshots, optional clips, and debug captures are
  unredacted. Event evidence can therefore contain identifiable people.
- Do not forward port 8001, place the service behind a public reverse proxy, or
  commit anything under `data/behavior_events/`.

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

For the current project-local supervised deployment:

```bash
./scripts/manage_behavior_service.sh start
./scripts/manage_behavior_service.sh status
```

Use `restart` after a configuration or code update and `stop` for a graceful
shutdown. The supervisor restarts a failed Uvicorn child after three seconds;
its PID is stored under `run/` and logs stay under `logs/`.

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
GET  /api/events        Recent phone/smoking behavior events
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
  face_detection_enabled: false
  mosaic_enabled: false
  unredacted_video_enabled: true
  unredacted_evidence_enabled: true
```

The optimized RK3588 profile uses `performance.detect_every_n_frames: 3`,
pins the process to big CPU cores `4-7`, reports a 30-frame rolling latency,
and streams 800x450 JPEG frames at quality 74. Person inference runs at about
10 Hz while tracking, drawing, video encoding, and Web streaming remain at the
camera frame rate.

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

No-mosaic runtime behavior:

- `face_detection_enabled`, `face_model_loaded`, and `mosaic_enabled` remain
  `false` in `/api/health` and `/api/stats`.
- `privacy_mode` is `no_mosaic`; `behavior_input_frame` is `raw`.
- The service does not create a face worker, face queue, optical-flow tracker,
  or mosaic operation. A missing RetinaFace file cannot degrade startup.
- Phone and smoking geometry uses configurable regions estimated from the
  person box; these regions are not face detections.
- Web and event outputs are unredacted. Keep the service and ignored evidence
  directory inside a controlled LAN environment.

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

### Phone and smoking behavior detection

The branch `feature/rk3588-no-mosaic-accuracy` keeps the original multi-class
RKNN behavior pipeline and independently improves phone recall and cigarette
false-positive rejection. Removing mosaic is a deployment choice; it is not
the reason accuracy changed.

The existing COCO YOLOv8n model supplies `person` and `cell phone` from one NPU
inference. A separate Apache-2.0 ModelScope DAMO-YOLO checkpoint supplies real
`cigarette` output from
`models/behavior_damoyolo_cigarette_int8.rknn`. It is SHA-verified, executes
every fifth frame, and is kept out of Git with the other model binaries. The
tracked manifest records its source, fixed revision, license, class order,
preprocessing contract, conversion toolchain, and hashes.

Small-phone recall is tuned independently from person detection. Full-frame
person and phone confidence remains `0.35`; low thresholds are used only inside
person-associated crops. Phone behavior still requires person association,
estimated head/body geometry, temporal persistence, and cooldown. Short
detector gaps are bridged by the phone cache and state machine.

For back-facing, side-facing, partially occluded, or small phones, one ROI pass
runs on the largest eligible person for each primary inference cycle. The
scheduled crop modes are YOLOv8n head/shoulders at `0.12`, YOLOv8n hands/torso
at `0.20`, and YOLO11n upper-body context at `0.35`. Results are deduplicated,
cached for eight primary frames with confidence decay, and mapped back to the
original frame. The final 235-image test measured phone precision 0.6935,
recall 0.6418, and F1 0.6667; phone-call hits improved from 19/42 with the
YOLOv8 RKNN crop baseline to 22/42 with the hybrid scheduler.

Cigarette postprocessing uses class-aware `0.35` IoU NMS plus containment
suppression to collapse nested boxes around one object. Raw candidates then
pass person-size/location checks, phone IoA conflict at `0.48`, explicit tool
conflict, and multi-frame continuity before they become verified cigarettes.
On the locked 235-image final set, confirmed smoking false positives fell from
4 to 0 while true-cigarette event recall stayed 0.1892 (21/111), a 0-point
change. Phone-call, screwdriver, tool, pen, and elongated-object groups each
produced zero confirmed smoking events.

Production smoking alerts require the cigarette to be assigned to a person and
near the configurable head/mouth proxy derived from that person's box for the
configured duration. This proxy preserves hand-to-mouth evidence without any
face detector. Smoke, flame, and lighter labels are auxiliary only and cannot
independently classify a person as smoking. A missing or invalid required
behavior model produces an explicit health error; it is never replaced with
fabricated or motion-based detections.

Model selection, conversion, and validation details:

- `docs/behavior_model_selection.md`
- `docs/cigarette_false_positive_analysis.md`
- `docs/no_mosaic_accuracy_deployment.md`
- `docs/behavior_model_build_environment.md`
- `models/behavior_model_manifest.json`
- `artifacts/behavior_model_validation.json`

Configuration, model conversion, event JSON, decision limits, tests, and
performance instructions are documented in:

- `docs/rk3588_behavior_detection.md`
- `docs/rk3588_behavior_acceptance_checklist.md`
- `docs/rk3588_behavior_benchmark.md`

Run all behavior and regression tests:

```bash
python -m pytest -q
```

Run the 300-frame RK3588 benchmark:

```bash
python scripts/benchmark_behavior_pipeline.py --frames 300
```

### Person detector selection

The motion detector is only a fallback for synthetic demos. It detects moving
foreground blobs and will mark non-person moving objects, so it should not be
used for the live camera deployment. The RK3588 optimized path defaults to
RKNN YOLO on the NPU and returns COCO `person` and `cell phone` in one pass.
The tracker still receives only `person`; the behavior layer receives both:

```yaml
detector:
  type: rknn-yolo
  model_path: models/yolov8n.rknn
  model_family: yolov8
  input_size: 640
  confidence_threshold: 0.35
  class_confidence_thresholds: {"person": 0.35, "cell phone": 0.35}
  nms_threshold: 0.45
  class_filter: ["person", "cell phone"]
  phone_roi_refinement:
    enabled: true
    confidence_threshold: 0.12
    detect_every_n_primary_frames: 1
    max_people: 1
    min_person_height_px: 160
    crop_modes:
      - {name: head_shoulders, detector: primary, confidence_threshold: 0.12}
      - {name: hands_torso, detector: primary, confidence_threshold: 0.20}
      - {name: upper_body_context, detector: context, confidence_threshold: 0.35}
    cache_primary_frames: 8
    cache_confidence_decay: 0.97
    cache_min_confidence: 0.08
    max_phone_area_ratio: 0.25
    nms_threshold: 0.35
    containment_threshold: 0.70
    context_model:
      enabled: true
      model_path: models/yolo11n.rknn
      model_family: yolo11
      class_filter: ["cell phone"]
  core_mask: "0_1_2"
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
