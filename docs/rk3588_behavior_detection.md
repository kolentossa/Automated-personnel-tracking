# RK3588 Phone and Smoking Behavior Detection

## Scope

This feature adds temporal behavior analysis to the existing local RK3588
person-tracking service. It supports these unified event types:

- `phone_call`: a phone remains close to a tracked person's face/head region.
- `phone_playing`: a phone remains inside the upper body and below the face.
- `unauthorized_photography`: a raised phone is geometrically aligned with a
  configured prohibited ROI.
- `smoking`: cigarette, hand-to-mouth, ignition, and smoke evidence is fused
  and associated with one tracked person.

No event is emitted from a single detection. Every event uses a configurable
duration, minimum evidence-frame count, confidence threshold, tolerated gap,
and cooldown. A continuous behavior emits one primary alert until its state
ends. State is keyed by the anonymous `track_id` and cleaned when that track
disappears.

## Current Model Status

The board currently has:

| Model | Purpose | Status |
| --- | --- | --- |
| `models/yolov8n.rknn` | COCO person and cell-phone detection | Present and NPU tested |
| `models/RetinaFace_mobile320.onnx` | Face detection for privacy and geometry | Present and CPU tested |
| `models/behavior_yolov8n.rknn` | Cigarette, smoke, flame, lighter, and hand detection | Not present |

The existing YOLOv8n model SHA-256 is:

```text
ff3a64e6fe180203128c8d42456b458d208d3a1e2217d63683af00d6194e82ea
```

Phone behavior can use the existing COCO model because the NPU postprocessor
now returns both `person` and `cell phone` from one inference. Smoking
inference remains unavailable until a properly licensed custom model is
converted and installed. The association, evidence fusion, state machine,
event output, tests, and RKNN interface do not fabricate missing detections.

## Data Flow

```text
V4L2/GStreamer, RTSP, or video file
  -> latest-frame camera thread
  -> YOLOv8n RKNN (person + cell phone)
  -> optional custom behavior RKNN
  -> person IoU tracking
  -> target-to-track association
  -> RetinaFace and optical-flow face boxes
  -> phone and smoking temporal state machines
  -> face/head mosaic
  -> async privacy-safe evidence writer and callbacks
  -> MJPEG stream and JSON API
```

The auxiliary model supports an independent frame interval. Its last result is
reused between inference frames, while the main camera thread still retains
only the newest frame. Disk and callback work runs on `behavior-event-writer`,
outside the frame-processing thread.

## Decision Logic

### Phone call

The phone must be assigned to one person and remain near a detected face. If a
face result is temporarily unavailable, the top portion of the person box is
used as a lower-confidence head estimate. Short phone appearances are removed
by the temporal state machine.

### Phone playing

The phone must be assigned to the person, lie inside the upper body, remain
below the face, and not satisfy the phone-call relation. The current model does
not estimate eye gaze or finger motion. `phone_below_face_attention_proxy` is a
geometric proxy, not a claim that gaze was directly recognized.

### Unauthorized photography

The phone must be raised and the vector from the person's body center to the
phone must align with the vector toward an enabled prohibited ROI. A person
whose center is already inside that ROI is ignored by this rule. The current
model cannot determine the physical phone-camera lens direction. The event
therefore means "sustained raised-phone geometry toward an ROI", not optical
proof that a photo was captured.

### Smoking

Smoking evidence is assigned to a specific track. Cigarette persistence,
cigarette near mouth, hand near mouth and cigarette, nearby lighter/flame, and
nearby smoke contribute to confidence. Smoke without a cigarette or correlated
human action does not produce `smoking`. This prevents ordinary steam, fog, or
distant light from directly classifying a person as smoking.

Small cigarette detections can disappear briefly without ending a candidate.
The allowed gap is controlled by `smoking.max_gap_frames`. A custom model is
required for real cigarette/smoke/flame inference.

## Configuration

The feature config is `configs/rk3588_behavior_detection.yaml`. It includes:

- `enabled` and per-feature phone/smoking switches.
- camera identity and camera, RTSP, or video source references.
- primary and optional behavior model paths, class order, confidence, NMS,
  frame interval, SHA-256, and RKNN core mask.
- class groups used by association instead of hard-coded class IDs.
- per-event duration, evidence frames, confidence, gap, and cooldown.
- inline JSON ROI definitions with normalized or pixel coordinates.
- evidence snapshot, clip, log, debug-frame, JPEG, and queue settings.
- logging level.

Example enabled ROI:

```yaml
phone:
  prohibited_rois: [{"id":"critical-equipment","enabled":true,"normalized":true,"x1":0.65,"y1":0.15,"x2":0.95,"y2":0.85}]
```

Example custom behavior model activation:

```yaml
models:
  behavior:
    enabled: true
    required: true
    model_path: models/behavior_yolov8n.rknn
    class_names: ["person", "cell phone", "cigarette", "smoke", "flame", "lighter", "hand"]
    class_filter: ["cigarette", "smoke", "flame", "lighter", "hand"]
    core_mask: "2"
    expected_sha256: "PUT_VERIFIED_SHA256_HERE"
```

`class_names` must exactly match the training dataset order. A mismatched order
will produce semantically wrong detections even when the tensor shapes are
valid.

The primary detector remains controlled by `config.yaml` unless
`models.primary.use_for_runtime: true` is set in the behavior config. This
preserves existing model-selection workflows while allowing one-file overrides
for a dedicated deployment.

## Input Sources

The main `config.yaml` supports all three source types:

```yaml
camera:
  source_type: camera  # camera, rtsp, or video
  camera_device: /dev/video11
  rtsp_url: ""
  video_file: data/sample.mp4
```

The existing `config.yaml` remains authoritative by default. Set
`source.use_for_runtime: true` in `configs/rk3588_behavior_detection.yaml` only
when the unified behavior config should override `source_type`, camera device,
RTSP URL, or video path.

RTSP credentials are used only to open the stream. Status APIs redact the user
and password from the displayed URL. Do not commit a real RTSP URL containing
credentials.

## Model Preparation on x86 Linux

Use a model whose license permits the intended deployment. Train or obtain an
ONNX detector with the exact classes configured under `class_names`. Do not
convert on the RK3588.

Install the Rockchip Toolkit2 version compatible with the board runtime in an
x86 Linux virtual environment, then run:

```bash
python scripts/convert_behavior_onnx_to_rknn.py \
  --onnx artifacts/behavior_yolov8n.onnx \
  --dataset artifacts/calibration.txt \
  --output models/behavior_yolov8n.rknn
```

The calibration file contains one representative image path per line. The
script prints SHA-256 after conversion. Put that value in
`models.behavior.expected_sha256`.

The postprocessor accepts the Rockchip model-zoo YOLOv8 multi-head layout and
common decoded YOLO rows. Validate output tensor layout, quantization, input
color order, and class mapping with representative images before deployment.

Copy and verify the model without adding it to Git:

```bash
scp models/behavior_yolov8n.rknn cat@RK3588_IP:/home/cat/projects/person-tracking/models/
ssh cat@RK3588_IP 'cd ~/projects/person-tracking && sha256sum models/behavior_yolov8n.rknn'
```

All `.rknn`, `.onnx`, `.pt`, and related model files remain ignored.

## Deployment

```bash
cd /home/cat/projects/person-tracking
source .venv/bin/activate
python -m unittest discover -s tests -v
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Useful endpoints:

```text
GET  /api/health
GET  /api/stats
GET  /api/events?limit=50
POST /api/reset-stats
GET  /video
```

`/api/health` reports `phone_detection_available`,
`smoking_detection_available`, `behavior_status`, auxiliary model status, and
behavior timing fields. `behavior_status=degraded` means phone detection is
usable but an enabled optional behavior model is missing. A missing required
model makes behavior health an error without crashing the camera service.

## Event JSON

```json
{
  "event_id": "a102c4f1-f1bd-41e4-8b5e-3d51629e9ad6",
  "event_type": "phone_call",
  "camera_id": "rk3588-camera-01",
  "timestamp": "2026-07-15T03:25:10.123456+00:00",
  "track_id": 12,
  "confidence": 0.84,
  "duration_ms": 1834,
  "bboxes": {
    "person": [120.0, 45.0, 420.0, 710.0],
    "phone": [286.0, 126.0, 322.0, 185.0],
    "face": [224.0, 72.0, 340.0, 224.0]
  },
  "snapshot_path": "data/behavior_events/snapshots/..._evidence.jpg",
  "annotated_snapshot_path": "data/behavior_events/snapshots/..._annotated.jpg",
  "video_clip_path": null,
  "evidence": ["phone_near_detected_face"],
  "persistence_status": "persisted",
  "persistence_error": ""
}
```

Evidence frames are generated after face/head mosaic. The unannotated snapshot
means no behavior boxes or labels have been drawn; it does not mean unmasked.
Video clips are represented in the event schema but disabled by default.

Register replaceable callbacks with:

```python
runtime.behavior_events.register_callback(my_callback)
```

Callbacks run on the event-writer thread. A callback or disk failure is logged
and reported in stats without stopping camera inference.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use deterministic mock detections. They verify state-machine logic and do
not claim model accuracy. Covered cases include short phone use, all three phone
events, smoking evidence fusion, smoke-only rejection, deduplication, cooldown,
track cleanup, evidence writes, unwritable paths, missing models, missing video,
and invalid/redacted RTSP configuration.

## Benchmark

Run on the RK3588 with the camera free:

```bash
python scripts/benchmark_behavior_pipeline.py --frames 300 \
  --json-output data/benchmark-behavior-rk3588.json
```

The result records input resolution, average FPS, average and P95 end-to-end
latency, stage averages, process CPU, peak memory, detector/NPU status, and NPU
load when the board exposes a readable devfreq load node. A local video can be
used with `--video`. Do not quote performance until this command has run on the
target configuration.

## Known Limits and Next Steps

- COCO phone detections can miss small, occluded, or motion-blurred phones.
- Face/head geometry is not full pose, hand-keypoint, eye-gaze, or action
  recognition.
- The photography rule cannot prove lens direction or that an image was saved.
- Real smoking inference requires the absent custom behavior RKNN model.
- Smoke appearance varies strongly across ventilation, lighting, steam, and
  camera exposure. Site-specific validation and likely fine tuning are needed.
- Configure ROIs per camera and validate perspective before enabling alerts.
- A future pose RKNN model can replace geometric proxies through the same
  behavior-engine input contract.
- Video-clip buffering can be added behind the existing optional schema without
  changing event consumers.
