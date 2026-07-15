# RK3588 Phone and Smoking Behavior Detection

## Scope

This feature adds temporal behavior analysis to the existing local RK3588
person-tracking service. It supports these unified event types:

- `phone_call`: a phone remains close to a tracked person's face/head region.
- `phone_playing`: a phone remains inside the upper body and below the face.
- `unauthorized_photography`: a raised phone is geometrically aligned with a
  configured prohibited ROI.
- `smoking`: a real cigarette or direct-smoking model output is associated
  with one tracked person and sustained near the mouth; ignition and smoke
  can only strengthen that direct evidence.

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
| `models/behavior_damoyolo_cigarette_int8.rknn` | Direct cigarette evidence | Present, SHA-verified, and NPU tested |

The existing YOLOv8n model SHA-256 is:

```text
ff3a64e6fe180203128c8d42456b458d208d3a1e2217d63683af00d6194e82ea
```

Phone behavior uses the existing COCO model because the NPU postprocessor
returns both `person` and `cell phone` from one inference. Smoking uses the
Apache-2.0 ModelScope DAMO-YOLO cigarette checkpoint converted to an INT8 RKNN.
The deployed behavior model SHA-256 is:

```text
d04c43a3a695c9985fbd03db1e0a2956763374fd686d949b8cd96cabdc7c5941
```

Its strict output contract is two decoded tensors, scores `[1,8400,2]` and
boxes `[1,8400,4]`; score channel 0 is `cigarette` and channel 1 is unused.
Details and measured candidate comparisons are in
`docs/behavior_model_selection.md` and `models/behavior_model_manifest.json`.

## Data Flow

```text
V4L2/GStreamer, RTSP, or video file
  -> latest-frame camera thread
  -> YOLOv8n RKNN (person + cell phone)
  -> DAMO-YOLO cigarette INT8 RKNN every fifth frame
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

With both RKNN instances active, their Python/native wrappers create cyclic
objects that default GC can retain for long bursts. Production config runs
`gc.collect()` followed by glibc `malloc_trim(0)` every 300 processed frames.
The project-local supervisor also sets `MALLOC_ARENA_MAX=2`; no system setting
is changed. Health and stats expose GC/trim counts and errors.

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

Smoking evidence is assigned to a specific track. A cigarette must be near the
detected or estimated mouth region, near an associated hand, or come from a
direct `smoking` model class. Merely persisting elsewhere in a person box is
disabled in production. Nearby lighter/flame and smoke can raise confidence
only after direct evidence exists. They cannot independently produce a person
`smoking` event. This reduces false alerts from steam, fog, distant lights, and
the model's known lighter false positive.

Small cigarette detections can disappear briefly without ending a candidate.
The allowed gap is controlled by `smoking.max_gap_frames`. A latched event must
remain absent for `smoking.rearm_absence_ms` before the same track can emit a
new event, so a brief detector gap does not duplicate a continuous episode.

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
- latest-frame queue and low-frequency memory maintenance interval.
- logging level.

Example enabled ROI:

```yaml
phone:
  prohibited_rois: [{"id":"critical-equipment","enabled":true,"normalized":true,"x1":0.65,"y1":0.15,"x2":0.95,"y2":0.85}]
```

Deployed behavior model configuration:

```yaml
models:
  behavior:
    enabled: true
    required: true
    model_path: models/behavior_damoyolo_cigarette_int8.rknn
    model_family: damoyolo
    input_size: 640
    confidence_threshold: 0.35
    nms_threshold: 0.70
    class_names: {"0": "cigarette", "1": "__unused__"}
    class_filter: ["cigarette"]
    core_mask: "0_1_2"
    detect_every_n_frames: 5
    expected_sha256: d04c43a3a695c9985fbd03db1e0a2956763374fd686d949b8cd96cabdc7c5941
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

## Reproducing the RKNN Model

The selected original checkpoint, static ONNX, calibration list, conversion
environment, hashes, and consistency measurements are documented in
`docs/behavior_model_build_environment.md`. Conversion used the project-local
Toolkit2 2.3.0 environment on the RK3588; it did not alter system Python.

Build the INT8 artifact with:

```bash
python scripts/convert_behavior_onnx_to_rknn.py \
  --onnx damo_cigarette_640_static_opset12.onnx \
  --dataset damo_cigarette_300.txt \
  --output models/behavior_damoyolo_cigarette_int8.rknn \
  --model-family damoyolo --input-size 640 --target rk3588
```

The 300-image calibration list contains 220 CigDet positives, 73 COCO
negatives, and 7 curated negatives. The DAMO postprocessor validates the exact
tensor count, dimensions, class-map width, and finite outputs, then performs
class-aware NMS. A mismatch is a hard inference error rather than an empty
detection result.

Copy and verify the model without adding it to Git:

```bash
scp models/behavior_damoyolo_cigarette_int8.rknn cat@RK3588_IP:/home/cat/projects/person-tracking/models/
ssh cat@RK3588_IP 'cd ~/projects/person-tracking && sha256sum models/behavior_damoyolo_cigarette_int8.rknn'
```

All `.rknn`, `.onnx`, `.pt`, and related model files remain ignored.

## Deployment

```bash
cd /home/cat/projects/person-tracking
source .venv/bin/activate
python -m unittest discover -s tests -v
./scripts/manage_behavior_service.sh start
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
`smoking_detection_available`, `behavior_model_loaded`, `behavior_status`,
auxiliary-model NPU status, and behavior timing fields. Production health must
report `status=ok`, `camera_status=online`, `npu_enabled=true`,
`phone_detection_available=true`, `smoking_detection_available=true`, and
`behavior_model_loaded=true`. A missing or invalid required model is an
explicit health error; it never silently falls back to mock or motion output.

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

Tests use deterministic detections for state-machine behavior; model accuracy
is measured separately on CigDet. Covered cases include all three phone events,
cigarette association and mouth proximity, direct smoking classes, smoke and
ignition-only rejection, distant-cigarette rejection, deduplication, track
cleanup, strict RKNN shapes/classes, evidence writes, missing models, camera
reconnect, missing video, and invalid/redacted RTSP configuration.

## Benchmark

Run on the RK3588 with the camera free:

```bash
python scripts/benchmark_behavior_pipeline.py --frames 300 \
  --json-output data/benchmark-behavior-rk3588.json

python scripts/monitor_behavior_stability.py \
  --duration-seconds 1800 --interval-seconds 10 \
  --output data/stability-behavior-rk3588-final.json
```

The result records input resolution, average FPS, average and P95 end-to-end
latency, stage averages, process CPU, peak memory, detector/NPU status, and NPU
load when the board exposes a readable devfreq load node. A local video can be
used with `--video`. Do not quote performance until this command has run on the
target configuration.

The stability monitor also fails on API/camera/process errors, worker restarts,
or positive RSS growth/slope above its configured limits. Use
`scripts/profile_native_memory.py` to isolate capture, primary RKNN, behavior
RKNN, and privacy paths if memory regresses.

## Known Limits and Next Steps

- COCO phone detections can miss small, occluded, or motion-blurred phones.
- Face/head geometry is not full pose, hand-keypoint, eye-gaze, or action
  recognition.
- The photography rule cannot prove lens direction or that an image was saved.
- The deployed model detects cigarettes, not smoke, flame, lighter, hand pose,
  or the full smoking action. The event relies on person/face geometry and time.
- CigDet AP50 is 0.7763 after INT8 conversion. The curated lighter and steam
  negatives produced false cigarette boxes, so mouth association is required.
- Smoke appearance varies strongly across ventilation, lighting, steam, and
  camera exposure. Site-specific validation and likely fine tuning are needed.
- Configure ROIs per camera and validate perspective before enabling alerts.
- A future pose RKNN model can replace geometric proxies through the same
  behavior-engine input contract.
- Video-clip buffering can be added behind the existing optional schema without
  changing event consumers.
