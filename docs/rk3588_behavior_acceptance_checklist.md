# RK3588 Behavior Detection Acceptance Checklist

Record date, commit SHA, camera, model hashes, config revision, and operator for
every acceptance run. Do not reuse results after changing a model or threshold.

## Installation and Safety

- [ ] Branch is `feature/rk3588-phone-smoking-detection`.
- [ ] `git status` contains no model, credential, snapshot, clip, or log files.
- [ ] `sha256sum models/yolov8n.rknn` matches the approved value.
- [ ] Custom behavior model license and SHA-256 are recorded, if enabled.
- [ ] RTSP credentials, if used, are outside Git and redacted in `/api/health`.
- [ ] Evidence paths resolve inside the project directory.
- [ ] Face/head mosaic is visible before evidence testing.

## Startup and Failure Handling

- [ ] `python -m unittest discover -s tests -v` passes.
- [ ] Missing required behavior model is reported clearly.
- [ ] Missing optional behavior model produces `behavior_status=degraded` and
      person/phone service continues.
- [ ] Invalid camera, disconnected camera, invalid RTSP, and missing video show
      an API error without terminating FastAPI.
- [ ] Unwritable evidence directory increments write failures without stopping
      the stream.
- [ ] Restart restores camera and model operation.

## Phone Scenarios

- [ ] Phone near ear for less than `duration_ms` does not alert.
- [ ] Sustained phone near ear emits one `phone_call`.
- [ ] Sustained low phone position emits one `phone_playing`.
- [ ] Raised phone aligned with an enabled prohibited ROI emits one
      `unauthorized_photography`.
- [ ] Person passing inside/outside the ROI without aligned raised-phone
      geometry does not emit photography.
- [ ] A phone belonging to an adjacent person is not assigned to the wrong ID.
- [ ] One continuous behavior does not repeatedly alert.
- [ ] Behavior can alert again only after ending and cooldown.

## Smoking Scenarios

- [ ] Sustained cigarette with hand-to-mouth evidence emits one `smoking`.
- [ ] Persistent associated cigarette follows configured policy.
- [ ] Ignition evidence contributes only near an associated person.
- [ ] Smoke plus correlated person/cigarette action contributes to smoking.
- [ ] Steam/fog/smoke without associated cigarette/action does not classify a
      person as smoking.
- [ ] Small/intermittent cigarette remains stable across configured gap frames.
- [ ] Adjacent people do not receive each other's cigarette/smoke evidence.

## Events and Privacy

- [ ] Event JSON contains every documented required field.
- [ ] `track_id`, bboxes, confidence, duration, and evidence are correct.
- [ ] Unannotated evidence has no behavior overlay and is still face-mosaiced.
- [ ] Annotated evidence shows ID, event type, confidence, duration, and boxes.
- [ ] `/api/events` returns recent events in newest-first order.
- [ ] JSONL output is valid one-event-per-line JSON.
- [ ] Callback failure does not stop inference.
- [ ] No unmasked raw frame is written by default.

## Performance

- [ ] Run `python scripts/benchmark_behavior_pipeline.py --frames 300`.
- [ ] Record input resolution and exact model hashes.
- [ ] Record average FPS and average/P95 latency.
- [ ] Record stage timing, CPU, peak memory, and NPU load availability/value.
- [ ] Compare with behavior disabled using the same camera scene and config.
- [ ] Confirm no frame queue growth or increasing stream delay over 30 minutes.
- [ ] Confirm NPU core masks do not cause runtime contention when two models are
      enabled.

## Signoff

```text
Date:
Commit:
Camera/source:
Primary model SHA-256:
Behavior model SHA-256:
Config SHA-256:
Average FPS:
Average latency ms:
P95 latency ms:
CPU percent:
Peak memory MB:
NPU load:
Open issues:
Operator:
```
