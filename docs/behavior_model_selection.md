# Smoking Behavior Model Selection

## Decision

The deployed direct smoking evidence model is the ModelScope
`iic/cv_tinynas_object-detection_damoyolo_cigarette` DAMO-YOLO-S checkpoint.
It was selected because it has an official source, Apache-2.0 licensing, a
single relevant `cigarette` target, substantially better measured CigDet AP50
than the smaller YOLOv8n candidate, and an RK3588 INT8 result that preserves
small-target accuracy within the project's five-point mAP budget.

Smoke, flame, and lighter are not presented as direct proof of smoking. The
deployed event requires a real cigarette model output associated with a person
track and sustained by the temporal state machine.

## Downloaded Candidates

| Candidate | Fixed source | Weight, SHA-256, size | License | Classes | Architecture/input | ONNX and RK3588 decision |
| --- | --- | --- | --- | --- | --- | --- |
| ModelScope DAMO cigarette | ModelScope tag `v1.1.0`, commit `b757d03ab58d9c30fe1e59b6af6d19431381d5ce`; DAMO code commit `319572eef367340267ab6ab8ae253527a71d7c3a` | `damoyolo_tinynasL25_S_cigarette.pt`; `daae5418929e166b92a9551c7d9686bd670cf8a7a6f0850d8d722cc3aa00079f`; 130,945,475 bytes | Apache-2.0 | Card: `cigarette`; verified head: channel 0 cigarette, channel 1 unused | DAMO-YOLO-S, TinyNAS L25, about 16.3M parameters/37.8 GFLOPs, 640x640 | Static ONNX export and RKNN 2.3.0 conversion passed. Selected for measured AP50 0.8118 and INT8 AP50 0.7763. |
| imvansh02 YOLOv8n smoking | Git commit `a51a53f74c4e0c8439e6f9ecdb683a435c3388eb` | `server/best.pt`; `6805e7a8a0b65601f0ea0faacc46c7c8521831892912f9d25f0a818c877a38fc`; 6,225,706 bytes | Checkpoint metadata says AGPL-3.0; repository has no standalone license file | `cigarette`, `face`, `smoking` | YOLOv8n, 3,011,433 parameters, 640x640 | Exportable and faster on CPU, but CigDet AP50 was only 0.5872. Rejected for lower cigarette recall/precision and weaker repository licensing clarity. |

PyTorch weights were inspected for unsafe globals before loading. The DAMO
checkpoint loaded with `weights_only=True`; the YOLOv8 checkpoint was loaded
only after allowlisting expected Torch and Ultralytics types. No unknown model
repository scripts were executed.

Other rejected searches included Hugging Face models with no model card or
license, contradictory architecture metadata, smoke-only classes, or much
larger YOLO11m/YOLOv8m weights. Smoke-only and fire-only models cannot satisfy
the direct human-smoking evidence requirement.

## Measured Candidate Accuracy

Both downloaded candidates were evaluated on all 111 labeled CigDet v1 test
images with the same IoU 0.5 evaluator.

| Candidate | AP50 | TP/FP/FN at 0.35 | Precision | Recall | F1 | Mean CPU inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DAMO cigarette | 0.8118 | 93 / 48 / 18 | 0.6596 | 0.8378 | 0.7381 | 424.28 ms |
| YOLOv8n smoking | 0.5872 | 67 / 48 / 44 | 0.5826 | 0.6036 | 0.5929 | 85.69 ms |

The YOLOv8n candidate produced useful detections on the three repository demo
images, including direct `smoking` and `cigarette` classes, but missed too many
of the independently labeled cigarette targets. Actual outputs, not README
claims, determined the selection.

## Validation Material

- CigDet v1: 557 cigarette images, 446 train and 111 test, CC BY 4.0,
  DOI `10.17632/6hyrr8typ7.1`. The downloaded archive SHA-256 is
  `dafc638d144dd79e39660c7cfdc044fd081c26684415d83a57e5dabdc9248b29`.
- COCO128: 128 COCO validation images used for people, phones, bottles, cups,
  eating utensils, food, and background negatives. Archive SHA-256:
  `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`.
- Curated Wikimedia Commons negatives: drinking water (CC0), eating food
  (CC0), holding a pen (public domain), phone near face (public domain), steam
  (CC BY-SA 4.0), fog/backlight (CC BY 2.0), butane lighter (CC BY-SA 4.0),
  warehouse (public domain in the US), garage (public-domain dedication),
  smoke cloud (CC BY 2.0), and machine room (public domain).

This set covers cigarette near mouth, actual smoking imagery, small distant
cigarettes, hand-to-mouth confounders, drinking, eating, pens, phones, steam,
fog, backlight, smoke, lighter, office, warehouse, garage, and machine-room
backgrounds. Large validation images and annotated renderings stay in ignored
project cache and are not committed.

## Negative Findings

The final INT8 RKNN produced no detection on 9 of 11 curated negatives. It
misclassified the close-up lighter as cigarette at 0.601 and produced a tiny
cigarette box in steam at 0.436. These are documented limitations. They do not
independently create a smoking event: the object must be assigned to a tracked
person, be near a detected or estimated face, and persist for the configured
duration. Site-specific fine tuning remains appropriate if these objects are
common in production.

Machine-readable measurements are in
`artifacts/behavior_model_validation.json`; binary details are in
`models/behavior_model_manifest.json`.

## Sources

- ModelScope model: https://www.modelscope.cn/models/iic/cv_tinynas_object-detection_damoyolo_cigarette/summary
- DAMO-YOLO code: https://github.com/tinyvision/DAMO-YOLO
- YOLOv8n candidate: https://github.com/imvansh02/Yolov8n-based-smoking-action-recognition-module
- CigDet v1: https://data.mendeley.com/datasets/6hyrr8typ7/1
- COCO128 archive: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
- Wikimedia Commons: https://commons.wikimedia.org/
