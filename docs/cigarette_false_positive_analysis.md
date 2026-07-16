# Cigarette False-Positive Analysis

## Problem

The cigarette detector sees a very small elongated target. Phone edges, tools,
pens, utensils, fingers, and reflective machine edges can therefore produce a
raw cigarette-like box. Nested model boxes could also draw two boxes around one
object. A raw candidate must not be treated as a smoking event.

## Locked Validation Set

Private material is stored under ignored
`data/private_accuracy_validation/private-accuracy-corpus-v3/` and is never
committed. The manifest SHA-256 is
`f7231d5d8f7a7264adf2ecca43ca9abd1500f4bda1b31be04079c9aeda277709`.

```text
total images: 803
train: 267
tuning: 301
final_test: 235
```

Splits are scene/source grouped rather than random adjacent frames. The final
set includes 111 CigDet cigarette positives, 67 phone positives, RK3588 event
evidence, phone calls, phone use, phone context/back/side views, tools,
screwdrivers, pens, food, drinking, and elongated-object negatives.

## Pipeline Audit

The model and conversion path were checked in this order:

```text
PyTorch checkpoint -> static ONNX -> RKNN non-quantized -> RKNN INT8
-> RK3588 Runtime 2.3.0 -> Web preprocessing/postprocessing
```

The DAMO class map is channel 0 `cigarette`; channel 1 is explicitly unused.
Input is RGB, uint8 NHWC at runtime, direct 640x640 resize, range 0-255,
mean 0, std 1. Outputs are scores `[1,8400,2]` and xyxy boxes `[1,8400,4]`.
No duplicate sigmoid or YOLO DFL decode is applied because DAMO exports decoded
scores and boxes.

Measured consistency:

| Stage | Result |
| --- | ---: |
| PyTorch AP50 | 0.8118 |
| ONNX AP50 | 0.806662 |
| Non-quantized RKNN AP50 | 0.808908 |
| INT8 RKNN AP50 | 0.776251 |
| INT8 AP50 change from ONNX | -0.030411 |
| PT/ONNX matched mean IoU | 0.9999993 |
| INT8 small-target recall | 0.764706 |

The raw confusion is present in the source detector and is slightly affected by
INT8, rather than being introduced by BGR/RGB, layout, output-head order, or
Web coordinate restoration.

## Implemented Verification

The production path separates three states:

1. `raw cigarette candidate`: direct DAMO model output.
2. `verified cigarette`: candidate passes context, conflict, and temporal tests.
3. `confirmed smoking`: verified evidence persists for the state-machine
   duration and remains associated with a person track.

The verifier applies:

- Class-aware NMS and containment suppression for duplicate/nested boxes.
- Person association and upper-body position checks.
- Candidate size relative to person height and area.
- A minimum vertical center ratio of `0.125` to reject implausible top-edge
  fragments.
- Phone conflict using `intersection_area / cigarette_area`, not IoU. At IoA
  `>=0.48`, sufficient phone confidence, and a winning confidence margin, the
  candidate is rejected as `phone_overlap`.
- Explicit tool-label overlap rejection as `tool_verifier`.
- Multiple fresh inference observations inside an 18-frame window.
- Position continuity relative to the associated person's diagonal.
- Event duration, minimum frames, cooldown, and rearm behavior per track ID.

Filter reasons are exposed in health/stats and the audit output:
`phone_overlap`, `tool_verifier`, `no_person_association`,
`insufficient_temporal_evidence`, `invalid_size_context`, and
`low_verified_confidence`.

## Final Results

The before column is the immediately preceding v2 verifier on the same 235
images. The after column changes only the calibrated person-center and phone
IoA boundary. True cigarette recall is unchanged.

| Image-level confirmed event metric | Before | After |
| --- | ---: | ---: |
| TP / FP / FN / TN | 21 / 4 / 90 / 120 | 21 / 0 / 90 / 124 |
| Precision | 0.8400 | 1.0000 |
| Recall | 0.1892 | 0.1892 |
| F1 | 0.3088 | 0.3182 |
| True-cigarette recall change | | 0.0 percentage points |

| Negative group | Raw candidates | Before confirmed | After confirmed |
| --- | ---: | ---: | ---: |
| Phone call | 19 | 1 | 0 |
| Phone context/back/side | 3 | 0 | 0 |
| Phone playing | 10 | 0 | 0 |
| RK3588 board tools | 3 | 0 | 0 |
| Screwdriver | 0 | 0 | 0 |
| Elongated objects | 1 | 0 | 0 |
| Other board negatives/review | 10 | 3 | 0 |

Across all negative groups, confirmed false events fell 4 to 0, a 100%
reduction. Screwdriver and elongated groups were already blocked by the earlier
tool/context verifier; the final run confirms they remain at zero. This report
does not claim a reduction where the measured baseline was already zero.

The final event proxy metrics on fixed evidence are:

```text
smoking precision: 1.0000
smoking recall:    0.1892
smoking F1:        0.3182
duplicate events:  0
average trigger:   5250 ms in the static repeated-frame harness
```

## Phone Recall Change

At the conservative YOLOv8 ROI baseline (`0.20`), phone precision/recall/F1 was
0.7255/0.5522/0.6271. The deployed YOLOv8 plus YOLO11 scheduler measured
0.6935/0.6418/0.6667: recall improved 8.96 points and F1 improved 3.96 points,
with a 3.20-point precision tradeoff. Phone-call image hits improved from
16/42 to 22/42. This is an independent model/ROI change, not a mosaic effect.

## Limitations

- Event recall on static CigDet images is deliberately conservative because a
  candidate also needs person association and temporal behavior evidence.
- The final set contains only four dedicated screwdriver images; site-specific
  repair footage should be added before claiming broad tool coverage.
- Raw model candidates still occur on hard negatives. The verified/confirmed
  distinction is therefore a required safety property, not optional cleanup.
- True e-cigarettes, lighters, and unusual phone accessories remain ambiguous
  and need additional labeled data if they are common at the deployment site.
