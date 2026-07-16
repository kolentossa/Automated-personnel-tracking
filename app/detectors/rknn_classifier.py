"""Small RKNN image classifier used to verify phone detector candidates."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from app.detectors.rknn_yolo import _resolve_core_mask
from vision.types import Frame


class RKNNPhoneVerifier:
    """Classify a candidate crop as ``not_phone`` or ``phone`` on the NPU."""

    def __init__(
        self,
        model_path: Path,
        *,
        input_size: int = 224,
        phone_class_id: int = 1,
        core_mask: Any = "0_1_2",
    ) -> None:
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:  # pragma: no cover - requires the board runtime
            raise RuntimeError("rknnlite.api.RKNNLite is not available") from exc

        self.model_path = Path(model_path)
        self.input_size = int(input_size)
        self.phone_class_id = int(phone_class_id)
        if self.input_size < 32:
            raise ValueError("Phone verifier input_size must be at least 32")
        if self.phone_class_id not in {0, 1}:
            raise ValueError("Phone verifier phone_class_id must be 0 or 1")
        self.last_profile: Dict[str, float] = _empty_profile()
        self.npu_enabled = False
        self._rknn = RKNNLite()
        result = self._rknn.load_rknn(str(self.model_path))
        if result != 0:
            raise RuntimeError(
                f"RKNN load_rknn failed with code {result}: {self.model_path}"
            )
        resolved_core_mask = _resolve_core_mask(RKNNLite, core_mask)
        result = (
            self._rknn.init_runtime(core_mask=resolved_core_mask)
            if resolved_core_mask is not None
            else self._rknn.init_runtime()
        )
        if result != 0:
            raise RuntimeError(f"RKNN init_runtime failed with code {result}")
        self.npu_enabled = True

    def predict_phone_probability(self, crop: Frame) -> float:
        started = time.monotonic()
        input_image = _prepare_classifier_input(crop, self.input_size)
        preprocessed = time.monotonic()
        outputs = self._rknn.inference(inputs=[input_image])
        inferred = time.monotonic()
        probability = _phone_probability(outputs, self.phone_class_id)
        finished = time.monotonic()
        self.last_profile = {
            "preprocess_ms": _ms(preprocessed - started),
            "inference_ms": _ms(inferred - preprocessed),
            "postprocess_ms": _ms(finished - inferred),
        }
        return probability

    def release(self) -> None:
        rknn = getattr(self, "_rknn", None)
        if rknn is not None:
            try:
                rknn.release()
            except Exception:
                pass


def _prepare_classifier_input(crop: Frame, input_size: int) -> np.ndarray:
    if crop is None or crop.size == 0:
        raise ValueError("Phone verifier received an empty crop")
    height, width = crop.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    square = crop[top : top + side, left : left + side]
    resized = cv2.resize(
        square, (input_size, input_size), interpolation=cv2.INTER_LINEAR
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0)


def _phone_probability(outputs: Any, phone_class_id: int = 1) -> float:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        count = len(outputs) if isinstance(outputs, (list, tuple)) else 0
        raise ValueError(
            f"Phone verifier output count mismatch: expected 1, got {count}"
        )
    values = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    if values.size != 2:
        raise ValueError(
            f"Phone verifier output shape mismatch: expected 2 values, got {tuple(np.asarray(outputs[0]).shape)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Phone verifier output contains NaN or infinity")
    total = float(values.sum())
    if (
        np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.isclose(total, 1.0, atol=1e-3)
    ):
        shifted = values - float(values.max())
        exponent = np.exp(shifted)
        values = exponent / float(exponent.sum())
    return float(values[int(phone_class_id)])


def _empty_profile() -> Dict[str, float]:
    return {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)
