from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import cv2
import numpy as np

from app.behavior.events import BehaviorEventManager
from app.behavior.types import BehaviorTrigger
from app.config import PROJECT_ROOT, load_accuracy_profile


class NoMosaicRuntimeTests(TestCase):
    def test_profile_requires_unredacted_output_and_disables_face_pipeline(
        self,
    ) -> None:
        profile = load_accuracy_profile()

        self.assertEqual(
            profile["privacy"],
            {
                "face_detection_enabled": False,
                "mosaic_enabled": False,
                "unredacted_video_enabled": True,
                "unredacted_evidence_enabled": True,
            },
        )

    def test_main_runtime_has_no_face_processor_dependency(self) -> None:
        source = (PROJECT_ROOT / "app" / "main.py").read_text(encoding="utf-8")

        self.assertNotIn("from app.privacy", source)
        self.assertNotIn("FaceMosaicProcessor", source)
        self.assertNotIn("face_boxes()", source)
        self.assertNotIn(".privacy.process(", source)

    def test_health_privacy_contract_is_explicit(self) -> None:
        source = (PROJECT_ROOT / "app" / "main.py").read_text(encoding="utf-8")

        for key in (
            "face_detection_enabled",
            "face_model_loaded",
            "mosaic_enabled",
            "privacy_mode",
            "unredacted_video_enabled",
            "unredacted_evidence_enabled",
        ):
            self.assertIn(f'"{key}"', source)

    def test_event_evidence_preserves_raw_high_frequency_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "snapshot_dir": "events",
                "event_log_path": "events.jsonl",
                "save_unannotated_snapshot": True,
                "save_annotated_snapshot": False,
                "jpeg_quality": 100,
                "queue_size": 2,
            }
            manager = BehaviorEventManager(config, "test-camera", project_root=root)
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
            frame[:, ::2] = 255
            trigger = BehaviorTrigger(
                "smoking",
                1,
                0.9,
                2000,
                (8, 8, 88, 92),
                {"cigarette": (45, 25, 52, 35)},
                ("verified_cigarette",),
            )
            manager.start()
            payload = manager.emit(trigger, frame)
            manager._queue.join()
            manager.stop()

            saved = cv2.imread(str(root / payload["snapshot_path"]))
            self.assertIsNotNone(saved)
            horizontal_change = np.abs(
                saved[:, 1:].astype(np.int16) - saved[:, :-1].astype(np.int16)
            ).mean()
            self.assertGreater(horizontal_change, 80.0)

    def test_dashboard_warns_that_faces_are_unredacted(self) -> None:
        html = (PROJECT_ROOT / "app" / "static" / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("当前服务未启用人脸遮挡", html)
        self.assertIn("不得将该服务暴露到公网", html)
