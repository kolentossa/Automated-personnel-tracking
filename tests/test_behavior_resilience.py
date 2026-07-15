from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from app.behavior.detector import BehaviorObjectDetector
from app.behavior.events import BehaviorEventManager
from app.behavior.types import BehaviorTrigger
from app.camera import CameraSource, _redact_rtsp_url


class BehaviorEventManagerTests(TestCase):
    def test_persists_unannotated_and_annotated_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = BehaviorEventManager(_event_config(), "camera-test", project_root=root)
            manager.start()
            manager.emit(_trigger(), np.zeros((240, 320, 3), dtype=np.uint8))
            manager._queue.join()
            recent = manager.recent(1)[0]
            manager.stop()
            self.assertEqual(recent["persistence_status"], "persisted")
            self.assertTrue((root / recent["snapshot_path"]).exists())
            self.assertTrue((root / recent["annotated_snapshot_path"]).exists())
            self.assertTrue((root / "logs/events.jsonl").exists())

    def test_unwritable_shape_is_reported_without_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocked = root / "blocked"
            blocked.write_text("not a directory", encoding="utf-8")
            config = _event_config()
            config["snapshot_dir"] = "blocked/snapshots"
            config["event_log_path"] = "blocked/events.jsonl"
            manager = BehaviorEventManager(config, "camera-test", project_root=root)
            manager.start()
            manager.emit(_trigger(), np.zeros((120, 160, 3), dtype=np.uint8))
            manager._queue.join()
            snapshot = manager.snapshot()
            manager.stop()
            self.assertGreaterEqual(snapshot["behavior_event_write_failures"], 1)
            self.assertTrue(snapshot["behavior_event_last_error"])


class FailurePathTests(TestCase):
    def test_missing_behavior_model_is_explicit_and_safe(self) -> None:
        detector = BehaviorObjectDetector({
            "enabled": True,
            "required": True,
            "model_path": "models/does-not-exist.rknn",
            "class_names": ["cigarette"],
            "class_filter": ["cigarette"],
        })
        self.assertFalse(detector.available)
        self.assertIn("not found", detector.error)
        self.assertEqual(detector.detect(np.zeros((32, 32, 3), dtype=np.uint8), 0), [])

    def test_missing_video_reports_error_without_crashing(self) -> None:
        camera = CameraSource({
            "source_type": "video",
            "video_file": "data/does-not-exist.mp4",
            "retry_interval_sec": 0,
        })
        self.assertIsNone(camera.read())
        self.assertEqual(camera.status, "error")
        self.assertIn("does not exist", camera.error)

    def test_connected_camera_read_failure_releases_for_reconnect(self) -> None:
        capture = _FailedCapture()
        camera = CameraSource({"source_type": "camera", "camera_device": "/dev/video0", "retry_interval_sec": 0})
        camera._capture = capture
        camera.selected_source = "/dev/video0"
        camera.status = "online"
        self.assertIsNone(camera.read())
        self.assertTrue(capture.released)
        self.assertIsNone(camera._capture)
        self.assertEqual(camera.status, "error")
        self.assertIn("Could not read", camera.error)

    def test_invalid_rtsp_config_is_rejected_and_credentials_are_redacted(self) -> None:
        camera = CameraSource({"source_type": "rtsp", "rtsp_url": "http://invalid", "retry_interval_sec": 0})
        self.assertIsNone(camera.read())
        self.assertIn("must start", camera.error)
        redacted = _redact_rtsp_url("rtsp://user:secret@192.0.2.1/live")
        self.assertNotIn("user", redacted)
        self.assertNotIn("secret", redacted)


def _event_config() -> dict:
    return {
        "snapshot_dir": "data/events",
        "video_clip_dir": "data/clips",
        "event_log_path": "logs/events.jsonl",
        "save_unannotated_snapshot": True,
        "save_annotated_snapshot": True,
        "save_video_clip": False,
        "jpeg_quality": 85,
        "queue_size": 2,
    }


def _trigger() -> BehaviorTrigger:
    return BehaviorTrigger(
        event_type="phone_call",
        track_id=3,
        confidence=0.88,
        duration_ms=1800,
        person_bbox=(50, 30, 200, 230),
        object_bboxes={"phone": (140, 55, 165, 95)},
        evidence=("phone_near_detected_face",),
    )


class _FailedCapture:
    def __init__(self) -> None:
        self.released = False

    def read(self):
        return False, None

    def release(self) -> None:
        self.released = True
