from __future__ import annotations

from unittest import TestCase

from scripts.monitor_behavior_stability import _summarise


class StabilityMonitorTests(TestCase):
    def test_memory_stability_accepts_bounded_rss(self) -> None:
        summary = _summarise(_samples(250.0, 252.0), 60.0, 32.0, 128.0)

        self.assertTrue(summary["memory_stable"])
        self.assertEqual(summary["rss_growth_mb"], 2.0)

    def test_memory_stability_rejects_continuous_growth(self) -> None:
        summary = _summarise(_samples(250.0, 300.0), 60.0, 32.0, 128.0)

        self.assertFalse(summary["memory_stable"])
        self.assertGreater(summary["rss_slope_mb_per_hour"], 128.0)


def _samples(start_rss: float, end_rss: float) -> list[dict]:
    base = {
        "api_error": "",
        "status": "ok",
        "camera_status": "online",
        "supervisor_alive": True,
        "worker_alive": True,
        "supervisor_pid": 10,
        "worker_pid": 11,
    }
    return [
        {**base, "elapsed_seconds": 0.0, "rss_mb": start_rss},
        {**base, "elapsed_seconds": 60.0, "rss_mb": end_rss},
    ]
