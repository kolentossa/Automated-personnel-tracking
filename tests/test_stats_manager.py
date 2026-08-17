from unittest import TestCase

from app.stats import StatsManager
from vision.types import TrackedPerson


FRAME_SHAPE = (100, 100)


def _track(x: float, track_id: int = 1) -> TrackedPerson:
    return TrackedPerson(track_id, (x - 2, 10, x + 2, 80), 0.9)


def _counter(*, cooldown_frames: int = 4, count_once: bool = False) -> StatsManager:
    return StatsManager(
        {
            "line": {"x1": 50, "y1": 0, "x2": 50, "y2": 100},
            "direction": {"mode": "left_to_right"},
            "cooldown_frames": cooldown_frames,
            "hysteresis_px": 8,
            "rearm_distance_px": 16,
            "confirmation_frames": 3,
            "track_state_ttl_frames": 20,
            "count_once_per_direction_per_track": count_once,
        }
    )


def _update(counter: StatsManager, positions: list[float]) -> None:
    for x in positions:
        counter.update_tracks([_track(x)], FRAME_SHAPE)


class StatsManagerCrossingTests(TestCase):
    def test_jitter_near_line_does_not_count(self):
        counter = _counter()
        _update(counter, [30, 30, 30])
        _update(counter, [45, 52, 47, 54, 44, 51, 48, 55] * 3)
        _update(counter, [40, 60, 40, 60])

        snapshot = counter.snapshot()
        self.assertEqual(snapshot["total_entered"], 0)
        self.assertEqual(snapshot["total_exited"], 0)
        self.assertEqual(snapshot["recent_events"], [])

    def test_crossing_requires_stable_opposite_side(self):
        counter = _counter()
        _update(counter, [30, 30, 30, 47, 53, 70, 70])
        self.assertEqual(counter.snapshot()["total_entered"], 0)

        _update(counter, [70])
        snapshot = counter.snapshot()
        self.assertEqual(snapshot["total_entered"], 1)
        self.assertEqual(snapshot["current_occupancy"], 1)

    def test_initial_side_can_arm_between_inner_and_outer_boundaries(self):
        counter = _counter(cooldown_frames=0)
        _update(counter, [40, 40, 40, 70, 70, 70])

        snapshot = counter.snapshot()
        self.assertEqual(snapshot["total_entered"], 1)
        self.assertEqual(snapshot["total_exited"], 0)

    def test_line_update_rebases_current_track_without_missing_next_crossing(self):
        counter = _counter(cooldown_frames=0)
        _update(counter, [30, 30, 30])

        counter.configure_counting(
            {"x1": 40, "y1": 0, "x2": 40, "y2": 100},
            "left_to_right",
        )
        self.assertEqual(counter.snapshot()["recent_events"], [])

        _update(counter, [60, 60, 60])
        snapshot = counter.snapshot()
        self.assertEqual(snapshot["total_entered"], 1)
        self.assertEqual(snapshot["total_exited"], 0)

    def test_line_lingering_after_crossing_does_not_repeat_event(self):
        counter = _counter()
        _update(counter, [30, 30, 30, 70, 70, 70])
        _update(counter, [45, 54, 47, 52, 44, 55, 49, 51] * 4)

        snapshot = counter.snapshot()
        self.assertEqual(snapshot["total_entered"], 1)
        self.assertEqual(snapshot["total_exited"], 0)
        self.assertEqual(len(snapshot["recent_events"]), 1)

    def test_stable_return_after_cooldown_counts_exit(self):
        counter = _counter(cooldown_frames=4)
        _update(counter, [30, 30, 30, 70, 70, 70])
        _update(counter, [70, 70, 70, 70, 30, 30, 30])

        snapshot = counter.snapshot()
        self.assertEqual(snapshot["total_entered"], 1)
        self.assertEqual(snapshot["total_exited"], 1)
        self.assertEqual(snapshot["current_occupancy"], 0)

    def test_transition_during_cooldown_is_delayed_not_swallowed(self):
        counter = _counter(cooldown_frames=6)
        _update(counter, [30, 30, 30, 70, 70, 70])
        _update(counter, [30, 30, 30])
        self.assertEqual(counter.snapshot()["total_exited"], 0)

        _update(counter, [30, 30, 30])
        snapshot = counter.snapshot()
        self.assertEqual(snapshot["total_entered"], 1)
        self.assertEqual(snapshot["total_exited"], 1)

    def test_same_track_counts_each_direction_only_once(self):
        counter = _counter(cooldown_frames=0, count_once=True)
        _update(counter, [30, 30, 30, 70, 70, 70])
        _update(counter, [30, 30, 30, 70, 70, 70, 30, 30, 30])

        snapshot = counter.snapshot()
        self.assertEqual(snapshot["total_entered"], 1)
        self.assertEqual(snapshot["total_exited"], 1)
        self.assertEqual(len(snapshot["recent_events"]), 2)

    def test_same_track_rearms_after_full_crossings(self):
        counter = _counter(cooldown_frames=0)
        _update(counter, [30, 30, 30, 70, 70, 70])
        _update(counter, [30, 30, 30, 70, 70, 70])

        snapshot = counter.snapshot()
        self.assertEqual(snapshot["total_entered"], 2)
        self.assertEqual(snapshot["total_exited"], 1)

    def test_inner_corridor_crossings_wait_for_outer_confirmation_zone(self):
        counter = _counter(cooldown_frames=0)
        _update(counter, [30, 30, 30])
        _update(counter, [40, 40, 40, 60, 60, 60] * 3)

        self.assertEqual(counter.snapshot()["total_entered"], 0)
        _update(counter, [70, 70, 70])
        self.assertEqual(counter.snapshot()["total_entered"], 1)
