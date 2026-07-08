from unittest import TestCase

from vision.counting import EntryExitCounter, LineCrossingConfig
from vision.types import TrackedPerson


class EntryExitCounterTests(TestCase):
    def test_counts_enter_when_track_crosses_left_to_right(self):
        counter = EntryExitCounter(LineCrossingConfig((50, 0), (50, 100), "positive_to_negative"))

        self.assertEqual(counter.update([TrackedPerson(1, (10, 10, 20, 60), 0.9)]), [])
        events = counter.update([TrackedPerson(1, (70, 10, 80, 60), 0.9)])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "ENTER")
        self.assertEqual(counter.snapshot()["current_people"], 1)
        self.assertEqual(counter.snapshot()["today_entered"], 1)

    def test_counts_exit_when_track_crosses_right_to_left(self):
        counter = EntryExitCounter(LineCrossingConfig((50, 0), (50, 100), "positive_to_negative"))
        counter.current_people_inside = 1

        self.assertEqual(counter.update([TrackedPerson(2, (70, 10, 80, 60), 0.9)]), [])
        events = counter.update([TrackedPerson(2, (10, 10, 20, 60), 0.9)])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "EXIT")
        self.assertEqual(counter.snapshot()["current_people"], 0)
        self.assertEqual(counter.snapshot()["today_exited"], 1)
