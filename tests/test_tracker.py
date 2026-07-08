from unittest import TestCase

from vision.tracking import ByteTrackTracker
from vision.types import Detection


class ByteTrackTrackerTests(TestCase):
    def test_preserves_id_for_overlapping_detection(self):
        tracker = ByteTrackTracker(iou_threshold=0.2)

        first = tracker.update([Detection((10, 10, 50, 80), 0.9)])
        second = tracker.update([Detection((14, 12, 54, 82), 0.88)])

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].track_id, second[0].track_id)
