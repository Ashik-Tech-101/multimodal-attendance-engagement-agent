import time
import unittest
from unittest.mock import patch

import numpy as np

from attendance_agent.live_camera import LiveCameraSession


class FakeVideoCapture:
    def __init__(self, _camera_index: int):
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, _property: int, _value: float) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == 3:
            return 640
        if property_id == 4:
            return 480
        return 0

    def read(self) -> tuple[bool, np.ndarray]:
        time.sleep(0.005)
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class FakeCascade:
    def __init__(self, _cascade_path: str):
        self.detections = iter(
            [
                np.array([[10, 20, 40, 40]]),
                np.array([[70, 20, 40, 40]]),
            ]
        )

    def empty(self) -> bool:
        return False

    def detectMultiScale(self, *_args, **_kwargs) -> np.ndarray:
        try:
            return next(self.detections)
        except StopIteration:
            return np.empty((0, 4), dtype=int)


class LiveCameraSessionTests(unittest.TestCase):
    def test_snapshot_includes_track_presence_and_movement_percentages(self) -> None:
        with (
            patch("attendance_agent.live_camera.cv2.VideoCapture", FakeVideoCapture),
            patch("attendance_agent.live_camera.cv2.CascadeClassifier", FakeCascade),
        ):
            camera = LiveCameraSession(min_presence_seconds=1)
            try:
                camera.update()
                time.sleep(0.02)
                snapshot = camera.update()
            finally:
                camera.close()

        self.assertEqual(snapshot["participants"][0]["participant_id"], "P001")
        self.assertGreater(snapshot["participants"][0]["activity_percent"], 0)
        self.assertGreaterEqual(snapshot["participants"][0]["presence_percent"], 0)
        self.assertLessEqual(snapshot["participants"][0]["presence_percent"], 100)
        self.assertEqual(snapshot["participants"][0]["visibility_status"], "Visible now")
        self.assertEqual(snapshot["participants"][0]["movement_summary"], "Frequent face-position changes")
        self.assertEqual(snapshot["frame_rgb"].shape, (480, 640, 3))

    def test_snapshot_marks_unseen_tracks_without_claiming_the_person_left(self) -> None:
        with (
            patch("attendance_agent.live_camera.cv2.VideoCapture", FakeVideoCapture),
            patch("attendance_agent.live_camera.cv2.CascadeClassifier", FakeCascade),
        ):
            camera = LiveCameraSession()
            try:
                camera.update()
                time.sleep(0.02)
                camera.update()
                time.sleep(0.02)
                snapshot = camera.update()
            finally:
                camera.close()

        self.assertEqual(snapshot["participants"][0]["visibility_status"], "Temporarily not detected")

    def test_snapshot_marks_long_gaps_as_out_of_view(self) -> None:
        with (
            patch("attendance_agent.live_camera.cv2.VideoCapture", FakeVideoCapture),
            patch("attendance_agent.live_camera.cv2.CascadeClassifier", FakeCascade),
        ):
            camera = LiveCameraSession()
            try:
                camera.update()
                camera.cascade.detections = iter([np.empty((0, 4), dtype=int)])
                camera.tracker.tracks[1]["last"] = -11
                snapshot = camera.update()
            finally:
                camera.close()

        self.assertEqual(snapshot["participants"][0]["visibility_status"], "Out of view (10s+)")


if __name__ == "__main__":
    unittest.main()
