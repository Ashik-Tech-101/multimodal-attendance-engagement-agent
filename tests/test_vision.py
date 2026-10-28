import unittest
from unittest.mock import patch

import cv2
import numpy as np

from attendance_agent.vision import CentroidTracker, analyze_image, analyze_video


class CentroidTrackerTests(unittest.TestCase):
    def test_update_returns_active_tracks_and_hides_missed_tracks(self) -> None:
        tracker = CentroidTracker()

        first = tracker.update([(10, 20, 30, 40)], timestamp=0, step=1)
        missed = tracker.update([], timestamp=1, step=1)
        seen_again = tracker.update([(12, 21, 30, 40)], timestamp=2, step=1)

        self.assertEqual(first, [{"participant_id": "P001", "x": 10, "y": 20, "w": 30, "h": 40}])
        self.assertEqual(missed, [])
        self.assertEqual(seen_again, [{"participant_id": "P001", "x": 12, "y": 21, "w": 30, "h": 40}])

    def test_tracker_accumulates_a_coarse_motion_signal(self) -> None:
        tracker = CentroidTracker()

        tracker.update([(10, 20, 30, 40)], timestamp=0, step=1)
        tracker.update([(20, 20, 30, 40)], timestamp=1, step=1)

        track = tracker.tracks[1]
        self.assertGreater(track["motion"], 0)
        self.assertEqual(track["motion_count"], 1)

    def test_analyze_image_returns_annotated_rgb_frame_and_face_count(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        encoded, buffer = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)

        result = analyze_image(buffer.tobytes())

        self.assertEqual(result["frame_rgb"].shape, (120, 160, 3))
        self.assertEqual(result["visible_faces"], 0)

    def test_analyze_image_rejects_invalid_image_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not decode the camera image"):
            analyze_image(b"not an image")


class FakeVideoCapture:
    def __init__(self, _path: str):
        self.frame_index = -1
        self.grabbed_frames = 0
        self.retrieved_frames = 0

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {
            cv2.CAP_PROP_FPS: 4,
            cv2.CAP_PROP_FRAME_COUNT: 8,
            cv2.CAP_PROP_FRAME_WIDTH: 160,
            cv2.CAP_PROP_FRAME_HEIGHT: 120,
        }.get(property_id, 0)

    def grab(self) -> bool:
        self.frame_index += 1
        self.grabbed_frames += 1
        return self.frame_index < 8

    def retrieve(self) -> tuple[bool, np.ndarray]:
        self.retrieved_frames += 1
        return True, np.zeros((120, 160, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class EmptyCascade:
    def __init__(self, _path: str):
        pass

    def empty(self) -> bool:
        return False

    def detectMultiScale(self, *_args, **_kwargs) -> np.ndarray:
        return np.empty((0, 4), dtype=int)


class AnalyzeVideoTests(unittest.TestCase):
    def test_intermediate_frames_are_grabbed_without_retrieving_image_data(self) -> None:
        capture = FakeVideoCapture("sample.mp4")
        with (
            patch("attendance_agent.vision.cv2.VideoCapture", return_value=capture),
            patch("attendance_agent.vision.cv2.CascadeClassifier", EmptyCascade),
        ):
            result = analyze_video("sample.mp4", sample_interval=1)

        self.assertEqual(capture.grabbed_frames, 9)
        self.assertEqual(capture.retrieved_frames, 2)
        self.assertEqual(result["video_samples"], 2)
        self.assertEqual(result["timeline"]["time_seconds"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
