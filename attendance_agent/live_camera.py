"""Incremental local webcam analysis for the Streamlit live view."""
from __future__ import annotations

import threading
import time
from datetime import timedelta

import cv2

from .vision import CentroidTracker

OUT_OF_VIEW_SECONDS = 10


class LiveCameraSession:
    def __init__(self, camera_index: int = 0, interval_seconds: int = 30, min_presence_seconds: int = 30):
        self.camera_index = int(camera_index)
        self.interval_seconds = max(10, int(interval_seconds))
        self.min_presence_seconds = max(1, int(min_presence_seconds))
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open camera {self.camera_index}. Check camera permissions or try another camera index.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.cascade.empty():
            self.cap.release()
            raise RuntimeError("OpenCV's bundled face detector could not be loaded.")
        diagonal = (self.width**2 + self.height**2) ** 0.5
        self.tracker = CentroidTracker(max_distance=max(110.0, diagonal * 0.08))
        self.started = time.monotonic()
        self.last_sample = self.started
        self.frame_count = 0
        self.current_bucket = 0
        self.bucket_samples = 0
        self.bucket_present = 0
        self.bucket_faces = 0
        self.bucket_motion = 0.0
        self.bucket_score = 0.0
        self.timeline: list[dict] = []
        self.previous_centers: dict[str, tuple[float, float]] = {}
        self.closed = False
        self._stop_capture = threading.Event()
        self._frame_ready = threading.Event()
        self._capture_lock = threading.Lock()
        self._latest_frame = None
        self._capture_error: Exception | None = None
        self._capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self._capture_thread.start()
        if not self._frame_ready.wait(timeout=2):
            self.close()
            raise RuntimeError("The camera opened but did not provide a frame within two seconds.")

    def _capture_frames(self) -> None:
        try:
            while not self._stop_capture.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    if not self._stop_capture.is_set():
                        with self._capture_lock:
                            self._capture_error = RuntimeError(
                                "The camera stopped returning frames. Check whether another app is using it."
                            )
                    return
                with self._capture_lock:
                    self._latest_frame = frame
                self._frame_ready.set()
        except Exception as exc:
            if not self._stop_capture.is_set():
                with self._capture_lock:
                    self._capture_error = exc

    def close(self) -> None:
        if not self.closed:
            self._stop_capture.set()
            self.cap.release()
            self._capture_thread.join(timeout=1)
            self.closed = True

    def _finish_bucket(self, bucket: int) -> None:
        if self.bucket_samples == 0:
            return
        start = self.current_bucket * self.interval_seconds
        self.timeline.append({
            "start_seconds": start,
            "end_seconds": start + self.interval_seconds,
            "time_label": str(timedelta(seconds=start))[:-3],
            "visible_faces_avg": round(self.bucket_faces / self.bucket_samples, 2),
            "presence_ratio": round(self.bucket_present / self.bucket_samples, 3),
            "motion_index": round(self.bucket_motion / self.bucket_samples, 3),
            "speech_activity": 0.0,
            "noise_level": 0.0,
            "engagement_score": round(self.bucket_score / self.bucket_samples, 1),
        })
        self.current_bucket = bucket
        self.bucket_samples = self.bucket_present = self.bucket_faces = 0
        self.bucket_motion = self.bucket_score = 0.0

    def update(self) -> dict:
        if self.closed:
            raise RuntimeError("This camera session has already stopped.")
        with self._capture_lock:
            capture_error = self._capture_error
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        if capture_error is not None:
            raise RuntimeError(f"Could not read from the camera: {capture_error}") from capture_error
        if frame is None:
            raise RuntimeError("The camera has not provided a frame yet.")
        now = time.monotonic()
        elapsed = max(0.0, now - self.started)
        step = min(max(0.0, now - self.last_sample), 1.0)
        self.last_sample = now
        self.frame_count += 1

        h, w = frame.shape[:2]
        scale = min(1.0, 640 / max(w, h))
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        found = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28))
        boxes = [(int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)) for x, y, bw, bh in found]
        active = self.tracker.update(boxes, elapsed, step)

        movements = []
        for person in active:
            x, y, bw, bh = person["x"], person["y"], person["w"], person["h"]
            center = (x + bw / 2, y + bh / 2)
            participant_id = person["participant_id"]
            old = self.previous_centers.get(participant_id)
            movement = 0.0
            if old:
                movement = min(
                    (((center[0] - old[0]) ** 2 + (center[1] - old[1]) ** 2) ** 0.5)
                    / max((w * w + h * h) ** 0.5, 1)
                    * 5,
                    1,
                )
            movements.append(movement)
            self.previous_centers[participant_id] = center
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (32, 205, 174), 2)
            cv2.putText(frame, participant_id, (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (32, 205, 174), 2)

        motion = sum(movements) / len(movements) if movements else 0.0
        is_present = int(bool(active))
        score = 100 * (0.55 * is_present + 0.20 * min(motion * 3, 1))
        bucket = int(elapsed // self.interval_seconds)
        if bucket > self.current_bucket:
            self._finish_bucket(bucket)
        self.bucket_samples += 1
        self.bucket_present += is_present
        self.bucket_faces += len(active)
        self.bucket_motion += motion
        self.bucket_score += score

        participants = []
        for tid, track in sorted(self.tracker.tracks.items()):
            participant_id = f"P{tid:03d}"
            seconds_since_seen = max(0.0, elapsed - track["last"])
            if track["missed"] == 0:
                visibility_status = "Visible now"
            elif seconds_since_seen >= OUT_OF_VIEW_SECONDS:
                visibility_status = "Out of view (10s+)"
            else:
                visibility_status = "Temporarily not detected"
            activity_percent = min(
                track["motion"] / max(track["motion_count"], 1) * 100,
                100,
            )
            participants.append({
                "participant_id": participant_id,
                "first_seen_seconds": round(track["first"], 1),
                "last_seen_seconds": round(track["last"], 1),
                "seconds_since_seen": round(seconds_since_seen, 1),
                "visibility_status": visibility_status,
                "presence_seconds": round(track["presence"], 1),
                "presence_percent": round(min(track["presence"] / max(elapsed, 1e-6) * 100, 100), 1),
                "activity_percent": round(activity_percent, 1),
                "movement_summary": (
                    "Frequent face-position changes"
                    if activity_percent >= 20
                    else "Some face-position changes"
                    if activity_percent >= 5
                    else "Little measured face-position change"
                ),
                "present": track["presence"] >= self.min_presence_seconds,
            })
        live_bucket = {
            "start_seconds": self.current_bucket * self.interval_seconds,
            "end_seconds": elapsed,
            "time_label": str(timedelta(seconds=self.current_bucket * self.interval_seconds))[:-3],
            "visible_faces_avg": round(self.bucket_faces / self.bucket_samples, 2),
            "presence_ratio": round(self.bucket_present / self.bucket_samples, 3),
            "motion_index": round(self.bucket_motion / self.bucket_samples, 3),
            "speech_activity": 0.0,
            "noise_level": 0.0,
            "engagement_score": round(self.bucket_score / self.bucket_samples, 1),
        }
        return {
            "frame_rgb": cv2.cvtColor(
                cv2.resize(
                    frame,
                    (
                        int(w * min(1.0, 960 / max(w, h))),
                        int(h * min(1.0, 960 / max(w, h))),
                    ),
                )
                if max(w, h) > 960
                else frame,
                cv2.COLOR_BGR2RGB,
            ),
            "elapsed_seconds": elapsed,
            "visible_faces": len(active),
            "detected_tracks": len(self.tracker.tracks),
            "average_score": round(sum(row["engagement_score"] for row in self.timeline + [live_bucket]) / max(len(self.timeline) + 1, 1), 1),
            "participants": participants,
            "timeline": self.timeline + [live_bucket],
            "video_samples": self.frame_count,
        }
