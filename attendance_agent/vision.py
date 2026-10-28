"""Anonymous face detection, centroid tracking, and presence aggregation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd


def analyze_image(image_bytes: bytes) -> dict:
    """Detect faces in one browser-captured image without assigning track identities."""
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the camera image. Capture it again and retry.")

    height, width = frame.shape[:2]
    scale = min(1.0, 720 / max(width, height))
    small = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("OpenCV's bundled face detector could not be loaded.")

    found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28))
    for x, y, box_width, box_height in found:
        x0, y0 = int(x / scale), int(y / scale)
        box_width, box_height = int(box_width / scale), int(box_height / scale)
        cv2.rectangle(frame, (x0, y0), (x0 + box_width, y0 + box_height), (32, 205, 174), 2)

    return {
        "frame_rgb": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        "visible_faces": len(found),
    }


class CentroidTracker:
    """Small deterministic tracker; IDs are ephemeral and are not identities."""
    def __init__(self, max_distance: float = 110.0, max_missed: int = 3):
        self.max_distance, self.max_missed = max_distance, max_missed
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, boxes: list[tuple[int,int,int,int]], timestamp: float, step: float) -> list[dict]:
        centers = [(x+w/2,y+h/2) for x,y,w,h in boxes]
        unmatched = set(self.tracks)
        matched_boxes: set[int] = set()
        pairs = []
        for tid, track in self.tracks.items():
            for index, center in enumerate(centers):
                dx,dy=center[0]-track["center"][0],center[1]-track["center"][1]
                pairs.append((dx*dx+dy*dy,tid,index))
        for dist2,tid,index in sorted(pairs):
            if tid not in unmatched or index in matched_boxes or dist2 > self.max_distance**2: continue
            track=self.tracks[tid]; center=centers[index]
            moved=(dist2**0.5)/max(track["frame_diagonal"],1)
            track["presence"] += step; track["last"] = timestamp; track["missed"] = 0
            track["motion"] += min(moved,1.0); track["motion_count"] += 1
            track["center"] = center; track["box"] = boxes[index]
            unmatched.remove(tid); matched_boxes.add(index)
        for tid in unmatched:
            self.tracks[tid]["missed"] += 1
        for index, center in enumerate(centers):
            if index in matched_boxes: continue
            tid=self.next_id; self.next_id+=1
            x,y,w,h=boxes[index]
            self.tracks[tid]={"center":center,"box":boxes[index],"first":timestamp,"last":timestamp,"presence":step,"missed":0,"motion":0.0,"motion_count":0,"frame_diagonal":max(1,(w*w+h*h)**0.5*4)}
        # A brief absence is tolerated; longer gaps end the live association, but historical tracks remain for reporting.
        for tid in list(self.tracks):
            if self.tracks[tid]["missed"] > self.max_missed:
                self.tracks[tid]["center"] = (-1e6,-1e6)
        return [
            {
                "participant_id": f"P{tid:03d}",
                "x": track["box"][0],
                "y": track["box"][1],
                "w": track["box"][2],
                "h": track["box"][3],
            }
            for tid, track in self.tracks.items()
            if track["missed"] == 0
        ]


def analyze_video(path: str | Path, sample_interval: float = 1.0, progress_callback: Callable[[float],None] | None = None) -> dict:
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError("OpenCV could not open this video. Check the file and codec.")
    fps=cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps<=0: cap.release(); raise ValueError("The video does not report a valid frame rate.")
    duration=frame_count/fps if frame_count else 0
    if duration<=0: cap.release(); raise ValueError("The video appears to be empty.")
    step=max(0.5,float(sample_interval)); stride=max(1,round(fps*step))
    cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    if cascade.empty(): cap.release(); raise RuntimeError("OpenCV face detector could not be loaded.")
    diagonal=(frame_width**2+frame_height**2)**0.5
    tracker=CentroidTracker(max_distance=max(110.0,diagonal*0.08)); timeline=[]; frame_index=0; sample_count=0; previous_centers={}
    while True:
        ok=cap.grab()
        if not ok: break
        if frame_index % stride != 0:
            frame_index+=1; continue
        ok,frame=cap.retrieve()
        if not ok: break
        timestamp=frame_index/fps
        h,w=frame.shape[:2]; scale=min(1.0,720/max(w,h));
        if scale<1: frame=cv2.resize(frame,(int(w*scale),int(h*scale)))
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(28,28))
        boxes=[(int(x/scale),int(y/scale),int(bw/scale),int(bh/scale)) for x,y,bw,bh in faces]
        active=tracker.update(boxes,timestamp,step); sample_count+=1
        motions=[]
        for item in active:
            cx=item["x"]+item["w"]/2; cy=item["y"]+item["h"]/2
            old=previous_centers.get(item["participant_id"])
            if old: motions.append(min((((cx-old[0])**2+(cy-old[1])**2)**0.5)/max(w*w+h*h,1)**0.5*5,1))
            previous_centers[item["participant_id"]]=(cx,cy)
        timeline.append({"time_seconds":timestamp,"visible_faces":len(active),"motion_index":sum(motions)/len(motions) if motions else 0.0})
        if progress_callback and frame_count: progress_callback(min(frame_index/frame_count,1.0))
        frame_index+=1
    cap.release()
    participants=[]
    for tid,t in sorted(tracker.tracks.items()):
        participants.append({"participant_id":f"P{tid:03d}","first_seen_seconds":round(t["first"],1),"last_seen_seconds":round(t["last"],1),"presence_seconds":round(min(t["presence"],duration),1),"motion_index":round(t["motion"]/max(t["motion_count"],1),3)})
    return {
        "duration_seconds": round(duration, 2),
        "sample_interval_seconds": step,
        "video_samples": sample_count,
        "participants": participants,
        "timeline": pd.DataFrame(timeline),
    }
