"""Timestamp-based combination of vision and optional audio context."""
from __future__ import annotations

import pandas as pd


def fuse_signals(vision_timeline: pd.DataFrame, audio: dict | None, duration_seconds: float, interval_seconds: int = 30) -> pd.DataFrame:
    if duration_seconds<=0: return pd.DataFrame()
    rows=[]
    for start in range(0,max(1,int(duration_seconds)),interval_seconds):
        end=min(start+interval_seconds,duration_seconds)
        v=vision_timeline[(vision_timeline.time_seconds>=start)&(vision_timeline.time_seconds<end)] if not vision_timeline.empty else pd.DataFrame()
        presence=float((v.visible_faces>0).mean()) if not v.empty else 0
        visible=float(v.visible_faces.mean()) if not v.empty else 0
        motion=float(v.motion_index.mean()) if not v.empty else 0
        speech=0.0; noise=0.0
        if audio:
            a=audio["timeline"]
            a=a[(a.end_seconds>start)&(a.start_seconds<end)]
            if not a.empty: speech=float(a.speech_activity.mean()); noise=float(a.noise_level.mean())
        # Explainable heuristic: presence carries the strongest weight; motion/audio are supporting context.
        score=100*(.55*presence+.20*min(motion*3,1)+.25*speech)
        rows.append({"start_seconds":start,"end_seconds":end,"time_label":f"{start//60:02d}:{start%60:02d}","visible_faces_avg":round(visible,2),"presence_ratio":round(presence,3),"motion_index":round(motion,3),"speech_activity":round(speech,3),"noise_level":round(noise,1),"engagement_score":round(score,1)})
    return pd.DataFrame(rows)
