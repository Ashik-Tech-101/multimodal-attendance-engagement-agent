"""Privacy-preserving audio feature extraction; no transcription or speaker ID."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def analyze_audio(path: str | Path, interval_seconds: int = 30) -> dict:
    try:
        import librosa
        y,sr=librosa.load(str(path),sr=16000,mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc
    if y.size==0: raise ValueError("The audio file contains no samples.")
    window=max(1,int(interval_seconds*sr)); rows=[]
    for start in range(0,len(y),window):
        chunk=y[start:start+window]
        rms=float(np.sqrt(np.mean(chunk.astype(np.float64)**2))) if len(chunk) else 0.0
        zcr=float(np.mean(librosa.feature.zero_crossing_rate(chunk))) if len(chunk)>1 else 0.0
        rows.append({"start_seconds":start/sr,"end_seconds":min((start+window)/sr,len(y)/sr),"rms":rms,"zcr":zcr})
    df=pd.DataFrame(rows)
    # A relative activity proxy avoids treating a fixed device gain as universal speech volume.
    floor=float(df["rms"].quantile(.2)); ceiling=float(df["rms"].quantile(.9))
    df["speech_activity"]=[float(np.clip((r-floor)/max(ceiling-floor,1e-6),0,1)) if r>floor*1.6 and z>.005 else 0.0 for r,z in zip(df.rms,df.zcr)]
    df["noise_level"]=(df["rms"].clip(0,1)*100).round(1)
    return {"timeline":df,"duration_seconds":len(y)/sr,"mean_rms":float(df.rms.mean()),"mean_activity":float(df.speech_activity.mean())}
