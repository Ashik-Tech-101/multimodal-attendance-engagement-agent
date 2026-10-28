# Class Tracker

A local-first classroom session analysis prototype. Class Tracker combines anonymous face-presence detections from video with optional room-level audio activity features, then produces attendance evidence, an explainable cohort timeline, and instructor prompts.

The Streamlit interface includes a dark live-classroom dashboard, a Teacher Central reflection view with in-session history, recording analysis, attendance review, and a methodology page.

## Capabilities

- Analyze a classroom video file with OpenCV frontal-face detection and lightweight centroid tracking.
- Process a live camera attached to the Streamlit computer with a background frame reader, annotated preview, temporary track IDs, live visibility status, per-track presence and movement-proxy percentages, interactive chart choices, and a stop control.
- Capture and analyze a single image from the browser camera when the app is opened from another device.
- Review a short in-memory history of sessions and compare descriptive group-level proxy trends in Teacher Central.
- Accumulate observed presence across sampled frames and apply a configurable attendance threshold.
- Optionally analyze a synchronized audio file for RMS energy and a simple speech-activity proxy (no transcription or speaker identification).
- Fuse signals into fixed time windows and produce a transparent 0–100 cohort indicator.
- Explore the report in Streamlit; download JSON and attendance CSV.
- Process media locally. The app does not persist uploaded files after processing.

## Quick start

Python 3.9+ is required; Python 3.10–3.12 is recommended for the widest native-wheel compatibility. Install [FFmpeg](https://ffmpeg.org/) if your audio formats need it. On Windows, use a recent Python installer and ensure Python is on PATH.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, create and activate `.venv` with `python3 -m venv .venv` and `source .venv/bin/activate`, then install the requirements and run `streamlit run app.py`.

Open the URL printed by Streamlit. Choose **Analyze recording** to upload a video and optionally an audio file from the same session start, or **Live monitor** to use the camera attached to the computer running Streamlit. Configure attendance duration and analysis intervals in the sidebar. Video analysis defaults to one face-detection sample every two seconds for faster processing; choose a shorter interval for more temporal detail. If the default camera is not found, try index `1`.

**Continuous camera** mode opens the camera attached to the computer running Streamlit through OpenCV; select **Stop camera** to release the device. **Capture a still image** requests browser camera permission and analyzes one captured image without tracking attendance. Browser camera access requires HTTPS, except for localhost during local development. Still images are sent to the Streamlit server for processing but are not saved. Live video is not recorded, and live audio is not captured.

## Project layout

```text
app.py                     Streamlit dashboard and local workflow
attendance_agent/
  vision.py                Face detection, anonymous tracking, presence
  live_camera.py           Incremental local webcam session handling
  audio.py                 Audio feature extraction
  fusion.py                Timestamp alignment and cohort scoring
  analytics.py             Session report and teaching prompts
requirements.txt           Compatible dependency ranges
```

## Workflow and scoring

1. **Vision:** The input video is sampled at the selected interval. Frames are resized for speed, and intermediate frames are skipped without retrieving full image data. Sampled frames are analyzed with OpenCV's bundled Haar frontal-face detector. A centroid tracker associates detections by image position. IDs such as `P001` are temporary within one analysis and are not identity claims.
2. **Attendance:** A track's observed sample intervals are accumulated. A track meets the attendance threshold when its observed duration reaches the configured minimum. Occlusion, camera movement, and detector misses can split tracks or undercount presence.
3. **Audio:** Librosa converts the optional audio to mono 16 kHz and calculates windowed RMS energy and zero-crossing rate. Their relative levels form a heuristic activity proxy. It does not separate speech from music/noise reliably and cannot tell instructor speech from learner speech.
4. **Fusion:** For each analysis window, the cohort score is `100 × (0.55 × presence ratio + 0.20 × normalized motion + 0.25 × audio activity)`. Without audio, the audio term is zero. The score is intentionally inspectable and adjustable in `attendance_agent/fusion.py`; it is not a validated learning or attention measurement.
5. **Live mode:** A background reader keeps the newest frame available while the Streamlit view updates the preview and face detector. Temporary track IDs, observed presence percentage, visibility status, and a coarse face-position movement proxy are shown with selectable interactive charts. “Out of view” means a face was not detected; it does not prove that a person has left. The face detector cannot determine phone use, studying, attention, or learning.
6. **Analytics:** Simple thresholds identify low or high cohort proxy periods and elevated RMS intervals. Prompts are decision support, not automated teaching decisions.

## Privacy and responsible use

- Use only with appropriate notice, consent, and institutional approval.
- The prototype performs face detection and temporary tracking, not face recognition. It does not infer emotion, identity, or individual engagement.
- It cannot determine phone use, studying, attention, or whether someone has actually left. “Out of view (10s+)” means only that a face was not detected for at least 10 seconds; occlusion and camera framing can produce the same result.
- The live activity timeline is based on camera visibility and coarse movement only. It cannot determine what a learner is doing, understanding, or feeling; it must not be used to grade, discipline, rank, or make consequential decisions about a learner.
- The interface does not run facial emotion analysis or rank students; camera movement and visibility are not reliable evidence of internal attention or learning.
- Confirm attendance manually. The output is evidence for review, not an authoritative record.
- Files are held in a temporary directory during analysis and removed when the run ends; reports are kept in Streamlit's in-memory session state until the session ends.
- For a real deployment, add access control, retention policy, informed-consent workflow, audit logging, accessibility review, representative validation, and institutional privacy/security review.

## Known limitations

Haar detection works best with clear, frontal faces and stable lighting. It may miss profiles, occluded faces, or people far from the camera. Centroid association is not robust to dense crowds or major camera motion. Video codec support depends on the local OpenCV build. Audio activity measures signal properties, not meaning or speaker identity. A single camera/microphone cannot reliably establish an individual's attention or participation.

## Troubleshooting

- **Video cannot open:** Re-export as H.264 MP4 or AVI; confirm OpenCV can decode the source.
- **No faces detected:** Use brighter footage, a more frontal view, and a closer camera; test a lower sample interval.
- **Audio decode fails:** Convert the track to WAV (mono or stereo, 16-bit PCM) or install FFmpeg and retry.
- **Slow analysis:** Increase the video sample interval or use a shorter clip. CPU-only processing is expected.
- **PowerShell blocks activation:** Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current terminal, then activate the environment.

## Milestone mapping

| Milestone | Implementation |
|---|---|
| Requirements and environment | `requirements.txt`, local virtual-environment setup above |
| Vision and attendance | `attendance_agent/vision.py`, `attendance_agent/live_camera.py` |
| Audio analysis | `attendance_agent/audio.py` |
| Multimodal fusion and scoring | `attendance_agent/fusion.py` |
| Suggestions and analytics | `attendance_agent/analytics.py` |
| Interface and export | `app.py` |
