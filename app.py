"""Streamlit interface for the local Multimodal Attendance & Engagement Agent."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from attendance_agent.analytics import build_report, instructor_suggestions
from attendance_agent.audio import analyze_audio
from attendance_agent.fusion import fuse_signals
from attendance_agent.live_camera import LiveCameraSession
from attendance_agent.vision import analyze_image, analyze_video


st.set_page_config(page_title="Class Tracker | Classroom insights", page_icon="◉", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family:'Segoe UI',sans-serif; color:#e7edf7; }
.stApp { background:radial-gradient(ellipse at 82% 0%,#14213a 0%,#0b1220 42%); }
.block-container { max-width:1480px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Segoe UI',sans-serif; letter-spacing:-.035em; color:#f3f6fc; }
h1 { font-size:2rem; }
h2 { font-size:1.55rem; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#111c30 0%,#0e1728 100%); border-right:1px solid #26344b; }
section[data-testid="stSidebar"] > div { padding:1.35rem 1rem 1.5rem; }
.brand-lockup { display:flex; align-items:center; gap:.75rem; }
.brand-mark { display:grid; place-items:center; width:2.55rem; height:2.55rem; border:1px solid #526da3; border-radius:12px; background:linear-gradient(145deg,#283f6a,#172746); color:#c8d8ff; font-size:.82rem; font-weight:800; letter-spacing:-.04em; box-shadow:0 6px 20px #050a1470; }
.brand { color:#f1f5fc; font-size:1.04rem; font-weight:750; letter-spacing:-.035em; }
.brand-sub { color:#93a2b9; font-size:.72rem; margin-top:.12rem; }
.hero { position:relative; display:block; overflow:hidden; padding:1.45rem 1.7rem; border:1px solid #2a3a56; border-radius:18px; background:linear-gradient(115deg,#172640 0%,#16233a 58%,#1a2a46 100%); margin-bottom:1.35rem; box-shadow:0 14px 36px #050a142b; }
.hero:after { content:""; position:absolute; width:240px; height:240px; right:8%; top:-175px; border:1px solid #5576b533; border-radius:50%; box-shadow:0 0 0 28px #5576b30d,0 0 0 58px #5576b30a; pointer-events:none; }
.hero-copy { position:relative; z-index:1; }
.hero h1 { margin:.25rem 0 .1rem; font-size:2.15rem; letter-spacing:-.055em; }
.hero p { color:#a6b4c9; margin:.25rem 0 0; font-size:.9rem; }
.eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.66rem; color:#93b5ff; font-weight:750; }
.privacy-badge { position:relative; z-index:1; display:inline-flex; margin-top:.85rem; color:#c5d5ef; border:1px solid #415473; background:#101a2bbd; border-radius:999px; padding:.4rem .7rem; font-size:.7rem; font-weight:650; }
.section-label { color:#93a4bd; text-transform:uppercase; font-size:.65rem; letter-spacing:.13em; font-weight:750; }
.live-pill { display:inline-flex; align-items:center; color:#bcd2ff; background:#1b2c4a; border:1px solid #3b527a; padding:.3rem .65rem; border-radius:999px; font-size:.7rem; letter-spacing:.02em; font-weight:700; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#17243a,#131f33); border:1px solid #2a3a53; padding:.9rem 1rem; border-radius:14px; box-shadow:0 5px 16px #050a1417; }
div[data-testid="stMetricLabel"] { color:#a5b3c8; font-size:.78rem; }
div[data-testid="stMetricValue"] { color:#f2f6fc; font-weight:700; letter-spacing:-.035em; }
div[data-testid="stMetricDelta"] { font-size:.75rem; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { background:linear-gradient(145deg,#141f32,#121c2d); border-color:#293950; border-radius:15px; }
div[data-testid="stTabs"] { margin-top:.3rem; }
div[data-testid="stTabs"] button { color:#aab7ca; font-weight:600; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#c6d7ff; border-bottom-color:#82a8ff; }
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] { background-color:#82a8ff; }
div[data-testid="stDataFrame"] { border:1px solid #2b3a51; border-radius:12px; overflow:hidden; }
div[data-testid="stAlert"] { border-radius:12px; border:1px solid #34445c; }
div[data-testid="stCaptionContainer"] { color:#98a8bf; }
button[kind="primary"] { color:#101a2c; background:linear-gradient(110deg,#a9c5ff,#83a9ff); border-color:#8baeff; font-weight:700; }
button[kind="primary"]:hover { color:#0d1728; background:#c0d3ff; border-color:#c0d3ff; }
div[data-testid="stTabs"] button[kind="primary"] { color:#101a2c; background:linear-gradient(110deg,#a9c5ff,#83a9ff); border-color:#8baeff; font-weight:700; }
div[data-testid="stTabs"] button[kind="primary"] p { color:#101a2c; }
div[data-testid="stTabs"] button[kind="primary"]:hover { color:#0d1728; background:#c0d3ff; border-color:#c0d3ff; }
button[kind="secondary"] { border-color:#3b4d68; }
[data-testid="stFileUploader"] section { background:#121d30; border-color:#34455f; border-radius:12px; }
div[data-testid="stRadio"] label p { font-size:.86rem; }
@media (max-width:700px) {
    .block-container { padding:1.1rem 1rem 2rem; }
    .hero { padding:1.2rem; }
    .hero h1 { font-size:1.8rem; }
    .privacy-badge { align-self:flex-start; }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="hero-copy"><div class="eyebrow">CLASSROOM OPERATIONS</div><h1>Class Tracker</h1><p>Attendance insights and session analysis, in one workspace.</p></div><div class="privacy-badge">● &nbsp;Private by design · Local processing</div></div>', unsafe_allow_html=True)


def add_history_entry(entry: dict) -> None:
    history = list(st.session_state.get("session_history", []))
    history.append(entry)
    st.session_state["session_history"] = history[-20:]


def end_live_session() -> None:
    camera = st.session_state.get("live_camera")
    if camera is not None:
        camera.close()
        st.session_state["live_camera"] = None
    final = st.session_state.get("live_snapshot")
    if final and not st.session_state.get("live_history_saved"):
        add_history_entry({
            "session_name": f"Live camera · {datetime.now().strftime('%b %d, %H:%M')}",
            "source": "Live camera",
            "duration_seconds": final["elapsed_seconds"],
            "detected_tracks": final["detected_tracks"],
            "present_tracks": sum(person["present"] for person in final["participants"]),
            "average_score": final["average_score"],
            "timeline": final["timeline"],
        })
        st.session_state["live_history_saved"] = True


@st.fragment(run_every=0.2)
def render_live_camera() -> None:
    """Poll the local camera in short rerun cycles so Stop remains responsive."""
    camera = st.session_state.get("live_camera")
    if camera is None:
        return
    if st.button("Stop camera", key="stop_live_camera", type="secondary"):
        end_live_session()
        st.rerun()
    try:
        snapshot = camera.update()
    except Exception as exc:
        camera.close()
        st.session_state["live_camera"] = None
        st.error(f"Live camera stopped: {exc}")
        st.rerun()
        return
    st.session_state["live_snapshot"] = {key: value for key, value in snapshot.items() if key != "frame_rgb"}
    main, rail = st.columns([2.35, 1], gap="medium")
    with main:
        st.markdown('<span class="live-pill">● LIVE CAMERA</span>', unsafe_allow_html=True)
        st.image(snapshot["frame_rgb"], caption="Live preview · temporary anonymous track labels", use_container_width=True)
        participants = pd.DataFrame(snapshot["participants"])
        visible_tracks = int((participants["visibility_status"] == "Visible now").sum()) if not participants.empty else 0
        threshold_met = int(participants["present"].sum()) if not participants.empty else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Visible now", snapshot["visible_faces"])
        c2.metric("Attendance threshold met", threshold_met)
        current = snapshot["timeline"][-1]
        c3.metric("Face movement proxy", f"{current['motion_index']:.2f}")
        elapsed = int(snapshot["elapsed_seconds"])
        c4.metric("Session time", f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        st.caption(f"{visible_tracks} of {snapshot['detected_tracks']} temporary tracks currently visible.")
        st.info(
            "This face detector cannot determine phone use, studying, attention, or whether someone "
            "has actually left. “Out of view” means the face has not been detected; occlusion and "
            "camera framing can cause the same result."
        )
    with rail:
        with st.container(border=True):
            st.markdown('<div class="section-label">Cohort pulse</div>', unsafe_allow_html=True)
            st.metric("Presence & movement proxy", f"{snapshot['average_score']:.0f} / 100")
            st.caption("A descriptive group signal. It is not a measure of attention, emotion, or learning.")
        with st.container(border=True):
            st.markdown('<div class="section-label">Audio context</div>', unsafe_allow_html=True)
            st.markdown("**Microphone:** not connected")
            st.caption("Live mode currently analyzes camera frames only. Add a synchronized audio file in Recording analysis for room-level activity cues.")
        with st.container(border=True):
            st.markdown('<div class="section-label">Observed camera cues</div>', unsafe_allow_html=True)
            current = snapshot["timeline"][-1]
            if snapshot["visible_faces"] == 0:
                st.warning("No faces are visible right now. Check camera framing, distance, and lighting.")
            elif current["presence_ratio"] < 0.55:
                st.info("Face visibility has been intermittent in this interval. Use a wider or steadier camera view if appropriate.")
            elif current["motion_index"] > 0:
                st.info(
                    f"Face-position changes were detected in the sampled view "
                    f"(movement index {current['motion_index']:.2f})."
                )
            else:
                st.success("Faces are visible in this sample; little face-position change was measured.")
            st.caption("These are camera-position cues only, not a classification of student activity, attention, or learning.")
    live_data = pd.DataFrame(snapshot["timeline"])
    if not live_data.empty:
        with st.container(border=True):
            st.markdown("#### Interactive live analytics")
            chart_view = st.selectbox(
                "Chart view",
                ["Cohort signals", "Visible track count", "Track comparison"],
                key="live_chart_view",
            )
            if chart_view == "Cohort signals":
                trend_data = live_data.assign(
                    **{
                        "Visible presence (%)": live_data["presence_ratio"] * 100,
                        "Face movement proxy (%)": live_data["motion_index"] * 100,
                        "Group proxy": live_data["engagement_score"],
                    }
                ).melt(
                    id_vars=["time_label"],
                    value_vars=["Visible presence (%)", "Face movement proxy (%)", "Group proxy"],
                    var_name="Signal",
                    value_name="Percent",
                )
                chart = px.line(
                    trend_data,
                    x="time_label",
                    y="Percent",
                    color="Signal",
                    markers=True,
                    range_y=[0, 100],
                    labels={"time_label": "Session time", "Percent": "Signal (%)"},
                    hover_data={"Percent": ":.1f"},
                )
                chart.update_traces(marker_size=6)
            elif chart_view == "Visible track count":
                chart = px.area(
                    live_data,
                    x="time_label",
                    y="visible_faces_avg",
                    markers=True,
                    labels={"time_label": "Session time", "visible_faces_avg": "Average visible tracks"},
                )
                chart.update_traces(line_color="#8c7cf6", fillcolor="rgba(140,124,246,0.2)")
            else:
                if participants.empty:
                    st.caption("Track comparisons appear after a face has been detected.")
                    chart = None
                else:
                    track_data = participants.melt(
                        id_vars=["participant_id"],
                        value_vars=["presence_percent", "activity_percent"],
                        var_name="Measure",
                        value_name="Percent",
                    )
                    track_data["Measure"] = track_data["Measure"].map(
                        {"presence_percent": "Presence", "activity_percent": "Movement proxy"}
                    )
                    chart = px.bar(
                        track_data,
                        x="participant_id",
                        y="Percent",
                        color="Measure",
                        barmode="group",
                        range_y=[0, 100],
                        labels={"participant_id": "Temporary track ID", "Percent": "Percent (%)"},
                        hover_data={"Percent": ":.1f"},
                    )
            if chart is not None:
                chart.update_layout(
                    plot_bgcolor="#111a2d",
                    paper_bgcolor="#111a2d",
                    font_color="#dbe5f2",
                    margin=dict(l=10, r=10, t=18, b=8),
                    yaxis_gridcolor="#26354c",
                    xaxis_gridcolor="#202c41",
                )
                st.plotly_chart(
                    chart,
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True},
                    key="live_selected_chart",
                )

    with st.container(border=True):
        st.markdown("#### Attendance review")
        if not participants.empty:
            st.caption(
                "Temporary IDs are not identities. Presence (%) is observed face visibility; the movement "
                "proxy only describes face-box position changes. A face that is out of view may be occluded."
            )
            visibility_filter = st.selectbox(
                "Show tracks",
                ["All tracks", "Visible now", "Not currently visible"],
                key="live_visibility_filter",
            )
            filtered_participants = participants
            if visibility_filter == "Visible now":
                filtered_participants = participants[participants["visibility_status"] == "Visible now"]
            elif visibility_filter == "Not currently visible":
                filtered_participants = participants[participants["visibility_status"] != "Visible now"]
            display_participants = filtered_participants.rename(
                columns={
                    "participant_id": "Track ID",
                    "visibility_status": "Visibility",
                    "seconds_since_seen": "Seconds since seen",
                    "presence_seconds": "Presence (s)",
                    "presence_percent": "Presence (%)",
                    "activity_percent": "Movement proxy (%)",
                    "movement_summary": "Observed movement",
                    "present": "Attendance threshold met",
                }
            )
            st.dataframe(display_participants, use_container_width=True, hide_index=True)
            st.download_button("Download attendance detections (CSV)", participants.to_csv(index=False), file_name="live_attendance_detections.csv", mime="text/csv", key="live_csv")
        else:
            st.caption("No anonymous face tracks have been detected yet.")

with st.sidebar:
    st.markdown('<div class="brand-lockup"><div class="brand-mark">CT</div><div><div class="brand">Class Tracker</div><div class="brand-sub">Classroom workspace</div></div></div>', unsafe_allow_html=True)
    st.divider()
    if st.session_state.get("live_camera") is not None:
        st.markdown('<div class="live-pill">● SESSION ACTIVE</div>', unsafe_allow_html=True)
        active_snapshot = st.session_state.get("live_snapshot", {})
        st.caption(f"◷ {active_snapshot.get('elapsed_seconds', 0) / 60:.1f} min · camera {st.session_state['live_camera'].camera_index}")
        st.metric("Visible now", active_snapshot.get("visible_faces", 0))
        if st.button("■  End live session", key="sidebar_end_live", use_container_width=True):
            end_live_session()
            st.rerun()
    else:
        st.markdown('<div class="section-label">Session ready</div>', unsafe_allow_html=True)
        st.caption("Local processing · no cloud upload")
    st.divider()
    st.markdown('<div class="section-label">Session setup</div>', unsafe_allow_html=True)
    session_name = st.text_input("Session name", "Classroom session")
    st.caption("Video and audio stay in this local app process.")
    st.markdown('<div class="section-label">Presence settings</div>', unsafe_allow_html=True)
    min_presence = st.slider("Minimum presence for attendance (seconds)", 5, 300, 30, 5)
    sample_interval = st.slider("Video sample interval (seconds)", 0.5, 5.0, 2.0, 0.5)
    st.markdown('<div class="section-label">Scoring settings</div>', unsafe_allow_html=True)
    interval_seconds = st.slider("Scoring interval (seconds)", 10, 120, 30, 10)
    st.caption("Engagement is a transparent cohort-level proxy from visible presence, motion, and speech activity. It is not a measure of learning, emotion, or an individual student’s attention.")

tab_live, tab_teacher, tab_run, tab_results, tab_about = st.tabs(["Live monitor", "Class overview", "Analyze recording", "Session report", "How it works"])

with tab_run:
    st.subheader("Add session media")
    left, right = st.columns(2)
    with left:
        video_file = st.file_uploader("Classroom video", type=["mp4", "mov", "avi", "mkv", "webm"], help="A frontal classroom view works best. Supported codecs depend on your OpenCV build.")
    with right:
        audio_file = st.file_uploader("Optional audio track", type=["wav", "mp3", "m4a", "flac", "ogg"], help="Use audio from the same session, starting at the same time as the video.")
    st.info("Use this only with appropriate participant notice and consent. The system assigns temporary anonymous track IDs; it does not identify people by name or recognize faces.", icon="🔒")
    run = st.button("Analyze session", type="primary", disabled=video_file is None, use_container_width=True)
    if run and video_file:
        progress = st.progress(0, text="Preparing local analysis…")
        try:
            with tempfile.TemporaryDirectory(prefix="classroom_insight_") as temp_dir:
                vp = Path(temp_dir) / video_file.name
                vp.write_bytes(video_file.getvalue())
                progress.progress(10, text="Detecting and tracking visible faces…")
                vision = analyze_video(vp, sample_interval=sample_interval, progress_callback=lambda p: progress.progress(10 + int(p * 0.55), text="Detecting and tracking visible faces…"))
                audio = None
                if audio_file:
                    ap = Path(temp_dir) / audio_file.name
                    ap.write_bytes(audio_file.getvalue())
                    progress.progress(70, text="Analyzing optional audio…")
                    audio = analyze_audio(ap, interval_seconds=interval_seconds)
            progress.progress(85, text="Fusing signals and preparing insights…")
            fused = fuse_signals(vision["timeline"], audio, duration_seconds=vision["duration_seconds"], interval_seconds=interval_seconds)
            report = build_report(session_name, vision, fused, min_presence_seconds=min_presence, audio_available=audio is not None)
            report["suggestions"] = instructor_suggestions(fused)
            report["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            st.session_state["analysis"] = report
            report_scores = [row["engagement_score"] for row in report["timeline"]]
            add_history_entry({
                "session_name": report["session_name"],
                "source": "Video recording",
                "duration_seconds": report["duration_seconds"],
                "detected_tracks": report["detected_tracks"],
                "present_tracks": sum(person["present"] for person in report["participants"]),
                "average_score": sum(report_scores) / len(report_scores) if report_scores else 0.0,
                "timeline": report["timeline"],
                "suggestions": report["suggestions"],
            })
            progress.progress(100, text="Analysis complete")
            st.success("Session analysis is ready. Review attendance detections before using them as a record.")
        except Exception as exc:
            progress.empty()
            st.error(f"Analysis could not be completed: {exc}")
            st.exception(exc)

with tab_live:
    st.markdown("## Live classroom")
    st.caption("Live camera analysis · temporary presence tracking · group-level observable cues")
    camera_mode = st.radio(
        "Camera access",
        [
            "Continuous camera on this computer",
            "Capture a still image",
        ],
        horizontal=True,
        disabled=st.session_state.get("live_camera") is not None,
        help="Continuous mode uses the camera attached to the computer running Streamlit. Still-image capture uses the browser camera.",
    )
    if camera_mode == "Capture a still image":
        st.caption("Your browser will ask for camera permission. Each captured image is sent to this Streamlit server for face detection and is not saved.")
        captured_image = st.camera_input("Capture a classroom frame", key="browser_camera_capture")
        if captured_image is not None:
            try:
                camera_result = analyze_image(captured_image.getvalue())
            except (RuntimeError, ValueError) as exc:
                st.error(f"Camera image could not be analyzed: {exc}")
            else:
                preview, summary = st.columns([2.35, 1], gap="medium")
                with preview:
                    st.image(camera_result["frame_rgb"], caption="Captured frame · detected faces outlined", use_container_width=True)
                with summary:
                    with st.container(border=True):
                        st.markdown('<div class="section-label">Frame analysis</div>', unsafe_allow_html=True)
                        st.metric("Faces visible in this frame", camera_result["visible_faces"])
                        st.caption("A single image does not track attendance or identify people. Review face detections manually.")
    else:
        camera_index = st.number_input("System camera index", min_value=0, max_value=10, value=0, step=1, help="Most computers use 0 for the built-in or default camera. Try 1 for another camera.")
        active_camera = st.session_state.get("live_camera")
        start_live = st.button("▶  Start live session", type="primary", disabled=active_camera is not None, use_container_width=True)
        if start_live:
            try:
                st.session_state["live_camera"] = LiveCameraSession(camera_index=int(camera_index), interval_seconds=interval_seconds, min_presence_seconds=min_presence)
                st.session_state.pop("live_snapshot", None)
                st.session_state["live_history_saved"] = False
                active_camera = st.session_state["live_camera"]
            except Exception as exc:
                st.error(f"Could not start the camera: {exc}")
        if st.session_state.get("live_camera") is not None:
            render_live_camera()
        else:
            left, right = st.columns([2.35, 1], gap="medium")
            with left:
                with st.container(border=True):
                    st.markdown('<span class="live-pill">● CAMERA STANDBY</span>', unsafe_allow_html=True)
                    st.markdown("### Start a local classroom session")
                    st.write("Select **Start live session** to open the system camera attached to the computer running Streamlit. Live footage is processed in memory and is not saved as a video.")
                    st.caption("If camera 0 does not open, check operating-system camera permissions or try index 1.")
            with right:
                with st.container(border=True):
                    st.markdown('<div class="section-label">Session signals</div>', unsafe_allow_html=True)
                    st.metric("Active tracks", 0)
                    st.metric("Audio input", "Not connected")
                    st.caption("Live audio is not captured. Synchronized audio analysis is available for uploaded recordings.")
            last_live = st.session_state.get("live_snapshot")
            if last_live:
                st.success(f"Last live session · {last_live['video_samples']} samples · {last_live['elapsed_seconds']:.1f} seconds.")
                st.download_button("Download last live report (JSON)", json.dumps(last_live, indent=2), file_name="live_session_report.json", mime="application/json", key="live_json_stopped")

with tab_teacher:
    st.markdown("## Teacher Central")
    st.caption("Session reflection and historical patterns · descriptive cohort data for instructor review")
    history = st.session_state.get("session_history", [])
    if history:
        average_signal = sum(item.get("average_score", 0) for item in history) / len(history)
        sessions_with_presence = sum(item.get("present_tracks", 0) for item in history)
        m1, m2, m3 = st.columns(3)
        m1.metric("Sessions analyzed", len(history))
        m2.metric("Average cohort proxy", f"{average_signal:.0f} / 100")
        m3.metric("Attendance threshold detections", sessions_with_presence)

        chart_rows = [{"Session": item["session_name"], "Group proxy": round(item.get("average_score", 0), 1)} for item in history[-8:]]
        col_chart, col_notes = st.columns([1.4, 1], gap="medium")
        with col_chart:
            with st.container(border=True):
                st.markdown("#### Class history comparison")
                history_chart = px.bar(pd.DataFrame(chart_rows), x="Session", y="Group proxy", range_y=[0, 100], labels={"Group proxy": "Average group signal"})
                history_chart.update_traces(marker_color="#8c7cf6", marker_line_width=0)
                history_chart.update_layout(plot_bgcolor="#111a2d", paper_bgcolor="#111a2d", font_color="#dbe5f2", margin=dict(l=8, r=8, t=12, b=8), yaxis_gridcolor="#26354c", xaxis_gridcolor="#202c41")
                st.plotly_chart(history_chart, use_container_width=True)
        with col_notes:
            with st.container(border=True):
                st.markdown("#### Reflection prompts")
                latest = history[-1]
                prompts = latest.get("suggestions", [])
                if prompts:
                    for prompt in prompts[:3]: st.markdown(f"- {prompt}")
                else:
                    st.write("Review camera coverage and learner feedback alongside the group timeline.")
                st.caption("These are review prompts from session-level patterns, not a rating of teaching quality.")
        with st.container(border=True):
            st.markdown("#### Recent sessions")
            history_frame = pd.DataFrame([{key: item.get(key) for key in ["session_name", "source", "duration_seconds", "detected_tracks", "present_tracks", "average_score"]} for item in reversed(history[-10:])])
            history_frame = history_frame.rename(columns={"session_name": "Session", "source": "Input", "duration_seconds": "Duration (s)", "detected_tracks": "Anonymous tracks", "present_tracks": "Threshold met", "average_score": "Group proxy"})
            st.dataframe(history_frame, use_container_width=True, hide_index=True)
    else:
        with st.container(border=True):
            st.markdown("### Your teaching reflection starts with a session")
            st.write("Run a live camera session or analyze a classroom recording. Class Tracker summarizes cohort-level patterns and keeps a short in-memory session history for this app session.")
            st.caption("No student ranking, identity recognition, or facial-emotion analysis is used.")

with tab_results:
    report = st.session_state.get("analysis")
    if not report:
        st.markdown("### Your session report will appear here")
        st.write("Add a video in **Analyze session** to generate attendance detections, an engagement timeline, audio context, and practical teaching suggestions.")
    else:
        people = report["participants"]
        timeline = pd.DataFrame(report["timeline"])
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Detected tracks", report["detected_tracks"])
        c2.metric("Attendance threshold met", sum(p["present"] for p in people))
        c3.metric("Session duration", f"{report['duration_seconds']/60:.1f} min")
        avg = timeline["engagement_score"].mean() if not timeline.empty else 0
        c4.metric("Average cohort signal", f"{avg:.0f}/100", help="A heuristic proxy, not an assessment of learning or individual attention.")
        st.subheader(report["session_name"])
        st.caption(f"Generated {report['generated_at']} · {report['video_samples']} sampled frames · face detection can miss or merge participants; review manually.")
        if not timeline.empty:
            trend_data = timeline.assign(
                **{
                    "Visible presence (%)": timeline["presence_ratio"] * 100,
                    "Face movement proxy (%)": timeline["motion_index"] * 100,
                    "Speech activity proxy (%)": timeline["speech_activity"] * 100,
                    "Group proxy": timeline["engagement_score"],
                }
            ).melt(
                id_vars=["time_label"],
                value_vars=["Visible presence (%)", "Face movement proxy (%)", "Speech activity proxy (%)", "Group proxy"],
                var_name="Signal",
                value_name="Percent",
            )
            trend_chart = px.line(
                trend_data,
                x="time_label",
                y="Percent",
                color="Signal",
                markers=True,
                range_y=[0, 100],
                title="Session signals over time",
                labels={"time_label": "Session time", "Percent": "Signal (%)"},
                hover_data={"Percent": ":.1f"},
            )
            trend_chart.update_traces(marker_size=6)
            trend_chart.update_layout(plot_bgcolor="#111a2d", paper_bgcolor="#111a2d", font_color="#dbe5f2", margin=dict(l=10,r=10,t=50,b=10), yaxis_gridcolor="#26354c", xaxis_gridcolor="#202c41")
            st.plotly_chart(trend_chart, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True}, key="report_session_signals")
            visible_chart = px.line(
                timeline,
                x="time_label",
                y="visible_faces_avg",
                markers=True,
                title="Average visible face tracks",
                labels={"time_label": "Session time", "visible_faces_avg": "Average visible tracks"},
            )
            visible_chart.update_traces(line_color="#8c7cf6", marker_size=6)
            visible_chart.update_layout(plot_bgcolor="#111a2d", paper_bgcolor="#111a2d", font_color="#dbe5f2", margin=dict(l=10,r=10,t=50,b=10), yaxis_gridcolor="#26354c", xaxis_gridcolor="#202c41")
            st.plotly_chart(visible_chart, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True}, key="report_visible_tracks")
            st.caption("These are descriptive camera/audio proxies, not measurements of individual attention or learning.")
        if people:
            participant_signals = pd.DataFrame(people).melt(
                id_vars=["participant_id"],
                value_vars=["presence_percent", "activity_percent"],
                var_name="Measure",
                value_name="Percent",
            )
            participant_signals["Measure"] = participant_signals["Measure"].map(
                {"presence_percent": "Presence", "activity_percent": "Movement proxy"}
            )
            participant_chart = px.bar(
                participant_signals,
                x="participant_id",
                y="Percent",
                color="Measure",
                barmode="group",
                range_y=[0, 100],
                title="Track-level presence and movement proxy",
                labels={"participant_id": "Temporary track ID", "Percent": "Percent (%)"},
                hover_data={"Percent": ":.1f"},
            )
            participant_chart.update_layout(plot_bgcolor="#111a2d", paper_bgcolor="#111a2d", font_color="#dbe5f2", margin=dict(l=10,r=10,t=50,b=10), yaxis_gridcolor="#26354c", xaxis_gridcolor="#202c41")
            st.plotly_chart(participant_chart, use_container_width=True, config={"displayModeBar": True}, key="report_participant_signals")
        col_a,col_b = st.columns([1.2,1])
        with col_a:
            st.subheader("Anonymous attendance detections")
            if people:
                attendance = pd.DataFrame(people)
                attendance["presence_min"] = (attendance["presence_seconds"] / 60).round(1)
                attendance["attendance"] = attendance["present"].map({True:"Threshold met",False:"Below threshold"})
                st.dataframe(attendance[["participant_id","visibility_status","presence_min","presence_percent","activity_percent","attendance","first_seen_seconds","last_seen_seconds"]].rename(columns={"participant_id":"Track","visibility_status":"Visibility at end","presence_min":"Presence (min)","presence_percent":"Presence (%)","activity_percent":"Movement proxy (%)","attendance":"Attendance status","first_seen_seconds":"First seen (s)","last_seen_seconds":"Last seen (s)"}), use_container_width=True, hide_index=True)
            else:
                st.warning("No face tracks detected. Try a brighter video with faces more clearly visible.")
        with col_b:
            st.subheader("Instructor suggestions")
            for item in report["suggestions"]:
                st.markdown(f"- {item}")
            if not report["suggestions"]: st.write("No strong pattern crossed the suggestion thresholds.")
            st.subheader("Audio context")
            st.write(report["audio_summary"])
        st.subheader("Session data")
        st.download_button("Download session report (JSON)", json.dumps(report, indent=2), file_name="classroom_insight_report.json", mime="application/json")
        if people:
            st.download_button("Download attendance detections (CSV)", pd.DataFrame(people).to_csv(index=False), file_name="attendance_detections.csv", mime="text/csv")

with tab_about:
    st.subheader("A transparent, local analysis pipeline")
    st.markdown("""
    1. **Vision:** OpenCV Haar cascade detects frontal faces at a configurable sampling rate. A centroid tracker associates nearby detections into temporary track IDs.
    2. **Presence:** Each track accumulates observed sample intervals. Attendance is marked only after the configured threshold; gaps may occur when a face is occluded or turned away.
    3. **Audio (optional):** Librosa estimates RMS energy, zero-crossing rate, and a speech-activity proxy in aligned time windows. It does not transcribe or identify speakers.
    4. **Fusion:** Time windows combine group face visibility, coarse motion, and speech proxy into a rule-based 0–100 cohort indicator.
    5. **Live workspace:** The camera panel updates temporary anonymous tracks and a cohort coverage timeline while the local camera is active. It does not save live video or capture live audio.
    6. **Analytics:** Pattern thresholds generate teaching prompts and a downloadable JSON/CSV summary; Teacher Central compares a short in-memory session history.

    **Known limits:** This is a project prototype, not a validated attendance system. Face detections are not identity verification. The system does not estimate facial emotion. Track IDs can split or merge, and speech activity cannot distinguish a teacher from students. Lighting, camera angle, occlusion, room acoustics, and codecs affect output. Review every attendance result and do not use the engagement proxy for grading, discipline, or high-stakes decisions.
    """)
