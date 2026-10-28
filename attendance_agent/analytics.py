"""Interpretable session summaries and non-prescriptive teaching prompts."""
from __future__ import annotations


def build_report(session_name: str, vision: dict, fused, min_presence_seconds: int = 30, audio_available: bool = False) -> dict:
    people=[]
    for p in vision["participants"]:
        q=dict(p)
        duration=max(float(vision["duration_seconds"]),1e-6)
        q["presence_percent"]=round(min(float(q["presence_seconds"])/duration*100,100),1)
        q["activity_percent"]=round(min(max(float(q.get("motion_index",0))*100,0),100),1)
        seconds_since_seen=max(float(vision["duration_seconds"])-float(q["last_seen_seconds"]),0)
        q["seconds_since_seen"]=round(seconds_since_seen,1)
        q["visibility_status"]=(
            "Visible near end"
            if seconds_since_seen <= float(vision.get("sample_interval_seconds",1))*1.5
            else "Not seen at end"
        )
        q["present"]=q["presence_seconds"]>=min_presence_seconds
        people.append(q)
    fused=fused.fillna(0)
    audio_text=(f"Audio analyzed. Mean speech-activity proxy {fused.speech_activity.mean():.2f}; mean relative RMS noise indicator {fused.noise_level.mean():.1f}/100. These are room-level cues and cannot identify who is speaking." if audio_available else "No usable audio signal was included; visual presence and motion cues only.")
    return {"session_name":session_name,"duration_seconds":vision["duration_seconds"],"video_samples":vision["video_samples"],"detected_tracks":len(people),"attendance_threshold_seconds":min_presence_seconds,"participants":people,"timeline":fused.to_dict(orient="records"),"audio_summary":audio_text,"method":"OpenCV Haar face detections + centroid association; heuristic cohort score"}


def instructor_suggestions(timeline) -> list[str]:
    if timeline.empty: return ["No timeline data was available. Check the video and try a shorter sampling interval."]
    tips=[]
    low=timeline[timeline.engagement_score<35]
    if len(low)>=2: tips.append("Several intervals had a low cohort signal. Consider a short check-in, worked example, or change of activity, then review whether participation shifts.")
    high=timeline[timeline.engagement_score>=70]
    if len(high): tips.append(f"Higher cohort signals appeared around {', '.join(high.time_label.head(3))}. Review what was happening in those segments and consider reusing that format.")
    if (timeline.noise_level>65).sum()>=2: tips.append("The audio RMS indicator was elevated in multiple intervals. Check room acoustics and ask learners whether the audio conditions are comfortable.")
    if not tips: tips.append("No sustained signal pattern crossed the configured thresholds. Use learner feedback and your own observations alongside this report.")
    return tips
