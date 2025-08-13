import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from attention_detector import AttentionDetector

# --------------------------
# App Setup
# --------------------------
st.set_page_config(page_title="📚 Attention & Cheating Detector", layout="centered")
# ---------------------------
# CUSTOM STYLING
# ---------------------------
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 900px;
        margin: auto;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    /* Mobile-friendly adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        img {
            max-width: 100% !important;
            height: auto !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Multi-Student Attention & Cheating Detector")

st.markdown("""
This AI-powered tool uses your webcam to detect attention in real-time.  
It tracks **eye closure** and **head pose** for each student (multi-face supported).  
Useful for monitoring engagement during online exams or virtual classes.
            
## 🚀 Features

- ✅ Multi-Student Detection (10+ faces)
- 👁️ Eye Closure Detection (sleepiness / inattentiveness)
- 🧠 Head Pose Estimation (look direction)
- 👀 Gaze Tracking (left/right/off-screen)
- 🌀 Motion Detection (fidgeting / distraction)
- ⚠️ Cheating Flagging based on combined metrics
- 🧾 Real-Time Feedback within frame (Attentive, Drowsy, Looking Away, etc.)
- 📈 Attention Logging & Plotly Charts
- 📦 CSV Export of Session Logs
- 🎛️ Adjustable Frame Skip, Min Motion Area, and Source (webcam/video)
""")

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.header("🎛️ Controls")
source = st.sidebar.radio("Video Source", ["📷 Webcam", "📂 Upload Video"])
frame_skip = st.sidebar.slider("Process every Nth frame", 1, 10, 2)
draw_landmarks = st.sidebar.checkbox("Draw Facial Landmarks", value=True)

# Buttons
if "run_session" not in st.session_state:
    st.session_state.run_session = False
if "session_log" not in st.session_state:
    st.session_state.session_log = []

if st.sidebar.button("▶️ Start Session", key="start_session"):
    st.session_state.run_session = True
    st.session_state.session_log = []
    st.toast("Session started")

if st.sidebar.button("🛑 Stop Session", key="stop_session"):
    st.session_state.run_session = False
    st.toast("Session stopped")

# --------------------------
# Main Logic
# --------------------------
video_col, charts_col = st.columns([2, 1])

with video_col:
    st.markdown("### 📹 Live Detection")
    frame_display = st.empty()

    detector = AttentionDetector(draw_landmarks=draw_landmarks)

    if st.session_state.run_session:
        if source == "📷 Webcam":
            cap = cv2.VideoCapture(0)
        else:
            uploaded = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])
            if uploaded:
                path = f"temp_{int(time.time())}.mp4"
                with open(path, "wb") as f:
                    f.write(uploaded.read())
                cap = cv2.VideoCapture(path)
            else:
                st.warning("Upload a video file.")
                st.stop()

        frame_count = 0
        while st.session_state.run_session and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            results = detector.detect(frame)

            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.session_log.append({
                "timestamp": timestamp,
                "attentive": results["attentive_count"],
                "total_faces": results["total_faces"],
                "cheating_flags": len(results["cheating_flags"]),
            })

            frame_display.image(results["annotated_frame"], channels="BGR", use_container_width=True)

        cap.release()
        st.session_state.run_session = False

# --------------------------
# Charts & Analytics
# --------------------------
with charts_col:
    st.markdown("### 📊 Attention & Cheating Logs")
    if st.session_state.session_log:
        df = pd.DataFrame(st.session_state.session_log)

        fig1 = px.line(df, x="timestamp", y="attentive", title="Attentive Students Over Time")
        fig2 = px.line(df, x="timestamp", y="total_faces", title="Total Students Detected")
        fig3 = px.bar(df, x="timestamp", y="cheating_flags", title="Cheating Events Detected")

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)

        # Pie chart summary
        latest = df.iloc[-1]
        attentive = latest.attentive
        inattentive = latest.total_faces - attentive
        pie = go.Figure(data=[
            go.Pie(labels=["Attentive", "Inattentive"], values=[attentive, inattentive], hole=0.5)
        ])
        pie.update_layout(title="Current Attention Split")
        st.plotly_chart(pie, use_container_width=True)

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV Log", csv, "attention_log.csv", mime="text/csv")
