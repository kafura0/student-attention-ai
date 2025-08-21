import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go

from attention_detector import AttentionDetector

# ---------- Page / Layout ----------
st.set_page_config(
    page_title="📚 Multi-Student Attention & Cheating Detector",
    page_icon="🎥",
    layout="wide",
)

# Minimal, mobile-friendly CSS
st.markdown(
    """
    <style>
      /* tighter top padding on mobile */
      .block-container { padding-top: 1rem; }
      /* make sidebar controls compact */
      section[data-testid="stSidebar"] .stButton>button { width: 100%; }
      /* video element should keep aspect and be responsive */
      img, video { max-height: 70vh; object-fit: contain; }
      /* data card look */
      .metric-card {
        padding: 0.75rem 1rem; border-radius: 14px; border: 1px solid #eaeaea;
        background: #fff;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar Controls ----------
st.sidebar.header("🎛️ Session & Video Settings")

# Video source (keep webcam first for local use)
source = st.sidebar.radio("Video Source", ["📷 Webcam", "📂 Upload Video"], index=0, key="src_radio")

frame_skip = st.sidebar.slider("Frame Skip (process every Nth frame)", 1, 10, 2, key="frame_skip")
target_fps = st.sidebar.slider("FPS Cap", 5, 30, 15, key="fps_cap")
max_faces = st.sidebar.slider("Max Faces", 1, 15, 8, key="max_faces")
min_motion_area = st.sidebar.slider("Min Motion Area (px)", 100, 6000, 800, step=50, key="min_motion")
draw_landmarks = st.sidebar.checkbox("Draw Landmarks", value=True, key="draw_lm")
refine_iris = st.sidebar.checkbox("Refine Iris (slower, better gaze)", value=False, key="refine_iris")
ema_alpha = st.sidebar.slider("Smoothing (EMA α)", 0.05, 0.8, 0.3, 0.05, key="ema_alpha")

st.sidebar.markdown("---")

# Start/Stop with unique keys (avoid duplicates)
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    start_clicked = st.button("▶️ Start", key="btn_start_main")
with col_btn2:
    stop_clicked = st.button("🛑 Stop", key="btn_stop_main")

# ---------- App State ----------
if "run" not in st.session_state:
    st.session_state.run = False
if "log" not in st.session_state:
    st.session_state.log = []       # list of rows (timestamp, attentive, total, cheating, face_id, eyes_closed, gaze, pitch, yaw, roll, motion)
if "ema_attentive" not in st.session_state:
    st.session_state.ema_attentive = None

# Handle start/stop
if start_clicked:
    st.session_state.run = True
    st.session_state.log = []
    st.session_state.ema_attentive = None

if stop_clicked:
    st.session_state.run = False

# ---------- Title / Intro ----------
st.title("📚 Multi-Student Attention & Cheating Detector")
st.markdown(
    """
    Real-time multi-face detection with **eye closure**, **head pose**, **motion**, and **gaze** analysis.
    Designed for class monitoring and exam proctoring.  
    - ✅ **Responsive video** (auto resizes)  
    - 📈 **Live analytics** with interactive Plotly charts  
    - 💾 **CSV log download**  
    """
)

# ---------- Initialize Detector ----------
# Keep a single instance for performance
@st.cache_resource(show_spinner=False)
def make_detector(_min_motion_area, _draw_landmarks, _refine_iris, _max_faces):
    return AttentionDetector(
        min_motion_area=_min_motion_area,
        draw_landmarks=_draw_landmarks,
        refine_iris=_refine_iris,
        max_faces=_max_faces,
        ema_alpha=ema_alpha,
    )

detector = make_detector(min_motion_area, draw_landmarks, refine_iris, max_faces)

# ---------- Layout ----------
left, right = st.columns([1.3, 1])  # video bigger than metrics

# Video container + live overlays
with left:
    st.subheader("🎥 Live Feed")
    video_placeholder = st.empty()
    # quick badges area
    badge = st.empty()

# Live metrics
with right:
    st.subheader("📊 Live Metrics")

    # placeholders for live KPIs and small charts
    kpi_cols = st.columns(3)
    kpi_attentive = kpi_cols[0].empty()
    kpi_total = kpi_cols[1].empty()
    kpi_cheat = kpi_cols[2].empty()

    # sparkline container
    sparkline = st.empty()

# ---------- Video Input Setup ----------
cap = None
temp_video_path = None

if st.session_state.run:
    if source == "📷 Webcam":
        # 0 is the default cam; CAP_DSHOW helps on Windows; ignored on Linux/Cloud
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    else:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="uploader")
        if uploaded is not None:
            temp_video_path = f"tmp_{int(time.time())}.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded.read())
            cap = cv2.VideoCapture(temp_video_path)

# ---------- Main Loop ----------
last_time = time.time()
frame_counter = 0

while st.session_state.run and cap is not None and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % frame_skip != 0:
        # show last frame to keep UI responsive
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        continue

    # Resize to a friendly size for speed and responsiveness
    frame = cv2.resize(frame, (960, 540))  # 16:9, scales well on mobile/desktop

    # Run detection
    results = detector.detect(frame)

    annotated = results["annotated_frame"]
    attentive_count = results["attentive_count"]
    total_faces = results["total_faces"]
    cheating_flags = results.get("cheating_flags", 0)
    faces = results.get("faces", [])
    metrics = results.get("metrics", {})  # optional raw values

    # EMA smoothing of attentive count (display only)
    if st.session_state.ema_attentive is None:
        st.session_state.ema_attentive = float(attentive_count)
    else:
        st.session_state.ema_attentive = (
            ema_alpha * attentive_count + (1 - ema_alpha) * st.session_state.ema_attentive
        )

    # --------- UI Update ----------
    # Badge color based on attention ratio & cheating
    attention_ratio = (attentive_count / max(total_faces, 1)) if total_faces else 0.0
    if cheating_flags > 0:
        badge.markdown(
            '<div class="metric-card" style="border-color:#ffd0d0;background:#fff5f5;">'
            f'🚨 <b>Cheating flags:</b> {cheating_flags}'
            "</div>",
            unsafe_allow_html=True,
        )
    elif attention_ratio >= 0.7:
        badge.markdown(
            '<div class="metric-card" style="border-color:#d9f7be;background:#f6ffed;">'
            '✅ <b>Class is attentive</b>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        badge.markdown(
            '<div class="metric-card" style="border-color:#ffe58f;background:#fffbe6;">'
            '👀 <b>Attention is low</b>'
            "</div>",
            unsafe_allow_html=True,
        )

    # Show annotated frame (auto-resize)
    video_placeholder.image(annotated, channels="BGR", use_container_width=True)

    # Live KPIs
    kpi_attentive.metric("Attentive", f"{int(round(st.session_state.ema_attentive))}/{total_faces}")
    kpi_total.metric("Faces", f"{total_faces}")
    kpi_cheat.metric("Cheating Flags", f"{cheating_flags}")

    # Sparkline (last ~120 points)
    now_str = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.append({
        "timestamp": now_str,
        "attentive": attentive_count,
        "faces": total_faces,
        "cheating": cheating_flags,
        # flatten per-face summary (first few faces to keep row size small)
        **({f"f{i}_eyes": f["eyes_closed"] for i, f in enumerate(faces[:4])}),
        **({f"f{i}_gaze": f["gaze_dir"] for i, f in enumerate(faces[:4])}),
    })

    short_df = pd.DataFrame(st.session_state.log[-120:])  # last N for the sparkline
    if not short_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=short_df["timestamp"], y=short_df["attentive"],
            mode="lines", name="Attentive"
        ))
        fig.add_trace(go.Scatter(
            x=short_df["timestamp"], y=short_df["faces"],
            mode="lines", name="Faces"
        ))
        fig.update_layout(
            height=180, margin=dict(l=10, r=10, t=0, b=0),
            legend=dict(orientation="h", y=1.2)
        )
        sparkline.plotly_chart(fig, use_container_width=True)

    # FPS cap
    elapsed = time.time() - last_time
    min_frame_time = 1.0 / max(target_fps, 1)
    if elapsed < min_frame_time:
        time.sleep(min_frame_time - elapsed)
    last_time = time.time()

# Release camera if used
if cap is not None:
    cap.release()

st.markdown("---")

# ---------- Attention Logs & Analytics (below live section) ----------
st.subheader("🧾 Attention Logs & Analytics")

if len(st.session_state.log) == 0:
    st.info("Start a session to populate logs.")
else:
    df = pd.DataFrame(st.session_state.log)

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Attention % over time
        df2 = df.copy()
        # compute % when faces>0 else 0
        df2["attention_pct"] = np.where(df2["faces"] > 0, df2["attentive"] / df2["faces"] * 100, 0)
        fig_pct = px.line(df2, x="timestamp", y="attention_pct", title="Attention % Over Time")
        fig_pct.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pct, use_container_width=True)

    with chart_col2:
        # Attentive vs Not Attentive (latest snapshot)
        latest = df.iloc[-1]
        att = int(latest["attentive"])
        faces = int(latest["faces"])
        not_att = max(faces - att, 0)
        fig_pie = px.pie(
            names=["Attentive", "Not Attentive"],
            values=[att, not_att],
            title="Current Class Snapshot"
        )
        fig_pie.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Full log table
    st.markdown("#### Full Session Log")
    st.dataframe(df, use_container_width=True)

    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name=f"attention_log_{int(time.time())}.csv",
        mime="text/csv",
        key="dl_csv_btn",
    )

# ---------- Helpful Tips ----------
with st.expander("💡 Performance Tips"):
    st.markdown(
        """
        - Use a **lower FPS cap** and **higher frame skip** on slower machines.
        - Disable **Refine Iris** unless precise gaze is critical.
        - Reduce **Max Faces** in large rooms for speed.
        - On Streamlit Cloud, browser webcams generally require a WebRTC component; for local desktop use, OpenCV is fine.
        """
    )
