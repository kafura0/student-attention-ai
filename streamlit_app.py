import streamlit as st
import cv2
import tempfile
from datetime import datetime
import numpy as np
from PIL import Image

# ---------------------------
# STREAMLIT CONFIG
# ---------------------------
st.set_page_config(
    page_title="Student Attention AI",
    page_icon="🎯",
    layout="centered"
)

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

# ---------------------------
# TITLE & DESCRIPTION
# ---------------------------
st.title("🎯 Student Attention AI")
st.markdown("Track student focus levels in **real-time** or from uploaded videos.")

# ---------------------------
# SIDEBAR SETTINGS
# ---------------------------
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Select Mode", ["Webcam", "Upload Video"])
skip_frames = st.sidebar.slider("Frame Skip (Performance)", min_value=1, max_value=10, value=2)
display_size = st.sidebar.selectbox("Display Size", ["640x480", "800x600", "320x240"])

width, height = map(int, display_size.split("x"))

# ---------------------------
# ATTENTION DETECTION (Placeholder Logic)
# Replace with your AI model inference
# ---------------------------
def detect_attention(frame):
    """
    Dummy attention detection function.
    Replace with actual model inference.
    """
    # Simulate attention score
    attention_score = np.random.randint(50, 101)
    status = "Attentive" if attention_score > 70 else "Distracted"
    return status, attention_score

# ---------------------------
# MAIN LOGIC
# ---------------------------
logs = []

if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file.name)
    else:
        cap = None
else:
    cap = cv2.VideoCapture(0)

if cap and cap.isOpened():
    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Frame skipping
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % skip_frames != 0:
            continue

        # Resize for performance
        frame = cv2.resize(frame, (width, height))

        # Attention detection
        status, score = detect_attention(frame)

        # Log results
        logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "status": status,
            "score": score
        })

        # Overlay info on frame
        cv2.putText(frame, f"{status} ({score}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if status == "Attentive" else (0, 0, 255), 2)

        # Convert for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

# ---------------------------
# ATTENTION LOGS BELOW VIDEO
# ---------------------------
if logs:
    st.markdown("## 📜 Attention Logs")
    st.dataframe(logs, use_container_width=True)
