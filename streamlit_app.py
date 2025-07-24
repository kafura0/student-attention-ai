'''
 Features:
Upload a video for analysis.

Shows real-time frames with attention score.

Summarizes average attention at the end.

Supports .mp4, .avi, .mov, .mkv.

'''

# streamlit_app.py

import streamlit as st
import tempfile
from utils.attention_detector import AttentionDetector

st.set_page_config(page_title="Student Attention Detector", layout="centered")

st.title("🎯 Student Attention Detector")
st.markdown("Upload a recorded classroom video to assess student attention levels.")

uploaded_file = st.file_uploader("Upload a video file (e.g. .mp4, .avi)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.video(temp_path)

    if st.button("🔍 Analyze Attention"):
        detector = AttentionDetector()
        score = detector.process_video(temp_path)
        st.success(f"✅ Attention Score: **{score}%**")

        if score > 70:
            st.info("👏 Good attention detected.")
        elif score > 40:
            st.warning("⚠️ Moderate attention. Could be improved.")
        else:
            st.error("❌ Low attention detected.")
