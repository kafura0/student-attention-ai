'''
Upload a .mp4 or .avi video

Run frame-by-frame attention analysis

Display results + average score

'''

import streamlit as st
import tempfile
import cv2
from utils.attention_detector import AttentionDetector

st.set_page_config(page_title="Student Attention AI", layout="centered")
st.title("🎯 Student Attention Detection")

uploaded_file = st.file_uploader("Upload a classroom video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save to temp location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    detector = AttentionDetector()

    scores = []
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        score = detector.process_frame(frame)
        if score is not None:
            scores.append(score)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Attention Score: {score:.2f}", channels="RGB")

    cap.release()

    if scores:
        st.success("✅ Analysis Complete")
        st.metric("Average Attention Score", round(sum(scores) / len(scores), 2))
    else:
        st.warning("No face detected in the video.")
