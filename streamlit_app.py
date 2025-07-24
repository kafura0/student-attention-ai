import cv2
import streamlit as st
import numpy as np
from datetime import datetime
from utils.attention_detector import AttentionDetector
import tempfile
import pandas as pd
import altair as alt

# Set Streamlit page config
st.set_page_config(page_title="Student Attention AI", layout="wide")

st.title("🎯 Student Attention Detector")
st.markdown("Monitor student attentiveness using AI-powered facial landmark detection.")

# Initialize session data
if 'session_data' not in st.session_state:
    st.session_state['session_data'] = []

# Sidebar controls
st.sidebar.header("📷 Input Source")
source = st.sidebar.radio("Select input:", ("Webcam", "Upload Video"))

start_session = st.sidebar.button("▶️ Start Session", key="start_button")
stop_session = st.sidebar.button("🛑 Stop Session", key="stop_button_1")
running = start_session and not stop_session

# Detector
detector = AttentionDetector()

frame_placeholder = st.empty()
status_text = st.empty()

# Capture video input
if source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file and start_session:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
elif source == "Webcam" and start_session:
    cap = cv2.VideoCapture(0)

# Process video stream
if running and 'cap' in locals():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        resized_frame = cv2.resize(frame, (640, 480))

        is_attentive, annotated_frame = detector.detect(resized_frame)

        # Record session data
        st.session_state['session_data'].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attentive": is_attentive
        })

        label = "✅ Attentive" if is_attentive else "⚠️ Not Attentive"
        color = (0, 255, 0) if is_attentive else (0, 0, 255)
        cv2.putText(annotated_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame_placeholder.image(annotated_frame, channels="BGR")
        status_text.markdown(f"**Status:** {label}")

        if stop_session:
            break

    cap.release()
    cv2.destroyAllWindows()

# Sidebar: CSV Export
st.sidebar.subheader("📄 Session Report")
if st.sidebar.button("📥 Export CSV"):
    df = pd.DataFrame(st.session_state['session_data'])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv, file_name="attention_log.csv", mime="text/csv")

# ============================
# 📊 Display Attention Charts
# ============================
st.subheader("📊 Attention Summary Charts")

if st.session_state['session_data']:
    df = pd.DataFrame(st.session_state['session_data'])

    # --- Pie chart ---
    attentive_count = df['attentive'].sum()
    inattentive_count = len(df) - attentive_count
    pie_df = pd.DataFrame({
        'Status': ['Attentive', 'Not Attentive'],
        'Count': [attentive_count, inattentive_count]
    })

    pie_chart = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Status", type="nominal"),
        tooltip=["Status", "Count"]
    ).properties(title="Attention Breakdown")

    # --- Timeline chart ---
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['status'] = df['attentive'].apply(lambda x: "Attentive" if x else "Not Attentive")

    time_chart = alt.Chart(df).mark_line(point=True).encode(
        x="timestamp:T",
        y=alt.Y('attentive:Q', title="Attention (1=Yes, 0=No)"),
        color=alt.Color("status:N"),
        tooltip=["timestamp", "status"]
    ).properties(title="Attention Over Time").interactive()

    # Display charts side by side
    col1, col2 = st.columns(2)
    col1.altair_chart(pie_chart, use_container_width=True)
    col2.altair_chart(time_chart, use_container_width=True)
else:
    st.info("Session data will appear here once the session starts.")
