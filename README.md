# 🎓 Student Attention & Cheating Detection AI

A real-time multi-student detection system using webcam or video input to track attention levels, detect cheating cues, and log engagement analytics using computer vision with **MediaPipe**, **OpenCV**, and **Streamlit**.

> 🔍 Ideal for online learning, remote exams, or in-class monitoring.

---

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

---

## 🖼️ Live Demo

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-spaces-lg.svg)](https://huggingface.co/spaces/YOUR_USERNAME/student-attention-ai)

---

## 🗂️ Project Folder Structure

```bash
student-attention-ai/
│
├── streamlit_app.py              # Main Streamlit UI app
├── attention_detector.py         # Core detection logic
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
│
├── sample_video.mp4              # Optional sample video
└── utils/                        # (Optional if split into modules)
    ├── __init__.py
    └── helper.py                 # Utility functions (if any)
```


## 🛠 Installation (Local)
```bash

# Clone the repo
git clone https://github.com/your-username/student-attention-ai.git
cd student-attention-ai

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.

```

## 🌍 Deployment (Hugging Face Spaces)
Requires a Hugging Face account

Create a new Space → choose Streamlit SDK

Push your files (ensure this structure exists):
```bash

git lfs install
git clone https://huggingface.co/spaces/YOUR_USERNAME/student-attention-ai
cd student-attention-ai
# Copy all project files here
git add .
git commit -m "Initial commit"
git push
```

## 📊 Output Analysis
** Real-time visual feedback (bounding boxes with attention labels)

** Sidebar controls to start/stop session

** Line chart showing attention over time (Plotly)

** Downloadable CSV session log

## 🔮 Roadmap / Future Improvements
** 🔐 Facial recognition for named student tracking

** 🗣️ Audio analysis for whispering/speech detection

** 🧠 Custom model integration (e.g., drowsiness, emotion detection)

** 📡 Remote teacher dashboard with live alerts

** 🌐 Cross-device sync (mobile + desktop)

** 🎓 LMS integration (Moodle, Google Classroom, etc.)

## 📘 License
MIT License — Free to use, modify and distribute with attribution.