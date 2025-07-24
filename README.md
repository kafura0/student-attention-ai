# 🧠 Student Attention AI

A lightweight attention detection system that analyzes student focus from video input. Built for virtual classrooms, it helps educators assess real-time or recorded engagement levels using computer vision and facial landmarks.

![Demo Screenshot](assets/demo_screenshot.png)

---

## 🚀 Features

- 📹 **Video Upload & Analysis** — Upload a pre-recorded lecture video
- 🧑‍💻 **Live Webcam Support** — Run real-time attention detection via webcam
- 🧠 **Blink & Head Pose Estimation** — Detect distraction based on blinking rate and head movement
- 📊 **Attention Scoring** — Generate frame-by-frame focus scores
- 🎛️ **Streamlit Interface** — Easy to use frontend with upload and results panel

---

## 🛠️ Tech Stack

- Python 3.10+
- OpenCV
- dlib
- imutils
- numpy
- Streamlit
- Mediapipe (optional future addition)

---

## 📂 Project Structure

student-attention-ai/
├── streamlit_app.py # Streamlit frontend app
├── live_test.py # Webcam-based real-time test
├── requirements.txt
├── utils/
│ └── attention_detector.py # Core logic: blink, gaze, scoring
├── assets/
│ ├── demo_screenshot.png # Optional demo image
│ └── attention_video_sample.mp4
└── README.md


---

## ▶️ How to Use

### 1. 🔧 Setup

```bash
git clone https://github.com/kafura0/student-attention-ai.git
cd student-attention-ai
pip install -r requirements.txt
```

### 2.  Run the Streamlit App (File Upload)
```
streamlit run streamlit_app.py
```
Upload a short classroom or face-focused video clip (.mp4/.mov).

### 3. 📡 Run Live Webcam Detector (Optional)

```
python live_test.py
```
Press q to exit.

### 💡 How Attention is Calculated
Attention scores are based on:

Eye aspect ratio (blink frequency)

Head pose direction

Face presence duration in frame

Each frame is assigned an attention score between 0 (distracted) and 1 (fully attentive).

### 🧩 Future Improvements
Replace dlib with MediaPipe or OpenVINO for faster inference

Add heatmaps or score graphs to Streamlit

Export attention scores as CSV

Multi-face tracking (for group classrooms)

Deploy to Streamlit Cloud or Hugging Face

### 🤝 Contributing
Feel free to fork, enhance, or suggest features via Issues or Pull Requests!

🧑‍🎓 Built with ❤️ by Kafura