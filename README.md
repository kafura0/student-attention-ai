🧠 Student Attention AI
Real-time Attention Detection App for students using webcam or uploaded videos. Detects attentiveness based on facial orientation and eye closure using computer vision and machine learning.

Built with Streamlit, MediaPipe, and OpenCV – deployable on both desktop and mobile via browser.

🚀 Features
✅ Live camera or uploaded video analysis

✅ Face detection with attention scoring

✅ Eye closure detection (EAR-based) for drowsiness detection

✅ Real-time annotated video display

✅ Attention logs exportable to CSV

✅ Visual attention timeline chart

✅ Easy deployment via Streamlit Cloud

📸 Demo
<img src="https://github.com/your-username/student-attention-ai/assets/demo.gif" width="600"/>
🛠️ Installation
bash
Copy
Edit
git clone https://github.com/your-username/student-attention-ai.git
cd student-attention-ai
pip install -r requirements.txt
streamlit run streamlit_app.py
📝 Python ≥ 3.8 recommended. For live webcam use, ensure camera permissions are allowed.

📂 Project Structure
bash
Copy
Edit
student-attention-ai/
│
├── streamlit_app.py           # Streamlit UI and logic
├── utils/
│   └── attention_detector.py  # Face + Eye Closure detection
├── logs/
│   └── attention_log.csv      # Exported attention sessions
├── requirements.txt
└── README.md
📈 Example Use Cases
🏫 Online learning platforms (Zoom/Google Meet)

🧪 Remote exam invigilation

👩‍🏫 Classroom analytics

💼 Employee monitoring (optional adaptation)

📤 Deployment (Streamlit Cloud)
Push the project to GitHub

Visit streamlit.io/cloud

Select your repo and deploy streamlit_app.py

Enjoy real-time attention tracking in the browser!

✅ TODOs / Suggestions for Improvement
Feature	Status	Suggestion
✅ Head pose + Eye closure	Done	Consider integrating dlib or deep learning for robustness
⏳ Multi-face detection	Not done	Currently handles only 1 face – extend for classrooms
⏳ Long-session reporting	Not done	Add attention heatmaps or daily summary
⏳ Notifications	Not done	Send alerts to user/teacher when attention drops
⏳ Face ID tagging	Not done	Save attention logs per student ID
⏳ Audio feedback	Not done	"You're losing focus" – optional voice feedback

🧠 Tech Stack
Streamlit – for UI and live demo

OpenCV – for frame capture & rendering

MediaPipe – for face landmarks

NumPy – for EAR calculations

Pandas – for log management & CSV export

👤 Author
Joan Kabura Njoroge
🌐 Website | 📧 Email | 🐙 GitHub

📄 License
MIT License – free to use and adapt with attribution.

🧪 Want to Contribute?
PRs are welcome! You can:

Add audio alerts

Improve multi-face support

Add database backend for attention logs