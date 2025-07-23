![Run Demo](final_demo.gif)

# 🧠 Student Attention Detection (AI Prototype)

An AI prototype that uses facial behavior tracking to infer student attention levels during online learning sessions.

## 🎯 Goal

To help educators monitor and improve student engagement by analyzing visual cues like head position and eye direction using machine learning.

## 🔍 Features

- 🔴 Real-time webcam feed processing
- 🧠 Face detection and landmark extraction using OpenCV
- 📐 Tracks head tilt, blink rate, and gaze direction
- 🧪 Early-stage attention scoring model

## 🛠 Tech Stack

- Python
- OpenCV
- Dlib / Mediapipe
- Numpy / Scikit-learn

## 🚀 Setup

```bash
git clone https://github.com/kafura0/student-attention-ai.git
cd student-attention-ai
pip install -r requirements.txt
python detect_attention.py
Ensure your webcam is enabled and functional.

⚙️ Example Workflow
Start video stream

Extract eye landmarks + head pose

Calculate heuristic attention score

Optionally, log timestamps and attention dips

📌 Limitations
Prototype only – not production-grade

Not yet trained on large datasets

Accuracy may vary based on lighting & device

👩🏽‍💻 Author
Created by Joan Kabura Njoroge to explore how AI can support remote learning environments.

yaml
Copy
Edit

---

Would you like help:
- Adding these directly to your repos?
- Creating screenshots or GIFs to include in the READMEs?
- Planning enhancements to either project (e.g. deploy NLP results in a Streamlit app)?

Once these are done, we’ll move on to:
✅ Resume  
✅ Cover letter  
✅ Job targeting  
✅ Mentorship circles

Just say the word!








Ask ChatGPT


* These values are used to issue warning if out of limit.
* The detected face bounding box and frame is passd to eye_closed_utils.py , which detects face features and finds the distance between the eye lids.
* Threshold is set to 0.25 if value less then that then eyes are closed.



           

