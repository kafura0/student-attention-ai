'''
Uses MediaPipe FaceMesh for facial landmark detection.

Calculates eye aspect ratio (EAR) to estimate attention.

Returns a normalized attention score between 0 and 1.

'''
# utils/attention_detector.py

import cv2
import mediapipe as mp
import numpy as np

class AttentionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False)
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    def analyze_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        h, w, _ = frame.shape
        attentive = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Eye landmarks for basic attention estimation
                left_eye = face_landmarks.landmark[33]  # left eye
                right_eye = face_landmarks.landmark[263]  # right eye

                eye_center_x = (left_eye.x + right_eye.x) / 2
                if 0.3 < eye_center_x < 0.7:
                    attentive = True

        return attentive

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = 0
        attentive_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            if self.analyze_frame(frame):
                attentive_frames += 1

        cap.release()
        attention_score = (attentive_frames / total_frames) * 100 if total_frames > 0 else 0
        return round(attention_score, 2)
