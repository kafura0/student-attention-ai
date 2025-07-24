# utils/attention_detector.py

import cv2
import numpy as np
import mediapipe as mp

class AttentionDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.drowsy_frames = 0
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 15

    def eye_aspect_ratio(self, eye_landmarks):
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (A + B) / (2.0 * C)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        attentive = True

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            face = results.multi_face_landmarks[0]
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face.landmark]

            # Eye indices from MediaPipe's face mesh
            left_eye_idx = [362, 385, 387, 263, 373, 380]  # Right eye in image
            right_eye_idx = [33, 160, 158, 133, 153, 144]  # Left eye in image

            left_eye = np.array([landmarks[i] for i in left_eye_idx], dtype='float')
            right_eye = np.array([landmarks[i] for i in right_eye_idx], dtype='float')

            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Check if eyes are closed
            if avg_ear < self.EAR_THRESHOLD:
                self.drowsy_frames += 1
                if self.drowsy_frames >= self.EAR_CONSEC_FRAMES:
                    attentive = False
            else:
                self.drowsy_frames = 0

            # Draw facial landmarks
            for idx in left_eye_idx + right_eye_idx:
                cv2.circle(frame, landmarks[idx], 2, (255, 255, 0), -1)

        else:
            attentive = False  # No face detected

        return attentive, frame
