'''
Features
Returns a score between 0-1 (closer to 1 = focused)

Uses nose-to-eye center distance as a proxy for attention

Modular: Can be imported into any script or app

Light enough for Streamlit or local webcam processing

'''
import cv2
import mediapipe as mp
import numpy as np

class AttentionDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        self.attention_threshold = 0.15  # tweakable

        # Eye landmark indices (MediaPipe reference)
        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]
        self.NOSE_TIP = 1

    def _get_landmark_coords(self, landmarks, idx, frame_shape):
        h, w = frame_shape[:2]
        point = landmarks[idx]
        return int(point.x * w), int(point.y * h)

    def _calculate_attention(self, landmarks, frame_shape):
        left_eye = self._get_landmark_coords(landmarks, self.LEFT_EYE[0], frame_shape)
        right_eye = self._get_landmark_coords(landmarks, self.RIGHT_EYE[0], frame_shape)
        nose_tip = self._get_landmark_coords(landmarks, self.NOSE_TIP, frame_shape)

        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        # Compute distance from nose to eye center (very crude "gaze" proxy)
        dx = abs(nose_tip[0] - eye_center[0])
        dy = abs(nose_tip[1] - eye_center[1])

        # Simple attention score: lower = more aligned = more attentive
        distance = np.sqrt(dx**2 + dy**2)
        score = max(0, 1 - (distance / (frame_shape[1] * self.attention_threshold)))
        return round(score, 2)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        attention_score = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            attention_score = self._calculate_attention(landmarks, frame.shape)

        return attention_score
