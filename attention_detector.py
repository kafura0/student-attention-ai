import cv2
import mediapipe as mp
import numpy as np

class AttentionDetector:
    def __init__(self, draw_landmarks=True):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.draw_landmarks = draw_landmarks
        self.min_motion_area = min_motion_area
        

        # Eye landmarks for left and right
        self.left_eye_ids = [33, 160, 158, 133, 153, 144]
        self.right_eye_ids = [362, 385, 387, 263, 373, 380]

    def _eye_aspect_ratio(self, eye_landmarks):
        # Calculate Eye Aspect Ratio (EAR) to detect eye closure
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        ear = (A + B) / (2.0 * C)
        return ear

    def detect(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        attentive_count = 0
        total_faces = 0
        cheating_flags = []

        if results.multi_face_landmarks:
            total_faces = len(results.multi_face_landmarks)
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = np.array([(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark])

                # Bounding Box
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Eye landmarks
                left_eye = np.array([landmarks[i] for i in self.left_eye_ids])
                right_eye = np.array([landmarks[i] for i in self.right_eye_ids])
                left_ear = self._eye_aspect_ratio(left_eye)
                right_ear = self._eye_aspect_ratio(right_eye)
                ear_avg = (left_ear + right_ear) / 2.0

                # Determine attention
                eye_closed = ear_avg < 0.21
                head_turned = self._is_head_turned(landmarks)

                label = "Attentive"
                if eye_closed:
                    label = "Eyes Closed"
                    cheating_flags.append("Eyes Closed")
                elif head_turned:
                    label = "Head Turned"
                    cheating_flags.append("Head Turned")
                else:
                    attentive_count += 1

                # Draw feedback
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if label != "Attentive" else (0, 255, 0), 2)

                if self.draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, face_landmarks,
                        self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    )

        return {
            "attentive_count": attentive_count,
            "total_faces": total_faces,
            "cheating_flags": cheating_flags,
            "annotated_frame": frame,
        }

    def _is_head_turned(self, landmarks):
        # Simple heuristic: Compare nose to eyes position
        left_eye_x = np.mean([landmarks[i][0] for i in self.left_eye_ids])
        right_eye_x = np.mean([landmarks[i][0] for i in self.right_eye_ids])
        nose_x = landmarks[1][0]

        # If nose is too far left/right of eye center, head is turned
        eye_center_x = (left_eye_x + right_eye_x) / 2
        offset_ratio = abs(nose_x - eye_center_x) / (right_eye_x - left_eye_x + 1e-6)
        return offset_ratio > 0.35
