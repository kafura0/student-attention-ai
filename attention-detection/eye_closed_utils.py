import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


eye_threshold = 0.25

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(
    "/var/www/virtualenv/attention-detection-pose-based/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def get_eye_closed_warning(frame, bounding_box):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subject = bounding_box

    shape = predict(gray, subject)
    shape = face_utils.shape_to_np(shape)  # converting to NumPy Array

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    left_eye_asp_ratio = eye_aspect_ratio(leftEye)
    right_eye_asp_ratio = eye_aspect_ratio(rightEye)

    eye_asp_ratio = (left_eye_asp_ratio + right_eye_asp_ratio) / 2.0
    if eye_asp_ratio < eye_threshold:
        return True, frame, eye_asp_ratio
