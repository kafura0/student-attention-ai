
import cv2
from utils.attention_detector import AttentionDetector

cap = cv2.VideoCapture(0)
detector = AttentionDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    score = detector.process_frame(frame)
    if score is not None:
        cv2.putText(frame, f'Attention Score: {score:.2f}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Live Attention Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
