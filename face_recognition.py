import os
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import joblib

CLASSIFIER_FILE = "face_classifier.pkl"

def recognize_faces():
    """Recognizes faces with dynamic confidence thresholding."""
    if not os.path.exists(CLASSIFIER_FILE):
        print("Error: Train the model first.")
        return

    clf, scaler, label_encoder = joblib.load(CLASSIFIER_FILE)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    app = FaceAnalysis()
    app.prepare(ctx_id=0)

    CONFIDENCE_THRESHOLD = 0.70  # Adjusted for better balance

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        faces = app.get(frame)
        for face in faces:
            embedding = scaler.transform([face.embedding])
            probabilities = clf.predict_proba(embedding)[0]
            confidence = np.max(probabilities)
            pred_num = np.argmax(probabilities)

            label = "Unknown"
            if confidence >= CONFIDENCE_THRESHOLD:
                label = label_encoder.inverse_transform([pred_num])[0]

            bbox = face.bbox.astype(int)
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_faces()