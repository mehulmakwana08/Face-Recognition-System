import os
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import joblib

# File for the trained classifier
CLASSIFIER_FILE = "face_classifier.pkl"

def recognize_faces():
    """
    Recognition mode:
    - Loads the trained classifier.
    - Opens the webcam and detects faces using InsightFace (which auto-aligns faces).
    - Computes embeddings and predicts the identity using the classifier.
    - Displays the name and confidence on the video frame.
    - If the classifier's confidence is below a defined threshold, the face is labeled as "Unknown".
    """
    if not os.path.exists(CLASSIFIER_FILE):
        print("Error: Classifier not found. Please train the model first (run train_model.py).")
        return

    clf = joblib.load(CLASSIFIER_FILE)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    app = FaceAnalysis()
    app.prepare(ctx_id=0)

    # Set a confidence threshold: below this, we label the face as "Unknown"
    CONFIDENCE_THRESHOLD = 0.9  # Adjust this value for better accuracy if needed

    print("Starting face recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        faces = app.get(frame)
        for face in faces:
            embedding = face.embedding
            # Predict identity using the trained classifier
            pred = clf.predict(embedding.reshape(1, -1))[0]
            proba = clf.predict_proba(embedding.reshape(1, -1))[0]
            confidence = np.max(proba)
            
            # If the confidence is below the threshold, label as "Unknown"
            if confidence < CONFIDENCE_THRESHOLD:
                label = "Unknown"
            else:
                label = f"{pred} ({confidence:.2f})"
            
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition (press 'q' to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_faces()
