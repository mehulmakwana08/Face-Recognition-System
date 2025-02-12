import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import joblib

EMBEDDINGS_FILE = "known_faces.npy"
CLASSIFIER_FILE = "face_classifier.pkl"

def train_model():
    """Trains a classifier using SVM for better recognition accuracy."""
    if not os.path.exists(EMBEDDINGS_FILE):
        print("Error: No face data found. Enroll faces first.")
        return

    embeddings_dict = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    X, y = [], []
    for person, embeddings in embeddings_dict.items():
        X.extend(embeddings)
        y.extend([person] * len(embeddings))

    if len(set(y)) < 2:
        print("Error: Need at least two different people.")
        return

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    print(f"Training model on {len(y)} samples...")

    # Using SVM with optimized parameters
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X, y)

    joblib.dump((clf, scaler, label_encoder), CLASSIFIER_FILE)
    print(f"Model saved as '{CLASSIFIER_FILE}'.")

if __name__ == '__main__':
    train_model()