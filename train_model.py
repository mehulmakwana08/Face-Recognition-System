import os
import numpy as np
from sklearn.svm import SVC
import joblib

EMBEDDINGS_FILE = "known_faces.npy"
CLASSIFIER_FILE = "face_classifier.pkl"

def train_model():
    """
    Loads face embeddings from file, trains an SVM classifier,
    and saves the trained classifier.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        print("Error: Embeddings file not found. Enroll some faces first.")
        return

    embeddings_dict = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    X = []
    y = []
    for person, embeddings in embeddings_dict.items():
        for emb in embeddings:
            X.append(emb)
            y.append(person)
            
    if len(np.unique(y)) < 2:
        print("Error: The number of classes has to be greater than one; please enroll faces for at least two different persons.")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"Training model on {len(y)} samples...")
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, CLASSIFIER_FILE)
    print(f"Model trained and saved as '{CLASSIFIER_FILE}'.")

if __name__ == '__main__':
    train_model()
