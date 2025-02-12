import os
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

IMAGES_DIR = "faces"  # Folder where images are stored
EMBEDDINGS_FILE = "known_faces.npy"
BLUR_THRESHOLD = 100  # Adjust as needed

def calculate_blur(image):
    """Calculate blur metric using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_images():
    """Processes static images and saves face embeddings."""
    print("=== Static Image Face Enrollment ===")

    app = FaceAnalysis()
    app.prepare(ctx_id=0)

    embeddings_dict = {}
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings_dict = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()

    for person_name in os.listdir(IMAGES_DIR):
        person_path = os.path.join(IMAGES_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        if person_name not in embeddings_dict:
            embeddings_dict[person_name] = []

        print(f"Processing images for: {person_name}")

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            frame = cv2.imread(image_path)

            if frame is None:
                print(f"Warning: Couldn't read {image_path}")
                continue

            faces = app.get(frame)
            if len(faces) == 1:
                face = faces[0]
                bbox = face.bbox.astype(int)
                face_roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                if face_roi.size == 0:
                    continue

                blur_score = calculate_blur(face_roi)
                if blur_score < BLUR_THRESHOLD:
                    print(f"Skipping {image_name} due to blurriness (Score: {blur_score:.1f})")
                    continue

                # Save face embedding
                embeddings_dict[person_name].append(face.embedding)
                print(f"Saved embedding for {image_name} (Blur: {blur_score:.1f})")

    # Save embeddings
    np.save(EMBEDDINGS_FILE, embeddings_dict)
    print("Enrollment complete. Embeddings saved.")

if __name__ == '__main__':
    process_images()
