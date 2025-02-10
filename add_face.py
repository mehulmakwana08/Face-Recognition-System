import os
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

# File to store the enrolled face embeddings
EMBEDDINGS_FILE = "known_faces.npy"

def auto_enroll():
    """
    Automatically captures and saves 100 face embeddings for a given person.
    The process:
      1. Prompts the user for the person's name.
      2. Opens the webcam and continuously captures frames.
      3. When exactly one face is detected, the face embedding is saved.
      4. The process repeats until 100 samples are collected.
    """
    print("=== Automatic Face Enrollment ===")
    person_name = input("Enter the name for this face: ").strip()
    if person_name == "":
        print("Error: Name cannot be empty.")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize the InsightFace model (handles face detection and alignment)
    app = FaceAnalysis()
    app.prepare(ctx_id=0)  # Use GPU if available, or set to -1 for CPU

    # Load existing embeddings if available, else create a new dictionary
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings_dict = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    else:
        embeddings_dict = {}

    # Optionally clear previous embeddings for this person (if any)
    if person_name in embeddings_dict:
        print(f"Warning: Previous embeddings found for {person_name}.")
        clear = input("Do you want to overwrite them? (Y/N): ").strip().lower()
        if clear == "y":
            embeddings_dict[person_name] = []
        else:
            print("Appending new samples to existing embeddings.")

    # Ensure the person has an entry in the dictionary
    if person_name not in embeddings_dict:
        embeddings_dict[person_name] = []

    sample_target = 100
    sample_count = len(embeddings_dict[person_name])
    print(f"Starting automatic capture for '{person_name}'.")
    print(f"Currently have {sample_count} samples. Target is {sample_target} samples.")
    print("Press 'q' at any time to exit early.")

    while sample_count < sample_target:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Detect faces in the frame
        faces = app.get(frame)
        if len(faces) == 1:
            # Only one face is accepted; if more (or less) are detected, skip the frame.
            face = faces[0]
            embedding = face.embedding
            embeddings_dict[person_name].append(embedding)
            sample_count += 1
            print(f"Captured sample {sample_count} for {person_name}.")
        else:
            # Optionally, print or log that the frame was skipped.
            print("Ensure exactly one face is visible. Skipping frame.")

        # Show the current frame (with bounding boxes drawn for feedback)
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {sample_count}/{sample_target}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Automatic Enrollment", frame)

        # Wait for 500 ms between frames (adjust delay as needed)
        key = cv2.waitKey(500) & 0xFF
        if key == ord('q'):
            print("Exiting early by user command.")
            break

    # Save the updated embeddings to file
    np.save(EMBEDDINGS_FILE, embeddings_dict)
    print(f"Enrollment complete for '{person_name}'. Total samples saved: {sample_count}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    auto_enroll()
