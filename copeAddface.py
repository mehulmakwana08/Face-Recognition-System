import os
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

EMBEDDINGS_FILE = "known_faces.npy"
BLUR_THRESHOLD = 100  # Adjust based on testing

def calculate_blur(frame):
    """Calculate blur metric using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def auto_enroll():
    """Automatically captures and saves face embeddings with quality checks."""
    print("=== Automatic Face Enrollment ===")
    person_name = input("Enter the name for this face: ").strip()
    if not person_name:
        print("Error: Name cannot be empty.")
        return

    # Create folder for saving images
    save_dir = os.path.join("faces", person_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Images will be saved to: {save_dir}")

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    app = FaceAnalysis()
    app.prepare(ctx_id=0)

    embeddings_dict = {}
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings_dict = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()

    if person_name not in embeddings_dict:
        embeddings_dict[person_name] = []

    sample_target = 100
    sample_count = len(embeddings_dict[person_name])
    print(f"Starting capture for '{person_name}'. Samples: {sample_count}/{sample_target}.")
    print("Press 'q' to exit early.")

    while sample_count < sample_target:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        faces = app.get(frame)
        if len(faces) == 1:
            face = faces[0]
            bbox = face.bbox.astype(int)
            face_roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            if face_roi.size == 0:
                continue  # Skip empty regions
            
            blur_score = calculate_blur(face_roi)
            if blur_score < BLUR_THRESHOLD:
                cv2.putText(frame, "Blurry!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Enrollment", frame)
                cv2.waitKey(1)
                continue

            # Save the face image
            img_path = os.path.join(save_dir, f"{person_name}_{sample_count}.jpg")
            cv2.imwrite(img_path, face_roi)
            
            embeddings_dict[person_name].append(face.embedding)
            sample_count += 1
            print(f"Captured sample {sample_count} (Blur: {blur_score:.1f})")

        # Visualization
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {sample_count}/{sample_target}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Enrollment", frame)

        if cv2.waitKey(200) & 0xFF == ord('q'):
            print("Exiting early.")
            break

    np.save(EMBEDDINGS_FILE, embeddings_dict)
    print(f"Enrollment complete. Total samples: {sample_count}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    auto_enroll()