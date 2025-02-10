# Face-Recognition-System
### **GitHub Repository Description: Face Recognition System**

# **Face Recognition System using InsightFace and SVM**  

This project is a real-time **face recognition system** using **InsightFace** for face detection and alignment, and **Support Vector Machines (SVM)** for classification. It consists of three main scripts:

1. **`add_face.py`** – Automatically captures and saves 100+ face embeddings per person.  
2. **`train_model.py`** – Trains an SVM classifier using stored embeddings.  
3. **`face_recognition.py`** – Recognizes faces in real time and labels unknown faces.

---

## **Features**
✅ **Automatic Face Enrollment:** Captures and saves 100 face samples per person.  
✅ **Real-Time Face Recognition:** Detects and classifies faces from a webcam feed.  
✅ **SVM-Based Classification:** Uses Support Vector Machines for better accuracy.  
✅ **Confidence Thresholding:** Unknown faces are labeled "Unknown" based on a confidence threshold.  
✅ **High Accuracy with InsightFace:** Uses deep learning-based embeddings for robust recognition.

---

## **Installation**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/face-recognition-system.git
   cd face-recognition-system
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy opencv-python insightface scikit-learn joblib
   ```

---

## **Usage**

### **1️⃣ Add New Faces (`add_face.py`)**
Automatically enrolls a person by capturing 100+ face samples.

```bash
python add_face.py
```
- Enter the person’s name when prompted.
- The script captures multiple samples and saves the embeddings.

### **2️⃣ Train the Model (`train_model.py`)**
Once enough faces are enrolled, train the classifier.

```bash
python train_model.py
```
- Trains an SVM model using the collected face embeddings.
- Saves the trained model as `face_classifier.pkl`.

### **3️⃣ Start Face Recognition (`face_recognition.py`)**
Run real-time face recognition with a confidence threshold.

```bash
python face_recognition.py
```
- Recognizes stored faces.
- Unknown faces are labeled `"Unknown"` if confidence is too low.

---

## **Configuration**
- **Adjust Confidence Threshold:** Change `CONFIDENCE_THRESHOLD` in `face_recognition.py` to fine-tune recognition accuracy.
- **Modify Sample Count:** Change `sample_target` in `add_face.py` to collect more than 100 samples.

---

## **Contributing**
Pull requests are welcome! If you'd like to improve the project, feel free to contribute. 🚀

---

## **License**
This project is licensed under the **MIT License**.

---

### **🔗 Connect with Me**
📧 Email: [your email]  
🌐 GitHub: [your GitHub profile]  
🔗 LinkedIn: [your LinkedIn profile]  

---

Let me know if you need any modifications before you upload this to GitHub! 🚀
