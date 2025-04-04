# 🚗 Drowsiness Detection System

## 📌 Overview
This **Drowsiness Detection System** is a real-time application that uses **OpenCV, TensorFlow, and Keras** to monitor a driver’s eyes and trigger an alarm if drowsiness is detected. The system captures live video, processes frames, and predicts whether the eyes are open or closed using a trained **CNN model**.

## 📸 Demo (ALERTED) 
<img src="/images/img1.png" width="270">  

---  

## 📸 Demo  (NOT-ALERTED)
<img src="/images/img2.png" width="270">  

---

## 📂 Dataset Setup
Before running the project, **download the dataset** from Kaggle and place it in the same directory as the Python script.

🔗 **[Download Dataset](https://www.kaggle.com/your-dataset-link)**

Ensure the dataset is structured as follows:
```
/drowsiness-detection/
│-- dataset_new/
│   ├── train/
│   │   ├── open/
│   │   ├── closed/
│   ├── test/
│   │   ├── open/
│   │   ├── closed/
│-- drowsiness_detection.py
│-- requirements.txt
```

---

## 🚀 Installation
### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/drowsiness-detection.git
cd drowsiness-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Ensure Correct File Paths
- Update **dataset paths** in `drowsiness_detection.py`.
- Set the correct path for the **alarm sound file** (`ALARME2.WAV`).

### Step 4: Run the Project
```bash
python drowsiness_detection.py
```

---

## ⚙️ How It Works
1. The system **captures video** from the webcam.
2. It **detects the face and eyes** using OpenCV’s Haarcascade classifiers.
3. The trained **CNN model** predicts if the eyes are open or closed.
4. A **drowsiness score** is calculated based on continuous eye closure.
5. If the score **exceeds a threshold**, an **alarm sound** is triggered.

---

## 🖼️ Screenshots
✅ **Normal State**
![Normal Eye Open](IMAGE_LINK)

❌ **Drowsy State**
![Drowsiness Alert](IMAGE_LINK)

---

## 🔧 Troubleshooting
**Issue: OpenCV window doesn’t open (Ubuntu/Linux)**
```bash
sudo apt install libgtk2.0-dev pkg-config
```

**Issue: Alarm sound file not found**
- Ensure `ALARME2.WAV` is placed correctly and update its path in the script.

---

## 🏆 Credits
- **Haarcascade Classifiers** - OpenCV
- **CNN Model** - TensorFlow/Keras
- **Dataset** - [Kaggle](https://www.kaggle.com/your-dataset-link)

📌 **Feel free to contribute and improve this project!**

🔗 **GitHub Repository**: [Your Repo Link](https://github.com/your-username/drowsiness-detection)

