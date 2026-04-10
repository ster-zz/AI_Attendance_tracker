# 🎓 AI Classroom System (Vision-Based Attendance & Monitoring)

An intelligent classroom management system that leverages Computer Vision and AI for automated attendance tracking, safety monitoring, and performance analytics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-orange)
![AI](https://img.shields.io/badge/AI-Face%20Recognition-red)

---

## 🚀 Overview

The **AI Classroom System** is designed to modernize educational environments. By using real-time video feeds from classroom cameras, the system automatically identifies students, logs their attendance based on their actual presence, and detects behavioral incidents (like missing ID cards) without manual intervention.

## ✨ Key Features

- **🛡️ Real-time Face Recognition**: High-accuracy student identification using state-of-the-art face embedding models.
- **🕒 Automated Attendance**: Tracks "Presence Time" and marks students as **Present** only if they meet a 75% class duration threshold.
- **📸 Incident Detection**: Automatically logs safety or compliance issues (e.g., student not wearing an ID card) with snapshot evidence.
- **📹 Video Analysis**: Support for live camera feeds or uploading recorded class videos for offline analysis.
- **📊 Analytics Dashboard**: Visual KPIs for attendance rates, recent activities, and student statistics.
- **📁 Student Database**: Easy synchronization of student profiles from image folders.
- **📑 Detailed Reports**: Generate and view comprehensive session-wise attendance reports.

---

## 🛠️ Tech Stack

- **Backend**: Python (Flask Framework)
- **Database**: SQLite3 (Scalable & Lightweight)
- **AI Modules**:
  - `face_recognition`: For individual student identification.
  - `Mediapipe`: Advanced landmark detection.
  - `Ultralytics (YOLOv8)`: Object detection (ID cards, etc.).
  - `OpenCV`: Video stream processing and manipulation.
- **Frontend**: Responsive HTML5, Vanilla CSS3, and JavaScript.

---

## 📂 Directory Structure

```text
AI_Classroom_System/
├── app.py                  # Main Flask Server & Routes
├── database.py             # Database Schema & Query Logic
├── config.py               # System Configuration (Keys, Paths)
├── models/                 # AI Model Implementations
│   └── face_recognition_module.py
├── data/
│   ├── student_images/     # Training data (Folder per student)
│   ├── uploads/            # Uploaded class videos
│   └── incidents/          # Snapshot evidence of incidents
├── templates/              # UI Components (Jinja2 Templates)
├── static/                 # CSS, JS, and Asset Files
├── classroom.db            # SQLite Database (Auto-generated)
└── requirements.txt        # Project Dependencies
```

---

## ⚙️ Installation & Setup

Follow these steps to get the system running on your local machine:

### 1. Prerequisite
Ensure you have **Python 3.8+** installed.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/ai-classroom-system.git
cd AI_Classroom_System
```

### 3. Create a Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Add Student Data
Place student images in `data/student_images/`. Create a folder for each student named after their full name:
```text
data/student_images/
├── John Doe/
│   └── photo1.jpg
└── Jane Smith/
    └── photo1.jpg
```

### 6. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 📖 Usage Guide

1.  **Syncing Students**: Go to the **Settings** or **Student Directory** page and click "Sync Database" to load your students from the folders.
2.  **Starting a Class**: On the Dashboard, click **Start Session**. This begins recording attendance.
3.  **Live Monitoring**: Use the **Live Feed** page to watch the AI identify students and detect incidents in real-time.
4.  **Uploading Videos**: If you have a recorded class, go to the **Video Upload** page. The system will process it in the background.
5.  **Viewing Reports**: Navigate to **Reports** and select a session to see who was present, partial, or absent.

---

## 🚀 How to Push to GitHub

To host this project on your own GitHub:

1.  Create a new repository on GitHub (e.g., `ai-classroom-system`).
2.  Open your terminal in the project folder.
3.  Initialize git (if not already):
    ```bash
    git init
    ```
4.  Add all files:
    ```bash
    git add .
    ```
5.  Commit your changes:
    ```bash
    git commit -m "Initial commit: AI Classroom System"
    ```
6.  Link to your repository and push:
    ```bash
    git remote add origin https://github.com/your-username/ai-classroom-system.git
    git branch -M main
    git push -u origin main
    ```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
*Created for the Mini Project - AI Classroom Monitoring System.*
