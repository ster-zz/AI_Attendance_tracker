# 📖 AI Attendance Tracker: Exhaustive Technical Manual (Phases 1-3)

This document is the ultimate guide to your project. It contains **actual code snippets** from every part of the system and explains exactly how they work, why they were written that way, and how they connect to the other parts.

---

## 📂 1. System Configuration (`config.py`)
This file sets the foundation. It tells the app where everything is located.

### 🔹 Code Snippet
```python
import os

# Base directory: Finds the absolute path where this file sits
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Database path: Joins the base directory with /data/classroom.db
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'classroom.db')

# Flask security: Used for encrypting cookies/session data
SECRET_KEY = os.environ.get('SECRET_KEY', 'default_secret_key_change_in_production')

# Debug Mode: Shows us detailed "Crash Logs" if the code fails
DEBUG = True
```
**Teacher Explanation:** We use a `config.py` file to keep our settings organized. Instead of typing the database path everywhere, we define it once here. The `SECRET_KEY` is a security requirement for the Flask website we built.

---

## 📂 2. The Database Memory (`database.py`)
This file is the "Central Nervous System" for data. We use **SQLite** because it is lightweight and doesn't require a separate server.

### 🔹 2.1 Schema Initialization (`init_db`)
```python
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table: Students - Stores names and face encodings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_encoding BLOB
        )
    ''')

    # Table: Attendance - Stores the actual "Check-ins"
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES Students(id)
        )
    ''')
    conn.commit()
    conn.close()
```
**Teacher Explanation:** When the app starts, it checks if these tables exist. `Students` keeps a permanent record of the kids, while `Attendance` keeps a record of every time they walked in front of the camera. We use "Foreign Keys" (`student_id`) to link the two tables together.

### 🔹 2.2 Synchronizing Students from Photos
```python
def sync_students_from_images():
    images_dir = os.path.join(BASE_DIR, 'data', 'student_images')
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get everyone we already know
    cursor.execute('SELECT name FROM Students')
    existing_students = set(row['name'] for row in cursor.fetchall())
    
    # Look for new folders in student_images/
    for item in os.listdir(images_dir):
        if os.path.isdir(os.path.join(images_dir, item)):
            if item not in existing_students:
                cursor.execute('INSERT INTO Students (name) VALUES (?)', (item,))
    conn.commit()
```
**Teacher Explanation:** This is a "No-Zero-Effort" feature. Instead of manually typing student names, the teacher just drops a folder named "Paul" into the `images` directory. This function sees the folder and automatically creates a new record in the database.

---

## 📂 3. The Web Application (`app.py`)
This file uses the **Flask** framework to create the Dashboard website.

### 🔹 3.1 The Main Dashboard Route
```python
@app.route('/')
def dashboard():
    students = get_all_students()
    sessions = get_all_sessions()
    active_session = get_active_session()
    
    return render_template('index.html', 
                          total_students=len(students),
                          total_sessions=len(sessions),
                          active_session=active_session)
```
**Teacher Explanation:** When you visit `http://127.0.0.1:5000`, this function runs. It gathers all the stats (total students, total sessions) and sends them to the `index.html` template. The `render_template` function is what "builds" the page for the browser.

### 🔹 3.2 Dynamic Context Processor
```python
@app.context_processor
def inject_global_data():
    students = get_all_students()
    active_session = get_active_session()
    return {
        'global_student_count': len(students),
        'global_active_session': active_session
    }
```
**Teacher Explanation:** This is a background helper. Usually, you only send data to a specific page. But since we want the "Student Count" to show up in the **Sidebar** of every single page, we use a `context_processor` to make this data "Global."

---

## 📂 4. The AI Heart (`face_recognition_module.py`)
This is the most complex part of the code. It handles the actual video processing.

### 🔹 4.1 Multi-Threaded Processing
We use **Threading** to keep the app fast. One thread runs the "Camera Display" and another thread runs the "AI Brain."
```python
def process_ai_frame():
    # ... inside a while loop ...
    # 1. Resize for speed
    small_frame = cv2.resize(frame_to_process, (0, 0), fx=0.5, fy=0.5)
    
    # 2. Use YOLOv8 to find where the face is
    results = face_detector(small_frame, verbose=False)
    
    # 3. Use dlib to identify WHO is in that spot
    face_encodings = face_recognition.face_encodings(rgb_full_frame, temp_face_locations)
```
**Teacher Explanation:** If we ran the AI on the main camera feed, the video would look like a slideshow (laggy). By using a separate background thread (`process_ai_frame`), the video stays smooth at 30fps while the AI works tirelessly in the background.

### 🔹 4.2 The Identification Logic
```python
# Compare the new face against every student we know
matches = face_recognition.compare_faces(known_face_encodings_global, face_encoding, tolerance=0.55)

# Calculate which match is the "Most Likely" (the smallest distance)
face_distances = face_recognition.face_distance(known_face_encodings_global, face_encoding)
best_match_index = np.argmin(face_distances)
```
**Teacher Explanation:** The AI calculates a "Distance" between faces. A distance of `0.0` is a perfect match. We set a `tolerance=0.55`, which means the face must be at least 45% similar to our stored photo to count as a match. This prevents "False Positives" (where it thinks you are someone else).

---

## 📂 5. Advanced ID Compliance (`id_detection_module.py`)
This is the Phase 3 feature that checks for student IDs.

### 🔹 5.1 Region of Interest (ROI)
```python
# Crop the frame to only look at the chest (below the face)
# face_bbox: (top, right, bottom, left)
chest_roi = frame[bottom : bottom + int(h * 1.5), left - 20 : right + 20]
```
**Teacher Explanation:** We don't want to scan the entire room for ID cards—that's too much work for the computer. Instead, we find the face, and then tell the computer: "Look exactly 1.5x face-heights below the face." This is the "Chest ROI."

### 🔹 5.2 Lanyard Color Masking
```python
# Convert to HSV (Hue, Saturation, Value) for better color detection
hsv = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)

# Define "Orange" range for the school lanyard
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])

# Create a 'mask' where ONLY orange things are visible
mask = cv2.inRange(hsv, lower_orange, upper_orange)
```
**Teacher Explanation:** Standard RGB colors are hard for computers to understand because of lighting. We convert to **HSV**, which allows the computer to see the "Orange" of the lanyard clearly, whether the room is bright or dim.

### 🔹 5.3 Shape & Aspect Ratio
```python
# Find the shape of the ID card
w, h = rect[1] # width and height
aspect_ratio = w / float(h)

# Check if the shape is a horizontal OR vertical rectangle
if (1.2 <= aspect_ratio <= 1.7) or (0.45 <= aspect_ratio <= 0.85):
    return True # Real ID Card Detected!
```
**Teacher Explanation:** An ID card is always a rectangle. If we find an orange blob that has the exact shape of a card (e.g., 1.5 times wider than it is tall), the system confirms it's a valid ID card.

---

## 📂 6. UI & Aesthetics (`dashboard.css`)
This is how we made the "Glass" look.

### 🔹 Code Snippet
```css
.card {
    background: rgba(255, 255, 255, 0.05); /* 5% White opacity */
    backdrop-filter: blur(20px);            /* Frosted glass effect */
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
```
**Teacher Explanation:** We used **Glassmorphism**. By combining a very low opacity background with a `backdrop-filter: blur`, we create a high-end, futuristic look that shows the teacher that this is a professional AI tool.

---

## 🚀 Presentation Talking Points
1.  **"Performance"**: We use Multi-threading and YOLOv8 to ensure the system never lags, even on standard laptops.
2.  **"Security"**: Our ID detection uses "Multi-Factor" logic (Biometric Face + Physical ID Shape + Lanyard Color).
3.  **"Efficiency"**: The automated student sync means zero manual data entry for the teacher—just drop photos into a folder and go.
