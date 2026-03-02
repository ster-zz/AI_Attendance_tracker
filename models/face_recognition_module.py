import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
import sqlite3
import threading
import time
import torch
from ultralytics import YOLO

# Strongly restrict PyTorch from consuming 100% of all CPU cores
# This leaves the remaining cores free for the OS to run the camera drivers smoothly
torch.set_num_threads(2)

from config import DATABASE_PATH
from database import get_db_connection, is_session_active, get_current_session_id

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'student_images')
ENCODINGS_FILE = os.path.join(BASE_DIR, 'data', 'encodings.pkl')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n-face.pt')

# Load the YOLO model once globally using Ultralytics
if os.path.exists(YOLO_MODEL_PATH):
    face_detector = YOLO(YOLO_MODEL_PATH)
else:
    print(f"CRITICAL WARNING: YOLOv8 PyTorch model not found at {YOLO_MODEL_PATH}. Recognition will fail.")
    face_detector = None

def encode_known_faces():
    """
    Loads images from data/student_images/, extracts Face Encodings 
    and saves them to data/encodings.pkl.
    Does NOT reconstruct the database, merely provides encodings.
    """
    if not os.path.exists(STUDENT_IMAGES_DIR):
        print(f"Warning: {STUDENT_IMAGES_DIR} not found.")
        return

    known_encodings = []
    known_names = []
    
    # Track the last modified times of images to know which ones we've already processed
    image_mtimes = {}
    
    # Load existing encodings if they exist to prevent recalculating old images
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                known_encodings = data.get("encodings", [])
                known_names = data.get("names", [])
                image_mtimes = data.get("image_mtimes", {})
        except Exception as e:
            print(f"Error loading existing encodings: {e}. Starting fresh.")
    
    new_encodings_added = 0
    
    # Iterate through all files in the student images directory
    for root, _, files in os.walk(STUDENT_IMAGES_DIR):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                
                # Get the last modified time of the file
                current_mtime = os.path.getmtime(path)
                
                # Check if we already processed this exact file and it hasn't changed
                # We use a unique key: combo of folder name + file name to prevent collision
                name = os.path.basename(root)
                file_key = f"{name}_{file}"
                
                if file_key in image_mtimes and image_mtimes[file_key] == current_mtime:
                    # Skip it! It's already successfully encoded in thepkl file
                    continue
                    
                try:
                    # Load image and encode
                    image = face_recognition.load_image_file(path)
                    
                    # Convert to RGB (OpenCV uses BGR, face_recognition expects RGB)
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                        image_mtimes[file_key] = current_mtime # Save timestamp
                        new_encodings_added += 1
                        print(f"Encoded NEW face: {name} (Image: {file})")
                    else:
                        print(f"Warning: No face found in {file}")
                
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    # Only save if we actually found something new
    if new_encodings_added > 0:
        data = {"encodings": known_encodings, "names": known_names, "image_mtimes": image_mtimes}
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"Encodings saved. Added {new_encodings_added} new encodings. Total: {len(known_names)}.")
    else:
        print(f"No new images to encode. Total existing encodings: {len(known_names)}.")

def load_encodings():
    """
    Loads saved face encodings from data/encodings.pkl.
    """
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    
    print("No encodings found. Please run encode_known_faces() first.")
    return [], []

def mark_attendance(name):
    """
    Marks the student "Present" ONLY IF:
    1. A session is currently ACTIVE.
    2. The student hasn't ALREADY been marked present in this session.
    """
    if not is_session_active():
        # Do nothing if no session is active.
        return
        
    session_id = get_current_session_id()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # First: We need the student's ID from the Students table based on their name.
    # Note: If student is not in DB yet, we skip or add placeholder (handling gracefully)
    cursor.execute("SELECT id FROM Students WHERE name = ?", (name,))
    student_record = cursor.fetchone()
    
    if not student_record:
        # Graceful handling if image exists but DB entry doesn't
        print(f"Student {name} recognized but not found in Students database. Skipping attendance.")
        conn.close()
        return
        
    student_id = student_record['id']
    
    # Second: Check if attendance already exists for this session AND this student
    cursor.execute('''
        SELECT id FROM Attendance 
        WHERE session_id = ? AND student_id = ?
    ''', (session_id, student_id))
    
    if cursor.fetchone() is None:
        # Not marked yet! Insert now.
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO Attendance (session_id, student_id, status, reason, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, student_id, "Present", "Face Recognition", timestamp))
        conn.commit()
        print(f"Attendance MARKED PRESENT for {name} at {timestamp}.")
    else:
        # Already marked
        pass
        
    conn.close()

# Threading Global Variables
current_frame = None
latest_face_locations = []
latest_face_names = []
known_face_encodings_global = []
known_face_names_global = []
thread_running = False
ai_is_processing = False

def process_ai_frame():
    """
    Background Thread: Continuously grabs the latest frame from the webcam,
    runs the heavy PyTorch YOLOv8 & dlib networks on it, and updates bounding boxes.
    Runs completely independently of the webcam render loop to prevent lag.
    """
    global current_frame, latest_face_locations, latest_face_names, thread_running, ai_is_processing
    
    while thread_running:
        if current_frame is None or ai_is_processing:
            time.sleep(0.01)
            continue
            
        ai_is_processing = True
        
        # Clone frame securely for background processing
        frame_to_process = current_frame.copy()
        
        # Resize frame to 1/2 size for faster face recognition processing
        small_frame = cv2.resize(frame_to_process, (0, 0), fx=0.5, fy=0.5)

        # Convert BGR (OpenCV) to RGB (face_recognition) for full-res encoding later
        rgb_full_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
        
        # Use YOLOv8 PyTorch for Face Location Detection on the FAST small frame
        temp_face_locations = []
        if face_detector is not None:
            # Run YOLO inference
            results = face_detector(small_frame, verbose=False)
            
            # Extract bounding boxes from YOLO results
            for box in results[0].boxes.xyxy: # returns [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = box.tolist()
                
                # Convert to integer and scale UP to match the full-resolution frame
                x_min = int(x_min * 2)
                y_min = int(y_min * 2)
                x_max = int(x_max * 2)
                y_max = int(y_max * 2)
                
                # Convert YOLO format to CSS (top, right, bottom, left) required by face_recognition
                temp_face_locations.append((y_min, x_max, y_max, x_min))
        
        # Pass the 100% resolution frame and 100% scale face locations into Dlib
        # This fixes the accuracy bug where Dlib was getting corrupted 50% resolution geometry
        temp_face_encodings = face_recognition.face_encodings(rgb_full_frame, temp_face_locations, num_jitters=1)

        temp_face_names = []
        for face_encoding in temp_face_encodings:
            # Compare against known encodings with STRICT tolerance (0.55 instead of default 0.60)
            matches = face_recognition.compare_faces(known_face_encodings_global, face_encoding, tolerance=0.55)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings_global, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names_global[best_match_index]
                    
                    # --- THE CORE ATTENDANCE TRIGGER ---
                    # If a match is found, only mark if the session is active.
                    if name != "Unknown":
                        mark_attendance(name)

            temp_face_names.append(name)
            
        # Safely overwrite globals with new frame data
        latest_face_locations = temp_face_locations
        latest_face_names = temp_face_names
        
        ai_is_processing = False
        time.sleep(3) # Wait 3 seconds between processing frames to massively reduce CPU load


def start_face_recognition():
    """
    Main Thread: Starts the live webcam render loop.
    Launches the background AI thread. Runs at 30fps.
    """
    global current_frame, latest_face_locations, latest_face_names, known_face_encodings_global, known_face_names_global, thread_running
    
    # 1. Load Encodings (Do not re-encode)
    known_face_encodings_global, known_face_names_global = load_encodings()
    
    if not known_face_encodings_global:
        print("No encodings available. Cannot start recognition.")
        return

    # 2. Start Video Capture using Windows DirectShow for instantly unblocked performance
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Force the camera physical hardware to a lower resolution to prevent buffer lag
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    
    # Check if webcam opened successfully
    if not video_capture.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Live Recognition Started. Press 'q' to quit.")

    # 3. Launch the Background AI Thread
    thread_running = True
    ai_thread = threading.Thread(target=process_ai_frame, daemon=True)
    ai_thread.start()

    while is_session_active():
        # Grab a single frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Push frame to global cache for the AI thread
        current_frame = frame.copy()

        # Render exactly what the AI thread most recently discovered
        for (top, right, bottom, left), name in zip(latest_face_locations, latest_face_names):
            # Draw a box around the face (No scaling needed, AI thread outputs native 100% boxes now)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
            
        # Visual Indicator for Active Session
        font = cv2.FONT_HERSHEY_DUPLEX
        if is_session_active():
            cv2.putText(frame, "STATUS: RECORDING ATTENDANCE", (20, 30), font, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "STATUS: IGNORED (NO ACTIVE SESSION)", (20, 30), font, 0.6, (0, 0, 255), 1)

        # Display the resulting image flawlessly at max fps
        cv2.imshow('Classroom Face Recognition [Phase 3]', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Gracefully shut down threads and camera
    thread_running = False
    ai_thread.join(timeout=2.0)
    video_capture.release()
    cv2.destroyAllWindows()
