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
torch.set_num_threads(2)

from config import DATABASE_PATH
from database import get_db_connection, is_session_active, get_current_session_id, log_incident

# AI detection modules
try:
    from models.id_detection_module import detect_id_card, save_incident_image
    from models.sleep_detection_module import check_sleep
except ImportError:
    from id_detection_module import detect_id_card, save_incident_image
    from sleep_detection_module import check_sleep

# ── Per-student incident cooldowns ───────────────────────────────────────────
_id_incident_cooldown = {}      # {student_name: datetime_of_last_incident}
_sleep_incident_cooldown = {}   # {student_name: datetime_of_last_incident}
_student_compliance_status = {}  # {student_name: {'id': bool, 'sleep': bool}}
ID_INCIDENT_COOLDOWN_SECONDS = 60
SLEEP_INCIDENT_COOLDOWN_SECONDS = 120 # Log sleeping less frequently

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'student_images')
ENCODINGS_FILE = os.path.join(BASE_DIR, 'data', 'encodings.pkl')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n-face.pt')

# Load the YOLO model once globally
if os.path.exists(YOLO_MODEL_PATH):
    face_detector = YOLO(YOLO_MODEL_PATH)
else:
    print(f"CRITICAL WARNING: YOLOv8 model not found at {YOLO_MODEL_PATH}.")
    face_detector = None

def encode_known_faces():
    """Extracts face encodings and saves to pkl."""
    if not os.path.exists(STUDENT_IMAGES_DIR):
        return
    known_encodings = []
    known_names = []
    image_mtimes = {}
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                known_encodings = data.get("encodings", [])
                known_names = data.get("names", [])
                image_mtimes = data.get("image_mtimes", {})
        except: pass
    new_added = 0
    for root, _, files in os.walk(STUDENT_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                mtime = os.path.getmtime(path)
                name = os.path.basename(root)
                key = f"{name}_{file}"
                if key in image_mtimes and image_mtimes[key] == mtime:
                    continue
                try:
                    img = face_recognition.load_image_file(path)
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        known_encodings.append(encs[0])
                        known_names.append(name)
                        image_mtimes[key] = mtime
                        new_added += 1
                except: pass
    if new_added > 0:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump({"encodings": known_encodings, "names": known_names, "image_mtimes": image_mtimes}, f)

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    return [], []

def mark_attendance(name):
    if not is_session_active(): return
    from database import mark_attendance_for_session
    session_id = get_current_session_id()
    if session_id:
        mark_attendance_for_session(name, session_id)

# Threading globals
current_frame = None
latest_face_locations = []
latest_face_names = []
known_face_encodings_global = []
known_face_names_global = []
thread_running = False
ai_is_processing = False
latest_processed_frame = None
frame_lock = threading.Lock()

def gen_frames():
    """Generator function for streaming frames to the web UI."""
    global latest_processed_frame
    while True:
        with frame_lock:
            if latest_processed_frame is None:
                # Provide a black frame if no stream is active
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Feed Offline", (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
            else:
                ret, buffer = cv2.imencode('.jpg', latest_processed_frame)
            
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # Cap at ~25 FPS

def process_ai_frame():
    global current_frame, latest_face_locations, latest_face_names, thread_running, ai_is_processing
    last_recognition_time = 0
    
    while thread_running:
        if current_frame is None or ai_is_processing:
            time.sleep(0.01)
            continue
            
        ai_is_processing = True
        now = time.time()
        frame_to_process = current_frame.copy()
        
        # 1. 🐢 SLOW PATH: IDENTIFICATION (Every 2 seconds)
        # Identifies WHO is in the frame.
        if now - last_recognition_time >= 2.0:
            small_frame = cv2.resize(frame_to_process, (0, 0), fx=0.5, fy=0.5)
            rgb_full = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            
            temp_locs = []
            if face_detector:
                results = face_detector(small_frame, verbose=False)
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = box.tolist()
                    temp_locs.append((int(y1*2), int(x2*2), int(y2*2), int(x1*2)))
            
            temp_encs = face_recognition.face_encodings(rgb_full, temp_locs, num_jitters=1)
            temp_names = []
            for i, enc in enumerate(temp_encs):
                matches = face_recognition.compare_faces(known_face_encodings_global, enc, tolerance=0.55)
                name = "Unknown"
                dists = face_recognition.face_distance(known_face_encodings_global, enc)
                if len(dists) > 0:
                    idx = np.argmin(dists)
                    if matches[idx]:
                        name = known_face_names_global[idx]
                        mark_attendance(name)
                temp_names.append(name)
            
            latest_face_locations, latest_face_names = temp_locs, temp_names
            last_recognition_time = now

        # 2. ⚡ FAST PATH: COMPLIANCE (Every 0.2 seconds)
        # Monitors SLEEP and ID status for recognized students at high frequency.
        for i, name in enumerate(latest_face_names):
            if name != "Unknown" and i < len(latest_face_locations):
                face_bbox = latest_face_locations[i]
                # High-frequency compliance checks
                _check_id_card(frame_to_process, name, face_bbox)
                _check_for_sleep(frame_to_process, name, face_bbox)
        
        ai_is_processing = False
        # Small sleep ensures 5-10 checks per second for very high accuracy
        time.sleep(0.1)

def _check_id_card(frame, student_name, face_bbox):
    global _id_incident_cooldown
    if face_bbox is None: return
    now = datetime.now()
    if student_name in _id_incident_cooldown:
        if (now - _id_incident_cooldown[student_name]).total_seconds() < ID_INCIDENT_COOLDOWN_SECONDS:
            return
    if not detect_id_card(frame, face_bbox):
        _do_log_incident(frame, student_name, 'no_id_card')
        _id_incident_cooldown[student_name] = now
        _student_compliance_status.setdefault(student_name, {})['id'] = False
    else:
        _id_incident_cooldown.pop(student_name, None)
        _student_compliance_status.setdefault(student_name, {})['id'] = True

def _check_for_sleep(frame, student_name, face_bbox):
    global _sleep_incident_cooldown
    if face_bbox is None: return
    now = datetime.now()
    if student_name in _sleep_incident_cooldown:
        if (now - _sleep_incident_cooldown[student_name]).total_seconds() < SLEEP_INCIDENT_COOLDOWN_SECONDS:
            return
    if check_sleep(frame, face_bbox, student_name):
        _do_log_incident(frame, student_name, 'sleeping')
        _sleep_incident_cooldown[student_name] = now
        _student_compliance_status.setdefault(student_name, {})['sleep'] = True
    else:
        _student_compliance_status.setdefault(student_name, {})['sleep'] = False

def _do_log_incident(frame, student_name, incident_type):
    """Internal helper to save image and log to DB."""
    img_path = save_incident_image(frame, student_name, incident_type)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM Students WHERE name = ?", (student_name,))
    row = cur.fetchone()
    conn.close()
    if row:
        sid = get_current_session_id()
        if sid:
            log_incident(sid, row['id'], incident_type, img_path)

def start_face_recognition():
    global current_frame, latest_face_locations, latest_face_names, known_face_encodings_global, known_face_names_global, thread_running, latest_processed_frame
    known_face_encodings_global, known_face_names_global = load_encodings()
    
    print("[DEBUG] Attempting to open camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened(): 
        print("[ERROR] Could not open camera. It might be in use by another process.")
        return
        
    print("[DEBUG] Camera opened successfully.")
    thread_running = True
    ai_t = threading.Thread(target=process_ai_frame, daemon=True); ai_t.start()
    
    if not is_session_active():
        print("[DEBUG] No active session detected when starting recognition.")

    while is_session_active():
        ret, frame = cap.read()
        if not ret: break
        current_frame = frame.copy()
        
        # Create a copy for the web UI annotations
        display_frame = frame.copy()
        
        for (t, r, b, l), name in zip(latest_face_locations, latest_face_names):
            status = _student_compliance_status.get(name, {'id': True, 'sleep': False})
            
            color = (0, 255, 0) # Green-Default
            label = name
            
            if status.get('sleep'):
                color = (0, 0, 255) # Red for Sleeping
                label = f"{name} | SLEEPING"
            elif not status.get('id'):
                color = (0, 165, 255) # Orange for No ID
                label = f"{name} | NO ID"
                
            cv2.rectangle(display_frame, (l, t), (r, b), color, 2)
            cv2.rectangle(display_frame, (l, b - 20), (r, b), color, cv2.FILLED)
            cv2.putText(display_frame, label, (l + 5, b - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
        # Update the global frame for web streaming
        with frame_lock:
            latest_processed_frame = display_frame
            
        # hit 'q' via terminal input not needed for web, but we keep the loop logic
        if not is_session_active():
            break
        time.sleep(0.01)

    thread_running = False; ai_t.join(timeout=2.0)
    cap.release()
    with frame_lock:
        latest_processed_frame = None

def process_uploaded_video_thread(filepath, session_id):
    """Processes an uploaded video for attendance."""
    from database import mark_attendance_for_session, finalize_session
    global known_face_encodings_global, known_face_names_global
    if not known_face_encodings_global:
        known_face_encodings_global, known_face_names_global = load_encodings()
    if not known_face_encodings_global:
        if os.path.exists(filepath): os.remove(filepath); return

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        if os.path.exists(filepath): os.remove(filepath); return
    
    # Calculate video duration for accurate session logging
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    
    # Analyze one frame every 3 seconds of video time
    skip_interval_seconds = 3
    skip_frames = int(fps * skip_interval_seconds)
    count = 0
    
    print(f"Analyzing uploaded video for Session {session_id} ({int(duration_seconds)}s duration)...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        count += 1
        if count % skip_frames != 0: continue
        
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = []
        if face_detector:
            res = face_detector(small, verbose=False)
            for box in res[0].boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                locs.append((int(y1*2), int(x2*2), int(y2*2), int(x1*2)))
        
        if locs:
            encs = face_recognition.face_encodings(rgb, locs, num_jitters=1)
            for enc in encs:
                matches = face_recognition.compare_faces(known_face_encodings_global, enc, tolerance=0.55)
                dists = face_recognition.face_distance(known_face_encodings_global, enc)
                if len(dists) > 0:
                    idx = np.argmin(dists)
                    if matches[idx]:
                        # Mark attendance with the 3s increment matching our skip interval
                        mark_attendance_for_session(known_face_names_global[idx], session_id, increment=skip_interval_seconds)
    
    cap.release()
    
    # Update the session with an accurate end time based on the video duration
    # This ensures 75% attendance logic works correctly for historical sessions
    from datetime import timedelta
    try:
        from database import get_db_connection
        conn = get_db_connection()
        row = conn.execute("SELECT start_time FROM Sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row:
            start_dt = datetime.strptime(row['start_time'], '%H:%M:%S')
            end_dt = start_dt + timedelta(seconds=duration_seconds)
            finalize_session(session_id, end_dt.strftime('%H:%M:%S'))
        conn.close()
    except Exception as e:
        print(f"Error finalizing session: {e}")

    if os.path.exists(filepath): os.remove(filepath)
    print(f"Analysis complete for Session {session_id}.")
