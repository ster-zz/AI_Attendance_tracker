import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye Landmark Indices (MediaPipe Face Mesh Refined Landmarks)
# Left Eye
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Right Eye
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Specific EAR indices (Vertical and Horizontal points)
L_V1, L_V2 = 386, 374
L_V3, L_V4 = 385, 373
L_H1, L_H2 = 362, 263

R_V1, R_V2 = 159, 145
R_V3, R_V4 = 158, 144
R_H1, R_H2 = 33, 133

# Thresholds
EAR_THRESHOLD = 0.26
SLEEP_DURATION_THRESHOLD = 1.5  # seconds

# Global state for tracking eye closure duration per student
_sleep_states = {} # {student_name: {'start_time': timestamp, 'is_closed': bool}}

def calculate_ear(landmarks, v1, v2, v3, v4, h1, h2):
    """
    Calculates the Eye Aspect Ratio (EAR).
    """
    def distance(p1_idx, p2_idx):
        p1 = landmarks[p1_idx]
        p2 = landmarks[p2_idx]
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    # Vertical distances
    v_dist1 = distance(v1, v2)
    v_dist2 = distance(v3, v4)
    # Horizontal distance
    h_dist = distance(h1, h2)

    ear = (v_dist1 + v_dist2) / (2.0 * h_dist)
    return ear

_last_mesh_frame_id = None
_last_mesh_results = None

def check_sleep(frame, face_bbox_recognition, student_name, current_time=None):
    """
    Checks if the student is sleeping based on EAR.
    
    Args:
        frame: The full BGR frame.
        face_bbox_recognition: (top, right, bottom, left) from face_recognition.
        student_name: Name of the identified student.
        current_time: Simulated timeline timestamp for video parsing.
        
    Returns:
        is_sleeping (bool): True if student has been closed-eyed for > threshold.
    """
    global _sleep_states, _last_mesh_frame_id, _last_mesh_results
    
    # O(1) Performance Cache: Only run heavy MediaPipe face meshing ONCE per 
    # video frame, even if there are N students being checked in that frame.
    if id(frame) != _last_mesh_frame_id:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _last_mesh_results = face_mesh.process(rgb_frame)
        _last_mesh_frame_id = id(frame)
        
    results = _last_mesh_results
    
    if not results.multi_face_landmarks:
        return False

    # Find the mesh landmark that matches the recognition bounding box
    # Since we can have multiple people, we find the one whose landmarks center fits the box
    h, w, _ = frame.shape
    best_match_landmarks = None
    min_dist = float('inf')
    
    top, right, bottom, left = face_bbox_recognition
    box_center_x = (left + right) / 2
    box_center_y = (top + bottom) / 2

    for face_landmarks in results.multi_face_landmarks:
        # Calculate center of landmarks
        lx = np.mean([lm.x for lm in face_landmarks.landmark]) * w
        ly = np.mean([lm.y for lm in face_landmarks.landmark]) * h
        
        dist = np.sqrt((lx - box_center_x)**2 + (ly - box_center_y)**2)
        # Match based on minimum distance to the recognition box center
        if dist < min_dist:
            min_dist = dist
            best_match_landmarks = face_landmarks.landmark

    if best_match_landmarks is None:
        return False

    # Calculate EAR
    left_ear = calculate_ear(best_match_landmarks, L_V1, L_V2, L_V3, L_V4, L_H1, L_H2)
    right_ear = calculate_ear(best_match_landmarks, R_V1, R_V2, R_V3, R_V4, R_H1, R_H2)
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Logic for sleep tracking
    now = current_time if current_time is not None else time.time()
    if student_name not in _sleep_states:
        _sleep_states[student_name] = {'start_time': None, 'is_sleeping': False}

    state = _sleep_states[student_name]

    if avg_ear < EAR_THRESHOLD:
        if state['start_time'] is None:
            state['start_time'] = now
        elif (now - state['start_time']) >= SLEEP_DURATION_THRESHOLD:
            state['is_sleeping'] = True
            return True
    else:
        state['start_time'] = None
        state['is_sleeping'] = False

    return False
