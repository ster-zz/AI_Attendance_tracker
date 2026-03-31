import cv2
import numpy as np
import os
import datetime

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ID_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'id_card_model.pt')
INCIDENT_IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'incident_images')

# Ensure incident images directory exists
os.makedirs(INCIDENT_IMAGES_DIR, exist_ok=True)

# ─── Try to load a custom-trained YOLOv8 ID card model ───────────────────────
# If a custom model (id_card_model.pt) is placed in the models/ folder,
# it will be used for accurate detection.
# Otherwise, we fall back to contour-based geometric detection.

_custom_model = None

if os.path.exists(ID_MODEL_PATH):
    try:
        from ultralytics import YOLO
        _custom_model = YOLO(ID_MODEL_PATH)
        print("ID Card Detection: Custom YOLOv8 model loaded.")
    except Exception as e:
        print(f"ID Card Detection: Failed to load custom model ({e}). Using contour fallback.")
else:
    print("ID Card Detection: No custom model found. Using contour-based detection fallback.")


# ─── Core Detection Function ──────────────────────────────────────────────────

def detect_id_card(frame: np.ndarray, face_bbox: tuple) -> bool:
    """
    Detects whether an ID card is visible in the frame near the student's chest.

    Strategy:
      1. If a custom YOLOv8 model is available  → use it (most accurate).
      2. Otherwise → crop the chest region below the face and use
         contour-based rectangle detection (practical fallback for demos).

    Args:
        frame     : Full BGR frame from the webcam.
        face_bbox : (top, right, bottom, left) — face_recognition CSS format.

    Returns:
        True  — ID card detected.
        False — No ID card detected (incident should be logged).
    """
    top, right, bottom, left = face_bbox
    face_height = bottom - top

    if _custom_model is not None:
        return _detect_with_yolo(frame, face_bbox)
    else:
        return _detect_with_contours(frame, top, right, bottom, left, face_height)


_last_yolo_frame_id = None
_last_yolo_results = None

def _detect_with_yolo(frame: np.ndarray, face_bbox: tuple) -> bool:
    """
    Uses the custom-trained YOLOv8 model to detect ID cards.
    Rule 1: If an object labeled 'id-card' AND 'tag' are both found under/near the face, return True.
    Rule 2: If only one is found under/near the face, it must have a confidence >= 50% (0.50).
    """
    global _last_yolo_frame_id, _last_yolo_results
    
    # Extract bounding box to assign IDs to the specific student
    top, right, bottom, left = face_bbox
    face_centerX = (left + right) / 2
    face_centerY = (top + bottom) / 2
    face_width = right - left
    
    # O(1) Performance Cache: Only run raw YOLOv8 inference ONCE per video frame, 
    # even if there are N students being checked in that single frame.
    if id(frame) != _last_yolo_frame_id:
        _last_yolo_results = _custom_model(frame, conf=0.1, verbose=False)
        _last_yolo_frame_id = id(frame)
        
    results = _last_yolo_results
    
    found_id_card = False
    found_tag = False
    high_conf_found = False

    for box in results[0].boxes:
        # Prevent stealing IDs from a neighboring student
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_centerX = (x1 + x2) / 2
        box_centerY = (y1 + y2) / 2
        
        # 1. The ID/Tag must physically be located below the center of the student's face
        if box_centerY < face_centerY:
            continue
            
        # 2. The ID/Tag must be horizontally aligned with the student's body
        # (Allows 1.5x face width margin for skewed sitting or tilted lanyards)
        if abs(box_centerX - face_centerX) > face_width * 1.5:
            continue

        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = _custom_model.names[class_id].lower()
        
        # Check if this box is an 'id-card'
        if 'id' in label or 'card' in label:
            found_id_card = True
            if conf >= 0.50:
                high_conf_found = True
                
        # Check if this box is a 'tag'
        elif 'tag' in label:
            found_tag = True
            if conf >= 0.50:
                high_conf_found = True

    # Check Rule 1: Both found
    if found_id_card and found_tag:
        return True
        
    # Check Rule 2: At least one found with >= 50% confidence
    if high_conf_found:
        return True
        
    return False


def _detect_with_contours(frame, top, right, bottom, left, face_height) -> bool:
    """
    Contour-based fallback:
    1. Crops a region below the face (chest area) — where an ID card would hang.
    2. Converts to grayscale → blurs → edge detection → finds contours.
    3. Checks if any contour has an aspect ratio matching a card (Horizontal or Vertical).
    4. Additionally checks for a lanyard color (blue/red/yellow/orange) above/on the card.
    """
    frame_h, frame_w = frame.shape[:2]

    # ── 1. Define chest search region ─────────────────────────────────────────
    # Expanded deep search (3.0x face height) to catch vertical or low-hanging cards
    chest_top    = min(bottom + int(face_height * 0.1), frame_h)
    chest_bottom = min(bottom + int(face_height * 3.0), frame_h)

    # Allow generous margins for students moving or angled cards
    margin = int((right - left) * 0.5)
    chest_left  = max(left  - margin, 0)
    chest_right = min(right + margin, frame_w)

    if chest_bottom <= chest_top or chest_right <= chest_left:
        return False

    roi = frame[chest_top:chest_bottom, chest_left:chest_right]

    # ── 2. Image Processing Pipeline ──────────────────────────────────────────
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = roi.shape[0] * roi.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter noise — cards must be ≥ 1.0% of the ROI area (loosened for deep crops)
        if area < roi_area * 0.01:
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) < 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if h == 0: continue

        aspect_ratio = float(w) / float(h)

        # Check for ID card aspect ratios:
        # 1. Horizontal: 1.2–2.2
        # 2. Vertical  : 0.4–1.0 (Refined for tags and vertical cards)
        is_card_shape = (1.2 <= aspect_ratio <= 2.2) or (0.4 <= aspect_ratio <= 1.0)

        if is_card_shape:
            # ── 3. Dual-Layer Color Verification (Lanyard/Tag) ────────────────
            ya1 = max(y - 30, 0); yb2 = y + int(h * 0.15)
            check_roi = roi[ya1:yb2, x:x+w]
            if check_roi.size > 0:
                hsv = cv2.cvtColor(check_roi, cv2.COLOR_BGR2HSV)
                masks = [
                    cv2.inRange(hsv, (100, 50, 50), (135, 255, 255)), # Blue
                    cv2.inRange(hsv, (0,   50, 50), (10,  255, 255)), # Red Low
                    cv2.inRange(hsv, (165, 50, 50), (180, 255, 255)), # Red High
                    cv2.inRange(hsv, (20,  40, 40), (42,  255, 255)), # Yellow/Gold
                    cv2.inRange(hsv, (2,   40, 40), (28,  255, 255)), # Orange/Amber (Much Wider)
                ]
                for mask in masks:
                    if cv2.countNonZero(mask) > check_roi.size * 0.05:
                        return True

            if (1.5 <= aspect_ratio <= 1.7) or (0.55 <= aspect_ratio <= 0.75):
                return True

    # ── 4. COLOR-BLOB FALLBACK (High Reliability) ─────────────────────────────
    # If no rectangular card shape was found, scan the WHOLE chest ROI for 
    # a significant "compliance color" blob.
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Target: High-vibrancy Orange or Blue (ignores skin/clothes)
    masks = [
        # Orange: Tightened hue (5-22) and high saturation (80+)
        cv2.inRange(hsv_roi, (5, 80, 80), (22, 255, 255)),
        # Blue: High saturation (90+)
        cv2.inRange(hsv_roi, (100, 90, 80), (135, 255, 255)),
    ]
    for mask in masks:
        # Require a more significant blob (4% of ROI) to avoid noise
        if cv2.countNonZero(mask) > roi_area * 0.04:
            return True

    return False


# ─── Incident Image Saving ───────────────────────────────────────────────────

def save_incident_image(frame: np.ndarray, student_name: str, incident_type: str) -> str:
    """
    Saves the current frame to static/incident_images/ as evidence.

    Args:
        frame         : BGR frame from the webcam.
        student_name  : Name of the student involved.
        incident_type : Short label like 'no_id_card'.

    Returns:
        Relative path (from project root) to the saved image, e.g.:
        'static/incident_images/John_Doe_no_id_card_20260302_203015.jpg'
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = student_name.replace(' ', '_')
    filename  = f"{safe_name}_{incident_type}_{timestamp}.jpg"
    full_path = os.path.join(INCIDENT_IMAGES_DIR, filename)

    cv2.imwrite(full_path, frame)
    print(f"Incident image saved: {filename}")

    # Return a web-friendly relative path for DB storage
    return f"static/incident_images/{filename}"
