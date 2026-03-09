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


def _detect_with_yolo(frame: np.ndarray, face_bbox: tuple) -> bool:
    """
    Uses the custom-trained YOLOv8 model to detect ID cards in the full frame.
    Returns True if any 'id_card' class bounding box is found.
    """
    results = _custom_model(frame, verbose=False)
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        label = _custom_model.names[class_id].lower()
        if 'id' in label or 'card' in label or 'badge' in label:
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
