# AI Classroom System — Bugs & Fixes Implemented

> This document is a living reference. Each section captures a confirmed bug, its root cause, the fix applied, and the outcome. New entries are added at the bottom.

---

## Table of Contents

1. [Camera Not Stopping When Session Ends](#1-camera-not-stopping-when-session-ends)

---

## 1. Camera Not Stopping When Session Ends

- **Date Identified**: 2026-03-01
- **Reported By**: User
- **Status**: ✅ Fixed

### 1.1 Bug Description
When the "End Session" button was clicked on the web UI, the database correctly updated the session to `active = 0`. However, the OpenCV webcam window remained open and continued running indefinitely. The camera had to be manually closed by pressing `q` on the keyboard.

### 1.2 Root Cause
The live recognition loop in `models/face_recognition_module.py` used an unconditional `while True:` loop with the only exit condition being a keypress (`'q'`):

```python
# BEFORE (Broken)
while True:
    ...
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

The loop had **no awareness of the session state**. Even though the Flask route correctly ended the session in SQLite, the background thread running the camera loop was completely disconnected from that state change.

### 1.3 Fix Applied
Changed the loop condition from `while True:` to `while is_session_active():`, which polls the database on every iteration:

```python
# AFTER (Fixed)
while is_session_active():
    ...
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

- **File Modified**: `models/face_recognition_module.py` — Line 150
- **Function Modified**: `start_face_recognition()`

### 1.4 How It Works Now
1. User clicks **"End Session"** in the web UI.
2. Flask sets `active = 0` in the `Sessions` table.
3. On the very next loop cycle, `is_session_active()` queries the DB and returns `False`.
4. The `while` condition fails → loop exits → `video_capture.release()` and `cv2.destroyAllWindows()` are called automatically.
5. The camera window closes cleanly with no user interaction required.

### 1.5 Outcome
- Camera now shuts down automatically the moment the session is ended via the UI.
- No manual `'q'` keypress required.
- The `'q'` shortcut still works as a manual override for early exit.

---

*Add new bug/fix entries below this line as they arise.*
