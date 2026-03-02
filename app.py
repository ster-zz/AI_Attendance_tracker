from flask import Flask, render_template, request, redirect, flash, url_for
from config import SECRET_KEY, DEBUG
from database import init_db, create_session, end_session, get_active_session, is_session_active, auto_recover_sessions, sync_students_from_images
from models.face_recognition_module import encode_known_faces, start_face_recognition
from datetime import datetime
import threading

app = Flask(__name__)
app.secret_key = SECRET_KEY

# ----------------- RECOVERY & STARTUP LOGIC -----------------
with app.app_context():
    # 1. Initialize schema and database
    init_db()
    
    # 2. Auto-recover any orphaned sessions left from a server crash
    now_str = datetime.now().strftime('%H:%M:%S')
    auto_recover_sessions(now_str)
    
    # 3. Auto-sync students from the data/student_images/ folder
    sync_students_from_images()
    
    # 4. Pre-load Face Encodings (Encode images in data/student_images automatically to data/encodings.pkl)
    encode_known_faces()
    print("Session recovery and face encoding complete. System ready.")

# ----------------- HELPER FUNCTIONS -----------------
def get_duration(start_time_str):
    """
    Helper function to calculate session duration safely.
    """
    try:
        now = datetime.now()
        start_dt = datetime.strptime(start_time_str, '%H:%M:%S').replace(
            year=now.year, month=now.month, day=now.day
        )
        duration_sec = (now - start_dt).total_seconds()
        
        minutes, seconds = divmod(int(duration_sec), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        return f"{minutes}m {seconds}s"
    except Exception:
        return "Unknown"

# ----------------- ROUTES -----------------
@app.route('/')
def index():
    """
    Home route.
    Retrieves the actual active session strictly from the database.
    """
    active_session_row = get_active_session()
    
    session_data = None
    if active_session_row:
        # Calculate dynamic duration
        duration = get_duration(active_session_row['start_time'])
        
        session_data = {
            "session_id": active_session_row['session_id'],
            "date": active_session_row['date'],
            "start_time": active_session_row['start_time'],
            "duration": duration
        }
        
    return render_template('index.html', session=session_data)

@app.route('/start_session', methods=['POST'])
def handle_start_session():
    """
    Validates and starts a session. Uses Flash messages.
    """
    if is_session_active():
        flash('An active session is already running. Please end it first.', 'error')
        return redirect(url_for('index'))

    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    start_time_str = now.strftime('%H:%M:%S')
    
    session_id = create_session(date_str, start_time_str)
    
    if session_id:
        flash(f'Session {session_id} started successfully at {start_time_str}.', 'success')
    else:
        # Failsafe error
        flash('Failed to start session. Database constraint blocked creation.', 'error')
        
    return redirect(url_for('index'))

@app.route('/end_session', methods=['POST'])
def handle_end_session():
    """
    Validates and ends the current active session.
    """
    active_session_row = get_active_session()
    
    if not active_session_row:
        flash('No active session found to end.', 'error')
        return redirect(url_for('index'))

    session_id = active_session_row['session_id']
    end_time_str = datetime.now().strftime('%H:%M:%S')
    
    success = end_session(session_id, end_time_str)
    
    if success:
        flash(f'Session {session_id} ended safely.', 'success')
    else:
        flash('Error encountered while ending the session.', 'error')
        
    return redirect(url_for('index'))

@app.route('/start_recognition', methods=['POST'])
def start_recognition_api():
    """
    Triggers the live face recognition module.
    Only allows starting if a session is currently active.
    """
    if not is_session_active():
        flash('Face recognition can only be started when a session is active.', 'warning')
        return redirect(url_for('index'))
    
    # We must run OpenCV loops in a separate thread so it doesn't block Flask from serving the website
    recognition_thread = threading.Thread(target=start_face_recognition)
    recognition_thread.daemon = True # Allows thread to close when main script exits
    recognition_thread.start()
    
    flash('Live Face Recognition started successfully. Open OpenCV window to monitor.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=DEBUG)
