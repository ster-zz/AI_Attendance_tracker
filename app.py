from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import os
from config import SECRET_KEY, DEBUG
from database import init_db, create_session, end_session, get_active_session, is_session_active, auto_recover_sessions, sync_students_from_images, create_historical_session
from models.face_recognition_module import encode_known_faces, start_face_recognition, process_uploaded_video_thread
from datetime import datetime
import threading

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Configure file uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit

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
def get_dashboard_context(active_page='dashboard'):
    """
    Helper function to gather all database context needed to render index.html
    """
    from database import get_dashboard_kpis, get_recent_activity, get_daily_attendance_log
    
    active_session_row = get_active_session()
    
    session_data = None
    if active_session_row:
        duration = get_duration(active_session_row['start_time'])
        session_data = {
            "session_id": active_session_row['session_id'],
            "date": active_session_row['date'],
            "start_time": active_session_row['start_time'],
            "duration": duration
        }
    
    kpis = get_dashboard_kpis()
    recent_activity = get_recent_activity(limit=4)
    daily_attendance = get_daily_attendance_log()
    
    return {
        "session": session_data,
        "kpis": kpis,
        "recent_activity": recent_activity,
        "daily_attendance": daily_attendance,
        "active_page": active_page
    }

@app.route('/')
def index():
    """
    Home route.
    """
    context = get_dashboard_context(active_page='dashboard')
    return render_template('index.html', **context)

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

@app.route('/handle_video_upload', methods=['POST'])
def handle_video_upload():
    """
    Handles class recording uploads securely, creates a historical session,
    and spins up a background thread to process the video so the server doesn't freeze.
    """
    if 'video' not in request.files:
        flash('No video file selected', 'error')
        return redirect(url_for('video_upload_page'))
        
    file = request.files['video']
    if file.filename == '':
        flash('No video file selected', 'error')
        return redirect(url_for('video_upload_page'))
        
    date_str = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
    subject_str = request.form.get('subject', 'Untitled Class')
    start_time_str = datetime.now().strftime('%H:%M:%S')
    
    if file:
        filename = secure_filename(file.filename)
        # Add timestamp to prevent overwriting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_name = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        
        try:
            file.save(filepath)
            
            # Create a DB historical session
            session_id = create_historical_session(date_str, start_time_str, subject=subject_str)
            
            # Start background processing thread
            processor_thread = threading.Thread(
                target=process_uploaded_video_thread, 
                args=(filepath, session_id)
            )
            processor_thread.daemon = True
            processor_thread.start()
            
            flash('Video uploaded successfully! The AI is now processing the attendance in the background.', 'success')
        except Exception as e:
            flash(f'Error saving or processing file: {e}', 'error')
            
    return redirect(url_for('video_upload_page'))

@app.route('/live')
def live_camera_page():
    context = get_dashboard_context(active_page='live')
    return render_template('index.html', **context)

@app.route('/upload')
def video_upload_page():
    context = get_dashboard_context(active_page='upload')
    return render_template('index.html', **context)

@app.route('/students')
def student_directory_page():
    context = get_dashboard_context(active_page='students')
    return render_template('index.html', **context)

@app.route('/reports')
def reports_page():
    context = get_dashboard_context(active_page='reports')
    return render_template('index.html', **context)

@app.route('/settings')
def settings_page():
    context = get_dashboard_context(active_page='settings')
    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(debug=DEBUG)
