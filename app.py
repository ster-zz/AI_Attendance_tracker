from flask import Flask, render_template, request, redirect, flash, url_for, Response
from werkzeug.utils import secure_filename
import os
from config import SECRET_KEY, DEBUG
from database import (
    init_db, create_session, end_session, get_active_session, is_session_active, 
    auto_recover_sessions, sync_students_from_images, create_historical_session,
    get_all_students, get_all_sessions, get_all_attendance, get_recent_incidents
)
from models.face_recognition_module import (
    encode_known_faces, start_face_recognition, process_uploaded_video_thread, gen_frames
)
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
    
    # 4. Pre-load Face Encodings
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
    
    context = {
        "session": session_data,
        "kpis": kpis,
        "recent_activity": recent_activity,
        "daily_attendance": daily_attendance,
        "active_page": active_page
    }

    # Add extra context for specific pages if needed
    if active_page == 'database':
        context['students'] = get_all_students()
        context['attendance'] = get_all_attendance()
    elif active_page == 'incidents':
        context['incidents'] = get_recent_incidents(limit=50)
    elif active_page == 'students':
        context['students'] = get_all_students()
    elif active_page == 'sessions':
        context['sessions'] = get_all_sessions()
    elif active_page == 'reports':
        from database import get_session_attendance_report
        session_id = request.args.get('session_id', type=int)
        all_sessions = get_all_sessions()
        context['all_sessions'] = all_sessions
        
        if session_id:
            # Find the specific session info
            selected_session = next((s for s in all_sessions if s['session_id'] == session_id), None)
            if selected_session:
                report = get_session_attendance_report(session_id)
                present = sum(1 for r in report if r['status'] == 'Present')
                absent = len(report) - present
                
                context['selected_session'] = selected_session
                context['report'] = report
                context['stats'] = {
                    'present': present,
                    'absent': absent,
                    'total': len(report),
                    'present_percent': int((present / len(report) * 100)) if len(report) > 0 else 0,
                    'absent_percent': int((absent / len(report) * 100)) if len(report) > 0 else 0
                }

    return context

# ----------------- ROUTES -----------------
@app.route('/')
def index():
    context = get_dashboard_context(active_page='dashboard')
    return render_template('index.html', **context)

@app.route('/start_session', methods=['POST'])
def handle_start_session():
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
        flash('Failed to start session.', 'error')
        
    return redirect(url_for('index'))

@app.route('/end_session', methods=['POST'])
def handle_end_session():
    active_session_row = get_active_session()
    if not active_session_row:
        flash('No active session found to end.', 'error')
        return redirect(url_for('index'))

    session_id = active_session_row['session_id']
    end_time_str = datetime.now().strftime('%H:%M:%S')
    
    if end_session(session_id, end_time_str):
        flash(f'Session {session_id} ended safely.', 'success')
    else:
        flash('Error encountered while ending the session.', 'error')
        
    return redirect(url_for('index'))

@app.route('/start_recognition', methods=['POST'])
def start_recognition_api():
    if not is_session_active():
        flash('Face recognition can only be started when a session is active.', 'warning')
        return redirect(url_for('index'))
    
    recognition_thread = threading.Thread(target=start_face_recognition)
    recognition_thread.daemon = True
    recognition_thread.start()
    
    flash('Live Face Recognition started successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/handle_video_upload', methods=['POST'])
def handle_video_upload():
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
        save_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        
        try:
            file.save(filepath)
            session_id = create_historical_session(date_str, start_time_str, subject=subject_str)
            
            processor_thread = threading.Thread(target=process_uploaded_video_thread, args=(filepath, session_id))
            processor_thread.daemon = True
            processor_thread.start()
            
            flash('Video uploaded successfully! The AI is now analyzing it live on the dashboard.', 'success')
            # Redirect to the main dashboard so they can watch the video processing live
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'Error: {e}', 'error')
            
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

@app.route('/sessions')
def view_sessions():
    context = get_dashboard_context(active_page='sessions')
    return render_template('index.html', **context)

@app.route('/incidents')
def view_incidents():
    context = get_dashboard_context(active_page='incidents')
    return render_template('index.html', **context)

@app.route('/database')
def view_database():
    context = get_dashboard_context(active_page='database')
    return render_template('index.html', **context)

@app.route('/reports')
def reports_page():
    context = get_dashboard_context(active_page='reports')
    return render_template('index.html', **context)

@app.route('/settings')
def settings_page():
    context = get_dashboard_context(active_page='settings')
    return render_template('index.html', **context)

@app.route('/sync_database', methods=['POST'])
def sync_database():
    """Manually triggers a sync of the student_images folder to the database."""
    sync_students_from_images()
    encode_known_faces()
    flash('Database and Encodings synced successfully with /data/student_images/', 'success')
    return redirect(request.referrer or url_for('index'))

if __name__ == '__main__':
    app.run(debug=DEBUG)
