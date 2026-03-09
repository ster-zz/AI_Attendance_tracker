import sqlite3
import os
from config import DATABASE_PATH
from datetime import datetime

def get_db_connection():
    """
    Establishes and returns a connection to the SQLite database.
    Row factory is set to sqlite3.Row for dict-like access.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the database schema with Phase 2 & 3 Updates.
    """
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table: Students
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_encoding BLOB
        )
    ''')

    # Table: Sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            subject TEXT,
            active INTEGER DEFAULT 0
        )
    ''')

    # Migration step just in case Phase 1 database already exists:
    # Try adding the 'active' and 'subject' columns safely if they don't exist
    try:
        cursor.execute('ALTER TABLE Sessions ADD COLUMN active INTEGER DEFAULT 0')
    except sqlite3.OperationalError: pass
    
    try:
        cursor.execute('ALTER TABLE Sessions ADD COLUMN subject TEXT')
    except sqlite3.OperationalError: pass

    # Table: Attendance
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            timestamp TEXT NOT NULL,
            presence_seconds INTEGER DEFAULT 0,
            FOREIGN KEY(session_id) REFERENCES Sessions(session_id),
            FOREIGN KEY(student_id) REFERENCES Students(id)
        )
    ''')

    # Table: Incidents (Phase 3)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            incident_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            FOREIGN KEY(session_id) REFERENCES Sessions(session_id),
            FOREIGN KEY(student_id) REFERENCES Students(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# ----------------- SESSION MANAGER FUNCTIONS -----------------

def get_active_session():
    """
    Retrieves the currently active session (active = 1), if any.
    Returns: A sqlite3.Row object containing the session data, or None.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Sessions WHERE active = 1 ORDER BY session_id DESC LIMIT 1')
    session = cursor.fetchone()
    conn.close()
    return session

def is_session_active():
    """
    Checks if there is any active session running.
    Returns: Boolean
    """
    return get_active_session() is not None

def get_current_session_id():
    """
    Retrieves the ID of the currently active session.
    Returns: session_id (int) or None if no active session.
    """
    session = get_active_session()
    return session['session_id'] if session else None

def create_session(date_str, start_time_str):
    """
    Creates a new session securely in the database.
    Sets active = 1.
    """
    if is_session_active():
        return None # Prevent creation if one is active
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Sessions (date, start_time, active)
        VALUES (?, ?, 1)
    ''', (date_str, start_time_str))
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

def create_historical_session(date_str, start_time_str, subject=None):
    """
    Creates a completed session directly (active=0).
    Used for video uploads where the class already happened.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Sessions (date, start_time, end_time, subject, active)
        VALUES (?, ?, ?, ?, 0)
    ''', (date_str, start_time_str, None, subject))
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

def finalize_session(session_id, end_time_str):
    """Updates the end time of a session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE Sessions SET end_time = ? WHERE session_id = ?
    ''', (end_time_str, session_id))
    conn.commit()
    conn.close()
    return True

def end_session(session_id, end_time_str):
    """
    Ends a specific session securely.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if this session actually exists and is active
    cursor.execute('SELECT active FROM Sessions WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    
    if not row or not row['active']:
        conn.close()
        return False
        
    cursor.execute('''
        UPDATE Sessions
        SET end_time = ?, active = 0
        WHERE session_id = ?
    ''', (end_time_str, session_id))
    conn.commit()
    conn.close()
    return True

def auto_recover_sessions(current_time_str):
    """
    Closes orphaned sessions on startup.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    # Find active sessions
    cursor.execute('SELECT session_id FROM Sessions WHERE active = 1')
    sessions = cursor.fetchall()
    
    for session in sessions:
        print(f"Recovering orphaned session: {session['session_id']}")
        cursor.execute('''
            UPDATE Sessions
            SET end_time = ?, active = 0
            WHERE session_id = ?
        ''', (f"{current_time_str} (Auto-closed)", session['session_id']))
        
    conn.commit()
    conn.close()

# ----------------- ATTENDANCE & INCIDENT LOGGING -----------------

def mark_attendance_for_session(name, session_id, increment=2):
    """
    Increments the student's presence time for a specific session.
    Default increment is 2 seconds (based on AI loop).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get student ID
    cursor.execute("SELECT id FROM Students WHERE name = ?", (name,))
    student_record = cursor.fetchone()
    
    if not student_record:
        conn.close()
        return False
        
    student_id = student_record['id']
    
    # Check if record exists
    cursor.execute('''
        SELECT id FROM Attendance 
        WHERE session_id = ? AND student_id = ?
    ''', (session_id, student_id))
    row = cursor.fetchone()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if row is None:
        cursor.execute('''
            INSERT INTO Attendance (session_id, student_id, status, reason, timestamp, presence_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, student_id, "Present", "System Detection", timestamp, increment))
    else:
        cursor.execute('''
            UPDATE Attendance 
            SET presence_seconds = presence_seconds + ?, 
                timestamp = ?
            WHERE session_id = ? AND student_id = ?
        ''', (increment, timestamp, session_id, student_id))
        
    conn.commit()
    conn.close()
    return True

def log_incident(session_id, student_id, incident_type, image_path=None):
    """
    Logs an incident (e.g., 'no_id_card').
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO Incidents (session_id, student_id, incident_type, timestamp, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, student_id, incident_type, timestamp, image_path))
        conn.commit()
        conn.close()
        print(f"Incident logged: {incident_type} for student_id={student_id} at {timestamp}")
        return True
    except Exception as e:
        print(f"Error logging incident: {e}")
        return False

# ----------------- DATA RETRIEVAL FUNCTIONS -----------------

def get_all_students():
    """
    Retrieves all registered students.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Students ORDER BY name ASC')
    students = cursor.fetchall()
    conn.close()
    return students

def get_all_sessions():
    """
    Retrieves all past and current sessions.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Sessions ORDER BY session_id DESC')
    sessions = cursor.fetchall()
    conn.close()
    return sessions

def get_all_attendance():
    """
    Retrieves the complete attendance record log, joined with 
    student names and session dates.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.id, a.session_id, a.student_id, a.status, a.reason, a.timestamp, a.presence_seconds,
               s.name AS student_name,
               sn.date AS session_date
        FROM Attendance a
        JOIN Students s ON a.student_id = s.id
        JOIN Sessions sn ON a.session_id = sn.session_id
        ORDER BY a.id DESC
    ''')
    attendance = cursor.fetchall()
    conn.close()
    return attendance

def get_recent_incidents(limit=20):
    """
    Retrieves the most recent incidents joined with student names.
    Used by the dashboard activity feed and incidents page.

    Returns:
        List of sqlite3.Row objects with: id, incident_type, timestamp,
        image_path, student name, session_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT i.id, i.session_id, i.incident_type, i.timestamp, i.image_path,
               s.name AS student_name
        FROM Incidents i
        JOIN Students s ON i.student_id = s.id
        ORDER BY i.id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def sync_students_from_images():
    """
    Scans the data/student_images/ directory and auto-adds any students.
    Folder names inside this directory are used as the exact student name.
    Example: data/student_images/John Doe/img1.jpg -> "John Doe"
    """
    from config import BASE_DIR
    
    images_dir = os.path.join(BASE_DIR, 'data', 'student_images')
    if not os.path.exists(images_dir):
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT name FROM Students')
    existing_students = set(row['name'] for row in cursor.fetchall())
    
    new_students_added = 0
    for item in os.listdir(images_dir):
        item_path = os.path.join(images_dir, item)
        # We only care about Folders now (each folder is a student)
        if os.path.isdir(item_path):
            name = item
            if name not in existing_students:
                cursor.execute('INSERT INTO Students (name) VALUES (?)', (name,))
                existing_students.add(name)
                new_students_added += 1
                
    if new_students_added > 0:
        conn.commit()
        print(f"Auto-synced {new_students_added} new students.")
        
    conn.close()

# ----------------- DASHBOARD KPI FUNCTIONS -----------------

def get_dashboard_kpis():
    """
    Calculates KPI stats for the Dashboard UI based on the ACTIVE session.
    If no session is active, returns 0 for present and total absent.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Total Students
    cursor.execute('SELECT COUNT(id) as total FROM Students')
    total_students = cursor.fetchone()['total'] or 0
    
    # Check if a session is active
    active_session_row = get_active_session()
    
    if not active_session_row:
        conn.close()
        return {
            "total": total_students,
            "present": 0,
            "absent": total_students,
            "present_percentage": 0,
            "absent_percentage": 100 if total_students > 0 else 0
        }
        
    active_session_id = active_session_row['session_id']
    
    # 2. Present in the ACTIVE session (Meeting 75% threshold)
    duration = get_session_duration_seconds(active_session_id)
    cursor.execute('''
        SELECT student_id, MAX(presence_seconds) as presence_seconds
        FROM Attendance
        WHERE session_id = ?
        GROUP BY student_id
    ''', (active_session_id,))
    
    present_today = 0
    records = cursor.fetchall()
    for rec in records:
        if rec['presence_seconds'] / duration >= 0.75:
            present_today += 1
    
    # 3. Absent in the ACTIVE session
    absent_today = total_students - present_today
    
    # Calculate percentage safely
    if total_students > 0:
        present_percentage = int((present_today / total_students) * 100)
        absent_percentage = 100 - present_percentage
    else:
        present_percentage = 0
        absent_percentage = 0
        
    conn.close()
    
    return {
        "total": total_students,
        "present": present_today,
        "absent": absent_today,
        "present_percentage": present_percentage,
        "absent_percentage": absent_percentage
    }

def get_recent_activity(limit=5):
    """
    Fetches the most recent live face detections for the currently active session.
    Used for the 'Activity Feed' panel.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    active_session_row = get_active_session()
    
    if not active_session_row:
        conn.close()
        return []
        
    cursor.execute('''
        SELECT s.name, a.timestamp, a.status, sess.date
        FROM Attendance a
        JOIN Students s ON a.student_id = s.id
        JOIN Sessions sess ON a.session_id = sess.session_id
        WHERE a.session_id = ?
        ORDER BY a.id DESC
        LIMIT ?
    ''', (active_session_row['session_id'], limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert rows to a list of dicts safely
    activity = []
    for r in rows:
        activity.append({
            "name": r["name"],
            "time": r["timestamp"], # Usually format HH:MM:SS
            "status": r["status"],
            "date": r["date"]
        })
    return activity

def get_session_duration_seconds(session_id):
    """Calculates the total duration of a session in seconds."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT start_time, end_time, active FROM Sessions WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return 0
        
    fmt = '%H:%M:%S'
    try:
        start_time = datetime.strptime(row['start_time'], fmt)
        if row['active']:
            now_time_str = datetime.now().strftime(fmt)
            end_time = datetime.strptime(now_time_str, fmt)
        else:
            # Handle potential "(Auto-closed)" suffix
            clean_end = row['end_time'].split(' ')[0] if row['end_time'] else row['start_time']
            end_time = datetime.strptime(clean_end, fmt)
            
        duration = (end_time - start_time).total_seconds()
        return max(duration, 1) # Minimum 1s to avoid div by zero
    except Exception as e:
        print(f"Error calculating duration: {e}")
        return 60 # Fallback 1 min

def get_daily_attendance_log():
    """
    Fetches all attendance records for the active session.
    Statuses are calculated based on 75% presence.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    active_session_row = get_active_session()
    if not active_session_row:
        conn.close()
        return []
        
    active_session_id = active_session_row['session_id']
    duration = get_session_duration_seconds(active_session_id)
    
    cursor.execute('''
        SELECT st.id as student_id, st.name, MIN(a.timestamp) as time_in, 
               MAX(a.presence_seconds) as presence_seconds
        FROM Attendance a
        JOIN Students st ON a.student_id = st.id
        WHERE a.session_id = ?
        GROUP BY st.id
    ''', (active_session_id,))
    
    tracked_students = {row['student_id']: row for row in cursor.fetchall()}
    
    cursor.execute('SELECT id, name FROM Students')
    all_students = cursor.fetchall()
    conn.close()
    
    attendance_log = []
    for student in all_students:
        s_id = student['id']
        s_name = student['name']
        
        if s_id in tracked_students:
            record = tracked_students[s_id]
            # 75% Threshold Logic
            presence_ratio = record['presence_seconds'] / duration
            status = "Present" if presence_ratio >= 0.75 else f"Partial ({int(presence_ratio*100)}%)"
            
            attendance_log.append({
                "student_id": s_id,
                "name": s_name,
                "time_in": record['time_in'],
                "status": status,
                "presence_per": int(presence_ratio * 100)
            })
        else:
            attendance_log.append({
                "student_id": s_id,
                "name": s_name,
                "time_in": "-- : --",
                "status": "Absent",
                "presence_per": 0
            })
            
    attendance_log.sort(key=lambda x: (x['status'] == 'Absent', x['name']))
    return attendance_log

def get_session_attendance_report(session_id):
    """
    Fetches attendance records for a specific session ID.
    Status is dynamically calculated based on 75% presence.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    duration = get_session_duration_seconds(session_id)
    
    cursor.execute('''
        SELECT st.id as student_id, st.name, MIN(a.timestamp) as time_in, 
               MAX(a.presence_seconds) as presence_seconds
        FROM Attendance a
        JOIN Students st ON a.student_id = st.id
        WHERE a.session_id = ?
        GROUP BY st.id
    ''', (session_id,))
    
    tracked_students = {row['student_id']: row for row in cursor.fetchall()}
    
    cursor.execute('SELECT id, name FROM Students')
    all_students = cursor.fetchall()
    conn.close()
    
    report = []
    for student in all_students:
        s_id = student['id']
        s_name = student['name']
        
        if s_id in tracked_students:
            record = tracked_students[s_id]
            presence_ratio = record['presence_seconds'] / duration
            status = "Present" if presence_ratio >= 0.75 else "Absent"
            
            report.append({
                "student_id": s_id,
                "name": s_name,
                "time_in": record['time_in'],
                "status": status,
                "presence_per": int(presence_ratio * 100)
            })
        else:
            report.append({
                "student_id": s_id,
                "name": s_name,
                "time_in": "-- : --",
                "status": "Absent",
                "presence_per": 0
            })
            
    report.sort(key=lambda x: (x['status'] == 'Absent', x['name']))
    return report
