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
    Initializes the database schema with Phase 2 Updates.
    Adds the `active` boolean column to the Sessions table.
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

    # Table: Sessions (Upgraded for Phase 2 with `active` column)
    # Adding active default 0 (false)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            active INTEGER DEFAULT 0
        )
    ''')

    # Migration step just in case Phase 1 database already exists:
    # Try adding the 'active' column safely if it doesn't exist
    try:
        cursor.execute('ALTER TABLE Sessions ADD COLUMN active INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        # Ignore if the column already exists
        pass

    # Table: Attendance
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES Sessions(session_id),
            FOREIGN KEY(student_id) REFERENCES Students(id)
        )
    ''')

    # Table: Incidents
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
    print("Database initialized successfully with Phase 2 schemas.")

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

def create_historical_session(date_str, start_time_str, end_time_str=None, subject=None):
    """
    Creates a completed session directly (active=0).
    Used for video uploads where the class already happened.
    We hijack the end_time field to store the Subject if provided natively 
    since we don't want to alter DB schema right now.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Sessions (date, start_time, end_time, active)
        VALUES (?, ?, ?, 0)
    ''', (date_str, start_time_str, subject if subject else end_time_str))
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

def mark_attendance_for_session(name, session_id):
    """
    Marks the student "Present" for a specific historical session.
    Used by the video upload background thread.
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
    
    # Check if already marked
    cursor.execute('''
        SELECT id FROM Attendance 
        WHERE session_id = ? AND student_id = ?
    ''', (session_id, student_id))
    
    if cursor.fetchone() is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO Attendance (session_id, student_id, status, reason, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, student_id, "Present", "Video Analysis", timestamp))
        conn.commit()
        
    conn.close()
    return True

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
    Checks to ensure no multiple sessions are active concurrently.
    Sets active = 1.
    Returns: The auto-generated session_id or None if an active session already exists.
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

def end_session(session_id, end_time_str):
    """
    Ends a specific session securely.
    Updates the end_time and sets active = 0.
    Returns: Boolean True if successful, False otherwise.
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
    Session Recovery Logic:
    Called on app startup. If the server crashed during an active session,
    we find all sessions where active = 1 and safely close them using the server startup time.
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

def sync_students_from_images():
    """
    Scans the data/student_images/ directory and auto-adds any students.
    Folder names inside this directory are used as the exact student name.
    Example: data/student_images/John Doe/img1.jpg -> "John Doe"
    """
    from config import BASE_DIR
    import os
    
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
        print(f"Auto-synced {new_students_added} new student(s) into the database.")
        
    conn.close()

# ----------------- DASHBOARD UI STATS FUNCTIONS -----------------

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
    cursor.execute('SELECT session_id FROM Sessions WHERE active = 1 ORDER BY session_id DESC LIMIT 1')
    active_session_row = cursor.fetchone()
    
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
    
    # 2. Present in the ACTIVE session
    cursor.execute('''
        SELECT COUNT(DISTINCT a.student_id) as present
        FROM Attendance a
        WHERE a.session_id = ? AND a.status = 'Present'
    ''', (active_session_id,))
    present_today = cursor.fetchone()['present'] or 0
    
    # 3. Absent in the ACTIVE session
    absent_today = total_students - present_today
    
    # Calculate percentage safely
    if total_students > 0:
        present_percentage = int((present_today / total_students) * 100)
        absent_percentage = int((absent_today / total_students) * 100)
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
    
    cursor.execute('SELECT session_id FROM Sessions WHERE active = 1 ORDER BY session_id DESC LIMIT 1')
    active_session_row = cursor.fetchone()
    
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

def get_daily_attendance_log():
    """
    Fetches all attendance records for the active session.
    If no session is active, returns an empty list.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT session_id FROM Sessions WHERE active = 1 ORDER BY session_id DESC LIMIT 1')
    active_session_row = cursor.fetchone()
    
    if not active_session_row:
        conn.close()
        return []
        
    active_session_id = active_session_row['session_id']
    
    # Group by student_id to show only the first time they were marked today, or just list all logs.
    cursor.execute('''
        SELECT st.id as student_id, st.name, MIN(a.timestamp) as time_in, a.status
        FROM Attendance a
        JOIN Students st ON a.student_id = st.id
        WHERE a.session_id = ?
        GROUP BY st.id
        ORDER BY time_in DESC
    ''', (active_session_id,))
    
    rows = cursor.fetchall()
    
    # Also fetch all students completely missing so we can list 'Absent' in the table clearly
    cursor.execute('SELECT id, name FROM Students')
    all_students = cursor.fetchall()
    conn.close()
    
    # Convert tracked students into a lookup set
    tracked_students = {row['student_id']: row for row in rows}
    
    attendance_log = []
    
    # Loop all students: if they have a log today, show it. Otherwise, show Absent.
    for student in all_students:
        s_id = student['id']
        s_name = student['name']
        
        if s_id in tracked_students:
            record = tracked_students[s_id]
            attendance_log.append({
                "student_id": s_id,
                "name": s_name,
                "time_in": record['time_in'],
                "status": record['status'] # Usually 'Present'
            })
        else:
            attendance_log.append({
                "student_id": s_id,
                "name": s_name,
                "time_in": "-- : --",
                "status": "Absent"
            })
            
    # Sort alphabetically or 'Present' first
    attendance_log.sort(key=lambda x: (x['status'] == 'Absent', x['name']))
    return attendance_log
