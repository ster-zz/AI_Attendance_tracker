import sqlite3
import os
from config import DATABASE_PATH

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

