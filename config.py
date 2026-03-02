import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Database configuration
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'classroom.db')

# Flask configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'default_secret_key_change_in_production')
DEBUG = True

# Session settings
SESSION_TIMEOUT = 3600  # Placeholder for session timeout in seconds (1 hour)
