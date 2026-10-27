import sqlite3
import sys
from pathlib import Path

# Set up project root path BEFORE importing from Data
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Now import project modules
from Data.Constants.constants import DB_PATH

with sqlite3.connect(str(DB_PATH)) as conn:
    # ... rest of your script ...
    # Database/db_builder.py
    try:
        conn.execute("ALTER TABLE weather_observations ADD COLUMN solar_radiation REAL;")
    except sqlite3.OperationalError:
        pass  # Column already exists
def init_db():
    """Initializes database connection and verifies schema tables."""
    pass