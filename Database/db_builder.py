from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "Database" / "incendios_euskadi.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Weather observations table (Euskalmet live feed + calculated proxy)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            municipality TEXT NOT NULL,
            station_id TEXT,
            temperature REAL,
            humidity REAL,
            wind_speed REAL,
            precip_7d REAL,
            fwi_proxy REAL
        )
    """
    )

    # Risk predictions table (Random Forest model output)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            municipality TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            probability REAL NOT NULL,
            weather_obs_id INTEGER,
            FOREIGN KEY(weather_obs_id) REFERENCES weather_observations(id)
        )
    """
    )

    conn.commit()
    conn.close()
    print(f"✅ Database initialized successfully at: {DB_PATH}")


if __name__ == "__main__":
    init_db()