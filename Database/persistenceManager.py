"""
Database Persistence Manager for FireModelProject.
Handles all read/write interactions with the SQLite database.
"""

from datetime import datetime
import logging
from pathlib import Path
import sqlite3
import sys
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from Data.Constants.constants import DB_PATH
from Database.db_builder import init_db

log = logging.getLogger("persistence_manager")
TABLE_WEATHER = "weather_observations"
TABLE_PREDICTIONS = "risk_predictions"
TABLE_DAILY_PRECIPITATION = "station_daily_precipitation"


def get_existing_weather_dates() -> set[str]:
    """Returns a set of all dates (YYYY-MM-DD) already saved in weather_observations."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        query = f"SELECT DISTINCT date FROM {TABLE_WEATHER}"
        df = pd.read_sql_query(query, conn)
        return set(df["date"].tolist())
    except Exception as e:
        log.error("Failed to fetch existing dates from database: %s", e)
        return set()
    finally:
        conn.close()


def get_cached_daily_precipitation(
    station_id: str,
    date: str,
) -> tuple[bool, float | None]:
    """Devuelve (existe, lluvia_diaria) de la caché persistente de estaciones."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_DAILY_PRECIPITATION} (
                station_id TEXT NOT NULL,
                date TEXT NOT NULL,
                precipitation REAL,
                PRIMARY KEY (station_id, date)
            )
            """
        )
        conn.commit()
        row = conn.execute(
            f"""SELECT precipitation FROM {TABLE_DAILY_PRECIPITATION}
                WHERE station_id = ? AND date = ?""",
            (station_id, date),
        ).fetchone()
        return (True, row[0]) if row else (False, None)
    finally:
        conn.close()


def save_cached_daily_precipitation(
    station_id: str,
    date: str,
    precipitation: float | None,
) -> None:
    """Guarda la lluvia diaria validada; NULL también queda cacheado como dato ausente."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_DAILY_PRECIPITATION} (
                station_id TEXT NOT NULL,
                date TEXT NOT NULL,
                precipitation REAL,
                PRIMARY KEY (station_id, date)
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO {TABLE_DAILY_PRECIPITATION} (station_id, date, precipitation)
            VALUES (?, ?, ?)
            ON CONFLICT(station_id, date) DO UPDATE SET
                precipitation = excluded.precipitation
            """,
            (station_id, date, precipitation),
        )
        conn.commit()
    finally:
        conn.close()


def save_weather_observations(df: pd.DataFrame | None) -> int:
    """Saves daily weather observations DataFrame to SQLite (upsert pattern)."""
    if df is None or df.empty:
        log.warning("No weather data provided to save.")
        return 0

    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.isolation_level = None
    written = replaced = failed = 0

    def num(v):
        return None if pd.isna(v) else float(v)

    try:
        conn.execute("BEGIN")
        for _, row in df.iterrows():
            conn.execute("SAVEPOINT row_save")
            try:
                cur = conn.execute(
                    f"DELETE FROM {TABLE_WEATHER} WHERE date = ? AND municipality = ?",
                    (row["date"], row["municipality"]),
                )
                replaced += cur.rowcount
                conn.execute(
                    f"""
                    INSERT INTO {TABLE_WEATHER}
                    (date, municipality, temperature, humidity, wind_speed, precip_7d, fwi_proxy)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["date"],
                        row["municipality"],
                        num(row["temperature"]),
                        num(row["humidity"]),
                        num(row["wind_speed"]),
                        num(row["precip_7d"]),
                        num(row["fwi_proxy"]),
                    ),
                )
                conn.execute("RELEASE row_save")
                written += 1
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK TO row_save")
                conn.execute("RELEASE row_save")
                failed += 1
                log.error("Failed to save %s | %s: %s", row["date"], row["municipality"], exc)
        conn.execute("COMMIT")
        log.info("Saved %d weather records to SQLite (%d replaced).", written, replaced)
        return written
    except Exception:
        log.exception("Transaction error writing weather observations to SQLite.")
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        return 0
    finally:
        conn.close()


def save_risk_predictions(df: pd.DataFrame | None) -> int:
    """Saves machine learning predictions to risk_predictions table (with auto-migration)."""
    if df is None or df.empty:
        log.warning("No risk predictions provided to save.")
        return 0

    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.cursor()

        # Check existing table schema
        table_info = cursor.execute(f"PRAGMA table_info({TABLE_PREDICTIONS})").fetchall()
        column_names = {col[1] for col in table_info}

        # Auto-migrate if table exists with an outdated schema missing 'date'
        if table_info and "date" not in column_names:
            log.info("Migrating '%s' table schema to include 'date' column...", TABLE_PREDICTIONS)
            cursor.execute(f"DROP TABLE {TABLE_PREDICTIONS}")
            column_names = set()

        if not column_names:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_PREDICTIONS} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    municipality TEXT NOT NULL,
                    risk_probability REAL,
                    risk_tier TEXT,
                    prediction_time TEXT,
                    UNIQUE(date, municipality)
                )
                """
            )
            conn.commit()

        latest_date = str(df["date"].iloc[0])

        # Delete previous predictions for this date to prevent UNIQUE collisions
        cursor.execute(f"DELETE FROM {TABLE_PREDICTIONS} WHERE date = ?", (latest_date,))
        conn.commit()

        save_df = df[["date", "municipality", "risk_probability", "risk_tier", "prediction_time"]]
        save_df.to_sql(TABLE_PREDICTIONS, conn, if_exists="append", index=False)
        log.info("Saved %d predictions for %s to '%s' table.", len(save_df), latest_date, TABLE_PREDICTIONS)
        return len(save_df)
    except Exception as e:
        log.error("Failed to save predictions to database: %s", e)
        return 0
    finally:
        conn.close()


def get_latest_weather_observations() -> pd.DataFrame:
    """Retrieves weather observations for the most recent date available in SQLite."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        query = f"""
            SELECT date, municipality, temperature, humidity, wind_speed, precip_7d, fwi_proxy 
            FROM {TABLE_WEATHER} 
            WHERE date = (SELECT MAX(date) FROM {TABLE_WEATHER})
        """
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()