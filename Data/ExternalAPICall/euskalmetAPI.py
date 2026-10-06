import datetime
import sqlite3
import sys
import time
from pathlib import Path
import jwt
import pandas as pd
import requests

# Configuración del directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from Data.Constants.constants import (
    API_URL,
    DB_PATH,
    JWT_AUD,
    JWT_ISS,
    JWT_VERSION,
    PRIVATE_KEY_PATH,
    USER_EMAIL,
)
from Database.db_builder import init_db


def generate_jwt() -> str:
    """Genera el token JWT firmado con RS256 para Open Data Euskadi."""
    if not PRIVATE_KEY_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la clave privada en: {PRIVATE_KEY_PATH}"
        )

    with open(PRIVATE_KEY_PATH, "r", encoding="utf-8") as f:
        private_key = f.read()

    now = int(time.time())
    payload = {
        "aud": JWT_AUD,
        "iss": JWT_ISS,
        "version": JWT_VERSION,
        "email": USER_EMAIL,
        "iat": now,
        "exp": now + 3600,
    }

    return jwt.encode(payload, private_key, algorithm="RS256")


def fetch_live_weather() -> pd.DataFrame | None:
    """Obtiene las estaciones meteorológicas y sus lecturas de Open Data Euskadi."""
    try:
        token = generate_jwt()
    except Exception as e:
        print(f"❌ Error al generar JWT: {e}")
        return None

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    print(f"📡 Conectando a Open Data Euskadi ({API_URL})...")

    try:
        res = requests.get(API_URL, headers=headers, timeout=12)
        res.raise_for_status()
        stations_data = res.json()

        station_ids = sorted(
            {
                s["stationId"]
                for s in stations_data
                if isinstance(s, dict) and "stationId" in s
            }
        )
        print(f"🔍 Se han localizado {len(station_ids)} estaciones en el catálogo.")

        records = []
        fecha_hoy = datetime.date.today().isoformat()

        for station_id in station_ids:
            station_url = f"{API_URL}/{station_id}/current"
            try:
                st_res = requests.get(station_url, headers=headers, timeout=5)
                if st_res.status_code != 200:
                    continue

                st_data = st_res.json()

                if st_data.get("stationType") != "METEOROLOGICAL":
                    continue

                muni_info = st_data.get("municipality", {})
                municipio = (
                    muni_info.get("SPANISH")
                    if isinstance(muni_info, dict)
                    else None
                )
                if not municipio:
                    name_info = st_data.get("name", {})
                    municipio = (
                        name_info.get("SPANISH")
                        if isinstance(name_info, dict)
                        else "Desconocido"
                    )

                readings_url = f"{API_URL}/{station_id}/readings/latest"
                rd_res = requests.get(readings_url, headers=headers, timeout=5)

                temp, humedad, viento, precip = 20.0, 50.0, 10.0, 0.0

                if rd_res.status_code == 200:
                    rd_data = rd_res.json()
                    measures = (
                        rd_data.get("measures", {})
                        if isinstance(rd_data, dict)
                        else {}
                    )
                    temp = measures.get("temperature", temp)
                    humedad = measures.get("humidity", humedad)
                    viento = measures.get("windSpeed", viento)
                    precip = measures.get("precipitation", precip)

                records.append(
                    {
                        "fecha": fecha_hoy,
                        "municipio": str(municipio).strip(),
                        "temperatura": float(temp),
                        "humedad": float(humedad),
                        "viento": float(viento),
                        "precipitacion_7d": float(precip),
                    }
                )

            except Exception:
                continue

        df = pd.DataFrame(records)
        print(f"📊 Procesados {len(df)} registros meteorológicos de municipios.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Error HTTP en la API: {e}")
        return None


def save_to_sqlite(df: pd.DataFrame | None) -> None:
    """Guarda los registros procesados en la base de datos SQLite."""
    if df is None or df.empty:
        print("⚠️ No hay datos para guardar.")
        return

    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    inserted_count = 0

    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT OR IGNORE INTO mediciones_clima 
            (fecha, municipio, temperatura, humedad, viento, precipitacion_7d)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["fecha"],
                row["municipio"],
                row["temperatura"],
                row["humedad"],
                row["viento"],
                row["precipitacion_7d"],
            ),
        )
        if cursor.rowcount > 0:
            inserted_count += 1

    conn.commit()
    conn.close()
    print(f"✅ Ingesta completada: {inserted_count} nuevos registros guardados en SQLite.")


if __name__ == "__main__":
    df_clima = fetch_live_weather()
    save_to_sqlite(df_clima)