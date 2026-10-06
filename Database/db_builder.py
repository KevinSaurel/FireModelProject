# Database/db_builder.py
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from Data.Constants.constants import DB_PATH


def init_db():
    # Asegurar que el directorio Database existe
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # 1. Tabla para almacenar las mediciones meteorológicas de la API
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS mediciones_clima (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TEXT NOT NULL,
        municipio TEXT NOT NULL,
        temperatura REAL,
        humedad REAL,
        viento REAL,
        precipitacion_7d REAL,
        UNIQUE(fecha, municipio)
    )
    """
    )

    # 2. Tabla para almacenar los resultados del modelo de Machine Learning
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS predicciones_riesgo (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TEXT NOT NULL,
        municipio TEXT NOT NULL,
        nivel_riesgo_num INTEGER,   -- 0: Bajo, 1: Medio, 2: Alto, 3: Extremo
        nivel_riesgo_label TEXT,    -- 'Bajo', 'Medio', 'Alto', 'Extremo'
        probabilidad REAL,          -- Confianza de la predicción (ej: 0.87)
        UNIQUE(fecha, municipio)
    )
    """
    )

    conn.commit()
    conn.close()
    print(
        f"✅ Base de datos '{DB_PATH.name}' creada e inicializada correctamente."
    )


if __name__ == "__main__":
    init_db()