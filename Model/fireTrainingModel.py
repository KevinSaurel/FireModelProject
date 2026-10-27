"""
Inferencia ML del modelo de incendios con viento en km/h y manejo transparente de
valores NULL ('Sin datos').
"""

from datetime import datetime
import logging
import math
from pathlib import Path
import sys
import json

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from Data.Constants.constants import MODEL_PATH

log = logging.getLogger("fire_training_model")
THRESHOLDS_PATH = MODEL_PATH.parent / "tier_thresholds.json"

def map_risk_level(
    probability: float | None,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Asigna tiers con cuantiles de positivos; conserva límites antiguos para modelos previos."""
    if probability is None or pd.isna(probability):
        return "Sin datos"
    if thresholds is not None:
        if probability < thresholds["low"]:
            return "Low"
        if probability < thresholds["medium"]:
            return "Medium"
        if probability < thresholds["high"]:
            return "High"
        return "Extreme"
    elif probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Medium"
    elif probability < 0.85:
        return "High"
    else:
        return "Extreme"


def predict_fire_risk(df_weather: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta inferencia ML en km/h (alineado con el modelo MITECO)."""
    if df_weather is None or df_weather.empty:
        log.error("❌ No se recibieron observaciones meteorológicas.")
        return pd.DataFrame()

    if not MODEL_PATH.exists():
        log.error("❌ Modelo no encontrado en: %s", MODEL_PATH)
        return pd.DataFrame()

    df = df_weather.copy()

    # 1. Asegurar formato numérico del viento en km/h (sin conversión a m/s)
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")

    # 2. Calcular FWI proxy directamente con viento en km/h
    def compute_fwi_strict(row):
        t, h, w, p = row["temperature"], row["humidity"], row["wind_speed"], row["precip_7d"]
        if any(pd.isna(v) or v is None for v in [t, h, w, p]):
            return None
        if float(h) <= 0:
            return None
        return round(
            max(0.0, (float(t) * float(w)) / (float(h) * (1.0 + math.log1p(max(0.0, float(p)))))),
            4,
        )

    df["fwi_proxy"] = df.apply(compute_fwi_strict, axis=1)

    # 3. Filtrar registros completos con la lista exacta de variables del modelo
    features = ["temperature", "humidity", "wind_speed", "precip_7d", "fwi_proxy"]
    valid_mask = df[features].notna().all(axis=1)

    df["risk_probability"] = None
    df["risk_tier"] = "Sin datos"
    df["prediction_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if valid_mask.any():
        X_valid = df.loc[valid_mask, features]
        
        log.info("🤖 Cargando modelo Random Forest (%s)...", MODEL_PATH.name)
        model = joblib.load(MODEL_PATH)

        probs = model.predict_proba(X_valid)[:, 1]
        df.loc[valid_mask, "risk_probability"] = probs
        thresholds = None
        if THRESHOLDS_PATH.exists():
            thresholds = json.loads(THRESHOLDS_PATH.read_text())
        else:
            log.warning("tier_thresholds.json not found; using legacy fixed thresholds.")
        df.loc[valid_mask, "risk_tier"] = df.loc[valid_mask, "risk_probability"].apply(
            lambda p: map_risk_level(p, thresholds)
        )
    log.info(
        "📊 Inferencia finalizada: %d municipios evaluados correctamente, %d sin datos suficientes.",
        valid_mask.sum(),
        (~valid_mask).sum(),
    )

    return df