"""
ML Inference, Evaluation, and Threshold Optimization for Wildfire Risk Prediction.
Supports live/historical inference (predict_fire_risk) and threshold grid search evaluation.
"""

from datetime import datetime
import json
import logging
import math
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from Data.Constants.constants import MODEL_PATH
except ImportError:
    MODEL_PATH = BASE_DIR / "Model" / "fireTrainModel.pkl"

MITECO_DATASET_PATH = BASE_DIR / "Data" / "MITECO" / "real_weather_training.csv"
THRESHOLDS_PATH = MODEL_PATH.parent / "tier_thresholds.json"

log = logging.getLogger("fire_training_model")

COLUMN_ALIASES = {
    "temp_mean": "temperature",
    "humidity_mean": "humidity",
    "humidity_min": "min_humidity",
    "wind_speed_mean": "wind_speed",
    "wind_speed_max": "max_wind_speed",
    "rain_1d": "precip_1d",
    "rain_7d": "precip_7d",
    "rain_14d": "precip_14d",
    "rain_30d": "precip_30d",
    "true_dry_streak": "dry_spell_days",
}


def compute_fwi_proxy(row: pd.Series) -> float | None:
    """Calculates FWI proxy based on temperature, wind speed, humidity, and 7-day rainfall."""
    t = row.get("temperature") if "temperature" in row else row.get("temp_mean")
    h = row.get("humidity") if "humidity" in row else row.get("humidity_mean")
    w = row.get("max_wind_speed") if "max_wind_speed" in row else row.get("wind_speed")
    p = row.get("precip_7d") if "precip_7d" in row else row.get("rain_7d")

    if any(v is None or pd.isna(v) for v in [t, h, w, p]):
        return None
    try:
        t_val = float(t)
        h_val = float(h)
        w_val = float(w)
        p_val = max(0.0, float(p))
        if h_val <= 0:
            return None
        return round(
            max(0.0, (t_val * w_val) / (h_val * (1.0 + math.log1p(p_val)))),
            4,
        )
    except (ValueError, TypeError):
        return None


def compute_vpd(row: pd.Series) -> float | None:
    """Calculates Vapor Pressure Deficit (VPD in kPa) using peak daily drying capacity."""
    t = row.get("temp_max") if "temp_max" in row else row.get("temperature")
    h = row.get("min_humidity") if "min_humidity" in row else row.get("humidity")

    if any(v is None or pd.isna(v) for v in [t, h]):
        return None
    try:
        t_val = float(t)
        h_val = max(0.0, min(100.0, float(h)))
        sat_vp = 0.61078 * math.exp((17.27 * t_val) / (t_val + 237.3))
        return round(sat_vp * (1.0 - h_val / 100.0), 4)
    except (ValueError, TypeError):
        return None


def map_risk_level(
    probability: float | None,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Maps probability scores to Low, Medium, High, and Extreme risk tiers."""
    if probability is None or pd.isna(probability):
        return "Sin datos"

    low_t = thresholds.get("low", 0.35) if thresholds else 0.35
    med_t = thresholds.get("medium", 0.50) if thresholds else 0.50
    high_t = thresholds.get("high", 0.70) if thresholds else 0.70

    if probability < low_t:
        return "Low"
    if probability < med_t:
        return "Medium"
    if probability < high_t:
        return "High"
    return "Extreme"


def prepare_feature_columns(df_in: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Ensures all feature columns exist, maps aliases, and calculates missing metrics."""
    df = df_in.copy()

    for src, dst in COLUMN_ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
        elif dst in df.columns and src not in df.columns:
            df[src] = df[dst]

    for col in ["temperature", "temp_max", "temp_min", "humidity", "min_humidity", "wind_speed", "max_wind_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for p_col in ["precip_1d", "precip_7d", "precip_14d", "precip_30d"]:
        if p_col in df.columns:
            df[p_col] = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0)

    if "fwi_proxy" in feature_cols and ("fwi_proxy" not in df.columns or df["fwi_proxy"].isna().all()):
        df["fwi_proxy"] = df.apply(compute_fwi_proxy, axis=1)

    if "vpd" in feature_cols and ("vpd" not in df.columns or df["vpd"].isna().all()):
        df["vpd"] = df.apply(compute_vpd, axis=1)

    if "date" in df.columns:
        dt_series = pd.to_datetime(df["date"], errors="coerce")
        if "month" in feature_cols and ("month" not in df.columns or df["month"].isna().all()):
            df["month"] = dt_series.dt.month
        if "day_of_year" in feature_cols and ("day_of_year" not in df.columns or df["day_of_year"].isna().all()):
            df["day_of_year"] = dt_series.dt.dayofyear

    if "temp_anomaly" in feature_cols and "temp_anomaly" not in df.columns:
        df["temp_anomaly"] = 0.0
    if "et0_14d" in feature_cols and "et0_14d" not in df.columns:
        df["et0_14d"] = 0.0

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def predict_fire_risk(df_weather: pd.DataFrame) -> pd.DataFrame:
    """Runs machine learning inference using the trained Random Forest model."""
    if df_weather is None or df_weather.empty:
        log.error("❌ No weather observations received.")
        return pd.DataFrame()

    if not MODEL_PATH.exists():
        log.error("❌ Model file not found at: %s", MODEL_PATH)
        return pd.DataFrame()

    log.info("🤖 Loading Random Forest model (%s)...", MODEL_PATH.name)
    model = joblib.load(MODEL_PATH)

    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = [
            "temperature", "temp_max", "temp_min", "humidity", "min_humidity",
            "wind_speed", "max_wind_speed", "precip_1d", "precip_7d", "precip_14d",
            "precip_30d", "dry_spell_days", "temp_anomaly", "et0_14d", "fwi_proxy",
            "vpd", "month", "day_of_year",
        ]

    df = prepare_feature_columns(df_weather, features)

    df["risk_probability"] = 0.0
    df["risk_tier"] = "Low"
    df["prediction_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    df["risk_probability"] = np.round(probs, 4)

    thresholds = None
    if THRESHOLDS_PATH.exists():
        try:
            thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Error reading tier_thresholds.json: %s", e)

    df["risk_tier"] = df["risk_probability"].apply(
        lambda p: map_risk_level(p, thresholds)
    )

    log.info(
        "📊 Inference completed: %d municipalities evaluated successfully.",
        len(df),
    )

    return df


def optimize_thresholds(y_test: np.ndarray, y_prob: np.ndarray):
    """Grid Search Optimization for decision thresholds across F1, F2, Precision, and Youden J."""
    print("\n" + "=" * 70)
    print("      🔍 RUNNING AUTOMATED THRESHOLD GRID SEARCH (100 STEPS) 🔍")
    print("=" * 70)

    threshold_steps = np.linspace(0.01, 0.99, 99)
    results = []

    for t in threshold_steps:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        youden = rec - fpr

        results.append({
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f2": f2,
            "youden": youden,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    df_res = pd.DataFrame(results)

    best_f1_row = df_res.loc[df_res["f1"].idxmax()]
    best_f2_row = df_res.loc[df_res["f2"].idxmax()]
    prec_80_df = df_res[df_res["precision"] >= 0.80]
    best_prec_80_row = prec_80_df.iloc[0] if not prec_80_df.empty else best_f1_row
    best_youden_row = df_res.loc[df_res["youden"].idxmax()]

    print(f"1️⃣ BEST OVERALL BALANCE (Max F1-Score):")
    print(f"   ► Optimal Threshold:   {best_f1_row['threshold']:.4f} ({best_f1_row['threshold']*100:.2f}%)")
    print(f"   ► Precision:           {best_f1_row['precision']*100:.2f}%")
    print(f"   ► Recall (Fires Hit):  {best_f1_row['recall']*100:.2f}% ({best_f1_row['tp']:,} fires)")
    print(f"   ► False Alarms (FP):   {best_f1_row['fp']:,}")
    print(f"   ► F1-Score:            {best_f1_row['f1']:.4f}")
    print("-" * 70)

    print(f"2️⃣ SAFETY FIRST (Max F2-Score - High Fire Recall Focus):")
    print(f"   ► Optimal Threshold:   {best_f2_row['threshold']:.4f} ({best_f2_row['threshold']*100:.2f}%)")
    print(f"   ► Precision:           {best_f2_row['precision']*100:.2f}%")
    print(f"   ► Recall (Fires Hit):  {best_f2_row['recall']*100:.2f}% ({best_f2_row['tp']:,} fires)")
    print(f"   ► False Alarms (FP):   {best_f2_row['fp']:,}")
    print("-" * 70)

    print(f"3️⃣ OPERATIONAL TARGET (Min 80% Precision Cutoff):")
    print(f"   ► Target Threshold:    {best_prec_80_row['threshold']:.4f} ({best_prec_80_row['threshold']*100:.2f}%)")
    print(f"   ► Precision:           {best_prec_80_row['precision']*100:.2f}%")
    print(f"   ► Recall (Fires Hit):  {best_prec_80_row['recall']*100:.2f}% ({best_prec_80_row['tp']:,} fires)")
    print(f"   ► False Alarms (FP):   {best_prec_80_row['fp']:,}")
    print("-" * 70)

    print(f"4️⃣ YOUDEN'S J INDEX OPTIMAL (Max ROC Separation):")
    print(f"   ► Optimal Threshold:   {best_youden_row['threshold']:.4f} ({best_youden_row['threshold']*100:.2f}%)")
    print(f"   ► Youden's J Score:    {best_youden_row['youden']:.4f}")
    print("=" * 70 + "\n")


def test_model():
    """Evaluates the model against holdout data and performs threshold grid search."""
    if not MODEL_PATH.exists():
        log.error("❌ Model not found at: %s", MODEL_PATH)
        return

    log.info("🤖 Loading pretrained model from %s...", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = [
            "temperature", "temp_max", "temp_min", "humidity", "min_humidity",
            "wind_speed", "max_wind_speed", "precip_1d", "precip_7d", "precip_14d",
            "precip_30d", "dry_spell_days", "temp_anomaly", "et0_14d", "fwi_proxy",
            "vpd", "month", "day_of_year",
        ]

    dataset_path = MITECO_DATASET_PATH if MITECO_DATASET_PATH.exists() else Path("real_weather_training.csv")
    if not dataset_path.exists():
        log.error("❌ Evaluation dataset not found.")
        return

    log.info("📥 Loading evaluation dataset from %s...", dataset_path)
    df = pd.read_csv(dataset_path)

    df_prep = prepare_feature_columns(df, feature_cols)
    target_col = "fire_risk"

    clean_df = df_prep.dropna(subset=feature_cols + [target_col]).copy()

    sample_cap = 3500000
    if len(clean_df) > sample_cap:
        log.info("Subsampling %d records for evaluation...", sample_cap)
        clean_df = clean_df.sample(n=sample_cap, random_state=42)

    X = clean_df[feature_cols]
    y = clean_df[target_col].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    log.info("⚡ Inference on holdout test set (%d samples)...", len(X_test))
    y_prob = model.predict_proba(X_test)[:, 1]

    optimize_thresholds(y_test.values, y_prob)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    test_model()