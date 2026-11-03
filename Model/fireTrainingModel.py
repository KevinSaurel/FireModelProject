"""
Inferencia ML, Evaluación y Optimización de Umbrales para Predicción de Incendios.
Soporta inferencia en tiempo real (predict_fire_risk) y búsqueda grid de umbrales óptimos.
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
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


def compute_fwi_proxy(row: pd.Series) -> float | None:
    """Calcula el índice FWI proxy estricto para una fila de datos meteorológicos."""
    t, h, w, p = row.get("temperature"), row.get("humidity"), row.get("wind_speed"), row.get("precip_7d")
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


def map_risk_level(
    probability: float | None,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Asigna tiers según los umbrales guardados o límites predeterminados."""
    if probability is None or pd.isna(probability):
        return "Sin datos"
    if thresholds is not None:
        if probability < thresholds.get("low", 0.2569):
            return "Low"
        if probability < thresholds.get("medium", 0.5016):
            return "Medium"
        if probability < thresholds.get("high", 0.7505):
            return "High"
        return "Extreme"
    elif probability < 0.25:
        return "Low"
    elif probability < 0.50:
        return "Medium"
    elif probability < 0.75:
        return "High"
    else:
        return "Extreme"


def prepare_feature_columns(df_in: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Asegura que todas las características requeridas por el modelo existan."""
    df = df_in.copy()

    if "wind_speed" in df.columns:
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")

    if "fwi_proxy" in feature_cols:
        if "fwi_proxy" not in df.columns or df["fwi_proxy"].isna().all():
            df["fwi_proxy"] = df.apply(compute_fwi_proxy, axis=1)

    if "date" in df.columns:
        dt_series = pd.to_datetime(df["date"], errors="coerce")
        if "month" in feature_cols and "month" not in df.columns:
            df["month"] = dt_series.dt.month
        if "day_of_year" in feature_cols and "day_of_year" not in df.columns:
            df["day_of_year"] = dt_series.dt.dayofyear

    for p_col in ["precip_1d", "precip_7d", "precip_14d", "precip_30d"]:
        if p_col in feature_cols and p_col in df.columns:
            df[p_col] = df[p_col].fillna(0.0)

    return df


def predict_fire_risk(df_weather: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta inferencia ML en vivo utilizando el modelo preentrenado."""
    if df_weather is None or df_weather.empty:
        log.error("❌ No se recibieron observaciones meteorológicas.")
        return pd.DataFrame()

    if not MODEL_PATH.exists():
        log.error("❌ Modelo no encontrado en: %s", MODEL_PATH)
        return pd.DataFrame()

    log.info("🤖 Cargando modelo Random Forest (%s)...", MODEL_PATH.name)
    model = joblib.load(MODEL_PATH)

    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = ["temperature", "humidity", "wind_speed", "precip_7d", "fwi_proxy"]

    df = prepare_feature_columns(df_weather, features)
    valid_mask = df[features].notna().all(axis=1)

    df["risk_probability"] = None
    df["risk_tier"] = "Sin datos"
    df["prediction_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if valid_mask.any():
        X_valid = df.loc[valid_mask, features]
        probs = model.predict_proba(X_valid)[:, 1]
        df.loc[valid_mask, "risk_probability"] = probs

        thresholds = None
        if THRESHOLDS_PATH.exists():
            try:
                thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("Error leyendo tier_thresholds.json: %s", e)

        df.loc[valid_mask, "risk_tier"] = df.loc[valid_mask, "risk_probability"].apply(
            lambda p: map_risk_level(p, thresholds)
        )

    log.info(
        "📊 Inferencia finalizada: %d municipios evaluados correctamente.",
        valid_mask.sum(),
    )

    return df


def optimize_thresholds(y_test: np.ndarray, y_prob: np.ndarray):
    """Búsqueda Grid Iterativa para encontrar los umbrales óptimos de decisión."""
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
        
        # F2 Score (Prioritizes Recall 2x over Precision)
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        
        # Youden's J Statistic = TPR - FPR
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

    # 1. Best Balance (Max F1)
    best_f1_row = df_res.loc[df_res["f1"].idxmax()]
    
    # 2. Safety First (Max F2 - Catching more fires)
    best_f2_row = df_res.loc[df_res["f2"].idxmax()]

    # 3. High Precision Goal (Precision >= 80%)
    prec_80_df = df_res[df_res["precision"] >= 0.80]
    best_prec_80_row = prec_80_df.iloc[0] if not prec_80_df.empty else best_f1_row

    # 4. Youden's Index Optimal Cutoff
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
    """Evaluación del modelo y optimización de umbrales sin reentrenar."""
    if not MODEL_PATH.exists():
        log.error("❌ Modelo no encontrado en: %s", MODEL_PATH)
        return

    log.info("🤖 Cargando modelo preentrenado desde %s...", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = ["temperature", "humidity", "wind_speed", "precip_7d", "fwi_proxy"]

    dataset_path = MITECO_DATASET_PATH if MITECO_DATASET_PATH.exists() else Path("real_weather_training.csv")
    if not dataset_path.exists():
        log.error("❌ Dataset de evaluación no encontrado.")
        return

    log.info("📥 Cargando dataset de evaluación desde %s...", dataset_path)
    df = pd.read_csv(dataset_path)

    df_prep = prepare_feature_columns(df, feature_cols)
    target_col = "fire_risk"

    clean_df = df_prep.dropna(subset=feature_cols + [target_col]).copy()

    sample_cap = 3500000
    if len(clean_df) > sample_cap:
        log.info("Submuestreando %d registros para evaluación...", sample_cap)
        clean_df = clean_df.sample(n=sample_cap, random_state=42)

    X = clean_df[feature_cols]
    y = clean_df[target_col].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    log.info("⚡ Inferencia en holdout test set (%d muestras)...", len(X_test))
    y_prob = model.predict_proba(X_test)[:, 1]

    # Ejecutar optimizador automático de umbrales
    optimize_thresholds(y_test.values, y_prob)


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    test_model()