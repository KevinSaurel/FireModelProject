"""
Entrenamiento del modelo de riesgo de incendios (MITECO).
Sin class_weight, validación temporal y umbrales de nivel guardados junto al modelo.
"""

import json
import logging
import math
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from Data.Constants.constants import MODEL_PATH

MITECO_DATASET_PATH = BASE_DIR / "Data" / "MITECO" / "real_weather_training.csv"
THRESHOLDS_PATH = MODEL_PATH.parent / "tier_thresholds.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("train_model_miteco")


def compute_fwi(row) -> float:
    t = float(row["temperature"]) if pd.notna(row["temperature"]) else 15.0
    h = max(float(row["humidity"]), 1.0) if pd.notna(row["humidity"]) else 50.0
    w = float(row["wind_speed"]) if pd.notna(row["wind_speed"]) else 2.5
    p = max(float(row["precip_7d"]), 0.0) if pd.notna(row["precip_7d"]) else 0.0
    return round(max(0.0, (t * w) / (h * (1.0 + math.log1p(p)))), 4)


def train_model():
    df = pd.read_csv(MITECO_DATASET_PATH, parse_dates=["date"])
    df["precip_7d"] = df["precip_7d"].fillna(0.0)
    df["fwi_proxy"] = df.apply(compute_fwi, axis=1)

    feature_cols = ["temperature", "humidity", "wind_speed", "precip_7d", "fwi_proxy"]
    target_col = "fire_risk"

    clean_df = df.dropna(subset=feature_cols + [target_col]).sort_values("date")
    log.info("Positive rate: %.3f", clean_df[target_col].mean())
    log.info("Positive rate by month:\n%s",
             clean_df.groupby(clean_df["date"].dt.month)[target_col].mean().round(2))

    # Temporal split: train on the past, test on the most recent 20%
    cut = int(len(clean_df) * 0.8)
    train_df, test_df = clean_df.iloc[:cut], clean_df.iloc[cut:]
    X_train, y_train = train_df[feature_cols], train_df[target_col].astype(int)
    X_test, y_test = test_df[feature_cols], test_df[target_col].astype(int)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )  # NO class_weight
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    log.info("ROC-AUC: %.4f | PR-AUC: %.4f | base rate: %.3f",
             roc_auc_score(y_test, y_prob),
             average_precision_score(y_test, y_prob),
             y_test.mean())
    log.info("Mean predicted prob on test: %.3f", y_prob.mean())

    # Tier thresholds from held-out probabilities (not training ones, which are overfit)
    s = pd.Series(y_prob)
    thresholds = {
        "low": float(s.quantile(0.60)),
        "medium": float(s.quantile(0.85)),
        "high": float(s.quantile(0.95)),
    }
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))
    log.info("Thresholds: %s", thresholds)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    log.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    train_model()