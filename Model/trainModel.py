"""
Random Forest Model Training for Forest Fire Risk Assessment.
Combines positive fire events (parsed from TTL RDF store) with sampled non-fire days
from Euskalmet. Incorporates enhanced meteorological variables (VPD, wind vectors,
multi-scale precipitation, FWI proxy, and cyclic seasonal features).
"""

import json
import logging
import math
from pathlib import Path
import random
import shutil
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from Data.Constants.constants import MODEL_PATH

TRAINING_DATASET_PATH = BASE_DIR / "Data" / "MITECO" / "real_weather_training.csv"
THRESHOLDS_PATH = MODEL_PATH.parent / "tier_thresholds.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("train_model_rf")

FEATURE_COLUMNS = [
    "temperature",
    "temp_max",
    "temp_min",
    "humidity",
    "min_humidity",
    "wind_speed",
    "max_wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "precip_1d",
    "precip_7d",
    "precip_30d",
    "dry_days",
    "vpd",
    "fwi_proxy",
    "month_sin",
    "month_cos",
    "is_summer",
]


def backup_existing_artifacts() -> list[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = []
    for path in (MODEL_PATH, THRESHOLDS_PATH):
        if not path.exists():
            continue
        backup_path = path.with_name(f"{path.name}.{timestamp}.bak")
        shutil.copy2(path, backup_path)
        backups.append(backup_path)
        log.info("Backup created: %s", backup_path)
    return backups


def fill_missing_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures all advanced features are calculated if missing from raw source."""
    if "temp_max" not in df.columns:
        df["temp_max"] = df["temperature"]
    if "temp_min" not in df.columns:
        df["temp_min"] = df["temperature"]
    if "min_humidity" not in df.columns:
        df["min_humidity"] = df["humidity"]
    if "max_wind_speed" not in df.columns:
        df["max_wind_speed"] = df["wind_speed"]
    if "precip_1d" not in df.columns:
        df["precip_1d"] = df.get("precip_7d", 0.0) / 7.0
    if "precip_30d" not in df.columns:
        df["precip_30d"] = df.get("precip_7d", 0.0) * 3.0
    if "dry_days" not in df.columns:
        df["dry_days"] = np.where(df["precip_1d"] < 1.0, 5, 0)
    if "wind_dir_sin" not in df.columns:
        df["wind_dir_sin"] = 0.0
    if "wind_dir_cos" not in df.columns:
        df["wind_dir_cos"] = 1.0

    # VPD calculation
    if "vpd" not in df.columns:
        t_c = df["temp_max"].fillna(20.0)
        h_pct = df["min_humidity"].fillna(50.0).clip(lower=1.0)
        e_sat = 0.61078 * np.exp((17.27 * t_c) / (t_c + 237.3))
        df["vpd"] = (e_sat * (1.0 - h_pct / 100.0)).round(4)

    # FWI proxy
    if "fwi_proxy" not in df.columns:
        t = df["temp_max"].fillna(20.0)
        w = df["max_wind_speed"].fillna(10.0)
        h = df["min_humidity"].fillna(50.0).clip(lower=1.0)
        p = df["precip_7d"].fillna(0.0).clip(lower=0.0)
        df["fwi_proxy"] = ((t * w) / (h * (1.0 + np.log1p(p)))).round(4)

    # Seasonality
    if "month_sin" not in df.columns and "date" in df.columns:
        months = pd.to_datetime(df["date"]).dt.month
        df["month_sin"] = np.sin(2 * np.pi * months / 12.0).round(4)
        df["month_cos"] = np.cos(2 * np.pi * months / 12.0).round(4)
        df["is_summer"] = months.isin([6, 7, 8, 9]).astype(int)

    return df


def train_model():
    log.info("Loading dataset from %s...", TRAINING_DATASET_PATH)
    df = pd.read_csv(TRAINING_DATASET_PATH, parse_dates=["date"])
    df = fill_missing_derived_features(df)

    target_col = "fire_risk"
    clean_df = df.dropna(subset=FEATURE_COLUMNS + [target_col]).sort_values("date").reset_index(drop=True)

    log.info("Total dataset records: %d", len(clean_df))
    log.info("Overall Fire (Positive) Rate: %.3f", clean_df[target_col].mean())

    # Temporal Train/Test Split (80% past train, 20% recent test)
    split_idx = int(len(clean_df) * 0.8)
    train_df = clean_df.iloc[:split_idx]
    test_df = clean_df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col].astype(int)
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[target_col].astype(int)

    log.info("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    log.info("📊 Validation Performance:")
    log.info("   ROC-AUC Score: %.4f", roc_auc)
    log.info("   PR-AUC Score:  %.4f", pr_auc)
    log.info("   Test Base Fire Rate: %.3f | Mean Predicted Prob: %.3f", y_test.mean(), y_prob.mean())

    # Feature Importance Logging
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    log.info("🌲 Top 10 Feature Importances:\n%s", importances.head(10).to_string())

    # Compute risk thresholds on held-out predictions
    s = pd.Series(y_prob)
    thresholds = {
        "low": float(s.quantile(0.50)),
        "medium": float(s.quantile(0.75)),
        "high": float(s.quantile(0.90)),
    }
    log.info("Risk Level Thresholds: %s", thresholds)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    backup_existing_artifacts()
    joblib.dump(rf, MODEL_PATH)
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    log.info("Model saved successfully to %s", MODEL_PATH)
    log.info("Thresholds saved successfully to %s", THRESHOLDS_PATH)


if __name__ == "__main__":
    train_model()