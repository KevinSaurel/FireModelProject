"""
High-Precision & Smooth Calibration Random Forest Model Training for Wildfire Risk.
Uses 18 engineered climate features with tuned depth (max_depth=13) and leaf smoothing.
"""

import json
import logging
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from Data.Constants.constants import MODEL_PATH
except ImportError:
    MODEL_PATH = BASE_DIR / "Model" / "fireTrainModel.pkl"

MITECO_DATASET_PATH = BASE_DIR / "Data" / "MITECO" / "real_weather_training.csv"
THRESHOLDS_PATH = MODEL_PATH.parent / "tier_thresholds.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("train_model_miteco")


def train_model():
    log.info("Loading dataset from %s...", MITECO_DATASET_PATH)
    df = pd.read_csv(MITECO_DATASET_PATH, parse_dates=["date"])

    feature_cols = [
        "temperature",
        "temp_max",
        "temp_min",
        "humidity",
        "min_humidity",
        "wind_speed",
        "max_wind_speed",
        "precip_1d",
        "precip_7d",
        "precip_14d",
        "precip_30d",
        "dry_spell_days",
        "temp_anomaly",
        "et0_14d",
        "fwi_proxy",
        "vpd",
        "month",
        "day_of_year",
    ]
    target_col = "fire_risk"

    clean_df = df.dropna(subset=feature_cols + [target_col])
    log.info(
        "Total dataset rows: %d | Overall positive rate: %.4f",
        len(clean_df),
        clean_df[target_col].mean(),
    )

    sample_cap = 3500000
    if len(clean_df) > sample_cap:
        log.info("Subsampling %d records for tuned Random Forest training...", sample_cap)
        clean_df = clean_df.sample(n=sample_cap, random_state=42)

    X = clean_df[feature_cols]
    y = clean_df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    log.info(
        "Training Calibrated Random Forest (300 trees, max_depth=13, min_samples_leaf=15) on %d samples...",
        len(X_train),
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=13,
        min_samples_leaf=15,
        min_samples_split=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)

    log.info("--- MODEL EVALUATION RESULTS ---")
    log.info("ROC-AUC: %.4f | PR-AUC: %.4f | Base test rate: %.4f", auc_roc, auc_pr, y_test.mean())
    log.info("Mean predicted probability on test set: %.4f", y_prob.mean())

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    log.info("\n--- FEATURE IMPORTANCES ---\n%s", importances.to_string())

    s = pd.Series(y_prob)
    thresholds = {
        "low": float(s.quantile(0.60)),
        "medium": float(s.quantile(0.85)),
        "high": float(s.quantile(0.95)),
    }
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))
    log.info("Thresholds saved to %s: %s", THRESHOLDS_PATH, thresholds)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    log.info("Model successfully saved to %s", MODEL_PATH)


if __name__ == "__main__":
    train_model()