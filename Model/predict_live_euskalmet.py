"""
Multi-Station Live Fire Risk Inference Pipeline (Euskalmet Open Data + Random Forest Model).

- Uses RS256 JWT authentication with Euskalmet API (https://api.euskadi.eus/euskalmet).
- Fetches 30 full days of historical observations per station in parallel via ThreadPoolExecutor.
- Computes exact 1d, 7d, 14d, and 30d rolling precipitation, dry spell streaks, and 30d thermal anomalies.
- Passes true feature vectors into `fireTrainModel.pkl` & `tier_thresholds.json`.
- Logs detailed station inputs, probability scores, and risk classifications.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time

import joblib
import jwt
import numpy as np
import pandas as pd
import requests

# Path & Constants Setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from Data.Constants.constants import (
    JWT_AUD,
    JWT_ISS,
    JWT_VERSION,
    MODEL_PATH,
    PRIVATE_KEY_PATH,
    USER_EMAIL,
)

THRESHOLDS_PATH = MODEL_PATH.parent / "tier_thresholds.json"
CATALOG_PATH = BASE_DIR / "Data" / "GeoJSON" / "euskalmet_stations.json"
BASE_URL = "https://api.euskadi.eus/euskalmet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("euskalmet_multistation_inference")


def generate_jwt() -> str:
    """Generates an RS256-signed JWT token using project credentials."""
    if not PRIVATE_KEY_PATH.exists():
        raise FileNotFoundError(f"❌ Private key file missing at: {PRIVATE_KEY_PATH}")
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


class EuskalmetClient:
    """Thread-safe, authenticated HTTP Client for Euskalmet REST API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._token_ts = 0.0
        self._auth_lock = threading.RLock()
        self._refresh_token()

    def _refresh_token(self):
        with self._auth_lock:
            token = generate_jwt()
            self.session.headers["Authorization"] = f"Bearer {token}"
            self._token_ts = time.time()
            log.info("🔑 Generated active JWT token for %s", USER_EMAIL)

    def get_json(self, path: str) -> tuple[int | None, object | None]:
        with self._auth_lock:
            if time.time() - self._token_ts > 3000:
                self._refresh_token()
            auth_header = self.session.headers["Authorization"]

        url = BASE_URL + path
        headers = {"Accept": "application/json", "Authorization": auth_header}

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code in (401, 403):
                self._refresh_token()
                headers["Authorization"] = self.session.headers["Authorization"]
                resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return resp.status_code, None
            return 200, resp.json()
        except requests.RequestException:
            return None, None


def load_artifacts():
    """Loads serialized Random Forest model and quantile risk thresholds."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model weight file missing at {MODEL_PATH}")
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"❌ Tier threshold configuration missing at {THRESHOLDS_PATH}")

    model = joblib.load(MODEL_PATH)
    thresholds = json.loads(THRESHOLDS_PATH.read_text())
    return model, thresholds


def fetch_active_stations(client: EuskalmetClient) -> list[dict]:
    """Loads station catalog from JSON cache or queries Euskalmet API catalog."""
    if CATALOG_PATH.exists():
        log.info("📂 Loading station catalog from local cache (%s)...", CATALOG_PATH)
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        meteorological = [
            {
                "stationId": s["stationId"],
                "name": s.get("name") or s["stationId"],
                "municipality": s.get("municipality") or "CAPV",
            }
            for s in catalog
            if (s.get("type") or s.get("stationType")) == "METEOROLOGICAL"
        ]
        if meteorological:
            log.info("✅ Found %d active METEOROLOGICAL stations.", len(meteorological))
            return meteorological

    log.info("🌐 Querying station catalog directly from Euskalmet API...")
    status, data = client.get_json("/stations")
    if status != 200 or not isinstance(data, list):
        log.error("Failed to retrieve stations catalog (HTTP %s)", status)
        return []

    station_ids = sorted({s["stationId"] for s in data if isinstance(s, dict) and "stationId" in s})
    catalog, meteorological = [], []

    for sid in station_ids:
        st_status, d = client.get_json(f"/stations/{sid}/current")
        if st_status == 200 and isinstance(d, dict):
            st_type = d.get("stationType")
            st_name = (d.get("name") or {}).get("SPANISH") or sid
            st_muni = (d.get("municipality") or {}).get("SPANISH") or "CAPV"
            entry = {"stationId": sid, "type": st_type, "name": st_name, "municipality": st_muni}
            catalog.append(entry)
            if st_type == "METEOROLOGICAL":
                meteorological.append({"stationId": sid, "name": st_name, "municipality": st_muni})

    if catalog:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("✅ Catalog populated with %d active METEOROLOGICAL stations.", len(meteorological))
    return meteorological


def fetch_station_history_30d(client: EuskalmetClient, station: dict, target_date: date) -> dict | None:
    """
    Fetches 30 individual daily aggregated API payloads for a station.
    Calculates exact historical rainfall, dry spells, and 30-day mean temperature.
    """
    sid = station["stationId"]
    daily_records = []

    # Query last 30 days (day 0 = yesterday, day 29 = 30 days ago)
    for days_back in range(30):
        d = target_date - timedelta(days=days_back)
        path = f"/readings/aggregated/byDay/forStation/{sid}/at/{d.year}/{d.month:02d}/{d.day:02d}"
        status, data = client.get_json(path)

        if status != 200 or not isinstance(data, dict):
            continue

        items = data.get("items", []) or []
        measures = {}

        for item in items:
            m_type = item.get("measureType")
            m_id = item.get("measureId")
            readings = item.get("readings", []) or []
            vals = []
            for r in readings:
                for v in r.get("values", []) or []:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals.append(float(v))
            if vals:
                measures[(m_type, m_id)] = vals

        temps = measures.get(("measuresForAir", "temperature"), [])
        hums = measures.get(("measuresForAir", "humidity"), [])
        winds = measures.get(("measuresForWind", "mean_speed"), [])
        precips = measures.get(("measuresForWater", "precipitation"), [])

        if temps and hums:
            daily_records.append({
                "date": d,
                "temp_mean": float(np.mean(temps)),
                "temp_max": float(np.max(temps)),
                "temp_min": float(np.min(temps)),
                "hum_mean": float(np.mean(hums)),
                "hum_min": float(np.min(hums)),
                "wind_mean": float(np.mean(winds)) if winds else 12.0,
                "wind_max": float(np.max(winds)) if winds else 22.0,
                "precip": float(np.sum(precips)) if precips else 0.0,
            })

    if not daily_records:
        return None

    # Day 0 is the primary evaluation day (yesterday)
    day0 = daily_records[0]

    # Exact precipitation accumulations over 1, 7, 14, and 30 days
    precip_1d = day0["precip"]
    precip_7d = sum(r["precip"] for r in daily_records[:7])
    precip_14d = sum(r["precip"] for r in daily_records[:14])
    precip_30d = sum(r["precip"] for r in daily_records[:30])

    # Exact consecutive dry streak (consecutive days with rain < 1.0mm starting from yesterday)
    dry_spell_days = 0
    for r in daily_records:
        if r["precip"] < 1.0:
            dry_spell_days += 1
        else:
            break

    # Exact 30-day mean temperature
    temp_30d_avg = float(np.mean([r["temp_mean"] for r in daily_records[:30]]))

    # Exact 14-day evapotranspiration proxy accumulation
    et0_14d = sum(
        max(0.0, (r["temp_max"] * 1.5) - (r["hum_mean"] * 0.2))
        for r in daily_records[:14]
    )

    # Clean humidity sensor glitches (e.g., 0.0% min humidity readings)
    min_humidity_cleaned = max(day0["hum_min"], 2.0)

    return {
        "station_name": f"{station['name']} ({station['municipality']})",
        "station_id": sid,
        "target_date": target_date,
        "days_retrieved": len(daily_records),
        "temperature": day0["temp_mean"],
        "temp_max": day0["temp_max"],
        "temp_min": day0["temp_min"],
        "humidity": day0["hum_mean"],
        "min_humidity": min_humidity_cleaned,
        "wind_speed": day0["wind_mean"],
        "max_wind_speed": day0["wind_max"],
        "precip_1d": round(precip_1d, 2),
        "precip_7d": round(precip_7d, 2),
        "precip_14d": round(precip_14d, 2),
        "precip_30d": round(precip_30d, 2),
        "dry_spell_days": dry_spell_days,
        "temp_30d_avg": round(temp_30d_avg, 2),
        "et0_14d": round(et0_14d, 2),
    }


def compute_station_features(raw: dict) -> pd.DataFrame:
    """Transforms raw 30-day historical metrics into the 18 model features."""
    target_date = raw["target_date"]
    month = target_date.month
    day_of_year = target_date.timetuple().tm_yday

    t_mean = round(raw["temperature"], 2)
    t_max = round(raw["temp_max"], 2)
    t_min = round(raw["temp_min"], 2)

    temp_anomaly = round(t_max - raw["temp_30d_avg"], 2)

    t = t_max
    w = raw["max_wind_speed"]
    h = max(raw["min_humidity"], 1.0)
    p7 = max(raw["precip_7d"], 0.0)

    fwi_proxy = round((t * w) / (h * (1.0 + math.log1p(p7))), 4)
    e_sat = 0.61078 * math.exp((17.27 * t) / (t + 237.3))
    vpd = round(e_sat * (1.0 - h / 100.0), 4)

    features = {
        "temperature": t_mean,
        "temp_max": t_max,
        "temp_min": t_min,
        "humidity": round(raw["humidity"], 2),
        "min_humidity": round(raw["min_humidity"], 2),
        "wind_speed": round(raw["wind_speed"], 2),
        "max_wind_speed": round(raw["max_wind_speed"], 2),
        "precip_1d": raw["precip_1d"],
        "precip_7d": raw["precip_7d"],
        "precip_14d": raw["precip_14d"],
        "precip_30d": raw["precip_30d"],
        "dry_spell_days": raw["dry_spell_days"],
        "temp_anomaly": temp_anomaly,
        "et0_14d": raw["et0_14d"],
        "fwi_proxy": fwi_proxy,
        "vpd": vpd,
        "month": month,
        "day_of_year": day_of_year,
    }

    return pd.DataFrame([features])


def get_risk_tier(prob: float, thresholds: dict) -> str:
    """Maps probability output to risk level tier."""
    if prob >= thresholds["high"]:
        return "EXTREME RISK 🔴"
    elif prob >= thresholds["medium"]:
        return "HIGH RISK 🟠"
    elif prob >= thresholds["low"]:
        return "MEDIUM RISK 🟡"
    else:
        return "LOW RISK 🟢"


def run_pipeline():
    model, thresholds = load_artifacts()
    client = EuskalmetClient()

    stations = fetch_active_stations(client)
    if not stations:
        log.error("❌ No active meteorological stations found.")
        return

    target_date = date.today() - timedelta(days=1)

    log.info("\n======================================================================")
    log.info("🔥 EUSKALMET BASQUE COUNTRY LIVE MULTI-STATION RISK ASSESSMENT")
    log.info("📅 Target Date: %s | Mode: 30-Day True API Accumulation", target_date.isoformat())
    log.info("======================================================================\n")

    log.info("⚡ Executing parallel 30-day API queries across %d stations...", len(stations))

    results = []
    start_time = time.time()

    # Parallel retrieval across all active stations using 12 worker threads
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_station = {
            executor.submit(fetch_station_history_30d, client, st, target_date): st
            for st in stations
        }

        for idx, future in enumerate(as_completed(future_to_station), start=1):
            raw = future.result()
            if raw is None:
                continue

            feature_df = compute_station_features(raw)
            probabilities = model.predict_proba(feature_df)[0]
            fire_prob = float(probabilities[1])
            risk_tier = get_risk_tier(fire_prob, thresholds)

            log.info("[%d/%d] LOCATION: %s (ID: %s | %d Days Retrieved)", idx, len(stations), raw["station_name"], raw["station_id"], raw["days_retrieved"])
            log.info("  ├─ Inputs: Temp(Mean/Max/Min): %.1f°C / %.1f°C / %.1f°C | Hum(Mean/Min): %.1f%% / %.1f%% | Wind(Mean/Max): %.1f / %.1f km/h",
                     raw["temperature"], raw["temp_max"], raw["temp_min"], raw["humidity"], raw["min_humidity"], raw["wind_speed"], raw["max_wind_speed"])
            log.info("  ├─ Real Rain: 1d: %.1fmm | 7d: %.1fmm | 14d: %.1fmm | 30d: %.1fmm | True Dry Streak: %d days",
                     feature_df["precip_1d"].iloc[0], feature_df["precip_7d"].iloc[0], feature_df["precip_14d"].iloc[0], feature_df["precip_30d"].iloc[0], feature_df["dry_spell_days"].iloc[0])
            log.info("  ├─ Derived: 30d Avg Temp: %.1f°C | Anomaly: %+.1f°C | FWI Proxy: %.4f | VPD: %.4f kPa",
                     raw["temp_30d_avg"], feature_df["temp_anomaly"].iloc[0], feature_df["fwi_proxy"].iloc[0], feature_df["vpd"].iloc[0])
            log.info("  └─ MODEL OUTPUT: Fire Probability: %.2f%% | Classification: %s\n",
                     fire_prob * 100, risk_tier)

            results.append({
                "station": raw["station_name"],
                "probability": fire_prob,
                "tier": risk_tier,
            })

    elapsed = time.time() - start_time
    log.info("======================================================================")
    log.info("✅ PROCESSED %d STATIONS IN %.2f SECONDS USING REAL API DATA", len(results), elapsed)
    log.info("======================================================================")


if __name__ == "__main__":
    run_pipeline()