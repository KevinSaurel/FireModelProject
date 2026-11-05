"""
Euskadi Fire Risk - live heatmap API (Random Forest model).

Place this file in FireModelProject/app/ next to index.html and run from the
project root:   uvicorn app.main:app --reload --port 8000
then open http://localhost:8000

Data flow (live only, no municipality layer):
  1. Euskalmet: real observations for the target day at every station
     (temp mean/max/min, humidity mean/min, wind mean/max, rain_1d).
  2. Open-Meteo: 30-day history at each station location (rain 7/14/30d,
     dry streak, 30d mean temp) + a coarse grid covering the territory,
     kept only where no Euskalmet station is within STATION_RADIUS_KM.
  3. Features -> scaler -> Random Forest -> fire probability per point,
     then IDW interpolation onto a regular grid for the heatmap.
"""

import datetime
import json
import logging
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import os

# If running in Cloud Run, recreate PEM keys from environment variables
if os.getenv("EUSKALMET_PRIVATE_KEY"):
    (ROOT_DIR / "private_key.pem").write_text(
        os.getenv("EUSKALMET_PRIVATE_KEY").replace("\\n", "\n")
    )
if os.getenv("EUSKALMET_PUBLIC_KEY"):
    (ROOT_DIR / "public_key.pem").write_text(
        os.getenv("EUSKALMET_PUBLIC_KEY").replace("\\n", "\n")
    )

from Data.ExternalAPICall.euskalmetAPI import (  # noqa: E402
    EuskalmetClient,
    get_station_day,
    load_station_catalog,
)

log = logging.getLogger("heatmap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

app = FastAPI(title="Euskadi Fire Risk - Live Heatmap")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/Static", StaticFiles(directory=APP_DIR / "Static"), name="static")
app.mount("/translations", StaticFiles(directory=APP_DIR / "translations"), name="translations")

INDEX_PATH = APP_DIR / "index.html"
MODEL_PATH = ROOT_DIR / "Model" / "fireTrainModel.pkl"
SCALER_PATH = ROOT_DIR / "Model" / "scaler.pkl"
TIER_THRESHOLDS_PATH = ROOT_DIR / "Model" / "tier_thresholds.json"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
EUSKADI_BOUNDS = {"lat_min": 42.47, "lat_max": 43.45, "lon_min": -3.45, "lon_max": -1.73}

CACHE_TTL = 3600           # seconds
FORECAST_DAYS_AHEAD = 1    # "forecast" mode predicts this many days after today (1 = tomorrow)
HISTORY_DAYS = 30          # window ending on the target day
GRID_LAT_POINTS = 9        # Open-Meteo base grid: 9 x 12 = 108 points
GRID_LON_POINTS = 12
STATION_RADIUS_KM = 12.0   # grid points closer than this to a station are dropped
DRY_THRESHOLD_MM = 0.1     # a day counts as "dry" below this rainfall
IDW_POWER = 2.0

DEFAULT_TIER_THRESHOLDS = {"low": 0.35, "medium": 0.50, "high": 0.70}


def load_tier_thresholds() -> dict[str, float]:
    try:
        configured = json.loads(TIER_THRESHOLDS_PATH.read_text(encoding="utf-8"))
        thresholds = {key: float(configured[key]) for key in DEFAULT_TIER_THRESHOLDS}
        if not 0 <= thresholds["low"] < thresholds["medium"] < thresholds["high"] <= 1:
            raise ValueError("Tier thresholds must be increasing values between 0 and 1")
        return thresholds
    except (OSError, KeyError, TypeError, ValueError) as exc:
        log.warning("Could not load tier thresholds from %s: %s", TIER_THRESHOLDS_PATH, exc)
        return DEFAULT_TIER_THRESHOLDS.copy()


TIER_THRESHOLDS = load_tier_thresholds()
TIER_LOW = TIER_THRESHOLDS["low"]
TIER_MEDIUM = TIER_THRESHOLDS["medium"]
TIER_HIGH = TIER_THRESHOLDS["high"]
PROB_CAP = TIER_HIGH

@app.get("/style.css")
def style():
    return FileResponse(APP_DIR / "style.css", media_type="text/css")


# Serves Info.html, LiveFireRisk.html, Detail.html... whatever the capitalisation
@app.get("/{page}.html")
def html_page(page: str):
    for f in APP_DIR.glob("*.html"):
        if f.stem.lower() == page.lower():
            return FileResponse(f)
    raise HTTPException(status_code=404, detail="Page not found")


# Fallback column order, only used if the pickles carry no feature_names_in_.
DEFAULT_FEATURES = [
    "temp_mean", "temp_max", "temp_min",
    "humidity_mean", "humidity_min",
    "wind_speed_mean", "wind_speed_max",
    "rain_1d", "rain_7d", "rain_14d", "rain_30d", "true_dry_streak",
    "temp_30d_avg", "temp_anomaly", "fwi_proxy", "vpd",
]

# model feature -> Open-Meteo daily variable
OM_MAP = {
    "temp_mean": "temperature_2m_mean",
    "temp_max": "temperature_2m_max",
    "temp_min": "temperature_2m_min",
    "humidity_mean": "relative_humidity_2m_mean",
    "humidity_min": "relative_humidity_2m_min",
    "wind_speed_mean": "wind_speed_10m_mean",
    "wind_speed_max": "wind_speed_10m_max",
}
OM_DAILY = list(OM_MAP.values()) + ["precipitation_sum", "et0_fao_evapotranspiration"]

# One cache entry per mode: {"live": {"ts": ..., "payload": ...}, "forecast": {...}}
_cache: dict = {}
_lock = threading.Lock()
_station_points_cache: list = []


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
try:
    model = joblib.load("Model/fireTrainModel.pkl")
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error crítico al cargar el modelo: {e}")
    model = None
scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None


def feature_names() -> list[str]:
    for obj in (scaler, model):
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return [str(n) for n in names]
    return DEFAULT_FEATURES


def predict_probabilities(feature_rows: list[dict]) -> np.ndarray:
    if model is None:
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    names = feature_names()
    n_expected = getattr(model, "n_features_in_", len(names))
    if len(names) != n_expected:
        raise RuntimeError(
            f"Model expects {n_expected} features but I am building {len(names)} ({names}). "
            "Check feature order/names against Model/trainModel.py."
        )
    missing = [n for n in names if n not in feature_rows[0]]
    if missing:
        raise RuntimeError(
            f"Model needs features I do not compute: {missing}. "
            f"Model expects: {names}. I compute: {sorted(feature_rows[0])}"
        )

    X = pd.DataFrame(feature_rows)[names]
    if scaler is not None:
        arr = scaler.transform(X if hasattr(scaler, "feature_names_in_") else X.values)
        X = pd.DataFrame(arr, columns=names) if hasattr(model, "feature_names_in_") else arr
    elif not hasattr(model, "feature_names_in_"):
        X = X.values

    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else -1
    return proba[:, idx]


def risk_tier(p: float) -> str:
    if p < TIER_LOW:
        return "LOW"
    if p < TIER_MEDIUM:
        return "MEDIUM"
    if p < TIER_HIGH:
        return "HIGH"
    return "EXTREME"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _last(daily: dict, key: str):
    arr = daily.get(key) or []
    return arr[-1] if arr and arr[-1] is not None else None


def make_features(daily: dict, obs: dict | None = None, day: datetime.date | None = None) -> dict | None:
    """
    daily: Open-Meteo daily series (HISTORY_DAYS days ending on the target day).
    obs:   optional Euskalmet observations for the target day; they override
           Open-Meteo's values for that day.
    """
    obs = obs or {}
    f: dict = {}
    for feat, key in OM_MAP.items():
        v = obs.get(feat)
        f[feat] = float(v) if v is not None else (_last(daily, key))
    if any(f[k] is None for k in OM_MAP):
        return None

    rain = [0.0 if v is None else float(v) for v in (daily.get("precipitation_sum") or [])]
    if not rain:
        return None
    if obs.get("rain_1d") is not None:
        rain[-1] = float(obs["rain_1d"])

    f["rain_1d"] = rain[-1]
    f["rain_7d"] = sum(rain[-7:])
    f["rain_14d"] = sum(rain[-14:])
    f["rain_30d"] = sum(rain[-30:])

    streak = 0
    for r in reversed(rain):
        if r < DRY_THRESHOLD_MM:
            streak += 1
        else:
            break
    f["true_dry_streak"] = streak

    temps = [v for v in (daily.get("temperature_2m_mean") or []) if v is not None]
    if not temps:
        return None
    f["temp_30d_avg"] = sum(temps) / len(temps)
    f["temp_anomaly"] = f["temp_mean"] - f["temp_30d_avg"]

    t, h, w, p = f["temp_mean"], f["humidity_mean"], f["wind_speed_mean"], f["rain_7d"]
    f["fwi_proxy"] = (t * w) / (h * (1 + math.log1p(max(0.0, p)))) if h > 0 else 0.0

    es = 0.6108 * math.exp(17.27 * t / (t + 237.3))   # saturation vapour pressure (kPa)
    f["vpd"] = es * (1 - min(max(h, 0.0), 100.0) / 100.0)

    # FAO-56 reference evapotranspiration summed over the last 14 days (mm)
    et0 = [float(v) for v in (daily.get("et0_fao_evapotranspiration") or [])[-14:] if v is not None]
    if not et0:
        return None
    f["et0_14d"] = sum(et0)

    # Calendar features
    if day is not None:
        f["month"] = day.month
        f["day_of_year"] = day.timetuple().tm_yday

    # Aliases: the trained model uses these column names
    f.update({
        "temperature": f["temp_mean"],
        "humidity": f["humidity_mean"],
        "min_humidity": f["humidity_min"],
        "wind_speed": f["wind_speed_mean"],
        "max_wind_speed": f["wind_speed_max"],
        "precip_1d": f["rain_1d"],
        "precip_7d": f["rain_7d"],
        "precip_14d": f["rain_14d"],
        "precip_30d": f["rain_30d"],
        "dry_spell_days": f["true_dry_streak"],
    })
    return f


# ---------------------------------------------------------------------------
# Source 1: Euskalmet stations (target-day observations)
# ---------------------------------------------------------------------------
def _station_obs(client: EuskalmetClient, st: dict, day: datetime.date) -> dict | None:
    if st.get("lat") is None or st.get("lon") is None:
        log.info("Skipping Euskalmet station %s: coordinates are missing", st.get("stationId"))
        return None
    station_id = st.get("stationId")
    started = time.perf_counter()
    log.info("Requesting Euskalmet observations: station=%s date=%s", station_id, day.isoformat())
    try:
        parsed, _ = get_station_day(client, st["stationId"], day)
    except Exception as exc:
        log.warning(
            "Euskalmet request failed: station=%s date=%s elapsed=%.2fs error=%s",
            station_id, day.isoformat(), time.perf_counter() - started, exc,
        )
        return None
    if not parsed or "temperature" not in parsed or "humidity" not in parsed:
        log.info(
            "Euskalmet response has no usable temperature/humidity: station=%s date=%s elapsed=%.2fs",
            station_id, day.isoformat(), time.perf_counter() - started,
        )
        return None

    t, h = parsed["temperature"]["values"], parsed["humidity"]["values"]
    obs = {
        "name": " ".join(str(st.get("name") or st.get("municipality") or st["stationId"]).split()),
        "lat": float(st["lat"]),
        "lon": float(st["lon"]),
        "temp_mean": sum(t) / len(t),
        "temp_max": max(t),
        "temp_min": min(t),
        "humidity_mean": sum(h) / len(h),
        "humidity_min": min(h),
    }
    w = parsed.get("mean_speed")
    if w:
        obs["wind_speed_mean"] = sum(w["values"]) / len(w["values"])
        obs["wind_speed_max"] = max(w["values"])
    r = parsed.get("precipitation")
    if r:
        obs["rain_1d"] = sum(r["values"])
    log.info(
        "Euskalmet response parsed: station=%s date=%s temperature_values=%d "
        "humidity_values=%d wind_data=%s precipitation_data=%s elapsed=%.2fs",
        station_id, day.isoformat(), len(t), len(h), bool(w), bool(r),
        time.perf_counter() - started,
    )
    return obs


def fetch_euskalmet_observations(day: datetime.date) -> list[dict]:
    started = time.perf_counter()
    log.info("Starting Euskalmet observation fetch for %s", day.isoformat())
    try:
        client = EuskalmetClient()
        catalog_started = time.perf_counter()
        log.info("Loading Euskalmet station catalog")
        stations = load_station_catalog(client)
        log.info(
            "Loaded Euskalmet station catalog: stations=%d elapsed=%.2fs",
            len(stations), time.perf_counter() - catalog_started,
        )
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(stations)))) as ex:
            results = list(ex.map(lambda s: _station_obs(client, s, day), stations))
    except Exception:
        log.exception("Euskalmet fetch failed")
        return []
    out = [r for r in results if r]
    log.info(
        "Euskalmet fetch complete: usable_stations=%d total_stations=%d elapsed=%.2fs",
        len(out), len(stations), time.perf_counter() - started,
    )
    return out


# ---------------------------------------------------------------------------
# Source 2: Open-Meteo (history at stations + full grid)
# ---------------------------------------------------------------------------
def fetch_open_meteo_history(coords: list[tuple[float, float]], day: datetime.date) -> list[dict | None]:
    results: list[dict | None] = [None] * len(coords)
    start_day = day - datetime.timedelta(days=HISTORY_DAYS - 1)
    batch_count = (len(coords) + 49) // 50
    log.info(
        "Starting Open-Meteo archive requests: endpoint=%s coordinates=%d batches=%d "
        "date_range=%s..%s daily_variables=%d",
        OPEN_METEO_URL, len(coords), batch_count, start_day.isoformat(), day.isoformat(), len(OM_DAILY),
    )

    for i in range(0, len(coords), 50):
        batch = coords[i : i + 50]
        batch_number = i // 50 + 1
        params = {
            "latitude": ",".join(f"{c[0]:.4f}" for c in batch),
            "longitude": ",".join(f"{c[1]:.4f}" for c in batch),
            "start_date": start_day.isoformat(),
            "end_date": day.isoformat(),
            "daily": ",".join(OM_DAILY),
            "wind_speed_unit": "kmh",
            "timezone": "Europe/Berlin",
        }
        started = time.perf_counter()
        log.info(
            "Requesting Open-Meteo archive: batch=%d/%d coordinates=%d date_range=%s..%s",
            batch_number, batch_count, len(batch), start_day.isoformat(), day.isoformat(),
        )
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=40)
            if resp.status_code != 200:
                log.error(
                    "Open-Meteo response failed: batch=%d/%d status=%d elapsed=%.2fs body=%s",
                    batch_number, batch_count, resp.status_code, time.perf_counter() - started,
                    resp.text[:200],
                )
                continue
            data = resp.json()
        except (requests.RequestException, ValueError):
            log.exception(
                "Open-Meteo request or JSON parsing failed: batch=%d/%d elapsed=%.2fs",
                batch_number, batch_count, time.perf_counter() - started,
            )
            continue

        data_list = data if isinstance(data, list) else [data]
        usable_count = 0
        for j, d in enumerate(data_list[: len(batch)]):
            if isinstance(d, dict) and d.get("daily"):
                results[i + j] = d["daily"]
                usable_count += 1
        log.info(
            "Open-Meteo response processed: batch=%d/%d status=%d response_items=%d "
            "usable_daily_series=%d elapsed=%.2fs",
            batch_number, batch_count, resp.status_code, len(data_list), usable_count,
            time.perf_counter() - started,
        )
        if i + 50 < len(coords):
            time.sleep(0.5)
    log.info(
        "Open-Meteo archive fetch complete: usable_series=%d requested_coordinates=%d",
        sum(d is not None for d in results), len(coords),
    )
    return results


# ---------------------------------------------------------------------------
# Heatmap build
# ---------------------------------------------------------------------------
def build_base_grid() -> list[tuple[float, float]]:
    lats = np.linspace(EUSKADI_BOUNDS["lat_min"], EUSKADI_BOUNDS["lat_max"], GRID_LAT_POINTS)
    lons = np.linspace(EUSKADI_BOUNDS["lon_min"], EUSKADI_BOUNDS["lon_max"], GRID_LON_POINTS)
    return [(float(a), float(o)) for a in lats for o in lons]


def idw_grid(samples: list[dict], resolution: int) -> list[dict]:
    s_lat = np.array([s["lat"] for s in samples])
    s_lon = np.array([s["lon"] for s in samples])
    s_val = np.array([s["probability"] for s in samples])

    g_lat = np.linspace(EUSKADI_BOUNDS["lat_min"], EUSKADI_BOUNDS["lat_max"], resolution)
    g_lon = np.linspace(EUSKADI_BOUNDS["lon_min"], EUSKADI_BOUNDS["lon_max"], resolution)
    mesh_lat, mesh_lon = np.meshgrid(g_lat, g_lon, indexing="ij")
    flat_lat, flat_lon = mesh_lat.ravel(), mesh_lon.ravel()

    cos_lat = math.cos(math.radians(43.0))
    d_lat = flat_lat[:, None] - s_lat[None, :]
    d_lon = (flat_lon[:, None] - s_lon[None, :]) * cos_lat
    dist = np.maximum(np.sqrt(d_lat**2 + d_lon**2), 1e-6)
    weights = 1.0 / dist**IDW_POWER
    probs = (weights * s_val[None, :]).sum(axis=1) / weights.sum(axis=1)

    return [
        {
            "lat": round(float(a), 4),
            "lon": round(float(o), 4),
            "probability": round(float(p), 4),
            "intensity": round(float(min(1.0, p / PROB_CAP)), 4),
        }
        for a, o, p in zip(flat_lat, flat_lon, probs)
    ]


def fetch_station_points() -> list[dict]:
    """Station names and coordinates only (no observations). Used by forecast mode,
    where there are no Euskalmet readings for tomorrow, only the locations."""
    global _station_points_cache
    if _station_points_cache:
        return _station_points_cache
    try:
        client = EuskalmetClient()
        stations = load_station_catalog(client)
    except Exception:
        log.exception("Euskalmet station catalog failed")
        return []
    out = []
    for st in stations:
        if st.get("lat") is None or st.get("lon") is None:
            continue
        out.append({
            "name": " ".join(str(st.get("name") or st.get("municipality") or st["stationId"]).split()),
            "lat": float(st["lat"]),
            "lon": float(st["lon"]),
        })
    log.info("Station locations loaded for forecast mode: %d", len(out))
    if out:
        _station_points_cache = out
    return out


def build_payload(resolution: int, mode: str = "live") -> dict:
    """mode="live": yesterday's real observations (Euskalmet) + Open-Meteo.
    mode="forecast": Open-Meteo forecast for tomorrow at every point, stations included."""
    today = datetime.date.today()
    if mode == "forecast":
        day = today + datetime.timedelta(days=FORECAST_DAYS_AHEAD)
    else:
        day = today - datetime.timedelta(days=1)
    started = time.perf_counter()
    log.info("Building fire-risk heatmap: mode=%s date=%s resolution=%d", mode, day.isoformat(), resolution)

    if mode == "forecast":
        observations = fetch_station_points()
        log.info("Heatmap input: station locations=%d (forecast mode)", len(observations))
    else:
        observations = fetch_euskalmet_observations(day)
        log.info("Heatmap input: Euskalmet observations=%d", len(observations))

    grid = build_base_grid()
    initial_grid_size = len(grid)
    if observations:
        grid = [
            (lat, lon) for lat, lon in grid
            if min(haversine_km(lat, lon, o["lat"], o["lon"]) for o in observations) > STATION_RADIUS_KM
        ]
    log.info(
        "Open-Meteo grid prepared: original_points=%d points_after_station_filter=%d "
        "station_exclusion_radius_km=%.1f",
        initial_grid_size, len(grid), STATION_RADIUS_KM,
    )

    coords = [(o["lat"], o["lon"]) for o in observations] + grid
    log.info(
        "Fetching weather history for heatmap: station_coordinates=%d grid_coordinates=%d total=%d",
        len(observations), len(grid), len(coords),
    )
    history = fetch_open_meteo_history(coords, day)

    samples: list[dict] = []
    n_obs = len(observations)
    missing_history = 0
    missing_features = 0
    for k, (lat, lon) in enumerate(coords):
        daily = history[k]
        if not daily:
            missing_history += 1
            continue
        is_station = k < n_obs
        station = observations[k] if is_station else None
        # Live: the station's real readings override Open-Meteo for the target day.
        # Forecast: nobody has measured tomorrow yet, so everything comes from Open-Meteo.
        obs = station if mode == "live" else None
        feats = make_features(daily, obs, day)
        if feats is None:
            missing_features += 1
            continue
        if mode == "forecast":
            source = "Open-Meteo (prevision)"
        else:
            source = "Euskalmet + Open-Meteo history" if is_station else "Open-Meteo"
        samples.append({
            "name": station["name"] if station else None,
            "lat": lat,
            "lon": lon,
            "source": source,
            "features": feats,
        })

    if not samples:
        log.warning(
            "Heatmap build has no usable samples: missing_history=%d missing_features=%d elapsed=%.2fs",
            missing_history, missing_features, time.perf_counter() - started,
        )
        return {"error": "No usable data from Euskalmet or Open-Meteo"}

    log.info(
        "Weather features prepared: usable_samples=%d missing_history=%d missing_features=%d",
        len(samples), missing_history, missing_features,
    )
    log.info("Predicting fire risk for %d samples", len(samples))
    probs = predict_probabilities([s["features"] for s in samples])
    for s, p in zip(samples, probs):
        s["probability"] = round(float(p), 4)
    log.info("Fire-risk predictions complete: samples=%d", len(probs))

    stations = [
        {
            "name": s["name"], "lat": s["lat"], "lon": s["lon"], "source": s["source"],
            "probability": s["probability"],
            "intensity": round(min(1.0, s["probability"] / PROB_CAP), 4),
            "risk_tier": risk_tier(s["probability"]),
            "temperature": round(s["features"]["temp_mean"], 1),
            "humidity": round(s["features"]["humidity_mean"], 1),
            "wind_speed": round(s["features"]["wind_speed_mean"], 1),
            "rain_7d": round(s["features"]["rain_7d"], 1),
            "dry_streak": s["features"]["true_dry_streak"],
        }
        for s in samples if s["name"]
    ]

    # Per-point details for the detail page (every point that feeds the map)
    names = feature_names()
    fi = getattr(model, "feature_importances_", None)
    importances = (
        sorted(
            ({"feature": n, "importance": round(float(v), 4)} for n, v in zip(names, fi)),
            key=lambda x: -x["importance"],
        )
        if fi is not None and len(fi) == len(names) else []
    )

    payload = {
        "mode": mode,
        "date": day.isoformat(),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "grid_size": resolution,
        "prob_cap": PROB_CAP,
        "counts": {
            "euskalmet": len(stations),
            "open_meteo": len(samples) - len(stations),
        },
        "points": idw_grid(samples, resolution),
        "stations": stations,
        "samples": [
            {
                "name": s["name"], "lat": s["lat"], "lon": s["lon"],
                "source": s["source"], "probability": s["probability"],
                "features": {n: round(float(s["features"][n]), 2) for n in names if n in s["features"]},
            }
            for s in samples
        ],
        "importances": importances,
    }
    log.info(
        "Heatmap build complete: points=%d stations=%d euskalmet=%d open_meteo=%d elapsed=%.2fs",
        len(payload["points"]), len(stations), payload["counts"]["euskalmet"],
        payload["counts"]["open_meteo"], time.perf_counter() - started,
    )
    return payload


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/fire-risk/heatmap")
def get_heatmap(
    resolution: int = Query(45, ge=10, le=120, description="Grid resolution"),
    refresh: bool = Query(False, description="Ignore the cache"),
    mode: str = Query("live", pattern="^(live|forecast)$", description="live = yesterday's data, forecast = tomorrow"),
):
    request_started = time.perf_counter()
    log.info("API request received: endpoint=/api/v1/fire-risk/heatmap mode=%s resolution=%d refresh=%s",
             mode, resolution, refresh)
    with _lock:
        entry = _cache.get(mode) or {"ts": 0.0, "payload": None}
        cached = entry["payload"]
        fresh = (time.time() - entry["ts"]) < CACHE_TTL
        if not refresh and fresh and cached and cached.get("grid_size") == resolution:
            log.info(
                "Serving cached heatmap: mode=%s resolution=%d cache_age=%.2fs",
                mode, resolution, time.time() - entry["ts"],
            )
            return cached

        log.info(
            "Building heatmap: mode=%s cache_status=%s refresh=%s cached_resolution=%s requested_resolution=%d",
            mode, "fresh_but_resolution_mismatch" if fresh and cached else "expired_or_missing",
            refresh, cached.get("grid_size") if cached else None, resolution,
        )
        try:
            payload = build_payload(resolution, mode)
        except Exception as exc:
            log.exception("Heatmap build failed")
            payload = {"error": str(exc)}

        if "error" in payload:
            if cached:
                log.warning(
                    "Heatmap build failed; serving stale cached payload: cached_resolution=%s "
                    "elapsed=%.2fs error=%s",
                    cached.get("grid_size"), time.perf_counter() - request_started, payload["error"],
                )
                return cached   # serve stale data if we have any
            log.error(
                "Heatmap request failed with no cached payload: elapsed=%.2fs error=%s",
                time.perf_counter() - request_started, payload["error"],
            )
            return payload
        _cache[mode] = {"ts": time.time(), "payload": payload}
        log.info(
            "API request complete: endpoint=/api/v1/fire-risk/heatmap mode=%s status=success "
            "points=%d stations=%d elapsed=%.2fs",
            mode, len(payload["points"]), len(payload["stations"]), time.perf_counter() - request_started,
        )
        return payload


@app.get("/")
def index():
    log.info("API request received: endpoint=/")
    return FileResponse(INDEX_PATH)