"""
Model v3 - serving side (used by the live/forecast heatmap and the historical simulation).

What lives here
  * loads Model/fireTrainModel_v3.pkl, fireModel_v3_meta.json and tier_thresholds_v3.json
  * thresholds_for(day): all-year thresholds, or the summer set for Jun-Sep
  * fetch_history(): ~130 days of daily ERA5/Open-Meteo weather per point, with a disk cache so the
    old part of the history (immutable) is downloaded only once per point
  * predict_many(): fireFeatures.build_features() on every point -> model -> probability

Nothing here touches the old model (Model/fireTrainModel.pkl), so the old code path keeps working.
If the v3 files are missing, READY is False and main.py silently keeps using the old model.
"""
import datetime
import json
import logging
import math
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from Model.fireFeatures import BASE_FEATURES, MIN_HISTORY_DAYS, RAW_COLUMNS, build_features

log = logging.getLogger("serveV3")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "Model" / "fireTrainModel_v3.pkl"
META_PATH = ROOT_DIR / "Model" / "fireModel_v3_meta.json"
THRESHOLDS_PATH = ROOT_DIR / "Model" / "tier_thresholds_v3.json"
HISTORY_DIR = ROOT_DIR / "Data" / "cache" / "history_v3"   # safe to delete: it is re-downloaded

HISTORY_DAYS = 130           # >= MIN_HISTORY_DAYS (120): the Drought Code and 60/90-day features need it
ARCHIVE_LAG_DAYS = 6         # the ERA5 archive trails real time; newer days come from the forecast API
GRID_DEG = 0.25              # model cells; latitude/longitude are model inputs and must be snapped to this
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
EUSKADI_BOUNDS = {"lat_min": 42.47, "lat_max": 43.45, "lon_min": -3.45, "lon_max": -1.73}

# Same variables as the training download (fetch_exact_weather.py): wind mean = mean of the hourly values.
_DAILY = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
          "relative_humidity_2m_mean", "relative_humidity_2m_min", "wind_speed_10m_max",
          "precipitation_sum", "et0_fao_evapotranspiration"]
_RENAME = {
    "time": "date", "temperature_2m_mean": "temperature", "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min", "relative_humidity_2m_mean": "humidity",
    "relative_humidity_2m_min": "min_humidity", "wind_speed_10m_mean": "wind_speed",
    "wind_speed_10m_max": "max_wind_speed", "precipitation_sum": "precip_1d",
    "et0_fao_evapotranspiration": "et0",
}
# What the detail pages show (old names, so the existing translations keep working) + 4 FWI codes
DISPLAY_FEATURES = BASE_FEATURES + ["ffmc", "dmc", "dc", "fwi"]
# Euskalmet observation key -> raw column (observations override the target day in live mode)
_OBS_MAP = {"temp_mean": "temperature", "temp_max": "temp_max", "temp_min": "temp_min",
            "humidity_mean": "humidity", "humidity_min": "min_humidity",
            "wind_speed_mean": "wind_speed", "wind_speed_max": "max_wind_speed", "rain_1d": "precip_1d"}

# ---------------------------------------------------------------------------
# Model + thresholds
# ---------------------------------------------------------------------------
READY = False
LOAD_ERROR = None
MODEL = None
FEATURES: list = []
SUMMER_MONTHS = (6, 7, 8, 9)
_THR: dict = {}


def _check_block(block: dict, name: str) -> dict:
    out = {k: float(block[k]) for k in ("low", "medium", "high", "very_high")}
    if not 0 <= out["low"] < out["medium"] < out["high"] < out["very_high"] <= 1:
        raise ValueError(f"{name}: thresholds must be increasing values between 0 and 1: {out}")
    return out


try:
    MODEL = joblib.load(MODEL_PATH)
    _meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    FEATURES = list(_meta["features"])
    _raw = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    _THR = {"all-year": _check_block(_raw, "all-year"),
            "summer": _check_block(_raw["summer"], "summer")}
    SUMMER_MONTHS = tuple(int(m) for m in _raw.get("summer_months", SUMMER_MONTHS))
    READY = True
    log.info("Model v3 loaded (%s, %d features)", _meta.get("variant"), len(FEATURES))
except Exception as exc:  # noqa: BLE001 - the app must still start with the old model
    LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    log.error("Model v3 NOT available, the old model stays in use: %s", LOAD_ERROR)


def thresholds_for(day: datetime.date) -> dict:
    """Tier cut-offs for a target day. Summer months use the summer set (see Model v3.md)."""
    season = "summer" if day.month in SUMMER_MONTHS else "all-year"
    return {**_THR[season], "season": season}


def tier_of(p: float, thr: dict) -> str:
    if p < thr["low"]:
        return "LOW"
    if p < thr["medium"]:
        return "MEDIUM"
    if p < thr["high"]:
        return "HIGH"
    if p < thr["very_high"]:
        return "VERY_HIGH"
    return "EXTREME"


def snap(x: float) -> float:
    """Centre of the 0.25-degree cell that contains x (the model's latitude/longitude inputs)."""
    return round(round(x / GRID_DEG) * GRID_DEG, 2)


def euskadi_cells() -> list[tuple[float, float]]:
    """Centres of the 0.25-degree model cells that overlap Euskadi."""
    b, h = EUSKADI_BOUNDS, GRID_DEG / 2
    lats = [snap(b["lat_min"]) + GRID_DEG * i for i in range(-1, 6)]
    lons = [snap(b["lon_min"]) + GRID_DEG * i for i in range(-1, 10)]
    return [(round(la, 2), round(lo, 2)) for la in lats for lo in lons
            if b["lat_min"] - h <= la <= b["lat_max"] + h and b["lon_min"] - h <= lo <= b["lon_max"] + h]


# ---------------------------------------------------------------------------
# Open-Meteo requests (budgeted: free limit is 600 weighted calls/minute)
# ---------------------------------------------------------------------------
_WEIGHT_PER_MIN = 500
_sent: deque = deque()
_rate_lock = threading.Lock()


def _throttle(weight: float) -> None:
    while True:
        with _rate_lock:
            now = time.monotonic()
            while _sent and now - _sent[0][0] > 60:
                _sent.popleft()
            if not _sent or sum(w for _, w in _sent) + weight <= _WEIGHT_PER_MIN:
                _sent.append((now, weight))
                return
            wait = 60 - (now - _sent[0][0]) + 0.5
        time.sleep(max(wait, 1.0))


def _request(url: str, coords: list, first: datetime.date, last: datetime.date) -> list:
    days = (last - first).days + 1
    weight = len(coords) * days / 14 * 9 / 10          # 8 daily + 1 hourly variable
    params = {
        "latitude": ",".join(f"{c[0]:.4f}" for c in coords),
        "longitude": ",".join(f"{c[1]:.4f}" for c in coords),
        "start_date": first.isoformat(), "end_date": last.isoformat(),
        "daily": ",".join(_DAILY), "hourly": "wind_speed_10m",
        "wind_speed_unit": "kmh", "timezone": "Europe/Madrid",
    }
    for _ in range(6):
        _throttle(weight)
        log.info("Open-Meteo %s: %d points, %s..%s (weight %.0f)", "archive" if url == ARCHIVE_URL else "forecast",
                 len(coords), first, last, weight)
        resp = requests.get(url, params=params, timeout=120)
        if resp.status_code == 429:
            try:
                reason = str(resp.json().get("reason", ""))
            except ValueError:
                reason = ""
            if "daily" in reason.lower():
                raise RuntimeError(f"Open-Meteo daily limit reached: {reason}")
            pause = 3700 if "hourly" in reason.lower() else 65
            log.warning("Open-Meteo 429 (%s); sleeping %ss", reason or "no reason", pause)
            time.sleep(pause)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else [data]
    raise RuntimeError("Open-Meteo still rate-limited after 6 attempts")


def _frame(item) -> pd.DataFrame | None:
    daily = item.get("daily") if isinstance(item, dict) else None
    if not daily or not daily.get("time"):
        return None
    df = pd.DataFrame(daily)
    hourly = item.get("hourly") or {}
    if hourly.get("time"):
        hours = pd.DataFrame({"day": [t[:10] for t in hourly["time"]], "w": hourly["wind_speed_10m"]})
        df["wind_speed_10m_mean"] = df["time"].map(hours.groupby("day")["w"].mean())
    else:
        df["wind_speed_10m_mean"] = np.nan
    df = df.rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    return df[RAW_COLUMNS]


# ---------------------------------------------------------------------------
# History with disk cache (archive part only: it never changes)
# ---------------------------------------------------------------------------
_mem: dict = {}
_mem_lock = threading.Lock()
_VALUE_COLUMNS = RAW_COLUMNS[1:]


def _key(lat: float, lon: float) -> str:
    return f"{lat:.4f}_{lon:.4f}"


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in _VALUE_COLUMNS},
                        index=pd.DatetimeIndex([], name="date"))


def _load_cached(key: str) -> pd.DataFrame:
    with _mem_lock:
        if key in _mem:
            return _mem[key]
    df = _empty()
    path = HISTORY_DIR / f"{key}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date")[_VALUE_COLUMNS]
        except Exception:  # noqa: BLE001 - a broken cache file is simply refetched
            log.warning("Ignoring unreadable history cache file %s", path)
            df = _empty()
    with _mem_lock:
        _mem[key] = df
    return df


def _store(key: str, df: pd.DataFrame) -> None:
    with _mem_lock:
        _mem[key] = df
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        tmp = HISTORY_DIR / f"{key}.tmp"
        df.to_csv(tmp)
        tmp.replace(HISTORY_DIR / f"{key}.csv")
    except OSError as exc:
        log.warning("History cache not written (%s); memory cache only", exc)


def fetch_history(coords: list, end_day: datetime.date, days: int = HISTORY_DAYS) -> list:
    """Daily weather (RAW_COLUMNS, oldest first) for `days` days ending on end_day, one frame per point
    (None if unavailable). Old days come from the ERA5 archive (disk-cached), the last few from the
    forecast API."""
    coords = [(round(float(a), 4), round(float(b), 4)) for a, b in coords]
    keys = [_key(*c) for c in coords]
    start = end_day - datetime.timedelta(days=days - 1)
    arch_end = min(end_day, datetime.date.today() - datetime.timedelta(days=ARCHIVE_LAG_DAYS))
    uniq = {}
    for c, k in zip(coords, keys):
        uniq.setdefault(k, c)

    # 1) archive part: download only the days that are not cached yet
    cached = {k: _load_cached(k) for k in uniq}
    if arch_end >= start:
        wanted = pd.date_range(start, arch_end)
        jobs = defaultdict(list)
        for k in uniq:
            missing = wanted.difference(cached[k].index)
            if len(missing):
                jobs[(missing.min().date(), missing.max().date())].append(k)
        for (first, last), ks in jobs.items():
            for b in range(0, len(ks), 50):
                batch = ks[b:b + 50]
                data = _request(ARCHIVE_URL, [uniq[k] for k in batch], first, last)
                for k, item in zip(batch, data):
                    fr = _frame(item)
                    if fr is None:
                        continue
                    fr = fr.dropna(subset=["temp_max", "min_humidity"]).set_index("date")
                    merged = pd.concat([cached[k], fr])
                    cached[k] = merged[~merged.index.duplicated(keep="last")].sort_index()
                    _store(k, cached[k])

    # 2) recent part (not cached: it is revised every day)
    recent = {}
    if end_day > arch_end:
        first = max(arch_end + datetime.timedelta(days=1), start)
        ks = list(uniq)
        for b in range(0, len(ks), 50):
            batch = ks[b:b + 50]
            data = _request(FORECAST_URL, [uniq[k] for k in batch], first, end_day)
            for k, item in zip(batch, data):
                fr = _frame(item)
                if fr is not None:
                    recent[k] = fr.set_index("date")

    out = []
    for k in keys:
        part = cached[k].loc[(cached[k].index >= pd.Timestamp(start)) & (cached[k].index <= pd.Timestamp(arch_end))]
        if k in recent:
            part = pd.concat([part, recent[k]])
        part = part[~part.index.duplicated(keep="last")].sort_index()
        out.append(part.reset_index()[RAW_COLUMNS] if len(part) else None)
    return out


def apply_observations(daily: pd.DataFrame, obs: dict, day: datetime.date) -> pd.DataFrame:
    """Live mode: real Euskalmet readings replace the model values of the target day."""
    daily = daily.copy()
    rows = daily.index[daily["date"] == pd.Timestamp(day)]
    for key, col in _OBS_MAP.items():
        if obs.get(key) is not None and len(rows):
            daily.loc[rows, col] = float(obs[key])
    return daily


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _clean(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily[RAW_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max()))
    df["precip_1d"] = df["precip_1d"].fillna(0.0)
    df = df.interpolate(limit=3, limit_direction="both")
    return df.rename_axis("date").reset_index()


def predict_many(items: list) -> list:
    """items: [(daily_frame, lat, lon), ...] -> [{"probability", "features", "cell"} | None, ...]
    Each frame needs >= MIN_HISTORY_DAYS days ending on the target day."""
    if not READY:
        raise RuntimeError(f"Model v3 is not loaded: {LOAD_ERROR}")
    rows, kept = [], []
    for n, (daily, lat, lon) in enumerate(items):
        try:
            frame = _clean(daily)
            if len(frame) < MIN_HISTORY_DAYS:
                continue
            f = build_features(frame).tail(1).iloc[0].to_dict()
        except Exception:  # noqa: BLE001
            log.exception("Feature build failed for point %s,%s", lat, lon)
            continue
        f["latitude"], f["longitude"] = snap(lat), snap(lon)
        try:
            if not all(math.isfinite(float(f[c])) for c in FEATURES):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({c: float(f[c]) for c in FEATURES})
        kept.append((n, f))
    results: list = [None] * len(items)
    if not rows:
        return results
    proba = MODEL.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]
    for (n, f), p in zip(kept, proba):
        results[n] = {
            "probability": float(p),
            "features": {c: round(float(f[c]), 2) for c in DISPLAY_FEATURES if c in f},
            "model_features": {c: round(float(f[c]), 2) for c in FEATURES},
            "cell": (f["latitude"], f["longitude"]),
        }
    return results
