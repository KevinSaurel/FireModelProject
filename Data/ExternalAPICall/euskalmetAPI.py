"""
Euskalmet Open Data API Client (OpenAPI 3.0 High-Feature Extraction).
Fully decoupled from local database/SQLite persistence.
Extracts raw observations, multi-scale accumulated precipitation, wind vectors,
Vapor Pressure Deficit (VPD), FWI proxies, and temporal embeddings.
"""

import argparse
from collections import Counter
import datetime
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import jwt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from Data.Constants.constants import (
    JWT_AUD,
    JWT_ISS,
    JWT_VERSION,
    PRIVATE_KEY_PATH,
    USER_EMAIL,
)

BASE_URL = "https://api.euskadi.eus/euskalmet"
STATIONS_CATALOG_PATH = BASE_DIR / "Data" / "GeoJSON" / "euskalmet_stations.json"

PRECIP_DAYS = 30
REQUEST_TIMEOUT = 30
REQUEST_PAUSE = 0.1
TOKEN_MAX_AGE = 50 * 60
SLOTS_PER_DAY = 144

MEASURES = {
    ("measuresForAir", "temperature"): "temperature",
    ("measuresForAir", "humidity"): "humidity",
    ("measuresForWind", "mean_speed"): "mean_speed",
    ("measuresForWind", "max_speed"): "max_speed",
    ("measuresForWind", "mean_direction"): "mean_direction",
    ("measuresForWater", "precipitation"): "precipitation",
}

VALID_RANGE = {
    "temperature": (-30.0, 50.0),
    "humidity": (0.0, 100.0),
    "mean_speed": (0.0, 150.0),
    "max_speed": (0.0, 200.0),
    "mean_direction": (0.0, 360.0),
    "precipitation": (0.0, 500.0),
}

MIN_COVERAGE = {
    "temperature": 0.5,
    "humidity": 0.5,
    "mean_speed": 0.5,
    "max_speed": 0.3,
    "mean_direction": 0.3,
    "precipitation": 0.8,
}

RAIN_ONLY = frozenset(["precipitation"])
log = logging.getLogger("euskalmet")


class AuthError(RuntimeError):
    """API rejected JWT."""


def setup_logging(verbose: bool = False) -> None:
    log.handlers.clear()
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)


def _ensure_logging() -> None:
    if not log.handlers:
        setup_logging()


def generate_jwt() -> str:
    if not PRIVATE_KEY_PATH.exists():
        raise FileNotFoundError(f"Private key missing at: {PRIVATE_KEY_PATH}")
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
    def __init__(self) -> None:
        self.session = requests.Session()
        self._configure_session(self.session)
        self.calls = 0
        self.counters: Counter = Counter()
        self._token_ts = 0.0
        self._session_local = threading.local()
        self._auth_lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._refresh_token()

    @staticmethod
    def _configure_session(session: requests.Session) -> None:
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers.update({"Accept": "application/json"})

    def _thread_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            self._configure_session(session)
            self._session_local.session = session
        session.headers["Authorization"] = self.session.headers["Authorization"]
        return session

    def _refresh_token(self) -> None:
        self.session.headers["Authorization"] = f"Bearer {generate_jwt()}"
        self._token_ts = time.time()
        log.info("JWT token generated for %s", USER_EMAIL)

    def get_json(self, path: str) -> tuple[int | None, object | None]:
        with self._auth_lock:
            if time.time() - self._token_ts > TOKEN_MAX_AGE:
                self._refresh_token()
            session = self._thread_session()
        url = BASE_URL + path
        resp = None
        for attempt in (1, 2):
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
            except requests.RequestException:
                with self._stats_lock:
                    self.counters["network error"] += 1
                return None, None
            with self._stats_lock:
                self.calls += 1
            time.sleep(REQUEST_PAUSE)
            if resp.status_code in (401, 403):
                if attempt == 1:
                    self._refresh_token()
                    session.headers["Authorization"] = self.session.headers["Authorization"]
                    continue
                raise AuthError(f"HTTP {resp.status_code} on {path}")
            break

        if resp.status_code != 200:
            with self._stats_lock:
                self.counters[f"HTTP {resp.status_code}"] += 1
            return resp.status_code, None
        try:
            return 200, resp.json()
        except ValueError:
            return 200, None


def load_station_catalog(client: EuskalmetClient, refresh: bool = False) -> list[dict]:
    if STATIONS_CATALOG_PATH.exists() and not refresh:
        catalog = json.loads(STATIONS_CATALOG_PATH.read_text(encoding="utf-8"))
    else:
        log.info("Downloading station catalog from Euskalmet API...")
        status, data = client.get_json("/stations")
        ids = sorted({s["stationId"] for s in data if isinstance(s, dict) and "stationId" in s})
        catalog = []
        for sid in ids:
            status, d = client.get_json(f"/stations/{sid}/current")
            if not isinstance(d, dict):
                continue
            pos = ((d.get("position") or {}).get("position") or {}).get("GOOGLE") or {}
            catalog.append({
                "stationId": sid,
                "type": d.get("stationType"),
                "name": (d.get("name") or {}).get("SPANISH"),
                "municipality": (d.get("municipality") or {}).get("SPANISH"),
                "province": (d.get("province") or {}).get("SPANISH"),
                "lat": pos.get("y"),
                "lon": pos.get("x"),
            })
        STATIONS_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATIONS_CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    return [s for s in catalog if s.get("type") == "METEOROLOGICAL"]


def parse_day(station_id: str, day: datetime.date, data: dict, wanted: frozenset[str] | None = None) -> dict:
    candidates: dict[str, list[tuple[str, int, list[float]]]] = {}
    for item in data.get("items", []) or []:
        name = MEASURES.get((item.get("measureType"), item.get("measureId")))
        if name is None or (wanted is not None and name not in wanted):
            continue
        raw_total, good = 0, []
        lo, hi = VALID_RANGE[name]
        readings = item.get("readings", []) or []
        for reading in readings:
            values = reading.get("values", []) or []
            for v in values:
                raw_total += 1
                if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi:
                    good.append(float(v))
        candidates.setdefault(name, []).append((item.get("sensorId"), raw_total, good))

    out = {}
    for name, options in candidates.items():
        sensor, raw_total, good = max(options, key=lambda o: len(o[2]))
        coverage = MIN_COVERAGE.get(name, 0.5)
        if (len(good) / SLOTS_PER_DAY) < coverage:
            continue
        out[name] = {"sensor": sensor, "values": good}
    return out


def get_station_day(client: EuskalmetClient, station_id: str, day: datetime.date, wanted: frozenset[str] | None = None):
    path = f"/readings/aggregated/byDay/forStation/{station_id}/at/{day.year}/{day.month:02d}/{day.day:02d}"
    status, data = client.get_json(path)
    if not isinstance(data, dict):
        return None, status
    return parse_day(station_id, day, data, wanted), status


def compute_accumulated_rain(client: EuskalmetClient, station_id: str, day: datetime.date, first_day: dict, window_days: int) -> float | None:
    if "precipitation" not in first_day:
        return None
    total = sum(first_day["precipitation"]["values"])
    for back in range(1, window_days):
        prev_day = day - datetime.timedelta(days=back)
        prev, status = get_station_day(client, station_id, prev_day, RAIN_ONLY)
        if prev is None or "precipitation" not in prev:
            return None
        precip = round(sum(prev["precipitation"]["values"]), 2)
        total += precip
    return round(total, 2)


def compute_consecutive_dry_days(client: EuskalmetClient, station_id: str, day: datetime.date, first_day: dict) -> int:
    dry_count = 0
    p1 = sum(first_day.get("precipitation", {}).get("values", [0.0]))
    if p1 >= 1.0:
        return 0
    dry_count += 1
    for back in range(1, 30):
        prev_day = day - datetime.timedelta(days=back)
        prev, status = get_station_day(client, station_id, prev_day, RAIN_ONLY)
        if prev and "precipitation" in prev:
            precip = sum(prev["precipitation"]["values"])
            if precip < 1.0:
                dry_count += 1
            else:
                break
        else:
            break
    return dry_count


def calculate_vpd(temp_c: float, humidity_pct: float) -> float:
    """Computes Vapor Pressure Deficit (VPD in kPa)."""
    e_sat = 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    vpd = e_sat * (1.0 - humidity_pct / 100.0)
    return round(max(0.0, vpd), 4)


def collect_day(client: EuskalmetClient, stations: list[dict], day: datetime.date) -> pd.DataFrame:
    started = time.monotonic()
    rows = []

    def collect_station(st: dict) -> dict | None:
        sid = st["stationId"]
        parsed, status = get_station_day(client, sid, day)
        if parsed is None or any(m not in parsed for m in ("temperature", "humidity")):
            return None

        t_vals = parsed["temperature"]["values"]
        h_vals = parsed["humidity"]["values"]
        
        temp_mean = float(np.mean(t_vals))
        temp_max = float(np.max(t_vals))
        temp_min = float(np.min(t_vals))

        hum_mean = float(np.mean(h_vals))
        hum_min = float(np.min(h_vals))

        wind = parsed.get("mean_speed")
        wind_max = parsed.get("max_speed")
        wind_dir = parsed.get("mean_direction")

        wind_avg = float(np.mean(wind["values"])) if wind else 0.0
        max_wind_val = float(np.max(wind_max["values"])) if wind_max else wind_avg
        dir_wind_val = float(np.mean(wind_dir["values"])) if wind_dir else 0.0

        wind_rad = math.radians(dir_wind_val)
        wind_sin = round(math.sin(wind_rad), 4)
        wind_cos = round(math.cos(wind_rad), 4)

        precip_1d = round(sum(parsed.get("precipitation", {}).get("values", [0.0])), 2)
        precip_7d = compute_accumulated_rain(client, sid, day, parsed, 7) or precip_1d
        precip_30d = compute_accumulated_rain(client, sid, day, parsed, 30) or precip_7d
        dry_days = compute_consecutive_dry_days(client, sid, day, parsed)

        vpd = calculate_vpd(temp_mean, hum_min)

        muni = " ".join(str(st.get("municipality") or st.get("name") or "Desconocido").split())
        prov = str(st.get("province") or "Desconocido").strip()

        return {
            "stationId": sid,
            "municipality": muni,
            "province": prov,
            "temperature": temp_mean,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "humidity": hum_mean,
            "min_humidity": hum_min,
            "wind_speed": wind_avg,
            "max_wind_speed": max_wind_val,
            "wind_direction": dir_wind_val,
            "wind_dir_sin": wind_sin,
            "wind_dir_cos": wind_cos,
            "precip_1d": precip_1d,
            "precip_7d": precip_7d,
            "precip_30d": precip_30d,
            "dry_days": dry_days,
            "vpd": vpd,
        }

    workers = min(8, len(stations))
    if workers:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(collect_station, station) for station in stations]
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    row = future.result()
                    if row is not None:
                        rows.append(row)
                except Exception:
                    log.exception("Error processing weather station.")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    val_cols = [
        "temperature", "temp_max", "temp_min", "humidity", "min_humidity",
        "wind_speed", "max_wind_speed", "wind_direction", "wind_dir_sin", "wind_dir_cos",
        "precip_1d", "precip_7d", "precip_30d", "dry_days", "vpd"
    ]
    for c in val_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_rules = {
        "temperature": "mean",
        "temp_max": "max",
        "temp_min": "min",
        "humidity": "mean",
        "min_humidity": "min",
        "wind_speed": "mean",
        "max_wind_speed": "max",
        "wind_direction": "mean",
        "wind_dir_sin": "mean",
        "wind_dir_cos": "mean",
        "precip_1d": "mean",
        "precip_7d": "mean",
        "precip_30d": "mean",
        "dry_days": "max",
        "vpd": "max",
        "province": "first",
    }
    out = df.groupby("municipality", as_index=False).agg(agg_rules).round(4)
    out.insert(0, "date", day.isoformat())

    # Temporal cyclic embeddings & summer indicator
    month = day.month
    out["month_sin"] = round(math.sin(2 * math.pi * month / 12.0), 4)
    out["month_cos"] = round(math.cos(2 * math.pi * month / 12.0), 4)
    out["is_summer"] = 1 if 6 <= month <= 9 else 0

    def fwi_proxy(row):
        t, w, h, p7 = row["temp_max"], row["max_wind_speed"], row["min_humidity"], row["precip_7d"]
        if pd.isna(t) or pd.isna(w) or pd.isna(h) or pd.isna(p7) or h <= 0:
            return None
        return round((t * w) / (h * (1.0 + math.log1p(max(0.0, p7)))), 4)

    out["fwi_proxy"] = out.apply(fwi_proxy, axis=1)
    log.info("Finished collecting %s (%d municipalities). Took %.1f s", day, len(out), time.monotonic() - started)
    return out


def GetVariablesForSpecificDay(target_date: str | datetime.date) -> pd.DataFrame | None:
    _ensure_logging()
    if isinstance(target_date, str):
        day = datetime.date.fromisoformat(target_date)
    else:
        day = target_date
    client = EuskalmetClient()
    stations = load_station_catalog(client)
    df = collect_day(client, stations, day)
    return df if not df.empty else None


get_variables_for_specific_day = GetVariablesForSpecificDay