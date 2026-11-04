"""
Euskalmet Open Data API Client.
Exposes:
  - get_live_weather_data(): Fetches live/yesterday weather without database access.
"""

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

PRECIP_DAYS = 7
REQUEST_TIMEOUT = 30
REQUEST_PAUSE = 0.1
TOKEN_MAX_AGE = 50 * 60
SLOTS_PER_DAY = 144

MEASURES = {
    ("measuresForAir", "temperature"): "temperature",
    ("measuresForAir", "humidity"): "humidity",
    ("measuresForWind", "mean_speed"): "mean_speed",
    ("measuresForWater", "precipitation"): "precipitation",
}

VALID_RANGE = {
    "temperature": (-30.0, 50.0),
    "humidity": (0.0, 100.0),
    "mean_speed": (0.0, 80.0),
    "precipitation": (0.0, 500.0),
}

MIN_COVERAGE = {
    "temperature": 0.5,
    "humidity": 0.5,
    "mean_speed": 0.5,
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
        log.info("JWT generated for %s", USER_EMAIL)

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
            except requests.RequestException as exc:
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
        log.info("Fetching station catalog from Euskalmet API...")
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
        for reading in item.get("readings", []) or []:
            for v in reading.get("values", []) or []:
                raw_total += 1
                if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi:
                    good.append(float(v))
        candidates.setdefault(name, []).append((item.get("sensorId"), raw_total, good))

    out = {}
    for name, options in candidates.items():
        sensor, raw_total, good = max(options, key=lambda o: len(o[2]))
        if (len(good) / SLOTS_PER_DAY) < MIN_COVERAGE[name]:
            continue
        out[name] = {"sensor": sensor, "values": good}
    return out


def get_station_day(client: EuskalmetClient, station_id: str, day: datetime.date, wanted: frozenset[str] | None = None):
    path = f"/readings/aggregated/byDay/forStation/{station_id}/at/{day.year}/{day.month:02d}/{day.day:02d}"
    status, data = client.get_json(path)
    if not isinstance(data, dict):
        return None, status
    return parse_day(station_id, day, data, wanted), status


def accumulate_rain(client: EuskalmetClient, station_id: str, day: datetime.date, first_day: dict, precip_days: int):
    if "precipitation" not in first_day:
        return None
    total = sum(first_day["precipitation"]["values"])
    for back in range(1, precip_days):
        prev_day = day - datetime.timedelta(days=back)
        prev, _ = get_station_day(client, station_id, prev_day, RAIN_ONLY)
        if prev is None:
            return None
        if "precipitation" not in prev:
            return None
        precipitation = round(sum(prev["precipitation"]["values"]), 2)
        total += precipitation
    return round(total, 2)


def collect_day(client: EuskalmetClient, stations: list[dict], day: datetime.date, precip_days: int) -> pd.DataFrame:
    started = time.monotonic()
    rows = []

    def collect_station(st: dict) -> dict | None:
        sid = st["stationId"]
        parsed, status = get_station_day(client, sid, day)
        if parsed is None or any(m not in parsed for m in ("temperature", "humidity")):
            return None

        rain = accumulate_rain(client, sid, day, parsed, precip_days)
        wind = parsed.get("mean_speed")
        temp = sum(parsed["temperature"]["values"]) / len(parsed["temperature"]["values"])
        hum = sum(parsed["humidity"]["values"]) / len(parsed["humidity"]["values"])
        wind_avg = (sum(wind["values"]) / len(wind["values"])) if wind else None

        muni = " ".join(str(st.get("municipality") or st.get("name") or "Desconocido").split())
        return {
            "stationId": sid,
            "municipality": muni,
            "temperature": temp,
            "humidity": hum,
            "wind_speed": wind_avg,
            "precip_7d": rain,
        }

    workers = min(8, len(stations))
    if workers:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(collect_station, station)
                for station in stations
            ]
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    row = future.result()
                except Exception:
                    log.exception("⚠️ Error procesando una estación meteorológica.")
                    continue
                if row is not None:
                    rows.append(row)
                if index % 25 == 0 or index == len(futures):
                    log.info("📊 Estaciones procesadas: %d/%d", index, len(futures))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    value_cols = ["temperature", "humidity", "wind_speed", "precip_7d"]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df.groupby("municipality", as_index=False)[value_cols].mean().round(2)
    out.insert(0, "date", day.isoformat())

    def fwi_proxy(row):
        t, w, h, p = row["temperature"], row["wind_speed"], row["humidity"], row["precip_7d"]
        if pd.isna(t) or pd.isna(w) or pd.isna(h) or pd.isna(p) or h <= 0 or p < 0:
            return None
        return (t * w) / (h * (1 + math.log1p(p)))

    out["fwi_proxy"] = out.apply(fwi_proxy, axis=1)
    log.info(
        "⏱️ Recopilación de %s completada en %.1f s (%d estaciones válidas).",
        day,
        time.monotonic() - started,
        len(out),
    )
    return out


# ---------------------------------------------------------------------------
# Public Exposed API Methods
# ---------------------------------------------------------------------------
def get_live_weather_data(target_day: datetime.date | None = None) -> pd.DataFrame | None:
    """Return a live/yesterday weather DataFrame without database access."""
    _ensure_logging()
    day = target_day or (datetime.date.today() - datetime.timedelta(days=1))
    client = EuskalmetClient()
    stations = load_station_catalog(client)
    df = collect_day(client, stations, day, PRECIP_DAYS)
    return df if not df.empty else None