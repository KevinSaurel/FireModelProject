"""Historical daily weather features from the Open-Meteo Archive API."""

import datetime
import logging
import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT = 45
MIN_ARCHIVE_DATE = datetime.date(1940, 1, 1)
BATCH_SIZE = 50
DRY_THRESHOLD_MM = 0.1

FEATURE_COLUMNS = [
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

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_min",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
]

log = logging.getLogger("open_meteo_historical")


def _daily_values(daily: Mapping[str, Any], variable: str, expected_days: list[str]) -> list[float] | None:
    times = daily.get("time")
    values = daily.get(variable)
    if not isinstance(times, list) or not isinstance(values, list):
        return None

    by_date = dict(zip(times, values))
    result: list[float] = []
    for day in expected_days:
        val = by_date.get(day)
        if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
            # Safe fallback for evapotranspiration/precipitation missing readings
            if variable in ("et0_fao_evapotranspiration", "precipitation_sum"):
                val = 0.0
            else:
                return None
        if not math.isfinite(val):
            return None
        result.append(float(val))
    return result


def _features_for_location(
    daily: Mapping[str, Any],
    target_date: datetime.date,
    expected_days: list[str],
) -> dict[str, float | int] | None:
    series = {
        variable: _daily_values(daily, variable, expected_days)
        for variable in DAILY_VARIABLES
    }
    if any(values is None for values in series.values()):
        return None

    target_index = len(expected_days) - 1
    target = {key: values[target_index] for key, values in series.items()}  # type: ignore[index]
    precipitation: list[float] = series["precipitation_sum"]  # type: ignore[assignment]
    temperatures: list[float] = series["temperature_2m_mean"]  # type: ignore[assignment]
    et0_series: list[float] = series["et0_fao_evapotranspiration"]  # type: ignore[assignment]

    target_rain = precipitation[-1]

    dry_spell_days = 0
    for amount in reversed(precipitation):
        if amount < DRY_THRESHOLD_MM:
            dry_spell_days += 1
        else:
            break

    temperature = target["temperature_2m_mean"]
    temp_max = target["temperature_2m_max"]
    temp_min = target["temperature_2m_min"]
    humidity = target["relative_humidity_2m_mean"]
    min_humidity = target["relative_humidity_2m_min"]
    wind_speed = target["wind_speed_10m_mean"]
    max_wind_speed = target["wind_speed_10m_max"]

    precip_7d = sum(precipitation[-7:])
    temp_anomaly = temperature - (sum(temperatures[-30:]) / max(1, len(temperatures[-30:])))

    # Compute FWI Proxy using peak fire weather conditions (temp_max, max_wind_speed, min_humidity)
    eff_temp = temp_max if temp_max is not None else temperature
    eff_wind = max(max_wind_speed, wind_speed) if max_wind_speed is not None else wind_speed
    eff_hum = min_humidity if (min_humidity is not None and min_humidity > 0) else humidity

    if eff_hum > 0:
        fwi_proxy = round(
            max(0.0, (eff_temp * eff_wind) / (eff_hum * (1.0 + math.log1p(max(precip_7d, 0.0))))),
            4,
        )
    else:
        fwi_proxy = 0.0

    # Compute VPD using peak daily atmospheric drying capacity (temp_max & min_humidity)
    eff_vpd_temp = temp_max if temp_max is not None else temperature
    eff_vpd_hum = min_humidity if min_humidity is not None else humidity

    sat_vp = 0.61078 * math.exp((17.27 * eff_vpd_temp) / (eff_vpd_temp + 237.3))
    vpd = round(sat_vp * (1.0 - min(max(eff_vpd_hum, 0.0), 100.0) / 100.0), 4)

    feats: dict[str, float | int] = {
        "temperature": temperature,
        "temp_mean": temperature,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "humidity": humidity,
        "humidity_mean": humidity,
        "min_humidity": min_humidity,
        "humidity_min": min_humidity,
        "wind_speed": wind_speed,
        "wind_speed_mean": wind_speed,
        "max_wind_speed": max_wind_speed,
        "wind_speed_max": max_wind_speed,
        "precip_1d": target_rain,
        "rain_1d": target_rain,
        "precip_7d": precip_7d,
        "rain_7d": precip_7d,
        "precip_14d": sum(precipitation[-14:]),
        "rain_14d": sum(precipitation[-14:]),
        "precip_30d": sum(precipitation[-30:]),
        "rain_30d": sum(precipitation[-30:]),
        "dry_spell_days": dry_spell_days,
        "true_dry_streak": dry_spell_days,
        "temp_anomaly": temp_anomaly,
        "temp_30d_avg": sum(temperatures[-30:]) / len(temperatures[-30:]),
        "et0_14d": sum(et0_series[-14:]),
        "fwi_proxy": fwi_proxy,
        "vpd": vpd,
        "month": target_date.month,
        "day_of_year": target_date.timetuple().tm_yday,
    }
    return feats


def fetch_historical_day_data(
    municipality_centroids: dict[str, tuple[float, float]],
    target_date_str: str,
) -> pd.DataFrame:
    """Fetch 31 days of archive weather and return target-day model features."""
    try:
        target_date = datetime.date.fromisoformat(target_date_str)
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD format") from exc

    if target_date > datetime.date.today() - datetime.timedelta(days=1):
        raise ValueError("Historical archive target date must be in the past")
    if target_date - datetime.timedelta(days=30) < MIN_ARCHIVE_DATE:
        raise ValueError("Open-Meteo Archive data is available from 1940 onward")
    if not municipality_centroids:
        raise ValueError("No municipality centroids are available")

    start_date = target_date - datetime.timedelta(days=30)
    expected_days = [
        (start_date + datetime.timedelta(days=offset)).isoformat()
        for offset in range(31)
    ]
    targets = [
        (name, float(coords[0]), float(coords[1]))
        for name, coords in municipality_centroids.items()
    ]
    rows: list[dict[str, Any]] = []

    for offset in range(0, len(targets), BATCH_SIZE):
        batch = targets[offset : offset + BATCH_SIZE]
        params = {
            "latitude": ",".join(f"{lat:.5f}" for _, lat, _ in batch),
            "longitude": ",".join(f"{lon:.5f}" for _, _, lon in batch),
            "start_date": start_date.isoformat(),
            "end_date": target_date.isoformat(),
            "daily": ",".join(DAILY_VARIABLES),
            "wind_speed_unit": "kmh",
            "timezone": "Europe/Madrid",
        }
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(f"Open-Meteo API Error: {payload.get('reason')}")
        except (requests.RequestException, ValueError) as exc:
            log.exception("Open-Meteo historical request failed for batch starting at %d", offset)
            raise RuntimeError("Could not retrieve historical weather from Open-Meteo") from exc

        locations = payload if isinstance(payload, list) else [payload]
        if len(locations) != len(batch):
            raise RuntimeError(
                f"Open-Meteo returned {len(locations)} locations for a batch of {len(batch)}"
            )

        for (municipality, latitude, longitude), location in zip(batch, locations):
            if not isinstance(location, dict) or not isinstance(location.get("daily"), dict):
                log.warning("No daily archive data returned for %s", municipality)
                continue
            features = _features_for_location(location["daily"], target_date, expected_days)
            if features is None:
                log.warning("Incomplete archive data returned for %s", municipality)
                continue
            rows.append({
                "date": target_date_str,
                "municipality": municipality,
                "latitude": latitude,
                "longitude": longitude,
                **features,
            })

    if not rows:
        raise RuntimeError(f"No complete historical weather data was returned for {target_date_str}")

    frame = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=[c for c in FEATURE_COLUMNS if c in frame.columns]).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No complete feature rows were available for {target_date_str}")
    return frame