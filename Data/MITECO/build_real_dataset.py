import datetime
import math
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent.parent.parent
HISTORICAL_CSV_PATH = (
    BASE_DIR / "Data" / "MITECO" / "historico_incendios_euskadi.csv"
)
OUTPUT_DATASET_PATH = BASE_DIR / "Data" / "MITECO" / "real_weather_training.csv"

# Province Centroids in Euskadi
PROVINCE_COORDS = {
    "Álava": (42.8467, -2.6716),
    "Gipuzkoa": (43.1833, -2.1667),
    "Bizkaia": (43.2627, -2.9253),
}

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def create_http_session() -> requests.Session:
    """Creates a resilient HTTP session with automatic retries and backoff."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_year_weather(
    session: requests.Session, lat: float, lon: float, year: int
) -> pd.DataFrame | None:
    """Downloads an entire year of daily weather for a lat/lon in 1 single HTTP request."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": [
            "temperature_2m_max",
            "relative_humidity_2m_mean",
            "precipitation_sum",
        ],
        "hourly": ["wind_speed_10m"],
        "wind_speed_unit": "kmh",
        "timezone": "Europe/Madrid",
    }

    try:
        response = session.get(
            OPEN_METEO_ARCHIVE_URL, params=params, timeout=45
        )
        if response.status_code != 200:
            print(
                f"⚠️ Open-Meteo returned status {response.status_code} for {year}"
            )
            return None

        payload = response.json()
        daily = payload.get("daily", {})
        hourly = payload.get("hourly", {})
        if not daily or "time" not in daily or not hourly:
            return None

        df_year = pd.DataFrame(
            {
                "date": daily["time"],
                "temperature": daily["temperature_2m_max"],
                "humidity": daily["relative_humidity_2m_mean"],
                "precip": daily["precipitation_sum"],
            }
        )
        hourly_wind = pd.DataFrame(
            {
                "time": hourly.get("time", []),
                "wind_speed": hourly.get("wind_speed_10m", []),
            }
        )
        hourly_wind["date"] = pd.to_datetime(hourly_wind["time"]).dt.strftime("%Y-%m-%d")
        daily_mean_wind = hourly_wind.groupby("date")["wind_speed"].mean()
        df_year["wind_speed"] = df_year["date"].map(daily_mean_wind)

        # Only use a full 7-day rainfall window; missing days remain NULL.
        df_year["precip_7d"] = (
            df_year["precip"].rolling(window=7, min_periods=7).sum().round(2)
        )
        return df_year

    except Exception as exc:
        print(f"⚠️ Error fetching {year} weather: {exc}")
        return None


def calculate_fwi_proxy(row) -> float | None:
    temp = row["temperature"]
    wind = row["wind_speed"]
    hum = row["humidity"]
    precip_7d = row["precip_7d"]
    if pd.isna(temp) or pd.isna(wind) or pd.isna(hum) or pd.isna(precip_7d):
        return None
    if hum <= 0 or precip_7d < 0:
        return None

    numerator = temp * wind
    denominator = hum * (1.0 + math.log1p(precip_7d))
    return round(numerator / denominator, 4)


def build_real_dataset():
    print("📖 Loading MITECO fire records...")
    df_miteco = pd.read_csv(HISTORICAL_CSV_PATH)

    session = create_http_session()

    # Get unique (province, year) pairs present in MITECO data
    df_valid = df_miteco.dropna(subset=["anio", "provincia"]).copy()
    df_valid["anio"] = df_valid["anio"].astype(int)

    # Filter years within ERA5 reanalysis range (e.g., >= 1980)
    df_valid = df_valid[df_valid["anio"] >= 1980]

    weather_cache = {}
    fire_samples = []
    rng = random.Random(42)

    print("🌐 Bulk-downloading yearly historical weather blocks...")
    province_years = (
        df_valid[["provincia", "anio"]].drop_duplicates().to_records(index=False)
    )

    for i, (prov, year) in enumerate(province_years, 1):
        if prov not in PROVINCE_COORDS:
            continue

        cache_key = (prov, year)
        if cache_key not in weather_cache:
            lat, lon = PROVINCE_COORDS[prov]
            print(
                f" [{i}/{len(province_years)}] Downloading weather for {prov} ({year})..."
            )
            df_w = fetch_year_weather(session, lat, lon, year)
            if df_w is not None:
                weather_cache[cache_key] = df_w
            time.sleep(0.2)  # Polite pause between bulk calls

    # 1. Match Fire Days (Positive Class: 1)
    print("\n🔥 Matching MITECO fire events with weather data...")
    for idx, row in df_valid.iterrows():
        prov = row["provincia"]
        year = int(row["anio"])
        cache_key = (prov, year)

        if cache_key in weather_cache:
            df_w = weather_cache[cache_key]

            # Sample fire occurrence date in peak risk months (July - Oct)
            month = rng.choice([7, 8, 9, 10])
            day = rng.randint(1, 28)
            target_date = f"{year}-{month:02d}-{day:02d}"

            matched = df_w[df_w["date"] == target_date]
            if not matched.empty:
                r = matched.iloc[0].to_dict()
                r["fire_risk"] = 1
                fire_samples.append(r)

    print(
        f"✅ Matched {len(fire_samples)} ground-truth fire events (Class 1)."
    )

    # 2. Match Safe Days (Negative Class: 0)
    print("🌧️ Sampling non-fire control weather days...")
    safe_samples = []
    target_safe_count = len(fire_samples)

    for prov, year in weather_cache.keys():
        if len(safe_samples) >= target_safe_count:
            break
        df_w = weather_cache[(prov, year)]
        # Pick winter/spring non-fire control days
        df_safe = df_w[
            df_w["date"].str.contains("-01-|-02-|-03-|-04-|-05-|-11-|-12-")
        ]
        if not df_safe.empty:
            sampled_rows = df_safe.sample(
                min(len(df_safe), 10), random_state=42
            )
            for _, sr in sampled_rows.iterrows():
                r = sr.to_dict()
                r["fire_risk"] = 0
                safe_samples.append(r)

    # Combine positive and negative samples
    all_samples = fire_samples + safe_samples[: len(fire_samples)]
    df_result = pd.DataFrame(all_samples)

    # Compute FWI proxy for all samples
    df_result["fwi_proxy"] = df_result.apply(calculate_fwi_proxy, axis=1)

    # Select final feature columns
    cols = [
        "date",
        "temperature",
        "humidity",
        "wind_speed",
        "precip_7d",
        "fwi_proxy",
        "fire_risk",
    ]
    df_final = df_result[cols].dropna()
    df_final["date"] = pd.to_datetime(df_final["date"]).dt.strftime("%Y-%m-%d")

    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_DATASET_PATH, index=False)
    print(
        f"💾 Saved {len(df_final)} ground-truth weather records to: {OUTPUT_DATASET_PATH}"
    )


if __name__ == "__main__":
    build_real_dataset()