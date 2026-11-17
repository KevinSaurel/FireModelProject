"""
Model v3 - historical simulation endpoint (GET /api/risk/historical), kept apart from main.py.

Roll back at any time: set USE_V3_HISTORICAL = False in app/main.py (this file is then never used).

How a date is scored
  * the model uses the same in-region weather grid as the live map;
  * weather comes from Open-Meteo through serveV3.fetch_history(), which reuses its Open-Meteo response cache;
  * municipalities and the map are interpolated (inverse-distance) from the cell probabilities.
The response keeps the same shape the pages already read (records / stations / open_meteo_points) and adds
`thresholds`, `season` and `history_days`.
"""
import datetime
import logging
import math
from functools import lru_cache

import numpy as np
import requests
from fastapi import APIRouter, HTTPException, Query

from Model import serveV3

log = logging.getLogger("historicalV3")

MIN_DATE = datetime.date(2010, 5, 1)       # the model was trained on 2010-2023 weather (+120 warm-up days)


def _idw(cells: np.ndarray, values: np.ndarray, lat: float, lon: float) -> float:
    cos_lat = math.cos(math.radians(43.0))
    d = np.maximum(np.hypot(cells[:, 0] - lat, (cells[:, 1] - lon) * cos_lat), 1e-6)
    w = 1.0 / d ** 2
    return float((w * values).sum() / w.sum())


@lru_cache(maxsize=24)
def _compute(date_str: str, grid: tuple[tuple[float, float], ...]) -> dict:
    day = datetime.date.fromisoformat(date_str)
    log.info("%s: loading Open-Meteo history for %d in-region grid points", date_str, len(grid))
    weather = serveV3.fetch_history(list(grid), day)
    available = [(cell, frame) for cell, frame in zip(grid, weather) if frame is not None]
    results = serveV3.predict_many([
        (frame, cell[0], cell[1]) for cell, frame in available
    ])
    ok = [(cell, result) for (cell, _), result in zip(available, results) if result is not None]
    if not ok:
        raise HTTPException(status_code=503, detail="No weather data available for this date")
    xy = np.array([c for c, _ in ok], dtype=float)
    prob = np.array([r["probability"] for _, r in ok], dtype=float)
    thr = serveV3.thresholds_for(day)

    records = []
    from Data.ExternalAPICall.openMeteoAPI import load_municipalities_centroids  # same source as before
    for name, (lat, lon) in load_municipalities_centroids().items():
        nearest = int(np.argmin(np.hypot(xy[:, 0] - lat, (xy[:, 1] - lon) * math.cos(math.radians(43.0)))))
        p = _idw(xy, prob, lat, lon)
        records.append({
            "date": date_str, "municipality": name, "latitude": lat, "longitude": lon,
            "risk_probability": round(p, 4), "risk_tier": serveV3.tier_of(p, thr),
            "model_inputs": ok[nearest][1]["model_features"],
            **ok[nearest][1]["features"],
        })
    return {
        "date": date_str, "model": "v3", "history_days": serveV3.HISTORY_DAYS, "thresholds": thr,
        "records": records,
        "open_meteo_points": [{"lat": float(c[0]), "lon": float(c[1]), "probability": round(float(p), 4),
                               "model_inputs": r["model_features"]}
                              for (c, r), p in zip(ok, prob)],
    }


def register(app, station_provider, grid_provider) -> None:
    router = APIRouter()

    @router.get("/api/risk/historical")
    def get_historical_risk(date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
        try:
            day = datetime.date.fromisoformat(date)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="date must be a valid YYYY-MM-DD date") from exc
        if day < MIN_DATE:
            raise HTTPException(status_code=422, detail=f"Model v3 covers dates from {MIN_DATE.isoformat()} on")
        if day >= datetime.date.today():
            raise HTTPException(status_code=422, detail="Choose a past date (yesterday or earlier)")
        try:
            stations = [{"name": s["name"], "lat": s["lat"], "lon": s["lon"], "probability": None}
                        for s in station_provider()]
        except Exception:  # noqa: BLE001 - station markers are optional
            log.exception("Station list unavailable for the historical map")
            stations = []
        try:
            grid = tuple((float(lat), float(lon)) for lat, lon in grid_provider(stations))
            payload = _compute(date, grid)
        except HTTPException:
            raise
        except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
            log.exception("Historical v3 request failed for %s", date)
            raise HTTPException(status_code=502, detail=f"Could not calculate historical risk: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("Historical v3 prediction failed for %s", date)
            raise HTTPException(status_code=503, detail="Historical risk prediction failed") from exc
        return {**payload, "stations": stations}

    app.include_router(router)
