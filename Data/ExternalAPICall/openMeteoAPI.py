"""
Servicio Plan B: Ingesta meteorológica desde Open-Meteo por coordenadas.
Se activa para los municipios sin estación Euskalmet o con datos incompletos.
Alineado exactamente con el conjunto de variables de Euskalmet:
  - temperature
  - humidity
  - wind_speed
  - max_wind_speed (racha máxima)
  - wind_direction (dirección media del viento)
  - precip_7d
  - fwi_proxy
"""

import datetime
import json
import logging
import math
from pathlib import Path
import re
import sys
import time
import unicodedata

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

GEOJSON_PATH = BASE_DIR / "Data" / "GeoJSON" / "municipios-euskadi.geojson"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

log = logging.getLogger("open_meteo")


def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]", "", ascii_name).lower()


def _get_polygon_centroid(coords) -> tuple[float, float]:
    lats, lons = [], []

    def extract_pts(pt_list):
        for item in pt_list:
            if isinstance(item[0], (int, float)):
                lons.append(item[0])
                lats.append(item[1])
            else:
                extract_pts(item)

    extract_pts(coords)
    if not lats or not lons:
        return 43.15, -2.5
    return sum(lats) / len(lats), sum(lons) / len(lons)


def load_municipalities_centroids() -> dict[str, tuple[float, float]]:
    """Carga municipios de Euskadi desde GeoJSON, descartando códigos y territorios compartidos."""
    if not GEOJSON_PATH.exists():
        log.error("GeoJSON no encontrado en: %s", GEOJSON_PATH)
        return {}

    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    centroids = {}
    forbidden = {"es21", "es211", "es212", "es213", "euskadi", "pais vasco"}

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        if str(props.get("fazeria", "0")) == "1":
            continue

        muni_name = None

        # Priorizar claves de nombre oficial del municipio
        for key in [
            "iz_ofizial",
            "iz_euskal",
            "ud_iz_e",
            "municipality",
            "NOMBRE",
            "NAME_2",
            "nombre",
            "MUNICIP",
            "municipio",
        ]:
            val = str(props.get(key, "")).strip()
            # Descartar si son códigos tipo E38, C040 o números de código INE
            if (
                val
                and not val.isdigit()
                and not re.match(r"^[A-Z]\d{2,3}$", val)
                and val.lower() not in forbidden
                and (len(val) > 2 or key == "iz_ofizial")
            ):
                muni_name = val
                break

        if not muni_name:
            continue

        coords = feature.get("geometry", {}).get("coordinates", [])
        if coords:
            lat, lon = _get_polygon_centroid(coords)
            centroids[muni_name] = (lat, lon)

    log.info("GeoJSON cargado correctamente: %d municipios localizados.", len(centroids))
    return centroids


def fetch_open_meteo_fallback(
    missing_municipalities: list[str],
    target_day: datetime.date | None = None,
) -> pd.DataFrame:
    """Consulta Open-Meteo por coordenadas en modo batch devolviendo las mismas variables que Euskalmet."""
    if not missing_municipalities:
        return pd.DataFrame()

    day = target_day or (datetime.date.today() - datetime.timedelta(days=1))
    start_day = day - datetime.timedelta(days=6)

    all_centroids = load_municipalities_centroids()
    norm_lookup = {normalize_name(k): k for k in all_centroids.keys()}

    targets = []
    for name in missing_municipalities:
        norm = normalize_name(name)
        if norm in norm_lookup:
            orig_name = norm_lookup[norm]
            lat, lon = all_centroids[orig_name]
            targets.append({"name": name, "lat": lat, "lon": lon})

    if not targets:
        return pd.DataFrame()

    log.info("📡 Consultando Plan B (Open-Meteo) para %d municipios...", len(targets))

    result_frames = []
    for start in range(0, len(targets), 50):
        batch = targets[start : start + 50]
        lats = [f"{target['lat']:.4f}" for target in batch]
        lons = [f"{target['lon']:.4f}" for target in batch]
        params = {
            "latitude": ",".join(lats),
            "longitude": ",".join(lons),
            "start_date": start_day.isoformat(),
            "end_date": day.isoformat(),
            "daily": (
                "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
            ),
            "hourly": "wind_speed_10m,wind_gusts_10m,wind_direction_10m",
            "wind_speed_unit": "kmh",
            "timezone": "Europe/Berlin",
        }

        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=20)
            if resp.status_code != 200:
                log.error(
                    "Error Open-Meteo HTTP %d para el lote %d-%d",
                    resp.status_code,
                    start + 1,
                    start + len(batch),
                )
                continue

            data_list = resp.json()
            if isinstance(data_list, dict):
                data_list = [data_list]
            if not isinstance(data_list, list):
                log.error(
                    "Respuesta Open-Meteo inválida para el lote %d-%d",
                    start + 1,
                    start + len(batch),
                )
                continue
            if len(data_list) != len(batch):
                log.warning(
                    "Open-Meteo devolvió %d respuestas para %d municipios en el lote %d-%d",
                    len(data_list),
                    len(batch),
                    start + 1,
                    start + len(batch),
                )

            results = []
            for target, data in zip(batch, data_list):
                if not isinstance(data, dict):
                    log.warning(
                        "Respuesta Open-Meteo inválida para %s; se omite",
                        target["name"],
                    )
                    continue

                daily = data.get("daily", {})
                temps = daily.get("temperature_2m_mean", [])
                hums = daily.get("relative_humidity_2m_mean", [])
                precips = daily.get("precipitation_sum", [])

                hourly = data.get("hourly", {})
                hourly_times = hourly.get("time", [])
                hourly_winds = hourly.get("wind_speed_10m", [])
                hourly_gusts = hourly.get("wind_gusts_10m", [])
                hourly_dirs = hourly.get("wind_direction_10m", [])

                temp = temps[-1] if (temps and temps[-1] is not None) else None
                hum = hums[-1] if (hums and hums[-1] is not None) else None

                # Filtrar valores horarios para el día objetivo
                valid_indices = [
                    i for i, ts in enumerate(hourly_times)
                    if isinstance(ts, str) and ts.startswith(day.isoformat())
                ]

                valid_winds = [
                    float(hourly_winds[i]) for i in valid_indices
                    if i < len(hourly_winds)
                    and isinstance(hourly_winds[i], (int, float))
                    and not isinstance(hourly_winds[i], bool)
                    and math.isfinite(hourly_winds[i])
                ]

                valid_gusts = [
                    float(hourly_gusts[i]) for i in valid_indices
                    if i < len(hourly_gusts)
                    and isinstance(hourly_gusts[i], (int, float))
                    and not isinstance(hourly_gusts[i], bool)
                    and math.isfinite(hourly_gusts[i])
                ]

                valid_dirs = [
                    float(hourly_dirs[i]) for i in valid_indices
                    if i < len(hourly_dirs)
                    and isinstance(hourly_dirs[i], (int, float))
                    and not isinstance(hourly_dirs[i], bool)
                    and math.isfinite(hourly_dirs[i])
                ]

                wind = sum(valid_winds) / len(valid_winds) if valid_winds else None
                max_wind = max(valid_gusts) if valid_gusts else (max(valid_winds) if valid_winds else None)
                wind_dir = sum(valid_dirs) / len(valid_dirs) if valid_dirs else None

                precip_7d = (
                    sum(precips)
                    if len(precips) == 7 and all(value is not None for value in precips)
                    else None
                )

                # Cálculo de FWI proxy alineado con Euskalmet
                fwi = None
                if (
                    temp is not None
                    and hum is not None
                    and hum > 0
                    and wind is not None
                    and precip_7d is not None
                    and precip_7d >= 0
                ):
                    fwi = round((temp * wind) / (hum * (1 + math.log1p(precip_7d))), 4)

                results.append({
                    "date": day.isoformat(),
                    "municipality": target["name"],
                    "temperature": round(temp, 2) if temp is not None else None,
                    "humidity": round(hum, 2) if hum is not None else None,
                    "wind_speed": round(wind, 2) if wind is not None else None,
                    "max_wind_speed": round(max_wind, 2) if max_wind is not None else None,
                    "wind_direction": round(wind_dir, 2) if wind_dir is not None else None,
                    "precip_7d": round(precip_7d, 2) if precip_7d is not None else None,
                    "fwi_proxy": fwi,
                })

            if results:
                result_frames.append(pd.DataFrame(results))
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            log.exception(
                "Error consultando Open-Meteo para el lote %d-%d: %s",
                start + 1,
                start + len(batch),
                exc,
            )
        except Exception:
            log.exception(
                "Error inesperado procesando Open-Meteo para el lote %d-%d",
                start + 1,
                start + len(batch),
            )
        finally:
            if start + 50 < len(targets):
                time.sleep(0.5)

    if not result_frames:
        return pd.DataFrame()

    df_out = pd.concat(result_frames, ignore_index=True)
    log.info("✅ Plan B devolvió datos para %d municipios.", len(df_out))
    return df_out