"""
Servicio Integrador de Análisis Meteorológico.
Euskalmet actúa como fuente primaria. Si cualquier parámetro (temperatura, humedad,
viento o precipitación acumulada) viene NULL o falta el municipio, se completa
quirúrgicamente con datos reales de Open-Meteo (Plan B).
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from Data.ExternalAPICall.euskalmetAPI import get_live_weather_data
from Data.ExternalAPICall.openMeteoAPI import (
    fetch_open_meteo_fallback,
    load_municipalities_centroids,
)
from Data.Constants.constants import MUNICIPALITY_ALIASES
from Database.persistenceManager import (
    get_latest_weather_observations,
    save_risk_predictions,
)
from Model.fireTrainingModel import predict_fire_risk

log = logging.getLogger("implementacion")


def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]", "", ascii_name).lower()


def _load_geojson_municipalities() -> tuple[pd.DataFrame, dict[str, str]]:
    """Carga códigos y nombres oficiales del GeoJSON para enlazar ambas fuentes."""
    geojson_path = BASE_DIR / "Data" / "GeoJSON" / "municipios-euskadi.geojson"
    with geojson_path.open(encoding="utf-8") as file:
        geojson = json.load(file)

    municipalities = []
    name_to_code = {}
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        if str(properties.get("fazeria", "0")) == "1":
            continue

        code = str(properties.get("ud_kodea", "")).strip()
        official_name = properties.get("iz_ofizial")
        if not code or not official_name:
            log.warning(
                "⚠️ Municipio GeoJSON sin código o nombre oficial: %s",
                properties,
            )
            continue

        municipalities.append({
            "muni_code": code,
            "municipality": official_name,
        })
        for key in ("iz_ofizial", "iz_euskal", "ud_iz_e"):
            name = properties.get(key)
            if name:
                name_to_code[normalize_name(name)] = code

    code_to_name = {
        row["muni_code"]: row["municipality"] for row in municipalities
    }
    for alias, target in MUNICIPALITY_ALIASES.items():
        target_code = name_to_code.get(normalize_name(target))
        if target_code is None:
            log.warning(
                "⚠️ Alias municipal sin destino GeoJSON: %s -> %s",
                alias,
                target,
            )
            continue
        name_to_code[normalize_name(alias)] = target_code

    return pd.DataFrame(municipalities), name_to_code


def _canonicalize_municipalities(
    df: pd.DataFrame,
    name_to_code: dict[str, str],
    code_to_name: dict[str, str],
    source: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()
    result["muni_code"] = result["municipality"].apply(
        lambda name: name_to_code.get(normalize_name(name))
    )
    unmatched = sorted(
        result.loc[result["muni_code"].isna(), "municipality"].dropna().unique()
    )
    if unmatched:
        log.warning(
            "⚠️ Municipios de %s descartados por no existir en GeoJSON: %s",
            source,
            unmatched,
        )

    result = result.dropna(subset=["muni_code"])
    result["municipality"] = result["muni_code"].map(code_to_name)
    if result.empty:
        return result

    value_columns = [
        column
        for column in ("temperature", "humidity", "wind_speed", "precip_7d")
        if column in result
    ]
    aggregations = {column: "mean" for column in value_columns}
    if "date" in result:
        aggregations["date"] = "first"
    return result.groupby(
        ["muni_code", "municipality"], as_index=False
    ).agg(aggregations)


def get_live_weather_analysis() -> pd.DataFrame:
    """
    1. Obtiene datos de Euskalmet (fuente primaria).
    2. Identifica municipios ausentes O municipios existentes con algún parámetro en NULL.
    3. Consulta Open-Meteo para obtener esos parámetros reales faltantes.
    4. Fusiona ambas fuentes: prevalece Euskalmet y Open-Meteo rellena los vacíos (NULLs).
    5. Ejecuta inferencia de ML con el dataset 100% completo y guarda resultados.
    """
    _ensure_logging()
    log.info("🌐 Solicitando datos primarios a Euskalmet API...")
    geo_municipalities, name_to_code = _load_geojson_municipalities()
    code_to_name = dict(zip(
        geo_municipalities["muni_code"],
        geo_municipalities["municipality"],
    ))
    df_euskalmet = None

    # Step 1: Obtener Euskalmet o fallback SQLite
    try:
        df_euskalmet = get_live_weather_data()
    except Exception as exc:
        log.warning("⚠️ Petición a Euskalmet fallida: %s", exc)

    if df_euskalmet is None or df_euskalmet.empty:
        log.warning("⚠️ Sin datos en directo de Euskalmet. Activando respaldo SQLite...")
        try:
            df_euskalmet = get_latest_weather_observations()
        except Exception as exc:
            log.error("❌ Fallo en lectura de base de datos: %s", exc)
            df_euskalmet = pd.DataFrame()

    df_euskalmet = _canonicalize_municipalities(
        df_euskalmet,
        name_to_code,
        code_to_name,
        "Euskalmet",
    )
    target_date = (
        df_euskalmet["date"].iloc[0]
        if (df_euskalmet is not None and not df_euskalmet.empty)
        else (datetime.now().date() - pd.Timedelta(days=1)).isoformat()
    )

    critical_cols = ["temperature", "humidity", "wind_speed", "precip_7d"]

    # Step 2: Detectar municipios con algún dato NULL en Euskalmet
    incomplete_euskalmet_munis = []
    if df_euskalmet is not None and not df_euskalmet.empty:
        mask_nulls = df_euskalmet[critical_cols].isna().any(axis=1)
        incomplete_euskalmet_munis = df_euskalmet.loc[mask_nulls, "municipality"].tolist()

    # Detectar municipios totalmente ausentes en Euskalmet
    all_centroids = load_municipalities_centroids()
    all_muni_names = list(all_centroids.keys())

    existing_codes = (
        set(df_euskalmet["muni_code"])
        if not df_euskalmet.empty
        else set()
    )
    absent_munis = [
        name
        for name in all_muni_names
        if name_to_code.get(normalize_name(name)) not in existing_codes
    ]

    # Lista consolidada de municipios que requieren datos de Open-Meteo
    munis_to_query_plan_b = list(dict.fromkeys(
        incomplete_euskalmet_munis + absent_munis
    ))

    # Step 3: Consultar Open-Meteo y fusionar vacíos con combine_first
    completed_from_open = 0
    added_as_absent = 0
    if munis_to_query_plan_b:
        log.info(
            "🔍 Plan B Activado: %d municipios necesitan completado/alta en Open-Meteo...",
            len(munis_to_query_plan_b),
        )
        target_dt = datetime.fromisoformat(target_date).date()
        df_openmeteo = fetch_open_meteo_fallback(munis_to_query_plan_b, target_day=target_dt)
        df_openmeteo = _canonicalize_municipalities(
            df_openmeteo,
            name_to_code,
            code_to_name,
            "Open-Meteo",
        )

        if not df_openmeteo.empty:
            if not df_euskalmet.empty:
                # Euskalmet conserva prioridad; Open-Meteo solo completa NULLs.
                df_eusk_idx = df_euskalmet.set_index("muni_code")
                df_open_idx = df_openmeteo.set_index("muni_code")
                open_for_existing = df_open_idx.reindex(df_eusk_idx.index)
                completed_mask = (
                    df_eusk_idx[critical_cols].isna()
                    & open_for_existing[critical_cols].notna()
                ).any(axis=1)
                completed_from_open = int(completed_mask.sum())
                added_as_absent = int(
                    (~df_open_idx.index.isin(df_eusk_idx.index)).sum()
                )

                # combine_first preserva Euskalmet y sustituye los NaN/NULLs por los valores de Open-Meteo
                df_combined_idx = df_eusk_idx.combine_first(df_open_idx)
                df_combined = df_combined_idx.reset_index()
            else:
                df_combined = df_openmeteo
                added_as_absent = len(df_openmeteo)
        else:
            df_combined = df_euskalmet
    else:
        df_combined = df_euskalmet

    log.info(
        "📊 Plan B: %d municipios completados desde Open-Meteo; %d municipios ausentes añadidos.",
        completed_from_open,
        added_as_absent,
    )

    if df_combined is None:
        df_combined = pd.DataFrame()
    if not df_combined.empty:
        df_combined = df_combined.drop_duplicates("muni_code", keep="first")
        df_combined = geo_municipalities.merge(
            df_combined.drop(columns=["municipality"], errors="ignore"),
            on="muni_code",
            how="left",
        )
    else:
        df_combined = geo_municipalities.copy()
    for column in critical_cols + ["fwi_proxy"]:
        if column not in df_combined:
            df_combined[column] = None
    df_combined["date"] = target_date
    log.info("✅ Dataset meteorológico consolidado para %d municipios.", len(df_combined))
    if len(df_combined) != len(geo_municipalities):
        log.error(
            "❌ El dataset final contiene %d filas para %d municipios GeoJSON.",
            len(df_combined),
            len(geo_municipalities),
        )

    # Step 4: Inferencia ML
    df_analyzed = predict_fire_risk(df_combined)

    if not df_analyzed.empty:
        save_risk_predictions(df_analyzed)

    return df_analyzed


GetLiveWeatherAnalisis = get_live_weather_analysis


def _ensure_logging() -> None:
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(ch)


if __name__ == "__main__":
    results = GetLiveWeatherAnalisis()
    if not results.empty:
        print(f"\nEvaluación finalizada con éxito para {len(results)} municipios.")