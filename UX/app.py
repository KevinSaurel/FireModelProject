"""
Panel Streamlit de Evaluación de Riesgo de Incendios en Euskadi.
"""

import json
import logging
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from Interface.implementacion import GetLiveWeatherAnalisis

GEOJSON_PATH = BASE_DIR / "Data" / "GeoJSON" / "municipios-euskadi.geojson"
log = logging.getLogger("streamlit_app")

st.set_page_config(
    page_title="Euskadi Fire Risk Assessment",
    page_icon="🔥",
    layout="wide",
)

COLOR_MAP = {
    "Low": "#2ecc71",       # Verde
    "Medium": "#f1c40f",    # Amarillo
    "High": "#e67e22",      # Naranja
    "Extreme": "#e74c3c",   # Rojo
    "Sin datos": "#95a5a6", # Gris
}


def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]", "", ascii_name).lower()


@st.cache_data(ttl=300)
def fetch_cached_analysis():
    return GetLiveWeatherAnalisis()


@st.cache_data(ttl=3600)
def load_and_prepare_geojson(df_municipalities: list[str]):
    if not GEOJSON_PATH.exists():
        st.error(f"GeoJSON missing at: {GEOJSON_PATH}")
        return None

    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    muni_lookup = {str(code) for code in df_municipalities}

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        code = str(props.get("ud_kodea", ""))
        feature["properties"]["matched_muni"] = code if code else None

    unmatched = [
        feature["properties"].get("iz_ofizial") or feature["properties"].get("ud_iz_e")
        for feature in geojson.get("features", [])
        if str(feature["properties"].get("fazeria", "0")) == "0"
        and feature["properties"].get("matched_muni") not in muni_lookup
    ]
    if unmatched:
        log.warning("Municipios del GeoJSON sin datos coincidentes: %s", unmatched)

    return geojson


def build_map_dataframe(df: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    """Completa el dataset del mapa con todos los polígonos del GeoJSON."""
    polygons = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        code = properties.get("matched_muni")
        if code:
            polygons.append({
                "muni_code": str(code),
                "municipality": (
                    properties.get("iz_ofizial")
                    or properties.get("iz_euskal")
                    or properties.get("ud_iz_e")
                ),
            })
    map_df = pd.DataFrame(polygons).drop_duplicates("muni_code")
    map_df = map_df.merge(
        df.drop(columns=["municipality"], errors="ignore"),
        on="muni_code",
        how="left",
    )
    map_df["risk_tier"] = map_df["risk_tier"].fillna("Sin datos")
    return map_df


def build_choropleth_fig(df: pd.DataFrame, geojson: dict):
    hover_cols = {
        "temperature": ":.1f °C",
        "wind_speed": ":.1f km/h",
        "fwi_proxy": ":.2f",
        "risk_probability": ":.2f",
        "municipality": False,
    }

    map_func = getattr(px, "choropleth_map", getattr(px, "choropleth_mapbox", None))
    style_key = "map_style" if hasattr(px, "choropleth_map") else "mapbox_style"

    kwargs = {
        "data_frame": df,
        "geojson": geojson,
        "locations": "muni_code",
        "featureidkey": "properties.matched_muni",
        "color": "risk_tier",
        "color_discrete_map": COLOR_MAP,
        "category_orders": {"risk_tier": ["Low", "Medium", "High", "Extreme", "Sin datos"]},
        "hover_name": "municipality",
        "hover_data": hover_cols,
        "center": {"lat": 43.15, "lon": -2.5},
        "zoom": 8.3,
        style_key: "carto-positron",
    }

    fig = map_func(**kwargs)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=550)
    return fig


def main():
    st.title("🔥 Sistema de Evaluación de Riesgo de Incendios Forestales (Euskadi)")
    st.markdown("Integración meteorológica multifuente (Euskalmet + Open-Meteo) e inferencia mediante Machine Learning.")

    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner(
        "Analizando observaciones meteorológicas (Euskalmet y Open-Meteo) y calculando riesgo..."
    ):
        df = fetch_cached_analysis()

    if df is None or df.empty:
        st.error("No se pudo obtener el análisis de riesgo.")
        return

    col1, col2, col3, col4 = st.columns(4)
    total_munis = len(df)
    evaluados = len(df[df["risk_tier"] != "Sin datos"])
    max_risk_row = df.dropna(subset=["risk_probability"])
    max_muni = max_risk_row.loc[max_risk_row["risk_probability"].idxmax()] if not max_risk_row.empty else None
    latest_update = df["prediction_time"].iloc[0]

    col1.metric("Municipios Evaluados", f"{evaluados} / {total_munis}")
    col2.metric("Zonas de Riesgo Alto/Extremo", f"{len(df[df['risk_tier'].isin(['High', 'Extreme'])])}")
    col3.metric("Punto de Mayor Riesgo", f"{max_muni['municipality'] if max_muni is not None else 'N/A'}", f"{max_muni['risk_probability']:.1%}" if max_muni is not None else "N/A")
    col4.metric("Última Actualización", f"{latest_update.split()[1]}")

    st.divider()

    map_col, table_col = st.columns([2, 1])
    geojson = load_and_prepare_geojson(df["muni_code"].dropna().unique().tolist())
    df_map = build_map_dataframe(df, geojson) if geojson else df

    with map_col:
        st.subheader("🗺️ Mapa Regional de Riesgo de Incendios")
        if geojson:
            fig = build_choropleth_fig(df_map, geojson)
            st.plotly_chart(fig, width="stretch")

    with table_col:
        st.subheader("📋 Detalle Municipal")
        selected_tier = st.multiselect(
            "Filtrar por Nivel de Riesgo",
            options=["Low", "Medium", "High", "Extreme", "Sin datos"],
            default=["Low", "Medium", "High", "Extreme", "Sin datos"],
        )

        filtered_df = df[df["risk_tier"].isin(selected_tier)].sort_values(
            "risk_probability", ascending=False, na_position="last"
        )

        st.dataframe(
            filtered_df[["municipality", "temperature", "wind_speed", "risk_probability", "risk_tier"]],
            column_config={
                "municipality": "Municipio",
                "temperature": st.column_config.NumberColumn("Temp (°C)", format="%.1f"),
                "wind_speed": st.column_config.NumberColumn("Viento (km/h)", format="%.1f"),
                "risk_probability": st.column_config.ProgressColumn(
                    "Prob. Riesgo", format="%.2f", min_value=0.0, max_value=1.0
                ),
                "risk_tier": "Nivel de Riesgo",
            },
            hide_index=True,
            width="stretch",
            height=480,
        )


if __name__ == "__main__":
    main()