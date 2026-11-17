"""
Shared feature builder + calibrated model wrapper for the wildfire model.

ONE source of truth: the training script (trainModelV3.py) AND the app (main.py,
fireTrainingModel.py) must call build_features() so both see exactly the same numbers.

Input of build_features(): ONE location, one row per calendar day, oldest first, columns
    date, temperature, temp_max, temp_min, humidity, min_humidity,
    wind_speed (daily MEAN), max_wind_speed (daily MAX), precip_1d, et0
Output: the same index order with `date` + every column in ALL_FEATURES.

History needed in the app: the Drought Code and the 60/90-day features have long memory.
Give build_features() at least MIN_HISTORY_DAYS (120) days ending on the target day, then
take the LAST row. With only 30 days the FWI-system codes are not reliable.

Fire Weather Index system: Van Wagner (1987), standard day-length tables (46N, a fair
approximation at 43N). Daily inputs approximate the noon observations:
    temperature = temp_max, relative humidity = min_humidity, wind = wind_speed (daily mean).
"""
import math

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "date", "temperature", "temp_max", "temp_min", "humidity", "min_humidity",
    "wind_speed", "max_wind_speed", "precip_1d", "et0",
]

# The 18 features of model v2 (same names, same order, same definitions).
BASE_FEATURES = [
    "temperature", "temp_max", "temp_min", "humidity", "min_humidity", "wind_speed",
    "max_wind_speed", "precip_1d", "precip_7d", "precip_14d", "precip_30d", "dry_spell_days",
    "temp_anomaly", "et0_14d", "fwi_proxy", "vpd", "month", "day_of_year",
]
FWI_FEATURES = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]
EXTRA_FEATURES = [
    "tmax_3d", "vpd_3d", "vpd_7d", "minhum_3d", "wind_max_3d", "hot_days_7d",
    "days_since_rain5", "precip_60d", "precip_90d", "tmax_anom_30d", "vpd_anom_30d",
]
ALL_FEATURES = BASE_FEATURES + FWI_FEATURES + EXTRA_FEATURES
LOCATION_FEATURES = ["latitude", "longitude"]   # static: the app supplies them for every point

WARMUP_DAYS = 120        # training drops this many first days of each series (code spin-up)
MIN_HISTORY_DAYS = 120   # what the app must feed to build_features()

# Day-length adjustment tables (index = month 1..12)
_LE = [None, 6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.2, 8.0, 7.0, 6.0]       # DMC
_LF = [None, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]       # DC


# ---------------------------------------------------------------------------
# Canadian Fire Weather Index system
# ---------------------------------------------------------------------------
def fwi_day(t, rh, w, r, month, ffmc0, dmc0, dc0):
    """One day of the FWI system. Returns (ffmc, dmc, dc, isi, bui, fwi)."""
    rh = min(max(rh, 0.0), 100.0)
    w = max(w, 0.0)
    r = max(r, 0.0)

    # --- Fine Fuel Moisture Code
    mo = 147.2 * (101.0 - ffmc0) / (59.5 + ffmc0)
    if r > 0.5:
        rf = r - 0.5
        mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        if mo > 150.0:
            mr += 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf)
        mo = min(mr, 250.0)
    ed = (0.942 * rh ** 0.679 + 11.0 * math.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - t) * (1.0 - math.exp(-0.115 * rh)))
    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * math.sqrt(w) * (1.0 - (rh / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * t)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = (0.618 * rh ** 0.753 + 10.0 * math.exp((rh - 100.0) / 10.0)
              + 0.18 * (21.1 - t) * (1.0 - math.exp(-0.115 * rh)))
        if mo < ew:
            k1 = (0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7)
                  + 0.0694 * math.sqrt(w) * (1.0 - ((100.0 - rh) / 100.0) ** 8))
            kw = k1 * 0.581 * math.exp(0.0365 * t)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    ffmc = min(max(59.5 * (250.0 - m) / (147.2 + m), 0.0), 101.0)

    # --- Duff Moisture Code
    rk = 1.894 * (max(t, -1.1) + 1.1) * (100.0 - rh) * _LE[month] * 1e-4
    p0 = dmc0
    if r > 1.5:
        re = 0.92 * r - 1.27
        mo_d = 20.0 + math.exp(5.6348 - p0 / 43.43)
        if p0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * p0)
        elif p0 <= 65.0:
            b = 14.0 - 1.3 * math.log(p0)
        else:
            b = 6.2 * math.log(p0) - 17.2
        mr_d = mo_d + 1000.0 * re / (48.77 + b * re)
        p0 = max(244.72 - 43.43 * math.log(mr_d - 20.0), 0.0)
    dmc = p0 + rk

    # --- Drought Code
    v = max(0.36 * (max(t, -2.8) + 2.8) + _LF[month], 0.0)
    d0 = dc0
    if r > 2.8:
        rd = 0.83 * r - 1.27
        qr = 800.0 * math.exp(-d0 / 400.0) + 3.937 * rd
        d0 = max(400.0 * math.log(800.0 / qr), 0.0)
    dc = d0 + 0.5 * v

    # --- Initial Spread Index
    fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    isi = 0.208 * math.exp(0.05039 * w) * 91.9 * math.exp(-0.1386 * fm) * (1.0 + fm ** 5.31 / 4.93e7)

    # --- Buildup Index and Fire Weather Index
    denom = dmc + 0.4 * dc
    if denom <= 0.0:
        u = 0.0
    elif dmc <= 0.4 * dc:
        u = 0.8 * dmc * dc / denom
    else:
        u = dmc - (1.0 - 0.8 * dc / denom) * (0.92 + (0.0114 * dmc) ** 1.7)
    u = max(u, 0.0)
    fd = 0.626 * u ** 0.809 + 2.0 if u <= 80.0 else 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * u))
    bb = 0.1 * isi * fd
    fwi = math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647) if bb > 1.0 else bb
    return ffmc, dmc, dc, isi, u, fwi


def fwi_series(temp, rh, wind, rain, months, start=(85.0, 6.0, 15.0)):
    """FWI codes for a whole daily series. Days with missing inputs carry the previous values."""
    n = len(temp)
    out = np.full((n, 6), np.nan)
    f, p, d = start
    isfinite = math.isfinite
    for i in range(n):
        t, h, w, r = temp[i], rh[i], wind[i], rain[i]
        if not (isfinite(t) and isfinite(h) and isfinite(w) and isfinite(r)):
            if i:
                out[i] = out[i - 1]
            continue
        res = fwi_day(t, h, w, r, int(months[i]), f, p, d)
        f, p, d = res[0], res[1], res[2]
        out[i] = res
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_COLUMNS if c not in daily.columns]
    if missing:
        raise ValueError(f"build_features: missing input columns {missing}")
    df = daily[RAW_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    tx, minh, wx = df["temp_max"], df["min_humidity"], df["max_wind_speed"]
    p1 = df["precip_1d"]
    out = pd.DataFrame({"date": df["date"]})
    for c in ("temperature", "temp_max", "temp_min", "humidity", "min_humidity",
              "wind_speed", "max_wind_speed", "precip_1d"):
        out[c] = df[c]

    # --- the 18 features of v2 (identical definitions)
    for n in (7, 14, 30):
        out[f"precip_{n}d"] = p1.rolling(n, min_periods=1).sum()
    is_dry = (p1 < 0.1).astype(int)
    out["dry_spell_days"] = is_dry.groupby((is_dry == 0).cumsum()).cumsum()
    out["temp_anomaly"] = df["temperature"] - df["temperature"].rolling(30, min_periods=1).mean()
    out["et0_14d"] = df["et0"].rolling(14, min_periods=1).sum()
    svp = 0.6108 * np.exp((17.27 * tx) / (tx + 237.3))
    vpd = np.maximum(0, svp * (1 - minh.clip(0, 100) / 100.0))
    out["vpd"] = vpd
    out["fwi_proxy"] = tx * wx / np.maximum(minh, 1.0) / (1.0 + np.log1p(np.maximum(out["precip_7d"], 0.0)))
    out["month"] = df["date"].dt.month
    out["day_of_year"] = df["date"].dt.day_of_year

    # --- Fire Weather Index system
    codes = fwi_series(tx.tolist(), minh.tolist(), df["wind_speed"].tolist(), p1.tolist(),
                       out["month"].tolist())
    for k, name in enumerate(FWI_FEATURES):
        out[name] = codes[:, k]

    # --- multi-day / long-memory features
    out["tmax_3d"] = tx.rolling(3, min_periods=1).mean()
    out["vpd_3d"] = vpd.rolling(3, min_periods=1).mean()
    out["vpd_7d"] = vpd.rolling(7, min_periods=1).mean()
    out["minhum_3d"] = minh.rolling(3, min_periods=1).mean()
    out["wind_max_3d"] = wx.rolling(3, min_periods=1).max()
    out["hot_days_7d"] = (tx >= 30).astype(float).rolling(7, min_periods=1).sum()
    heavy = (p1 >= 5).astype(int)
    out["days_since_rain5"] = (1 - heavy).groupby(heavy.cumsum()).cumsum().clip(upper=90)
    out["precip_60d"] = p1.rolling(60, min_periods=1).sum()
    out["precip_90d"] = p1.rolling(90, min_periods=1).sum()
    out["tmax_anom_30d"] = tx - tx.rolling(30, min_periods=1).mean()
    out["vpd_anom_30d"] = vpd - vpd.rolling(30, min_periods=1).mean()
    return out[["date"] + ALL_FEATURES]


# ---------------------------------------------------------------------------
# Model wrapper: predict_proba returns probabilities on the REAL base rate
# ---------------------------------------------------------------------------
def correct_prior(p, beta):
    """Undo the effect of keeping only a fraction `beta` of the non-fire days when training."""
    p = np.asarray(p, dtype=float)
    return beta * p / (beta * p + (1.0 - p))


class FireRiskModel:
    """
    Wraps one forest (or two, one for summer months) and applies the prior correction, so the
    app can call predict_proba() and compare with tier_thresholds directly.
    `main` is used for every month except `summer_months` when `summer` is given.
    """

    def __init__(self, main, beta_main, summer=None, beta_summer=None, summer_months=(6, 7, 8, 9)):
        self.main, self.beta_main = main, float(beta_main)
        self.summer = summer
        self.beta_summer = None if beta_summer is None else float(beta_summer)
        self.summer_months = tuple(summer_months)
        self.feature_names_in_ = np.asarray(main.feature_names_in_)
        self.n_features_in_ = len(self.feature_names_in_)
        self.classes_ = np.array([0, 1])
        # gradient-boosting models have no feature_importances_; only expose it when available
        imps = [m.feature_importances_ for m in (main, summer)
                if m is not None and hasattr(m, "feature_importances_")]
        if imps:
            self.feature_importances_ = np.mean(imps, axis=0)

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        X = X[list(self.feature_names_in_)]
        p = correct_prior(self.main.predict_proba(X)[:, 1], self.beta_main)
        if self.summer is not None:
            mask = X["month"].isin(self.summer_months).to_numpy()
            if mask.any():
                p[mask] = correct_prior(self.summer.predict_proba(X[mask])[:, 1], self.beta_summer)
        return np.column_stack([1.0 - p, p])