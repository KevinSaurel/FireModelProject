# Wildfire Risk Model v3 — how it works

Project: FireModelProject (PFG) · Region: northern Spain (Euskadi + neighbouring Asturias, Cantabria, Navarra) · Final model: **variant H** (gradient boosting, 35 weather features + latitude/longitude)

---

## 1. What the model answers

> "Given the weather of the last ~4 months at this 0.25° grid cell, how likely is it that a wildfire is recorded there **today**?"

- **One prediction = one grid cell × one day.** A cell is a 0.25° × 0.25° square (≈ 28 km N–S × 20 km E–W at 43°N). 106 cells are used (the ones near recorded fires).
- **Output**: a probability between 0 and 1 (`0.05` = 5 %), later turned into a tier: LOW / MEDIUM / HIGH / EXTREME.
- A "fire" is any fire in the MITECO records (EGIF/PIF) located in that cell on that day — any size, any cause. 53 % of records are intentional burns, mostly livestock burns in Asturias in March; in summer the share of lightning (87 % of lightning fires are in Jun–Sep) and smokers (69 %) is much higher.
- Important: the probability is a **relative ranking tool** more than a literal chance (see §8).

---

## 2. Data

| Piece | Source | Notes |
|---|---|---|
| Fire records | MITECO Excel (`Xlsx_..._1.xlsx`, sheet `Informe`, 61,912 rows) | Detection date `Detectado`, coordinates, cause. **Ends on 2023-12-31.** |
| Weather | ERA5 reanalysis through Open-Meteo archive API | Daily values per cell, 2010-01-01 → today, cached in `Data/MITECO/weather_cache/cell_<lat>_<lon>.csv` |
| Labels | `weather_all_days_natural.csv` | One row per (date, cell) with `fire_risk` = 1 if ≥1 fire in that cell that day |

Size of the dataset used by v3: 639,252 cell-days (106 cells), of which 529,258 up to 2023-12-31. About 3 % are fire cell-days.
Days after 2023-12-31 are dropped in training because there are no fire records for them (they would all look like "no fire" and teach the model that nothing burns).

The first 120 days of every cell are skipped (warm-up of the fire-weather codes).

---

## 3. Inputs (the 37 values the model receives)

All features are computed by **one function**, `Model/fireFeatures.py → build_features()`, used by both the training script and the app, so they cannot drift apart.

**Raw daily inputs (10 columns)** for each cell, one row per day, oldest first:
`date, temperature (daily mean), temp_max, temp_min, humidity (mean), min_humidity, wind_speed (daily mean), max_wind_speed (daily max), precip_1d, et0 (evapotranspiration)`

### 3.1 The 18 v2 features (same names and definitions as v2)

| Feature | Meaning |
|---|---|
| `temperature`, `temp_max`, `temp_min` | Daily mean / max / min air temperature (°C) |
| `humidity`, `min_humidity` | Daily mean / minimum relative humidity (%) |
| `wind_speed`, `max_wind_speed` | Daily mean / max wind at 10 m (km/h) |
| `precip_1d` | Rain that day (mm) |
| `precip_7d`, `precip_14d`, `precip_30d` | Rain accumulated over the last 7 / 14 / 30 days |
| `dry_spell_days` | Consecutive days with < 0.1 mm rain |
| `temp_anomaly` | Today's mean temperature minus the average of the last 30 days |
| `et0_14d` | Evapotranspiration summed over 14 days (how much the ground has been drying) |
| `vpd` | Vapour-pressure deficit: how thirsty the air is. `0.6108·exp(17.27·Tmax/(Tmax+237.3))·(1 − min_humidity/100)` |
| `fwi_proxy` | Home-made danger index: `Tmax · max_wind / max(min_humidity,1) / (1 + ln(1 + precip_7d))` |
| `month`, `day_of_year` | Calendar (the model learns the season from these) |

### 3.2 Canadian Fire Weather Index system (6 features)

The standard fire-danger system used by fire agencies worldwide (Van Wagner 1987), computed day by day with memory of past days:

| Feature | Meaning | Memory |
|---|---|---|
| `ffmc` | Fine Fuel Moisture Code — dryness of leaf litter / grass; reacts in hours | ~1 day |
| `dmc` | Duff Moisture Code — dryness of loose organic layers | ~2 weeks |
| `dc` | Drought Code — deep drought of heavy fuels / soil | ~2–3 months |
| `isi` | Initial Spread Index — how fast a fire would spread (FFMC + wind) | — |
| `bui` | Build-Up Index — total fuel available (DMC + DC) | — |
| `fwi` | Fire Weather Index — final intensity rating | — |

Approximation: noon values are replaced by `temp_max`, `min_humidity` and the daily mean wind. Day-length tables are for 46°N (fair at 43°N).

### 3.3 Multi-day / long-memory features (11)

`tmax_3d`, `vpd_3d`, `vpd_7d`, `minhum_3d` (3-/7-day averages), `wind_max_3d` (max wind of the last 3 days), `hot_days_7d` (days ≥ 30 °C in the last week), `days_since_rain5` (days since ≥ 5 mm, capped at 90), `precip_60d`, `precip_90d` (long rain totals), `tmax_anom_30d`, `vpd_anom_30d` (today vs the last 30 days).

### 3.4 Location (2)

`latitude`, `longitude` of the cell centre, snapped to the 0.25° grid. **At serving time the app must use the same snapped centres** (e.g. `round(x*4)/4` as in `fetch_exact_weather.py`), never raw coordinates.

**Total: 18 + 6 + 11 + 2 = 37 inputs.** (35 weather + 2 location.) The exact order is saved in `Model/fireModel_v3_meta.json`.

**History the app must provide:** at least **120 days** of daily weather ending on the target day (the Drought Code and 60/90-day features need it). We tested 120 vs 240 days: probabilities change by ~1 % and the ranking is identical (rank correlation 0.999), so 120 is enough. The old app used 30 days, which is not enough for v3.

---

## 4. Output

| Output | Description |
|---|---|
| `predict_proba(X)[:, 1]` | Probability of ≥ 1 recorded fire in that cell that day, **already corrected to the real fire rate** (see §6) |
| Tier | From thresholds in `Model/tier_thresholds_v3.json` |
| `fireModel_v3_meta.json` | Variant, feature list/order, history required, whether lat/lon are needed |

Files produced by `trainModelV3.py`:
`Model/fireTrainModel_v3.pkl` (the model wrapper), `Model/tier_thresholds_v3.json`, `Model/fireModel_v3_meta.json`.

The pickle contains a `FireRiskModel` object defined in `Model/fireFeatures.py`; that file must be importable (as `Model.fireFeatures`, exact capitalisation) whenever the pickle is loaded.

### Tiers

Thresholds are quantiles of the validation-year predictions (a "percentile band", not an absolute danger scale):

| Tier | Rule | Share of days |
|---|---|---|
| LOW | below the 60th percentile | ~60 % |
| MEDIUM | 60th–85th | ~25 % |
| HIGH | 85th–95th | ~10 % |
| EXTREME | above the 95th | ~5 % |

Two sets are saved: **all-year** (top level of the JSON) and **summer** (`"summer"` key, computed inside Jun–Sep only). Use the summer set for months 6–9 so that a hot July day is compared with other summer days, not with January.

Latest thresholds (from the `--final` run; they change a little at every retrain):
all-year `low 0.0139 / medium 0.0669 / high 0.1816`; summer `low 0.0159 / medium 0.0584 / high 0.1224`.

---

## 5. How the model works (the learning part)

### 5.1 Variant H = gradient boosting

Gradient boosting builds many small decision trees **one after another**; each new tree tries to fix the mistakes of all the previous ones, and their outputs are added up into one score (turned into a probability). Contrast with a random forest, where trees are independent and are averaged.

Settings (scikit-learn `HistGradientBoostingClassifier`):

| Parameter | Value | What it does |
|---|---|---|
| `max_iter` | 300 | Up to 300 trees (rounds). Early stopping (automatic for > 10k rows) may stop sooner when an internal 10 % check set stops improving |
| `learning_rate` | 0.05 | Each tree contributes only 5 % of its correction → slower but more robust |
| `max_depth` | 6 | Each tree asks at most 6 questions in a row |
| `min_samples_leaf` | 40 | A leaf must contain ≥ 40 examples (stops memorising single days) |
| `l2_regularization` | 1.0 | Penalises extreme leaf values |
| `random_state` | 42 | Reproducibility |

So the "forest size" of H is **≤ 300 shallow trees added in sequence**. (The other variants use a Random Forest: 300 trees, `max_depth 13`, `min_samples_leaf 20`, `min_samples_split 40`, `max_features = sqrt` = about 6 features tried per split — `--trees` changes the 300.)

### 5.2 Why latitude/longitude help (and what they actually do)

With lat/lon the model learns **which cells burn more**: e.g. Asturias cells have many more recorded fires than the Basque coast, partly because of land use and burning practices, not only weather. Weather then decides *when* the fire-prone cells flare up. Gradient boosting can use these two numbers precisely, which is why H beats the same features in a forest (F).

Consequence: part of the score is a **fixed spatial base level**. A cell with no recorded fire history will inherit the level of nearby cells.

### 5.3 Training procedure

1. Build the 37 features for every cell-day.
2. Split **by year**, never randomly: train 2010–2018, validation 2019–2020, test 2021–2023 (the `--final` run: train 2010–2020, validation 2021–2022, test 2023 only).
3. **Undersample** the training set to 1 fire : 5 non-fire days (all fire days are kept; non-fire days are randomly subsampled, keeping ~17.5 %). Validation and test keep the real, unbalanced rate.
4. Fit the model.
5. **Prior correction**: because non-fire days were thinned, raw probabilities are too high. The wrapper applies `p' = β·p / (β·p + 1 − p)` with β = fraction of non-fire days kept (≈ 0.175). This is built into `FireRiskModel.predict_proba`, so callers never do it.
6. Compute tier thresholds from validation predictions.
7. Evaluate on the untouched test years, with a paired bootstrap over days to tell real gains from noise.

### 5.4 The seven variants compared

| Variant | Features | Learner |
|---|---|---|
| A | 18 (v2 set) | Random Forest |
| B | A + FWI system | RF |
| C | A + multi-day | RF |
| D | all 35 weather features | RF |
| E | D with separate Jun–Sep forest (`--seasonal`) | RF |
| F | D + lat/lon | RF |
| G | D | Gradient boosting |
| **H** | **D + lat/lon** | **Gradient boosting** |

**H is chosen automatically** (`--select balanced`, best average of validation ROC over all days and over Jun–Sep). You understood correctly: the final v3 is variant **H**.

---

## 6. Results

Final check on 2023 (never used for choosing anything; model trained on 2010–2020, thresholds from 2021–2022):

| | D (no location) | F (RF + loc) | **H** |
|---|---|---|---|
| ROC-AUC, all days | 0.836 | 0.904 | **0.927** |
| ROC-AUC, Jun–Sep | 0.690 | 0.840 | **0.860** |
| PR-AUC, all days (base 0.024) | 0.157 | 0.356 | **0.428** |
| PR-AUC, Jun–Sep (base 0.013) | 0.037 | 0.122 | **0.137** |

Bootstrap gain of H over D: +0.090 ROC all days, +0.171 in Jun–Sep (both "real").

Fire rate per tier in 2023, summer thresholds, Jun–Sep: LOW 0.26 % → MEDIUM 1.27 % → HIGH 2.97 % → EXTREME 13.0 %. Rates rise with the tier, which is what a good tier system must do.

Day-level ("was there any fire in the region that day?"): ROC 0.78–0.79; for "3+ fire cells": 0.77–0.80.

Metric guide: **ROC-AUC** = chance a random fire cell-day is scored higher than a random non-fire one (0.5 = coin, 1 = perfect). **PR-AUC** = precision/recall trade-off, compare with the base rate (a random model scores the base rate).

---

## 7. What changed vs v1 and v2

| | **v1** (original) | **v2** | **v3** |
|---|---|---|---|
| Fire data | ~248 fire days; one label per day copied to every cell; old weather rows (1971–92, constant wind in parts) | MITECO + ERA5: 18,029 real fire cell-days; one label per cell-day | Same data, but days without fire records (after 2023) removed |
| Weather source | Open-Meteo calls at runtime | ERA5 grid cache | Same cache |
| Evaluation | Random row split → leaked neighbouring days; reported ROC 0.93 was inflated | Split by year; honest ROC ≈ 0.79, PR 0.066 | Split by year + bootstrap + untouched final year |
| Features | 18, some zero-filled when missing, mismatched FWI fallback | 18 | 37 (FWI system, multi-day memory, location) |
| Model | Random Forest | Random Forest | Gradient boosting (H) |
| Class balance | Weighting | 1:5 undersampling + prior correction (but saved **uncorrected** probabilities against corrected thresholds — a bug) | 1:5 + correction built into the saved model |
| Tiers | Fixed 0.35 / 0.50 / 0.70 hard-coded in the HTML | Validation quantiles (all-year) | Validation quantiles, all-year **and** summer-specific |
| Summer behaviour | Low everywhere | Summer ROC ≈ 0.69, EXTREME almost never reached | Summer ROC ≈ 0.83–0.86, summer EXTREME ≈ 5 % of days |
| App history | 30 days | 30 days | ≥ 120 days |
| Shared code | Separate in training and app | Separate | One `build_features()` for both |

Why v1 looked good but failed on real summer dates: random splitting put neighbouring days in train and test, and the one-label-per-day data could not tell cells apart. The real difference between hot and ordinary days was never measured.

Why v2 → v3 gains: (a) +0.015 ROC from FWI and multi-day features, small but statistically real; (b) the big jump (+0.09 ROC) comes from location + gradient boosting.

---

## 8. Limits to know about

1. **Absolute probabilities are about 1.2–1.7× too high** on recent years (2023: predicted mean 4.0 % vs actual 2.4 %). Fire counts have been drifting down, and the model learned the older, higher rate. Present the output as **risk level / relative ranking**, not "exactly N % chance".
2. **Euskadi evidence is thinner.** About 60 % of records are in Asturias; Euskadi has ~3,000 records. In 2023 Euskadi had only 13 summer fire cell-days, so metrics there are noisy. In the earlier test years H was better than the no-location model for Euskadi (ROC 0.82 vs 0.76); in 2023 it was about equal (0.83 vs 0.85). Treat it as "good, not proven" for the Basque Country alone.
3. **It predicts recorded fires of all causes.** Intentional burns cannot be predicted from weather; they set an upper limit on accuracy. Weather explains lightning/smoker/escaped-burn days best.
4. **Spatial prior.** Cells far from the 106 trained cells have no history of their own.
5. **ERA5 is ~5–6 days behind real time** in the archive API; for "live" risk the app uses forecast/recent data, which differs slightly from ERA5. Keep that in mind when comparing live and historical results.
6. **Fire records end 2023-12-31.** Retrain when MITECO publishes newer years.
7. A fire is a record, not a size: a 0.01 ha burn and a 500 ha fire both count as 1.

---

## 9. How to use / retrain

```
python .\Model\trainModelV3.py --trees 100 --bootstrap 30     # compare variants, saves the best (H)
python .\Model\trainModelV3.py --final                        # one-time test on the untouched year
python .\Model\trainModelV3.py --variant H --date 2022-07-18  # force H and score one date
python .\Model\trainModelV3.py --seasonal                     # also train variant E
python .\Data\MITECO\inspectFires.py                          # inspect the fire Excel
```

Loading in the app:

```python
import joblib, json
from Model.fireFeatures import build_features        # must be importable
model = joblib.load("Model/fireTrainModel_v3.pkl")
meta  = json.load(open("Model/fireModel_v3_meta.json"))
thr   = json.load(open("Model/tier_thresholds_v3.json"))

feats = build_features(last_120_plus_days_of_weather_for_one_cell).tail(1)
feats["latitude"], feats["longitude"] = cell_lat, cell_lon      # snapped to 0.25°
p = model.predict_proba(feats[meta["features"]])[:, 1][0]
t = thr["summer"] if feats["month"].iloc[0] in (6, 7, 8, 9) else thr
tier = "LOW" if p <= t["low"] else "MEDIUM" if p <= t["medium"] else "HIGH" if p <= t["high"] else "EXTREME"
```

Do **not** use `scaler.pkl`: v3 was never trained with a scaler.

Note on the saved file: a `--final` run saves a model trained on 2010–2020 only. For the production model, retrain on all years up to 2023 (planned `--production` run) and keep thresholds from out-of-sample predictions.

---

## 10. Glossary

- **Cell**: 0.25° × 0.25° grid square; the unit of prediction.
- **Cell-day**: one cell on one day; one row of data.
- **Undersampling**: keeping only a share of the majority class (non-fire days) in training.
- **Prior correction**: formula that restores real-world probabilities after undersampling.
- **Validation / test**: validation picks the variant and thresholds; test is only for the final honest score.
- **ROC-AUC / PR-AUC**: ranking-quality scores (see §6).
- **VPD**: vapour-pressure deficit; how strongly the air pulls moisture out of plants.
- **FWI system**: Canadian Fire Weather Index codes (FFMC, DMC, DC, ISI, BUI, FWI).
