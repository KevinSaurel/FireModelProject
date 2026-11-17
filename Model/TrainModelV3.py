"""
Wildfire model v3 (iteration harness): richer features, location, gradient boosting, season-aware
tiers, and a "showcase" report of real summer days.

Put fireFeatures.py and this file in  <project>/Model/ . Run from the project root:
    python .\\Model\\trainModelV3.py                      # all variants, default settings
    python .\\Model\\trainModelV3.py --trees 100          # quicker
    python .\\Model\\trainModelV3.py --variant G          # force which variant is saved
    python .\\Model\\trainModelV3.py --final              # ONE-TIME check on the untouched years
    python .\\Model\\trainModelV3.py --production         # FINAL model for the app (variant H, all years)

Reads   Data/MITECO/weather_cache/cell_*.csv        raw daily weather per cell (already downloaded)
        Data/MITECO/weather_all_days_natural.csv    only for the fire labels
Writes  Model/fireTrainModel_v3.pkl, Model/tier_thresholds_v3.json, Model/fireModel_v3_meta.json

Honest evaluation while we iterate
    * train = years before --val-start, validation = --val-start..--test-start, test = up to --holdout-start.
    * Years from --holdout-start on (default 2023-01-01) are NOT used at all until you run --final,
      so we do not tune against them.
    * --final: train < 2021, validation 2021-2022 (thresholds), test = the untouched years.

Variants (same rows, same split)
    A v2 features (18)          B A + FWI system          C A + multi-day features
    D all 35 features           F D + latitude/longitude  G D with gradient boosting
    H D + location, gradient boosting                     E D with a separate Jun-Sep forest (--seasonal)
The saved model returns probabilities on the real base rate (the correction is built in).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def find_root():
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "Model").is_dir() and (p / "Data").is_dir():
            return p
    return here


ROOT = find_root()
sys.path.insert(0, str(ROOT))
try:
    from Model.fireFeatures import (  # noqa: E402
        ALL_FEATURES, BASE_FEATURES, EXTRA_FEATURES, FWI_FEATURES, LOCATION_FEATURES,
        MIN_HISTORY_DAYS, RAW_COLUMNS, WARMUP_DAYS, FireRiskModel, build_features,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fireFeatures import (  # noqa: E402
        ALL_FEATURES, BASE_FEATURES, EXTRA_FEATURES, FWI_FEATURES, LOCATION_FEATURES,
        MIN_HISTORY_DAYS, RAW_COLUMNS, WARMUP_DAYS, FireRiskModel, build_features,
    )
    print("!! imported fireFeatures as a top-level module. The saved .pkl will only load where "
          "'fireFeatures' is importable. Put the file in <project>/Model/ to avoid this.")

EUSKADI = dict(lat=(42.47, 43.45), lon=(-3.45, -1.73))
SUMMER = (6, 7, 8, 9)
LABELS = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH", "EXTREME"]
RAW_RENAME = {
    "time": "date",
    "temperature_2m_mean": "temperature",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "relative_humidity_2m_mean": "humidity",
    "relative_humidity_2m_min": "min_humidity",
    "wind_speed_10m_mean": "wind_speed",
    "wind_speed_10m_max": "max_wind_speed",
    "precipitation_sum": "precip_1d",
    "et0_fao_evapotranspiration": "et0",
}


def header(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(cache_dir, labels_path, max_cells):
    files = sorted(cache_dir.glob("cell_*.csv"))
    if max_cells:
        files = files[:max_cells]
    if not files:
        raise SystemExit(f"No cell_*.csv files in {cache_dir}")
    frames, skipped, raws = [], [], {}
    for f in files:
        m = re.match(r"cell_(-?\d+\.\d+)_(-?\d+\.\d+)\.csv$", f.name)
        if not m:
            continue
        raw = pd.read_csv(f).rename(columns=RAW_RENAME)
        lacking = [c for c in RAW_RENAME.values() if c not in raw.columns]
        if lacking:
            skipped.append((f.name, lacking))
            continue
        feats = build_features(raw)
        lat, lon = round(float(m.group(1)), 2), round(float(m.group(2)), 2)
        feats["latitude"], feats["longitude"] = lat, lon
        frames.append(feats.iloc[WARMUP_DAYS:])
        raw["date"] = pd.to_datetime(raw["date"])
        raws[(lat, lon)] = raw[RAW_COLUMNS].sort_values("date").reset_index(drop=True)
    if skipped:
        print(f"!! skipped {len(skipped)} cell files with missing columns, e.g. {skipped[0]}. "
              "Delete those files and re-run fetch_exact_weather.py to download them again.")
    feats = pd.concat(frames, ignore_index=True)

    lab = pd.read_csv(labels_path, usecols=["date", "latitude", "longitude", "fire_risk"], parse_dates=["date"])
    lab["latitude"] = lab["latitude"].round(2)
    lab["longitude"] = lab["longitude"].round(2)
    df = feats.merge(lab, on=["date", "latitude", "longitude"], how="inner")
    before = len(df)
    df = df.dropna(subset=ALL_FEATURES + ["fire_risk"])
    df["fire_risk"] = df["fire_risk"].astype(int)
    print(f"cells used: {len(frames)}   rows after joining labels: {before:,}   "
          f"dropped for missing values: {before - len(df):,}   "
          f"first {WARMUP_DAYS} days of every cell skipped (FWI spin-up)")
    return df, raws


def sample_train(df, ratio, seed):
    pos, neg = df[df["fire_risk"] == 1], df[df["fire_risk"] == 0]
    n = min(len(neg), ratio * len(pos))
    keep = neg.sample(n=n, random_state=seed)
    return pd.concat([pos, keep]).sample(frac=1, random_state=seed), n / max(len(neg), 1)


def make_estimator(kind, trees, seed):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=trees, max_depth=13, min_samples_leaf=20, min_samples_split=40,
            max_features="sqrt", n_jobs=-1, random_state=seed)
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6, min_samples_leaf=40,
        l2_regularization=1.0, random_state=seed)


def fit_model(kind, train_df, feats, ratio, trees, seed=42):
    s, beta = sample_train(train_df, ratio, seed)
    est = make_estimator(kind, trees, seed).fit(s[feats], s["fire_risk"])
    return est, beta


# ---------------------------------------------------------------------------
# Metrics and reports
# ---------------------------------------------------------------------------
def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-p)
    out = {"roc": roc_auc_score(y, p), "pr": average_precision_score(y, p), "base": y.mean()}
    for frac in (0.05, 0.10):
        k = max(1, int(len(p) * frac))
        out[f"recall@{int(frac * 100)}%"] = y[order[:k]].sum() / y.sum()
    return out


def f3(m, key):
    return f"{m[key]:.3f}" if m else "  -  "


def tier_of(p, thr):
    return pd.cut(
        p,
        [-1, thr["low"], thr["medium"], thr["high"], thr["very_high"], 2],
        labels=LABELS,
        right=False,
    )


def tier_table(p, y, thr):
    t = pd.DataFrame({"tier": tier_of(p, thr), "fire": y}).groupby("tier", observed=False)["fire"] \
        .agg(["size", "sum", "mean"])
    t["share"] = (t["size"] / max(t["size"].sum(), 1) * 100).round(1)
    t["mean"] = (t["mean"] * 100).round(2)
    t = t[["size", "sum", "mean", "share"]]
    t.columns = ["cell_days", "fire_days", "fire_rate_%", "share_of_days_%"]
    return t.to_string()


def thresholds_from(p):
    return {
        "low": float(np.quantile(p, 0.50)), 
        "medium": float(np.quantile(p, 0.75)),
        "high": float(np.quantile(p, 0.85)),
        "very_high": float(np.quantile(p, 0.95))
    }

def showcase(te, p_te, thr_summer, top=8):
    """What a person sees when they try summer dates."""
    m = te["month"].isin(SUMMER).to_numpy()
    d = te.loc[m, ["date", "fire_risk"]].copy()
    d["p"] = p_te[m]
    d["orange_red"] = d["p"] >= thr_summer["medium"]
    g = d.groupby("date").agg(fire_cells=("fire_risk", "sum"), cells=("fire_risk", "size"),
                              mean_p=("p", "mean"), max_p=("p", "max"), orange_red=("orange_red", "mean"))
    g["pctile"] = g["mean_p"].rank(pct=True) * 100
    show = lambda x: x.assign(mean_p=(x["mean_p"] * 100).round(1), max_p=(x["max_p"] * 100).round(1),
                              orange_red=(x["orange_red"] * 100).round(0), pctile=x["pctile"].round(0)
                              ).rename(columns={"mean_p": "mean_%", "max_p": "max_%",
                                                "orange_red": "cells_orange/red_%", "pctile": "summer_pctile"})
    cols = ["fire_cells", "cells", "mean_%", "max_%", "cells_orange/red_%", "summer_pctile"]
    print(f"Summer test days: {len(g)}.  Days with at least one fire cell: {int((g['fire_cells'] > 0).sum())}")
    print(f"\nThe {top} summer days with MOST fire cells (did the model notice them?):")
    print(show(g.sort_values("fire_cells", ascending=False).head(top))[cols].to_string())
    print(f"\nThe {top} summer days the model rated RISKIEST (how many fires did they have?):")
    print(show(g.sort_values("mean_p", ascending=False).head(top))[cols].to_string())
    for label, y in (("any fire cell that day", g["fire_cells"] > 0), ("3+ fire cells that day", g["fire_cells"] >= 3)):
        if y.nunique() == 2:
            print(f"day-level ROC-AUC for '{label}': mean prob {roc_auc_score(y, g['mean_p']):.3f}, "
                  f"max prob {roc_auc_score(y, g['max_p']):.3f}")


def window_check(df, raws, model, feats, start, n_samples, windows, seed=0):
    """Rebuild features from only the last W days (what the app has) and compare probabilities."""
    cand = df[df["date"] >= start]
    pick = cand.sample(n=min(n_samples, len(cand)), random_state=seed)
    p_cont = model.predict_proba(pick[feats])[:, 1]
    fwi_cols = [c for c in FWI_FEATURES if c in feats]
    print(f"{len(pick)} random test cell-days; mean probability with full history {p_cont.mean():.4f}")
    for w in windows:
        rows = []
        for d, la, lo in zip(pick["date"], pick["latitude"], pick["longitude"]):
            raw = raws[(la, lo)]
            rows.append(build_features(raw[raw["date"] <= d].tail(w)).tail(1))
        wf = pd.concat(rows, ignore_index=True)
        for c in LOCATION_FEATURES:
            wf[c] = pick[c].to_numpy()
        p_win = model.predict_proba(wf[feats])[:, 1]
        rank = pd.Series(p_cont).corr(pd.Series(p_win), method="spearman")
        gaps = "  ".join(f"{c} {np.nanmean(np.abs(wf[c].to_numpy() - pick[c].to_numpy())):.1f}" for c in fwi_cols)
        print(f"window {w:3d} days: mean |change in probability| {np.mean(np.abs(p_cont - p_win)):.4f} "
              f"({np.mean(np.abs(p_cont - p_win)) / max(p_cont.mean(), 1e-9) * 100:.0f}% of the mean), "
              f"rank correlation {rank:.3f}\n                  mean |gap| of FWI codes: {gaps}")


def run_production(df, raws, args):
    """Final model: trained on ALL years with fire records; thresholds from out-of-sample predictions."""
    feats_by_var = {"D": ("rf", ALL_FEATURES), "F": ("rf", ALL_FEATURES + LOCATION_FEATURES),
                    "G": ("gb", ALL_FEATURES), "H": ("gb", ALL_FEATURES + LOCATION_FEATURES)}
    letter = "H" if args.variant == "auto" else args.variant
    if letter not in feats_by_var:
        raise SystemExit("--production supports --variant D, F, G or H (default H).")
    kind, feats = feats_by_var[letter]
    oos_from = pd.Timestamp(args.production_oos_from)
    header(f"PRODUCTION build, variant {letter}")
    # 1) out-of-sample predictions -> thresholds and an honest score
    tr, oos = df[df["date"] < oos_from], df[df["date"] >= oos_from]
    est, beta = fit_model(kind, tr, feats, args.ratio, args.trees)
    m0 = FireRiskModel(est, beta)
    p_oos = m0.predict_proba(oos[feats])[:, 1]
    y_oos = oos["fire_risk"].to_numpy()
    summer_oos = oos["month"].isin(SUMMER).to_numpy()
    thr_all, thr_summer = thresholds_from(p_oos), thresholds_from(p_oos[summer_oos])
    print(f"out-of-sample check: trained {tr['date'].min().date()}..{tr['date'].max().date()}, "
          f"predicted {oos['date'].min().date()}..{oos['date'].max().date()} ({len(oos):,} cell-days)")
    ma, ms = metrics(y_oos, p_oos), metrics(y_oos[summer_oos], p_oos[summer_oos])
    print(f"  ROC all {ma['roc']:.3f}  PR {ma['pr']:.3f}   |   Jun-Sep ROC {ms['roc']:.3f}  PR {ms['pr']:.3f}")
    print(f"  mean predicted {p_oos.mean():.4f} vs actual {y_oos.mean():.4f}")
    print("all-year thresholds:", {k: round(v, 4) for k, v in thr_all.items()})
    print("summer thresholds  :", {k: round(v, 4) for k, v in thr_summer.items()})
    print("\nJun-Sep out-of-sample, summer thresholds:\n" +
          tier_table(p_oos[summer_oos], y_oos[summer_oos], thr_summer))
    # 2) final fit on everything
    est, beta = fit_model(kind, df, feats, args.ratio, args.trees)
    model = FireRiskModel(est, beta)
    print(f"\nfinal model trained on {df['date'].min().date()}..{df['date'].max().date()} "
          f"({int(df['fire_risk'].sum()):,} fire cell-days of {len(df):,})")
    if args.window_check > 0:
        header("Serving check (history window)")
        window_check(df, raws, model, feats, str(oos_from.date()), args.window_check, (MIN_HISTORY_DAYS, 240))
    if args.no_save:
        return
    out_dir = ROOT / "Model"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(model, out_dir / "fireTrainModel_v3.pkl")
    (out_dir / "tier_thresholds_v3.json").write_text(json.dumps(
        {**thr_all, "summer": thr_summer, "summer_months": list(SUMMER)}, indent=2))
    (out_dir / "fireModel_v3_meta.json").write_text(json.dumps({
        "variant": letter, "features": list(feats), "min_history_days": MIN_HISTORY_DAYS,
        "needs_latitude_longitude": any(c in feats for c in LOCATION_FEATURES),
        "production": True, "trained_through": str(df["date"].max().date()),
        "thresholds_from": f"out-of-sample predictions {oos_from.date()}..{df['date'].max().date()}",
        "ratio": args.ratio, "seasonal_model": False,
        "note": "predict_proba already returns probabilities on the real base rate; build features "
                "with Model/fireFeatures.build_features(), add snapped 0.25-degree latitude/longitude; "
                "tier_thresholds_v3.json: all-year at top level, 'summer' set for Jun-Sep.",
    }, indent=2))
    header("Saved (production)")
    print(f"{out_dir / 'fireTrainModel_v3.pkl'}\n{out_dir / 'tier_thresholds_v3.json'}\n{out_dir / 'fireModel_v3_meta.json'}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "Data" / "MITECO" / "weather_cache")
    ap.add_argument("--labels", type=Path, default=ROOT / "Data" / "MITECO" / "weather_all_days_natural.csv")
    ap.add_argument("--val-start", default="2019-01-01")
    ap.add_argument("--test-start", default="2021-01-01")
    ap.add_argument("--data-end", default="2023-12-31",
                    help="last date covered by the MITECO fire records; later days have NO fire records "
                         "(they would all be labelled 'no fire') so they are dropped")
    ap.add_argument("--holdout-start", default="2023-01-01",
                    help="years from here on stay untouched until --final")
    ap.add_argument("--final", action="store_true",
                    help="train <2021, validate 2021-2022, test on the untouched years (use once)")
    ap.add_argument("--production", action="store_true",
                    help="final build: train on ALL years with fire records (variant H by default)")
    ap.add_argument("--production-oos-from", default="2021-01-01",
                    help="thresholds come from predictions of a model trained before this date")
    ap.add_argument("--ratio", type=int, default=5)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--select", choices=["balanced", "overall", "summer"], default="balanced",
                    help="balanced = validation ROC-AUC over all days and over Jun-Sep, averaged")
    ap.add_argument("--variant", default="auto", choices=["auto", "A", "B", "C", "D", "E", "F", "G", "H"])
    ap.add_argument("--baseline", default="D", choices=["A", "B", "C", "D", "F", "G", "H"])
    ap.add_argument("--seasonal", action="store_true", help="also train variant E (separate Jun-Sep forest)")
    ap.add_argument("--bootstrap", type=int, default=100, help="resamples for the gain check (0 = skip)")
    ap.add_argument("--window-check", type=int, default=300)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if args.final:
        args.val_start, args.test_start = "2021-01-01", args.holdout_start
    t0 = time.time()
    header("Data" + ("   [FINAL CHECK on the untouched years]" if args.final else ""))
    df, raws = load_data(args.cache_dir, args.labels, args.max_cells)
    n_after = int((df["date"] > args.data_end).sum())
    df = df[df["date"] <= args.data_end]
    print(f"dropped {n_after:,} rows after {args.data_end} (no fire records exist for those days)")
    if args.production:
        run_production(df, raws, args)
        return
    if not args.final:
        n_hold = int((df["date"] >= args.holdout_start).sum())
        df = df[df["date"] < args.holdout_start]
        print(f"held out (not used): {n_hold:,} rows from {args.holdout_start} on")
    tr = df[df["date"] < args.val_start]
    va = df[(df["date"] >= args.val_start) & (df["date"] < args.test_start)]
    te = df[df["date"] >= args.test_start]
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        print(f"{name:11s} {part['date'].min().date()} .. {part['date'].max().date()}  rows {len(part):>8,}  "
              f"fire cell-days {int(part['fire_risk'].sum()):>6,} ({part['fire_risk'].mean() * 100:.2f}%)")
    va_summer = va["month"].isin(SUMMER).to_numpy()
    te_summer = te["month"].isin(SUMMER).to_numpy()
    te_eus = (te["latitude"].between(*EUSKADI["lat"]) & te["longitude"].between(*EUSKADI["lon"])).to_numpy()
    y_va, y_te = va["fire_risk"].to_numpy(), te["fire_risk"].to_numpy()
    print(f"Jun-Sep rows: validation {int(va_summer.sum()):,} ({int(y_va[va_summer].sum())} fire), "
          f"test {int(te_summer.sum()):,} ({int(y_te[te_summer].sum())} fire)")

    variants = {
        "A v2 features (18)": ("rf", BASE_FEATURES),
        "B A + FWI system": ("rf", BASE_FEATURES + FWI_FEATURES),
        "C A + multi-day": ("rf", BASE_FEATURES + EXTRA_FEATURES),
        "D all features": ("rf", ALL_FEATURES),
        "F D + location": ("rf", ALL_FEATURES + LOCATION_FEATURES),
        "G D, gradient boosting": ("gb", ALL_FEATURES),
        "H D + location, gb": ("gb", ALL_FEATURES + LOCATION_FEATURES),
    }
    results = {}
    header("Training variants")
    for name, (kind, feats) in variants.items():
        est, beta = fit_model(kind, tr, feats, args.ratio, args.trees)
        model = FireRiskModel(est, beta)
        results[name] = dict(model=model, feats=feats, p_va=model.predict_proba(va[feats])[:, 1],
                             p_te=model.predict_proba(te[feats])[:, 1])
        print(f"  trained {name}   ({time.time() - t0:.0f}s elapsed)")
    if args.seasonal:
        s_tr = tr["month"].isin(SUMMER)
        rf_s, b_s = fit_model("rf", tr[s_tr], ALL_FEATURES, args.ratio, args.trees)
        rf_r, b_r = fit_model("rf", tr[~s_tr], ALL_FEATURES, args.ratio, args.trees)
        model = FireRiskModel(rf_r, b_r, rf_s, b_s, SUMMER)
        results["E D + season split"] = dict(model=model, feats=ALL_FEATURES,
                                             p_va=model.predict_proba(va[ALL_FEATURES])[:, 1],
                                             p_te=model.predict_proba(te[ALL_FEATURES])[:, 1])
        print(f"  trained E D + season split   ({time.time() - t0:.0f}s elapsed)")

    # ---- comparison table ---------------------------------------------------------------
    header("Results (ROC-AUC / PR-AUC).  Compare PR-AUC with the base rate of the same slice")
    te_eus_sum = te_eus & te_summer
    rows = []
    for name, r in results.items():
        mv, mt = metrics(y_va, r["p_va"]), metrics(y_te, r["p_te"])
        ms = metrics(y_te[te_summer], r["p_te"][te_summer])
        me = metrics(y_te[te_eus], r["p_te"][te_eus])
        mes = metrics(y_te[te_eus_sum], r["p_te"][te_eus_sum]) if y_te[te_eus_sum].sum() >= 20 else None
        rows.append({"variant": name, "val ROC": f3(mv, "roc"), "val PR": f3(mv, "pr"),
                     "test ROC": f3(mt, "roc"), "test PR": f3(mt, "pr"),
                     "JunSep ROC": f3(ms, "roc"), "JunSep PR": f3(ms, "pr"),
                     "Euskadi ROC": f3(me, "roc"), "Eus JunSep ROC": f3(mes, "roc")})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nbase rates (test): all days {y_te.mean():.3f}   Jun-Sep {y_te[te_summer].mean():.3f}   "
          f"Euskadi {y_te[te_eus].mean():.3f}   Euskadi Jun-Sep {y_te[te_eus_sum].mean():.3f} "
          f"({int(y_te[te_eus_sum].sum())} fire cell-days)")

    # ---- is the gain real? ---------------------------------------------------------------
    base_name = next(k for k in results if k.startswith(args.baseline + " "))
    if args.bootstrap > 0:
        header(f"Gain over {base_name}: paired bootstrap over test DAYS ({args.bootstrap} resamples)")
        day_codes, days = pd.factorize(te["date"])
        order = np.argsort(day_codes, kind="stable")
        bounds = np.cumsum(np.bincount(day_codes, minlength=len(days)))[:-1]
        idx_by_day = np.split(order, bounds)
        rng = np.random.default_rng(0)
        base = results[base_name]["p_te"]
        deltas = {k: {"all": [], "summer": []} for k in results if k != base_name}
        for _ in range(args.bootstrap):
            pick = rng.integers(0, len(days), len(days))
            idx = np.concatenate([idx_by_day[j] for j in pick])
            sidx = idx[te_summer[idx]]
            if y_te[idx].min() == y_te[idx].max() or y_te[sidx].min() == y_te[sidx].max():
                continue
            b_all, b_sum = roc_auc_score(y_te[idx], base[idx]), roc_auc_score(y_te[sidx], base[sidx])
            for k in deltas:
                p = results[k]["p_te"]
                deltas[k]["all"].append(roc_auc_score(y_te[idx], p[idx]) - b_all)
                deltas[k]["summer"].append(roc_auc_score(y_te[sidx], p[sidx]) - b_sum)

        def ci(v):
            if not v:
                return "n/a"
            lo, mid, hi = np.percentile(v, [2.5, 50, 97.5])
            return f"{mid:+.3f} [{lo:+.3f}, {hi:+.3f}] " + ("real" if lo > 0 else ("WORSE" if hi < 0 else "noise"))

        for k, d in deltas.items():
            print(f"{k:24s} all days: {ci(d['all']):38s} Jun-Sep: {ci(d['summer'])}")
        print("(change in ROC-AUC; 'real' = the 95% interval is entirely above 0)")

    # ---- choose --------------------------------------------------------------------------
    def val_key(r):
        m_all, m_sum = metrics(y_va, r["p_va"]), metrics(y_va[va_summer], r["p_va"][va_summer])
        if not m_all or not m_sum:
            return -1
        return {"overall": m_all["pr"], "summer": m_sum["pr"],
                "balanced": 0.5 * (m_all["roc"] + m_sum["roc"])}[args.select]

    if args.variant == "auto":
        best = max(results, key=lambda k: val_key(results[k]))
        why = f"best validation score, selection = {args.select}"
    else:
        matches = [k for k in results if k.startswith(args.variant + " ")]
        if not matches:
            raise SystemExit(f"Variant {args.variant} was not trained (E needs --seasonal).")
        best, why = matches[0], "forced by --variant"
    chosen = results[best]
    header(f"Chosen ({why}): {best}")
    model, feats, p_va, p_te = chosen["model"], chosen["feats"], chosen["p_va"], chosen["p_te"]
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        print("top feature importances:\n" + imp.head(12).round(3).to_string())
    else:
        print("(gradient boosting: no built-in feature importances)")

    # ---- thresholds: all-year and summer-specific -----------------------------------------
    thr_all = thresholds_from(p_va)
    thr_summer = thresholds_from(p_va[va_summer])
    header("Tier thresholds")
    print("all-year (validation quantiles 60/85/95):", {k: round(v, 4) for k, v in thr_all.items()})
    print("summer-only (same quantiles within Jun-Sep):", {k: round(v, 4) for k, v in thr_summer.items()})
    print(f"\nFire rate per tier, all test days, all-year thresholds:\n{tier_table(p_te, y_te, thr_all)}")
    print(f"\nFire rate per tier, test Jun-Sep, all-year thresholds (what testers see today):\n"
          f"{tier_table(p_te[te_summer], y_te[te_summer], thr_all)}")
    print(f"\nFire rate per tier, test Jun-Sep, summer thresholds (proposed):\n"
          f"{tier_table(p_te[te_summer], y_te[te_summer], thr_summer)}")
    print(f"\nmean predicted probability on test: {p_te.mean():.4f}  (actual fire rate {y_te.mean():.4f})")

    header("Showcase: real summer days in the test years (tiers use the summer thresholds)")
    showcase(te, p_te, thr_summer)

    if args.window_check > 0:
        header("Serving check: features rebuilt from only the last W days of weather")
        window_check(te, raws, model, feats, args.test_start, args.window_check, (MIN_HISTORY_DAYS, 240))
        print("Small changes (a few % of the mean, rank correlation > 0.98) mean the app can use that window.")

    if args.date:
        header(args.date)
        day = df[df["date"] == args.date]
        if day.empty:
            print("This date is not in the data (outside the cached period, in the skipped warm-up, or held out).")
        else:
            p_day = model.predict_proba(day[feats])[:, 1]
            summer_day = int(day["month"].iloc[0]) in SUMMER
            thr = thr_summer if summer_day else thr_all
            if day["date"].iloc[0] < pd.Timestamp(args.val_start):
                print("!! this date is in the TRAINING years, so the score is optimistic.")
            tiers = tier_of(p_day, thr).value_counts().to_dict()
            print(f"{len(day)} cells.  mean probability {p_day.mean():.3f}   max {p_day.max():.3f}")
            print(f"tiers with {'summer' if summer_day else 'all-year'} thresholds: {tiers}")
            print(f"fire cells that day: {int(day['fire_risk'].sum())}")

    if args.no_save:
        return
    out_dir = ROOT / "Model"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(model, out_dir / "fireTrainModel_v3.pkl")
    (out_dir / "tier_thresholds_v3.json").write_text(json.dumps(
        {**thr_all, "summer": thr_summer, "summer_months": list(SUMMER)}, indent=2))
    (out_dir / "fireModel_v3_meta.json").write_text(json.dumps({
        "variant": best, "features": list(feats), "min_history_days": MIN_HISTORY_DAYS,
        "needs_latitude_longitude": any(c in feats for c in LOCATION_FEATURES),
        "train_before": args.val_start, "test_from": args.test_start, "ratio": args.ratio,
        "seasonal_model": bool(model.summer is not None), "final_check": bool(args.final),
        "note": "predict_proba already returns probabilities on the real base rate; feature values "
                "must come from Model/fireFeatures.build_features(); tier_thresholds_v3.json has "
                "all-year thresholds at the top level and a 'summer' set for Jun-Sep.",
    }, indent=2))
    header("Saved")
    print(f"{out_dir / 'fireTrainModel_v3.pkl'}\n{out_dir / 'tier_thresholds_v3.json'}\n{out_dir / 'fireModel_v3_meta.json'}")
    print("The app must build features with Model/fireFeatures.build_features() and give it >= "
          f"{MIN_HISTORY_DAYS} days of history. Do NOT use scaler.pkl.")


if __name__ == "__main__":
    main()