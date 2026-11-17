"""
Retrain the wildfire Random Forest the honest way.

Needs weather_all_days_natural.csv (written by:  python fetch_exact_weather.py --save-full ; no API calls).

    python train_model_v2.py                       # default split: train <2019, val 2019-20, test >=2021
    python train_model_v2.py --date 2022-07-18     # also scores that day
    python train_model_v2.py --trees 100           # faster test run

What it does differently from trainModel.py
  * Splits by YEAR (train -> validation -> test), never by random rows.
  * Samples non-fire days to 1:5 in the TRAINING part only; validation and test keep the real fire rate.
  * No class_weight (the 1:5 sampling already rebalances); probabilities are then corrected back to
    the real base rate, so "0.20" means roughly a 20% chance, not an artefact of the sampling.
  * Compares the model with trivial baselines (month average, fwi_proxy alone). If it can't beat them
    it is not worth shipping.
  * Tests with and without month/day_of_year and keeps the better variant on the validation years.
  * Tier thresholds come from the validation years' natural distribution.
  * Saves Model/fireTrainModel_v2.pkl and Model/tier_thresholds_v2.json. Nothing existing is overwritten.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ALL_FEATURES = [
    "temperature", "temp_max", "temp_min", "humidity", "min_humidity", "wind_speed",
    "max_wind_speed", "precip_1d", "precip_7d", "precip_14d", "precip_30d", "dry_spell_days",
    "temp_anomaly", "et0_14d", "fwi_proxy", "vpd", "month", "day_of_year",
]
CALENDAR = ["month", "day_of_year"]
EUSKADI = dict(lat=(42.47, 43.45), lon=(-3.45, -1.73))


def find_root():
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "Model").is_dir() and (p / "Data").is_dir():
            return p
    return here


ROOT = find_root()


def header(text):
    print("\n" + "=" * 74 + f"\n{text}\n" + "=" * 74)


def sample_train(df, ratio, seed):
    pos, neg = df[df["fire_risk"] == 1], df[df["fire_risk"] == 0]
    n = min(len(neg), ratio * len(pos))
    keep = neg.sample(n=n, random_state=seed)
    return pd.concat([pos, keep]).sample(frac=1, random_state=seed), n / max(len(neg), 1)


def correct_prior(p, beta):
    """Undo the effect of keeping only a fraction `beta` of the non-fire days when training."""
    return beta * p / (beta * p + (1 - p))


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-p)
    out = {"roc": roc_auc_score(y, p), "pr": average_precision_score(y, p), "base": y.mean()}
    for frac in (0.05, 0.10):
        k = max(1, int(len(p) * frac))
        out[f"recall@{int(frac * 100)}%"] = y[order[:k]].sum() / y.sum()
    return out


def fmt(m):
    if m is None:
        return "not enough fire days in this slice"
    return (f"ROC-AUC {m['roc']:.3f}  PR-AUC {m['pr']:.3f} (base {m['base']:.3f})  "
            f"fires caught in top 5% of days {m['recall@5%'] * 100:4.1f}%  top 10% {m['recall@10%'] * 100:4.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "Data" / "MITECO" / "weather_all_days_natural.csv")
    ap.add_argument("--val-start", default="2019-01-01")
    ap.add_argument("--test-start", default="2021-01-01")
    ap.add_argument("--ratio", type=int, default=5, help="non-fire : fire in the training sample")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--date", default=None, help="also score this date, e.g. 2022-07-18")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found. Run:  python fetch_exact_weather.py --save-full  "
                         "(no API calls once all cells are cached)")
    df = pd.read_csv(args.csv, parse_dates=["date"]).dropna(subset=ALL_FEATURES + ["fire_risk"])
    df["fire_risk"] = df["fire_risk"].astype(int)

    tr = df[df["date"] < args.val_start]
    va = df[(df["date"] >= args.val_start) & (df["date"] < args.test_start)]
    te = df[df["date"] >= args.test_start]
    header("Data")
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        print(f"{name:11s} {part['date'].min().date()} .. {part['date'].max().date()}  "
              f"rows {len(part):>8,}  fire cell-days {int(part['fire_risk'].sum()):>6,} "
              f"({part['fire_risk'].mean() * 100:.2f}%)")
    if te["fire_risk"].sum() < 50 or va["fire_risk"].sum() < 50:
        print("!! very few fire days in validation or test; results will be noisy. "
              "Adjust --val-start / --test-start.")

    train_s, beta = sample_train(tr, args.ratio, seed=42)
    print(f"\ntraining sample: {int(train_s['fire_risk'].sum()):,} fire + "
          f"{int((train_s['fire_risk'] == 0).sum()):,} non-fire (kept {beta * 100:.1f}% of non-fire days)")

    # ---- baselines ------------------------------------------------------------------
    header("Baselines (the model must beat these)")
    month_rate = tr.groupby("month")["fire_risk"].mean()
    print("month average only :", fmt(metrics(te["fire_risk"], te["month"].map(month_rate).fillna(0))))
    print("fwi_proxy alone    :", fmt(metrics(te["fire_risk"], te["fwi_proxy"])))
    print("vpd alone          :", fmt(metrics(te["fire_risk"], te["vpd"])))

    # ---- model variants ---------------------------------------------------------------
    variants = {
        "all 18 features": ALL_FEATURES,
        "no month/day_of_year": [f for f in ALL_FEATURES if f not in CALENDAR],
    }
    fitted = {}
    header("Random Forest variants (validation = years used to choose; test = untouched years)")
    for name, feats in variants.items():
        rf = RandomForestClassifier(
            n_estimators=args.trees, max_depth=13, min_samples_leaf=20, min_samples_split=40,
            max_features="sqrt", n_jobs=-1, random_state=42,
        ).fit(train_s[feats], train_s["fire_risk"])
        p_va = correct_prior(rf.predict_proba(va[feats])[:, 1], beta)
        p_te = correct_prior(rf.predict_proba(te[feats])[:, 1], beta)
        m_va, m_te = metrics(va["fire_risk"], p_va), metrics(te["fire_risk"], p_te)
        fitted[name] = dict(model=rf, feats=feats, p_va=p_va, p_te=p_te, m_va=m_va)
        print(f"\n[{name}]\n  validation: {fmt(m_va)}\n  test      : {fmt(m_te)}")

    best = max(fitted, key=lambda k: (fitted[k]["m_va"] or {"pr": -1})["pr"])
    chosen = fitted[best]
    print(f"\n-> chosen variant (best validation PR-AUC): {best}")
    rf, feats, p_te = chosen["model"], chosen["feats"], chosen["p_te"]

    header("Does it hold up where it matters?")
    in_eus = te["latitude"].between(*EUSKADI["lat"]) & te["longitude"].between(*EUSKADI["lon"])
    print(f"test, Euskadi cells only ({int(in_eus.sum()):,} rows, {int(te.loc[in_eus, 'fire_risk'].sum())} fire days):")
    print("  ", fmt(metrics(te.loc[in_eus, "fire_risk"], p_te[in_eus.values])))
    summer = te["month"].isin([6, 7, 8, 9]).values
    print(f"test, Jun-Sep only: {fmt(metrics(te.loc[summer, 'fire_risk'], p_te[summer]))}")
    print(f"test, Oct-May only: {fmt(metrics(te.loc[~summer, 'fire_risk'], p_te[~summer]))}")
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nfeature importances:\n" + imp.round(3).to_string())

    # ---- thresholds --------------------------------------------------------------------
    p_va = chosen["p_va"]
    thresholds = {"low": float(np.quantile(p_va, 0.60)),
                  "medium": float(np.quantile(p_va, 0.85)),
                  "high": float(np.quantile(p_va, 0.95))}
    header("Tier thresholds (from the validation years' natural distribution)")
    te_tier = pd.cut(p_te, [-1, thresholds["low"], thresholds["medium"], thresholds["high"], 2],
                     labels=["LOW", "MEDIUM", "HIGH", "EXTREME"])
    stats = pd.DataFrame({"tier": te_tier, "fire": te["fire_risk"].values}).groupby("tier", observed=True)["fire"] \
        .agg(["size", "sum", "mean"])
    stats["mean"] = (stats["mean"] * 100).round(2)
    stats.columns = ["cell_days", "fire_days", "fire_rate_%"]
    print(json.dumps({k: round(v, 4) for k, v in thresholds.items()}))
    print("\nTest years by tier (fire rate should rise with the tier):")
    print(stats.to_string())

    # ---- one date ---------------------------------------------------------------------
    if args.date:
        header(f"{args.date}")
        day = df[df["date"] == args.date]
        if day.empty:
            print("This date is not in the file.")
        else:
            p_day = correct_prior(rf.predict_proba(day[feats])[:, 1], beta)
            daily = pd.Series(p_te, index=te.index).groupby(te["date"]).mean()
            pct = (daily < p_day.mean()).mean() * 100
            if day["date"].iloc[0] < pd.Timestamp(args.val_start):
                print("!! this date is in the TRAINING years, so the score is optimistic.")
            tiers = pd.cut(p_day, [-1, thresholds["low"], thresholds["medium"], thresholds["high"], 2],
                           labels=["LOW", "MEDIUM", "HIGH", "EXTREME"]).value_counts().to_dict()
            print(f"{len(day)} cells.  mean probability {p_day.mean():.3f}   max {p_day.max():.3f}")
            print(f"this day ranks above {pct:.0f}% of the days in the test years; tiers: {tiers}")
            print(f"fire cells that day: {int(day['fire_risk'].sum())}")

    if args.no_save:
        return
    out_dir = ROOT / "Model"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(rf, out_dir / "fireTrainModel_v2.pkl")
    (out_dir / "tier_thresholds_v2.json").write_text(json.dumps(thresholds, indent=2))
    header("Saved")
    print(f"{out_dir / 'fireTrainModel_v2.pkl'}\n{out_dir / 'tier_thresholds_v2.json'}")
    print("features, in order:", feats)
    print("Do NOT use Model/scaler.pkl with this model: it was never trained with one.")


if __name__ == "__main__":
    main()