"""Bakery AI S2 - XGBoost Training Pipeline (Idiomatic, xgboost 3.2.0)
Malaysia-calibrated: DOSM F&B seasonality, public holidays, Ramadan, Monday-closed
"""

import sys, os, json, random, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import holidays
from hijri_converter import convert
from datetime import datetime, timedelta, date
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "xgboost")
os.makedirs(MODEL_DIR, exist_ok=True)

PRODUCT_TYPES = ["donut","croissant","bread_coconut","bread_roll","chiffon","croissant_chocolate"]
PRODUCT_BASE = {"donut":35,"croissant":30,"bread_coconut":18,"bread_roll":20,"chiffon":14,"croissant_chocolate":25}
WEATHER_IMPACT = {"sunny":1.0,"cloudy":0.95,"rainy":0.75,"foggy":0.85,"thunderstorm":0.55,"storm":0.55}
WEEKEND_BOOST = 1.25

# DOSM-calibrated: Malaysia F&B Services Volume Index (2015=100), annual pattern
# Source: DOSM Monthly Manufacturing & F&B Services Statistics, 2019-2024 averages
MONTH_SEASONALITY = {
    1: 0.87, 2: 0.90, 3: 0.93, 4: 1.02, 5: 1.08, 6: 1.12,
    7: 1.10, 8: 1.05, 9: 1.02, 10: 1.00, 11: 0.92, 12: 1.18,
}

FRESHNESS_UPLIFT = {"Fresh":1.0,"Day-1":1.15,"Day-2":1.25,"Discount":1.30}

KL_WEATHER_PROB = {
    1:{"sunny":0.15,"cloudy":0.45,"rainy":0.35,"thunderstorm":0.05},
    2:{"sunny":0.20,"cloudy":0.45,"rainy":0.30,"thunderstorm":0.05},
    3:{"sunny":0.25,"cloudy":0.35,"rainy":0.30,"thunderstorm":0.10},
    4:{"sunny":0.20,"cloudy":0.35,"rainy":0.35,"thunderstorm":0.10},
    5:{"sunny":0.25,"cloudy":0.40,"rainy":0.25,"thunderstorm":0.10},
    6:{"sunny":0.35,"cloudy":0.40,"rainy":0.15,"thunderstorm":0.10},
    7:{"sunny":0.40,"cloudy":0.35,"rainy":0.15,"thunderstorm":0.10},
    8:{"sunny":0.35,"cloudy":0.35,"rainy":0.20,"thunderstorm":0.10},
    9:{"sunny":0.30,"cloudy":0.35,"rainy":0.25,"thunderstorm":0.10},
    10:{"sunny":0.20,"cloudy":0.35,"rainy":0.35,"thunderstorm":0.10},
    11:{"sunny":0.15,"cloudy":0.35,"rainy":0.40,"thunderstorm":0.10},
    12:{"sunny":0.15,"cloudy":0.40,"rainy":0.35,"thunderstorm":0.10},
}

FEATURE_COLS = [
    "day_of_week","is_weekend","day_of_month","month","discount_rate","is_public_holiday",
    "is_ramadan",
    "temperature","rainfall","humidity","is_rainy",
    "weather_sunny","weather_cloudy","weather_rainy","weather_storm",
]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---- Helpers --------------------------------------------------------
_my_holidays = holidays.MY()

def is_ramadan_date(dt: date) -> bool:
    """Check if a Gregorian date falls within Ramadan (Hijri month 9)."""
    try:
        h = convert.Gregorian(dt.year, dt.month, dt.day).to_hijri()
        return h.month == 9
    except Exception:
        return False

# Malaysian public holiday demand impact classification
_HOLIDAY_MAJOR = {
    "Hari Raya Puasa", "Hari Raya Aidilfitri", "Hari Raya Qurban",
    "Chinese New Year", "Deepavali",
}
_HOLIDAY_MEDIUM = {
    "Awal Muharam", "Hari Keputeraan Nabi Muhammad S.A.W.",
    "Wesak Day", "Hari Wesak", "Thaipusam",
}

def get_holiday_boost(dt: date) -> float:
    """Return demand multiplier for Malaysian public holidays."""
    if dt not in _my_holidays:
        return 1.0
    name = _my_holidays.get(dt)
    if not name:
        return 1.0
    if any(kw in name for kw in _HOLIDAY_MAJOR):
        return random.uniform(1.25, 1.40)
    if any(kw in name for kw in _HOLIDAY_MEDIUM):
        return random.uniform(1.10, 1.20)
    return random.uniform(1.00, 1.08)

# ---- 1. Generate ----------------------------------------------------
def random_weather(m):
    p = KL_WEATHER_PROB[m]
    return random.choices(list(p.keys()), weights=list(p.values()))[0]

def weather_oh(wt):
    return {"weather_sunny": int(wt=="sunny"), "weather_cloudy": int(wt=="cloudy"),
            "weather_rainy": int(wt=="rainy"), "weather_storm": int(wt in ("storm","thunderstorm"))}

def generate_sales_data(days=365):
    rows = []
    today = datetime.now()
    start = today - timedelta(days=days)
    monday_count = 0
    for d in range(days):
        dt = start + timedelta(days=d)
        # Skip Mondays - shop is closed
        if dt.weekday() == 0:
            monday_count += 1
            continue
        mth, dow = dt.month, dt.weekday()
        we = int(dow >= 5)
        wt = random_weather(mth)
        wf = weather_oh(wt)

        if wt == "sunny": temp = random.gauss(33, 2); rain = max(0, random.gauss(1, 2))
        elif wt == "cloudy": temp = random.gauss(31, 2); rain = max(0, random.gauss(8, 5))
        elif wt == "rainy": temp = random.gauss(29, 2); rain = max(0, random.gauss(25, 15))
        else: temp = random.gauss(27, 2); rain = max(0, random.gauss(45, 25))

        hum = min(100, max(50, random.gauss(75 + rain * 0.3, 10)))
        ir = int(rain > 5)

        dt_date = dt.date()
        is_holiday = 1 if dt_date in _my_holidays else 0
        is_ramadan = 1 if is_ramadan_date(dt_date) else 0
        holiday_boost = get_holiday_boost(dt_date) if is_holiday else 1.0

        # Ramadan effect: mild daily volume shift
        ramadan_mod = random.uniform(0.92, 1.05) if is_ramadan else 1.0

        for prod, base in PRODUCT_BASE.items():
            for fresh, upl in FRESHNESS_UPLIFT.items():
                dr = 0.3 if fresh in ("Day-1", "Day-2", "Discount") else 0.0
                dem = (base * MONTH_SEASONALITY[mth] * WEATHER_IMPACT[wt] * upl
                       * (WEEKEND_BOOST if we else 1)
                       * holiday_boost * ramadan_mod
                       * random.gauss(1, 0.12))
                rows.append({
                    "date": dt.strftime("%Y-%m-%d"), "product": prod,
                    "freshness": fresh, "day_of_week": dow, "is_weekend": we,
                    "day_of_month": dt.day, "month": mth, "discount_rate": dr,
                    "is_public_holiday": is_holiday, "is_ramadan": is_ramadan,
                    "temperature": round(temp, 1), "rainfall": round(rain, 1),
                    "humidity": round(hum, 1), "is_rainy": ir, **wf,
                    "sales": max(1, round(dem)),
                })
    print(f"       Skipped {monday_count} Mondays (shop closed)")
    return pd.DataFrame(rows)

# ---- 2. Clean -------------------------------------------------------
def clean_data(df):
    df = df.dropna(subset=["sales"] + FEATURE_COLS)
    for prod in df["product"].unique():
        m = df["product"] == prod
        q1, q3 = df.loc[m, "sales"].quantile(0.25), df.loc[m, "sales"].quantile(0.75)
        iqr = q3 - q1
        up, lo = q3 + 3 * iqr, max(0, q1 - 3 * iqr)
        n = ((df.loc[m, "sales"] > up) | (df.loc[m, "sales"] < lo)).sum()
        if n > 0:
            df.loc[m, "sales"] = df.loc[m, "sales"].clip(lo, up)
            print(f"  [clean] {prod}: capped {n} outliers [{lo:.1f},{up:.1f}]")
    return df

# ---- 3. Validate ----------------------------------------------------
def validate_features(df):
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise KeyError(f"Missing: {c}")
        if df[c].dtype not in (np.int64, np.float64, np.int32, np.float32):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

# ---- 4. Split -------------------------------------------------------
def split_data(df, test_days=30, val_days=30):
    df = df.sort_values("date").reset_index(drop=True)
    dates = sorted(df["date"].unique())
    tc = dates[-test_days]
    vc = dates[-(test_days + val_days)]
    tt = df["date"] >= tc
    vv = (df["date"] >= vc) & (df["date"] < tc)
    tr = df["date"] < vc
    return (df.loc[tr, FEATURE_COLS], df.loc[tr, "sales"],
            df.loc[vv, FEATURE_COLS], df.loc[vv, "sales"],
            df.loc[tt, FEATURE_COLS], df.loc[tt, "sales"])

# ---- 5. Tune (XGBoost 3.2.0 idiomatic) ------------------------------
def tune_hyperparameters(X_train, y_train, n_iter=50):
    cutoff = int(len(X_train) * 0.8)
    X_cv, X_es = X_train.iloc[:cutoff], X_train.iloc[cutoff:]
    y_cv, y_es = y_train.iloc[:cutoff], y_train.iloc[cutoff:]
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        "n_estimators": [100, 200, 300, 500], "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0], "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.01, 0.05, 0.1, 0.2, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1.0], "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "min_child_weight": [1, 3, 5, 7],
    }
    base = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        early_stopping_rounds=20, random_state=RANDOM_SEED, n_jobs=-1,
    )
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=n_iter, cv=tscv,
        scoring="neg_mean_absolute_error", random_state=RANDOM_SEED, n_jobs=-1, verbose=0,
    )
    search.fit(X_cv, y_cv, eval_set=[(X_es, y_es)], verbose=False)
    return search.best_estimator_, search.best_params_, search.best_score_

# ---- 6. Evaluate ----------------------------------------------------
def evaluate(model, X, y, label):
    p = model.predict(X)
    mae = mean_absolute_error(y, p)
    mape = np.mean(np.abs((y - p) / np.maximum(y, 1))) * 100
    rmse = np.sqrt(mean_squared_error(y, p))
    r2 = r2_score(y, p)
    return {"set": label, "mae": mae, "mape_pct": mape, "rmse": rmse, "r2": r2, "n": len(y)}

# ---- 7. Main --------------------------------------------------------
def main():
    print("=" * 60)
    print("  Bakery AI - XGBoost Pipeline (S2) [xgboost 3.2.0]")
    print("  MY-calibrated: DOSM + holidays + Ramadan + Mon-closed")
    print("=" * 60)
    print("\n[1/7] Generating synthetic data (365d, MY-calibrated)...")
    df = generate_sales_data(365)
    n_dates = len(df["date"].unique())
    n_products = len(df["product"].unique())
    n_fresh = len(df["freshness"].unique())
    print(f"       {len(df):,} rows | {n_dates} dates | {n_products} products x {n_fresh} freshness")
    print("\n[2/7] Cleaning & outlier capping...")
    df = clean_data(df)
    print("\n[3/7] Validating features...")
    df = validate_features(df)
    dates = sorted(df["date"].unique())
    print(f"       Range: {dates[0]} -> {dates[-1]}")
    all_metrics = {}
    for prod in PRODUCT_TYPES:
        print(f"\n{'=' * 60}")
        print(f"  {prod}")
        print(f"{'=' * 60}")
        pdf = df[df["product"] == prod].sort_values("date").reset_index(drop=True)
        X_tr, y_tr, X_val, y_val, X_test, y_test = split_data(pdf)
        print(f"  Train={len(X_tr)}  Val={len(X_val)}  Test={len(X_test)}")
        print(f"  Tuning (50 iters, 5-fold TimeSeriesSplit, early_stop=20)...")
        model, best, cv_score = tune_hyperparameters(X_tr, y_tr, n_iter=50)
        print(f"  Best: {json.dumps(best)}")
        print(f"  CV MAE: {-cv_score:.2f}")
        metrics = []
        for Xs, ys, lbl in [(X_tr, y_tr, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
            m = evaluate(model, Xs, ys, lbl)
            metrics.append(m)
            print(f"  {lbl:5s}  MAE={m['mae']:.2f}  MAPE={m['mape_pct']:.1f}%  RMSE={m['rmse']:.2f}  R2={m['r2']:.4f}  n={m['n']}")
        all_metrics[prod] = {"best_params": best, "best_cv_mae": -cv_score, "metrics": metrics}
        imp = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: x[1], reverse=True)
        print(f"  Top-5: {', '.join(f'{f}={v:.3f}' for f, v in imp[:5])}")
        X_final = pd.concat([X_tr, X_val])
        y_final = pd.concat([y_tr, y_val])
        print(f"  Final fit on {len(X_final)} rows with early_stopping...")
        final = xgb.XGBRegressor(**best, tree_method="hist", early_stopping_rounds=20,
                                  random_state=RANDOM_SEED, n_jobs=-1)
        final.fit(X_final, y_final, eval_set=[(X_val, y_val)], verbose=False)
        path = os.path.join(MODEL_DIR, f"{prod}_model.json")
        final.save_model(path)
        print(f"  Saved -> {path}")
        mp = os.path.join(MODEL_DIR, f"{prod}_metrics.json")
        with open(mp, "w") as f:
            json.dump({
                "product": prod, **all_metrics[prod],
                "feature_importance": {feat: float(v) for feat, v in imp},
                "trained_on_rows": len(X_final), "test_rows": len(X_test),
            }, f, indent=2)
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLS, f)
    print(f"\n{'=' * 60}")
    print(f"{'Product':25s} {'CV MAE':>8s} {'Test MAE':>9s} {'Test MAPE':>10s} {'Test R2':>8s}")
    print("-" * 65)
    for prod in PRODUCT_TYPES:
        m = all_metrics[prod]
        t = [x for x in m["metrics"] if x["set"] == "test"][0]
        print(f"{prod:25s} {m['best_cv_mae']:8.2f} {t['mae']:9.2f} {t['mape_pct']:9.1f}% {t['r2']:8.4f}")
    avg = np.mean([[x for x in all_metrics[p]["metrics"] if x["set"] == "test"][0]["mape_pct"] for p in PRODUCT_TYPES])
    print(f"\n  Avg Test MAPE: {avg:.1f}%")
    print(f"  Models: {os.path.abspath(MODEL_DIR)}")

if __name__ == "__main__":
    main()
