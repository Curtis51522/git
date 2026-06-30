#!/usr/bin/env python
"""XGBoost Weekly Quantile Regression (2026-06-30)
==================================================
Predicts weekly total quantity per product.
Compares:
  - Naive baseline: lag_1w (last week's value)
  - Quantile Q50 / Q90 models
"""

import os, json, joblib, warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

WEEKLY_FEATURES = [
    "product_id", "category", "weekly_tickets", "week_of_year", "month",
    "is_holiday_week",
    "lag_1w", "lag_4w_avg", "lag_8w_avg",
    "is_day1_w", "is_top3_w", "discount_pct_w",
    "is_member_week", "is_new_product_w", "is_competitor_w", "is_rainy_w",
]
TARGET = "quantity_w"
QUANTILES = [0.10, 0.50, 0.90]


def load_weekly_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "weekly_train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "weekly_val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "weekly_test.csv"))
    print(f"Loaded: Train={len(train)}  Val={len(val)}  Test={len(test)}")
    return train, val, test


def get_xy(df):
    return df[WEEKLY_FEATURES].copy(), df[TARGET].copy()


def compute_wape(y_true, y_pred):
    """Weighted Absolute Percentage Error"""
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true[mask] - y_pred[mask])) / np.sum(y_true[mask]) * 100)


def compute_baseline(test_df):
    """Naive baseline: use lag_1w (last week's quantity) as prediction."""
    y_true = test_df[TARGET].values
    y_pred = test_df["lag_1w"].values
    y_pred = np.maximum(y_pred, 0)
    wape = compute_wape(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"WAPE": round(wape, 2), "MAE": round(mae, 2), "RMSE": round(rmse, 2)}


def train_quantile(train_df, val_df, quantile_alpha):
    qname = int(quantile_alpha * 100)
    print(f"\n--- Training Q{qname} ---")
    combined = pd.concat([train_df, val_df], ignore_index=True)
    X, y = get_xy(combined)

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile_alpha,
        enable_categorical=True,
        tree_method="hist",
        max_depth=5,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y, eval_set=[(X, y)], verbose=False)

    fname = os.path.join(OUT_DIR, f"weekly_quantile_q{qname}.pkl")
    joblib.dump(model, fname)
    print(f"  Saved -> {fname}")
    return model


def evaluate(test_df, models):
    X_test, y_test = get_xy(test_df)

    results = {}
    # Baseline
    baseline = compute_baseline(test_df)
    print(f"\n  Baseline (lag_1w): WAPE={baseline['WAPE']}%  MAE={baseline['MAE']}  RMSE={baseline['RMSE']}")
    results["baseline"] = baseline

    # Each quantile
    for q, model in models.items():
        raw = model.predict(X_test)
        pred = np.maximum(raw, 0)
        wape = compute_wape(y_test, pred)
        mae = float(np.mean(np.abs(y_test - pred)))
        rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
        qname = f"Q{int(q*100)}"
        print(f"  {qname}: WAPE={wape:.2f}%  MAE={mae:.2f}  RMSE={rmse:.2f}")
        results[qname] = {"WAPE": round(wape, 2), "MAE": round(mae, 2), "RMSE": round(rmse, 2)}

    # Coverage for Q50/Q90 interval
    if 0.50 in models and 0.90 in models:
        q50_pred = np.maximum(models[0.50].predict(X_test), 0)
        q90_pred = np.maximum(models[0.90].predict(X_test), 0)
        coverage = ((y_test >= q50_pred) & (y_test <= q90_pred)).mean()
        width = np.mean(q90_pred - q50_pred)
        print(f"  Coverage Q50-Q90: {coverage:.4f}  (target 0.40)")
        print(f"  Interval width: {width:.2f}")
        results["coverage_Q50_Q90"] = round(float(coverage), 4)
        results["interval_width"] = round(float(width), 2)

    # WAPE improvement
    if "Q50" in results:
        improvement = baseline["WAPE"] - results["Q50"]["WAPE"]
        results["Q50_vs_baseline"] = round(improvement, 2)
        print(f"\n  Q50 improvement over baseline: {improvement:.1f} pp")

    return results, y_test


def per_product_wape(test_df, models):
    """Compute WAPE per product for the best model (Q50)."""
    if 0.50 not in models:
        return
    model = models[0.50]
    test_df = test_df.copy()
    X_test = test_df[WEEKLY_FEATURES]
    y_test = test_df[TARGET].values
    test_df["pred_q50"] = np.maximum(model.predict(X_test), 0)

    results = []
    for pid in sorted(test_df["product_id"].unique()):
        sub = test_df[test_df["product_id"] == pid]
        yt = sub[TARGET].values
        yp = sub["pred_q50"].values
        if yt.sum() == 0:
            continue
        wape = float(np.sum(np.abs(yt - yp)) / np.sum(yt) * 100)
        results.append({"product_id": pid, "samples": len(yt), "weekly_wape": round(wape, 1)})

    df_p = pd.DataFrame(results).sort_values("weekly_wape", ascending=True)
    print("\n--- Per-Product Weekly WAPE (Q50) ---")
    print(df_p.to_string(index=False))
    df_p.to_csv(os.path.join(OUT_DIR, "weekly_per_product.csv"), index=False)

    # Coarsen: bread vs beverage
    test_df["category"] = (test_df["product_id"] >= 30).astype(int)
    for cat, label in [(0, "Bakery"), (1, "Beverage")]:
        sub = test_df[test_df["category"] == cat]
        yt = sub[TARGET].values
        yp = sub["pred_q50"].values
        wape = float(np.sum(np.abs(yt - yp)) / np.sum(yt) * 100)
        print(f"  {label}: WAPE={wape:.2f}%")


def main():
    print("=" * 60)
    print("  XGBoost Weekly Quantile Regression")
    print("=" * 60)

    train, val, test = load_weekly_data()
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")

    models = {}
    for q in QUANTILES:
        models[q] = train_quantile(train, val, q)

    print("\n--- Evaluation on Weekly Test Set ---")
    results, y_test = evaluate(test, models)
    per_product_wape(test, models)

    with open(os.path.join(OUT_DIR, "weekly_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("  Done. Outputs -> s2_forecasting/outputs/weekly_*")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
