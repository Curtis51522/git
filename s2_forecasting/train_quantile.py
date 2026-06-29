import os, json, joblib, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "product_id", "category", "daily_tickets", "day_of_week", "month",
    "is_weekend", "is_holiday",
    "lag_1", "lag_7_avg", "lag_30_avg", "roll_std_7", "roll_std_14", "trend_7",
    "is_day1", "is_top3", "discount_pct",
    "is_member_day", "is_rainy",
    "temp_mean", "temp_range", "is_cold_day", "is_hot_day",
]
TARGET = "quantity"
QUANTILES = [0.10, 0.50, 0.90]

PARAM_GRID_Q50 = {
    "n_estimators": [200, 500],
    "max_depth": [3, 4],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 1.5],
    "min_child_weight": [10, 20],
    "tweedie_variance_power": [1.2, 1.5, 1.8],
}

PARAM_GRID_QUANTILE = {
    "n_estimators": [200, 500],
    "max_depth": [3, 4],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 1.5],
    "min_child_weight": [10, 20],
}


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "xgboost_train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "xgboost_val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "xgboost_test.csv"))
    print(f"Loaded from CSVs: Train={len(train)}  Val={len(val)}  Test={len(test)}")
    return train, val, test


def get_xy(df):
    return df[FEATURES].copy(), df[TARGET].copy()


def tune_q50(train_df, val_df):
    print("\n--- Tuning Q50 (Tweedie, 3-fold CV) ---")
    combined = pd.concat([train_df, val_df], ignore_index=True)
    X, y = get_xy(combined)
    grid = GridSearchCV(
        xgb.XGBRegressor(objective="reg:tweedie",
                         enable_categorical=True, tree_method="hist",
                         random_state=42, n_jobs=-1, verbosity=0),
        PARAM_GRID_Q50, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=-1, verbose=1,
    )
    grid.fit(X, y)
    print(f"\n  Best params: {grid.best_params_}")
    print(f"  Best CV MAE: {-grid.best_score_:.4f}")
    return grid.best_params_


def train_quantile(train_df, val_df, quantile_alpha, best_params, is_tweedie=False):
    qname = int(quantile_alpha * 100)
    label = "Tweedie" if is_tweedie else "Quantile"
    print(f"\n--- Training Q{qname} ({label}) ---")
    combined = pd.concat([train_df, val_df], ignore_index=True)
    X, y = get_xy(combined)
    X_val, y_val = get_xy(val_df)

    if is_tweedie:
        tw_power = best_params.get("tweedie_variance_power", 1.5)
        params = {k: v for k, v in best_params.items() if k != "tweedie_variance_power"}
        model = xgb.XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=tw_power,
            enable_categorical=True,
            tree_method="hist",
            early_stopping_rounds=50,
            **params,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        params = best_params
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile_alpha,
            enable_categorical=True,
            tree_method="hist",
            early_stopping_rounds=50,
            **params,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    fname = os.path.join(OUT_DIR, f"quantile_model_q{qname}.pkl")
    joblib.dump(model, fname)
    print(f"  Saved -> {fname}")
    return model


def conformal_calibrate(val_df, models):
    print("\n--- Conformal Calibration on Val Set (target 80% coverage) ---")
    X_val, y_val = get_xy(val_df)
    q50_preds = np.maximum(models[0.50].predict(X_val), 0)

    per_product = {}
    for pid in sorted(val_df["product_id"].unique()):
        mask = val_df["product_id"].values == pid
        residuals = np.abs(y_val.values[mask] - q50_preds[mask])
        half_w = float(np.quantile(residuals, 0.80))
        half_w = max(half_w, 0.5)
        per_product[str(pid)] = round(half_w, 1)

    all_residuals = np.abs(y_val.values - q50_preds)
    global_half = round(max(float(np.quantile(all_residuals, 0.80)), 0.5), 1)

    calibration = {"per_product": per_product, "global": global_half}

    out_path = os.path.join(OUT_DIR, "conformal_calibration.json")
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"  Saved -> {out_path}")
    print(f"  Global half_width: {global_half}")
    vals = list(per_product.values())
    print(f"  Per-product range: {min(vals)} to {max(vals)}")

    covered = 0
    total = 0
    for pid in sorted(val_df["product_id"].unique()):
        mask = val_df["product_id"].values == pid
        hw = per_product.get(str(pid), global_half)
        covered += ((y_val.values[mask] >= q50_preds[mask] - hw) & (y_val.values[mask] <= q50_preds[mask] + hw)).sum()
        total += mask.sum()
    print(f"  Val coverage (calibrated): {covered / total * 100:.1f}%")
    return calibration


def evaluate(test_df, models, calibration):
    X_test, y_test = get_xy(test_df)

    q50_preds = np.maximum(models[0.50].predict(X_test), 0)
    q10_preds = np.maximum(models[0.10].predict(X_test), 0)
    q90_preds = np.maximum(models[0.90].predict(X_test), 0)

    print("\n--- Evaluation on Test Set (Conformal 80% intervals) ---")
    covered = 0
    total = len(y_test)
    per_prod_cov = {}
    for pid in sorted(test_df["product_id"].unique()):
        mask = test_df["product_id"].values == pid
        hw = calibration["per_product"].get(str(pid), calibration["global"])
        yt = y_test.values[mask]
        qp = q50_preds[mask]
        lower = qp - hw
        upper = qp + hw
        c = ((yt >= lower) & (yt <= upper)).sum()
        covered += c
        per_prod_cov[str(pid)] = round(c / mask.sum() * 100, 1)

    cov_80 = covered / total
    avg_width = np.mean([calibration["per_product"].get(str(pid), calibration["global"]) * 2 for pid in test_df["product_id"].unique()])
    cov_raw = ((y_test.values >= q10_preds) & (y_test.values <= q90_preds)).mean()
    w_raw = np.mean(q90_preds - q10_preds)

    mae = np.mean(np.abs(y_test.values - q50_preds))
    wape = np.sum(np.abs(y_test.values - q50_preds)) / np.sum(y_test.values) * 100
    rmse = float(np.sqrt(np.mean((y_test.values - q50_preds) ** 2)))

    metrics = {
        "model": "XGBoost Tweedie Q50 + Quantile Q10/Q90",
        "features": len(FEATURES),
        "test_period": f"{test_df['date'].min()} to {test_df['date'].max()}",
        "interval_method": "Conformal 80%",
        "overall": {
            "WAPE": round(wape, 1),
            "MAE": round(mae, 1),
            "RMSE": round(rmse, 1),
            "conformal_coverage_80": round(cov_80 * 100, 1),
            "conformal_avg_width": round(avg_width, 1),
            "raw_Q10Q90_coverage": round(cov_raw * 100, 1),
            "raw_Q10Q90_width": round(w_raw, 1),
        },
        "per_product_coverage": per_prod_cov,
        "best_params": str(models[0.50].get_params()),
    }

    print(f"  Conformal Coverage: {cov_80*100:.1f}%  (raw Q10-Q90: {cov_raw*100:.1f}%)")
    print(f"  Conformal Width:    {avg_width:.1f}  (raw Q10-Q90: {w_raw:.1f})")
    print(f"  WAPE: {wape:.1f}%  MAE: {mae:.1f}  RMSE: {rmse:.1f}")

    per_product = {}
    for pid in sorted(test_df["product_id"].unique()):
        mask = test_df["product_id"].values == pid
        yt = y_test.values[mask]
        qp = q50_preds[mask]
        hw = calibration["per_product"].get(str(pid), calibration["global"])
        per_product[str(pid)] = {
            "MAE": round(float(np.mean(np.abs(yt - qp))), 1),
            "RMSE": round(float(np.sqrt(np.mean((yt - qp)**2))), 1),
            "WAPE": round(float(np.sum(np.abs(yt - qp)) / np.sum(yt) * 100), 1),
            "conformal_half": hw,
            "conformal_coverage": per_prod_cov.get(str(pid), 0),
            "samples": int(mask.sum()),
            "mean_daily": round(float(np.mean(yt)), 1),
        }

    metrics["per_product"] = per_product
    out_path = os.path.join(OUT_DIR, "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics, {"q50": q50_preds, "q10": q10_preds, "q90": q90_preds}, y_test


def main():
    print("=" * 60)
    print("  XGBoost Tweedie Q50 + Conformal Calibration")
    print("=" * 60)
    train, val, test = load_data()

    # Tune Q50 with Tweedie
    best_q50 = tune_q50(train, val)
    with open(os.path.join(OUT_DIR, "tweedie_best_params.json"), "w") as f:
        json.dump(best_q50, f, indent=2, default=str)

    # Q10/Q90 use quantile params from simpler grid
    print("\n--- Tuning Q10 (Quantile, 3-fold CV) ---")
    combined = pd.concat([train, val], ignore_index=True)
    X_all, y_all = get_xy(combined)
    qt_grid = GridSearchCV(
        xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10,
                         enable_categorical=True, tree_method="hist",
                         random_state=42, n_jobs=-1, verbosity=0),
        PARAM_GRID_QUANTILE, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=-1, verbose=1,
    )
    qt_grid.fit(X_all, y_all)
    best_qt = qt_grid.best_params_
    print(f"  Best Q10 params: {best_qt}")

    models = {}
    models[0.50] = train_quantile(train, val, 0.50, best_q50, is_tweedie=True)
    models[0.10] = train_quantile(train, val, 0.10, best_qt, is_tweedie=False)
    models[0.90] = train_quantile(train, val, 0.90, best_qt, is_tweedie=False)

    calibration = conformal_calibrate(val, models)
    metrics, preds, y_test = evaluate(test, models, calibration)

    print(f"\n{'='*60}")
    print("  Done. Outputs -> s2_forecasting/outputs/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
