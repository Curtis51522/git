import os, json, joblib, warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from s2_forecasting.feature_contract import FORECAST_FEATURES
from s2_forecasting.quantile_utils import enforce_quantile_monotonicity
warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = FORECAST_FEATURES
TARGET = "quantity"
QUANTILES = [0.10, 0.50, 0.90]
CV_SPLITS = 3
EXPERIMENT_ROLE = "proposed_probabilistic_forecast"
VALIDATION_DESIGN = (
    "train-only rolling-origin model selection and fitting; independent "
    "chronological validation calibration; untouched chronological test evaluation"
)
CORE_INTERVAL_METHOD = "Core pre-runtime-bias conformal 80%"
CORE_INTERVAL_SCOPE = "core_pre_runtime_bias_conformal"
RUNTIME_TRANSFORM_NOTE = (
    "The live endpoint applies operation-specific bakery/beverage bias scaling "
    "and integer rounding. Those dynamic transforms are not evaluated by this "
    "core artifact-level interval metric."
)

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


def predict_postprocessed_quantiles(df, models):
    features, _ = get_xy(df)
    raw_q10 = np.maximum(models[0.10].predict(features), 0)
    raw_q50 = np.maximum(models[0.50].predict(features), 0)
    raw_q90 = np.maximum(models[0.90].predict(features), 0)
    corrected = enforce_quantile_monotonicity(raw_q10, raw_q50, raw_q90)
    corrected["q50_changed_count"] = int(
        np.count_nonzero(raw_q50 != corrected["q50"])
    )
    return corrected


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def _split_metadata(df, split_name):
    return {
        "split": split_name,
        "row_count": int(len(df)),
        "period": f"{df['date'].min()} to {df['date'].max()}",
    }


def build_deployment_refit_frame(train_df, val_df, test_df):
    frames = (train_df, val_df, test_df)
    if any(frame.empty for frame in frames):
        raise ValueError("Deployment refit requires non-empty train, validation, and test splits")
    periods = [pd.to_datetime(frame["date"]) for frame in frames]
    if not periods[0].max() < periods[1].min() < periods[2].min():
        raise ValueError("Deployment refit splits must be strictly chronological")
    combined = pd.concat(frames, ignore_index=True)
    sort_columns = ["date"]
    if "product_id" in combined.columns:
        sort_columns.append("product_id")
    return combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def write_deployment_metadata(
    train_df,
    val_df,
    test_df,
    refit_df,
    *,
    output_dir=None,
):
    metadata = {
        "schema_version": 1,
        "run_timestamp": _utc_timestamp(),
        "evaluation": {
            "model_selection": _split_metadata(train_df, "train"),
            "calibration": _split_metadata(val_df, "validation"),
            "test": _split_metadata(test_df, "test"),
            "test_evaluated_once": True,
        },
        "deployment_refit": {
            **_split_metadata(refit_df, "complete_active_year"),
            "purpose": "runtime_forecasting_after_held_out_evaluation",
            "held_out_metric_claim": False,
        },
        "calibration_note": (
            "Runtime models are refitted on the complete active year after held-out "
            "evaluation. The saved conformal residual widths remain sourced from the "
            "chronological validation design and are not reported as refit metrics."
        ),
    }
    target_dir = output_dir or OUT_DIR
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, "deployment_metadata.json")
    with open(output_path, "w", encoding="ascii") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return metadata


def build_date_aware_cv(df, n_splits=CV_SPLITS, date_col="date"):
    """Build rolling-origin CV folds using complete forecast dates as units."""
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    dates = pd.to_datetime(df[date_col])
    unique_dates = pd.Series(dates.dropna().unique()).sort_values().reset_index(drop=True)
    if len(unique_dates) <= n_splits:
        raise ValueError(
            f"Need more than {n_splits} unique dates for date-aware cross-validation"
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    date_values = dates.reset_index(drop=True)
    for train_date_idx, val_date_idx in splitter.split(unique_dates):
        train_dates = set(pd.to_datetime(unique_dates.iloc[train_date_idx]))
        val_dates = set(pd.to_datetime(unique_dates.iloc[val_date_idx]))
        train_idx = np.flatnonzero(date_values.isin(train_dates).to_numpy())
        val_idx = np.flatnonzero(date_values.isin(val_dates).to_numpy())
        yield train_idx, val_idx


def tune_q50(train_df):
    print("\n--- Tuning Q50 (Tweedie, date-aware rolling CV) ---")
    X, y = get_xy(train_df)
    cv_splits = list(build_date_aware_cv(train_df, n_splits=CV_SPLITS))
    grid = GridSearchCV(
        xgb.XGBRegressor(objective="reg:tweedie",
                         enable_categorical=True, tree_method="hist",
                         random_state=42, n_jobs=-1, verbosity=0),
        PARAM_GRID_Q50, cv=cv_splits, scoring="neg_mean_absolute_error",
        n_jobs=-1, verbose=1,
    )
    grid.fit(X, y)
    print(f"\n  Best params: {grid.best_params_}")
    print(f"  Best CV MAE: {-grid.best_score_:.4f}")
    return grid.best_params_


def train_quantile(train_df, quantile_alpha, best_params, is_tweedie=False):
    qname = int(quantile_alpha * 100)
    label = "Tweedie" if is_tweedie else "Quantile"
    print(f"\n--- Training Q{qname} ({label}) ---")
    X, y = get_xy(train_df)

    if is_tweedie:
        tw_power = best_params.get("tweedie_variance_power", 1.5)
        params = {k: v for k, v in best_params.items() if k != "tweedie_variance_power"}
        model = xgb.XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=tw_power,
            enable_categorical=True,
            tree_method="hist",
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
            **params,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    model.fit(X, y)
    fname = os.path.join(OUT_DIR, f"quantile_model_q{qname}.pkl")
    joblib.dump(model, fname)
    print(f"  Saved -> {fname}")
    return model


def conformal_calibrate(val_df, models, model_fit_df=None):
    print("\n--- Conformal Calibration on Val Set (target 80% coverage) ---")
    _, y_val = get_xy(val_df)
    corrected = predict_postprocessed_quantiles(val_df, models)
    q50_preds = corrected["q50"]

    per_product = {}
    for pid in sorted(val_df["product_id"].unique()):
        mask = val_df["product_id"].values == pid
        residuals = np.abs(y_val.values[mask] - q50_preds[mask])
        half_w = float(np.quantile(residuals, 0.80))
        half_w = max(half_w, 0.5)
        per_product[str(pid)] = round(half_w, 1)

    all_residuals = np.abs(y_val.values - q50_preds)
    global_half = round(max(float(np.quantile(all_residuals, 0.80)), 0.5), 1)

    calibration = {
        "method": "chronological split conformal calibration",
        "target_coverage": 0.80,
        "run_timestamp": _utc_timestamp(),
        "model_fit": (
            _split_metadata(model_fit_df, "train")
            if model_fit_df is not None
            else {"split": "train"}
        ),
        "calibration_split": _split_metadata(val_df, "validation"),
        "test_usage": "untouched until final evaluation",
        "prediction_postprocessing": {
            "algorithm": "enforce_quantile_monotonicity",
            "applied_before_residual_scoring": True,
            "raw_quantile_crossing_count": corrected["crossing_count"],
            "q50_changed_count": corrected["q50_changed_count"],
        },
        "per_product": per_product,
        "global": global_half,
    }

    out_path = os.path.join(OUT_DIR, "conformal_calibration.json")
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"  Saved -> {out_path}")
    print(f"  Global half_width: {global_half}")
    vals = list(per_product.values())
    print(f"  Per-product range: {min(vals)} to {max(vals)}")
    print(
        "  Validation raw crossings / changed Q50: "
        f"{corrected['crossing_count']} / {corrected['q50_changed_count']}"
    )

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
    _, y_test = get_xy(test_df)
    corrected = predict_postprocessed_quantiles(test_df, models)
    pre_correction_crossing_count = corrected["crossing_count"]
    q50_changed_count = corrected["q50_changed_count"]
    q10_preds = corrected["q10"]
    q50_preds = corrected["q50"]
    q90_preds = corrected["q90"]

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
        "run_timestamp": _utc_timestamp(),
        "row_count": int(len(test_df)),
        "model": "XGBoost Tweedie Q50 + Quantile Q10/Q90",
        "experiment_role": EXPERIMENT_ROLE,
        "validation_design": VALIDATION_DESIGN,
        "features": len(FEATURES),
        "feature_contract": "s2_forecasting.feature_contract.FORECAST_FEATURES",
        "test_period": f"{test_df['date'].min()} to {test_df['date'].max()}",
        "interval_method": CORE_INTERVAL_METHOD,
        "interval_scope": CORE_INTERVAL_SCOPE,
        "runtime_transform_evaluated": False,
        "runtime_transform_note": RUNTIME_TRANSFORM_NOTE,
        "model_fit": calibration.get("model_fit", {"split": "train"}),
        "calibration_split": calibration.get(
            "calibration_split",
            {"split": "validation"},
        ),
        "prediction_postprocessing": {
            "algorithm": "enforce_quantile_monotonicity",
            "applied_before_evaluation": True,
        },
        "monitoring_diagnostics": {
            "raw_quantile_crossing_count": pre_correction_crossing_count,
            "raw_quantile_crossing_rate_pct": round(
                pre_correction_crossing_count / total * 100,
                4,
            ),
            "q50_changed_count": q50_changed_count,
        },
        "overall": {
            "WAPE": round(wape, 1),
            "MAE": round(mae, 1),
            "RMSE": round(rmse, 1),
            "conformal_coverage_80": round(cov_80 * 100, 1),
            "conformal_avg_width": round(avg_width, 1),
            "raw_Q10Q90_coverage": round(cov_raw * 100, 1),
            "raw_Q10Q90_width": round(w_raw, 1),
            "pre_correction_crossing_count": pre_correction_crossing_count,
            "quantile_crossing_count": 0,
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
    best_q50 = tune_q50(train)
    with open(os.path.join(OUT_DIR, "tweedie_best_params.json"), "w") as f:
        json.dump(best_q50, f, indent=2, default=str)

    # Q10/Q90 use quantile params from simpler grid
    print("\n--- Tuning Q10 (Quantile, date-aware rolling CV) ---")
    X_all, y_all = get_xy(train)
    cv_splits = list(build_date_aware_cv(train, n_splits=CV_SPLITS))
    qt_grid = GridSearchCV(
        xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10,
                         enable_categorical=True, tree_method="hist",
                         random_state=42, n_jobs=-1, verbosity=0),
        PARAM_GRID_QUANTILE, cv=cv_splits, scoring="neg_mean_absolute_error",
        n_jobs=-1, verbose=1,
    )
    qt_grid.fit(X_all, y_all)
    best_qt = qt_grid.best_params_
    print(f"  Best Q10 params: {best_qt}")

    models = {}
    models[0.50] = train_quantile(train, 0.50, best_q50, is_tweedie=True)
    models[0.10] = train_quantile(train, 0.10, best_qt, is_tweedie=False)
    models[0.90] = train_quantile(train, 0.90, best_qt, is_tweedie=False)

    calibration = conformal_calibrate(val, models, model_fit_df=train)
    metrics, preds, y_test = evaluate(test, models, calibration)

    deployment_refit = build_deployment_refit_frame(train, val, test)
    train_quantile(
        deployment_refit,
        0.50,
        best_q50,
        is_tweedie=True,
    )
    train_quantile(
        deployment_refit,
        0.10,
        best_qt,
        is_tweedie=False,
    )
    train_quantile(
        deployment_refit,
        0.90,
        best_qt,
        is_tweedie=False,
    )
    write_deployment_metadata(train, val, test, deployment_refit)

    print(f"\n{'='*60}")
    print("  Done. Outputs -> s2_forecasting/outputs/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
