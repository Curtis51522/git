"""Candidate A: log-scale S2 quantile forecast experiment.

This script trains separate Q10/Q50/Q90 XGBoost quantile models on log1p
quantity, converts predictions back to demand units, and reports whether the
log target reduces relative interval width. It writes candidate artifacts only
and does not replace the deployed S2 quantile models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from s2_forecasting.feature_contract import FORECAST_FEATURES
from s2_forecasting.quantile_utils import enforce_quantile_monotonicity


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
TARGET = "quantity"
QUANTILES = [0.10, 0.50, 0.90]
EXPERIMENT_ID = "CandidateA_LogQuantile"
EXPERIMENT_ROLE = "candidate_probabilistic_forecast"
MODEL_SCOPE = "candidate_not_deployed"
VALIDATION_DESIGN = "chronological train/validation fit with chronological holdout test set"

CANDIDATE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 10,
}


def transform_log_target(values) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    if np.any(values_arr < 0):
        raise ValueError("Target values must be non-negative for log1p transform")
    return np.log1p(values_arr)


def inverse_log_target(values) -> np.ndarray:
    return np.maximum(np.expm1(np.asarray(values, dtype=float)), 0.0)


def _wape(actual: pd.Series, predicted: pd.Series) -> float:
    actual_sum = float(actual.sum())
    if actual_sum <= 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / actual_sum * 100)


def _coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    if len(actual) == 0:
        return float("nan")
    return float(((actual >= lower) & (actual <= upper)).mean() * 100)


def _relative_width(width: pd.Series, q50: pd.Series) -> np.ndarray:
    return np.asarray(width, dtype=float) / np.maximum(np.asarray(q50, dtype=float), 1.0)


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df[FORECAST_FEATURES].copy(), transform_log_target(df[TARGET].values)


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "xgboost_train.csv")
    val = pd.read_csv(data_dir / "xgboost_val.csv")
    test = pd.read_csv(data_dir / "xgboost_test.csv")
    return train, val, test


def train_log_quantile_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    quantile_alpha: float,
    params: Mapping[str, float | int] | None = None,
) -> xgb.XGBRegressor:
    params = dict(params or CANDIDATE_PARAMS)
    X_train, y_train = get_xy(train_df)
    X_val, y_val = get_xy(val_df)
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile_alpha,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def predict_log_quantiles(
    test_df: pd.DataFrame,
    models: Mapping[float, xgb.XGBRegressor],
) -> dict[str, np.ndarray]:
    features = test_df[FORECAST_FEATURES].copy()
    return {
        f"q{int(alpha * 100)}": np.maximum(model.predict(features), 0.0)
        for alpha, model in models.items()
    }


def build_prediction_frame(
    test_df: pd.DataFrame,
    log_predictions: Mapping[str, np.ndarray | list[float]],
) -> pd.DataFrame:
    required = {"q10", "q50", "q90"}
    missing = required - set(log_predictions)
    if missing:
        raise ValueError(f"Missing log prediction arrays: {sorted(missing)}")
    if TARGET not in test_df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if "product_id" not in test_df.columns:
        raise ValueError("Missing product_id column")

    corrected = enforce_quantile_monotonicity(
        inverse_log_target(log_predictions["q10"]),
        inverse_log_target(log_predictions["q50"]),
        inverse_log_target(log_predictions["q90"]),
    )
    frame = test_df[["date", "product_id", TARGET]].copy()
    frame = frame.rename(columns={TARGET: "actual"})
    frame["had_quantile_crossing"] = corrected["crossing_mask"]
    frame["q10"] = corrected["q10"]
    frame["q50"] = corrected["q50"]
    frame["q90"] = corrected["q90"]
    frame["has_quantile_crossing"] = (frame["q10"] > frame["q50"]) | (frame["q50"] > frame["q90"])
    frame["raw_interval_width"] = frame["q90"] - frame["q10"]
    frame["raw_relative_width"] = _relative_width(frame["raw_interval_width"], frame["q50"])
    frame["raw_interval_covered"] = (frame["actual"] >= frame["q10"]) & (frame["actual"] <= frame["q90"])
    frame["absolute_error"] = np.abs(frame["actual"] - frame["q50"])
    frame["squared_error"] = (frame["actual"] - frame["q50"]) ** 2
    frame["signed_error"] = frame["q50"] - frame["actual"]
    return frame


def summarize_candidate_metrics(frame: pd.DataFrame, test_period: str) -> dict:
    frame = frame.copy()
    if "raw_interval_width" not in frame.columns:
        frame["raw_interval_width"] = frame["q90"] - frame["q10"]
    if "raw_relative_width" not in frame.columns:
        frame["raw_relative_width"] = _relative_width(frame["raw_interval_width"], frame["q50"])
    if "absolute_error" not in frame.columns:
        frame["absolute_error"] = np.abs(frame["actual"] - frame["q50"])
    if "squared_error" not in frame.columns:
        frame["squared_error"] = (frame["actual"] - frame["q50"]) ** 2
    if "signed_error" not in frame.columns:
        frame["signed_error"] = frame["q50"] - frame["actual"]

    actual = frame["actual"]
    q50 = frame["q50"]
    width = frame["raw_interval_width"]
    metrics = {
        "id": EXPERIMENT_ID,
        "name": "Log-scale quantile candidate",
        "role": EXPERIMENT_ROLE,
        "scope": MODEL_SCOPE,
        "description": "Trains Q10/Q50/Q90 quantile models on log1p quantity and evaluates outputs in demand units.",
        "validation_design": VALIDATION_DESIGN,
        "feature_contract": "s2_forecasting.feature_contract.FORECAST_FEATURES",
        "features": len(FORECAST_FEATURES),
        "test_period": test_period,
        "overall": {
            "WAPE": round(_wape(actual, q50), 1),
            "MAE": round(float(frame["absolute_error"].mean()), 1),
            "RMSE": round(float(np.sqrt(frame["squared_error"].mean())), 1),
            "raw_Q10Q90_coverage": round(_coverage(actual, frame["q10"], frame["q90"]), 1),
            "raw_Q10Q90_width": round(float(width.mean()), 1),
            "avg_raw_relative_width": round(float(frame["raw_relative_width"].mean()), 4),
            "bias_q50_minus_actual": round(float(frame["signed_error"].mean()), 4),
            "pre_correction_crossing_count": int(frame["had_quantile_crossing"].sum()),
            "quantile_crossing_count": int(frame["has_quantile_crossing"].sum()),
        },
    }
    return metrics


def run_experiment(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUT_DIR,
    params: Mapping[str, float | int] | None = None,
) -> dict:
    train_df, val_df, test_df = load_data(data_dir)
    models = {
        alpha: train_log_quantile_model(train_df, val_df, alpha, params)
        for alpha in QUANTILES
    }
    log_predictions = predict_log_quantiles(test_df, models)
    prediction_frame = build_prediction_frame(test_df, log_predictions)
    test_period = f"{test_df['date'].min()} to {test_df['date'].max()}"
    metrics = summarize_candidate_metrics(prediction_frame, test_period)
    metrics["params"] = dict(params or CANDIDATE_PARAMS)

    output_dir.mkdir(parents=True, exist_ok=True)
    for alpha, model in models.items():
        joblib.dump(model, output_dir / f"log_quantile_model_q{int(alpha * 100)}.pkl")
    with (output_dir / "log_quantile_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    prediction_frame.to_csv(output_dir / "log_quantile_predictions.csv", index=False)
    return metrics


def main() -> None:
    metrics = run_experiment()
    overall = metrics["overall"]
    print("S2 log-scale quantile candidate saved.")
    print(f"WAPE: {overall['WAPE']}%")
    print(f"Coverage: {overall['raw_Q10Q90_coverage']}%")
    print(f"Relative width: {overall['avg_raw_relative_width']}")
    print(f"Crossing count: {overall['quantile_crossing_count']}")


if __name__ == "__main__":
    main()
