"""Candidate C: relative conformal calibration for S2 intervals.

This experiment keeps the deployed Q50 model unchanged. It calibrates
normalized residuals on the validation set and converts the relative
half-width back into demand-unit intervals on the chronological test set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from s2_forecasting.feature_contract import FORECAST_FEATURES


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
TARGET = "quantity"
EXPERIMENT_ID = "CandidateC_RelativeConformal"
EXPERIMENT_ROLE = "candidate_uncertainty_calibration"
MODEL_SCOPE = "candidate_not_deployed"
VALIDATION_DESIGN = "validation normalized-residual calibration with chronological holdout test set"
DEFAULT_COVERAGE_QUANTILE = 0.80
DEFAULT_DENOMINATOR_FLOOR = 1.0
DEFAULT_MIN_RELATIVE_HALF_WIDTH = 0.10
DEFAULT_MIN_ABSOLUTE_HALF_WIDTH = 0.5


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


def build_relative_calibration(
    q50,
    actual,
    quantile: float = DEFAULT_COVERAGE_QUANTILE,
    min_relative_half_width: float = DEFAULT_MIN_RELATIVE_HALF_WIDTH,
    denominator_floor: float = DEFAULT_DENOMINATOR_FLOOR,
    min_absolute_half_width: float = DEFAULT_MIN_ABSOLUTE_HALF_WIDTH,
) -> dict:
    q50_arr = np.asarray(q50, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    if q50_arr.shape != actual_arr.shape:
        raise ValueError("q50 and actual arrays must have matching shapes")
    if len(q50_arr) == 0:
        raise ValueError("q50 must not be empty")

    denominator = np.maximum(q50_arr, denominator_floor)
    normalized_residuals = np.abs(actual_arr - q50_arr) / denominator
    relative_half_width = max(
        float(np.quantile(normalized_residuals, quantile)),
        min_relative_half_width,
    )
    return {
        "method": "relative conformal calibration",
        "target_coverage_quantile": quantile,
        "relative_half_width": round(relative_half_width, 6),
        "denominator_floor": denominator_floor,
        "min_relative_half_width": min_relative_half_width,
        "min_absolute_half_width": min_absolute_half_width,
    }


def apply_relative_calibration(q50, calibration: Mapping[str, object]) -> np.ndarray:
    q50_arr = np.asarray(q50, dtype=float)
    relative_half_width = float(calibration.get("relative_half_width", DEFAULT_MIN_RELATIVE_HALF_WIDTH))
    denominator_floor = float(calibration.get("denominator_floor", DEFAULT_DENOMINATOR_FLOOR))
    min_absolute_half_width = float(calibration.get("min_absolute_half_width", DEFAULT_MIN_ABSOLUTE_HALF_WIDTH))
    widths = np.maximum(q50_arr, denominator_floor) * relative_half_width
    return np.maximum(widths, min_absolute_half_width)


def build_relative_prediction_frame(
    test_df: pd.DataFrame,
    q50,
    calibration: Mapping[str, object],
) -> pd.DataFrame:
    if TARGET not in test_df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if "product_id" not in test_df.columns:
        raise ValueError("Missing product_id column")

    frame = test_df[["date", "product_id", TARGET]].copy()
    frame = frame.rename(columns={TARGET: "actual"})
    frame["q50"] = np.maximum(np.asarray(q50, dtype=float), 0.0)
    frame["half_width"] = apply_relative_calibration(frame["q50"].values, calibration)
    frame["lower"] = np.maximum(frame["q50"] - frame["half_width"], 0.0)
    frame["upper"] = frame["q50"] + frame["half_width"]
    frame["width"] = frame["upper"] - frame["lower"]
    frame["relative_width"] = _relative_width(frame["width"], frame["q50"])
    frame["covered"] = (frame["actual"] >= frame["lower"]) & (frame["actual"] <= frame["upper"])
    frame["absolute_error"] = np.abs(frame["actual"] - frame["q50"])
    frame["squared_error"] = (frame["actual"] - frame["q50"]) ** 2
    frame["signed_error"] = frame["q50"] - frame["actual"]
    return frame


def summarize_relative_metrics(frame: pd.DataFrame, test_period: str, calibration: Mapping[str, object]) -> dict:
    frame = frame.copy()
    if "width" not in frame.columns:
        frame["width"] = frame["upper"] - frame["lower"]
    if "relative_width" not in frame.columns:
        frame["relative_width"] = _relative_width(frame["width"], frame["q50"])
    if "absolute_error" not in frame.columns:
        frame["absolute_error"] = np.abs(frame["actual"] - frame["q50"])
    if "squared_error" not in frame.columns:
        frame["squared_error"] = (frame["actual"] - frame["q50"]) ** 2
    if "signed_error" not in frame.columns:
        frame["signed_error"] = frame["q50"] - frame["actual"]

    actual = frame["actual"]
    q50 = frame["q50"]
    return {
        "id": EXPERIMENT_ID,
        "name": "Relative conformal candidate",
        "role": EXPERIMENT_ROLE,
        "scope": MODEL_SCOPE,
        "description": "Calibrates interval width using normalized residuals while keeping the deployed Q50 model fixed.",
        "validation_design": VALIDATION_DESIGN,
        "feature_contract": "s2_forecasting.feature_contract.FORECAST_FEATURES",
        "features": len(FORECAST_FEATURES),
        "test_period": test_period,
        "calibration": calibration,
        "overall": {
            "WAPE": round(_wape(actual, q50), 1),
            "MAE": round(float(frame["absolute_error"].mean()), 1),
            "RMSE": round(float(np.sqrt(frame["squared_error"].mean())), 1),
            "coverage_80": round(_coverage(actual, frame["lower"], frame["upper"]), 1),
            "avg_width": round(float(frame["width"].mean()), 1),
            "avg_relative_width": round(float(frame["relative_width"].mean()), 4),
            "bias_q50_minus_actual": round(float(frame["signed_error"].mean()), 4),
        },
    }


def load_q50_model(output_dir: Path = OUT_DIR):
    return joblib.load(output_dir / "quantile_model_q50.pkl")


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_df = pd.read_csv(data_dir / "xgboost_val.csv")
    test_df = pd.read_csv(data_dir / "xgboost_test.csv")
    return val_df, test_df


def run_experiment(data_dir: Path = DATA_DIR, output_dir: Path = OUT_DIR) -> dict:
    val_df, test_df = load_data(data_dir)
    model = load_q50_model(output_dir)
    val_q50 = np.maximum(model.predict(val_df[FORECAST_FEATURES].copy()), 0.0)
    test_q50 = np.maximum(model.predict(test_df[FORECAST_FEATURES].copy()), 0.0)

    calibration = build_relative_calibration(val_q50, val_df[TARGET].values)
    prediction_frame = build_relative_prediction_frame(test_df, test_q50, calibration)
    test_period = f"{test_df['date'].min()} to {test_df['date'].max()}"
    metrics = summarize_relative_metrics(prediction_frame, test_period, calibration)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "relative_conformal_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    prediction_frame.to_csv(output_dir / "relative_conformal_predictions.csv", index=False)
    return metrics


def main() -> None:
    metrics = run_experiment()
    overall = metrics["overall"]
    print("S2 relative conformal candidate saved.")
    print(f"WAPE: {overall['WAPE']}%")
    print(f"Coverage: {overall['coverage_80']}%")
    print(f"Relative width: {overall['avg_relative_width']}")
    print(f"Average width: {overall['avg_width']}")


if __name__ == "__main__":
    main()
