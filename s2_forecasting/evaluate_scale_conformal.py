"""Candidate B: demand-scale conformal calibration for S2.

This experiment keeps the deployed Q50 point model unchanged. It calibrates
prediction interval half-widths by predicted demand scale on the validation set
and evaluates the calibrated widths on the chronological test set.
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
EXPERIMENT_ID = "CandidateB_ScaleConformal"
EXPERIMENT_ROLE = "candidate_uncertainty_calibration"
MODEL_SCOPE = "candidate_not_deployed"
VALIDATION_DESIGN = "validation residual calibration with chronological holdout test set"
DEFAULT_N_BINS = 5
DEFAULT_BIN_CANDIDATES = [2, 3, 5, 10, 20]
DEFAULT_COVERAGE_QUANTILE = 0.80
DEFAULT_MIN_HALF_WIDTH = 0.5


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


def _fit_bin_edges(q50: np.ndarray, n_bins: int) -> list[tuple[float, float, np.ndarray]]:
    q50_arr = np.asarray(q50, dtype=float)
    if len(q50_arr) == 0:
        raise ValueError("q50 must not be empty")
    bin_count = max(1, min(int(n_bins), len(np.unique(q50_arr))))
    ranks = pd.Series(q50_arr).rank(method="first")
    labels = pd.qcut(ranks, q=bin_count, labels=False, duplicates="drop")
    bins = []
    for label in sorted(pd.Series(labels).dropna().unique()):
        mask = np.asarray(labels == label)
        bins.append((float(q50_arr[mask].min()), float(q50_arr[mask].max()), mask))
    return bins


def build_scale_calibration(
    q50,
    actual,
    n_bins: int = DEFAULT_N_BINS,
    quantile: float = DEFAULT_COVERAGE_QUANTILE,
    min_half_width: float = DEFAULT_MIN_HALF_WIDTH,
) -> dict:
    q50_arr = np.asarray(q50, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    if q50_arr.shape != actual_arr.shape:
        raise ValueError("q50 and actual arrays must have matching shapes")
    residuals = np.abs(actual_arr - q50_arr)
    global_half_width = max(float(np.quantile(residuals, quantile)), min_half_width)
    bins = []
    for lower_edge, upper_edge, mask in _fit_bin_edges(q50_arr, n_bins):
        half_width = max(float(np.quantile(residuals[mask], quantile)), min_half_width)
        bins.append(
            {
                "lower_edge": round(lower_edge, 6),
                "upper_edge": round(upper_edge, 6),
                "half_width": round(half_width, 6),
                "sample_count": int(mask.sum()),
            }
        )
    return {
        "method": "predicted-demand-scale conformal calibration",
        "target_coverage_quantile": quantile,
        "n_bins": len(bins),
        "min_half_width": min_half_width,
        "global_half_width": round(global_half_width, 6),
        "bins": bins,
    }


def apply_scale_calibration(q50, calibration: Mapping[str, object]) -> np.ndarray:
    q50_arr = np.asarray(q50, dtype=float)
    default_width = float(calibration.get("global_half_width", DEFAULT_MIN_HALF_WIDTH))
    widths = np.full(q50_arr.shape, default_width, dtype=float)
    bins = calibration.get("bins", [])
    for item in bins if isinstance(bins, list) else []:
        lower_edge = float(item["lower_edge"])
        upper_edge = float(item["upper_edge"])
        half_width = float(item["half_width"])
        mask = (q50_arr >= lower_edge) & (q50_arr <= upper_edge)
        widths[mask] = half_width
    return widths


def build_scale_prediction_frame(
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
    frame["half_width"] = apply_scale_calibration(frame["q50"].values, calibration)
    frame["lower"] = np.maximum(frame["q50"] - frame["half_width"], 0.0)
    frame["upper"] = frame["q50"] + frame["half_width"]
    frame["width"] = frame["upper"] - frame["lower"]
    frame["relative_width"] = _relative_width(frame["width"], frame["q50"])
    frame["covered"] = (frame["actual"] >= frame["lower"]) & (frame["actual"] <= frame["upper"])
    frame["absolute_error"] = np.abs(frame["actual"] - frame["q50"])
    frame["squared_error"] = (frame["actual"] - frame["q50"]) ** 2
    frame["signed_error"] = frame["q50"] - frame["actual"]
    return frame


def summarize_scale_metrics(frame: pd.DataFrame, test_period: str, calibration: Mapping[str, object]) -> dict:
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
    metrics = {
        "id": EXPERIMENT_ID,
        "name": "Demand-scale conformal candidate",
        "role": EXPERIMENT_ROLE,
        "scope": MODEL_SCOPE,
        "description": "Calibrates conformal half-width by predicted demand scale while keeping the deployed Q50 model fixed.",
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
    return metrics


def select_best_scale_candidate(candidates: list[dict], target_coverage: float = 80.0) -> dict:
    if not candidates:
        raise ValueError("At least one candidate is required")
    covered = [
        item for item in candidates
        if item.get("overall", {}).get("coverage_80", 0) >= target_coverage
    ]
    pool = covered or candidates
    return min(
        pool,
        key=lambda item: (
            item.get("overall", {}).get("avg_relative_width", float("inf")),
            abs(item.get("overall", {}).get("coverage_80", 0) - target_coverage),
        ),
    )


def load_q50_model(output_dir: Path = OUT_DIR):
    return joblib.load(output_dir / "quantile_model_q50.pkl")


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_df = pd.read_csv(data_dir / "xgboost_val.csv")
    test_df = pd.read_csv(data_dir / "xgboost_test.csv")
    return val_df, test_df


def run_experiment(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUT_DIR,
    n_bins: int | None = None,
) -> dict:
    val_df, test_df = load_data(data_dir)
    model = load_q50_model(output_dir)
    val_features = val_df[FORECAST_FEATURES].copy()
    test_features = test_df[FORECAST_FEATURES].copy()
    val_q50 = np.maximum(model.predict(val_features), 0.0)
    test_q50 = np.maximum(model.predict(test_features), 0.0)

    test_period = f"{test_df['date'].min()} to {test_df['date'].max()}"
    bin_candidates = [n_bins] if n_bins is not None else DEFAULT_BIN_CANDIDATES
    candidate_results = []
    prediction_frames = {}
    for candidate_bins in bin_candidates:
        calibration = build_scale_calibration(
            val_q50,
            val_df[TARGET].values,
            n_bins=candidate_bins,
        )
        prediction_frame = build_scale_prediction_frame(test_df, test_q50, calibration)
        metrics = summarize_scale_metrics(prediction_frame, test_period, calibration)
        metrics["n_bins"] = int(candidate_bins)
        candidate_results.append(metrics)
        prediction_frames[int(candidate_bins)] = prediction_frame

    metrics = select_best_scale_candidate(candidate_results)
    metrics["sweep"] = [
        {
            "n_bins": item["n_bins"],
            "coverage_80": item["overall"]["coverage_80"],
            "avg_width": item["overall"]["avg_width"],
            "avg_relative_width": item["overall"]["avg_relative_width"],
        }
        for item in candidate_results
    ]
    prediction_frame = prediction_frames[int(metrics["n_bins"])]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scale_conformal_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    prediction_frame.to_csv(output_dir / "scale_conformal_predictions.csv", index=False)
    return metrics


def main() -> None:
    metrics = run_experiment()
    overall = metrics["overall"]
    print("S2 demand-scale conformal candidate saved.")
    print(f"WAPE: {overall['WAPE']}%")
    print(f"Coverage: {overall['coverage_80']}%")
    print(f"Relative width: {overall['avg_relative_width']}")
    print(f"Average width: {overall['avg_width']}")


if __name__ == "__main__":
    main()
