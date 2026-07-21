"""Diagnose S2 prediction interval usefulness.

This script does not train models. It loads the deployed S2 Q10/Q50/Q90 models,
scores the chronological test split, and reports whether prediction intervals
are useful after accounting for demand scale.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from s2_forecasting.feature_contract import FORECAST_FEATURES
from s2_forecasting.quantile_utils import enforce_quantile_monotonicity


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
TARGET = "quantity"
DEFAULT_HIGH_UNCERTAINTY_THRESHOLD = 0.60
CORE_INTERVAL_SCOPE = "core_pre_runtime_bias_conformal"
RUNTIME_TRANSFORM_NOTE = (
    "The live endpoint applies operation-specific bakery/beverage bias scaling "
    "and integer rounding. These diagnostics do not evaluate that dynamic transform."
)


def _safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    denominator_arr = np.maximum(np.asarray(denominator, dtype=float), 1.0)
    return np.asarray(numerator, dtype=float) / denominator_arr


def _wape(actual: pd.Series, predicted: pd.Series) -> float:
    actual_sum = float(actual.sum())
    if actual_sum <= 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / actual_sum * 100)


def _coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    if len(actual) == 0:
        return float("nan")
    covered = ((actual >= lower) & (actual <= upper)).mean()
    return float(covered * 100)


def _round_or_none(value: float, digits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def load_product_names(output_dir: Path = OUT_DIR) -> dict[int, str]:
    path = output_dir / "product_id_map.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        product_to_id = json.load(f)
    return {int(product_id): name for name, product_id in product_to_id.items()}


def load_quantile_models(output_dir: Path = OUT_DIR) -> dict[str, object]:
    return {
        "q10": joblib.load(output_dir / "quantile_model_q10.pkl"),
        "q50": joblib.load(output_dir / "quantile_model_q50.pkl"),
        "q90": joblib.load(output_dir / "quantile_model_q90.pkl"),
    }


def load_calibration(output_dir: Path = OUT_DIR) -> dict:
    path = output_dir / "conformal_calibration.json"
    if not path.exists():
        return {"per_product": {}, "global": 0.5}
    with path.open("r", encoding="utf-8") as f:
        calibration = json.load(f)
    calibration.setdefault("per_product", {})
    calibration.setdefault("global", 0.5)
    return calibration


def predict_quantiles(test_df: pd.DataFrame, models: Mapping[str, object]) -> dict[str, np.ndarray]:
    features = test_df[FORECAST_FEATURES].copy()
    return {
        name: np.maximum(model.predict(features), 0)
        for name, model in models.items()
    }


def build_interval_frame(
    test_df: pd.DataFrame,
    predictions: Mapping[str, np.ndarray | list[float]],
    calibration: Mapping[str, object],
) -> pd.DataFrame:
    required_prediction_keys = {"q10", "q50", "q90"}
    missing = required_prediction_keys - set(predictions)
    if missing:
        raise ValueError(f"Missing prediction arrays: {sorted(missing)}")
    if TARGET not in test_df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if "product_id" not in test_df.columns:
        raise ValueError("Missing product_id column")

    frame = test_df[["date", "product_id", TARGET]].copy()
    frame = frame.rename(columns={TARGET: "actual"})
    corrected = enforce_quantile_monotonicity(
        predictions["q10"],
        predictions["q50"],
        predictions["q90"],
    )
    frame["had_quantile_crossing"] = corrected["crossing_mask"]
    frame["q10"] = corrected["q10"]
    frame["q50"] = corrected["q50"]
    frame["q90"] = corrected["q90"]

    per_product = calibration.get("per_product", {}) if isinstance(calibration, Mapping) else {}
    global_half_width = float(calibration.get("global", 0.5)) if isinstance(calibration, Mapping) else 0.5
    frame["conformal_half_width"] = frame["product_id"].map(
        lambda product_id: float(per_product.get(str(int(product_id)), global_half_width))
    )

    frame["has_quantile_crossing"] = (frame["q10"] > frame["q50"]) | (frame["q50"] > frame["q90"])
    frame["raw_lower"] = frame["q10"]
    frame["raw_upper"] = frame["q90"]
    frame["conformal_lower"] = np.maximum(frame["q50"] - frame["conformal_half_width"], 0)
    frame["conformal_upper"] = frame["q50"] + frame["conformal_half_width"]

    frame["raw_interval_width"] = frame["raw_upper"] - frame["raw_lower"]
    frame["conformal_interval_width"] = frame["conformal_upper"] - frame["conformal_lower"]
    frame["raw_relative_width"] = _safe_divide(frame["raw_interval_width"], frame["q50"])
    frame["conformal_relative_width"] = _safe_divide(frame["conformal_interval_width"], frame["q50"])
    frame["raw_upper_risk"] = _safe_divide(frame["raw_upper"] - frame["q50"], frame["q50"])
    frame["raw_lower_risk"] = _safe_divide(frame["q50"] - frame["raw_lower"], frame["q50"])

    frame["raw_interval_covered"] = (
        (frame["actual"] >= frame["raw_lower"]) & (frame["actual"] <= frame["raw_upper"])
    )
    frame["conformal_interval_covered"] = (
        (frame["actual"] >= frame["conformal_lower"]) & (frame["actual"] <= frame["conformal_upper"])
    )
    frame["absolute_error"] = np.abs(frame["actual"] - frame["q50"])
    frame["signed_error"] = frame["q50"] - frame["actual"]
    return frame


def summarize_by_product(
    interval_frame: pd.DataFrame,
    product_names: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    product_names = product_names or {}
    rows = []
    for product_id, group in interval_frame.groupby("product_id", sort=True):
        actual = group["actual"]
        q50 = group["q50"]
        rows.append(
            {
                "product_id": int(product_id),
                "product_name": product_names.get(int(product_id), str(int(product_id))),
                "sample_count": int(len(group)),
                "avg_actual": round(float(actual.mean()), 4),
                "avg_q50": round(float(q50.mean()), 4),
                "avg_raw_interval_width": round(float(group["raw_interval_width"].mean()), 4),
                "avg_raw_relative_width": round(float(group["raw_relative_width"].mean()), 4),
                "avg_conformal_interval_width": round(float(group["conformal_interval_width"].mean()), 4),
                "avg_conformal_relative_width": round(float(group["conformal_relative_width"].mean()), 4),
                "raw_interval_coverage_pct": round(_coverage(actual, group["raw_lower"], group["raw_upper"]), 4),
                "conformal_interval_coverage_pct": round(
                    _coverage(actual, group["conformal_lower"], group["conformal_upper"]), 4
                ),
                "wape_pct": round(_wape(actual, q50), 4),
                "bias_q50_minus_actual": round(float(group["signed_error"].mean()), 4),
                "pre_correction_crossing_count": int(group["had_quantile_crossing"].sum()),
                "quantile_crossing_count": int(group["has_quantile_crossing"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_demand_decile(interval_frame: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if interval_frame.empty:
        return pd.DataFrame()
    frame = interval_frame.copy()
    bin_count = max(1, min(n_bins, len(frame)))
    ranked_q50 = frame["q50"].rank(method="first")
    frame["demand_decile"] = pd.qcut(ranked_q50, q=bin_count, labels=False) + 1

    rows = []
    for demand_decile, group in frame.groupby("demand_decile", sort=True):
        actual = group["actual"]
        q50 = group["q50"]
        rows.append(
            {
                "demand_decile": int(demand_decile),
                "sample_count": int(len(group)),
                "avg_actual": round(float(actual.mean()), 4),
                "avg_q50": round(float(q50.mean()), 4),
                "avg_raw_interval_width": round(float(group["raw_interval_width"].mean()), 4),
                "avg_raw_relative_width": round(float(group["raw_relative_width"].mean()), 4),
                "avg_conformal_interval_width": round(float(group["conformal_interval_width"].mean()), 4),
                "avg_conformal_relative_width": round(float(group["conformal_relative_width"].mean()), 4),
                "raw_interval_coverage_pct": round(_coverage(actual, group["raw_lower"], group["raw_upper"]), 4),
                "conformal_interval_coverage_pct": round(
                    _coverage(actual, group["conformal_lower"], group["conformal_upper"]), 4
                ),
                "wape_pct": round(_wape(actual, q50), 4),
                "bias_q50_minus_actual": round(float(group["signed_error"].mean()), 4),
                "pre_correction_crossing_count": int(group["had_quantile_crossing"].sum()),
                "quantile_crossing_count": int(group["has_quantile_crossing"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_overall(
    interval_frame: pd.DataFrame,
    high_uncertainty_threshold: float = DEFAULT_HIGH_UNCERTAINTY_THRESHOLD,
    product_names: Mapping[int, str] | None = None,
) -> dict:
    product_names = product_names or {}
    by_product = summarize_by_product(interval_frame, product_names)
    high_uncertainty = by_product[
        by_product["avg_raw_relative_width"] >= high_uncertainty_threshold
    ]
    widest_relative_product = None
    if not by_product.empty:
        widest = by_product.sort_values("avg_raw_relative_width", ascending=False).iloc[0]
        widest_relative_product = str(widest["product_name"])

    return {
        "sample_count": int(len(interval_frame)),
        "product_count": int(interval_frame["product_id"].nunique()),
        "conformal_interval_scope": CORE_INTERVAL_SCOPE,
        "runtime_transform_evaluated": False,
        "raw_interval_coverage_pct": round(
            _coverage(interval_frame["actual"], interval_frame["raw_lower"], interval_frame["raw_upper"]),
            4,
        ),
        "conformal_interval_coverage_pct": round(
            _coverage(
                interval_frame["actual"],
                interval_frame["conformal_lower"],
                interval_frame["conformal_upper"],
            ),
            4,
        ),
        "avg_q50": round(float(interval_frame["q50"].mean()), 4),
        "avg_raw_interval_width": round(float(interval_frame["raw_interval_width"].mean()), 4),
        "avg_raw_relative_width": round(float(interval_frame["raw_relative_width"].mean()), 4),
        "avg_conformal_interval_width": round(float(interval_frame["conformal_interval_width"].mean()), 4),
        "avg_conformal_relative_width": round(float(interval_frame["conformal_relative_width"].mean()), 4),
        "wape_pct": round(_wape(interval_frame["actual"], interval_frame["q50"]), 4),
        "bias_q50_minus_actual": round(float(interval_frame["signed_error"].mean()), 4),
        "pre_correction_crossing_count": int(interval_frame["had_quantile_crossing"].sum()),
        "quantile_crossing_count": int(interval_frame["has_quantile_crossing"].sum()),
        "high_uncertainty_threshold": high_uncertainty_threshold,
        "high_uncertainty_product_count": int(len(high_uncertainty)),
        "widest_relative_product": widest_relative_product,
    }


def run_diagnostics(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUT_DIR,
    high_uncertainty_threshold: float = DEFAULT_HIGH_UNCERTAINTY_THRESHOLD,
) -> dict:
    test_df = pd.read_csv(data_dir / "xgboost_test.csv")
    models = load_quantile_models(output_dir)
    calibration = load_calibration(output_dir)
    predictions = predict_quantiles(test_df, models)
    interval_frame = build_interval_frame(test_df, predictions, calibration)
    product_names = load_product_names(output_dir)

    by_product = summarize_by_product(interval_frame, product_names)
    by_decile = summarize_by_demand_decile(interval_frame)
    overall = summarize_overall(interval_frame, high_uncertainty_threshold, product_names)

    diagnostics = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(test_df)),
        "test_period": f"{test_df['date'].min()} to {test_df['date'].max()}",
        "module": "S2 interval diagnostics",
        "purpose": "Evaluate prediction interval usefulness by demand scale.",
        "runtime_transform_note": RUNTIME_TRANSFORM_NOTE,
        "intervals": {
            "raw": "Q10 to Q90",
            "conformal": (
                "Core pre-runtime-bias Q50 plus or minus product-level "
                "conformal half-width"
            ),
        },
        "overall": overall,
        "outputs": {
            "by_product": "interval_by_product.csv",
            "by_demand_decile": "interval_by_demand_decile.csv",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "interval_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=_round_or_none)
    by_product.to_csv(output_dir / "interval_by_product.csv", index=False)
    by_decile.to_csv(output_dir / "interval_by_demand_decile.csv", index=False)
    return diagnostics


def main() -> None:
    diagnostics = run_diagnostics()
    overall = diagnostics["overall"]
    print("S2 interval diagnostics saved.")
    print(f"Raw coverage: {overall['raw_interval_coverage_pct']}%")
    print(f"Raw relative width: {overall['avg_raw_relative_width']}")
    print(f"High-uncertainty products: {overall['high_uncertainty_product_count']}")


if __name__ == "__main__":
    main()
