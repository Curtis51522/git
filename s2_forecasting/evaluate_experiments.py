"""Build a thesis-ready S2 experiment summary from existing metric files."""

import json
import os
from pathlib import Path
from typing import Any, Mapping

from s2_forecasting.feature_contract import FEATURE_GROUPS, FORECAST_FEATURES

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
SUMMARY_FILENAME = "s2_experiment_results.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _pick_metrics(source: Mapping[str, Any], mapping: Mapping[str, str]) -> dict:
    metrics = {}
    for output_key, source_key in mapping.items():
        value = source.get(source_key)
        if value is not None:
            metrics[output_key] = value
    return metrics


def _build_supplementary_experiments(output_path: Path) -> list[dict]:
    classifier_metrics = _load_json(output_path / "classifier_metrics.json")
    weekly_metrics = _load_json(output_path / "weekly_metrics.json")
    experiments = []

    if classifier_metrics:
        experiments.append(
            {
                "id": "AuxClassifier",
                "name": "Auxiliary high-demand risk classifier",
                "role": classifier_metrics.get(
                    "experiment_role",
                    "auxiliary_high_demand_risk_classifier",
                ),
                "scope": classifier_metrics.get("model_scope", "auxiliary_risk_signal"),
                "description": (
                    "Classifies product-day high-demand risk using the shared "
                    "27-feature forecast-time contract. Metrics are classification "
                    "signals and are not directly comparable with WAPE."
                ),
                "validation_design": classifier_metrics.get("validation_design"),
                "metrics": _pick_metrics(
                    classifier_metrics,
                    {
                        "Accuracy": "test_Accuracy",
                        "Precision": "test_Precision",
                        "Recall": "test_Recall",
                        "F1": "test_F1",
                        "ROC_AUC": "test_ROC_AUC",
                    },
                ),
            }
        )

    if weekly_metrics:
        q50_metrics = weekly_metrics.get("Q50", {})
        baseline_metrics = weekly_metrics.get("baseline", {})
        experiments.append(
            {
                "id": "WeeklyEventAware",
                "name": "Weekly event-aware supplementary forecast",
                "role": weekly_metrics.get(
                    "experiment_role",
                    "supplementary_weekly_event_aware_forecast",
                ),
                "scope": weekly_metrics.get("model_scope", "weekly_supplementary"),
                "description": (
                    "Forecasts weekly product demand with reserved new-product and "
                    "competitor event fields for supplementary scenario analysis."
                ),
                "validation_design": weekly_metrics.get("validation_design"),
                "metrics": {
                    "baseline_WAPE": baseline_metrics.get("WAPE"),
                    "Q50_WAPE": q50_metrics.get("WAPE"),
                    "Q50_MAE": q50_metrics.get("MAE"),
                    "Q50_RMSE": q50_metrics.get("RMSE"),
                    "coverage_Q50_Q90": weekly_metrics.get("coverage_Q50_Q90"),
                    "interval_width": weekly_metrics.get("interval_width"),
                },
            }
        )

    return experiments


def build_experiment_summary(output_dir: os.PathLike | str = OUT_DIR) -> dict:
    output_path = Path(output_dir)
    baseline_metrics = _load_json(output_path / "metrics.json")
    proposed_metrics = _load_json(output_path / "test_metrics.json")
    proposed_overall = proposed_metrics.get("overall", {})
    supplementary_experiments = _build_supplementary_experiments(output_path)

    experiments = [
        {
            "id": "B0",
            "name": "Historical moving-average baseline",
            "role": "simple_business_baseline",
            "description": "Uses lag_7_avg directly as the demand prediction.",
            "metrics": _pick_metrics(
                baseline_metrics,
                {
                    "MAE": "baseline_MAE",
                    "RMSE": "baseline_RMSE",
                    "MAPE": "baseline_MAPE",
                    "R2": "baseline_R2",
                },
            ),
        },
        {
            "id": "B1",
            "name": "Deterministic XGBoost baseline",
            "role": "point_forecast_ml_baseline",
            "description": (
                "Uses the shared S2 forecast-time feature contract with a "
                "deterministic XGBoost point forecast objective."
            ),
            "metrics": _pick_metrics(
                baseline_metrics,
                {
                    "MAE": "xgboost_test_MAE",
                    "RMSE": "xgboost_test_RMSE",
                    "MAPE": "xgboost_test_MAPE",
                    "R2": "xgboost_test_R2",
                },
            ),
        },
        {
            "id": "Proposed",
            "name": "Tweedie Q50 with conformal interval",
            "role": "probabilistic_decision_support_model",
            "description": (
                "Uses the shared S2 forecast-time feature contract with a "
                "Tweedie Q50 point forecast and conformal prediction interval."
            ),
            "metrics": {
                "WAPE": proposed_overall.get("WAPE"),
                "MAE": proposed_overall.get("MAE"),
                "RMSE": proposed_overall.get("RMSE"),
                "coverage_80": proposed_overall.get("conformal_coverage_80"),
                "avg_interval_width": proposed_overall.get("conformal_avg_width"),
            },
        },
    ]

    return {
        "module": "S2 demand forecasting",
        "summary_file": SUMMARY_FILENAME,
        "feature_contract": {
            "source": "s2_forecasting.feature_contract",
            "total_features": len(FORECAST_FEATURES),
            "groups": FEATURE_GROUPS,
        },
        "validation_design": {
            "model_selection": "date-aware rolling-origin cross-validation",
            "final_evaluation": "chronological holdout test set",
        },
        "experiments": experiments,
        "supplementary_experiments": supplementary_experiments,
        "ablation_plan": [
            {
                "group": group,
                "removed_features": features,
            }
            for group, features in FEATURE_GROUPS.items()
            if group not in {"product_identity", "calendar"}
        ],
        "paper_claim": {
            "primary_model": "Proposed",
            "baseline_chain": ["B0", "B1", "Proposed"],
            "supplementary_chain": [
                item["id"] for item in supplementary_experiments
            ],
            "claim": (
                "S2 improves from a simple historical baseline to a deterministic "
                "machine-learning baseline, then adds calibrated uncertainty for "
                "downstream production and inventory decisions."
            ),
        },
    }


def write_experiment_summary(output_dir: os.PathLike | str = OUT_DIR) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = build_experiment_summary(output_path)
    out_file = output_path / SUMMARY_FILENAME
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return out_file


def main():
    out_file = write_experiment_summary()
    print(f"Saved S2 experiment summary -> {out_file}")


if __name__ == "__main__":
    main()
