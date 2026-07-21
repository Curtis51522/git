"""Build a provenance-checked S2 experiment acceptance summary."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from s2_forecasting.feature_contract import FEATURE_GROUPS, FORECAST_FEATURES


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
SUMMARY_FILENAME = "s2_experiment_results.json"
PROVENANCE_FIELDS = ("run_timestamp", "row_count", "test_period")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    return data if isinstance(data, dict) else {}


def _pick_metrics(source: Mapping[str, Any], mapping: Mapping[str, str]) -> dict:
    return {
        output_key: source[source_key]
        for output_key, source_key in mapping.items()
        if source.get(source_key) is not None
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_artifacts(paths: list[Path]) -> list[dict]:
    return [
        {"path": path.name, "sha256": _sha256(path)}
        for path in paths
        if path.exists()
    ]


def _extract_provenance(source: Mapping[str, Any], source_path: Path) -> dict:
    missing = [field for field in PROVENANCE_FIELDS if source.get(field) in (None, "")]
    if missing:
        raise ValueError(
            f"{source_path.name} provenance missing fields: {', '.join(missing)}"
        )
    try:
        row_count = int(source["row_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_path.name} provenance row_count is invalid") from exc
    if row_count <= 0:
        raise ValueError(f"{source_path.name} provenance row_count must be positive")
    try:
        datetime.fromisoformat(str(source["run_timestamp"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{source_path.name} provenance run_timestamp is invalid") from exc
    return {
        "run_timestamp": str(source["run_timestamp"]),
        "row_count": row_count,
        "test_period": str(source["test_period"]),
        "source_artifacts": _source_artifacts([source_path]),
    }


def _actual_test_contract(output_path: Path, fallback: Mapping[str, Any]) -> dict:
    test_path = output_path.parent.parent / "data" / "xgboost_test.csv"
    if not test_path.exists():
        return {
            "row_count": int(fallback["row_count"]),
            "test_period": str(fallback["test_period"]),
        }
    test_dates = pd.read_csv(test_path, usecols=["date"])["date"]
    return {
        "row_count": int(len(test_dates)),
        "test_period": f"{test_dates.min()} to {test_dates.max()}",
    }


def _require_contract(provenance: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for field in ("row_count", "test_period"):
        if provenance[field] != expected[field]:
            raise ValueError(
                f"{label} provenance {field}={provenance[field]!r} does not match "
                f"current test contract {expected[field]!r}"
            )


def _validated_optional_provenance(
    source: Mapping[str, Any],
    source_path: Path,
    expected: Mapping[str, Any],
    prediction_path: Path | None = None,
) -> dict:
    provenance = _extract_provenance(source, source_path)
    _require_contract(provenance, expected, source_path.name)
    paths = [source_path]
    if prediction_path is not None:
        if not prediction_path.exists():
            raise ValueError(
                f"{source_path.name} provenance prediction artifact missing: "
                f"{prediction_path.name}"
            )
        predictions = pd.read_csv(prediction_path, usecols=["date"])
        prediction_contract = {
            "row_count": int(len(predictions)),
            "test_period": (
                f"{predictions['date'].min()} to {predictions['date'].max()}"
            ),
        }
        _require_contract(prediction_contract, expected, prediction_path.name)
        paths.append(prediction_path)
    provenance["source_artifacts"] = _source_artifacts(paths)
    return provenance


def _excluded_entry(experiment_id: str, source_path: Path, reason: str) -> dict:
    return {
        "id": experiment_id,
        "reason": reason,
        "source_artifacts": _source_artifacts([source_path]),
    }


def _build_candidate_experiments(
    output_path: Path,
    expected: Mapping[str, Any],
) -> tuple[list[dict], list[dict]]:
    configs = [
        {
            "id": "CandidateA_LogQuantile",
            "metrics_file": "log_quantile_metrics.json",
            "prediction_file": None,
            "name": "Log-scale quantile candidate",
            "role": "candidate_probabilistic_forecast",
            "scope": "candidate_not_deployed",
            "description": "Trains Q10/Q50/Q90 quantile models on log1p quantity.",
            "metric_keys": {
                "WAPE": "WAPE",
                "MAE": "MAE",
                "RMSE": "RMSE",
                "raw_Q10Q90_coverage": "raw_Q10Q90_coverage",
                "avg_raw_relative_width": "avg_raw_relative_width",
                "quantile_crossing_count": "quantile_crossing_count",
            },
        },
        {
            "id": "CandidateB_ScaleConformal",
            "metrics_file": "scale_conformal_metrics.json",
            "prediction_file": "scale_conformal_predictions.csv",
            "name": "Demand-scale conformal candidate",
            "role": "candidate_uncertainty_calibration",
            "scope": "candidate_not_deployed",
            "description": "Selects scale bins on validation and evaluates test once.",
            "metric_keys": {
                "WAPE": "WAPE",
                "MAE": "MAE",
                "RMSE": "RMSE",
                "coverage_80": "coverage_80",
                "avg_width": "avg_width",
                "avg_relative_width": "avg_relative_width",
            },
        },
        {
            "id": "CandidateC_RelativeConformal",
            "metrics_file": "relative_conformal_metrics.json",
            "prediction_file": "relative_conformal_predictions.csv",
            "name": "Relative conformal candidate",
            "role": "candidate_uncertainty_calibration",
            "scope": "candidate_not_deployed",
            "description": "Calibrates interval width using normalized residuals.",
            "metric_keys": {
                "WAPE": "WAPE",
                "MAE": "MAE",
                "RMSE": "RMSE",
                "coverage_80": "coverage_80",
                "avg_width": "avg_width",
                "avg_relative_width": "avg_relative_width",
            },
        },
    ]
    experiments = []
    excluded = []
    for config in configs:
        source_path = output_path / config["metrics_file"]
        source = _load_json(source_path)
        if not source:
            excluded.append(
                _excluded_entry(config["id"], source_path, "missing or empty artifact")
            )
            continue
        prediction_path = (
            output_path / config["prediction_file"]
            if config["prediction_file"]
            else None
        )
        try:
            provenance = _validated_optional_provenance(
                source,
                source_path,
                expected,
                prediction_path,
            )
        except ValueError as exc:
            excluded.append(_excluded_entry(config["id"], source_path, str(exc)))
            continue
        overall = source.get("overall", {})
        experiments.append(
            {
                "id": source.get("id", config["id"]),
                "name": source.get("name", config["name"]),
                "role": source.get("role", config["role"]),
                "scope": source.get("scope", config["scope"]),
                "description": source.get("description", config["description"]),
                "validation_design": source.get("validation_design"),
                "metrics": _pick_metrics(overall, config["metric_keys"]),
                "provenance": provenance,
            }
        )
    return experiments, excluded


def _build_supplementary_experiments(
    output_path: Path,
    expected: Mapping[str, Any],
) -> tuple[list[dict], list[dict]]:
    configs = [
        ("AuxClassifier", "classifier_metrics.json"),
        ("WeeklyEventAware", "weekly_metrics.json"),
    ]
    experiments = []
    excluded = []
    for experiment_id, filename in configs:
        source_path = output_path / filename
        source = _load_json(source_path)
        if not source:
            excluded.append(
                _excluded_entry(experiment_id, source_path, "missing or empty artifact")
            )
            continue
        try:
            provenance = _validated_optional_provenance(source, source_path, expected)
        except ValueError as exc:
            excluded.append(_excluded_entry(experiment_id, source_path, str(exc)))
            continue
        experiments.append(
            {
                "id": experiment_id,
                "name": source.get("name", experiment_id),
                "role": source.get("experiment_role"),
                "scope": source.get("model_scope"),
                "validation_design": source.get("validation_design"),
                "metrics": source.get("overall", {}),
                "provenance": provenance,
            }
        )
    return experiments, excluded


def build_experiment_summary(output_dir: os.PathLike | str = OUT_DIR) -> dict:
    output_path = Path(output_dir)
    baseline_path = output_path / "metrics.json"
    proposed_path = output_path / "test_metrics.json"
    baseline_metrics = _load_json(baseline_path)
    proposed_metrics = _load_json(proposed_path)

    baseline_provenance = _extract_provenance(baseline_metrics, baseline_path)
    proposed_provenance = _extract_provenance(proposed_metrics, proposed_path)
    expected = _actual_test_contract(output_path, baseline_provenance)
    _require_contract(baseline_provenance, expected, baseline_path.name)
    _require_contract(proposed_provenance, expected, proposed_path.name)

    proposed_overall = proposed_metrics.get("overall", {})
    experiments = [
        {
            "id": "B0",
            "name": baseline_metrics.get(
                "naive_baseline_name",
                "B0 lag_7_avg",
            ),
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
            "provenance": baseline_provenance,
        },
        {
            "id": "B1",
            "name": baseline_metrics.get(
                "experiment_name",
                "B1 deterministic XGBoost",
            ),
            "role": "point_forecast_ml_baseline",
            "description": "Deterministic XGBoost point forecast baseline.",
            "metrics": _pick_metrics(
                baseline_metrics,
                {
                    "MAE": "xgboost_test_MAE",
                    "RMSE": "xgboost_test_RMSE",
                    "MAPE": "xgboost_test_MAPE",
                    "R2": "xgboost_test_R2",
                },
            ),
            "provenance": baseline_provenance,
        },
        {
            "id": "Proposed",
            "name": "Tweedie Q50 with conformal interval",
            "role": "probabilistic_decision_support_model",
            "description": (
                "Train-only Q50 with independent validation split-conformal "
                "calibration. Metrics cover the core pre-runtime-bias interval."
            ),
            "interval_scope": proposed_metrics.get("interval_scope"),
            "runtime_transform_evaluated": proposed_metrics.get(
                "runtime_transform_evaluated",
                False,
            ),
            "runtime_transform_note": proposed_metrics.get("runtime_transform_note"),
            "metrics": {
                "WAPE": proposed_overall.get("WAPE"),
                "MAE": proposed_overall.get("MAE"),
                "RMSE": proposed_overall.get("RMSE"),
                "coverage_80": proposed_overall.get("conformal_coverage_80"),
                "avg_interval_width": proposed_overall.get("conformal_avg_width"),
            },
            "provenance": proposed_provenance,
        },
    ]

    candidate_experiments, candidate_excluded = _build_candidate_experiments(
        output_path,
        expected,
    )
    supplementary_experiments, supplementary_excluded = (
        _build_supplementary_experiments(output_path, expected)
    )

    return {
        "module": "S2 demand forecasting",
        "summary_file": SUMMARY_FILENAME,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "acceptance_contract": expected,
        "feature_contract": {
            "source": "s2_forecasting.feature_contract",
            "total_features": len(FORECAST_FEATURES),
            "groups": FEATURE_GROUPS,
        },
        "validation_design": {
            "model_selection": "train-only date-aware rolling-origin cross-validation",
            "calibration": "independent chronological validation split",
            "final_evaluation": "chronological holdout test set evaluated once",
        },
        "experiments": experiments,
        "candidate_experiments": candidate_experiments,
        "supplementary_experiments": supplementary_experiments,
        "excluded_artifacts": candidate_excluded + supplementary_excluded,
        "ablation_plan": [
            {"group": group, "removed_features": features}
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
                "S2 improves from a lag-7 baseline to a deterministic model, "
                "then adds leakage-free core conformal uncertainty estimates."
            ),
        },
    }


def write_experiment_summary(output_dir: os.PathLike | str = OUT_DIR) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = build_experiment_summary(output_path)
    out_file = output_path / SUMMARY_FILENAME
    with out_file.open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2, default=str)
    return out_file


def main() -> None:
    out_file = write_experiment_summary()
    print(f"Saved S2 experiment summary -> {out_file}")


if __name__ == "__main__":
    main()
