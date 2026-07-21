import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pytest


def test_quantile_training_uses_train_only_and_date_aware_cv(monkeypatch, tmp_path):
    from s2_forecasting import train_quantile

    df = pd.DataFrame(
        {
            "date": [
                "2026-01-03",
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-01",
                "2026-01-02",
            ],
            "product_id": [0, 0, 0, 1, 1, 1],
            "quantity": [3, 1, 2, 6, 4, 5],
        }
    )

    splits = list(train_quantile.build_date_aware_cv(df, n_splits=2))

    assert len(splits) == 2
    for train_idx, val_idx in splits:
        train_dates = pd.to_datetime(df.iloc[train_idx]["date"])
        val_dates = pd.to_datetime(df.iloc[val_idx]["date"])
        assert train_dates.max() < val_dates.min()
        assert len(set(train_idx).intersection(set(val_idx))) == 0

    assert list(inspect.signature(train_quantile.tune_q50).parameters) == ["train_df"]
    assert list(inspect.signature(train_quantile.train_quantile).parameters) == [
        "train_df",
        "quantile_alpha",
        "best_params",
        "is_tweedie",
    ]

    class RecordingModel:
        def __init__(self):
            self.fit_rows = None
            self.fit_kwargs = None

        def fit(self, features, target, **kwargs):
            self.fit_rows = len(features)
            self.fit_kwargs = kwargs

    recording_model = RecordingModel()
    monkeypatch.setattr(
        train_quantile.xgb,
        "XGBRegressor",
        lambda **kwargs: recording_model,
    )
    monkeypatch.setattr(train_quantile.joblib, "dump", lambda model, path: None)
    monkeypatch.setattr(train_quantile, "OUT_DIR", str(tmp_path))

    train_df = pd.DataFrame(
        {
            **{feature: [0.0, 1.0] for feature in train_quantile.FEATURES},
            "quantity": [1.0, 2.0],
        }
    )
    trained = train_quantile.train_quantile(
        train_df,
        quantile_alpha=0.50,
        best_params={"n_estimators": 1},
        is_tweedie=True,
    )

    assert trained is recording_model
    assert recording_model.fit_rows == len(train_df)
    assert recording_model.fit_kwargs == {}


def test_deployment_refit_frame_uses_complete_active_year():
    from s2_forecasting import train_quantile

    train = pd.DataFrame(
        {"date": ["2025-06-24", "2026-01-31"], "quantity": [1.0, 2.0]}
    )
    validation = pd.DataFrame(
        {"date": ["2026-02-01", "2026-03-31"], "quantity": [3.0, 4.0]}
    )
    test = pd.DataFrame(
        {"date": ["2026-04-01", "2026-06-23"], "quantity": [5.0, 6.0]}
    )

    refit = train_quantile.build_deployment_refit_frame(
        train, validation, test
    )

    assert len(refit) == 6
    assert refit["date"].tolist() == [
        "2025-06-24",
        "2026-01-31",
        "2026-02-01",
        "2026-03-31",
        "2026-04-01",
        "2026-06-23",
    ]
    assert refit["date"].min() == "2025-06-24"
    assert refit["date"].max() == "2026-06-23"


def test_deployment_metadata_distinguishes_evaluation_from_refit(tmp_path):
    from s2_forecasting import train_quantile

    train = pd.DataFrame({"date": ["2025-06-24", "2026-01-31"]})
    validation = pd.DataFrame({"date": ["2026-02-01", "2026-03-31"]})
    test = pd.DataFrame({"date": ["2026-04-01", "2026-06-23"]})
    refit = train_quantile.build_deployment_refit_frame(
        train, validation, test
    )

    metadata = train_quantile.write_deployment_metadata(
        train,
        validation,
        test,
        refit,
        output_dir=tmp_path,
    )

    assert metadata["evaluation"]["test"]["period"] == (
        "2026-04-01 to 2026-06-23"
    )
    assert metadata["deployment_refit"]["period"] == (
        "2025-06-24 to 2026-06-23"
    )
    assert metadata["deployment_refit"]["held_out_metric_claim"] is False
    assert (tmp_path / "deployment_metadata.json").exists()


def test_calibration_and_deployment_share_crossing_row_q50(monkeypatch, tmp_path):
    from api import module2_forecast
    from s2_forecasting import train_quantile

    class FixedModel:
        def __init__(self, predictions):
            self.predictions = np.asarray(predictions, dtype=float)

        def predict(self, features):
            assert len(features) == len(self.predictions)
            return self.predictions.copy()

        def get_params(self):
            return {}

    frame = pd.DataFrame(
        {
            **{feature: [0.0, 0.0] for feature in train_quantile.FEATURES},
            "date": ["2026-01-01", "2026-01-02"],
            "product_id": [0, 0],
            "quantity": [10.0, 5.0],
        }
    )
    raw_q10 = np.asarray([10.0, 2.0])
    raw_q50 = np.asarray([20.0, 5.0])
    raw_q90 = np.asarray([12.0, 8.0])
    models = {
        0.10: FixedModel(raw_q10),
        0.50: FixedModel(raw_q50),
        0.90: FixedModel(raw_q90),
    }
    deployed = module2_forecast.enforce_quantile_monotonicity(
        raw_q10,
        raw_q50,
        raw_q90,
    )
    monkeypatch.setattr(train_quantile, "OUT_DIR", str(tmp_path))

    calibration = train_quantile.conformal_calibrate(frame, models)
    metrics, predictions, _ = train_quantile.evaluate(frame, models, calibration)

    assert (
        train_quantile.enforce_quantile_monotonicity
        is module2_forecast.enforce_quantile_monotonicity
    )
    assert calibration["per_product"]["0"] == 1.6
    assert calibration["prediction_postprocessing"] == {
        "algorithm": "enforce_quantile_monotonicity",
        "applied_before_residual_scoring": True,
        "raw_quantile_crossing_count": 1,
        "q50_changed_count": 1,
    }
    np.testing.assert_array_equal(predictions["q50"], deployed["q50"])
    assert metrics["monitoring_diagnostics"] == {
        "raw_quantile_crossing_count": 1,
        "raw_quantile_crossing_rate_pct": 50.0,
        "q50_changed_count": 1,
    }


def test_xgboost_script_is_explicit_baseline_experiment():
    from s2_forecasting import train_xgboost

    assert train_xgboost.EXPERIMENT_ROLE == "deterministic_xgboost_baseline"
    assert train_xgboost.EXPERIMENT_NAME == "B1 deterministic XGBoost"
    assert train_xgboost.NAIVE_BASELINE_NAME == "B0 lag_7_avg"
    assert train_xgboost.FEATURES == train_xgboost.BASELINE_FEATURES
    assert "lag_7_avg" in train_xgboost.BASELINE_DESCRIPTION


def test_scale_conformal_splits_validation_by_complete_dates():
    from s2_forecasting.evaluate_scale_conformal import (
        split_validation_chronologically,
    )

    validation = pd.DataFrame(
        {
            "date": [
                "2026-01-03",
                "2026-01-01",
                "2026-01-02",
                "2026-01-04",
                "2026-01-03",
                "2026-01-01",
                "2026-01-02",
                "2026-01-04",
            ],
            "product_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "quantity": [3, 1, 2, 4, 6, 4, 5, 7],
        }
    )

    calibration, selection = split_validation_chronologically(
        validation,
        selection_fraction=0.25,
    )

    calibration_dates = pd.to_datetime(calibration["date"])
    selection_dates = pd.to_datetime(selection["date"])
    assert calibration_dates.max() < selection_dates.min()
    assert set(calibration_dates).isdisjoint(set(selection_dates))
    assert sorted(selection["date"].unique()) == ["2026-01-04"]


def test_scale_conformal_selects_on_validation_before_single_test_evaluation(
    monkeypatch,
    tmp_path,
):
    from s2_forecasting import evaluate_scale_conformal
    from s2_forecasting.feature_contract import FORECAST_FEATURES

    def make_frame(dates):
        rows = []
        for index, date_value in enumerate(dates):
            row = {feature: 0.0 for feature in FORECAST_FEATURES}
            row.update(
                {
                    "date": date_value,
                    "product_id": index % 2,
                    "quantity": float(index + 1),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    validation = make_frame(pd.date_range("2026-01-01", periods=10).strftime("%Y-%m-%d"))
    test = make_frame(pd.date_range("2026-02-01", periods=4).strftime("%Y-%m-%d"))
    validation.to_csv(tmp_path / "xgboost_val.csv", index=False)
    test.to_csv(tmp_path / "xgboost_test.csv", index=False)

    class ConstantModel:
        def predict(self, features):
            return np.full(len(features), 3.0)

    observed_candidates = []

    def select_from_validation(candidates, target_coverage=80.0):
        observed_candidates.extend(candidates)
        assert all(
            item["evaluation_split"] == "validation_selection"
            for item in candidates
        )
        return candidates[0]

    monkeypatch.setattr(
        evaluate_scale_conformal,
        "load_q50_model",
        lambda output_dir: ConstantModel(),
    )
    monkeypatch.setattr(
        evaluate_scale_conformal,
        "select_best_scale_candidate",
        select_from_validation,
    )

    metrics = evaluate_scale_conformal.run_experiment(
        data_dir=tmp_path,
        output_dir=tmp_path / "outputs",
    )

    assert observed_candidates
    assert metrics["selection"]["evaluation_split"] == "validation_selection"
    assert metrics["final_evaluation"]["evaluation_split"] == "test_once"
    assert metrics["row_count"] == len(test)


def test_classifier_is_auxiliary_risk_experiment_on_forecast_contract():
    from s2_forecasting import train_classifier
    from s2_forecasting.feature_contract import FORECAST_FEATURES

    assert train_classifier.EXPERIMENT_ROLE == "auxiliary_high_demand_risk_classifier"
    assert train_classifier.VALIDATION_DESIGN == "date-aware rolling-origin cross-validation"
    assert train_classifier.FEATURES == FORECAST_FEATURES
    assert train_classifier.CV_SPLITS == 3


def test_weekly_model_is_supplementary_event_aware_experiment():
    from s2_forecasting import train_weekly
    from s2_forecasting.feature_contract import WEEKLY_RESERVED_SCENARIO_FEATURES

    assert train_weekly.EXPERIMENT_ROLE == "supplementary_weekly_event_aware_forecast"
    assert train_weekly.MODEL_SCOPE == "weekly_supplementary"
    assert all(
        feature in train_weekly.WEEKLY_FEATURES
        for feature in WEEKLY_RESERVED_SCENARIO_FEATURES
    )


def test_experiment_summary_builds_b0_b1_proposed_closure(tmp_path):
    from s2_forecasting.evaluate_experiments import build_experiment_summary

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (broken_dir / "test_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        build_experiment_summary(broken_dir)

    output_dir = tmp_path
    run_timestamp = "2026-07-19T01:00:00+00:00"
    test_period = "2025-07-01 to 2026-06-23"
    row_count = 4
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_timestamp": run_timestamp,
                "row_count": row_count,
                "test_period": test_period,
                "experiment_name": "B1 deterministic XGBoost",
                "naive_baseline_name": "B0 lag_7_avg",
                "baseline_MAE": 2.3,
                "baseline_RMSE": 7.1,
                "baseline_MAPE": 36.0,
                "xgboost_test_MAE": 2.7,
                "xgboost_test_RMSE": 8.8,
                "xgboost_test_MAPE": 35.8,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "run_timestamp": run_timestamp,
                "row_count": row_count,
                "test_period": test_period,
                "model": "XGBoost Tweedie Q50 + Quantile Q10/Q90",
                "features": 27,
                "interval_method": "Core pre-runtime-bias conformal 80%",
                "interval_scope": "core_pre_runtime_bias_conformal",
                "runtime_transform_evaluated": False,
                "overall": {
                    "WAPE": 30.0,
                    "MAE": 1.9,
                    "RMSE": 6.7,
                    "conformal_coverage_80": 78.5,
                    "conformal_avg_width": 4.8,
                },
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "scale_conformal_metrics.json").write_text(
        json.dumps(
            {
                "id": "CandidateB_ScaleConformal",
                "run_timestamp": run_timestamp,
                "row_count": row_count,
                "test_period": test_period,
                "overall": {
                    "WAPE": 29.0,
                    "MAE": 1.8,
                    "RMSE": 6.5,
                    "coverage_80": 81.0,
                    "avg_width": 4.6,
                    "avg_relative_width": 0.42,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": ["2025-07-01", "2025-07-01", "2026-06-23", "2026-06-23"],
            "product_id": [0, 1, 0, 1],
            "actual": [1, 2, 3, 4],
        }
    ).to_csv(output_dir / "scale_conformal_predictions.csv", index=False)
    (output_dir / "log_quantile_metrics.json").write_text(
        json.dumps(
            {
                "id": "CandidateA_LogQuantile",
                "run_timestamp": "2026-07-13T01:00:00+00:00",
                "row_count": row_count,
                "test_period": "2025-07-01 to 2026-06-29",
                "overall": {"WAPE": 37.4},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "classifier_metrics.json").write_text("{}", encoding="utf-8")
    (output_dir / "weekly_metrics.json").write_text("{}", encoding="utf-8")

    summary = build_experiment_summary(output_dir)

    assert summary["module"] == "S2 demand forecasting"
    assert summary["feature_contract"]["total_features"] == 27
    assert [item["id"] for item in summary["experiments"]] == ["B0", "B1", "Proposed"]
    assert summary["experiments"][0]["name"] == "B0 lag_7_avg"
    assert summary["experiments"][1]["name"] == "B1 deterministic XGBoost"
    assert summary["experiments"][2]["name"] == "Tweedie Q50 with conformal interval"
    assert summary["experiments"][2]["metrics"]["coverage_80"] == 78.5
    assert [item["id"] for item in summary["candidate_experiments"]] == [
        "CandidateB_ScaleConformal",
    ]
    assert summary["supplementary_experiments"] == []
    assert {
        item["id"] for item in summary["excluded_artifacts"]
    } >= {
        "CandidateA_LogQuantile",
        "AuxClassifier",
        "WeeklyEventAware",
    }

    included = summary["experiments"] + summary["candidate_experiments"]
    for experiment in included:
        provenance = experiment["provenance"]
        assert provenance["run_timestamp"] == run_timestamp
        assert provenance["row_count"] == row_count
        assert provenance["test_period"] == test_period
        assert provenance["source_artifacts"]
        assert all(
            len(artifact["sha256"]) == 64
            for artifact in provenance["source_artifacts"]
        )

    metrics_hash = hashlib.sha256(
        (output_dir / "metrics.json").read_bytes()
    ).hexdigest()
    assert summary["experiments"][0]["provenance"]["source_artifacts"] == [
        {"path": "metrics.json", "sha256": metrics_hash}
    ]
    assert summary["paper_claim"]["primary_model"] == "Proposed"
    assert summary["paper_claim"]["supplementary_chain"] == []
