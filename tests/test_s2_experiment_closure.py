import json

import pandas as pd


def test_quantile_training_uses_date_aware_cv_without_future_leakage():
    from s2_forecasting.train_quantile import build_date_aware_cv

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

    splits = list(build_date_aware_cv(df, n_splits=2))

    assert len(splits) == 2
    for train_idx, val_idx in splits:
        train_dates = pd.to_datetime(df.iloc[train_idx]["date"])
        val_dates = pd.to_datetime(df.iloc[val_idx]["date"])
        assert train_dates.max() < val_dates.min()
        assert len(set(train_idx).intersection(set(val_idx))) == 0


def test_xgboost_script_is_explicit_baseline_experiment():
    from s2_forecasting import train_xgboost

    assert train_xgboost.EXPERIMENT_ROLE == "deterministic_xgboost_baseline"
    assert train_xgboost.BASELINE_NAME == "B1 deterministic XGBoost"
    assert train_xgboost.FEATURES == train_xgboost.BASELINE_FEATURES
    assert "lag_7_avg" in train_xgboost.BASELINE_DESCRIPTION


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

    output_dir = tmp_path
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
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
                "model": "XGBoost Tweedie Q50 + Quantile Q10/Q90",
                "features": 27,
                "interval_method": "Conformal 80%",
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
    (output_dir / "classifier_metrics.json").write_text(
        json.dumps(
            {
                "experiment_role": "auxiliary_high_demand_risk_classifier",
                "validation_design": "date-aware rolling-origin cross-validation",
                "features": 27,
                "test_Accuracy": 0.75,
                "test_F1": 0.52,
                "test_ROC_AUC": 0.84,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "weekly_metrics.json").write_text(
        json.dumps(
            {
                "experiment_role": "supplementary_weekly_event_aware_forecast",
                "model_scope": "weekly_supplementary",
                "baseline": {"WAPE": 40.0, "MAE": 4.5, "RMSE": 7.0},
                "Q50": {"WAPE": 35.0, "MAE": 4.0, "RMSE": 6.5},
                "coverage_Q50_Q90": 0.45,
                "interval_width": 2.2,
            }
        ),
        encoding="utf-8",
    )

    summary = build_experiment_summary(output_dir)

    assert summary["module"] == "S2 demand forecasting"
    assert summary["feature_contract"]["total_features"] == 27
    assert [item["id"] for item in summary["experiments"]] == ["B0", "B1", "Proposed"]
    assert summary["experiments"][0]["name"] == "Historical moving-average baseline"
    assert summary["experiments"][1]["name"] == "Deterministic XGBoost baseline"
    assert summary["experiments"][2]["name"] == "Tweedie Q50 with conformal interval"
    assert summary["experiments"][2]["metrics"]["coverage_80"] == 78.5
    assert [item["id"] for item in summary["supplementary_experiments"]] == [
        "AuxClassifier",
        "WeeklyEventAware",
    ]
    assert summary["supplementary_experiments"][0]["metrics"]["ROC_AUC"] == 0.84
    assert summary["supplementary_experiments"][1]["metrics"]["Q50_WAPE"] == 35.0
    assert summary["paper_claim"]["primary_model"] == "Proposed"
    assert summary["paper_claim"]["supplementary_chain"] == [
        "AuxClassifier",
        "WeeklyEventAware",
    ]
