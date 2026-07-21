import json

import numpy as np
import pandas as pd


def test_log_target_transform_round_trips_non_negative_values():
    from s2_forecasting.train_quantile_log import inverse_log_target, transform_log_target

    values = np.array([0.0, 1.0, 4.0, 20.0])
    restored = inverse_log_target(transform_log_target(values))

    assert np.allclose(restored, values)


def test_log_target_transform_rejects_negative_values():
    from s2_forecasting.train_quantile_log import transform_log_target

    try:
        transform_log_target(np.array([1.0, -1.0]))
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative target values to fail")


def test_log_quantile_prediction_summary_enforces_monotonic_outputs():
    from s2_forecasting.train_quantile_log import build_prediction_frame

    test_df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "product_id": [1, 1],
            "quantity": [10, 20],
        }
    )
    log_predictions = {
        "q10": np.log1p([12, 6]),
        "q50": np.log1p([11, 8]),
        "q90": np.log1p([9, 14]),
    }

    frame = build_prediction_frame(test_df, log_predictions)

    assert list(frame["q10"].round(6)) == [9.0, 6.0]
    assert list(frame["q50"].round(6)) == [11.0, 8.0]
    assert list(frame["q90"].round(6)) == [12.0, 14.0]
    assert list(frame["had_quantile_crossing"]) == [True, False]
    assert int(frame["has_quantile_crossing"].sum()) == 0


def test_log_quantile_metrics_report_candidate_scope():
    from s2_forecasting.train_quantile_log import summarize_candidate_metrics

    frame = pd.DataFrame(
        {
            "actual": [10.0, 20.0],
            "q10": [8.0, 15.0],
            "q50": [10.0, 18.0],
            "q90": [12.0, 25.0],
            "had_quantile_crossing": [False, True],
            "has_quantile_crossing": [False, False],
            "product_id": [1, 1],
        }
    )
    metrics = summarize_candidate_metrics(frame, test_period="2026-01-01 to 2026-01-02")

    assert metrics["id"] == "CandidateA_LogQuantile"
    assert metrics["scope"] == "candidate_not_deployed"
    assert metrics["overall"]["WAPE"] == 6.7
    assert metrics["overall"]["raw_Q10Q90_coverage"] == 100.0
    assert metrics["overall"]["pre_correction_crossing_count"] == 1
    assert metrics["overall"]["quantile_crossing_count"] == 0


def test_experiment_summary_includes_log_quantile_candidate_when_available(tmp_path):
    from s2_forecasting.evaluate_experiments import build_experiment_summary

    provenance = {
        "run_timestamp": "2026-01-03T00:00:00+00:00",
        "row_count": 2,
        "test_period": "2026-01-01 to 2026-01-02",
    }
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                **provenance,
                "baseline_MAE": 2.0,
                "xgboost_test_MAE": 1.8,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "test_metrics.json").write_text(
        json.dumps(
            {
                **provenance,
                "overall": {
                    "WAPE": 30.0,
                    "MAE": 1.9,
                    "conformal_coverage_80": 78.9,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "log_quantile_metrics.json").write_text(
        json.dumps(
            {
                **provenance,
                "id": "CandidateA_LogQuantile",
                "name": "Log-scale quantile candidate",
                "role": "candidate_probabilistic_forecast",
                "scope": "candidate_not_deployed",
                "description": "Trains quantile models on log1p quantity.",
                "validation_design": "chronological holdout test set",
                "overall": {
                    "WAPE": 29.0,
                    "MAE": 1.8,
                    "RMSE": 6.2,
                    "raw_Q10Q90_coverage": 80.0,
                    "avg_raw_relative_width": 0.7,
                    "quantile_crossing_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_experiment_summary(tmp_path)

    assert [item["id"] for item in summary["experiments"]] == ["B0", "B1", "Proposed"]
    assert summary["candidate_experiments"][0]["id"] == "CandidateA_LogQuantile"
    assert summary["candidate_experiments"][0]["scope"] == "candidate_not_deployed"
    assert summary["paper_claim"]["primary_model"] == "Proposed"
