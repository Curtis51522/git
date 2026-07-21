import json

import numpy as np
import pandas as pd


def test_relative_calibration_uses_normalized_residuals():
    from s2_forecasting.evaluate_relative_conformal import build_relative_calibration

    calibration = build_relative_calibration(
        q50=np.array([0.2, 2.0, 10.0, 20.0]),
        actual=np.array([0.5, 3.0, 12.0, 26.0]),
        quantile=0.80,
        min_relative_half_width=0.10,
    )

    assert calibration["method"] == "relative conformal calibration"
    assert calibration["relative_half_width"] >= 0.10
    assert calibration["denominator_floor"] == 1.0


def test_apply_relative_calibration_scales_width_by_prediction_size():
    from s2_forecasting.evaluate_relative_conformal import apply_relative_calibration

    calibration = {
        "relative_half_width": 0.25,
        "denominator_floor": 1.0,
        "min_absolute_half_width": 0.5,
    }

    widths = apply_relative_calibration(np.array([0.5, 4.0, 20.0]), calibration)

    assert list(widths) == [0.5, 1.0, 5.0]


def test_relative_prediction_frame_reports_intervals_and_relative_width():
    from s2_forecasting.evaluate_relative_conformal import build_relative_prediction_frame

    test_df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "product_id": [1, 1],
            "quantity": [10.0, 20.0],
        }
    )
    calibration = {
        "relative_half_width": 0.20,
        "denominator_floor": 1.0,
        "min_absolute_half_width": 0.5,
    }

    frame = build_relative_prediction_frame(test_df, q50=np.array([10.0, 18.0]), calibration=calibration)

    assert list(frame["half_width"]) == [2.0, 3.6]
    assert list(frame["lower"]) == [8.0, 14.4]
    assert list(frame["upper"]) == [12.0, 21.6]
    assert list(frame["covered"]) == [True, True]
    assert round(float(frame.loc[0, "relative_width"]), 4) == 0.4


def test_relative_metrics_report_candidate_scope():
    from s2_forecasting.evaluate_relative_conformal import summarize_relative_metrics

    frame = pd.DataFrame(
        {
            "actual": [10.0, 20.0],
            "q50": [10.0, 18.0],
            "lower": [8.0, 14.4],
            "upper": [12.0, 21.6],
            "covered": [True, True],
            "half_width": [2.0, 3.6],
            "relative_width": [0.4, 0.4],
            "absolute_error": [0.0, 2.0],
            "squared_error": [0.0, 4.0],
            "signed_error": [0.0, -2.0],
        }
    )

    metrics = summarize_relative_metrics(
        frame,
        test_period="2026-01-01 to 2026-01-02",
        calibration={"relative_half_width": 0.2},
    )

    assert metrics["id"] == "CandidateC_RelativeConformal"
    assert metrics["scope"] == "candidate_not_deployed"
    assert metrics["overall"]["WAPE"] == 6.7
    assert metrics["overall"]["coverage_80"] == 100.0
    assert metrics["overall"]["avg_relative_width"] == 0.4


def test_experiment_summary_includes_relative_conformal_candidate_when_available(tmp_path):
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
    (tmp_path / "relative_conformal_metrics.json").write_text(
        json.dumps(
            {
                **provenance,
                "id": "CandidateC_RelativeConformal",
                "name": "Relative conformal candidate",
                "role": "candidate_uncertainty_calibration",
                "scope": "candidate_not_deployed",
                "description": "Calibrates interval width using normalized residuals.",
                "validation_design": "validation residual calibration with chronological holdout test set",
                "overall": {
                    "WAPE": 29.9,
                    "MAE": 1.9,
                    "RMSE": 6.5,
                    "coverage_80": 80.0,
                    "avg_width": 4.8,
                    "avg_relative_width": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"date": ["2026-01-01", "2026-01-02"]}).to_csv(
        tmp_path / "relative_conformal_predictions.csv",
        index=False,
    )

    summary = build_experiment_summary(tmp_path)

    assert summary["candidate_experiments"][0]["id"] == "CandidateC_RelativeConformal"
    assert summary["candidate_experiments"][0]["metrics"]["avg_relative_width"] == 0.75
    assert summary["paper_claim"]["primary_model"] == "Proposed"
