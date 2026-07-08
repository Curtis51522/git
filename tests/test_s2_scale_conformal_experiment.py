import json

import numpy as np
import pandas as pd


def test_scale_bins_keep_ordered_edges_and_min_width():
    from s2_forecasting.evaluate_scale_conformal import build_scale_calibration

    calibration = build_scale_calibration(
        q50=np.array([1, 2, 3, 10, 11, 12], dtype=float),
        actual=np.array([1, 3, 5, 10, 14, 18], dtype=float),
        n_bins=2,
        quantile=0.80,
        min_half_width=0.5,
    )

    assert calibration["n_bins"] == 2
    assert calibration["bins"][0]["lower_edge"] <= calibration["bins"][0]["upper_edge"]
    assert calibration["bins"][1]["lower_edge"] <= calibration["bins"][1]["upper_edge"]
    assert calibration["bins"][0]["half_width"] >= 0.5
    assert calibration["bins"][1]["half_width"] >= 0.5
    assert calibration["global_half_width"] >= 0.5


def test_apply_scale_calibration_uses_matching_bin_widths():
    from s2_forecasting.evaluate_scale_conformal import apply_scale_calibration

    calibration = {
        "global_half_width": 3.0,
        "bins": [
            {"lower_edge": 0.0, "upper_edge": 5.0, "half_width": 1.0},
            {"lower_edge": 5.0, "upper_edge": 20.0, "half_width": 4.0},
        ],
    }

    widths = apply_scale_calibration(np.array([2.0, 5.0, 12.0, 30.0]), calibration)

    assert list(widths) == [1.0, 4.0, 4.0, 3.0]


def test_scale_conformal_prediction_frame_reports_relative_width_and_coverage():
    from s2_forecasting.evaluate_scale_conformal import build_scale_prediction_frame

    test_df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "product_id": [1, 1],
            "quantity": [10.0, 20.0],
        }
    )
    calibration = {
        "global_half_width": 3.0,
        "bins": [
            {"lower_edge": 0.0, "upper_edge": 15.0, "half_width": 2.0},
            {"lower_edge": 15.0, "upper_edge": 30.0, "half_width": 6.0},
        ],
    }

    frame = build_scale_prediction_frame(test_df, q50=np.array([10.0, 18.0]), calibration=calibration)

    assert list(frame["half_width"]) == [2.0, 6.0]
    assert list(frame["lower"]) == [8.0, 12.0]
    assert list(frame["upper"]) == [12.0, 24.0]
    assert list(frame["covered"]) == [True, True]
    assert round(float(frame.loc[0, "relative_width"]), 4) == 0.4


def test_scale_conformal_metrics_report_candidate_scope():
    from s2_forecasting.evaluate_scale_conformal import summarize_scale_metrics

    frame = pd.DataFrame(
        {
            "actual": [10.0, 20.0],
            "q50": [10.0, 18.0],
            "lower": [8.0, 12.0],
            "upper": [12.0, 24.0],
            "covered": [True, True],
            "half_width": [2.0, 6.0],
            "relative_width": [0.4, 0.6667],
            "absolute_error": [0.0, 2.0],
            "squared_error": [0.0, 4.0],
            "signed_error": [0.0, -2.0],
        }
    )

    metrics = summarize_scale_metrics(frame, test_period="2026-01-01 to 2026-01-02", calibration={"bins": []})

    assert metrics["id"] == "CandidateB_ScaleConformal"
    assert metrics["scope"] == "candidate_not_deployed"
    assert metrics["overall"]["WAPE"] == 6.7
    assert metrics["overall"]["coverage_80"] == 100.0
    assert metrics["overall"]["avg_relative_width"] == 0.5333


def test_select_best_scale_candidate_prefers_coverage_then_relative_width():
    from s2_forecasting.evaluate_scale_conformal import select_best_scale_candidate

    candidates = [
        {"n_bins": 2, "overall": {"coverage_80": 70.0, "avg_relative_width": 0.5}},
        {"n_bins": 3, "overall": {"coverage_80": 80.0, "avg_relative_width": 1.0}},
        {"n_bins": 5, "overall": {"coverage_80": 82.0, "avg_relative_width": 0.9}},
    ]

    best = select_best_scale_candidate(candidates, target_coverage=80.0)

    assert best["n_bins"] == 5


def test_experiment_summary_includes_scale_conformal_candidate_when_available(tmp_path):
    from s2_forecasting.evaluate_experiments import build_experiment_summary

    (tmp_path / "metrics.json").write_text(
        json.dumps({"baseline_MAE": 2.0, "xgboost_test_MAE": 1.8}),
        encoding="utf-8",
    )
    (tmp_path / "test_metrics.json").write_text(
        json.dumps({"overall": {"WAPE": 30.0, "MAE": 1.9, "conformal_coverage_80": 78.9}}),
        encoding="utf-8",
    )
    (tmp_path / "scale_conformal_metrics.json").write_text(
        json.dumps(
            {
                "id": "CandidateB_ScaleConformal",
                "name": "Demand-scale conformal candidate",
                "role": "candidate_uncertainty_calibration",
                "scope": "candidate_not_deployed",
                "description": "Calibrates conformal width by predicted demand scale.",
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

    summary = build_experiment_summary(tmp_path)

    assert summary["candidate_experiments"][0]["id"] == "CandidateB_ScaleConformal"
    assert summary["candidate_experiments"][0]["metrics"]["avg_relative_width"] == 0.75
    assert summary["paper_claim"]["primary_model"] == "Proposed"
