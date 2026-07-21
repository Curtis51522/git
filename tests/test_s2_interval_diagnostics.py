import pandas as pd


def test_interval_diagnostics_summarize_overall_product_and_deciles():
    from s2_forecasting.diagnose_intervals import (
        build_interval_frame,
        summarize_by_demand_decile,
        summarize_by_product,
        summarize_overall,
    )

    test_df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
            "product_id": [1, 1, 2, 2],
            "quantity": [10, 14, 40, 60],
        }
    )
    predictions = {
        "q10": [8, 10, 30, 48],
        "q50": [11, 13, 45, 55],
        "q90": [16, 18, 70, 92],
    }
    calibration = {"per_product": {"1": 3.0, "2": 18.0}, "global": 5.0}

    frame = build_interval_frame(test_df, predictions, calibration)
    product_names = {1: "croissant", 2: "macaron"}
    overall = summarize_overall(
        frame,
        high_uncertainty_threshold=0.60,
        product_names=product_names,
    )
    by_product = summarize_by_product(frame, product_names)
    by_decile = summarize_by_demand_decile(frame, n_bins=2)

    assert list(frame["raw_interval_width"]) == [8, 8, 40, 44]
    assert round(float(frame.loc[0, "raw_relative_width"]), 4) == round(8 / 11, 4)
    assert round(float(frame.loc[2, "raw_upper_risk"]), 4) == round((70 - 45) / 45, 4)
    assert round(float(frame.loc[2, "raw_lower_risk"]), 4) == round((45 - 30) / 45, 4)

    assert overall["sample_count"] == 4
    assert overall["raw_interval_coverage_pct"] == 100.0
    assert overall["conformal_interval_coverage_pct"] == 100.0
    assert overall["conformal_interval_scope"] == "core_pre_runtime_bias_conformal"
    assert overall["runtime_transform_evaluated"] is False
    assert overall["high_uncertainty_product_count"] == 2
    assert overall["widest_relative_product"] == "macaron"

    assert list(by_product["product_name"]) == ["croissant", "macaron"]
    assert by_product.loc[0, "sample_count"] == 2
    assert by_product.loc[1, "product_name"] == "macaron"
    assert by_product.loc[1, "avg_q50"] == 50.0
    assert by_product.loc[1, "raw_interval_coverage_pct"] == 100.0

    assert list(by_decile["demand_decile"]) == [1, 2]
    assert by_decile.loc[0, "sample_count"] == 2
    assert by_decile.loc[1, "avg_q50"] == 50.0


def test_interval_diagnostics_detect_quantile_crossing():
    from s2_forecasting.diagnose_intervals import build_interval_frame, summarize_overall

    test_df = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "product_id": [1],
            "quantity": [10],
        }
    )
    predictions = {"q10": [12], "q50": [11], "q90": [9]}
    frame = build_interval_frame(test_df, predictions, {"per_product": {}, "global": 2.0})
    overall = summarize_overall(frame)

    assert bool(frame.loc[0, "had_quantile_crossing"]) is True
    assert bool(frame.loc[0, "has_quantile_crossing"]) is False
    assert frame.loc[0, "q10"] == 9
    assert frame.loc[0, "q50"] == 11
    assert frame.loc[0, "q90"] == 12
    assert overall["pre_correction_crossing_count"] == 1
    assert overall["quantile_crossing_count"] == 0


def test_quantile_post_processing_orders_predictions():
    from s2_forecasting.quantile_utils import enforce_quantile_monotonicity

    corrected = enforce_quantile_monotonicity(
        q10=[10, 7, 4],
        q50=[8, 8, 9],
        q90=[6, 12, 5],
    )

    assert list(corrected["q10"]) == [6, 7, 4]
    assert list(corrected["q50"]) == [8, 8, 5]
    assert list(corrected["q90"]) == [10, 12, 9]
    assert list(corrected["crossing_mask"]) == [True, False, True]
    assert corrected["crossing_count"] == 2
