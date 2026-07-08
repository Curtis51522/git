from pathlib import Path

import pandas as pd


def test_preprocessing_accepts_precomputed_beverage_features(monkeypatch, tmp_path):
    from s2_forecasting import preprocess
    from s2_forecasting.feature_contract import FORECAST_FEATURES

    rows = [
        {
            "date": "2023-01-01",
            "product_name": "baguette",
            "ticket_id": 1,
            "daily_tickets": 120,
            "quantity": 3,
            "is_day1": 0,
            "is_top3": 1,
            "discount_pct": 0,
            "is_member_day": 0,
            "is_new_product": 0,
            "is_competitor": 0,
            "is_rainy": 0,
            "large_ratio": 0.5,
            "cold_ratio": 0.25,
            "sweetness_avg": 1.5,
            "ice_avg": 1.0,
            "temp_hot_ratio": 0.6,
        },
        {
            "date": "2025-01-01",
            "product_name": "baguette",
            "ticket_id": 2,
            "daily_tickets": 80,
            "quantity": 4,
            "is_day1": 0,
            "is_top3": 1,
            "discount_pct": 0,
            "is_member_day": 0,
            "is_new_product": 0,
            "is_competitor": 0,
            "is_rainy": 0,
            "large_ratio": 0.4,
            "cold_ratio": 0.2,
            "sweetness_avg": 1.0,
            "ice_avg": 0.5,
            "temp_hot_ratio": 0.5,
        },
        {
            "date": "2025-07-01",
            "product_name": "baguette",
            "ticket_id": 3,
            "daily_tickets": 90,
            "quantity": 5,
            "is_day1": 0,
            "is_top3": 1,
            "discount_pct": 0,
            "is_member_day": 0,
            "is_new_product": 0,
            "is_competitor": 0,
            "is_rainy": 0,
            "large_ratio": 0.3,
            "cold_ratio": 0.1,
            "sweetness_avg": 0.8,
            "ice_avg": 0.25,
            "temp_hot_ratio": 0.4,
        },
    ]
    raw_csv = tmp_path / "bakery_sales_raw.csv"
    pd.DataFrame(rows).to_csv(raw_csv, index=False)

    monkeypatch.setattr(preprocess, "RAW_CSV", str(raw_csv))
    monkeypatch.setattr(preprocess, "DATA_DIR", str(Path(tmp_path)))

    train_df, val_df, test_df = preprocess.run_preprocessing(verbose=False)

    expected_columns = ["date"] + FORECAST_FEATURES + ["quantity"]
    assert list(train_df.columns) == expected_columns
    assert list(val_df.columns) == expected_columns
    assert list(test_df.columns) == expected_columns
    assert "is_new_product" not in train_df.columns
    assert "is_competitor" not in train_df.columns
    assert train_df["large_ratio"].max() == 0.5
    assert train_df["daily_tickets"].max() == 120
    assert val_df["daily_tickets"].max() == 80
    assert test_df["daily_tickets"].max() == 90
