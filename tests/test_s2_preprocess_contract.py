from pathlib import Path
import inspect
from datetime import date

import pandas as pd
import pytest


def build_raw_fixture(tmp_path, end_date, start_date="2025-06-24"):
    dates = pd.date_range(start_date, end_date, freq="D")
    raw_path = tmp_path / "bakery_sales_raw.csv"
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "product_name": "baguette",
            "ticket_id": range(1, len(dates) + 1),
            "quantity": 1,
        }
    ).to_csv(raw_path, index=False)
    return raw_path


def test_s2_test_set_stops_before_operation(monkeypatch, tmp_path):
    from s2_forecasting import preprocess

    raw_path = build_raw_fixture(tmp_path, end_date="2026-06-24")
    monkeypatch.setattr(preprocess, "RAW_CSV", str(raw_path))
    _, _, test = preprocess.run_preprocessing(verbose=False)

    assert test["date"].max() == "2026-06-23"
    assert not (test["date"] >= "2026-06-24").any()


def test_s2_train_set_starts_at_approved_history(monkeypatch, tmp_path):
    from s2_forecasting import preprocess

    raw_path = build_raw_fixture(
        tmp_path,
        end_date="2026-06-23",
        start_date="2021-01-01",
    )
    monkeypatch.setattr(preprocess, "RAW_CSV", str(raw_path))
    train, _, _ = preprocess.run_preprocessing(verbose=False)

    assert train["date"].min() == "2025-06-24"
    assert train["date"].max() == "2026-01-31"


def test_s2_active_year_split_is_chronological(monkeypatch, tmp_path):
    from s2_forecasting import preprocess

    raw_path = build_raw_fixture(tmp_path, end_date="2026-06-23")
    monkeypatch.setattr(preprocess, "RAW_CSV", str(raw_path))
    train, validation, test = preprocess.run_preprocessing(verbose=False)

    assert (train["date"].min(), train["date"].max()) == (
        "2025-06-24",
        "2026-01-31",
    )
    assert (validation["date"].min(), validation["date"].max()) == (
        "2026-02-01",
        "2026-03-31",
    )
    assert (test["date"].min(), test["date"].max()) == (
        "2026-04-01",
        "2026-06-23",
    )
    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()


def test_product_categories_accept_legacy_numeric_values():
    from s2_forecasting import preprocess

    raw = pd.DataFrame(
        [
            {"product_name": "americano", "category": 1},
            {"product_name": "macaron", "category": 0},
            {"product_name": "latte", "category": 1.0},
            {"product_name": "baguette", "category": 0.0},
        ]
    )

    categories = preprocess.resolve_product_categories(
        raw,
        ["americano", "macaron", "latte", "baguette"],
    )

    assert categories == {
        "americano": 1,
        "macaron": 0,
        "latte": 1,
        "baguette": 0,
    }


def test_product_categories_accept_task3_string_values():
    from s2_forecasting import preprocess

    raw = pd.DataFrame(
        [
            {"product_name": "croissant", "category": "bakery"},
            {"product_name": "latte", "category": "beverage"},
        ]
    )

    categories = preprocess.resolve_product_categories(raw, ["croissant", "latte"])

    assert categories == {"croissant": 1, "latte": 0}


def test_product_categories_reject_unknown_values():
    from s2_forecasting import preprocess

    raw = pd.DataFrame([{"product_name": "baguette", "category": "other"}])

    with pytest.raises(ValueError, match="Invalid category for baguette: other"):
        preprocess.resolve_product_categories(raw, ["baguette"])


def test_product_categories_reject_fractional_numeric_values():
    from s2_forecasting import preprocess

    raw = pd.DataFrame([{"product_name": "baguette", "category": 0.5}])

    with pytest.raises(ValueError, match=r"Invalid category for baguette: 0\.5"):
        preprocess.resolve_product_categories(raw, ["baguette"])


def test_product_categories_reject_nan_values():
    from s2_forecasting import preprocess

    raw = pd.DataFrame([{"product_name": "baguette", "category": float("nan")}])

    with pytest.raises(ValueError, match="Invalid category for baguette: nan"):
        preprocess.resolve_product_categories(raw, ["baguette"])


def test_weekly_preprocessing_uses_real_product_categories(monkeypatch, tmp_path):
    from s2_forecasting import preprocess_weekly

    rows = []
    for product_name, category in [("americano", 1), ("macaron", 0)]:
        rows.append(
            {
                "date": "2023-01-02",
                "product_name": product_name,
                "category": category,
                "ticket_id": product_name,
                "quantity": 3,
                "is_day1": 0,
                "is_top3": 0,
                "discount_pct": 0,
                "is_member_day": 0,
                "is_new_product": 0,
                "is_competitor": 0,
                "is_rainy": 0,
            }
        )
    raw_csv = tmp_path / "bakery_sales_raw.csv"
    pd.DataFrame(rows).to_csv(raw_csv, index=False)
    monkeypatch.setattr(preprocess_weekly, "RAW_CSV", str(raw_csv))

    train_df, _, _ = preprocess_weekly.run_weekly_preprocessing(verbose=False)

    assert train_df.set_index("product_id")["category"].to_dict() == {0: 1, 1: 0}


def test_weekly_metrics_keep_preprocessed_category():
    from s2_forecasting import train_weekly

    source = inspect.getsource(train_weekly.per_product_wape)

    assert '(test_df["product_id"] >= 30)' not in source


def test_weekly_holiday_feature_excludes_ordinary_weekends():
    from s2_forecasting import preprocess_weekly

    assert preprocess_weekly.is_named_holiday(date(2026, 7, 11)) is False
    assert preprocess_weekly.is_named_holiday(date(2026, 10, 1)) is True


def test_preprocessing_accepts_precomputed_beverage_features(monkeypatch, tmp_path):
    from s2_forecasting import preprocess
    from s2_forecasting.feature_contract import FORECAST_FEATURES

    rows = [
        {
            "date": "2025-06-24",
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
            "date": "2026-02-01",
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
            "date": "2026-04-01",
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
