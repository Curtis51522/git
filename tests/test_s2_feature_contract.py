import asyncio
from datetime import datetime

import numpy as np


def test_forecast_feature_contract_is_canonical():
    from config.settings import FORECAST_FEATURE_COLS
    from s2_forecasting.feature_contract import (
        FORECAST_FEATURES,
        FEATURE_GROUPS,
        FEATURE_METADATA,
        RESERVED_SCENARIO_FEATURES,
    )
    from s2_forecasting import train_quantile
    from api import module2_forecast

    assert len(FORECAST_FEATURES) == 27
    assert FORECAST_FEATURE_COLS == FORECAST_FEATURES
    assert train_quantile.FEATURES == FORECAST_FEATURES
    assert module2_forecast.FORECAST_FEATURE_ORDER == FORECAST_FEATURES
    assert set(FEATURE_METADATA) == set(FORECAST_FEATURES)
    assert "weather" in FEATURE_GROUPS
    assert "beverage_behavior_proxy" in FEATURE_GROUPS
    assert set(RESERVED_SCENARIO_FEATURES) == {"is_new_product", "is_competitor"}
    assert not set(RESERVED_SCENARIO_FEATURES).intersection(FORECAST_FEATURES)


def test_feature_importance_uses_deployed_q50_model(monkeypatch):
    from s2_forecasting.feature_contract import FORECAST_FEATURES
    from api import module2_forecast

    class DummyModel:
        feature_importances_ = np.linspace(0.01, 0.27, len(FORECAST_FEATURES))

    monkeypatch.setattr(module2_forecast, "_get_unified_quantile", lambda q: DummyModel())

    result = asyncio.run(module2_forecast.get_feature_importance())

    assert result["status"] == "ok"
    assert result["model"] == "quantile_model_q50"
    assert result["total_features"] == len(FORECAST_FEATURES)
    assert {item["feature"] for item in result["ranked"]} == set(FORECAST_FEATURES)
    assert "lagged_history" in result["grouped"]
    assert "traffic_proxy" in result["grouped"]


def test_today_features_returns_full_forecast_contract(monkeypatch):
    from s2_forecasting.feature_contract import FORECAST_FEATURES
    from api import module2_forecast

    sample = {name: index for index, name in enumerate(FORECAST_FEATURES)}
    monkeypatch.setattr(
        module2_forecast,
        "build_forecast_features",
        lambda forecast_date, product="": dict(sample),
    )
    monkeypatch.setattr(
        module2_forecast,
        "_list_business_events",
        lambda date="": [
            {
                "id": 1,
                "event_type": "competitor_activity",
                "label": "Competitor activity",
                "start_date": "2026-07-01",
                "end_date": "2026-07-14",
                "products": ["croissant"],
                "discount_pct": 10.0,
                "note": "Nearby store opening promotion",
                "active": True,
            }
        ],
    )

    result = asyncio.run(module2_forecast.get_today_features(date="2026-07-07"))

    assert result["status"] == "ok"
    assert result["date"] == "2026-07-07"
    assert list(result["features"].keys()) == FORECAST_FEATURES
    assert result["feature_contract"]["total_features"] == len(FORECAST_FEATURES)
    assert result["feature_contract"]["source"] == "s2_forecasting.feature_contract"
    assert set(result["reserved_scenario_features"]) == {"is_new_product", "is_competitor"}
    assert "is_new_product" not in result["features"]
    assert "is_competitor" not in result["features"]
    assert "is_new_product" not in result["interpretations"]
    assert "is_competitor" not in result["interpretations"]
    assert result["business_events"][0]["event_type"] == "competitor_activity"
    reserved = result["reserved_scenario_features"]
    assert reserved["is_competitor"]["active"] is True
    assert reserved["is_competitor"]["value"] == 1
    assert reserved["is_competitor"]["model_input"] is False
    assert reserved["is_new_product"]["active"] is False
    assert reserved["is_competitor"]["events"][0]["event_type"] == "competitor_activity"


def test_build_forecast_features_covers_contract(monkeypatch):
    from s2_forecasting.feature_contract import FORECAST_FEATURES
    from api import module2_forecast

    monkeypatch.setattr(module2_forecast, "_init_frozen_meta", lambda: None)
    monkeypatch.setattr(module2_forecast, "_frozen_meta", {
        "last_lag": {0: (2.0, 3.0, 4.0, 0.5, 0.7, -1.0)},
        "last_daily_tickets": 42.0,
        "top3_products": [0],
        "rainy_dates": set(),
        "holiday_dates": [],
    })
    monkeypatch.setattr(module2_forecast, "_get_product_id_map", lambda: {"baguette": 0})
    monkeypatch.setattr(module2_forecast, "_get_lag", lambda product, date, days_back: 0.0)
    monkeypatch.setattr(module2_forecast, "_get_rolling_avg", lambda product, date, window: 0.0)
    monkeypatch.setattr(module2_forecast, "_get_daily_tickets", lambda date: 0.0)
    monkeypatch.setattr(module2_forecast, "_get_is_day1", lambda product: 0)
    monkeypatch.setattr(module2_forecast, "_get_is_holiday", lambda date: 0)
    monkeypatch.setattr(module2_forecast, "_get_weather", lambda date: (26.0, 8.0, 0, 1))

    features = module2_forecast.build_forecast_features(datetime(2026, 7, 7), "baguette")

    assert list(features.keys()) == FORECAST_FEATURES
    assert set(features) == set(FORECAST_FEATURES)
