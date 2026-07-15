import asyncio

import numpy as np


class DummyQuantileModel:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.array([self.value])


def test_forecast_api_uses_post_processed_q50(monkeypatch):
    from api import module2_forecast

    models = {
        "q10": DummyQuantileModel(12),
        "q50": DummyQuantileModel(20),
        "q90": DummyQuantileModel(9),
    }

    monkeypatch.setattr(module2_forecast, "_get_unified_quantile", lambda q: models[q])
    monkeypatch.setattr(module2_forecast, "_get_product_id_map", lambda: {"baguette": 0})
    monkeypatch.setattr(module2_forecast, "_get_conformal_half", lambda product: 2.0)
    monkeypatch.setattr(
        module2_forecast,
        "build_forecast_features",
        lambda forecast_date, product="", **kwargs: {
            name: 0 for name in module2_forecast.FORECAST_FEATURE_ORDER
        },
    )

    def fail_db():
        raise RuntimeError("No database needed for this test")

    monkeypatch.setattr(module2_forecast, "get_db", fail_db)

    result = module2_forecast._do_forecast("baguette", 1, use_cache=False, start_date="2026-07-07")

    forecast = result["forecasts"][0]
    assert forecast["predicted_demand"] == 12
    assert forecast["lower_bound"] == 10
    assert forecast["upper_bound"] == 14
    assert forecast["interval_width"] == 4
    assert forecast["relative_width"] == 0.333
    assert forecast["uncertainty_level"] == "low"
    assert forecast["interval_method"] == "Conformal 80%"


def test_forecast_api_marks_wide_relative_intervals_as_high_uncertainty(monkeypatch):
    from api import module2_forecast

    models = {
        "q10": DummyQuantileModel(10),
        "q50": DummyQuantileModel(12),
        "q90": DummyQuantileModel(14),
    }

    monkeypatch.setattr(module2_forecast, "_get_unified_quantile", lambda q: models[q])
    monkeypatch.setattr(module2_forecast, "_get_product_id_map", lambda: {"melon_bread": 0})
    monkeypatch.setattr(module2_forecast, "_get_conformal_half", lambda product: 8.0)
    monkeypatch.setattr(
        module2_forecast,
        "build_forecast_features",
        lambda forecast_date, product="", **kwargs: {
            name: 0 for name in module2_forecast.FORECAST_FEATURE_ORDER
        },
    )
    monkeypatch.setattr(module2_forecast, "get_db", lambda: (_ for _ in ()).throw(RuntimeError("No database needed")))

    result = module2_forecast._do_forecast("melon_bread", 1, use_cache=False, start_date="2026-07-07")

    forecast = result["forecasts"][0]
    assert forecast["interval_width"] == 16
    assert forecast["relative_width"] == 1.333
    assert forecast["uncertainty_level"] == "high"


def test_multiday_forecast_rolls_q50_forward_without_future_actuals(monkeypatch):
    from api import module2_forecast

    models = {
        "q10": DummyQuantileModel(8),
        "q50": DummyQuantileModel(10),
        "q90": DummyQuantileModel(12),
    }
    captured_histories = []

    def fake_features(
        forecast_date,
        product="",
        sales_history=None,
        daily_tickets=None,
    ):
        captured_histories.append(dict(sales_history or {}))
        return {name: 0 for name in module2_forecast.FORECAST_FEATURE_ORDER}

    monkeypatch.setattr(module2_forecast, "_get_unified_quantile", lambda q: models[q])
    monkeypatch.setattr(module2_forecast, "_get_product_id_map", lambda: {"baguette": 0})
    monkeypatch.setattr(module2_forecast, "_get_conformal_half", lambda product: 2.0)
    monkeypatch.setattr(
        module2_forecast,
        "_get_product_daily_sales",
        lambda product: {
            "2026-07-14": 9.0,
            "2026-07-15": 99.0,
        },
    )
    monkeypatch.setattr(module2_forecast, "_get_daily_tickets", lambda date: 50.0)
    monkeypatch.setattr(
        module2_forecast,
        "_get_recent_category_bias_factors",
        lambda start_date, products, models: {"bakery": 1.2, "beverage": 1.0},
    )
    monkeypatch.setattr(module2_forecast, "build_forecast_features", fake_features)
    monkeypatch.setattr(
        module2_forecast,
        "get_db",
        lambda: (_ for _ in ()).throw(RuntimeError("No database needed")),
    )

    result = module2_forecast._do_forecast(
        "baguette",
        2,
        use_cache=False,
        start_date="2026-07-15",
    )

    assert len(result["forecasts"]) == 2
    assert captured_histories[0] == {"2026-07-14": 9.0}
    assert captured_histories[1] == {
        "2026-07-14": 9.0,
        "2026-07-15": 12.0,
    }


def test_recent_bias_factor_requires_evidence_and_limits_adjustment():
    from api import module2_forecast

    assert module2_forecast._compute_bias_factor(140, 100, 7) == 1.4
    assert module2_forecast._compute_bias_factor(300, 100, 7) == 1.5
    assert module2_forecast._compute_bias_factor(50, 100, 7) == 0.75
    assert module2_forecast._compute_bias_factor(140, 100, 2) == 1.0
    assert module2_forecast._compute_bias_factor(140, 0, 7) == 1.0


def test_forecast_interval_context_treats_one_unit_ranges_as_low_uncertainty():
    from api import module2_forecast

    context = module2_forecast._build_interval_context(
        prediction=1.0,
        lower_bound=1,
        upper_bound=2,
    )

    assert context["interval_width"] == 1
    assert context["relative_width"] == 1.0
    assert context["uncertainty_level"] == "low"


def test_forecast_interval_context_treats_small_low_volume_ranges_as_low_uncertainty():
    from api import module2_forecast

    context = module2_forecast._build_interval_context(
        prediction=2.0,
        lower_bound=1,
        upper_bound=3,
    )

    assert context["interval_width"] == 2
    assert context["relative_width"] == 1.0
    assert context["uncertainty_level"] == "low"


def test_refresh_forecast_passes_selected_start_date(monkeypatch):
    from api import module2_forecast

    calls = []

    def fake_forecast(product, days, use_cache, start_date=None):
        calls.append(
            {
                "product": product,
                "days": days,
                "use_cache": use_cache,
                "start_date": start_date,
            }
        )
        return {"status": "ok"}

    monkeypatch.setattr(module2_forecast, "_do_forecast", fake_forecast)

    result = asyncio.run(
        module2_forecast.refresh_forecast(
            product="baguette",
            days=3,
            date="2026-06-30",
        )
    )

    assert result == {"status": "ok"}
    assert calls == [
        {
            "product": "baguette",
            "days": 3,
            "use_cache": False,
            "start_date": "2026-06-30",
        }
    ]
