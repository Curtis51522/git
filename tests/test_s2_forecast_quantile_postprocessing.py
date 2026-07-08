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
        lambda forecast_date, product="": {name: 0 for name in module2_forecast.FORECAST_FEATURE_ORDER},
    )

    def fail_db():
        raise RuntimeError("No database needed for this test")

    monkeypatch.setattr(module2_forecast, "get_db", fail_db)

    result = module2_forecast._do_forecast("baguette", 1, use_cache=False, start_date="2026-07-07")

    forecast = result["forecasts"][0]
    assert forecast["predicted_demand"] == 12
    assert forecast["lower_bound"] == 10
    assert forecast["upper_bound"] == 14
