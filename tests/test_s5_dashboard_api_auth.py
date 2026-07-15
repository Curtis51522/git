import asyncio
import json

import s5_agent.agents.inventory as inventory_module
from s5_agent.agents.forecast_accuracy import ForecastAccuracyAgent
from s5_agent.agents.inventory import InventoryAgent
from s5_agent.agents.promo import PromoAgent
from s5_agent.core import dashboard_api


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_dashboard_api_forwards_authorization_with_standard_library(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse({"data": {"today_revenue": 2984.0}})

    monkeypatch.setattr(dashboard_api, "urlopen", fake_urlopen)

    result = dashboard_api.fetch_dashboard_json(
        "http://127.0.0.1:8002/s4/revenue/daily?date=2026-06-30",
        {"_authorization": "Bearer manager-token"},
    )

    assert result == {"data": {"today_revenue": 2984.0}}
    assert captured == {
        "url": "http://127.0.0.1:8002/s4/revenue/daily?date=2026-06-30",
        "authorization": "Bearer manager-token",
        "timeout": 10,
    }


def test_inventory_agent_uses_dashboard_auth(monkeypatch):
    captured = {}

    def fake_fetch(url, params, timeout=10):
        captured["url"] = url
        captured["authorization"] = params.get("_authorization")
        captured["timeout"] = timeout
        return {"inventory": []}

    monkeypatch.setattr(
        inventory_module,
        "fetch_dashboard_json",
        fake_fetch,
        raising=False,
    )
    monkeypatch.setattr(
        inventory_module,
        "S4_DASHBOARD_URL",
        "http://127.0.0.1:9/s4/inventory/dashboard",
    )

    result = asyncio.run(
        InventoryAgent().fetch(
            {
                "date": "2026-07-15",
                "module": "inventory",
                "product": "all",
                "_authorization": "Bearer manager-token",
            }
        )
    )

    assert result == {"inventory": []}
    assert captured == {
        "url": "http://127.0.0.1:9/s4/inventory/dashboard?date=2026-07-15",
        "authorization": "Bearer manager-token",
        "timeout": 10,
    }


def test_promo_agent_accepts_full_graph_params_and_uses_dashboard_auth(monkeypatch):
    captured = {}

    def fake_fetch(url, params, timeout=10):
        captured["url"] = url
        captured["authorization"] = params.get("_authorization")
        return {"data": {"today_discount": 30.0, "today_revenue": 1000.0}}

    monkeypatch.setattr("s5_agent.agents.promo.fetch_dashboard_json", fake_fetch)

    result = asyncio.run(
        PromoAgent("PromotionSignalAgent").fetch(
            {
                "date": "2026-06-30",
                "module": "promotion_mix",
                "product": "all",
                "_authorization": "Bearer manager-token",
            }
        )
    )

    assert result["success"] is True
    assert result["data"]["total_discount"] == 30.0
    assert result["data"]["discount_rate"] == 0.03
    assert captured["authorization"] == "Bearer manager-token"


def test_forecast_accuracy_agent_uses_dashboard_auth(monkeypatch):
    captured = {}

    def fake_fetch(url, params, timeout=10):
        captured["url"] = url
        captured["authorization"] = params.get("_authorization")
        return {
            "metrics": {
                "overall": {
                    "WAPE": 29.9,
                    "conformal_coverage_80": 78.9,
                    "conformal_avg_width": 5.0,
                }
            }
        }

    monkeypatch.setattr(
        "s5_agent.agents.forecast_accuracy.fetch_dashboard_json",
        fake_fetch,
    )

    result = asyncio.run(
        ForecastAccuracyAgent("ForecastAccuracyAgent").fetch(
            {"_authorization": "Bearer manager-token"}
        )
    )

    assert result["data"]["overall"]["WAPE"] == 29.9
    assert captured == {
        "url": "http://127.0.0.1:8002/s2/accuracy",
        "authorization": "Bearer manager-token",
    }
