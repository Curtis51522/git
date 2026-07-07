import asyncio
import sys
import types


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params=None):
        self.sql = sql
        self.params = params

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class FakeDB:
    def __init__(self):
        self.calls = 0

    def cursor(self, dictionary=False):
        self.calls += 1
        if self.calls == 1:
            return FakeCursor([("croissant",)])
        return FakeCursor([("latte", 18.0)])


def test_combo_returns_business_event_context_without_scoring_boost(monkeypatch):
    from api.module4_frontend import bff

    def fake_q(db, table):
        class Builder:
            def select(self, columns="*"):
                return self

            def gt(self, column, value):
                return self

            def neq(self, column, value):
                return self

            def execute(self):
                return FakeResponse([
                    {
                        "product_name": "croissant",
                        "quantity": 8,
                        "freshness_status": "Fresh",
                    }
                ])

        return Builder()

    fake_freshness = types.SimpleNamespace(
        get_discount_rate=lambda freshness: 0.0,
        update_all_freshness=lambda: None,
    )
    fake_pairing = types.SimpleNamespace(
        get_pairing_matrix=lambda: {"croissant": {"latte": 0.9}}
    )
    monkeypatch.setattr(bff, "get_db", lambda: FakeDB())
    monkeypatch.setattr(bff, "q", fake_q)
    monkeypatch.setitem(sys.modules, "api.freshness_service", fake_freshness)
    monkeypatch.setitem(sys.modules, "api.module4_frontend.pairing_llm", fake_pairing)

    result = asyncio.run(bff.get_combo({
        "items": [{"product_name": "latte", "quantity": 1, "freshness": "Fresh"}],
        "business_events": [
            {
                "id": 1,
                "event_type": "competitor_activity",
                "label": "Competitor activity",
                "products": ["croissant"],
                "discount_pct": 10.0,
                "active": True,
            }
        ],
    }))

    assert result["status"] == "ok"
    assert result["recommendations"]
    rec = result["recommendations"][0]
    assert rec["product_name"] == "croissant"
    assert rec["business_event_context"]["event_type"] == "competitor_activity"
    assert rec["business_event_context"]["discount_pct"] == 10.0
    assert "business_event_boost" not in rec
    assert "business_event_score" not in rec["scoring_breakdown"]
