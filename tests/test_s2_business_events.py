import asyncio
from datetime import date


def test_business_event_reserved_summary_maps_event_types():
    from api import module2_forecast

    events = [
        {"event_type": "new_product_launch", "active": True},
        {"event_type": "competitor_activity", "active": True},
    ]

    summary = module2_forecast._build_reserved_scenario_summary(events)

    assert summary["is_new_product"]["active"] is True
    assert summary["is_new_product"]["value"] == 1
    assert summary["is_new_product"]["model_input"] is False
    assert summary["is_competitor"]["active"] is True
    assert summary["is_competitor"]["value"] == 1
    assert summary["is_competitor"]["model_input"] is False


def test_business_event_reserved_summary_ignores_unknown_event_type():
    from api import module2_forecast

    summary = module2_forecast._build_reserved_scenario_summary([
        {"event_type": "unknown_event", "active": True}
    ])

    assert summary["is_new_product"]["active"] is False
    assert summary["is_competitor"]["active"] is False


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []
        self.lastrowid = 10

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows


class FakeDB:
    def __init__(self, rows=None):
        self.cursor_obj = FakeCursor(rows)

    def cursor(self, dictionary=False):
        return self.cursor_obj


def test_list_business_events_filters_by_selected_date(monkeypatch):
    from api import module2_forecast

    rows = [
        {
            "id": 10,
            "event_type": "competitor_activity",
            "start_date": date(2026, 7, 1),
            "end_date": date(2026, 7, 14),
            "products": '["croissant", "baguette"]',
            "discount_pct": 10.0,
            "note": "Nearby store opening promotion",
            "active": 1,
        }
    ]
    fake_db = FakeDB(rows)
    monkeypatch.setattr(module2_forecast, "get_db", lambda: fake_db)

    result = module2_forecast._list_business_events("2026-07-07")

    assert result[0]["id"] == 10
    assert result[0]["label"] == "Competitor activity"
    assert result[0]["products"] == ["croissant", "baguette"]
    assert result[0]["active"] is True
    sql, params = fake_db.cursor_obj.executed[0]
    assert "start_date <= %s" in sql
    assert "end_date >= %s" in sql
    assert params == ("2026-07-07", "2026-07-07")


def test_validate_business_event_payload_rejects_invalid_discount():
    from api import module2_forecast

    payload = {
        "event_type": "competitor_activity",
        "start_date": "2026-07-01",
        "end_date": "2026-07-14",
        "products": ["croissant"],
        "discount_pct": 150,
    }

    try:
        module2_forecast._validate_business_event_payload(payload)
    except ValueError as exc:
        assert "discount_pct" in str(exc)
    else:
        raise AssertionError("Expected invalid discount to raise ValueError")


def test_get_business_events_endpoint_returns_reserved_summary(monkeypatch):
    from api import module2_forecast

    events = [
        {
            "id": 10,
            "event_type": "competitor_activity",
            "label": "Competitor activity",
            "start_date": "2026-07-01",
            "end_date": "2026-07-14",
            "products": ["croissant"],
            "discount_pct": 10.0,
            "note": "Nearby store opening promotion",
            "active": True,
        }
    ]
    monkeypatch.setattr(module2_forecast, "_list_business_events", lambda date="": events)

    result = asyncio.run(module2_forecast.get_business_events(date="2026-07-07"))

    assert result["status"] == "ok"
    assert result["date"] == "2026-07-07"
    assert result["events"] == events
    assert result["reserved_scenario_features"]["is_competitor"]["active"] is True
    assert result["reserved_scenario_features"]["is_competitor"]["model_input"] is False
