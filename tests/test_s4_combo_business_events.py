import asyncio
import sys
import types


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeCursor:
    def __init__(self, connection, rows):
        self.connection = connection
        self.rows = rows
        self.lastrowid = None
        self.rowcount = 0
        self.closed = False

    def execute(self, sql, params=None):
        self.sql = sql
        self.params = params
        if "INSERT INTO recommendation_events" in sql:
            self.connection.next_event_id += 1
            self.lastrowid = self.connection.next_event_id
            self.rowcount = 1
            self.connection.event_inserts.append(
                {"id": self.lastrowid, "params": params}
            )

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeDB:
    def __init__(self, autocommit):
        self.autocommit = autocommit
        self.calls = 0
        self.cursors = []
        self.event_inserts = []
        self.next_event_id = 700
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self, dictionary=False):
        self.calls += 1
        if self.calls == 1:
            rows = [("croissant",)]
        elif self.calls == 2:
            rows = [("latte", 18.0)]
        else:
            rows = []
        cursor = FakeCursor(self, rows)
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class FakeDBFactory:
    def __init__(self):
        self.connections = []

    def __call__(self, *, autocommit=True):
        connection = FakeDB(autocommit=autocommit)
        self.connections.append(connection)
        return connection


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
    db_factory = FakeDBFactory()
    monkeypatch.setattr(bff, "get_db", db_factory)
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

    transactional_connections = [
        connection
        for connection in db_factory.connections
        if connection.autocommit is False
    ]
    assert len(transactional_connections) == 1
    assert all(connection.closed for connection in db_factory.connections)
    connection = transactional_connections[0]
    assert connection.autocommit is False
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.closed is True
    assert all(cursor.closed for cursor in connection.cursors)
    assert len(connection.event_inserts) == len(result["recommendations"])
    assert [event["id"] for event in connection.event_inserts] == [
        recommendation["recommendation_event_id"]
        for recommendation in result["recommendations"]
    ]
