import sys
import types

from s5_agent.agents.inventory import InventoryAgent


class FakeCursor:
    def __init__(self, db):
        self.db = db

    def execute(self, sql, params=None):
        self.db.queries.append((sql, params))

    def fetchall(self):
        return self.db.result_sets.pop(0)

    def close(self):
        self.db.closed_cursors += 1


class FakeDb:
    def __init__(self):
        self.queries = []
        self.closed_cursors = 0
        self.result_sets = [
            [("croissant", "Fresh", 8), ("croissant", "Day-1", 3)],
            [("croissant", 5.9)],
        ]

    def cursor(self):
        return FakeCursor(self)


def test_inventory_freshness_uses_current_db_client(monkeypatch):
    fake_db = FakeDb()
    fake_mysql_client = types.SimpleNamespace(get_db=lambda: fake_db)
    monkeypatch.setitem(sys.modules, "db.mysql_client", fake_mysql_client)

    result = InventoryAgent()._query_db_freshness({"croissant"})

    assert result == {
        "croissant": {
            "Fresh": 8,
            "Day-1": 3,
            "qty": 11,
            "selling_price": 5.9,
        }
    }
    assert "batch_inventory" in fake_db.queries[0][0]
    assert fake_db.queries[0][1] == ["croissant"]
