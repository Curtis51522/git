import asyncio
from pathlib import Path

from api.module4_frontend import bff


class OrderHistoryCursor:
    def __init__(self):
        self.sql = ""
        self.params = ()

    def execute(self, sql, params=None):
        self.sql = " ".join(sql.split())
        self.params = tuple(params or ())

    def fetchall(self):
        return [
            (
                222683,
                "RCP-20260715184500-000",
                "18:45:00",
                42.8,
                "dine_in",
                "paid",
                3,
            )
        ]

    def close(self):
        pass


class OrderHistoryConnection:
    def __init__(self):
        self.cursor_instance = OrderHistoryCursor()

    def cursor(self):
        return self.cursor_instance


def test_orders_today_returns_database_order_id(monkeypatch):
    connection = OrderHistoryConnection()
    monkeypatch.setattr(bff, "get_db", lambda: connection)

    result = asyncio.run(bff.orders_today("2026-07-15"))

    assert "SELECT id, ticket_id" in connection.cursor_instance.sql
    assert connection.cursor_instance.params == ("2026-07-15",)
    assert result["orders"][0]["order_id"] == 222683
    assert result["orders"][0]["ticket_id"] == "RCP-20260715184500-000"


def test_pos_order_history_displays_database_order_id():
    html = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )

    assert "var displayOrderId='#'+o.order_id;" in html
    assert "o.ticket_id.slice(-6)" not in html
