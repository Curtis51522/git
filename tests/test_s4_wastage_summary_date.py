import asyncio

from api.module4_frontend import bff


class FakeCursor:
    def __init__(self):
        self.executed_sql = ""
        self.executed_params = None

    def execute(self, sql, params=None):
        self.executed_sql = sql
        self.executed_params = params

    def fetchall(self):
        return [
            ("Bread Flour", 1.25, 0.02, 0.016, "2026-07-02"),
        ]


class FakeDb:
    def __init__(self):
        self.cursor_instance = FakeCursor()

    def cursor(self):
        return self.cursor_instance


def test_wastage_summary_uses_selected_date(monkeypatch):
    fake_db = FakeDb()
    monkeypatch.setattr(bff, "get_db", lambda: fake_db)

    result = asyncio.run(bff.wastage_summary(date="2026-07-02"))

    assert result["status"] == "ok"
    assert result["date"] == "2026-07-02"
    assert fake_db.cursor_instance.executed_params == ("2026-07-02",)
    assert "check_date <= %s" in fake_db.cursor_instance.executed_sql
    assert "MAX(mw.id)" in fake_db.cursor_instance.executed_sql
    assert result["summary"][0]["check_date"] == "2026-07-02"
