import asyncio
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import HTTPException

from api import module1_yolo
from api.module4_frontend import bff


class HistoryCursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql = ""
        self.params = ()
        self.closed = False

    def execute(self, sql, params=None):
        self.sql = " ".join(sql.split())
        self.params = tuple(params or ())

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class HistoryDb:
    def __init__(self, rows):
        self.cursor_instance = HistoryCursor(rows)
        self.dictionary_requested = False
        self.closed = False

    def cursor(self, dictionary=False):
        self.dictionary_requested = dictionary
        return self.cursor_instance

    def close(self):
        self.closed = True


class HistoricalRevenueCursor:
    def __init__(self):
        self.rows = []

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        if "SUM(oi.line_total) as total_revenue" in normalized:
            self.rows = [
                ("macaron", "2026-07-18", "2026-07-18", 223.0, 215.0),
            ]
        elif "AS revenue_adjustment" in normalized:
            self.rows = [
                ("2026-07-18", "2026-07-18", 9.0, -4.75),
            ]
        elif "transaction_type = 'return'" in normalized:
            self.rows = []
        elif "freshness_status = 'Expired'" in normalized:
            self.rows = []
        else:
            raise AssertionError(f"Unexpected SQL: {normalized}")

    def fetchall(self):
        return self.rows


class HistoricalRevenueDb:
    def __init__(self):
        self.cursor_instance = HistoricalRevenueCursor()

    def cursor(self):
        return self.cursor_instance


class InventorySnapshotCursor:
    def __init__(self, snapshot_rows=None):
        self.rows = []
        self.executed = []
        self.snapshot_rows = snapshot_rows or [
            ("bagel", datetime(2026, 7, 1, 6, 15), 2),
        ]

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self.executed.append((normalized, tuple(params or ())))
        if "LEFT JOIN inventory_transactions it" in normalized:
            self.rows = self.snapshot_rows
        elif "SUM(bi.quantity_remaining)" in normalized:
            self.rows = [("bagel", "Day-1", 0)]
        elif "FROM order_items oi JOIN orders o" in normalized:
            self.rows = [("bagel", "Fresh", 9)]
        elif "FROM raw_materials rm" in normalized:
            self.rows = []
        elif "FROM batch_inventory bi JOIN products p" in normalized:
            self.rows = [(0,)]
        elif "stock_quantity * unit_price" in normalized:
            self.rows = [(0,)]
        elif "SELECT material_name, stock_quantity, unit" in normalized:
            self.rows = []
        elif "SELECT material_name, SUM(quantity) as total" in normalized:
            self.rows = []
        else:
            raise AssertionError(f"Unexpected SQL: {normalized}")

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0]


class InventorySnapshotDb:
    def __init__(self, snapshot_rows=None):
        self.cursor_instance = InventorySnapshotCursor(snapshot_rows)

    def cursor(self):
        return self.cursor_instance


def test_inventory_dashboard_reconstructs_selected_date_from_batch_movements(
    monkeypatch,
):
    database = InventorySnapshotDb()
    monkeypatch.setattr(bff, "get_db", lambda: database)

    result = asyncio.run(bff.inventory_dashboard(date="2026-07-02"))

    assert result["bread_stock"] == [
        {
            "product_name": "bagel",
            "fresh_qty": 0,
            "day1_qty": 2,
            "total_qty": 2,
        }
    ]
    assert result["fresh_total"] == 0
    assert result["day1_total"] == 2
    assert result["snapshot_basis"] == "historical_close"
    assert result["snapshot_label"] == "Closing Bread Stock (2026-07-02)"
    snapshot_queries = [
        query for query in database.cursor_instance.executed
        if "LEFT JOIN inventory_transactions it" in query[0]
    ]
    assert snapshot_queries
    assert snapshot_queries[0][1] == (
        "2026-07-03 00:00:00",
        "2026-07-03 00:00:00",
    )


def test_inventory_dashboard_uses_current_time_for_today(monkeypatch):
    database = InventorySnapshotDb()
    monkeypatch.setattr(bff, "datetime", FixedDateTime)
    monkeypatch.setattr(bff, "get_db", lambda: database)

    result = asyncio.run(bff.inventory_dashboard(date="2026-07-15"))

    assert result["snapshot_basis"] == "current_live"
    assert result["snapshot_label"] == "Current Bread Stock"
    assert result["balance_time"] == "2026-07-15 12:30:00"
    snapshot_queries = [
        query
        for query in database.cursor_instance.executed
        if "LEFT JOIN inventory_transactions it" in query[0]
    ]
    assert snapshot_queries[0][1] == (
        "2026-07-15 12:30:00",
        "2026-07-15 12:30:00",
    )


def test_inventory_dashboard_excludes_stock_older_than_day1(monkeypatch):
    database = InventorySnapshotDb(
        [
            ("bagel", datetime(2026, 7, 2, 6, 15), 3),
            ("croissant", datetime(2026, 7, 1, 6, 15), 2),
            ("brownie", datetime(2026, 6, 30, 6, 15), 4),
        ]
    )
    monkeypatch.setattr(bff, "get_db", lambda: database)

    result = asyncio.run(bff.inventory_dashboard(date="2026-07-02"))

    assert result["fresh_total"] == 3
    assert result["day1_total"] == 2
    assert result["bread_stock"] == [
        {
            "product_name": "bagel",
            "fresh_qty": 3,
            "day1_qty": 0,
            "total_qty": 3,
        },
        {
            "product_name": "croissant",
            "fresh_qty": 0,
            "day1_qty": 2,
            "total_qty": 2,
        },
    ]
    assert result["overdue_total"] == 4
    assert result["overdue_stock"] == [
        {
            "product_name": "brownie",
            "overdue_qty": 4,
            "oldest_production_date": "2026-06-30",
        }
    ]


def test_finished_product_inflow_history_uses_selected_date(monkeypatch):
    rows = [
        {
            "id": 14323,
            "batch_id": "BATCH_20260714061500000000_apple_pie",
            "product_name": "apple_pie",
            "quantity_opening": 0,
            "quantity_baked": 11,
            "quantity_sold": 7,
            "quantity_discarded": 2,
            "quantity_outflow_total": 9,
            "quantity_closing": 2,
            "transaction_time": FixedDateTime(2026, 7, 14, 6, 15),
        }
    ]
    db = HistoryDb(rows)
    monkeypatch.setattr(module1_yolo, "datetime", FixedDateTime)
    monkeypatch.setattr(module1_yolo, "get_db", lambda: db)

    result = asyncio.run(
        module1_yolo.get_inflow_history(date="2026-07-14", limit=100)
    )

    assert result["status"] == "ok"
    assert result["date"] == "2026-07-14"
    assert result["remaining_label"] == "Left at Close"
    record = result["records"][0]
    assert record["quantity_baked"] == 11
    assert record["quantity_opening"] == 0
    assert record["quantity_sold"] == 7
    assert record["quantity_discarded"] == 2
    assert record["quantity_other_outflow"] == 0
    assert record["quantity_left"] == 2
    assert record["quantity_carried_to_day1"] == 2
    assert record["data_quality_issue"] is False
    assert record["transaction_time"] == "2026-07-14 06:15:00"
    assert "LEFT JOIN inventory_transactions it" in db.cursor_instance.sql
    assert "LEFT JOIN batch_inventory" not in db.cursor_instance.sql
    assert "p.category = 'bakery'" in db.cursor_instance.sql
    assert db.cursor_instance.params == (
        "2026-07-14 00:00:00",
        "2026-07-15 00:00:00",
        100,
    )
    assert db.dictionary_requested
    assert db.cursor_instance.closed
    assert db.closed


def test_finished_product_inflow_history_flags_negative_balance(monkeypatch):
    rows = [
        {
            "id": 1,
            "batch_id": "BATCH_TEST",
            "product_name": "bagel",
            "quantity_baked": 4,
            "quantity_sold": 3,
            "quantity_discarded": 2,
            "quantity_outflow_total": 6,
            "transaction_time": datetime(2026, 7, 14, 7, 0),
        }
    ]
    db = HistoryDb(rows)
    monkeypatch.setattr(module1_yolo, "get_db", lambda: db)

    result = asyncio.run(
        module1_yolo.get_inflow_history(date="2026-07-14", limit=100)
    )

    record = result["records"][0]
    assert record["quantity_other_outflow"] == 1
    assert record["quantity_left"] == 0
    assert record["data_quality_issue"] is True


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 7, 15, 12, 30)


def test_finished_product_inflow_history_uses_current_time_for_today(monkeypatch):
    db = HistoryDb([])
    monkeypatch.setattr(module1_yolo, "datetime", FixedDateTime)
    monkeypatch.setattr(module1_yolo, "get_db", lambda: db)

    result = asyncio.run(
        module1_yolo.get_inflow_history(date="2026-07-15", limit=100)
    )

    assert result["remaining_label"] == "Left Now"
    assert result["snapshot_basis"] == "current_live"
    assert db.cursor_instance.params == (
        "2026-07-15 00:00:00",
        "2026-07-15 12:30:00",
        100,
    )


def test_finished_product_inflow_history_rejects_invalid_date():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            module1_yolo.get_inflow_history(date="2026-7-14", limit=100)
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Date must use YYYY-MM-DD format"


def test_raw_material_restock_history_uses_selected_date(monkeypatch):
    rows = [
        {
            "id": 68,
            "material_name": "Lids",
            "quantity": 34,
            "unit": "pcs",
            "reference": "manual_restock",
            "created_at": datetime(2026, 7, 14, 14, 21, 27),
        }
    ]
    db = HistoryDb(rows)
    monkeypatch.setattr(bff, "get_db", lambda: db)

    result = asyncio.run(
        bff.inventory_restock_history(date="2026-07-14", limit=100)
    )

    assert result["status"] == "ok"
    assert result["date"] == "2026-07-14"
    assert result["records"][0]["created_at"] == "2026-07-14 14:21:27"
    assert "transaction_type = 'restock'" in db.cursor_instance.sql
    assert db.cursor_instance.params == (
        "2026-07-14 00:00:00",
        "2026-07-15 00:00:00",
        100,
    )
    assert db.dictionary_requested
    assert db.cursor_instance.closed
    assert db.closed


def test_raw_material_restock_history_reports_latest_date_when_selected_day_is_empty(monkeypatch):
    class EmptyRestockCursor(HistoryCursor):
        def __init__(self):
            super().__init__([])
            self.executed = []

        def execute(self, sql, params=None):
            super().execute(sql, params)
            self.executed.append((self.sql, self.params))

        def fetchone(self):
            return {"latest_date": "2026-07-14", "record_count": 6}

    db = HistoryDb([])
    db.cursor_instance = EmptyRestockCursor()
    monkeypatch.setattr(bff, "get_db", lambda: db)

    result = asyncio.run(
        bff.inventory_restock_history(date="2026-07-15", limit=100)
    )

    assert result["count"] == 0
    assert result["latest_record_date"] == "2026-07-14"
    assert result["latest_record_count"] == 6
    assert len(db.cursor_instance.executed) == 2
    assert "created_at < %s" in db.cursor_instance.executed[1][0]
    assert db.cursor_instance.executed[1][1] == ("2026-07-16 00:00:00",)


def test_dine_type_breakdown_counts_paid_orders():
    cursor = HistoryCursor(
        [
            ("dine_in", 29),
            ("takeaway", 23),
        ]
    )

    result = bff._get_dine_type_breakdown(cursor, "2026-07-14")

    assert result == {"Dine-in": 29, "Takeaway": 23}
    assert "state IN ('paid','completed')" in cursor.sql
    assert cursor.params == ("2026-07-14",)


def test_revenue_change_is_unavailable_without_a_valid_baseline():
    assert bff._revenue_change(120.0, 100.0) == 20.0
    assert bff._revenue_change(120.0, None) is None
    assert bff._revenue_change(120.0, 0.0) is None


def test_order_basket_metrics_explain_average_order_value_change():
    metrics = bff._order_basket_metrics(
        revenue=2637.5,
        orders=45,
        items=215,
        previous_revenue=4179.6,
        previous_orders=52,
        previous_items=344,
    )

    assert metrics == {
        "today_items": 215,
        "items_per_order": 4.78,
        "items_per_order_change": -27.8,
        "revenue_per_item": 12.27,
        "revenue_per_item_change": 1.0,
    }


def test_recent_revenue_baseline_excludes_selected_date_and_uses_completed_days():
    cursor = HistoryCursor(
        [
            ("2026-07-13", 49, 3970.8),
            ("2026-07-12", 52, 4179.6),
            ("2026-07-11", 47, 3744.0),
        ]
    )

    result = bff._get_recent_revenue_baseline(cursor, "2026-07-14", limit=7)

    assert result == {
        "day_count": 3,
        "start_date": "2026-07-11",
        "end_date": "2026-07-13",
        "avg_revenue": 3964.8,
        "avg_orders": 49.33,
        "avg_order_value": 80.37,
    }
    assert "o.order_date < %s" in cursor.sql
    assert "ORDER BY o.order_date DESC" in cursor.sql
    assert cursor.params == ("2026-07-14", 7)


def test_expired_product_breakdown_supports_promotion_decisions():
    cursor = HistoryCursor(
        [
            ("croissant", 8, 24.0, 32, 384.0, 240.0),
            ("baguette", 10, 20.0, 5, 60.0, 18.0),
        ]
    )

    result = bff._get_expired_product_breakdown(cursor, "2026-07-14")

    assert result == [
        {
            "name": "Croissant",
            "expired_qty": 8,
            "expired_cost": 24.0,
            "sold_qty": 32,
            "revenue": 384.0,
            "profit": 240.0,
            "margin_pct": 62.5,
            "sell_through_pct": 80.0,
            "loss_share_pct": 54.55,
        },
        {
            "name": "Baguette",
            "expired_qty": 10,
            "expired_cost": 20.0,
            "sold_qty": 5,
            "revenue": 60.0,
            "profit": 18.0,
            "margin_pct": 30.0,
            "sell_through_pct": 33.33,
            "loss_share_pct": 45.45,
        },
    ]
    assert "freshness_status = 'Expired'" in cursor.sql
    assert "reason = 'expired'" not in cursor.sql
    assert "state IN ('paid','completed')" in cursor.sql
    assert cursor.params == (
        "2026-07-14",
        "2026-07-14",
        "2026-07-14",
        "2026-07-14",
    )


def test_closing_loss_endpoint_returns_all_products_for_date_range(monkeypatch):
    rows = [
        ("cream_horn", 16, 73.6, 4, 100.0, 85.2),
        ("apple_pie", 10, 61.6, 5, 90.0, 60.0),
        ("chocopie", 20, 56.6, 7, 140.0, 98.0),
        ("baguette", 3, 7.5, 30, 360.0, 240.0),
        ("croissant", 2, 6.0, 40, 480.0, 300.0),
        ("donut", 1, 2.5, 25, 250.0, 150.0),
    ]
    db = HistoryDb(rows)
    monkeypatch.setattr(bff, "get_db", lambda: db)

    result = asyncio.run(
        bff.revenue_closing_loss(start="2026-07-08", end="2026-07-14")
    )

    assert result["status"] == "ok"
    assert result["data"]["start"] == "2026-07-08"
    assert result["data"]["end"] == "2026-07-14"
    assert result["data"]["total_expired_qty"] == 52
    assert result["data"]["total_expired_cost"] == 207.8
    assert result["data"]["product_count"] == 6
    assert len(result["data"]["products"]) == 6
    assert result["data"]["products"][0] == {
        "name": "Cream Horn",
        "expired_qty": 16,
        "expired_cost": 73.6,
        "sold_qty": 4,
        "revenue": 100.0,
        "profit": 85.2,
        "margin_pct": 85.2,
        "sell_through_pct": 20.0,
        "loss_share_pct": 35.42,
    }
    assert "LIMIT 5" not in db.cursor_instance.sql
    assert db.cursor_instance.params == (
        "2026-07-08",
        "2026-07-14",
        "2026-07-08",
        "2026-07-14",
    )
    assert db.cursor_instance.closed
    assert db.closed


def test_closing_loss_endpoint_rejects_reversed_date_range():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            bff.revenue_closing_loss(start="2026-07-14", end="2026-07-08")
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Start date must not be after end date"


def test_frontend_exposes_inventory_histories_and_order_type_chart():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )
    history_source = source[
        source.index("async function loadInventoryHistories") : source.index(
            "async function loadInvWastageRates"
        )
    ]

    assert 'id="inv-inflow-history"' in source
    assert 'id="inv-restock-history"' in source
    assert "loadInventoryHistories" in source
    assert "/s1/inflow/history?date=" in source
    assert "/s4/inventory/restock/history?date=" in source
    assert 'id="rev-dine-chart"' in source
    assert "m.dine_type" in source
    assert "chart-third" in source
    assert "function ratioLegendFormatter" in source
    assert "legend:{bottom:0" in source
    assert "No raw-material restock was recorded on" in source
    assert "Latest restock:" in source
    assert "Select that date to view details." in source
    assert "Baked Product Stock Records" in source
    assert 'id="inv-bread-chart-title"' in source
    assert "d.snapshot_basis" in source
    assert "No bread stock remains now" in source
    assert "No bread stock remained at closing" in source
    assert "d.overdue_total" in source
    assert "Expired Stock Pending Disposal" in source
    assert "Not included in sellable stock" in source
    assert "t(inflow.remaining_label)" in source
    assert "row.batch_id" in source
    assert "row.product_name" in source
    assert "capName(row.product_name||'')" in history_source
    assert "row.transaction_time" in source
    assert "row.quantity_baked" in source
    assert "row.quantity_opening" in source
    assert "row.quantity_sold" in source
    assert "row.quantity_discarded" in source
    assert "row.quantity_carried_to_day1" in source
    assert "row.quantity_left" in source
    assert "row.data_quality_issue" in source
    assert "t('Check data')" in source
    assert "colspan=\"8\"" in source
    assert "'Opening':'Opening'" in source
    assert "'Carried to Day-1':'Carried to Day-1'" in source
    assert "'Current Bread Stock':'Current Bread Stock'" in source
    assert "'Closing Bread Stock':'Closing Bread Stock'" in source
    assert "'Baked':'Baked'" in source
    assert "'Sold':'Sold'" in source
    assert "'Discarded':'Discarded'" in source
    assert "'Left Now':'Left Now'" in source
    assert "'Left at Close':'Left at Close'" in source
    assert "'Check data':'Check data'" in source
    assert "'Baked':'\\u70d8\\u5236'" in source
    assert "'Sold':'\\u5df2\\u552e'" in source
    assert "'Discarded':'\\u5df2\\u62a5\\u635f'" in source
    assert "'Left Now':'\\u5f53\\u524d\\u5269\\u4f59'" in source
    assert "'Left at Close':'\\u6253\\u70ca\\u65f6\\u5269\\u4f59'" in source
    assert "'Check data':'\\u68c0\\u67e5\\u6570\\u636e'" in source
    assert "row.quantity_remaining" not in history_source


def test_revenue_frontend_exposes_closing_loss_daily_and_range_views():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )

    assert 'id="rev-closing-loss-table"' in source
    assert 'id="rev-loss-day-btn"' in source
    assert 'id="rev-loss-range-btn"' in source
    assert 'id="rev-loss-range-controls"' in source
    assert "function setClosingLossMode(mode)" in source
    assert "function loadClosingLossData(selectedDate)" in source
    assert "/s4/revenue/closing-loss?" in source
    assert "m.expired_products" not in source
    assert "loadClosingLossData(date);" in source
    assert 'overflow-x:auto;max-height:420px' in source
    assert 'position:sticky;top:0;z-index:1' in source
    assert "t('Discarded Product Loss')" in source
    assert "t('Discarded')" in source
    assert "t('discarded units')" in source
    assert "t('Unsold Product Loss')" not in source
    assert "t('unsold units')" not in source


def test_revenue_kpis_show_na_when_previous_day_is_unavailable():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )

    assert "function revenueChangeLabel(value)" in source
    assert "value==null?t('N/A')" in source
    assert "revenueChangeLabel(m.revenue_change)" in source
    assert "revenueChangeLabel(m.profit_change)" in source
    assert "revenueChangeLabel(m.orders_change)" in source
    assert "revenueChangeLabel(m.avg_change)" in source


def test_historical_sales_separates_order_adjustments_from_products(monkeypatch):
    database = HistoricalRevenueDb()
    monkeypatch.setattr(bff, "get_db", lambda: database)

    result = asyncio.run(
        bff.revenue_historical(
            start="2026-07-18",
            end="2026-07-18",
            granularity="day",
            category="total",
        )
    )

    data = result["data"]
    assert [product["name"] for product in data["products"]] == ["Macaron"]
    assert data["order_adjustments"] == {
        "total_revenue": 9.0,
        "total_profit": -4.75,
        "periods": {
            "2026-07-18": {"revenue": 9.0, "profit": -4.75},
        },
    }


def test_historical_sales_frontend_renders_adjustments_outside_chart():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )

    assert 'id="rev-historical-adjustment"' in source
    assert "d.order_adjustments||{}" in source
    assert "adjustments.periods||{}" in source
    assert "Separate order charges/costs" in source
    assert "names.push(t(p.name));" in source


def test_hourly_chart_separates_closing_adjustment_from_trading_hours():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )

    assert "'Closing adjustment':'Closing adjustment'" in source
    assert (
        "'Unsold products discarded at closing':"
        "'Unsold products discarded at closing'"
    ) in source
    assert "Already deducted from today's profit" in source
    assert "'Closing inventory loss':'Closing inventory loss'" not in source
    assert "'Separate order charges/costs':'Separate order charges/costs'" in source
    assert 'id="rev-hourly-adjustment"' in source
    assert "var closingIndex=dd.hours.indexOf('Closing adjustment');" in source
    assert "chartHours.splice(closingIndex,1);" in source
    assert "data:chartHours.map(function(hour){return t(hour);})" in source
    assert "data:dd.hours.map(function(hour){return t(hour);})" not in source
    assert "names.push(t(p.name));" in source
