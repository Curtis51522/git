from datetime import datetime
from pathlib import Path

from api.module4_frontend import bff


REPO_ROOT = Path(__file__).resolve().parents[1]
BFF_SOURCE = (REPO_ROOT / "api" / "module4_frontend" / "bff.py").read_text(encoding="utf-8")
SCHEDULER_SOURCE = (REPO_ROOT / "s3_scheduling" / "scheduler.py").read_text(
    encoding="utf-8"
)


def test_checkout_uses_raw_material_units_for_recipe_transactions():
    assert "JOIN raw_materials rm ON rm.material_name = pr.material_name" in BFF_SOURCE
    assert "rm.unit" in BFF_SOURCE
    assert "(mat_name, 'outflow', actual_used_qty, 'kg', receipt_id)" not in BFF_SOURCE
    assert "SELECT material_name, stock_quantity, unit" in BFF_SOURCE
    assert 'material_info[material_name]["unit"]' in BFF_SOURCE
    assert "INSERT INTO material_transactions" in BFF_SOURCE


def test_checkout_does_not_apply_wastage_to_piece_based_materials():
    assert 'if unit != "pcs" and (category or "") != "packaging":' in BFF_SOURCE
    assert 'required *= Decimal("1") + products[product_name]["wastage_pct"]' in BFF_SOURCE


def test_inventory_stock_queries_filter_untracked_materials():
    assert BFF_SOURCE.count("track_inventory = 1") >= 7
    assert '.select("material_name, stock_quantity, unit, track_inventory")' in SCHEDULER_SOURCE
    assert 'if not bool(r.get("track_inventory", True))' in SCHEDULER_SOURCE


def test_manual_restock_rejects_untracked_materials():
    assert "SELECT stock_quantity, unit, track_inventory FROM raw_materials" in BFF_SOURCE
    assert "Material is not stock-tracked" in BFF_SOURCE


def test_material_transaction_history_remains_available():
    assert "FROM material_transactions" in BFF_SOURCE
    assert "transaction_type = 'outflow'" in BFF_SOURCE


class WastageCursor:
    def __init__(self, last_check):
        self.last_check = last_check
        self.current_row = None
        self.executed_sql = []

    def execute(self, sql, _params=None):
        normalized = " ".join(sql.split()).lower()
        self.executed_sql.append(normalized)
        if "select check_date, actual_stock, created_at" in normalized:
            self.current_row = self.last_check
        elif "select stock_quantity from raw_materials" in normalized:
            self.current_row = (80.0,)
        elif (
            "from material_transactions" in normalized
            and "transaction_type = 'outflow'" in normalized
        ):
            self.current_row = (5.0,)
        elif (
            "from material_transactions" in normalized
            and "transaction_type in ('inflow','restock')" in normalized
        ):
            self.current_row = (2.0 if self.last_check else 10.0,)
        else:
            raise AssertionError(f"Unexpected SQL: {normalized}")

    def fetchone(self):
        return self.current_row


class WastageDb:
    def __init__(self, last_check):
        self.cursor_instance = WastageCursor(last_check)
        self.commit_count = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commit_count += 1


def test_theoretical_material_use_reads_outflow_transactions(monkeypatch):
    database = WastageDb(
        ("2026-07-13", 100.0, datetime(2026, 7, 13, 20, 45))
    )
    monkeypatch.setattr(bff, "get_db", lambda: database)

    result = bff._get_theoretical("Bread Flour")

    assert result[:4] == (97.0, 5.0, 2.0, 100.0)
    sql = " ".join(database.cursor_instance.executed_sql)
    assert "material_transactions" in sql
    assert "transaction_type = 'outflow'" in sql
    assert "order_items" not in sql
    assert "insert into material_wastage_log" not in sql
    assert database.commit_count == 0


def test_first_theoretical_view_is_read_only_and_reconstructs_opening_stock(
    monkeypatch,
):
    database = WastageDb(None)
    monkeypatch.setattr(bff, "get_db", lambda: database)

    result = bff._get_theoretical("All-Purpose Flour")

    assert result[:4] == (80.0, 5.0, 10.0, 75.0)
    sql = " ".join(database.cursor_instance.executed_sql)
    assert "insert into material_wastage_log" not in sql
    assert database.commit_count == 0


def test_wastage_summary_filters_untracked_materials():
    start = BFF_SOURCE.index("async def wastage_summary")
    end = BFF_SOURCE.index("async def inventory_restock_history", start)
    body = BFF_SOURCE[start:end]

    assert body.count("rm.track_inventory = 1") == 2
