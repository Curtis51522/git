from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import json

import pytest

from scripts import migrate_realistic_recipes as migration_module
from scripts.migrate_realistic_recipes import migrate
from scripts import reconcile_production_materials as reconciliation_module
from scripts.reconcile_production_materials import (
    calculate_expected_outflows,
    calculate_stock_adjustments,
    reconcile,
)
from scripts.recipe_catalog import load_catalog


CATALOG_PATH = Path("data/reference/product_recipe_catalog.json")


class MigrationCursor:
    def __init__(self, connection, dictionary=False):
        self.connection = connection
        self.dictionary = dictionary
        self.rows = []

    def execute(self, sql, params=None):
        params = params or ()
        normalized = " ".join(sql.split()).lower()
        self.connection.executed.append((normalized, tuple(params)))

        if "from information_schema.columns" in normalized:
            self.connection.in_transaction = True
            self.rows = [(1,)] if self.connection.track_column else []
            return
        if normalized.startswith("alter table raw_materials add column"):
            self.connection.track_column = True
            self.connection.in_transaction = False
            for material in self.connection.raw_materials.values():
                material["track_inventory"] = True
            self.rows = []
            return
        if normalized.startswith("select product_name from products"):
            names = sorted(
                name
                for name, product in self.connection.products.items()
                if product["category"].lower() == "bakery"
            )
            self.rows = (
                [{"product_name": name} for name in names]
                if self.dictionary
                else [(name,) for name in names]
            )
            return
        if normalized.startswith("insert into raw_materials"):
            (
                name,
                stock,
                unit,
                price,
                category,
                reorder_point,
                track_inventory,
            ) = params
            existing_stock = self.connection.raw_materials.get(name, {}).get(
                "stock_quantity", stock
            )
            self.connection.raw_materials[name] = {
                "stock_quantity": Decimal(str(existing_stock)),
                "unit": unit,
                "unit_price": Decimal(str(price)),
                "category": category,
                "reorder_point": Decimal(str(reorder_point)),
                "track_inventory": bool(track_inventory),
            }
            self.rows = []
            return
        if normalized.startswith("delete from product_recipes"):
            self.connection.product_recipes[params[0]] = {}
            self.rows = []
            return
        if normalized.startswith("insert into product_recipes"):
            product_name, material_name, quantity = params
            self.connection.product_recipes.setdefault(product_name, {})[
                material_name
            ] = Decimal(str(quantity))
            self.rows = []
            return
        if normalized.startswith("update products set material_cost"):
            cost, product_name = params
            self.connection.products[product_name]["material_cost"] = Decimal(
                str(cost)
            )
            self.rows = []
            return
        if normalized.startswith("select pr.product_name"):
            rows = []
            for product_name in sorted(self.connection.product_recipes):
                for material_name, quantity in sorted(
                    self.connection.product_recipes[product_name].items()
                ):
                    rows.append(
                        {
                            "product_name": product_name,
                            "material_name": material_name,
                            "quantity_per_unit": quantity,
                            "unit": self.connection.raw_materials[material_name]["unit"],
                        }
                    )
            self.rows = rows if self.dictionary else [tuple(row.values()) for row in rows]
            return
        if normalized.startswith("select product_name, material_cost from products"):
            rows = [
                {
                    "product_name": name,
                    "material_cost": product["material_cost"],
                }
                for name, product in sorted(self.connection.products.items())
                if product["category"].lower() == "bakery"
            ]
            self.rows = rows if self.dictionary else [tuple(row.values()) for row in rows]
            return
        raise AssertionError(f"Unexpected SQL: {normalized}")

    def executemany(self, sql, parameter_rows):
        for params in parameter_rows:
            self.execute(sql, params)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        return None


class MigrationConnection:
    def __init__(self, product_names):
        self.products = {
            name: {"category": "bakery", "material_cost": Decimal("0.00")}
            for name in product_names
        }
        self.raw_materials = {
            "Bread Flour": {
                "stock_quantity": Decimal("12.345678"),
                "unit": "kg",
                "unit_price": Decimal("8.00"),
                "category": "flour",
                "reorder_point": Decimal("5.000000"),
            }
        }
        self.product_recipes = {name: {} for name in product_names}
        self.track_column = False
        self.commit_count = 0
        self.rollback_count = 0
        self.executed = []
        self._transaction_snapshot = None
        self.close_count = 0
        self.in_transaction = False

    def cursor(self, dictionary=False):
        return MigrationCursor(self, dictionary=dictionary)

    def start_transaction(self):
        if self.in_transaction:
            raise RuntimeError("Transaction already in progress")
        self._transaction_snapshot = self.snapshot()
        self.in_transaction = True

    def commit(self):
        self.commit_count += 1
        self._transaction_snapshot = None
        self.in_transaction = False

    def rollback(self):
        self.rollback_count += 1
        if self._transaction_snapshot is not None:
            state = self._transaction_snapshot
            self.products = deepcopy(state["products"])
            self.raw_materials = deepcopy(state["raw_materials"])
            self.product_recipes = deepcopy(state["product_recipes"])
            self._transaction_snapshot = None
        self.in_transaction = False

    def snapshot(self):
        return {
            "products": deepcopy(self.products),
            "raw_materials": deepcopy(self.raw_materials),
            "product_recipes": deepcopy(self.product_recipes),
            "track_column": self.track_column,
        }

    def close(self):
        self.close_count += 1


def _expected_products():
    return {
        product["product_name"] for product in load_catalog(CATALOG_PATH)["products"]
    }


def test_migration_rolls_back_when_catalog_product_is_unknown():
    connection = MigrationConnection({"baguette"})

    with pytest.raises(ValueError, match="Catalog product mismatch"):
        migrate(connection, CATALOG_PATH)

    assert connection.rollback_count == 1
    assert connection.commit_count == 0
    assert connection.product_recipes == {"baguette": {}}


def test_migration_is_idempotent_and_preserves_existing_stock():
    connection = MigrationConnection(_expected_products())

    migrate(connection, CATALOG_PATH)
    first_state = connection.snapshot()
    migrate(connection, CATALOG_PATH)

    assert connection.snapshot() == first_state
    assert connection.commit_count == 3
    assert connection.rollback_count == 0
    assert connection.raw_materials["Bread Flour"]["stock_quantity"] == Decimal(
        "12.345678"
    )


def test_migration_closes_schema_check_transaction_before_data_transaction():
    connection = MigrationConnection(_expected_products())
    connection.track_column = True

    migrate(connection, CATALOG_PATH)

    assert connection.commit_count == 2
    assert connection.rollback_count == 0


def test_migration_uses_parameterized_product_scoped_recipe_replacement():
    connection = MigrationConnection(_expected_products())

    migrate(connection, CATALOG_PATH)

    delete_calls = [
        call
        for call in connection.executed
        if call[0].startswith("delete from product_recipes")
    ]
    assert len(delete_calls) == 30
    assert all("where product_name = %s" in sql for sql, _ in delete_calls)
    assert all(len(params) == 1 for _, params in delete_calls)
    assert not any("truncate" in sql for sql, _ in connection.executed)


def test_migration_cli_defaults_to_validation_without_database_connection(capsys):
    def unexpected_connection(**_kwargs):
        raise AssertionError("Validation mode must not connect to the database")

    report = migration_module.main([], connection_factory=unexpected_connection)
    output = json.loads(capsys.readouterr().out)

    assert report == output
    assert report["mode"] == "validate-only"
    assert report["applied"] is False
    assert report["product_count"] == 30
    assert report["material_count"] == 37
    assert report["recipe_row_count"] == 223
    assert report["untracked_materials"] == ["Water"]


def test_migration_cli_validate_only_flag_does_not_connect(capsys):
    def unexpected_connection(**_kwargs):
        raise AssertionError("Validation mode must not connect to the database")

    report = migration_module.main(
        ["--validate-only"],
        connection_factory=unexpected_connection,
    )
    capsys.readouterr()

    assert report["mode"] == "validate-only"
    assert report["applied"] is False


def test_migration_cli_requires_apply_flag_for_database_changes(capsys):
    connection = MigrationConnection(_expected_products())

    report = migration_module.main(
        ["--apply"],
        connection_factory=lambda **_kwargs: connection,
    )
    output = json.loads(capsys.readouterr().out)

    assert report == output
    assert report["mode"] == "apply"
    assert report["applied"] is True
    assert connection.commit_count == 1
    assert connection.close_count == 1


def _reconciliation_state():
    batches = [
        {
            "product_name": "brownie",
            "quantity_initial": 2,
            "production_time": datetime(2026, 7, 14, 5, 45),
        },
        {
            "product_name": "bread_roll",
            "quantity_initial": 3,
            "production_time": datetime(2026, 7, 14, 6, 15),
        },
    ]
    recipes = [
        {
            "product_name": "brownie",
            "wastage_pct": Decimal("0.05"),
            "material_name": "Bread Flour",
            "quantity_per_unit": Decimal("0.100000"),
            "unit": "kg",
            "category": "flour",
            "track_inventory": True,
        },
        {
            "product_name": "brownie",
            "wastage_pct": Decimal("0.05"),
            "material_name": "Water",
            "quantity_per_unit": Decimal("0.020000"),
            "unit": "L",
            "category": "utility",
            "track_inventory": False,
        },
        {
            "product_name": "bread_roll",
            "wastage_pct": Decimal("0.10"),
            "material_name": "Bread Flour",
            "quantity_per_unit": Decimal("0.050000"),
            "unit": "kg",
            "category": "flour",
            "track_inventory": True,
        },
    ]
    existing_outflows = [
        {
            "material_name": "Bread Flour",
            "quantity": Decimal("0.300000"),
            "unit": "kg",
            "reference": "production:20260714054500000000",
            "created_at": datetime(2026, 7, 14, 5, 45),
        }
    ]
    stocks = {
        "Bread Flour": {
            "stock_quantity": Decimal("10.000000"),
            "track_inventory": True,
        },
        "Water": {
            "stock_quantity": Decimal("0.000000"),
            "track_inventory": False,
        },
    }
    return {
        "batches": batches,
        "recipes": recipes,
        "existing_outflows": existing_outflows,
        "stocks": stocks,
    }


def test_expected_outflows_preserve_production_groups_and_water_contract():
    state = _reconciliation_state()

    outflows = calculate_expected_outflows(state["batches"], state["recipes"])

    assert set(outflows) == {
        "production:20260714054500000000",
        "production:20260714061500000000",
    }
    first = outflows["production:20260714054500000000"]
    second = outflows["production:20260714061500000000"]
    assert first["Bread Flour"]["quantity"] == Decimal("0.210000")
    assert first["Water"]["quantity"] == Decimal("0.040000")
    assert first["Water"]["track_inventory"] is False
    assert second["Bread Flour"]["quantity"] == Decimal("0.165000")


def test_stock_adjustments_ignore_water_and_block_negative_stock():
    state = _reconciliation_state()
    outflows = calculate_expected_outflows(state["batches"], state["recipes"])

    adjustments = calculate_stock_adjustments(
        state["stocks"], state["existing_outflows"], outflows
    )

    assert adjustments == {
        "Bread Flour": {
            "old_outflow": Decimal("0.300000"),
            "expected_outflow": Decimal("0.375000"),
            "current_stock": Decimal("10.000000"),
            "new_stock": Decimal("9.925000"),
        }
    }

    state["stocks"]["Bread Flour"]["stock_quantity"] = Decimal("0.010000")
    with pytest.raises(ValueError, match="negative stock"):
        calculate_stock_adjustments(
            state["stocks"], state["existing_outflows"], outflows
        )


def test_reconcile_dry_run_reports_without_starting_a_transaction(monkeypatch):
    state = _reconciliation_state()

    class ReadOnlyConnection:
        def start_transaction(self):
            raise AssertionError("dry-run must not start a transaction")

        def commit(self):
            raise AssertionError("dry-run must not commit")

        def rollback(self):
            raise AssertionError("dry-run must not roll back")

    monkeypatch.setattr(
        reconciliation_module,
        "_load_reconciliation_state",
        lambda connection, production_date, lock: state,
    )

    report = reconcile(ReadOnlyConnection(), date(2026, 7, 14), dry_run=True)

    assert report["production_date"] == "2026-07-14"
    assert report["batch_units"] == 5
    assert report["production_group_count"] == 2
    assert report["changed_material_count"] == 1


def test_reconciliation_delete_is_date_scoped_to_production_outflows():
    source = Path("scripts/reconcile_production_materials.py").read_text(
        encoding="utf-8"
    )

    assert "DELETE FROM material_transactions" in source
    assert "transaction_type = 'outflow'" in source
    assert "reference LIKE 'production:%'" in source
    assert "DATE(created_at) = %s" in source
    assert "DELETE FROM material_transactions WHERE reference = %s" not in source
