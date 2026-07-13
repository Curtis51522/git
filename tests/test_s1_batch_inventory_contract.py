import asyncio
import copy
import inspect
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from models.schemas import DeductRequest


class MemoryCursor:
    def __init__(self, db, dictionary=False):
        self.db = db
        self.dictionary = dictionary
        self.rows = []
        self.rowcount = 0
        self.closed = False

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        params = tuple(params or ())
        self.rowcount = 0

        if normalized.startswith("SELECT * FROM batch_inventory"):
            product_names = set(params)
            self.rows = [
                dict(row)
                for row in self.db.tables["batch_inventory"]
                if row["product_name"] in product_names
                and (row.get("quantity_remaining") or 0) > 0
                and (
                    "freshness_status IN" not in normalized
                    or row.get("freshness_status") in {"Fresh", "Day-1"}
                )
            ]
            self.rows.sort(key=lambda row: row["production_time"])
            return

        if normalized.startswith(
            "SELECT p.product_name, p.category, p.wastage_pct"
        ):
            rows = []
            for product_name in params:
                product = self.db.tables["products"].get(product_name)
                if product is None:
                    continue
                recipes = self.db.tables["product_recipes"].get(product_name, [])
                if not recipes:
                    rows.append(
                        (
                            product_name,
                            product["category"],
                            product["wastage_pct"],
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    )
                    continue
                for material_name, quantity_per_unit in recipes:
                    material = self.db.tables["raw_materials"].get(material_name)
                    rows.append(
                        (
                            product_name,
                            product["category"],
                            product["wastage_pct"],
                            material_name,
                            quantity_per_unit,
                            material["stock_quantity"] if material else None,
                            material["unit"] if material else None,
                            material["category"] if material else None,
                        )
                    )
            self.rows = rows
            return

        if normalized.startswith(
            "UPDATE raw_materials SET stock_quantity = stock_quantity -"
        ):
            quantity, material_name, minimum = params
            material = self.db.tables["raw_materials"].get(material_name)
            if material and material["stock_quantity"] >= minimum:
                material["stock_quantity"] -= quantity
                self.rowcount = 1
            return

        if normalized.startswith("INSERT INTO material_transactions"):
            material_name, transaction_type, quantity, unit, reference = params
            self.db.tables["material_transactions"].append(
                {
                    "material_name": material_name,
                    "transaction_type": transaction_type,
                    "quantity": quantity,
                    "unit": unit,
                    "reference": reference,
                }
            )
            self.rowcount = 1
            return

        raise AssertionError(f"Unexpected SQL: {normalized}")

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class MemoryDB:
    def __init__(self):
        self.autocommit = True
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False
        self.tables = {
            "batch_inventory": [],
            "inventory_transactions": [],
            "products": {
                "bread_coconut": {
                    "category": "bakery",
                    "wastage_pct": Decimal("0.05"),
                }
            },
            "product_recipes": {
                "bread_coconut": [
                    ("Bread Flour", Decimal("0.1")),
                ]
            },
            "raw_materials": {
                "Bread Flour": {
                    "stock_quantity": Decimal("10"),
                    "unit": "kg",
                    "category": "flour",
                }
            },
            "material_transactions": [],
        }
        self.committed_tables = copy.deepcopy(self.tables)

    def cursor(self, dictionary=False):
        return MemoryCursor(self, dictionary=dictionary)

    def commit(self):
        self.commit_count += 1
        self.committed_tables = copy.deepcopy(self.tables)

    def rollback(self):
        self.rollback_count += 1
        self.tables = copy.deepcopy(self.committed_tables)

    def close(self):
        self.closed = True


class MemoryQuery:
    def __init__(self, db, table):
        self.db = db
        self.table = table
        self.insert_data = None
        self.update_data = None
        self.select_data = None
        self.batch_id = None

    def select(self, columns):
        self.select_data = columns
        return self

    def insert(self, data):
        self.insert_data = dict(data)
        return self

    def update(self, data):
        self.update_data = dict(data)
        return self

    def eq(self, column, value):
        assert column == "batch_id"
        self.batch_id = value
        return self

    def execute(self):
        if self.select_data is not None:
            rows = [
                dict(row)
                for row in self.db.tables[self.table]
                if self.batch_id is None or row.get("batch_id") == self.batch_id
            ]
            return SimpleNamespace(data=rows)
        if self.insert_data is not None:
            self.db.tables[self.table].append(self.insert_data)
            return None
        if self.update_data is not None:
            for row in self.db.tables[self.table]:
                if row.get("batch_id") == self.batch_id:
                    row.update(self.update_data)
                    return None
            raise AssertionError(f"Unknown batch: {self.batch_id}")
        raise AssertionError("Query executed without insert or update data")


def _install_memory_db(monkeypatch):
    monkeypatch.setenv("BAKERY_SKIP_DB_INIT", "1")
    from api import module1_yolo

    db = MemoryDB()
    monkeypatch.setattr(module1_yolo, "get_db", lambda **_kwargs: db)
    monkeypatch.setattr(module1_yolo, "q", lambda active_db, table: MemoryQuery(active_db, table))
    return module1_yolo, db


def _batch(quantity_remaining):
    return {
        "batch_id": "BATCH-1",
        "product_name": "bread_coconut",
        "quantity": quantity_remaining,
        "quantity_initial": quantity_remaining,
        "quantity_remaining": quantity_remaining,
        "freshness_status": "Fresh",
        "production_time": "2026-07-11T08:00:00",
    }


def _sale_request(items, receipt_id="RCP-TEST-1"):
    priced_items = [
        {
            **item,
            "unit_price": item.get("unit_price", 1.45),
            "discount_applied": item.get("discount_applied", 0.0),
        }
        for item in items
    ]
    return DeductRequest(items=priced_items, receipt_id=receipt_id)


def test_confirmed_inflow_initializes_inventory_quantity_fields(monkeypatch):
    module1_yolo, db = _install_memory_db(monkeypatch)

    result = asyncio.run(module1_yolo.inflow_batch(DeductRequest(items=[
        {"product_name": "bread_coconut", "quantity": 10, "tray_color": "green"}
    ])))

    assert result["status"] == "ok"
    batch = db.tables["batch_inventory"][0]
    assert batch["quantity"] == 10
    assert batch["quantity_initial"] == 10
    assert batch["quantity_remaining"] == 10
    assert db.tables["raw_materials"]["Bread Flour"]["stock_quantity"] == Decimal(
        "8.95"
    )
    assert len(db.tables["material_transactions"]) == 1
    material_transaction = db.tables["material_transactions"][0]
    assert material_transaction["material_name"] == "Bread Flour"
    assert material_transaction["transaction_type"] == "outflow"
    assert material_transaction["quantity"] == Decimal("1.050000")
    assert material_transaction["unit"] == "kg"
    assert material_transaction["reference"].startswith("production:")


def test_confirmed_inflow_rolls_back_when_production_material_is_insufficient(
    monkeypatch,
):
    module1_yolo, db = _install_memory_db(monkeypatch)
    db.tables["raw_materials"]["Bread Flour"]["stock_quantity"] = Decimal("0")
    db.committed_tables = copy.deepcopy(db.tables)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            module1_yolo.inflow_batch(
                DeductRequest(
                    items=[
                        {
                            "product_name": "bread_coconut",
                            "quantity": 10,
                            "tray_color": "green",
                        }
                    ]
                )
            )
        )

    assert exc_info.value.status_code == 409
    assert "Bread Flour" in str(exc_info.value.detail)
    assert db.tables["batch_inventory"] == []
    assert db.tables["inventory_transactions"] == []
    assert db.tables["material_transactions"] == []
    assert db.rollback_count == 1


def test_scanned_inflow_initializes_inventory_quantity_fields(monkeypatch):
    module1_yolo, db = _install_memory_db(monkeypatch)

    class FakeUpload:
        content_type = "image/png"

        async def read(self):
            return b"image-bytes"

    class FakeImage:
        shape = (1, 1, 3)

    monkeypatch.setattr(module1_yolo.cv2, "imdecode", lambda *_args: FakeImage())
    monkeypatch.setattr(module1_yolo, "detect_products", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        module1_yolo,
        "aggregate_results",
        lambda _results: [
            {"product_name": "bread_coconut", "quantity": 10, "tray_color": "green"}
        ],
    )

    result = asyncio.run(module1_yolo.inflow_scan(FakeUpload()))

    assert result["status"] == "ok"
    batch = db.tables["batch_inventory"][0]
    assert batch["quantity"] == 10
    assert batch["quantity_initial"] == 10
    assert batch["quantity_remaining"] == 10


def test_search_uses_query_builder_and_preserves_public_q_parameter(monkeypatch):
    module1_yolo, db = _install_memory_db(monkeypatch)
    db.tables["batch_inventory"].append(
        {
            "batch_id": "BATCH-SEARCH-1",
            "product_name": "bread_coconut",
            "quantity": 4,
            "freshness_status": "Fresh",
            "sales_area": "Fresh Area",
            "production_time": "2026-07-11T08:00:00",
        }
    )

    result = asyncio.run(module1_yolo.search_products("BATCH-SEARCH-1"))

    assert result["match_type"] == "batch_id"
    assert result["results"][0]["batch_id"] == "BATCH-SEARCH-1"
    parameters = inspect.signature(module1_yolo.search_products).parameters
    assert "q" not in parameters
    assert parameters["keyword"].default.alias == "q"


def test_confirmed_inflow_is_immediately_available_for_fifo_deduction(monkeypatch):
    module1_yolo, db = _install_memory_db(monkeypatch)
    request = DeductRequest(items=[
        {"product_name": "bread_coconut", "quantity": 10, "tray_color": "green"}
    ])
    asyncio.run(module1_yolo.inflow_batch(request))

    result = asyncio.run(module1_yolo.deduct_inventory(_sale_request([
        {"product_name": "bread_coconut", "quantity": 1, "freshness": "Fresh"}
    ])))

    assert result.status == "ok"
    assert result.errors == []
    assert result.deducted[0]["quantity_deducted"] == 1
    assert db.tables["batch_inventory"][0]["quantity_remaining"] == 9


def test_duplicate_request_lines_share_the_locked_batch_balance(monkeypatch):
    module1_yolo, db = _install_memory_db(monkeypatch)
    db.tables["batch_inventory"].append(_batch(quantity_remaining=5))

    result = asyncio.run(module1_yolo.deduct_inventory(_sale_request([
        {"product_name": "bread_coconut", "quantity": 3},
        {"product_name": "bread_coconut", "quantity": 3},
    ])))

    assert result.status == "partial"
    assert sum(row["quantity_deducted"] for row in result.deducted) == 5
    assert db.tables["batch_inventory"][0]["quantity_remaining"] == 0
    assert sum(row["quantity"] for row in db.tables["inventory_transactions"]) == 5
    assert result.errors == ["Insufficient stock for 'bread_coconut': short by 1"]


def test_duplicate_request_lines_can_complete_with_shared_stock(monkeypatch):
    module1_yolo, db = _install_memory_db(monkeypatch)
    db.tables["batch_inventory"].append(_batch(quantity_remaining=5))

    result = asyncio.run(module1_yolo.deduct_inventory(_sale_request([
        {"product_name": "bread_coconut", "quantity": 2},
        {"product_name": "bread_coconut", "quantity": 2},
    ])))

    assert result.status == "ok"
    assert sum(row["quantity_deducted"] for row in result.deducted) == 4
    assert db.tables["batch_inventory"][0]["quantity_remaining"] == 1


def test_fifo_deduction_never_selects_expired_batches(monkeypatch):
    module1_yolo, db = _install_memory_db(monkeypatch)
    expired = _batch(quantity_remaining=5)
    expired.update(
        {
            "batch_id": "BATCH-EXPIRED",
            "freshness_status": "Expired",
            "production_time": "2026-07-09T08:00:00",
        }
    )
    fresh = _batch(quantity_remaining=5)
    fresh.update(
        {
            "batch_id": "BATCH-FRESH",
            "freshness_status": "Fresh",
            "production_time": "2026-07-11T08:00:00",
        }
    )
    db.tables["batch_inventory"].extend([expired, fresh])

    result = asyncio.run(module1_yolo.deduct_inventory(_sale_request([
        {"product_name": "bread_coconut", "quantity": 1}
    ])))

    assert result.status == "ok"
    assert result.deducted == [
        {
            "product_name": "bread_coconut",
            "batch_id": "BATCH-FRESH",
            "quantity_deducted": 1,
            "remaining_after": 4,
        }
    ]
    assert expired["quantity_remaining"] == 5


@pytest.mark.parametrize("quantity", [None, 0, -1, 1.5, True, "2"])
def test_direct_deduction_rejects_quantity_that_is_not_a_positive_integer(
    quantity, monkeypatch
):
    module1_yolo, db = _install_memory_db(monkeypatch)
    db.tables["batch_inventory"].append(_batch(quantity_remaining=5))

    result = asyncio.run(module1_yolo.deduct_inventory(_sale_request([
        {"product_name": "bread_coconut", "quantity": quantity}
    ])))

    assert result.status == "error"
    assert result.deducted == []
    assert result.errors == [
        "Invalid quantity for 'bread_coconut': expected a positive integer"
    ]
    assert db.tables["batch_inventory"][0]["quantity_remaining"] == 5
    assert db.tables["inventory_transactions"] == []


@pytest.mark.parametrize("quantity", [None, 0, -1, 1.5, True, "2"])
def test_confirmed_inflow_rejects_quantity_that_is_not_a_positive_integer(
    quantity, monkeypatch
):
    module1_yolo, db = _install_memory_db(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(module1_yolo.inflow_batch(DeductRequest(items=[
            {"product_name": "bread_coconut", "quantity": quantity}
        ])))

    assert exc_info.value.status_code == 400
    assert "positive integer" in str(exc_info.value.detail)
    assert db.tables["batch_inventory"] == []
    assert db.tables["inventory_transactions"] == []


def test_fifo_outflow_preserves_checkout_receipt_id(monkeypatch):
    module1_yolo, db = _install_memory_db(monkeypatch)
    db.tables["batch_inventory"].append(_batch(quantity_remaining=5))

    result = asyncio.run(module1_yolo.deduct_inventory(_sale_request(
        items=[{"product_name": "bread_coconut", "quantity": 2}],
        receipt_id="RCP-TRACE-1",
    )))

    assert result.status == "ok"
    assert db.tables["inventory_transactions"] == [
        {
            "transaction_type": "outflow",
            "batch_id": "BATCH-1",
            "product_name": "bread_coconut",
            "quantity": 2,
            "unit_price": 1.45,
            "discount_applied": 0.0,
            "freshness_status": "Fresh",
            "receipt_id": "RCP-TRACE-1",
            "disposition": "sold",
        }
    ]


def test_fifo_deduction_requires_receipt_and_canonical_item_price(monkeypatch):
    module1_yolo, db = _install_memory_db(monkeypatch)
    db.tables["batch_inventory"].append(_batch(quantity_remaining=5))

    missing_receipt = asyncio.run(
        module1_yolo.deduct_inventory(
            DeductRequest(
                items=[
                    {
                        "product_name": "bread_coconut",
                        "quantity": 1,
                        "unit_price": 1.45,
                    }
                ]
            )
        )
    )
    missing_price = asyncio.run(
        module1_yolo.deduct_inventory(
            DeductRequest(
                items=[{"product_name": "bread_coconut", "quantity": 1}],
                receipt_id="RCP-TEST-2",
            )
        )
    )

    assert missing_receipt.status == "error"
    assert missing_receipt.errors == ["receipt_id required for sold inventory"]
    assert missing_price.status == "error"
    assert missing_price.errors == [
        "Invalid canonical unit price for 'bread_coconut'"
    ]
    assert db.tables["batch_inventory"][0]["quantity_remaining"] == 5
    assert db.tables["inventory_transactions"] == []
