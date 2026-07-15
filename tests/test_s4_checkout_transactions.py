import asyncio
import copy
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException


class RecordingConnection:
    def __init__(
        self,
        autocommit=False,
        fail_on_receipt=False,
        fail_on_refund_update=False,
        fail_on_material_transaction=False,
    ):
        self.autocommit = autocommit
        self.fail_on_receipt = fail_on_receipt
        self.fail_on_refund_update = fail_on_refund_update
        self.fail_on_material_transaction = fail_on_material_transaction
        self.events = []
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False
        self.cursors = []
        self.product_rows = {
            "croissant": (1.45, 0.5, 0.0),
            "latte": (18.0, 2.0, 0.0),
        }
        self.recipes = {
            "croissant": [("Bread Flour", 0.1, "kg", "ingredient")],
            "latte": [("Coffee Beans", 0.02, "kg", "ingredient")],
        }
        self.refund_items = [
            ("croissant", 1, "Fresh", None),
            ("seasonal_drink", 2, "Fresh", "large"),
        ]
        self.refund_outflows = [
            (501, "BATCH-1", "croissant", 1, 1.45, 0, "Fresh"),
            (502, None, "seasonal_drink", 2, 8.0, 0, "Fresh"),
        ]
        self.state = {
            "batch_remaining": 5,
            "inventory_transactions": [],
            "orders": [],
            "order_items": [],
            "materials": {
                "Cup Regular": 10,
                "Cup Large": 10,
                "Packaging Box": 10,
                "Packaging Bag": 10,
                "Bread Flour": 10,
                "Coffee Beans": 10,
            },
            "material_transactions": [],
            "payments": [],
            "receipts": [],
        }
        self.material_tracking = {
            material_name: True for material_name in self.state["materials"]
        }
        self.material_prices = {
            "Cup Regular": 0.0,
            "Cup Large": 0.0,
            "Packaging Box": 0.30,
            "Packaging Bag": 0.15,
            "Bread Flour": 8.0,
            "Coffee Beans": 80.0,
        }
        self.initial_state = copy.deepcopy(self.state)

    def cursor(self, dictionary=False):
        cursor = RecordingCursor(self, dictionary=dictionary)
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        self.commit_count += 1
        self.events.append(("commit", None))

    def rollback(self):
        self.rollback_count += 1
        self.state = copy.deepcopy(self.initial_state)
        self.events.append(("rollback", None))

    def close(self):
        self.closed = True
        self.events.append(("connection_close", None))


class RecordingCursor:
    def __init__(self, db, dictionary=False):
        self.db = db
        self.dictionary = dictionary
        self.rows = []
        self.lastrowid = 101
        self.closed = False
        self.rowcount = 0

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        params = tuple(params or ())
        self.rowcount = 0
        self.db.events.append((normalized, params))

        if normalized.startswith("SELECT product_name, selling_price"):
            requested = set(params)
            self.rows = [
                (name, values[0], values[1], values[2])
                for name, values in self.db.product_rows.items()
                if name in requested
            ]
        elif normalized.startswith("SELECT product_name, material_cost"):
            requested = set(params)
            self.rows = [
                (name, values[1], values[2])
                for name, values in self.db.product_rows.items()
                if name in requested
            ]
        elif normalized.startswith("SELECT COUNT(*)"):
            self.rows = [(1,)]
        elif normalized.startswith("SELECT pr.product_name"):
            requested = set(params)
            self.rows = [
                (product_name, *recipe)
                for product_name, recipes in self.db.recipes.items()
                if product_name in requested
                for recipe in recipes
            ]
        elif normalized.startswith("SELECT pr.material_name"):
            self.rows = list(self.db.recipes.get(params[0], []))
        elif normalized.startswith(
            "SELECT material_name, stock_quantity, unit, unit_price FROM raw_materials"
        ):
            requested = set(params)
            self.rows = [
                (
                    material_name,
                    quantity,
                    "pcs" if material_name.startswith(("Cup", "Packaging")) else "kg",
                    self.db.material_prices[material_name],
                )
                for material_name, quantity in self.db.state["materials"].items()
                if material_name in requested
            ]
        elif normalized.startswith(
            "SELECT material_name, stock_quantity, unit FROM raw_materials"
        ):
            requested = set(params)
            self.rows = [
                (
                    material_name,
                    quantity,
                    "pcs" if material_name.startswith(("Cup", "Packaging")) else "kg",
                )
                for material_name, quantity in self.db.state["materials"].items()
                if material_name in requested
            ]
        elif normalized.startswith(
            "SELECT stock_quantity, unit, track_inventory FROM raw_materials"
        ):
            material_name = params[0]
            if material_name in self.db.state["materials"]:
                unit = (
                    "pcs"
                    if material_name.startswith(("Cup", "Packaging"))
                    else "kg"
                )
                self.rows = [
                    (
                        self.db.state["materials"][material_name],
                        unit,
                        self.db.material_tracking.get(material_name, True),
                    )
                ]
            else:
                self.rows = []
        elif normalized.startswith(
            "SELECT stock_quantity, unit FROM raw_materials"
        ):
            material_name = params[0]
            if material_name in self.db.state["materials"]:
                unit = (
                    "pcs"
                    if material_name.startswith(("Cup", "Packaging"))
                    else "kg"
                )
                self.rows = [
                    (self.db.state["materials"][material_name], unit)
                ]
            else:
                self.rows = []
        elif normalized.startswith("INSERT INTO orders"):
            self.db.state["orders"].append(params)
        elif normalized.startswith("INSERT INTO order_items"):
            self.db.state["order_items"].append(params)
        elif normalized.startswith("UPDATE raw_materials SET stock_quantity = stock_quantity -"):
            if len(params) == 1:
                quantity, material_name, minimum = 1, params[0], 1
            else:
                quantity, material_name = params[:2]
                minimum = params[2] if len(params) > 2 else quantity
            if self.db.state["materials"].get(material_name, 0) >= minimum:
                self.db.state["materials"][material_name] -= quantity
                self.rowcount = 1
            else:
                self.rowcount = 0
        elif normalized.startswith("UPDATE raw_materials SET stock_quantity = stock_quantity +"):
            if len(params) == 1:
                quantity, material_name = 1, params[0]
            else:
                quantity, material_name = params
            if material_name in self.db.state["materials"]:
                self.db.state["materials"][material_name] += quantity
                self.rowcount = 1
        elif normalized.startswith("INSERT INTO material_transactions"):
            if self.db.fail_on_material_transaction:
                raise RuntimeError("material transaction write failed")
            self.db.state["material_transactions"].append(params)
        elif normalized.startswith("INSERT INTO payments"):
            self.db.state["payments"].append(params)
        elif normalized.startswith("INSERT INTO receipts"):
            if self.db.fail_on_receipt:
                raise RuntimeError("receipt write failed")
            self.db.state["receipts"].append(params)
        elif normalized.startswith("SELECT id, state, dine_type FROM orders"):
            self.rows = [(101, "paid", "takeaway")]
        elif normalized.startswith("SELECT product_name, quantity, freshness, coffee_size"):
            self.rows = list(self.db.refund_items)
        elif normalized.startswith(
            "SELECT id, batch_id, product_name, quantity, unit_price"
        ):
            self.rows = list(self.db.refund_outflows)
        elif normalized.startswith("INSERT INTO inventory_transactions"):
            self.db.state["inventory_transactions"].append(params)
        elif normalized.startswith("UPDATE orders SET state = 'refunded'"):
            if self.db.fail_on_refund_update:
                raise RuntimeError("refund state write failed")
            self.db.state["orders"] = [(101, "refunded")]

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def close(self):
        self.closed = True
        self.db.events.append(("cursor_close", None))


class RecordingQuery:
    def __init__(self, db, table):
        self.db = db
        self.table = table
        self.data = None

    def insert(self, data):
        self.data = dict(data)
        return self

    def execute(self):
        self.db.events.append((f"QUERY INSERT {self.table}", dict(self.data)))
        self.db.state[self.table].append(dict(self.data))
        return types.SimpleNamespace(data=[self.data])


@pytest.fixture
def bff_module():
    mysql_stub = types.ModuleType("db.mysql_client")
    mysql_stub.get_db = lambda **_kwargs: None
    mysql_stub.q = lambda db, table: RecordingQuery(db, table)
    sys.modules.pop("api.module4_frontend.bff", None)
    with patch.dict(sys.modules, {"db.mysql_client": mysql_stub}):
        module = importlib.import_module("api.module4_frontend.bff")
    yield module
    sys.modules.pop("api.module4_frontend.bff", None)


def _checkout_dependencies(monkeypatch, bff, db, deduction_status="ok"):
    calls = []

    def get_db(*, autocommit=True):
        calls.append(autocommit)
        assert autocommit is False
        return db

    async def deduct_inventory(request, db=None):
        assert db is not None
        assert db is not None and db.autocommit is False
        assert request.receipt_id
        assert request.items[0]["unit_price"] == 1.5
        assert request.items[0]["discount_applied"] == 0.1
        quantity = request.items[0]["quantity"]
        db.state["batch_remaining"] -= quantity
        db.state["inventory_transactions"].append({
            "type": "fifo",
            "quantity": quantity,
            "receipt_id": getattr(request, "receipt_id", None),
        })
        db.events.append(("FIFO DEDUCTION", quantity))
        errors = ["Insufficient stock"] if deduction_status == "partial" else []
        return types.SimpleNamespace(
            status=deduction_status,
            deducted=[{"product_name": "croissant", "quantity_deducted": quantity}],
            errors=errors,
        )

    async def discounts(_items):
        return {
            "croissant": {
                "discount_pct": 10,
                "source": "s5_dynamic",
                "strategy": "clearance",
                "reason": "Boundary test",
            }
        }

    monkeypatch.setattr(bff, "get_db", get_db)
    monkeypatch.setattr(bff, "get_product_prices", lambda: {"croissant": 1.45, "latte": 18.0})
    monkeypatch.setattr(bff, "_fetch_validated_dynamic_discounts", discounts)

    yolo_stub = types.ModuleType("api.module1_yolo")
    yolo_stub.deduct_inventory = deduct_inventory
    freshness_stub = types.ModuleType("api.freshness_service")
    freshness_stub.get_discount_rate = lambda freshness: 0.2 if freshness == "Day-1" else 0.0
    return calls, yolo_stub, freshness_stub


def _checkout_payload():
    return {
        "items": [
            {
                "product_name": "croissant",
                "quantity": 1,
                "freshness": "Fresh",
                "discount_rate": 0.1,
            },
            {
                "product_name": "latte",
                "quantity": 1,
                "size": "regular",
                "temperature": "hot",
                "sugar": "normal",
                "ice_level": "none",
            },
        ],
        "payment_method": "card",
        "dine_type": "dine_in",
    }


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True, "2"])
def test_checkout_rejects_quantity_that_is_not_a_positive_integer(
    quantity, monkeypatch, bff_module
):
    db = RecordingConnection()
    _, yolo_stub, freshness_stub = _checkout_dependencies(
        monkeypatch, bff_module, db
    )
    payload = _checkout_payload()
    payload["items"] = [payload["items"][1]]
    payload["items"][0]["quantity"] = quantity

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(bff_module.checkout_complete(payload))

    assert exc_info.value.status_code == 400
    assert "positive integer" in str(exc_info.value.detail)
    assert db.commit_count == 0
    assert db.state == db.initial_state


@pytest.mark.parametrize("freshness", [None, "Expired", "Unknown"])
def test_checkout_requires_saleable_bakery_freshness(
    freshness, monkeypatch, bff_module
):
    db = RecordingConnection()
    _, yolo_stub, freshness_stub = _checkout_dependencies(
        monkeypatch, bff_module, db
    )
    payload = _checkout_payload()
    payload["items"] = [payload["items"][0]]
    if freshness is None:
        payload["items"][0].pop("freshness")
    else:
        payload["items"][0]["freshness"] = freshness

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(bff_module.checkout_complete(payload))

    assert exc_info.value.status_code == 400
    assert "freshness" in str(exc_info.value.detail).lower()
    assert db.commit_count == 0
    assert db.state == db.initial_state


def test_checkout_uses_one_canonical_beverage_price_for_receipt_and_outflow(
    monkeypatch, bff_module
):
    db = RecordingConnection()
    _, yolo_stub, freshness_stub = _checkout_dependencies(
        monkeypatch, bff_module, db
    )
    monkeypatch.setattr(
        bff_module,
        "get_product_prices",
        lambda: {"croissant": 1.45},
    )

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        result = asyncio.run(bff_module.checkout_complete(_checkout_payload()))

    receipt_latte = next(
        item for item in result["receipt"]["items"]
        if item["product_name"] == "latte"
    )
    outflow_latte = next(
        item for item in db.state["inventory_transactions"]
        if isinstance(item, dict) and item.get("product_name") == "latte"
    )
    assert receipt_latte["unit_price"] == 18.0
    assert outflow_latte["unit_price"] == 18.0


def test_checkout_rejects_missing_canonical_product_price_before_writes(
    monkeypatch, bff_module
):
    db = RecordingConnection()
    db.product_rows.pop("latte")
    _, yolo_stub, freshness_stub = _checkout_dependencies(
        monkeypatch, bff_module, db
    )

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(bff_module.checkout_complete(_checkout_payload()))

    assert exc_info.value.status_code == 409
    assert "canonical price" in str(exc_info.value.detail).lower()
    assert db.commit_count == 0
    assert db.state == db.initial_state


def test_checkout_rejects_missing_recipe_before_writes(monkeypatch, bff_module):
    db = RecordingConnection()
    db.recipes.pop("latte")
    _, yolo_stub, freshness_stub = _checkout_dependencies(
        monkeypatch, bff_module, db
    )
    payload = _checkout_payload()
    payload["items"] = [payload["items"][1]]

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(bff_module.checkout_complete(payload))

    assert exc_info.value.status_code == 409
    assert "missing product recipe" in str(exc_info.value.detail).lower()
    assert db.commit_count == 0
    assert db.state == db.initial_state


def test_checkout_rejects_non_positive_recipe_quantity_before_writes(
    monkeypatch, bff_module
):
    db = RecordingConnection()
    db.recipes["latte"] = [
        ("Coffee Beans", -0.1, "kg", "ingredient")
    ]
    _, yolo_stub, freshness_stub = _checkout_dependencies(
        monkeypatch, bff_module, db
    )
    payload = _checkout_payload()
    payload["items"] = [payload["items"][1]]

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(bff_module.checkout_complete(payload))

    assert exc_info.value.status_code == 409
    assert "invalid recipe quantity" in str(exc_info.value.detail).lower()
    assert db.commit_count == 0
    assert db.state == db.initial_state


@pytest.mark.parametrize(
    ("material_name", "dine_type", "items"),
    [
        (
            "Cup Regular",
            "dine_in",
            [
                {
                    "product_name": "latte",
                    "quantity": 1,
                    "size": "regular",
                    "temperature": "hot",
                    "sugar": "normal",
                    "ice_level": "none",
                }
            ],
        ),
        (
            "Packaging Box",
            "takeaway",
            [
                {
                    "product_name": "croissant",
                    "quantity": 1,
                    "freshness": "Fresh",
                }
            ],
        ),
    ],
)
def test_checkout_rejects_insufficient_material_before_any_commit(
    material_name, dine_type, items, monkeypatch, bff_module
):
    db = RecordingConnection()
    db.state["materials"][material_name] = 0
    db.initial_state = copy.deepcopy(db.state)
    _, yolo_stub, freshness_stub = _checkout_dependencies(
        monkeypatch, bff_module, db
    )
    payload = {
        "items": items,
        "payment_method": "card",
        "dine_type": dine_type,
    }

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(bff_module.checkout_complete(payload))

    assert exc_info.value.status_code == 409
    assert material_name in str(exc_info.value.detail)
    assert db.commit_count == 0
    assert db.state == db.initial_state


def test_checkout_does_not_consume_bakery_recipe_materials(
    monkeypatch, bff_module
):
    db = RecordingConnection()
    db.state["materials"]["Bread Flour"] = 0
    db.initial_state = copy.deepcopy(db.state)
    _, yolo_stub, freshness_stub = _checkout_dependencies(
        monkeypatch, bff_module, db
    )
    payload = _checkout_payload()
    payload["items"] = [payload["items"][0]]

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        result = asyncio.run(bff_module.checkout_complete(payload))

    assert result["status"] == "ok"
    assert db.state["materials"]["Bread Flour"] == 0
    assert all(
        transaction[0] != "Bread Flour"
        for transaction in db.state["material_transactions"]
    )


def test_checkout_commits_once_after_receipt_on_one_transaction(monkeypatch, bff_module):
    db = RecordingConnection()
    calls, yolo_stub, freshness_stub = _checkout_dependencies(monkeypatch, bff_module, db)

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        result = asyncio.run(bff_module.checkout_complete(_checkout_payload()))

    assert result["status"] == "ok"
    assert calls == [False]
    assert db.commit_count == 1
    assert db.rollback_count == 0
    assert len(db.state["receipts"]) == 1
    assert result["receipt"]["items"][0]["unit_price"] == 1.5
    assert result["receipt"]["items"][0]["discount_amount"] == 0.1
    assert result["receipt"]["items"][0]["line_total"] == 1.4
    receipt_event = next(i for i, event in enumerate(db.events) if str(event[0]).startswith("INSERT INTO receipts"))
    commit_event = next(i for i, event in enumerate(db.events) if event[0] == "commit")
    assert receipt_event < commit_event
    assert db.closed
    assert all(cursor.closed for cursor in db.cursors)
    order_row = db.state["orders"][0]
    line_profit_total = sum(row[6] for row in db.state["order_items"])
    assert order_row[6] == pytest.approx(16.9)
    assert order_row[6] == pytest.approx(line_profit_total)


def test_takeaway_profit_deducts_packaging_material_cost(monkeypatch, bff_module):
    db = RecordingConnection()
    _, yolo_stub, freshness_stub = _checkout_dependencies(monkeypatch, bff_module, db)
    payload = _checkout_payload()
    payload["dine_type"] = "takeaway"

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        result = asyncio.run(bff_module.checkout_complete(payload))

    order_row = db.state["orders"][0]
    line_profit_total = sum(row[6] for row in db.state["order_items"])
    assert result["receipt"]["total"] == pytest.approx(19.7)
    assert order_row[6] == pytest.approx(line_profit_total + 0.30 - 0.45)


def test_checkout_links_fifo_and_beverage_outflows_to_receipt(
    monkeypatch, bff_module
):
    db = RecordingConnection()
    _, yolo_stub, freshness_stub = _checkout_dependencies(
        monkeypatch, bff_module, db
    )

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        result = asyncio.run(bff_module.checkout_complete(_checkout_payload()))

    receipt_id = result["receipt"]["receipt_id"]
    fifo = next(
        item for item in db.state["inventory_transactions"]
        if item.get("type") == "fifo"
    )
    beverage = next(
        item for item in db.state["inventory_transactions"]
        if item.get("product_name") == "latte"
    )
    assert fifo["receipt_id"] == receipt_id
    assert beverage["receipt_id"] == receipt_id
    assert beverage["disposition"] == "sold"


def test_checkout_rolls_back_fifo_and_all_later_writes_after_receipt_failure(
    monkeypatch, bff_module
):
    db = RecordingConnection(fail_on_receipt=True)
    calls, yolo_stub, freshness_stub = _checkout_dependencies(monkeypatch, bff_module, db)

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        result = asyncio.run(bff_module.checkout_complete(_checkout_payload()))

    assert result["status"] == "error"
    assert calls == [False]
    assert db.commit_count == 0
    assert db.rollback_count == 1
    assert db.state == db.initial_state
    assert result["deducted"] == []
    assert any(event[0] == "FIFO DEDUCTION" for event in db.events)
    assert any(str(event[0]).startswith("INSERT INTO receipts") for event in db.events)
    assert db.closed


def test_checkout_rolls_back_partial_fifo_result_before_order_writes(monkeypatch, bff_module):
    db = RecordingConnection()
    calls, yolo_stub, freshness_stub = _checkout_dependencies(
        monkeypatch, bff_module, db, deduction_status="partial"
    )

    with patch.dict(
        sys.modules,
        {"api.module1_yolo": yolo_stub, "api.freshness_service": freshness_stub},
    ):
        result = asyncio.run(bff_module.checkout_complete(_checkout_payload()))

    assert result["status"] == "partial"
    assert calls == [False]
    assert db.commit_count == 0
    assert db.rollback_count == 1
    assert db.state == db.initial_state
    assert result["deducted"] == []
    assert "0 items deducted" in result["message"]
    assert not any(str(event[0]).startswith("INSERT INTO orders") for event in db.events)
    assert db.closed


def test_query_builder_preserves_autocommit_but_does_not_commit_explicit_transaction():
    connections = []

    class Connector:
        @staticmethod
        def connect(**kwargs):
            connection = RecordingConnection(autocommit=kwargs["autocommit"])
            connections.append(connection)
            return connection

    mysql_package = types.ModuleType("mysql")
    connector_module = types.ModuleType("mysql.connector")
    connector_module.connect = Connector.connect
    mysql_package.connector = connector_module
    module_name = "test_mysql_client_contract"
    spec = importlib.util.spec_from_file_location(module_name, Path("db/mysql_client.py"))
    module = importlib.util.module_from_spec(spec)

    with patch.dict(
        sys.modules,
        {"mysql": mysql_package, "mysql.connector": connector_module, module_name: module},
    ):
        spec.loader.exec_module(module)

    transaction = module.get_db(autocommit=False)
    module.q(transaction, "inventory_transactions").insert({"quantity": 1}).execute()
    assert transaction.autocommit is False
    assert transaction.commit_count == 0
    assert transaction.cursors[-1].closed

    regular = module.get_db()
    module.q(regular, "inventory_transactions").insert({"quantity": 1}).execute()
    assert regular.autocommit is True
    assert regular.commit_count == 1


def test_refund_records_non_sellable_returns_without_restocking(
    monkeypatch, bff_module
):
    db = RecordingConnection()
    monkeypatch.setattr(bff_module, "get_db", lambda **_kwargs: db)
    monkeypatch.setattr(bff_module, "is_beverage", lambda name: name == "seasonal_drink")

    result = asyncio.run(bff_module.refund_order(
        {
            "ticket_id": "RCP-1",
            "reason": "Customer return",
        },
        user={"sub": "manager", "role": "manager"},
    ))

    assert result["status"] == "ok"
    assert db.state["materials"] == db.initial_state["materials"]
    assert db.state["inventory_transactions"] == [
        (
            "return",
            "BATCH-1",
            "croissant",
            1,
            1.45,
            0,
            "Fresh",
            "RCP-1",
            501,
            "non_sellable",
            "Customer return",
            "manager",
        ),
        (
            "return",
            None,
            "seasonal_drink",
            2,
            8.0,
            0,
            "Fresh",
            "RCP-1",
            502,
            "non_sellable",
            "Customer return",
            "manager",
        ),
    ]
    assert db.state["orders"] == [(101, "refunded")]
    assert db.commit_count == 1
    assert result["returned_units"] == 3
    assert result["disposition"] == "non_sellable"
    assert db.closed
    assert all(cursor.closed for cursor in db.cursors)

    order_lock = next(
        event[0]
        for event in db.events
        if str(event[0]).startswith("SELECT id, state, dine_type FROM orders")
    )
    assert order_lock.endswith("FOR UPDATE")


def test_restock_uses_one_locked_transaction(monkeypatch, bff_module):
    db = RecordingConnection()
    calls = []

    def get_db(*, autocommit=True):
        calls.append(autocommit)
        return db

    monkeypatch.setattr(bff_module, "get_db", get_db)

    result = asyncio.run(
        bff_module.inventory_restock(
            {"material_name": "Bread Flour", "quantity": 2.5}
        )
    )

    assert calls == [False]
    assert result["previous_stock"] == 10
    assert result["new_stock"] == 12.5
    assert db.state["materials"]["Bread Flour"] == 12.5
    assert db.state["material_transactions"] == [
        ("Bread Flour", "restock", 2.5, "kg", "manual_restock")
    ]
    assert any("FOR UPDATE" in event[0] for event in db.events)
    assert db.commit_count == 1
    assert db.rollback_count == 0
    assert db.closed


def test_restock_rolls_back_stock_when_audit_write_fails(monkeypatch, bff_module):
    db = RecordingConnection(fail_on_material_transaction=True)
    monkeypatch.setattr(bff_module, "get_db", lambda **_kwargs: db)

    with pytest.raises(RuntimeError, match="material transaction write failed"):
        asyncio.run(
            bff_module.inventory_restock(
                {"material_name": "Bread Flour", "quantity": 2.5}
            )
        )

    assert db.state == db.initial_state
    assert db.commit_count == 0
    assert db.rollback_count == 1
    assert db.closed


def test_restock_rejects_untracked_utility(monkeypatch, bff_module):
    db = RecordingConnection()
    db.state["materials"]["Water"] = 0
    db.material_tracking["Water"] = False
    db.initial_state = copy.deepcopy(db.state)
    monkeypatch.setattr(bff_module, "get_db", lambda **_kwargs: db)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            bff_module.inventory_restock(
                {"material_name": "Water", "quantity": 2.5}
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Material is not stock-tracked"
    assert db.state["materials"]["Water"] == 0
    assert db.commit_count == 0
    assert db.rollback_count == 1
    assert db.closed


@pytest.mark.parametrize("quantity", [None, 0, -1, True, "2", float("nan"), float("inf")])
def test_restock_rejects_invalid_quantity_before_database_access(
    quantity, monkeypatch, bff_module
):
    def unexpected_database_access(**_kwargs):
        raise AssertionError("invalid input must not access the database")

    monkeypatch.setattr(bff_module, "get_db", unexpected_database_access)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            bff_module.inventory_restock(
                {"material_name": "Bread Flour", "quantity": quantity}
            )
        )

    assert exc_info.value.status_code == 400


def test_refund_rolls_back_every_restoration_when_final_state_write_fails(
    monkeypatch, bff_module
):
    db = RecordingConnection(fail_on_refund_update=True)
    monkeypatch.setattr(bff_module, "get_db", lambda **_kwargs: db)
    monkeypatch.setattr(bff_module, "is_beverage", lambda name: name == "seasonal_drink")

    with pytest.raises(RuntimeError, match="refund state write failed"):
        asyncio.run(bff_module.refund_order(
            {
                "ticket_id": "RCP-1",
                "reason": "Customer return",
            },
            user={"sub": "manager", "role": "manager"},
        ))

    assert db.commit_count == 0
    assert db.rollback_count == 1
    assert db.state == db.initial_state
    assert db.closed


def test_refund_rejects_missing_original_outflow_evidence(
    monkeypatch, bff_module
):
    db = RecordingConnection()
    db.refund_outflows = []
    monkeypatch.setattr(bff_module, "get_db", lambda **_kwargs: db)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(bff_module.refund_order(
            {
                "ticket_id": "RCP-1",
                "reason": "Customer return",
            },
            user={"sub": "manager", "role": "manager"},
        ))

    assert exc_info.value.status_code == 409
    assert "allocation" in str(exc_info.value.detail).lower()
    assert db.commit_count == 0
    assert db.rollback_count == 1
    assert db.state == db.initial_state


def test_refund_requires_non_empty_reason(monkeypatch, bff_module):
    db = RecordingConnection()
    monkeypatch.setattr(bff_module, "get_db", lambda **_kwargs: db)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(bff_module.refund_order({"ticket_id": "RCP-1"}))

    assert exc_info.value.status_code == 400
    assert "reason" in str(exc_info.value.detail).lower()
    assert db.commit_count == 0
    assert db.state == db.initial_state


def test_fifo_deduction_locks_selected_batches_without_public_route():
    source = Path("api/module1_yolo.py").read_text(encoding="utf-8")

    assert "ORDER BY product_name, production_time, batch_id FOR UPDATE" in source
    assert "db = get_db()" in source
    assert "db.autocommit = False" in source
    assert "db.commit()" in source
    assert "db.rollback()" in source
    assert '"/deduct"' not in source
    assert "deduct_inventory_endpoint" not in source
