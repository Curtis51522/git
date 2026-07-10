import asyncio

from models.schemas import DeductRequest


class MemoryCursor:
    def __init__(self, db, dictionary=False):
        self.db = db
        self.dictionary = dictionary
        self.rows = []

    def execute(self, sql, params=None):
        if not sql.startswith("SELECT * FROM batch_inventory"):
            raise AssertionError(f"Unexpected SQL: {sql}")
        product_names = set(params or ())
        self.rows = [
            dict(row)
            for row in self.db.tables["batch_inventory"]
            if row["product_name"] in product_names
            and (row.get("quantity_remaining") or 0) > 0
        ]
        self.rows.sort(key=lambda row: row["production_time"])

    def fetchall(self):
        return self.rows

    def close(self):
        return None


class MemoryDB:
    def __init__(self):
        self.tables = {
            "batch_inventory": [],
            "inventory_transactions": [],
        }

    def cursor(self, dictionary=False):
        return MemoryCursor(self, dictionary=dictionary)


class MemoryQuery:
    def __init__(self, db, table):
        self.db = db
        self.table = table
        self.insert_data = None
        self.update_data = None
        self.batch_id = None

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
    from api import module1_yolo

    db = MemoryDB()
    monkeypatch.setattr(module1_yolo, "get_db", lambda: db)
    monkeypatch.setattr(module1_yolo, "q", lambda active_db, table: MemoryQuery(active_db, table))
    return module1_yolo, db


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


def test_confirmed_inflow_is_immediately_available_for_fifo_deduction(monkeypatch):
    module1_yolo, db = _install_memory_db(monkeypatch)
    request = DeductRequest(items=[
        {"product_name": "bread_coconut", "quantity": 10, "tray_color": "green"}
    ])
    asyncio.run(module1_yolo.inflow_batch(request))

    result = asyncio.run(module1_yolo.deduct_inventory(DeductRequest(items=[
        {"product_name": "bread_coconut", "quantity": 1, "freshness": "Fresh"}
    ])))

    assert result.status == "ok"
    assert result.errors == []
    assert result.deducted[0]["quantity_deducted"] == 1
    assert db.tables["batch_inventory"][0]["quantity_remaining"] == 9
