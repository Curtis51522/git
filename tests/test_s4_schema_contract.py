from pathlib import Path

from api import freshness_service
import re


CANONICAL_SCHEMA = Path("schema.sql")
EXPECTED_TABLES = frozenset(
    {
        "attendance_records",
        "attendance_correction_log",
        "batch_inventory",
        "business_events",
        "detection_log",
        "employees",
        "inventory_transactions",
        "material_inventory",
        "material_transactions",
        "material_wastage_log",
        "order_items",
        "orders",
        "payments",
        "product_inventory",
        "product_recipes",
        "products",
        "raw_materials",
        "recommendation_events",
        "receipts",
        "shift_schedule",
        "sick_leave_log",
        "sick_replacements",
        "users",
    }
)
DESTRUCTIVE_LEGACY_SCRIPTS = (
    Path("scripts/rebuild_mysql_schema.py"),
    Path("scripts/rebuild_v2.py"),
    Path("scripts/import_sales_to_mysql.py"),
)
OBSOLETE_CHECKOUT_MIGRATION = Path("scripts/migrate_checkout_schema.py")

CHECKOUT_COLUMNS = (
    "discount_rate",
    "line_total",
    "line_profit",
    "freshness",
    "coffee_size",
    "coffee_temp",
    "coffee_ice",
    "coffee_sugar",
)
ORDER_COLUMNS = (
    "ticket_id",
    "order_date",
    "subtotal",
    "discount_total",
    "state",
    "dine_type",
)


def _schema_source():
    assert CANONICAL_SCHEMA.exists(), "schema.sql must be the single database schema"
    return CANONICAL_SCHEMA.read_text(encoding="utf-8")


def _table_sql(table_name):
    match = re.search(
        rf"CREATE TABLE `{re.escape(table_name)}` \((.*?)\) ENGINE=InnoDB",
        _schema_source(),
        flags=re.DOTALL,
    )
    assert match, f"{table_name} missing from {CANONICAL_SCHEMA}"
    return match.group(1).lower()


def test_canonical_schema_is_mysql_structure_only():
    source = _schema_source()
    tables = re.findall(r"CREATE TABLE `([^`]+)`", source)

    assert "canonical MySQL schema" in source
    assert len(tables) == 23
    assert set(tables) == EXPECTED_TABLES
    assert "CREATE DATABASE IF NOT EXISTS `bakery_ai`" in source
    assert "USE `bakery_ai`;" in source
    assert "INSERT INTO" not in source.upper()
    assert "ON CONFLICT" not in source.upper()
    assert "TIMESTAMPTZ" not in source.upper()
    assert not re.search(r"\bSERIAL\b", source, flags=re.IGNORECASE)
    assert not re.search(r"\bAUTO_INCREMENT=\d+", source, flags=re.IGNORECASE)


def test_canonical_schema_keeps_safe_defaults():
    users = _table_sql("users")
    products = _table_sql("products")

    assert "`password_hash` varchar(255) not null" in users
    password_line = next(line for line in users.splitlines() if "password_hash" in line)
    assert "default" not in password_line
    assert "`wastage_pct` decimal(5,2) default 0.05" in products


def test_attendance_schema_allows_one_record_per_employee_date():
    attendance = _table_sql("attendance_records")

    assert "unique key `uq_attendance_emp_date` (`emp_id`,`date`)" in attendance
    assert "key `idx_emp_date` (`emp_id`,`date`)" not in attendance


def test_attendance_corrections_and_sick_replacements_are_persisted():
    corrections = _table_sql("attendance_correction_log")
    sick_log = _table_sql("sick_leave_log")
    replacements = _table_sql("sick_replacements")

    assert "`attendance_date` date not null" in corrections
    assert "`reason` varchar(255) not null" in corrections
    assert "`corrected_by` varchar(50) not null" in corrections
    assert "`leave_date` date not null" in sick_log
    assert "`replacement_employee_id` varchar(20)" in replacements


def test_canonical_schema_uses_mysql_native_json_columns():
    business_events = _table_sql("business_events")
    detection_log = _table_sql("detection_log")
    employees = _table_sql("employees")
    receipts = _table_sql("receipts")

    assert "`products` json default null" in business_events
    assert "`bbox` json default null" in detection_log
    assert "`skills` json not null default (json_array('bakery'))" in employees
    assert "`unavailable_dates` json not null default (json_array())" in employees
    assert "`items` json not null" in receipts
    assert "json_valid" not in _schema_source().lower()


def test_every_authoritative_order_items_schema_has_coffee_size():
    assert "coffee_size" in _table_sql("order_items")


def test_every_order_items_rebuild_keeps_the_checkout_contract():
    order_items = _table_sql("order_items")
    for column in CHECKOUT_COLUMNS:
        assert column in order_items, f"{column} missing from {CANONICAL_SCHEMA}"


def test_every_order_rebuild_keeps_the_checkout_contract():
    orders = _table_sql("orders")
    for column in ORDER_COLUMNS:
        assert column in orders, f"{column} missing from {CANONICAL_SCHEMA}"


def test_every_product_schema_supports_runtime_pricing_and_wastage():
    products = _table_sql("products")
    for column in ("unit_price", "selling_price", "wastage_pct"):
        assert column in products, f"{column} missing from {CANONICAL_SCHEMA}"


def test_raw_material_schema_declares_tracked_inventory_default_true():
    raw_materials = _table_sql("raw_materials")

    assert "`track_inventory` tinyint(1) not null default 1" in raw_materials


def test_every_order_schema_accepts_generated_receipt_ids():
    assert "`ticket_id` varchar(50)" in _table_sql("orders")


def test_order_children_reference_their_parent_order():
    source = _schema_source().lower()

    assert "constraint `fk_order_items_order`" in source
    assert "foreign key (`order_id`) references `orders` (`id`) on delete cascade" in source
    assert "constraint `fk_payments_order`" in source


def test_checkout_inventory_transactions_use_declared_trace_columns():
    source = Path("api/module4_frontend/bff.py").read_text(encoding="utf-8")

    for column in (
        "beverage_size",
        "beverage_temp",
        "beverage_sweetness",
        "beverage_ice",
    ):
        assert f'"{column}"' not in source

    refund_insert = source[source.index("INSERT INTO inventory_transactions") :]
    refund_insert = refund_insert[: refund_insert.index(")", refund_insert.index("VALUES"))]
    for column in (
        "receipt_id",
        "reversal_of_transaction_id",
        "disposition",
        "reason",
        "performed_by",
    ):
        assert column in refund_insert


def test_revenue_views_exclude_refunded_orders_and_expose_return_cost():
    source = Path("api/module4_frontend/bff.py").read_text(encoding="utf-8")
    daily = source[source.index('async def revenue_daily') : source.index('async def revenue_hourly')]
    hourly = source[source.index('async def revenue_hourly') : source.index('async def revenue_historical')]
    historical = source[source.index('async def revenue_historical') :]

    assert daily.count("state IN ('paid','completed')") >= 10
    assert hourly.count("state IN ('paid','completed')") >= 4
    assert historical.count("state IN ('paid','completed')") >= 2
    assert "state IN ('paid','draft')" not in daily
    assert "_get_non_sellable_return_cost" in daily
    assert '"non_sellable_return_cost"' in daily
    assert "transaction_type = 'return'" in source
    assert "disposition = 'non_sellable'" in source
    assert "order_cost * discount_ratio" not in source
    assert "order_profit = order_total - order_cost" in source
    assert "_get_non_sellable_return_cost_by_hour" in hourly
    assert '"non_sellable_return_cost"' in hourly
    assert "_get_non_sellable_return_cost_by_period" in historical


def test_revenue_views_apply_expired_cost_consistently():
    source = Path("api/module4_frontend/bff.py").read_text(encoding="utf-8")
    daily = source[source.index('async def revenue_daily') : source.index('async def revenue_hourly')]
    hourly = source[source.index('async def revenue_hourly') : source.index('async def revenue_historical')]
    historical = source[source.index('async def revenue_historical') :]

    assert daily.count("_get_expired_cost(") >= 3
    assert '"expired_cost"' in daily
    assert '"Closing adjustment"' in hourly
    assert '"expired_cost"' in hourly
    assert "_get_expired_cost_by_period" in historical
    assert "_get_order_adjustments_by_period" in historical
    assert '"order_adjustments"' in historical
    assert '"__order_adjustments__"' not in historical


def test_daily_revenue_exposes_promotion_loss_inputs():
    source = Path("api/module4_frontend/bff.py").read_text(encoding="utf-8")
    daily = source[source.index('async def revenue_daily') : source.index('async def revenue_hourly')]

    assert "_get_expired_product_breakdown(cur, date, expired_cost)" in daily
    assert "_get_sold_bread_sku_count(cur, date)" in daily
    assert '"expired_products": expired_products' in daily
    assert '"sold_bread_sku_count": sold_bread_sku_count' in daily


def test_revenue_category_uses_beverages_key_end_to_end():
    backend_source = Path("api/module4_frontend/bff.py").read_text(encoding="utf-8")
    daily = backend_source[
        backend_source.index("async def revenue_daily") :
        backend_source.index("async def revenue_hourly")
    ]
    frontend_source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )

    assert 'cat_data = {"Bread": 0, "Beverages": 0}' in daily
    assert 'else "Beverages"' in daily
    assert 'else "Coffee"' not in daily
    assert "m.category.Beverages" in frontend_source
    assert "m.category.Coffee" not in frontend_source


def test_expired_inventory_records_cost_snapshot():
    source = Path("api/freshness_service.py").read_text(encoding="utf-8")
    expiration = source[source.index('if new_freshness == "Expired"') :]

    assert 'select("product_name,material_cost")' in source
    assert '"unit_price": product_costs.get(' in expiration
    assert '"unit_price": 0' not in expiration


def test_future_production_batch_remains_fresh_before_production_date():
    assert (
        freshness_service.get_freshness(
            "2026-07-18 05:00:00",
            "2026-07-17",
        )
        == "Fresh"
    )


class _FreshnessResult:
    def __init__(self, data):
        self.data = data


class _FreshnessQuery:
    def __init__(self, tables, table):
        self.tables = tables
        self.table = table
        self.operation = "select"
        self.filters = []
        self.payload = None

    def select(self, _columns):
        self.operation = "select"
        return self

    def gt(self, field, value):
        self.filters.append(("gt", field, value))
        return self

    def neq(self, field, value):
        self.filters.append(("neq", field, value))
        return self

    def order(self, _field, desc=False):
        return self

    def insert(self, payload):
        self.operation = "insert"
        self.payload = dict(payload)
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = dict(payload)
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def eq(self, field, value):
        self.filters.append(("eq", field, value))
        return self

    def _matches(self, row):
        for operator, field, value in self.filters:
            current = row.get(field)
            if operator == "gt" and not (current or 0) > value:
                return False
            if operator == "neq" and current == value:
                return False
            if operator == "eq" and current != value:
                return False
        return True

    def execute(self):
        rows = self.tables.setdefault(self.table, [])
        if self.operation == "select":
            return _FreshnessResult(
                [dict(row) for row in rows if self._matches(row)]
            )
        if self.operation == "insert":
            rows.append(dict(self.payload))
            return _FreshnessResult([dict(self.payload)])
        if self.operation == "update":
            for row in rows:
                if self._matches(row):
                    row.update(self.payload)
            return _FreshnessResult([])
        if self.operation == "delete":
            self.tables[self.table] = [
                row for row in rows if not self._matches(row)
            ]
            return _FreshnessResult([])
        raise AssertionError(self.operation)


def test_freshness_aging_uses_remaining_inventory_only(monkeypatch):
    tables = {
        "batch_inventory": [
            {
                "batch_id": "sold-out",
                "product_name": "croissant",
                "quantity": 10,
                "quantity_remaining": 0,
                "production_time": "2026-07-11 06:00:00",
                "freshness_status": "Fresh",
            },
            {
                "batch_id": "remaining",
                "product_name": "croissant",
                "quantity": 10,
                "quantity_remaining": 2,
                "production_time": "2026-07-11 06:00:00",
                "freshness_status": "Fresh",
            },
        ],
        "products": [
            {"product_name": "croissant", "material_cost": 1.48},
        ],
        "inventory_transactions": [],
    }
    monkeypatch.setattr(freshness_service, "get_db", lambda: object())
    monkeypatch.setattr(
        freshness_service,
        "q",
        lambda _db, table: _FreshnessQuery(tables, table),
    )

    result = freshness_service.update_all_freshness("2026-07-14")

    assert result["expired_cleared"] == 1
    assert [row["batch_id"] for row in tables["batch_inventory"]] == [
        "sold-out",
        "remaining",
    ]
    expired_batch = tables["batch_inventory"][1]
    assert expired_batch["quantity"] == 0
    assert expired_batch["quantity_remaining"] == 0
    assert expired_batch["freshness_status"] == "Expired"
    assert expired_batch["tray_color"] == "black"
    assert tables["inventory_transactions"] == [
        {
            "transaction_type": "outflow",
            "batch_id": "remaining",
            "product_name": "croissant",
            "quantity": 2,
            "unit_price": 1.48,
            "discount_applied": 1.0,
            "freshness_status": "Expired",
            "disposition": "discarded",
            "reason": "day1_unsold",
            "performed_by": "freshness_service",
        }
    ]


def test_freshness_aging_never_moves_day1_inventory_back_to_fresh(monkeypatch):
    tables = {
        "batch_inventory": [
            {
                "batch_id": "day1-batch",
                "product_name": "baguette",
                "quantity": 1,
                "quantity_remaining": 1,
                "production_time": "2026-07-17 05:00:00",
                "freshness_status": "Day-1",
                "tray_color": "yellow",
            },
        ],
        "products": [],
        "inventory_transactions": [],
    }
    monkeypatch.setattr(freshness_service, "get_db", lambda: object())
    monkeypatch.setattr(
        freshness_service,
        "q",
        lambda _db, table: _FreshnessQuery(tables, table),
    )

    result = freshness_service.update_all_freshness("2026-07-17")

    assert result["updated"] == 0
    assert tables["batch_inventory"][0]["freshness_status"] == "Day-1"
    assert tables["batch_inventory"][0]["tray_color"] == "yellow"


def test_sellable_batches_filter_on_remaining_inventory(monkeypatch):
    tables = {
        "batch_inventory": [
            {
                "batch_id": "sold-out",
                "quantity": 10,
                "quantity_remaining": 0,
                "freshness_status": "Fresh",
            },
            {
                "batch_id": "available",
                "quantity": 10,
                "quantity_remaining": 3,
                "freshness_status": "Fresh",
            },
        ]
    }
    monkeypatch.setattr(freshness_service, "get_db", lambda: object())
    monkeypatch.setattr(
        freshness_service,
        "q",
        lambda _db, table: _FreshnessQuery(tables, table),
    )

    result = freshness_service.get_sellable_batches()

    assert [row["batch_id"] for row in result.data] == ["available"]


def test_daily_revenue_keeps_return_only_days_visible():
    source = Path("api/module4_frontend/bff.py").read_text(encoding="utf-8")
    daily = source[
        source.index("async def revenue_daily") : source.index("async def revenue_hourly")
    ]

    cost_lookup = daily.index("_get_non_sellable_return_cost(cur, date)")
    no_sales_branch = daily.index("if not row or not row[0]:")
    assert cost_lookup < no_sales_branch
    assert "non_sellable_return_cost <= 0" in daily


def test_obsolete_narrow_migration_name_is_removed():
    assert not Path("scripts/migrate_add_coffee_size.py").exists()


def test_obsolete_standalone_checkout_migration_and_compiled_residue_are_removed():
    assert not OBSOLETE_CHECKOUT_MIGRATION.exists()
    assert not tuple(
        Path("scripts/__pycache__").glob("migrate_checkout_schema*.pyc")
    )


def test_destructive_legacy_schema_entrypoints_are_removed():
    for path in DESTRUCTIVE_LEGACY_SCRIPTS:
        assert not path.exists(), f"destructive legacy schema entrypoint remains: {path}"
