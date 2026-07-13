from pathlib import Path
import re


CANONICAL_SCHEMA = Path("schema.sql")
EXPECTED_TABLES = frozenset(
    {
        "attendance_records",
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
        "receipts",
        "shift_schedule",
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
    assert len(tables) == 19
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
