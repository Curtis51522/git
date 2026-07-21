from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.module4_frontend.beverage_options import is_beverage
from kpi.attendance import build_schedule_windows, derive_attendance_status
from scripts.rebuild_guangzhou_sales_history import (
    BAKERY_SHARE_CAP,
    BASKET_MEAN_MAX,
    BASKET_MEAN_MIN,
    BASKET_SIZE_WEIGHTS,
    HISTORY_END,
    HISTORY_START,
    OPERATION_START,
    ORDER_RANGES,
    PAYMENT_RANGES,
    STORE_CLOSE,
    STORE_OPEN,
    TAKEAWAY_RANGE,
    TOP_THREE_MAX,
    TOP_THREE_MIN,
)
from scripts.reconcile_production_materials import (
    QUANTITY_PRECISION,
    calculate_expected_outflows,
)


OPERATION_END = date(2026, 7, 24)
HISTORY_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "guangzhou_rebuild_manifest.json"
)
FULL_MATERIAL_CHECK_DATES = (
    date(2026, 6, 24),
    date(2026, 7, 1),
    date(2026, 7, 8),
    date(2026, 7, 15),
    date(2026, 7, 22),
)
PAID_STATES = frozenset({"paid", "completed"})
NORMALIZED_STATES = PAID_STATES | frozenset({"refunded"})
DEFAULT_CHECKOUT_WASTAGE_PCT = Decimal("0.03")
MATERIAL_QUANTITY_TOLERANCE = Decimal("0.001")
ONE_ITEM_SHARE_RANGE = (
    BASKET_SIZE_WEIGHTS[1] / sum(BASKET_SIZE_WEIGHTS.values()) - 0.05,
    BASKET_SIZE_WEIGHTS[1] / sum(BASKET_SIZE_WEIGHTS.values()) + 0.05,
)
TWO_ITEM_SHARE_RANGE = (
    BASKET_SIZE_WEIGHTS[2] / sum(BASKET_SIZE_WEIGHTS.values()) - 0.05,
    BASKET_SIZE_WEIGHTS[2] / sum(BASKET_SIZE_WEIGHTS.values()) + 0.05,
)
PACKAGING_MATERIALS = frozenset(
    {"Box", "Cup Large", "Cup Regular", "Packaging Bag", "Packaging Box"}
)
MIN_DAILY_BAKERY_COVERAGE = 25
MIN_DAILY_BEVERAGE_COVERAGE = 12
MIN_AVERAGE_DAILY_BAKERY_COVERAGE = 28.0
MIN_AVERAGE_DAILY_BEVERAGE_COVERAGE = 14.0
MAX_AVERAGE_DAILY_BAKERY_COVERAGE = 29.5
MAX_AVERAGE_DAILY_BEVERAGE_COVERAGE = 14.8
MIN_BAKERY_PRODUCT_DAILY_AVERAGE = 2.0
MIN_BEVERAGE_PRODUCT_DAILY_AVERAGE = 3.0
BAKERY_TOP_SEVEN_RANGE = (0.55, 0.65)

HISTORICAL_TABLES = (
    "orders",
    "order_items",
    "payments",
    "products",
)
OPERATIONAL_TABLES = (
    "attendance_records",
    "batch_inventory",
    "business_events",
    "inventory_transactions",
    "material_transactions",
    "material_wastage_log",
    "order_items",
    "orders",
    "payments",
    "product_recipes",
    "products",
    "raw_materials",
    "receipts",
    "recommendation_events",
    "shift_schedule",
)

TABLE_QUERIES = {
    "attendance_records": "SELECT * FROM attendance_records",
    "batch_inventory": "SELECT * FROM batch_inventory",
    "business_events": "SELECT * FROM business_events",
    "inventory_transactions": "SELECT * FROM inventory_transactions",
    "material_transactions": "SELECT * FROM material_transactions",
    "material_wastage_log": "SELECT * FROM material_wastage_log",
    "order_items": "SELECT * FROM order_items",
    "orders": "SELECT * FROM orders",
    "payments": "SELECT * FROM payments",
    "product_recipes": "SELECT * FROM product_recipes",
    "products": "SELECT * FROM products",
    "raw_materials": "SELECT * FROM raw_materials",
    "receipts": "SELECT * FROM receipts",
    "recommendation_events": "SELECT * FROM recommendation_events",
    "shift_schedule": "SELECT * FROM shift_schedule",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    observed: object
    expected: str
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class BasketMetrics:
    order_count: int
    item_count: int
    items_per_order: float
    average_order_value: float
    one_item_share: float
    two_item_share: float


@dataclass
class DataSnapshot:
    available_tables: frozenset[str]
    attendance_records: tuple[dict, ...] = field(default_factory=tuple)
    batch_inventory: tuple[dict, ...] = field(default_factory=tuple)
    business_events: tuple[dict, ...] = field(default_factory=tuple)
    inventory_transactions: tuple[dict, ...] = field(default_factory=tuple)
    material_transactions: tuple[dict, ...] = field(default_factory=tuple)
    material_wastage_log: tuple[dict, ...] = field(default_factory=tuple)
    order_items: tuple[dict, ...] = field(default_factory=tuple)
    orders: tuple[dict, ...] = field(default_factory=tuple)
    payments: tuple[dict, ...] = field(default_factory=tuple)
    product_recipes: tuple[dict, ...] = field(default_factory=tuple)
    products: tuple[dict, ...] = field(default_factory=tuple)
    raw_materials: tuple[dict, ...] = field(default_factory=tuple)
    receipts: tuple[dict, ...] = field(default_factory=tuple)
    recommendation_events: tuple[dict, ...] = field(default_factory=tuple)
    shift_schedule: tuple[dict, ...] = field(default_factory=tuple)


def _result(name, observed, expected, failures=()) -> CheckResult:
    normalized = tuple(str(failure) for failure in failures)
    return CheckResult(name, not normalized, observed, expected, normalized)


def _decimal(value) -> Decimal:
    if value is None or value == "":
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _receipt_items(receipt) -> tuple[Mapping, ...]:
    raw_items = receipt.get("items") if receipt else None
    if isinstance(raw_items, str):
        try:
            raw_items = json.loads(raw_items)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ()
    if not isinstance(raw_items, list):
        return ()
    return tuple(item for item in raw_items if isinstance(item, Mapping))


def _packaging_costs_by_reference(material_transactions, raw_materials):
    unit_prices = {
        str(row.get("material_name") or ""): _decimal(row.get("unit_price"))
        for row in raw_materials
    }
    costs = defaultdict(lambda: Decimal("0"))
    for row in material_transactions:
        material_name = str(row.get("material_name") or "")
        if (
            material_name not in {"Packaging Bag", "Packaging Box"}
            or str(row.get("transaction_type") or "").lower() != "outflow"
        ):
            continue
        reference = str(row.get("reference") or "")
        costs[reference] += _decimal(row.get("quantity")) * unit_prices.get(
            material_name, Decimal("0")
        )
    return costs


def _as_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _as_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if value is None or value == "":
        return None
    text = str(value).strip().replace("T", " ")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _as_time(value) -> time | None:
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, time):
        return value
    if isinstance(value, timedelta):
        seconds = int(value.total_seconds()) % (24 * 60 * 60)
        return time(seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    if value is None or value == "":
        return None
    try:
        return time.fromisoformat(str(value))
    except ValueError:
        return None


def _date_range(start: date, end: date) -> tuple[date, ...]:
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))


def _paid_orders(rows: Iterable[Mapping]) -> list[Mapping]:
    return [row for row in rows if str(row.get("state") or "").lower() in PAID_STATES]


def _normalized_orders(rows: Iterable[Mapping]) -> list[Mapping]:
    return [
        row
        for row in rows
        if str(row.get("state") or "").lower() in NORMALIZED_STATES
    ]


def _rows_for_period(rows, column, start, end):
    selected = []
    for row in rows:
        value = _as_date(row.get(column))
        if value is not None and start <= value <= end:
            selected.append(row)
    return selected


def _category_map(products: Iterable[Mapping]) -> dict[str, str]:
    return {
        str(row.get("product_name") or ""): str(row.get("category") or "").lower()
        for row in products
        if row.get("product_name")
    }


def _is_beverage_product(product_name: str, categories: Mapping[str, str]) -> bool:
    category = categories.get(product_name, "")
    return category in {"beverage", "drink", "coffee"} or is_beverage(product_name)


def summarize_orders(rows: Iterable[Mapping]) -> BasketMetrics:
    baskets: dict[str, int] = defaultdict(int)
    totals: dict[str, Decimal] = {}
    for index, row in enumerate(rows):
        ticket_id = str(row.get("ticket_id") or row.get("id") or index)
        baskets[ticket_id] += int(row.get("quantity", row.get("item_count", 0)) or 0)
        totals.setdefault(ticket_id, _decimal(row.get("total_amount")))
    order_count = len(baskets)
    item_count = sum(baskets.values())
    revenue = sum(totals.values(), Decimal("0"))
    return BasketMetrics(
        order_count=order_count,
        item_count=item_count,
        items_per_order=round(item_count / order_count, 6) if order_count else 0.0,
        average_order_value=round(float(revenue / order_count), 6) if order_count else 0.0,
        one_item_share=(sum(value == 1 for value in baskets.values()) / order_count if order_count else 0.0),
        two_item_share=(sum(value == 2 for value in baskets.values()) / order_count if order_count else 0.0),
    )


def validate_inventory(batch_rows, raw_material_rows=()) -> CheckResult:
    failures = []
    for row in batch_rows:
        batch_id = str(row.get("batch_id") or "<unknown>")
        if _decimal(row.get("quantity_remaining")) < 0:
            failures.append(f"Negative finished stock in batch {batch_id}")
    for row in raw_material_rows:
        material_name = str(row.get("material_name") or "<unknown>")
        if _decimal(row.get("stock_quantity")) < 0:
            failures.append(f"Negative raw stock for {material_name}")
    observed = {"batches": len(batch_rows), "raw_materials": len(raw_material_rows)}
    return _result(
        "operational_inventory_nonnegative",
        observed,
        "Raw and finished stock are nonnegative",
        failures,
    )


def validate_history_range(
    orders,
    start: date = HISTORY_START,
    end: date = HISTORY_END,
    cutoff: date = OPERATION_START,
    expected_order_count: int | None = None,
    observed_item_count: int | None = None,
    expected_item_count: int | None = None,
) -> CheckResult:
    orders = list(orders)
    dates = sorted({_as_date(row.get("order_date")) for row in orders} - {None})
    failures = []
    if expected_order_count is not None and len(orders) != expected_order_count:
        failures.append(
            f"Historical GZ order count is {len(orders)}, expected manifest count {expected_order_count}"
        )
    if (
        expected_item_count is not None
        and observed_item_count != expected_item_count
    ):
        failures.append(
            f"Historical GZ item row count is {observed_item_count if observed_item_count is not None else '<unknown>'}, expected manifest output count {expected_item_count}"
        )
    if not dates:
        failures.append("Historical orders are empty")
    else:
        if dates[0] != start:
            failures.append(f"Historical start date is {dates[0].isoformat()}, expected {start.isoformat()}")
        if dates[-1] != end:
            failures.append(f"Historical end date is {dates[-1].isoformat()}, expected {end.isoformat()}")
        for order_date in dates:
            if order_date >= cutoff:
                failures.append(
                    f"Historical order on or after operation cutoff: {order_date.isoformat()}"
                )
        present = set(dates)
        for expected_date in _date_range(start, end):
            if expected_date not in present:
                failures.append(f"Missing historical trading date {expected_date.isoformat()}")
    observed = {
        "start": dates[0].isoformat() if dates else None,
        "end": dates[-1].isoformat() if dates else None,
        "trading_dates": len(dates),
        "orders": len(orders),
        "item_rows": observed_item_count,
    }
    count_text = (
        f" with {expected_order_count} GZ orders"
        if expected_order_count is not None
        else ""
    )
    return _result(
        "historical_date_range",
        observed,
        f"Continuous history from {start.isoformat()} through {end.isoformat()} before {cutoff.isoformat()}{count_text}",
        failures,
    )


def validate_history_manifest(manifest) -> CheckResult:
    failures = []
    if not isinstance(manifest, Mapping):
        manifest = {}
        failures.append("Historical manifest root is not an object")
    date_range = manifest.get("date_range")
    if not isinstance(date_range, Mapping):
        date_range = {}
        failures.append("Historical manifest date_range is not an object")
    manifest_start = _as_date(date_range.get("start"))
    manifest_end = _as_date(date_range.get("end"))
    if manifest_start != HISTORY_START:
        failures.append(
            f"Historical manifest start is {manifest_start.isoformat() if manifest_start else '<invalid>'}, expected {HISTORY_START.isoformat()}"
        )
    if manifest_end != HISTORY_END:
        failures.append(
            f"Historical manifest end is {manifest_end.isoformat() if manifest_end else '<invalid>'}, expected {HISTORY_END.isoformat()}"
        )
    raw_order_count = manifest.get("order_count")
    try:
        order_count = int(raw_order_count)
    except (TypeError, ValueError):
        order_count = None
    if order_count is None or order_count <= 0:
        failures.append(
            f"Historical manifest order_count is {raw_order_count!r}, expected a positive integer"
        )
    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, Mapping):
        row_counts = {}
        failures.append("Historical manifest row_counts is not an object")
    raw_item_count = row_counts.get("output")
    try:
        item_count = int(raw_item_count)
    except (TypeError, ValueError):
        item_count = None
    if item_count is None or item_count < 0:
        failures.append(
            f"Historical manifest output row count is {raw_item_count!r}, expected a nonnegative integer"
        )
    return _result(
        "historical_manifest_contract",
        {
            "start": manifest_start.isoformat() if manifest_start else None,
            "end": manifest_end.isoformat() if manifest_end else None,
            "order_count": order_count,
            "item_count": item_count,
        },
        "Task 3 manifest range and reconstructed order/item counts are valid",
        failures,
    )


def validate_normalized_transactions(
    orders,
    items,
    payments,
    receipts=None,
    *,
    name="historical_normalized_relationships",
    source_orders=None,
    scope_start: date | None = None,
    scope_end: date | None = None,
    material_transactions=(),
    raw_materials=(),
) -> CheckResult:
    receipts = None if receipts is None else list(receipts)
    source_orders = list(orders if source_orders is None else source_orders)
    failures = []
    orders_by_id = {}
    ticket_ids = set()
    for order in orders:
        order_id = order.get("id")
        ticket_id = str(order.get("ticket_id") or "").strip()
        if order_id in orders_by_id:
            failures.append(f"Duplicate order id {order_id}")
        orders_by_id[order_id] = order
        if not ticket_id:
            failures.append(f"Order {order_id} has empty ticket_id")
        elif ticket_id in ticket_ids:
            failures.append(f"Duplicate ticket_id {ticket_id}")
        ticket_ids.add(ticket_id)

    source_orders_by_id = {row.get("id"): row for row in source_orders}
    packaging_costs = _packaging_costs_by_reference(
        material_transactions, raw_materials
    )

    def date_is_in_scope(value) -> bool:
        row_date = _as_date(value)
        if scope_start is None or scope_end is None:
            return True
        return row_date is not None and scope_start <= row_date <= scope_end

    scope_label = name.removesuffix("_normalized_relationships")
    items_by_order = defaultdict(list)
    for item in items:
        order_id = item.get("order_id")
        if order_id not in source_orders_by_id:
            failures.append(f"Item {item.get('id')} references missing order {order_id}")
        elif order_id in orders_by_id:
            items_by_order[order_id].append(item)
        elif date_is_in_scope(source_orders_by_id[order_id].get("order_date")):
            failures.append(
                f"Item {item.get('id')} belongs to non-normalized order {order_id} in {scope_label} scope"
            )
    payments_by_order = defaultdict(list)
    for payment in payments:
        order_id = payment.get("order_id")
        if order_id not in source_orders_by_id:
            failures.append(
                f"Payment {payment.get('id')} references missing order {order_id}"
            )
        elif order_id in orders_by_id:
            payments_by_order[order_id].append(payment)
        elif date_is_in_scope(payment.get("payment_date")):
            source_order = source_orders_by_id[order_id]
            if date_is_in_scope(source_order.get("order_date")):
                failures.append(
                    f"Payment {payment.get('id')} belongs to non-normalized order {order_id} in {scope_label} scope"
                )
            else:
                payment_date = _as_date(payment.get("payment_date"))
                failures.append(
                    f"Payment {payment.get('id')} date {payment_date.isoformat() if payment_date else '<invalid>'} belongs to order {order_id} outside {scope_label} scope"
                )
    receipts_by_ticket = defaultdict(list)
    for receipt in receipts or []:
        if not date_is_in_scope(receipt.get("created_at")):
            continue
        receipt_id = str(receipt.get("receipt_id") or "")
        if receipt_id not in ticket_ids:
            failures.append(
                f"Receipt {receipt_id or '<empty>'} has no normalized order in {scope_label} scope"
            )
        else:
            receipts_by_ticket[receipt_id].append(receipt)

    for order_id, order in orders_by_id.items():
        order_items = items_by_order.get(order_id, [])
        if not order_items:
            failures.append(f"Order {order_id} has no item rows")
            continue
        ticket_id = str(order.get("ticket_id") or "")
        order_receipts = receipts_by_ticket.get(ticket_id, [])
        receipt = order_receipts[0] if len(order_receipts) == 1 else None
        packaging_cost = packaging_costs.get(ticket_id, Decimal("0"))
        quantity = sum(int(item.get("quantity") or 0) for item in order_items)
        subtotal = sum(
            (_decimal(item.get("unit_price")) * int(item.get("quantity") or 0) for item in order_items),
            Decimal("0"),
        )
        line_total = sum((_decimal(item.get("line_total")) for item in order_items), Decimal("0"))
        line_profit = sum((_decimal(item.get("line_profit")) for item in order_items), Decimal("0"))
        if int(order.get("item_count") or 0) != quantity:
            failures.append(
                f"Order {order_id} item_count {order.get('item_count')} does not equal item quantity {quantity}"
            )
        if _decimal(order.get("subtotal")) != subtotal:
            failures.append(
                f"Order {order_id} subtotal {order.get('subtotal')} does not equal pre-discount item total {subtotal}"
            )
        expected_total = line_total
        if _decimal(order.get("total_amount")) != expected_total:
            failures.append(
                f"Order {order_id} total_amount {order.get('total_amount')} does not equal item line total {expected_total}"
            )
        expected_discount = subtotal - line_total
        if _decimal(order.get("discount_total")) != expected_discount:
            failures.append(
                f"Order {order_id} discount_total {order.get('discount_total')} does not equal {expected_discount}"
            )
        expected_profit = line_profit - packaging_cost
        profit_difference = abs(
            _decimal(order.get("total_profit")) - expected_profit
        )
        profit_tolerance = Decimal("0.01") * max(len(order_items), 1)
        if profit_difference > profit_tolerance:
            failures.append(
                f"Order {order_id} total_profit {order.get('total_profit')} differs from item profit after packaging cost {expected_profit} by more than {profit_tolerance}"
            )
        order_payments = payments_by_order.get(order_id, [])
        if len(order_payments) != 1:
            failures.append(f"Order {order_id} has {len(order_payments)} payment rows")
        else:
            payment = order_payments[0]
            if _decimal(payment.get("amount")) != _decimal(order.get("total_amount")):
                failures.append(
                    f"Order {order_id} payment amount {payment.get('amount')} does not equal total_amount {order.get('total_amount')}"
                )
            payment_date = _as_date(payment.get("payment_date"))
            order_date = _as_date(order.get("order_date"))
            if payment_date != order_date:
                failures.append(
                    f"Order {order_id} payment date {payment_date.isoformat() if payment_date else '<invalid>'} does not equal order date {order_date.isoformat() if order_date else '<invalid>'}"
                )
        if receipts is not None:
            if len(order_receipts) != 1:
                failures.append(f"Order {order_id} has {len(order_receipts)} receipt rows")
            else:
                if _decimal(receipt.get("total")) != _decimal(order.get("total_amount")):
                    failures.append(
                        f"Order {order_id} receipt total {receipt.get('total')} does not equal total_amount {order.get('total_amount')}"
                    )
                receipt_date = _as_date(receipt.get("created_at"))
                order_date = _as_date(order.get("order_date"))
                if receipt_date != order_date:
                    failures.append(
                        f"Order {order_id} receipt date {receipt_date.isoformat() if receipt_date else '<invalid>'} does not equal order date {order_date.isoformat() if order_date else '<invalid>'}"
                    )

    observed = {
        "orders": len(orders),
        "items": sum(len(rows) for rows in items_by_order.values()),
        "payments": sum(len(rows) for rows in payments_by_order.values()),
        "receipts": sum(len(rows) for rows in receipts_by_ticket.values()),
    }
    receipt_text = ", and receipt" if receipts is not None else ""
    return _result(
        name,
        observed,
        f"Every normalized order reconciles to item, payment{receipt_text} evidence",
        failures,
    )


def validate_daily_order_ranges(orders) -> CheckResult:
    counts = Counter(_as_date(row.get("order_date")) for row in orders)
    counts.pop(None, None)
    failures = []
    for order_date, count in sorted(counts.items()):
        day_type = "weekend" if order_date.weekday() >= 5 else "friday" if order_date.weekday() == 4 else "weekday"
        low, high = ORDER_RANGES[day_type]
        if not low <= count <= high:
            failures.append(
                f"Historical orders on {order_date.isoformat()} are {count}, expected {low}-{high} for {day_type}"
            )
    if not counts:
        failures.append("Historical daily order counts are empty")
    observed = {day.isoformat(): count for day, count in sorted(counts.items())}
    return _result(
        "historical_daily_order_ranges",
        observed,
        "Daily order counts satisfy approved weekday, Friday, and weekend ranges",
        failures,
    )


def validate_basket_profile(orders) -> CheckResult:
    metrics = summarize_orders(
        {
            "ticket_id": row.get("ticket_id") or row.get("id"),
            "quantity": row.get("item_count"),
            "total_amount": row.get("total_amount"),
        }
        for row in orders
    )
    failures = []
    if metrics.order_count == 0:
        failures.append("Historical basket profile is empty")
    if not BASKET_MEAN_MIN <= metrics.items_per_order <= BASKET_MEAN_MAX:
        failures.append(
            f"Mean items per order {metrics.items_per_order:.6f} outside {BASKET_MEAN_MIN}-{BASKET_MEAN_MAX}"
        )
    if not ONE_ITEM_SHARE_RANGE[0] <= metrics.one_item_share <= ONE_ITEM_SHARE_RANGE[1]:
        failures.append(
            f"One-item share {metrics.one_item_share:.6f} outside {ONE_ITEM_SHARE_RANGE[0]:.2f}-{ONE_ITEM_SHARE_RANGE[1]:.2f}"
        )
    if not TWO_ITEM_SHARE_RANGE[0] <= metrics.two_item_share <= TWO_ITEM_SHARE_RANGE[1]:
        failures.append(
            f"Two-item share {metrics.two_item_share:.6f} outside {TWO_ITEM_SHARE_RANGE[0]:.2f}-{TWO_ITEM_SHARE_RANGE[1]:.2f}"
        )
    return _result(
        "historical_basket_profile",
        asdict(metrics),
        "Mean basket 1.7-1.9 with approved one-item and two-item shares",
        failures,
    )


def validate_sales_boundary(orders) -> CheckResult:
    failures = []
    parsed_times = []
    for row in orders:
        order_id = row.get("id", row.get("ticket_id"))
        order_time = _as_time(row.get("order_time"))
        if order_time is None:
            failures.append(f"Order {order_id} has invalid order_time")
        elif not STORE_OPEN <= order_time <= STORE_CLOSE:
            failures.append(f"Order {order_id} time {order_time.isoformat()} is outside 06:00-19:00")
        else:
            parsed_times.append(order_time)
    observed = {
        "minimum": min(parsed_times).isoformat() if parsed_times else None,
        "maximum": max(parsed_times).isoformat() if parsed_times else None,
    }
    return _result(
        "historical_sales_boundary",
        observed,
        "All sales occur from 06:00 through 19:00 inclusive",
        failures,
    )


def validate_service_payment_shares(orders, payments) -> CheckResult:
    failures = []
    service_counts = Counter(str(row.get("dine_type") or "").lower() for row in orders)
    order_total = sum(service_counts.values())
    delivery = sum(count for key, count in service_counts.items() if "delivery" in key)
    takeaway = sum(count for key, count in service_counts.items() if key in {"takeaway", "take_away", "take-away"})
    takeaway_share = takeaway / order_total if order_total else 0.0
    if delivery:
        failures.append(f"Delivery orders found: {delivery}")
    if not TAKEAWAY_RANGE[0] <= takeaway_share <= TAKEAWAY_RANGE[1]:
        failures.append(
            f"Takeaway share {takeaway_share:.6f} outside {TAKEAWAY_RANGE[0]}-{TAKEAWAY_RANGE[1]}"
        )

    payment_counts = Counter(str(row.get("payment_method") or "").lower() for row in payments)
    payment_total = sum(payment_counts.values())
    payment_shares = {}
    for method, bounds in PAYMENT_RANGES.items():
        share = payment_counts.get(method, 0) / payment_total if payment_total else 0.0
        payment_shares[method] = share
        if not bounds[0] <= share <= bounds[1]:
            failures.append(
                f"{method.upper()} payment share {share:.6f} outside {bounds[0]}-{bounds[1]}"
            )
    observed = {
        "delivery_orders": delivery,
        "takeaway_share": takeaway_share,
        "payment_shares": payment_shares,
    }
    return _result(
        "historical_service_payment_shares",
        observed,
        "No delivery, takeaway 0.75-0.85, and approved QR/card/cash shares",
        failures,
    )


def validate_product_coverage(items, products) -> CheckResult:
    sold = {str(row.get("product_name") or "") for row in items if int(row.get("quantity") or 0) > 0}
    catalog = {str(row.get("product_name") or "") for row in products if row.get("product_name")}
    categories = _category_map(products)
    failures = [f"Historical product has no sales: {name}" for name in sorted(catalog - sold)]
    sold_bakery = any(not _is_beverage_product(name, categories) for name in sold)
    sold_beverage = any(_is_beverage_product(name, categories) for name in sold)
    if not sold_bakery:
        failures.append("Historical sales have no bakery product coverage")
    if not sold_beverage:
        failures.append("Historical sales have no beverage product coverage")
    observed = {"catalog_products": len(catalog), "sold_products": len(sold & catalog)}
    return _result(
        "historical_product_coverage",
        observed,
        "Every catalog product is sold with bakery and beverage coverage",
        failures,
    )


def validate_bakery_concentration(items, products) -> CheckResult:
    categories = _category_map(products)
    units = Counter()
    for row in items:
        product_name = str(row.get("product_name") or "")
        if product_name and not _is_beverage_product(product_name, categories):
            units[product_name] += int(row.get("quantity") or 0)
    total = sum(units.values())
    failures = []
    shares = {name: quantity / total for name, quantity in units.items()} if total else {}
    for name, share in sorted(shares.items()):
        if share > BAKERY_SHARE_CAP:
            failures.append(f"Bakery product {name} share {share:.6f} exceeds {BAKERY_SHARE_CAP}")
    top3_share = sum(quantity for _, quantity in units.most_common(3)) / total if total else 0.0
    if not total:
        failures.append("Historical bakery concentration has no units")
    elif not TOP_THREE_MIN <= top3_share <= TOP_THREE_MAX:
        failures.append(
            f"Top-three bakery share {top3_share:.6f} outside {TOP_THREE_MIN}-{TOP_THREE_MAX}"
        )
    observed = {
        "maximum_product_share": max(shares.values(), default=0.0),
        "top_three_share": top3_share,
        "bakery_units": total,
    }
    return _result(
        "historical_bakery_concentration",
        observed,
        "One bakery product is at most 0.15 and top three total 0.35-0.45",
        failures,
    )


def validate_historical_product_mix(orders, items, products) -> CheckResult:
    order_dates = {
        row.get("id"): _as_date(row.get("order_date"))
        for row in orders
        if row.get("id") is not None
    }
    categories = _category_map(products)
    bakery_products = sorted(
        name for name in categories if not _is_beverage_product(name, categories)
    )
    beverage_products = sorted(
        name for name in categories if _is_beverage_product(name, categories)
    )
    daily_units = defaultdict(Counter)
    annual_units = Counter()
    for row in items:
        order_date = order_dates.get(row.get("order_id"))
        product_name = str(row.get("product_name") or "")
        quantity = int(row.get("quantity") or 0)
        if order_date is None or product_name not in categories or quantity <= 0:
            continue
        daily_units[order_date][product_name] += quantity
        annual_units[product_name] += quantity

    failures = []
    if not daily_units:
        failures.append("Historical product-mix evidence is empty")
    bakery_coverage = [
        sum(units[name] > 0 for name in bakery_products)
        for units in daily_units.values()
    ]
    beverage_coverage = [
        sum(units[name] > 0 for name in beverage_products)
        for units in daily_units.values()
    ]
    minimum_bakery_coverage = min(bakery_coverage, default=0)
    minimum_beverage_coverage = min(beverage_coverage, default=0)
    average_bakery_coverage = (
        sum(bakery_coverage) / len(bakery_coverage) if bakery_coverage else 0.0
    )
    average_beverage_coverage = (
        sum(beverage_coverage) / len(beverage_coverage) if beverage_coverage else 0.0
    )
    if minimum_bakery_coverage < MIN_DAILY_BAKERY_COVERAGE:
        failures.append(
            f"Minimum daily bakery coverage {minimum_bakery_coverage} is below {MIN_DAILY_BAKERY_COVERAGE}"
        )
    if minimum_beverage_coverage < MIN_DAILY_BEVERAGE_COVERAGE:
        failures.append(
            f"Minimum daily beverage coverage {minimum_beverage_coverage} is below {MIN_DAILY_BEVERAGE_COVERAGE}"
        )
    if average_bakery_coverage < MIN_AVERAGE_DAILY_BAKERY_COVERAGE:
        failures.append(
            f"Average daily bakery coverage {average_bakery_coverage:.3f} is below {MIN_AVERAGE_DAILY_BAKERY_COVERAGE:.1f}"
        )
    if average_beverage_coverage < MIN_AVERAGE_DAILY_BEVERAGE_COVERAGE:
        failures.append(
            f"Average daily beverage coverage {average_beverage_coverage:.3f} is below {MIN_AVERAGE_DAILY_BEVERAGE_COVERAGE:.1f}"
        )
    if average_bakery_coverage > MAX_AVERAGE_DAILY_BAKERY_COVERAGE:
        failures.append(
            f"Average daily bakery coverage {average_bakery_coverage:.3f} exceeds {MAX_AVERAGE_DAILY_BAKERY_COVERAGE:.1f}"
        )
    if average_beverage_coverage > MAX_AVERAGE_DAILY_BEVERAGE_COVERAGE:
        failures.append(
            f"Average daily beverage coverage {average_beverage_coverage:.3f} exceeds {MAX_AVERAGE_DAILY_BEVERAGE_COVERAGE:.1f}"
        )

    day_count = len(daily_units)
    bakery_daily_averages = {
        name: annual_units[name] / day_count if day_count else 0.0
        for name in bakery_products
    }
    beverage_daily_averages = {
        name: annual_units[name] / day_count if day_count else 0.0
        for name in beverage_products
    }
    minimum_bakery_daily_average = min(bakery_daily_averages.values(), default=0.0)
    minimum_beverage_daily_average = min(beverage_daily_averages.values(), default=0.0)
    if minimum_bakery_daily_average < MIN_BAKERY_PRODUCT_DAILY_AVERAGE:
        failures.append(
            f"Lowest bakery product daily average {minimum_bakery_daily_average:.3f} is below {MIN_BAKERY_PRODUCT_DAILY_AVERAGE:.1f}"
        )
    if minimum_beverage_daily_average < MIN_BEVERAGE_PRODUCT_DAILY_AVERAGE:
        failures.append(
            f"Lowest beverage product daily average {minimum_beverage_daily_average:.3f} is below {MIN_BEVERAGE_PRODUCT_DAILY_AVERAGE:.1f}"
        )

    bakery_total = sum(annual_units[name] for name in bakery_products)
    top_seven_share = (
        sum(sorted((annual_units[name] for name in bakery_products), reverse=True)[:7])
        / bakery_total
        if bakery_total
        else 0.0
    )
    if not BAKERY_TOP_SEVEN_RANGE[0] <= top_seven_share <= BAKERY_TOP_SEVEN_RANGE[1]:
        failures.append(
            f"Top-seven bakery share {top_seven_share:.6f} outside {BAKERY_TOP_SEVEN_RANGE[0]}-{BAKERY_TOP_SEVEN_RANGE[1]}"
        )

    observed = {
        "minimum_daily_bakery_coverage": minimum_bakery_coverage,
        "minimum_daily_beverage_coverage": minimum_beverage_coverage,
        "average_daily_bakery_coverage": average_bakery_coverage,
        "average_daily_beverage_coverage": average_beverage_coverage,
        "minimum_bakery_product_daily_average": minimum_bakery_daily_average,
        "minimum_beverage_product_daily_average": minimum_beverage_daily_average,
        "bakery_top_seven_share": top_seven_share,
    }
    return _result(
        "historical_product_mix_depth",
        observed,
        "Daily catalog coverage, product floors, and top-seven concentration satisfy the active-year contract",
        failures,
    )


def validate_operational_completeness(
    orders,
    items,
    payments,
    receipts,
    *,
    start: date = OPERATION_START,
    end: date = OPERATION_END,
) -> CheckResult:
    paid = _paid_orders(orders)
    orders_by_id = {row.get("id"): row for row in paid}
    orders_by_ticket = {str(row.get("ticket_id") or ""): row for row in paid}
    order_dates = Counter(_as_date(row.get("order_date")) for row in paid)
    item_dates = Counter(
        _as_date(orders_by_id[row.get("order_id")].get("order_date"))
        for row in items
        if row.get("order_id") in orders_by_id
    )
    payment_dates = Counter(
        _as_date(orders_by_id[row.get("order_id")].get("order_date"))
        for row in payments
        if row.get("order_id") in orders_by_id
    )
    receipt_dates = Counter(
        _as_date(row.get("created_at"))
        for row in receipts
        if str(row.get("receipt_id") or "") in orders_by_ticket
        and _as_date(row.get("created_at"))
        == _as_date(orders_by_ticket[str(row.get("receipt_id") or "")].get("order_date"))
    )
    failures = []
    for operation_date in _date_range(start, end):
        for label, counts in (
            ("orders", order_dates),
            ("order items", item_dates),
            ("payments", payment_dates),
            ("receipts", receipt_dates),
        ):
            if not counts.get(operation_date):
                failures.append(f"Missing operational {label} on {operation_date.isoformat()}")
    observed = {
        operation_date.isoformat(): {
            "orders": order_dates.get(operation_date, 0),
            "items": item_dates.get(operation_date, 0),
            "payments": payment_dates.get(operation_date, 0),
            "receipts": receipt_dates.get(operation_date, 0),
        }
        for operation_date in _date_range(start, end)
    }
    return _result(
        "operational_daily_completeness",
        observed,
        "Every operational date has order, item, payment, and receipt evidence",
        failures,
    )


def validate_operational_product_coverage(
    orders,
    items,
    products,
    *,
    start: date | None = None,
    end: date | None = None,
) -> CheckResult:
    order_dates = {
        row.get("id"): _as_date(row.get("order_date"))
        for row in orders
        if row.get("id") is not None
    }
    categories = _category_map(products)
    bakery_products = {
        name for name in categories if not _is_beverage_product(name, categories)
    }
    beverage_products = {
        name for name in categories if _is_beverage_product(name, categories)
    }
    daily_products = defaultdict(set)
    for row in items:
        operation_date = order_dates.get(row.get("order_id"))
        product_name = str(row.get("product_name") or "")
        if (
            operation_date is not None
            and product_name in categories
            and int(row.get("quantity") or 0) > 0
        ):
            daily_products[operation_date].add(product_name)

    operation_dates = sorted(set(order_dates.values()) - {None})
    bakery_coverage = [
        len(daily_products[operation_date] & bakery_products)
        for operation_date in operation_dates
    ]
    beverage_coverage = [
        len(daily_products[operation_date] & beverage_products)
        for operation_date in operation_dates
    ]
    minimum_bakery = min(bakery_coverage, default=0)
    minimum_beverage = min(beverage_coverage, default=0)
    average_bakery = (
        sum(bakery_coverage) / len(bakery_coverage) if bakery_coverage else 0.0
    )
    average_beverage = (
        sum(beverage_coverage) / len(beverage_coverage)
        if beverage_coverage
        else 0.0
    )

    weekly_recurrence_evaluated = not (
        start is not None and end is not None and (end - start).days < 6
    )
    weekly_missing = {}
    if weekly_recurrence_evaluated:
        weekly_products = defaultdict(set)
        for operation_date, names in daily_products.items():
            weekly_products[operation_date.isocalendar()[:2]].update(names)
        catalog = bakery_products | beverage_products
        for week, sold_products in sorted(weekly_products.items()):
            missing = sorted(catalog - sold_products)
            if missing:
                weekly_missing[f"{week[0]}-W{week[1]:02d}"] = missing

    failures = []
    if minimum_bakery < MIN_DAILY_BAKERY_COVERAGE:
        failures.append(
            f"Minimum operational daily bakery coverage {minimum_bakery} "
            f"is below {MIN_DAILY_BAKERY_COVERAGE}"
        )
    if minimum_beverage < MIN_DAILY_BEVERAGE_COVERAGE:
        failures.append(
            f"Minimum operational daily beverage coverage {minimum_beverage} "
            f"is below {MIN_DAILY_BEVERAGE_COVERAGE}"
        )
    if average_bakery < MIN_AVERAGE_DAILY_BAKERY_COVERAGE:
        failures.append(
            f"Average operational daily bakery coverage {average_bakery:.3f} "
            f"is below {MIN_AVERAGE_DAILY_BAKERY_COVERAGE:.1f}"
        )
    if average_beverage < MIN_AVERAGE_DAILY_BEVERAGE_COVERAGE:
        failures.append(
            f"Average operational daily beverage coverage {average_beverage:.3f} "
            f"is below {MIN_AVERAGE_DAILY_BEVERAGE_COVERAGE:.1f}"
        )
    for week, missing in weekly_missing.items():
        failures.append(
            f"Operational product recurrence missing {len(missing)} catalog "
            f"products in {week}: {', '.join(missing[:5])}"
        )

    observed = {
        "days": len(operation_dates),
        "minimum_daily_bakery_coverage": minimum_bakery,
        "minimum_daily_beverage_coverage": minimum_beverage,
        "average_daily_bakery_coverage": average_bakery,
        "average_daily_beverage_coverage": average_beverage,
        "weekly_recurrence_evaluated": weekly_recurrence_evaluated,
        "weeks_with_missing_products": weekly_missing,
    }
    return _result(
        "operational_product_coverage",
        observed,
        "Daily bakery/beverage depth is realistic and every catalog product recurs weekly",
        failures,
    )


def validate_business_event_evidence(
    events,
    receipts,
    *,
    start: date = OPERATION_START,
    end: date = OPERATION_END,
) -> CheckResult:
    required_types = set()
    if start <= date(2026, 7, 5) and end >= date(2026, 7, 3):
        required_types.add("new_product_launch")
    if start <= date(2026, 7, 19) and end >= date(2026, 7, 17):
        required_types.add("competitor_activity")

    selected_events = []
    for event in events:
        event_start = _as_date(event.get("start_date"))
        event_end = _as_date(event.get("end_date"))
        if (
            event.get("active") in {0, False}
            or event_start is None
            or event_end is None
            or event_start > end
            or event_end < start
        ):
            continue
        selected_events.append((event, event_start, event_end))

    receipt_rows = []
    for receipt in receipts:
        receipt_date = _as_date(receipt.get("created_at"))
        if receipt_date is None:
            continue
        for item in _receipt_items(receipt):
            receipt_rows.append((receipt_date, item))

    failures = []
    event_signatures = {}
    for event, event_start, event_end in selected_events:
        raw_products = event.get("products")
        try:
            products = (
                json.loads(raw_products)
                if isinstance(raw_products, str)
                else list(raw_products or [])
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            products = []
        signature = (
            str(event.get("event_type") or ""),
            event_start,
            event_end,
            tuple(sorted(str(product) for product in products)),
            round(float(event.get("discount_pct") or 0), 4),
        )
        event_id = event.get("id", "<unknown>")
        if signature in event_signatures:
            failures.append(
                f"Duplicate business event {event_id} matches event "
                f"{event_signatures[signature]}"
            )
        else:
            event_signatures[signature] = event_id
    event_types = {
        str(event.get("event_type") or "")
        for event, _, _ in selected_events
    }
    for event_type in sorted(required_types - event_types):
        failures.append(f"Missing required operational business event: {event_type}")

    matched_receipts = 0
    for event, event_start, event_end in selected_events:
        raw_products = event.get("products")
        try:
            products = (
                json.loads(raw_products)
                if isinstance(raw_products, str)
                else list(raw_products or [])
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            products = []
        event_id = event.get("id", "<unknown>")
        if not products:
            failures.append(f"Business event {event_id} has no product scope")
            continue
        discount_pct = float(event.get("discount_pct") or 0)
        for product_name in products:
            matches = [
                item
                for receipt_date, item in receipt_rows
                if event_start <= receipt_date <= event_end
                and item.get("product_name") == product_name
                and str(item.get("discount_source") or "") == "business_event"
                and float(item.get("discount_pct") or 0) + 0.001 >= discount_pct
            ]
            if not matches:
                failures.append(
                    f"Business event {event_id} has no matching discounted "
                    f"receipt evidence for {product_name}"
                )
            else:
                matched_receipts += len(matches)

    observed = {
        "events": len(selected_events),
        "event_types": sorted(event_types),
        "matched_discounted_receipts": matched_receipts,
    }
    return _result(
        "operational_business_event_evidence",
        observed,
        "Planned business events have frontend records and matching discounted receipts",
        failures,
    )


def validate_batch_equations(batches, transactions, products) -> CheckResult:
    categories = _category_map(products)
    batches_by_id = {
        str(row.get("batch_id") or ""): row
        for row in batches
        if row.get("batch_id")
    }
    outflows = defaultdict(lambda: {"sold": 0, "discarded": 0, "other": 0})
    failures = []
    for row in transactions:
        if str(row.get("transaction_type") or "").lower() != "outflow":
            continue
        transaction_id = row.get("id")
        batch_id = str(row.get("batch_id") or "")
        product_name = str(row.get("product_name") or "")
        beverage = _is_beverage_product(product_name, categories)
        if not batch_id:
            if not beverage:
                failures.append(
                    f"Bakery inventory transaction {transaction_id} has no batch_id"
                )
            continue
        batch = batches_by_id.get(batch_id)
        if batch is None:
            failures.append(
                f"Inventory transaction {transaction_id} references missing batch {batch_id}"
            )
            continue
        batch_product = str(batch.get("product_name") or "")
        if product_name and product_name != batch_product:
            failures.append(
                f"Inventory transaction {transaction_id} product {product_name} does not match batch {batch_id} product {batch_product}"
            )
        transaction_time = _as_datetime(row.get("transaction_time"))
        production_time = _as_datetime(batch.get("production_time"))
        if transaction_time is not None and production_time is not None and transaction_time < production_time:
            failures.append(
                f"Inventory transaction {transaction_id} predates batch {batch_id} production"
            )
        if str(row.get("disposition") or "").lower() == "sold" and not row.get("receipt_id"):
            failures.append(f"Sold inventory transaction {transaction_id} has no receipt_id")
        disposition = str(row.get("disposition") or "").lower()
        bucket = disposition if disposition in {"sold", "discarded"} else "other"
        outflows[batch_id][bucket] += abs(int(row.get("quantity") or 0))
    for batch in batches:
        batch_id = str(batch.get("batch_id") or "<unknown>")
        product_name = str(batch.get("product_name") or "")
        if _is_beverage_product(product_name, categories):
            failures.append(f"Beverage batch inventory found for {batch_id}")
            continue
        initial = int(batch.get("quantity_initial") if batch.get("quantity_initial") is not None else batch.get("quantity") or 0)
        remaining = int(batch.get("quantity_remaining") or 0)
        sold = outflows[batch.get("batch_id")]["sold"]
        discarded = outflows[batch.get("batch_id")]["discarded"]
        other = outflows[batch.get("batch_id")]["other"]
        if other:
            failures.append(f"Batch {batch_id} has unsupported outflow quantity {other}")
        if initial != sold + discarded + remaining:
            failures.append(
                f"Batch {batch_id} equation failed: {initial} != {sold} + {discarded} + {remaining}"
            )
    return _result(
        "operational_batch_equations",
        {"batches": len(batches), "transactions": len(transactions)},
        "Bakery initial equals sold plus discarded plus remaining and beverages have no batches",
        failures,
    )


def validate_material_outflow_scope(
    transactions,
    recipes,
    products,
    raw_materials,
    *,
    receipt_ids=frozenset(),
    batches=(),
    orders=(),
    items=(),
    receipts=(),
) -> CheckResult:
    from api.module4_frontend.bff import (
        TAKEAWAY_BOX_CAPACITY,
        TAKEAWAY_BOX_MIN_BAKERY_UNITS,
    )

    categories = _category_map(products)
    products_by_name = {
        str(row.get("product_name") or ""): row for row in products
    }
    materials_by_name = {
        str(row.get("material_name") or ""): row for row in raw_materials
    }
    material_scopes = defaultdict(set)
    for recipe in recipes:
        product_name = str(recipe.get("product_name") or "")
        material_name = str(recipe.get("material_name") or "")
        scope = "beverage" if _is_beverage_product(product_name, categories) else "bakery"
        material_scopes[material_name].add(scope)
    for material in raw_materials:
        name = str(material.get("material_name") or "")
        category = str(material.get("category") or "").lower()
        if category == "packaging" or name in PACKAGING_MATERIALS:
            material_scopes[name].add("packaging")
    failures = []

    expected = Counter()
    expected_kind = {}
    production_reference_dates = {}
    bakery_batches = [
        row
        for row in batches
        if not _is_beverage_product(str(row.get("product_name") or ""), categories)
    ]
    enriched_recipes = []
    for recipe in recipes:
        product_name = str(recipe.get("product_name") or "")
        material_name = str(recipe.get("material_name") or "")
        product = products_by_name.get(product_name)
        material = materials_by_name.get(material_name)
        if product is None or material is None:
            continue
        enriched_recipes.append(
            {
                **recipe,
                "wastage_pct": product.get("wastage_pct"),
                "unit": material.get("unit"),
                "category": material.get("category"),
                "track_inventory": material.get("track_inventory", 1),
            }
        )
    if bakery_batches:
        try:
            production_expected = calculate_expected_outflows(
                bakery_batches, enriched_recipes
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"Cannot derive production material outflows: {exc}")
            production_expected = {}
        for reference, material_rows in production_expected.items():
            for material_name, values in material_rows.items():
                operation_date = _as_date(values.get("created_at"))
                key = (reference, material_name, operation_date)
                expected[key] += _decimal(values.get("quantity"))
                expected_kind[key] = "production"
                production_reference_dates[reference] = operation_date

    normalized_orders = _normalized_orders(orders)
    orders_by_id = {row.get("id"): row for row in normalized_orders}
    items_by_order = defaultdict(list)
    for item in items:
        if item.get("order_id") in orders_by_id:
            items_by_order[item.get("order_id")].append(item)
    receipt_dates = {
        str(row.get("receipt_id") or ""): _as_date(row.get("created_at"))
        for row in receipts
    }
    known_receipt_ids = set(receipt_ids) | set(receipt_dates)
    for order in normalized_orders:
        order_id = order.get("id")
        reference = str(order.get("ticket_id") or "")
        operation_date = _as_date(order.get("order_date"))
        bakery_quantity = 0
        for item in items_by_order.get(order_id, []):
            product_name = str(item.get("product_name") or "")
            quantity = int(item.get("quantity") or 0)
            if not _is_beverage_product(product_name, categories):
                bakery_quantity += quantity
                continue
            product_recipes = [
                row for row in enriched_recipes if row.get("product_name") == product_name
            ]
            if not product_recipes:
                failures.append(f"Missing product recipe: {product_name}")
            for recipe in product_recipes:
                material_name = str(recipe.get("material_name") or "")
                required = _decimal(recipe.get("quantity_per_unit")) * quantity
                if str(recipe.get("unit") or "").lower() != "pcs" and str(
                    recipe.get("category") or ""
                ).lower() != "packaging":
                    wastage_pct = products_by_name[product_name].get("wastage_pct")
                    required *= Decimal("1") + (
                        DEFAULT_CHECKOUT_WASTAGE_PCT
                        if wastage_pct is None
                        else _decimal(wastage_pct)
                    )
                key = (reference, material_name, operation_date)
                expected[key] += required.quantize(QUANTITY_PRECISION)
                expected_kind[key] = "checkout"
            cup_name = (
                "Cup Large"
                if str(item.get("coffee_size") or "").lower() == "large"
                else "Cup Regular"
            )
            key = (reference, cup_name, operation_date)
            expected[key] += Decimal(quantity).quantize(QUANTITY_PRECISION)
            expected_kind[key] = "checkout"
        if str(order.get("dine_type") or "").lower() == "takeaway":
            bag_key = (reference, "Packaging Bag", operation_date)
            expected[bag_key] += Decimal("1.000000")
            expected_kind[bag_key] = "checkout"
            if bakery_quantity >= TAKEAWAY_BOX_MIN_BAKERY_UNITS:
                box_quantity = (
                    bakery_quantity + TAKEAWAY_BOX_CAPACITY - 1
                ) // TAKEAWAY_BOX_CAPACITY
                box_key = (reference, "Packaging Box", operation_date)
                expected[box_key] += Decimal(box_quantity).quantize(
                    QUANTITY_PRECISION
                )
                expected_kind[box_key] = "checkout"

    actual = Counter()
    for row in transactions:
        if str(row.get("transaction_type") or "").lower() != "outflow":
            continue
        transaction_id = row.get("id")
        material_name = str(row.get("material_name") or "")
        reference = str(row.get("reference") or "")
        transaction_date = _as_date(row.get("created_at"))
        scopes = material_scopes.get(material_name, set())
        if reference.startswith("production:"):
            if "bakery" not in scopes:
                failures.append(
                    f"Non-bakery material {material_name} has production outflow {transaction_id}"
                )
            expected_date = production_reference_dates.get(reference)
            if expected_date is None:
                failures.append(
                    f"Material outflow {transaction_id} has untraceable production reference {reference}"
                )
            elif transaction_date != expected_date:
                failures.append(
                    f"Material outflow {transaction_id} reference {reference} date {transaction_date.isoformat() if transaction_date else '<invalid>'} does not match production date {expected_date.isoformat()}"
                )
        elif reference in known_receipt_ids:
            if not scopes.intersection({"beverage", "packaging"}):
                failures.append(
                    f"Bakery material {material_name} has checkout outflow {transaction_id}"
                )
            expected_date = receipt_dates.get(reference)
            if expected_date is not None and transaction_date != expected_date:
                failures.append(
                    f"Material outflow {transaction_id} reference {reference} date {transaction_date.isoformat() if transaction_date else '<invalid>'} does not match receipt date {expected_date.isoformat()}"
                )
        else:
            failures.append(
                f"Material outflow {transaction_id} for {material_name} has unknown source {reference or '<empty>'}"
            )
        actual[(reference, material_name, transaction_date)] += _decimal(
            row.get("quantity")
        )

    enforce_expected = bool(batches or orders or items or receipts)
    if enforce_expected:
        for key, expected_quantity in sorted(
            expected.items(), key=lambda item: tuple(str(value) for value in item[0])
        ):
            actual_quantity = actual.get(key, Decimal("0"))
            reference, material_name, operation_date = key
            kind = expected_kind[key]
            if actual_quantity == 0:
                failures.append(
                    f"Missing {kind} material outflow {reference}/{material_name} on {operation_date.isoformat() if operation_date else '<invalid>'}: expected {expected_quantity.quantize(QUANTITY_PRECISION)}"
                )
            elif actual_quantity != expected_quantity:
                failures.append(
                    f"{kind.title()} material outflow {reference}/{material_name} on {operation_date.isoformat() if operation_date else '<invalid>'} is {actual_quantity}, expected {expected_quantity.quantize(QUANTITY_PRECISION)}"
                )
        for key, actual_quantity in sorted(
            actual.items(), key=lambda item: tuple(str(value) for value in item[0])
        ):
            if key not in expected:
                reference, material_name, operation_date = key
                failures.append(
                    f"Unexpected material outflow {reference}/{material_name} on {operation_date.isoformat() if operation_date else '<invalid>'}: {actual_quantity}"
                )
    return _result(
        "operational_material_outflow_scope",
        {"outflows": sum(str(row.get("transaction_type") or "").lower() == "outflow" for row in transactions)},
        "Bakery materials flow at production; beverage and packaging materials flow at checkout",
        failures,
    )


def validate_restock_timing(transactions, raw_materials) -> CheckResult:
    final_stock = {
        str(row.get("material_name") or ""): _decimal(row.get("stock_quantity"))
        for row in raw_materials
    }
    grouped = defaultdict(list)
    for row in transactions:
        grouped[str(row.get("material_name") or "")].append(row)
    failures = []
    for material_name, rows in sorted(grouped.items()):
        inflow = sum(
            (_decimal(row.get("quantity")) for row in rows if str(row.get("transaction_type") or "").lower() in {"inflow", "restock"}),
            Decimal("0"),
        )
        outflow = sum(
            (_decimal(row.get("quantity")) for row in rows if str(row.get("transaction_type") or "").lower() == "outflow"),
            Decimal("0"),
        )
        balance = final_stock.get(material_name, Decimal("0")) + outflow - inflow
        for row in sorted(rows, key=lambda item: (_as_datetime(item.get("created_at")) or datetime.min, int(item.get("id") or 0))):
            quantity = _decimal(row.get("quantity"))
            transaction_type = str(row.get("transaction_type") or "").lower()
            if quantity < 0:
                failures.append(f"Material transaction {row.get('id')} has negative quantity")
                continue
            if transaction_type in {"inflow", "restock"}:
                balance += quantity
            elif transaction_type == "outflow":
                balance -= quantity
            if balance < 0:
                failures.append(
                    f"Material {material_name} became negative at transaction {row.get('id')}"
                )
    return _result(
        "operational_restock_before_shortage",
        {"materials": len(grouped), "transactions": len(transactions)},
        "Material balances never become negative before a restock",
        failures,
    )


def validate_discrete_material_quantities(
    raw_materials,
    transactions,
    checks,
) -> CheckResult:
    piece_materials = {
        str(row.get("material_name") or "")
        for row in raw_materials
        if str(row.get("unit") or "").strip().lower() == "pcs"
    }
    failures = []

    def is_fractional(value):
        quantity = _decimal(value)
        return quantity != quantity.to_integral_value()

    for row in sorted(
        raw_materials,
        key=lambda item: str(item.get("material_name") or ""),
    ):
        material_name = str(row.get("material_name") or "")
        if material_name not in piece_materials:
            continue
        for field_name in ("stock_quantity", "reorder_point"):
            if row.get(field_name) is not None and is_fractional(row.get(field_name)):
                failures.append(
                    f"Material {material_name} {field_name} must be a whole piece count"
                )

    for row in transactions:
        material_name = str(row.get("material_name") or "")
        if material_name in piece_materials and is_fractional(row.get("quantity")):
            failures.append(
                f"Material transaction {row.get('id')} for {material_name} must use a whole piece quantity"
            )

    quantity_fields = (
        "theoretical_stock",
        "actual_stock",
        "theoretical_consumed",
        "actual_consumed",
        "wastage_qty",
    )
    for row in checks:
        material_name = str(row.get("material_name") or "")
        if material_name not in piece_materials:
            continue
        for field_name in quantity_fields:
            if row.get(field_name) is not None and is_fractional(row.get(field_name)):
                failures.append(
                    f"Material check {row.get('id')} {field_name} for {material_name} must be a whole piece count"
                )

    return _result(
        "operational_discrete_material_quantities",
        {
            "piece_materials": len(piece_materials),
            "transactions": len(transactions),
            "checks": len(checks),
        },
        "Piece-based material stock, movements, and checks use whole-number quantities",
        failures,
    )


def validate_weekly_material_checks(
    checks,
    raw_materials,
    *,
    start: date = OPERATION_START,
    end: date = OPERATION_END,
) -> CheckResult:
    tracked = sorted(
        str(row.get("material_name") or "")
        for row in raw_materials
        if bool(row.get("track_inventory", 1)) and row.get("material_name")
    )
    checked = defaultdict(set)
    for row in checks:
        check_date = _as_date(row.get("check_date"))
        if check_date:
            checked[check_date].add(str(row.get("material_name") or ""))
    failures = []
    if not tracked:
        failures.append("No tracked raw materials found")
    required_dates = tuple(
        check_date
        for check_date in FULL_MATERIAL_CHECK_DATES
        if start <= check_date <= end
    )
    for check_date in required_dates:
        missing = sorted(set(tracked) - checked.get(check_date, set()))
        if missing:
            failures.append(
                f"Full material check {check_date.isoformat()} missing {', '.join(missing)}"
            )
    return _result(
        "operational_weekly_material_checks",
        {day.isoformat(): len(checked.get(day, set())) for day in required_dates},
        "All tracked materials are checked on each required Wednesday",
        failures,
    )


def validate_material_check_variance(
    checks,
    *,
    start: date = OPERATION_START,
    end: date = OPERATION_END,
) -> CheckResult:
    checks_by_date = defaultdict(list)
    for row in checks:
        check_date = _as_date(row.get("check_date"))
        if check_date is not None:
            checks_by_date[check_date].append(row)

    required_dates = tuple(
        check_date
        for check_date in FULL_MATERIAL_CHECK_DATES
        if start <= check_date <= end and check_date != OPERATION_START
    )
    failures = []
    observed = {}
    for check_date in required_dates:
        positive_rows = [
            row
            for row in checks_by_date.get(check_date, ())
            if _decimal(row.get("wastage_qty")) > 0
        ]
        observed[check_date.isoformat()] = len(positive_rows)
        if not positive_rows:
            failures.append(
                f"Material check {check_date.isoformat()} has no recorded operational variance"
            )
            continue
        if len(positive_rows) > 6:
            failures.append(
                f"Material check {check_date.isoformat()} records variance for more than 6 materials"
            )
        excessive = sorted(
            str(row.get("material_name") or "")
            for row in positive_rows
            if _decimal(row.get("wastage_rate")) > Decimal("0.03")
        )
        if excessive:
            failures.append(
                f"Material check {check_date.isoformat()} exceeds 3% variance for {', '.join(excessive)}"
            )
    return _result(
        "operational_material_check_variance",
        observed,
        "Each post-baseline weekly check records small variance for 1 to 6 materials",
        failures,
    )


def _canonical_schedule_windows(schedules):
    schedules_by_date = defaultdict(list)
    for row in schedules:
        schedule_date = _as_date(row.get("schedule_date"))
        if schedule_date is not None:
            schedules_by_date[schedule_date].append(row)
    windows_by_date = {
        schedule_date: build_schedule_windows(rows, schedule_date.isoformat())
        for schedule_date, rows in schedules_by_date.items()
    }
    return schedules_by_date, windows_by_date


def validate_schedule_attendance(
    schedules,
    records,
    *,
    start: date | None = None,
    end: date | None = None,
) -> CheckResult:
    scheduled = {
        (_as_date(row.get("schedule_date")), str(row.get("employee_id") or ""))
        for row in schedules
    }
    record_counts = Counter(
        (_as_date(row.get("date")), str(row.get("emp_id") or ""))
        for row in records
    )
    _, windows_by_date = _canonical_schedule_windows(schedules)
    failures = []
    if start is not None and end is not None:
        scheduled_dates = {schedule_date for schedule_date, _ in scheduled}
        for operation_date in _date_range(start, end):
            if operation_date not in scheduled_dates:
                failures.append(
                    f"Missing operational schedule on {operation_date.isoformat()}"
                )
    for schedule_date, employee_id in sorted(
        scheduled, key=lambda value: tuple(str(item) for item in value)
    ):
        if (
            schedule_date is None
            or employee_id not in windows_by_date.get(schedule_date, {})
        ):
            failures.append(
                f"Schedule {employee_id or '<empty>'} on {schedule_date.isoformat() if schedule_date else '<invalid>'} has no canonical window"
            )
        count = record_counts.get((schedule_date, employee_id), 0)
        if count > 1:
            failures.append(
                f"Duplicate attendance records for {employee_id} on {schedule_date.isoformat() if schedule_date else '<invalid>'}: {count}"
            )
    for record in records:
        record_date = _as_date(record.get("date"))
        employee_id = str(record.get("emp_id") or "")
        if (record_date, employee_id) not in scheduled:
            failures.append(
                f"Attendance {employee_id} on {record_date.isoformat() if record_date else '<invalid>'} has no schedule"
            )
    return _result(
        "operational_schedule_attendance",
        {"schedule_rows": len(schedules), "attendance_rows": len(records)},
        "Every scheduled employee has a canonical window and attendance records are unique and schedule-backed",
        failures,
    )


def validate_punch_status(schedules, records) -> CheckResult:
    schedules_by_date, windows_by_date = _canonical_schedule_windows(schedules)
    failures = []
    for schedule_date, rows in sorted(schedules_by_date.items()):
        windows = windows_by_date[schedule_date]
        employee_ids = {
            str(row.get("employee_id") or "") for row in rows
        }
        for employee_id in sorted(employee_ids):
            if employee_id not in windows:
                failures.append(
                    f"Schedule {employee_id or '<empty>'} on {schedule_date.isoformat()} has no canonical window"
                )
                continue
    for record in records:
        record_date = _as_date(record.get("date"))
        employee_id = str(record.get("emp_id") or "")
        if record_date is None:
            failures.append(f"Attendance {employee_id} has invalid date")
            continue
        window = windows_by_date.get(record_date, {}).get(employee_id)
        if not window:
            continue
        expected = derive_attendance_status(record, window)
        observed = str(record.get("status") or "")
        if observed != expected:
            failures.append(
                f"Attendance {employee_id} on {record_date.isoformat()} has status {observed}, expected {expected}"
            )
    return _result(
        "operational_punch_status",
        {"attendance_rows": len(records)},
        "Recorded punch times and statuses agree with schedule windows; missing records represent absence",
        failures,
    )


def validate_top3_events(
    events,
    orders,
    *,
    start: date = OPERATION_START,
    end: date = OPERATION_END,
) -> CheckResult:
    orders_by_id = {row.get("id"): row for row in orders}
    requests = defaultdict(list)
    counts = defaultdict(lambda: {"shown": 0, "selected": 0, "purchased": 0})
    failures = []
    for event in events:
        event_id = event.get("id")
        operation_date = _as_date(event.get("operation_date"))
        shown_at = _as_datetime(event.get("shown_at"))
        selected_at = _as_datetime(event.get("selected_at"))
        request_id = str(event.get("request_id") or "")
        requests[request_id].append(event)
        shown_date = shown_at.date() if shown_at is not None else None
        if shown_date != operation_date:
            failures.append(
                f"Recommendation event {event_id} shown_at {shown_date.isoformat() if shown_date else '<invalid>'} does not match operation_date {operation_date.isoformat() if operation_date else '<invalid>'}"
            )
        if event.get("selected_at") is not None:
            selected_date = selected_at.date() if selected_at is not None else None
            if selected_date != operation_date:
                failures.append(
                    f"Recommendation event {event_id} selected_at {selected_date.isoformat() if selected_date else '<invalid>'} does not match operation_date {operation_date.isoformat() if operation_date else '<invalid>'}"
                )
            if shown_at is not None and selected_at is not None:
                try:
                    selection_precedes_exposure = selected_at < shown_at
                except TypeError:
                    selection_precedes_exposure = True
                if selection_precedes_exposure:
                    failures.append(
                        f"Recommendation event {event_id} selected_at precedes shown_at"
                    )
        if operation_date:
            counts[operation_date]["shown"] += 1
            if event.get("selected_at") is not None:
                counts[operation_date]["selected"] += 1
            if event.get("purchased_order_id") is not None:
                counts[operation_date]["purchased"] += 1
        if event.get("purchased_order_id") is not None and event.get("selected_at") is None:
            failures.append(f"Recommendation event {event_id} was purchased without selection")
        purchased_order_id = event.get("purchased_order_id")
        if purchased_order_id is not None:
            order = orders_by_id.get(purchased_order_id)
            if order is None:
                failures.append(
                    f"Recommendation event {event_id} references missing order {purchased_order_id}"
                )
            elif _as_date(order.get("order_date")) != operation_date:
                failures.append(f"Recommendation event {event_id} purchase date does not match order")
    for request_id, request_events in sorted(requests.items()):
        ranks = sorted(int(row.get("rank_position") or 0) for row in request_events)
        if ranks != [1, 2, 3]:
            failures.append(
                f"Recommendation request {request_id or '<empty>'} ranks are {ranks}, expected [1, 2, 3]"
            )
    for operation_date in _date_range(start, end):
        if counts[operation_date]["shown"] == 0:
            failures.append(f"Missing Top 3 exposure on {operation_date.isoformat()}")
    observed = {day.isoformat(): dict(counts[day]) for day in _date_range(start, end)}
    return _result(
        "operational_top3_events",
        observed,
        "Each day has measurable three-rank exposure and selection/purchase lifecycle consistency",
        failures,
    )


def validate_revenue_reconciliation(orders, items, receipts=()) -> CheckResult:
    paid = _paid_orders(orders)
    order_dates = {row.get("id"): _as_date(row.get("order_date")) for row in paid}
    header = defaultdict(lambda: {"orders": 0, "items": 0, "revenue": Decimal("0")})
    source = defaultdict(lambda: {"items": 0, "revenue": Decimal("0")})
    for order in paid:
        order_date = _as_date(order.get("order_date"))
        header[order_date]["orders"] += 1
        header[order_date]["items"] += int(order.get("item_count") or 0)
        header[order_date]["revenue"] += _decimal(order.get("total_amount"))
    for item in items:
        order_date = order_dates.get(item.get("order_id"))
        if order_date is None:
            continue
        source[order_date]["items"] += int(item.get("quantity") or 0)
        source[order_date]["revenue"] += _decimal(item.get("line_total"))
    failures = []
    for operation_date in sorted(header):
        if header[operation_date]["items"] != source[operation_date]["items"]:
            failures.append(
                f"Revenue BI items on {operation_date.isoformat()} are {header[operation_date]['items']}, normalized items are {source[operation_date]['items']}"
            )
        if header[operation_date]["revenue"] != source[operation_date]["revenue"]:
            failures.append(
                f"Revenue BI total on {operation_date.isoformat()} is {header[operation_date]['revenue']}, normalized revenue is {source[operation_date]['revenue']}"
            )
    observed = {
        day.isoformat(): {
            "orders": values["orders"],
            "items": values["items"],
            "revenue": values["revenue"],
        }
        for day, values in sorted(header.items())
        if day is not None
    }
    return _result(
        "operational_revenue_reconciliation",
        observed,
        "Revenue BI order, item, and revenue totals equal normalized source rows",
        failures,
    )


def validate_inventory_source_equations(items, inventory_transactions, orders, products) -> CheckResult:
    categories = _category_map(products)
    tickets = {
        row.get("id"): (
            str(row.get("ticket_id") or ""),
            _as_date(row.get("order_date")),
        )
        for row in orders
    }
    sold = Counter()
    for row in inventory_transactions:
        if (
            str(row.get("transaction_type") or "").lower() == "outflow"
            and str(row.get("disposition") or "").lower() == "sold"
        ):
            sold[
                (
                    _as_date(row.get("transaction_time")),
                    str(row.get("receipt_id") or ""),
                    str(row.get("product_name") or ""),
                )
            ] += abs(int(row.get("quantity") or 0))
    expected = Counter()
    for item in items:
        product_name = str(item.get("product_name") or "")
        if _is_beverage_product(product_name, categories):
            continue
        ticket_id, order_date = tickets.get(item.get("order_id"), ("", None))
        expected[(order_date, ticket_id, product_name)] += int(
            item.get("quantity") or 0
        )
    failures = []
    for (order_date, ticket_id, product_name), quantity in sorted(
        expected.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        actual = sold.get((order_date, ticket_id, product_name), 0)
        if actual != quantity:
            order_id = next(
                (
                    key
                    for key, value in tickets.items()
                    if value == (ticket_id, order_date)
                ),
                "<unknown>",
            )
            if order_date is None:
                failures.append(
                    f"Order {order_id} bakery quantity for {product_name} is {quantity}, inventory sold outflow is {actual}"
                )
            else:
                failures.append(
                    f"Order {order_id} on {order_date.isoformat()} bakery quantity for {product_name} is {quantity}, inventory sold outflow is {actual}"
                )
    for (transaction_date, ticket_id, product_name), quantity in sorted(
        sold.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        if (
            not _is_beverage_product(product_name, categories)
            and (transaction_date, ticket_id, product_name) not in expected
        ):
            if transaction_date is None:
                failures.append(
                    f"Inventory sold outflow for unknown bakery order {ticket_id or '<empty>'}/{product_name} is {quantity}"
                )
            else:
                failures.append(
                    f"Inventory sold outflow on {transaction_date.isoformat()} for {ticket_id or '<empty>'}/{product_name} has no matching bakery order"
                )
    return _result(
        "operational_inventory_source_equations",
        {"expected_lines": len(expected), "sold_lines": len(sold)},
        "Bakery order quantities equal sold finished-inventory outflows",
        failures,
    )


def validate_wastage_source_equations(checks) -> CheckResult:
    from api.module4_frontend.bff import (
        _calculate_material_wastage,
        _normalize_actual_consumed,
        _normalize_material_wastage,
    )

    failures = []
    for row in checks:
        check_id = row.get("id")
        wastage_qty = _decimal(row.get("wastage_qty"))
        wastage_rate = _decimal(row.get("wastage_rate"))
        if wastage_qty < 0 or wastage_rate < 0:
            failures.append(f"Negative material wastage in check {check_id}")
            continue
        theoretical_stock = _decimal(row.get("theoretical_stock"))
        theoretical_consumed = _decimal(row.get("theoretical_consumed"))
        stored_actual = _normalize_actual_consumed(
            row.get("actual_consumed"), theoretical_consumed
        )
        stored_wastage, stored_rate = _normalize_material_wastage(
            wastage_qty, wastage_rate, theoretical_consumed
        )
        reconstructed_reference = theoretical_stock + theoretical_consumed
        actual_consumed, _, expected_wastage, expected_rate = _calculate_material_wastage(
            reference_stock=reconstructed_reference,
            restocked=0,
            actual_stock=_decimal(row.get("actual_stock")),
            theoretical_consumed=theoretical_consumed,
        )
        if abs(_decimal(stored_actual) - _decimal(actual_consumed)) > MATERIAL_QUANTITY_TOLERANCE:
            failures.append(f"Material check {check_id} actual_consumed does not reconcile")
        wastage_delta = abs(
            _decimal(stored_wastage) - _decimal(expected_wastage)
        )
        legacy_rounding_delta = (
            Decimal("0") < wastage_delta <= MATERIAL_QUANTITY_TOLERANCE
        )
        if wastage_delta > MATERIAL_QUANTITY_TOLERANCE:
            failures.append(f"Material check {check_id} wastage_qty does not reconcile")
        if (
            _decimal(stored_rate) != _decimal(expected_rate)
            and not legacy_rounding_delta
        ):
            failures.append(f"Material check {check_id} wastage_rate does not reconcile")
    return _result(
        "operational_wastage_source_equations",
        {"checks": len(checks)},
        "Stored material checks match the canonical wastage calculation",
        failures,
    )


def _required_table_check(scope: str, table: str, available: frozenset[str]) -> CheckResult:
    missing = table not in available
    return _result(
        f"{scope}_required_table_{table}",
        {"available": not missing},
        f"Required {scope} table {table} exists",
        (f"Missing {scope} table {table}",) if missing else (),
    )


def collect_snapshot(connection, scope="all") -> DataSnapshot:
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("START TRANSACTION READ ONLY")
        cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()"
        )
        available = frozenset(
            str(value)
            for row in cursor.fetchall()
            for key, value in row.items()
            if str(key).lower() == "table_name"
        )
        needed = set()
        if scope in {"all", "historical"}:
            needed.update(HISTORICAL_TABLES)
        if scope in {"all", "operational"}:
            needed.update(OPERATIONAL_TABLES)
        data = {}
        for table in TABLE_QUERIES:
            if table not in needed or table not in available:
                continue
            cursor.execute(TABLE_QUERIES[table])
            data[table] = tuple(dict(row) for row in cursor.fetchall())
        return DataSnapshot(available_tables=available, **data)
    finally:
        try:
            cursor.close()
        finally:
            connection.rollback()


def _period_snapshot(snapshot: DataSnapshot, start: date, end: date):
    orders = _rows_for_period(snapshot.orders, "order_date", start, end)
    order_ids = {row.get("id") for row in orders}
    items = [row for row in snapshot.order_items if row.get("order_id") in order_ids]
    payments = [row for row in snapshot.payments if row.get("order_id") in order_ids]
    receipts = _rows_for_period(snapshot.receipts, "created_at", start, end)
    return orders, items, payments, receipts


def _batch_lifecycle_rows(snapshot: DataSnapshot, start: date, end: date):
    period_batches = _rows_for_period(
        snapshot.batch_inventory,
        "production_time",
        start,
        end,
    )
    period_transactions = _rows_for_period(
        snapshot.inventory_transactions,
        "transaction_time",
        start,
        end,
    )
    relevant_batch_ids = {
        str(row.get("batch_id") or "")
        for row in (*period_batches, *period_transactions)
        if row.get("batch_id")
    }
    period_transaction_rows = {id(row) for row in period_transactions}
    batches = [
        row
        for row in snapshot.batch_inventory
        if str(row.get("batch_id") or "") in relevant_batch_ids
    ]
    transactions = [
        row
        for row in snapshot.inventory_transactions
        if str(row.get("batch_id") or "") in relevant_batch_ids
        or id(row) in period_transaction_rows
    ]
    return batches, transactions


def _load_history_manifest():
    return json.loads(HISTORY_MANIFEST_PATH.read_text(encoding="ascii"))


def run_checks(
    snapshot: DataSnapshot,
    scope="all",
    *,
    history_manifest=None,
    operation_start: date = OPERATION_START,
    operation_end: date = OPERATION_END,
) -> tuple[CheckResult, ...]:
    if operation_start < OPERATION_START or operation_end > OPERATION_END:
        raise ValueError(
            f"Operational range must stay within {OPERATION_START}:{OPERATION_END}"
        )
    if operation_start > operation_end:
        raise ValueError("Operational range start must not be after end")
    results = []
    if scope in {"all", "historical"}:
        table_results = [
            _required_table_check("historical", table, snapshot.available_tables)
            for table in HISTORICAL_TABLES
        ]
        results.extend(table_results)
        if all(result.passed for result in table_results):
            manifest_error = None
            if history_manifest is None:
                try:
                    history_manifest = _load_history_manifest()
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    manifest_error = exc
            if manifest_error is None:
                manifest_result = validate_history_manifest(history_manifest)
                expected_history_count = manifest_result.observed["order_count"]
                expected_history_item_count = manifest_result.observed["item_count"]
            else:
                manifest_result = _result(
                    "historical_manifest_contract",
                    {"path": str(HISTORY_MANIFEST_PATH)},
                    "Task 3 manifest range and reconstructed order/item counts are valid",
                    (f"Cannot read historical manifest: {manifest_error}",),
                )
                expected_history_count = None
                expected_history_item_count = None
            reconstructed = [
                row
                for row in snapshot.orders
                if str(row.get("ticket_id") or "").startswith("GZ-")
            ]
            reconstructed_order_ids = {row.get("id") for row in reconstructed}
            reconstructed_item_rows = [
                row
                for row in snapshot.order_items
                if row.get("order_id") in reconstructed_order_ids
            ]
            normalized = _normalized_orders(reconstructed)
            paid = _paid_orders(reconstructed)
            paid_order_ids = {row.get("id") for row in paid}
            item_rows = [
                row
                for row in snapshot.order_items
                if row.get("order_id") in paid_order_ids
            ]
            payment_rows = [
                row
                for row in snapshot.payments
                if row.get("order_id") in paid_order_ids
            ]
            results.append(manifest_result)
            results.extend(
                (
                    validate_history_range(
                        reconstructed,
                        expected_order_count=expected_history_count,
                        observed_item_count=len(reconstructed_item_rows),
                        expected_item_count=expected_history_item_count,
                    ),
                    validate_normalized_transactions(
                        normalized,
                        snapshot.order_items,
                        snapshot.payments,
                        source_orders=snapshot.orders,
                        scope_start=HISTORY_START,
                        scope_end=HISTORY_END,
                    ),
                    validate_daily_order_ranges(paid),
                    validate_basket_profile(paid),
                    validate_sales_boundary(paid),
                    validate_service_payment_shares(paid, payment_rows),
                    validate_product_coverage(item_rows, snapshot.products),
                    validate_bakery_concentration(item_rows, snapshot.products),
                    validate_historical_product_mix(paid, item_rows, snapshot.products),
                )
            )
    if scope in {"all", "operational"}:
        table_results = [
            _required_table_check("operational", table, snapshot.available_tables)
            for table in OPERATIONAL_TABLES
        ]
        results.extend(table_results)
        if all(result.passed for result in table_results):
            orders, items, payments, receipts = _period_snapshot(
                snapshot, operation_start, operation_end
            )
            normalized = _normalized_orders(orders)
            normalized_order_ids = {row.get("id") for row in normalized}
            normalized_item_rows = [
                row for row in items if row.get("order_id") in normalized_order_ids
            ]
            paid = _paid_orders(orders)
            paid_order_ids = {row.get("id") for row in paid}
            item_rows = [row for row in items if row.get("order_id") in paid_order_ids]
            payment_rows = [
                row for row in payments if row.get("order_id") in paid_order_ids
            ]
            receipt_ids = {str(row.get("receipt_id") or "") for row in receipts}
            batches = _rows_for_period(
                snapshot.batch_inventory,
                "production_time",
                operation_start,
                operation_end,
            )
            inventory_transactions = _rows_for_period(
                snapshot.inventory_transactions,
                "transaction_time",
                operation_start,
                operation_end,
            )
            lifecycle_batches, lifecycle_inventory_transactions = _batch_lifecycle_rows(
                snapshot,
                operation_start,
                operation_end,
            )
            material_transactions = _rows_for_period(
                snapshot.material_transactions,
                "created_at",
                operation_start,
                operation_end,
            )
            material_checks = _rows_for_period(
                snapshot.material_wastage_log,
                "check_date",
                operation_start,
                operation_end,
            )
            schedules = _rows_for_period(
                snapshot.shift_schedule,
                "schedule_date",
                operation_start,
                operation_end,
            )
            attendance = _rows_for_period(
                snapshot.attendance_records,
                "date",
                operation_start,
                operation_end,
            )
            events = _rows_for_period(
                snapshot.recommendation_events,
                "operation_date",
                operation_start,
                operation_end,
            )
            business_events = [
                row
                for row in snapshot.business_events
                if (_as_date(row.get("start_date")) or date.max) <= operation_end
                and (_as_date(row.get("end_date")) or date.min) >= operation_start
            ]
            results.extend(
                (
                    validate_operational_completeness(
                        paid,
                        item_rows,
                        payment_rows,
                        receipts,
                        start=operation_start,
                        end=operation_end,
                    ),
                    validate_operational_product_coverage(
                        paid,
                        item_rows,
                        snapshot.products,
                        start=operation_start,
                        end=operation_end,
                    ),
                    validate_normalized_transactions(
                        normalized,
                        snapshot.order_items,
                        snapshot.payments,
                        snapshot.receipts,
                        name="operational_normalized_relationships",
                        source_orders=snapshot.orders,
                        scope_start=operation_start,
                        scope_end=operation_end,
                        material_transactions=material_transactions,
                        raw_materials=snapshot.raw_materials,
                    ),
                    validate_batch_equations(
                        lifecycle_batches,
                        lifecycle_inventory_transactions,
                        snapshot.products,
                    ),
                    validate_material_outflow_scope(
                        material_transactions,
                        snapshot.product_recipes,
                        snapshot.products,
                        snapshot.raw_materials,
                        receipt_ids=receipt_ids,
                        batches=batches,
                        orders=normalized,
                        items=normalized_item_rows,
                        receipts=receipts,
                    ),
                    validate_inventory(batches, snapshot.raw_materials),
                    validate_restock_timing(material_transactions, snapshot.raw_materials),
                    validate_discrete_material_quantities(
                        snapshot.raw_materials,
                        material_transactions,
                        material_checks,
                    ),
                    validate_weekly_material_checks(
                        material_checks,
                        snapshot.raw_materials,
                        start=operation_start,
                        end=operation_end,
                    ),
                    validate_material_check_variance(
                        material_checks,
                        start=operation_start,
                        end=operation_end,
                    ),
                    validate_schedule_attendance(
                        schedules,
                        attendance,
                        start=operation_start,
                        end=operation_end,
                    ),
                    validate_punch_status(schedules, attendance),
                    validate_top3_events(
                        events,
                        normalized,
                        start=operation_start,
                        end=operation_end,
                    ),
                    validate_business_event_evidence(
                        business_events,
                        receipts,
                        start=operation_start,
                        end=operation_end,
                    ),
                    validate_revenue_reconciliation(
                        normalized, normalized_item_rows, receipts
                    ),
                    validate_inventory_source_equations(
                        normalized_item_rows,
                        inventory_transactions,
                        normalized,
                        snapshot.products,
                    ),
                    validate_wastage_source_equations(material_checks),
                )
            )
    return tuple(results)


def _json_default(value):
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_report(path: Path, scope: str, results: Iterable[CheckResult]) -> dict:
    results = tuple(results)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "read_only": True,
        "passed": all(result.passed for result in results),
        "checks": [asdict(result) for result in results],
    }
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return payload


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an ISO date in YYYY-MM-DD format: {value}"
        ) from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the Guangzhou historical and operational data contract read-only."
    )
    parser.add_argument(
        "--scope",
        choices=("all", "historical", "operational"),
        default="all",
    )
    parser.add_argument("--start", type=_parse_iso_date)
    parser.add_argument("--end", type=_parse_iso_date)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be provided together")
    if args.start is not None:
        if args.scope != "operational":
            parser.error("--start and --end are supported only for operational scope")
        if args.start < OPERATION_START or args.end > OPERATION_END:
            parser.error(
                f"Operational range must stay within {OPERATION_START}:{OPERATION_END}"
            )
        if args.start > args.end:
            parser.error("--start must not be after --end")
    return args


def main(argv=None, *, connect=None) -> int:
    args = parse_args(argv)
    if connect is None:
        from db.mysql_client import get_db

        connect = get_db
    connection = None
    try:
        connection = connect(autocommit=False)
        snapshot = collect_snapshot(connection, args.scope)
        results = run_checks(
            snapshot,
            args.scope,
            operation_start=args.start or OPERATION_START,
            operation_end=args.end or OPERATION_END,
        )
    except Exception as exc:
        results = (
            _result(
                "database_read",
                {"error_type": type(exc).__name__},
                "Database snapshot is readable in a read-only transaction",
                (f"Database read failed: {exc}",),
            ),
        )
    finally:
        if connection is not None:
            connection.close()
    payload = write_report(args.output, args.scope, results)
    failing_names = [result.name for result in results if not result.passed]
    for name in failing_names:
        print(name)
    if not failing_names:
        print("All required checks passed")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
