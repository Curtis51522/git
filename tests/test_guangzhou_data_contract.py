from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import json

from scripts.validate_guangzhou_data_contract import (
    _batch_lifecycle_rows,
    CheckResult,
    DataSnapshot,
    collect_snapshot,
    main,
    parse_args,
    run_checks,
    summarize_orders,
    validate_batch_equations,
    validate_bakery_concentration,
    validate_business_event_evidence,
    validate_daily_order_ranges,
    validate_history_range,
    validate_historical_product_mix,
    validate_inventory,
    validate_inventory_source_equations,
    validate_material_check_variance,
    validate_material_outflow_scope,
    validate_normalized_transactions,
    validate_operational_completeness,
    validate_operational_product_coverage,
    validate_product_coverage,
    validate_punch_status,
    validate_revenue_reconciliation,
    validate_restock_timing,
    validate_sales_boundary,
    validate_schedule_attendance,
    validate_service_payment_shares,
    validate_top3_events,
    validate_wastage_source_equations,
    validate_weekly_material_checks,
)


def _order(
    order_id: int,
    order_date: date,
    *,
    ticket_id: str | None = None,
    item_count: int = 2,
    total_amount: str = "24.0",
    dine_type: str = "takeaway",
    order_time: time = time(8, 30),
):
    return {
        "id": order_id,
        "ticket_id": ticket_id or f"T-{order_id}",
        "order_date": order_date,
        "order_time": order_time,
        "subtotal": Decimal(total_amount),
        "discount_total": Decimal("0.0"),
        "total_amount": Decimal(total_amount),
        "total_profit": Decimal("10.0"),
        "item_count": item_count,
        "state": "paid",
        "dine_type": dine_type,
    }


def _items(order_id: int):
    return [
        {
            "id": order_id * 10 + 1,
            "order_id": order_id,
            "product_name": "croissant",
            "quantity": 1,
            "unit_price": Decimal("12.0"),
            "discount_rate": Decimal("0"),
            "line_total": Decimal("12.0"),
            "line_profit": Decimal("5.0"),
        },
        {
            "id": order_id * 10 + 2,
            "order_id": order_id,
            "product_name": "latte",
            "quantity": 1,
            "unit_price": Decimal("12.0"),
            "discount_rate": Decimal("0"),
            "line_total": Decimal("12.0"),
            "line_profit": Decimal("5.0"),
        },
    ]


def _payment(order_id: int, payment_date: date, method: str = "qr"):
    return {
        "id": order_id,
        "order_id": order_id,
        "payment_method": method,
        "amount": Decimal("24.0"),
        "payment_date": payment_date,
    }


def _receipt(order_id: int, created_at: datetime):
    return {
        "id": order_id,
        "receipt_id": f"T-{order_id}",
        "subtotal": 24.0,
        "discount_total": 0.0,
        "total": 24.0,
        "created_at": created_at,
    }


def _check(results, name):
    return next(result for result in results if result.name == name)


def test_validator_computes_basket_metrics():
    rows = [
        {"ticket_id": "A", "quantity": 1, "total_amount": 12.0},
        {"ticket_id": "B", "quantity": 2, "total_amount": 24.0},
    ]

    result = summarize_orders(rows)

    assert result.order_count == 2
    assert result.item_count == 3
    assert result.items_per_order == 1.5
    assert result.average_order_value == 18.0


def test_every_validator_returns_check_result_and_negative_inventory_is_precise():
    result = validate_inventory(
        [{"batch_id": "B1", "quantity_remaining": -1}],
        [{"material_name": "Flour", "stock_quantity": Decimal("-0.1")}],
    )

    assert isinstance(result, CheckResult)
    assert result.name == "operational_inventory_nonnegative"
    assert result.failures == (
        "Negative finished stock in batch B1",
        "Negative raw stock for Flour",
    )


def test_history_range_requires_exact_continuous_dates_and_cutoff():
    result = validate_history_range(
        [
            {"order_date": date(2025, 6, 24)},
            {"order_date": date(2025, 6, 26)},
            {"order_date": date(2026, 6, 24)},
        ]
    )

    assert isinstance(result, CheckResult)
    assert not result.passed
    assert "Historical end date is 2026-06-24, expected 2026-06-23" in result.failures
    assert "Historical order on or after operation cutoff: 2026-06-24" in result.failures
    assert "Missing historical trading date 2025-06-25" in result.failures


def test_run_checks_uses_gz_provenance_and_manifest_contract():
    start = date(2025, 6, 24)
    end = date(2026, 6, 23)
    orders = [
        _order(
            index + 1,
            start + timedelta(days=index),
            ticket_id=f"GZ-{start + timedelta(days=index):%Y%m%d}-0001",
            item_count=1,
            total_amount="12.0",
        )
        for index in range((end - start).days + 1)
    ]
    operational_order = _order(
        900001,
        date(2026, 6, 24),
        ticket_id="POS-20260624-0001",
        item_count=1,
        total_amount="12.0",
    )
    manifest = {
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "order_count": len(orders),
        "row_counts": {"output": 0},
    }
    snapshot = DataSnapshot(
        available_tables=frozenset({"orders", "order_items", "payments", "products"}),
        orders=tuple(orders + [operational_order]),
    )

    results = run_checks(
        snapshot,
        scope="historical",
        history_manifest=manifest,
    )

    assert _check(results, "historical_manifest_contract").passed
    assert _check(results, "historical_date_range").passed

    leaked = DataSnapshot(
        available_tables=snapshot.available_tables,
        orders=snapshot.orders
        + (
            _order(
                900002,
                date(2026, 6, 24),
                ticket_id="GZ-20260624-0001",
                item_count=1,
                total_amount="12.0",
            ),
        ),
    )
    leakage = _check(
        run_checks(
            leaked,
            scope="historical",
            history_manifest=manifest,
        ),
        "historical_date_range",
    )

    assert f"Historical GZ order count is {len(orders) + 1}, expected manifest count {len(orders)}" in leakage.failures
    assert "Historical order on or after operation cutoff: 2026-06-24" in leakage.failures

    missing = DataSnapshot(
        available_tables=snapshot.available_tables,
        orders=tuple(orders[:-1] + [operational_order]),
    )
    incomplete = _check(
        run_checks(
            missing,
            scope="historical",
            history_manifest=manifest,
        ),
        "historical_date_range",
    )

    assert f"Historical GZ order count is {len(orders) - 1}, expected manifest count {len(orders)}" in incomplete.failures
    assert "Historical end date is 2026-06-22, expected 2026-06-23" in incomplete.failures

    item_manifest = deepcopy(manifest)
    item_manifest["row_counts"]["output"] = 1
    item_count = _check(
        run_checks(
            snapshot,
            scope="historical",
            history_manifest=item_manifest,
        ),
        "historical_date_range",
    )

    assert "Historical GZ item row count is 0, expected manifest output count 1" in item_count.failures


def test_normalized_transactions_reconcile_orders_items_payments_and_receipts():
    day = date(2026, 6, 24)
    orders = [_order(1, day)]
    items = _items(1)
    payments = [_payment(1, day)]
    receipts = [_receipt(1, datetime(2026, 6, 24, 8, 31))]

    result = validate_normalized_transactions(
        orders,
        items,
        payments,
        receipts,
        name="operational_normalized_relationships",
    )

    assert result == CheckResult(
        name="operational_normalized_relationships",
        passed=True,
        observed={"orders": 1, "items": 2, "payments": 1, "receipts": 1},
        expected="Every normalized order reconciles to item, payment, and receipt evidence",
    )

    broken = deepcopy(items)
    broken[0]["line_total"] = Decimal("11.0")
    failure = validate_normalized_transactions(
        orders,
        broken,
        payments,
        receipts,
        name="operational_normalized_relationships",
    )
    assert "Order 1 total_amount 24.0 does not equal item line total 23.0" in failure.failures
    assert "Order 1 discount_total 0.0 does not equal 1.0" in failure.failures


def test_normalized_transactions_reject_cross_date_payment_and_receipt():
    day = date(2026, 6, 24)
    result = validate_normalized_transactions(
        [_order(1, day)],
        _items(1),
        [_payment(1, date(2026, 6, 25))],
        [_receipt(1, datetime(2026, 6, 25, 8, 31))],
        name="operational_normalized_relationships",
    )

    assert result.failures == (
        "Order 1 payment date 2026-06-25 does not equal order date 2026-06-24",
        "Order 1 receipt date 2026-06-25 does not equal order date 2026-06-24",
    )


def test_takeaway_packaging_cost_reconciles_without_customer_fee():
    day = date(2026, 6, 24)
    order = _order(1, day)
    order["total_profit"] = Decimal("9.85")
    payment = _payment(1, day)
    receipt = _receipt(1, datetime(2026, 6, 24, 8, 31))
    receipt["items"] = json.dumps(
        [
            {"product_name": "croissant", "line_total": 12.0},
            {"product_name": "latte", "line_total": 12.0},
        ]
    )
    material_transactions = [
        {
            "id": 1,
            "material_name": "Packaging Bag",
            "transaction_type": "outflow",
            "quantity": Decimal("1"),
            "reference": "T-1",
            "created_at": datetime(2026, 6, 24, 8, 31),
        }
    ]
    raw_materials = [
        {
            "material_name": "Packaging Bag",
            "unit_price": Decimal("0.15"),
        }
    ]

    normalized = validate_normalized_transactions(
        [order],
        _items(1),
        [payment],
        [receipt],
        name="operational_normalized_relationships",
        material_transactions=material_transactions,
        raw_materials=raw_materials,
    )
    revenue = validate_revenue_reconciliation([order], _items(1), [receipt])

    assert normalized.passed
    assert revenue.passed
    assert revenue.observed[day.isoformat()]["items"] == 2
    assert revenue.observed[day.isoformat()]["revenue"] == Decimal("24.0")


def test_run_checks_rejects_orphan_cross_scope_and_extra_operational_rows():
    day = date(2026, 6, 24)
    paid_order = _order(1, day)
    excluded_order = _order(2, day)
    excluded_order["state"] = "pending"
    outside_order = _order(3, date(2026, 6, 23))
    orphan_item = {
        "id": 999,
        "order_id": 999,
        "product_name": "croissant",
        "quantity": 1,
        "unit_price": Decimal("12.0"),
        "line_total": Decimal("12.0"),
        "line_profit": Decimal("5.0"),
    }
    excluded_item = dict(orphan_item, id=200, order_id=2)
    orphan_payment = dict(_payment(999, day), id=999)
    excluded_payment = dict(_payment(2, day), id=200)
    cross_scope_payment = dict(_payment(3, day), id=300)
    extra_receipt = dict(
        _receipt(99, datetime(2026, 6, 24, 9, 0)),
        receipt_id="EXTRA",
    )
    snapshot = DataSnapshot(
        available_tables=frozenset(
            {
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
            }
        ),
        orders=(paid_order, excluded_order, outside_order),
        order_items=tuple(_items(1) + [excluded_item, orphan_item]),
        payments=(
            _payment(1, day),
            excluded_payment,
            cross_scope_payment,
            orphan_payment,
        ),
        receipts=(
            _receipt(1, datetime(2026, 6, 24, 8, 31)),
            extra_receipt,
        ),
    )

    result = _check(
        run_checks(snapshot, scope="operational"),
        "operational_normalized_relationships",
    )

    assert "Item 999 references missing order 999" in result.failures
    assert "Payment 999 references missing order 999" in result.failures
    assert "Item 200 belongs to non-normalized order 2 in operational scope" in result.failures
    assert "Payment 200 belongs to non-normalized order 2 in operational scope" in result.failures
    assert "Payment 300 date 2026-06-24 belongs to order 3 outside operational scope" in result.failures
    assert "Receipt EXTRA has no normalized order in operational scope" in result.failures


def test_run_checks_accepts_refunded_evidence_but_excludes_paid_metrics():
    day = date(2026, 6, 24)
    paid_order = _order(1, day)
    refunded_order = _order(2, day)
    refunded_order["state"] = "refunded"
    snapshot = DataSnapshot(
        available_tables=frozenset(
            {
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
            }
        ),
        orders=(paid_order, refunded_order),
        order_items=tuple(_items(1) + _items(2)),
        payments=(_payment(1, day), _payment(2, day)),
        receipts=(
            _receipt(1, datetime(2026, 6, 24, 8, 31)),
            _receipt(2, datetime(2026, 6, 24, 9, 31)),
        ),
    )

    results = run_checks(snapshot, scope="operational")

    assert _check(results, "operational_normalized_relationships").passed
    assert _check(results, "operational_daily_completeness").observed[
        day.isoformat()
    ]["orders"] == 1
    assert _check(results, "operational_revenue_reconciliation").observed[
        day.isoformat()
    ]["orders"] == 1


def test_historical_profile_checks_use_approved_ranges_and_catalog_coverage():
    monday = date(2026, 6, 22)
    friday = date(2026, 6, 19)
    saturday = date(2026, 6, 20)
    orders = (
        [_order(i, monday) for i in range(1, 111)]
        + [_order(i, friday) for i in range(111, 231)]
        + [_order(i, saturday) for i in range(231, 366)]
    )
    products = [
        {"product_name": "croissant", "category": "bakery"},
        {"product_name": "latte", "category": "beverage"},
    ]
    items = [
        {"product_name": "croissant", "quantity": 40},
        {"product_name": "latte", "quantity": 20},
    ]

    assert validate_daily_order_ranges(orders).passed
    assert validate_sales_boundary(orders).passed
    assert validate_product_coverage(items, products).passed

    missing = validate_product_coverage(items[:1], products)
    assert "Historical product has no sales: latte" in missing.failures
    assert "Historical sales have no beverage product coverage" in missing.failures


def test_historical_channel_and_concentration_failures_are_named():
    orders = [
        _order(1, date(2026, 6, 23), dine_type="delivery"),
        _order(2, date(2026, 6, 23), dine_type="takeaway"),
    ]
    payments = [
        _payment(1, date(2026, 6, 23), "qr"),
        _payment(2, date(2026, 6, 23), "qr"),
    ]
    channel = validate_service_payment_shares(orders, payments)
    assert "Delivery orders found: 1" in channel.failures
    assert "Takeaway share 0.500000 outside 0.75-0.85" in channel.failures

    concentration = validate_bakery_concentration(
        [
            {"product_name": "a", "quantity": 16},
            {"product_name": "b", "quantity": 84},
        ],
        [
            {"product_name": "a", "category": "bakery"},
            {"product_name": "b", "category": "bakery"},
        ],
    )
    assert "Bakery product b share 0.840000 exceeds 0.15" in concentration.failures


def test_historical_product_mix_requires_depth_without_flattening_leaders():
    start = date(2026, 6, 17)
    bakery_products = [f"bread_{index:02d}" for index in range(30)]
    beverage_products = [f"drink_{index:02d}" for index in range(15)]
    products = [
        {"product_name": name, "category": "bakery"}
        for name in bakery_products
    ] + [
        {"product_name": name, "category": "beverage"}
        for name in beverage_products
    ]
    orders = []
    items = []
    for day_offset in range(7):
        order_id = day_offset + 1
        orders.append(_order(order_id, start + timedelta(days=day_offset)))
        omitted_bakery = bakery_products[day_offset]
        omitted_beverage = beverage_products[day_offset]
        items.extend(
            {
                "order_id": order_id,
                "product_name": name,
                "quantity": 15 if index < 7 else 3,
            }
            for index, name in enumerate(bakery_products)
            if name != omitted_bakery
        )
        items.extend(
            {
                "order_id": order_id,
                "product_name": name,
                "quantity": 4,
            }
            for name in beverage_products
            if name != omitted_beverage
        )

    result = validate_historical_product_mix(orders, items, products)
    assert result.passed
    assert 0.55 <= result.observed["bakery_top_seven_share"] <= 0.65

    sparse_items = [
        row
        for row in items
        if not (
            row["order_id"] == 1
            and row["product_name"] in bakery_products[10:]
        )
    ]
    sparse = validate_historical_product_mix(orders, sparse_items, products)
    assert not sparse.passed
    assert "Minimum daily bakery coverage 9 is below 25" in sparse.failures

    fully_dense_items = []
    for day_offset in range(7):
        order_id = day_offset + 1
        fully_dense_items.extend(
            {
                "order_id": order_id,
                "product_name": name,
                "quantity": 15 if index < 7 else 3,
            }
            for index, name in enumerate(bakery_products)
        )
        fully_dense_items.extend(
            {
                "order_id": order_id,
                "product_name": name,
                "quantity": 4,
            }
            for name in beverage_products
        )
    fully_dense = validate_historical_product_mix(
        orders, fully_dense_items, products
    )
    assert not fully_dense.passed
    assert (
        "Average daily bakery coverage 30.000 exceeds 29.5"
        in fully_dense.failures
    )
    assert (
        "Average daily beverage coverage 15.000 exceeds 14.8"
        in fully_dense.failures
    )


def test_operational_completeness_requires_every_date_and_each_evidence_type():
    orders = [_order(1, date(2026, 6, 24))]
    result = validate_operational_completeness(
        orders,
        _items(1),
        [_payment(1, date(2026, 6, 24))],
        [_receipt(1, datetime(2026, 6, 24, 8, 31))],
    )

    assert isinstance(result, CheckResult)
    assert not result.passed
    assert "Missing operational orders on 2026-06-25" in result.failures
    assert "Missing operational receipts on 2026-06-25" in result.failures


def test_operational_product_coverage_rejects_excessive_zero_sale_products():
    start = date(2026, 6, 24)
    products = [
        {"product_name": f"bread_{index:02d}", "category": "bakery"}
        for index in range(30)
    ] + [
        {"product_name": f"drink_{index:02d}", "category": "beverage"}
        for index in range(15)
    ]
    orders = [_order(1, start), _order(2, start + timedelta(days=1))]
    items = []
    for order in orders:
        items.extend(
            {
                "order_id": order["id"],
                "product_name": product["product_name"],
                "quantity": 1,
            }
            for product in products
        )

    result = validate_operational_product_coverage(orders, items, products)
    assert result.passed
    assert result.observed["minimum_daily_bakery_coverage"] == 30
    assert result.observed["minimum_daily_beverage_coverage"] == 15

    sparse = validate_operational_product_coverage(
        orders,
        [
            item
            for item in items
            if item["product_name"] in {"bread_00", "drink_00"}
        ],
        products,
    )
    assert not sparse.passed
    assert "Minimum operational daily bakery coverage 1 is below 25" in sparse.failures
    assert "Minimum operational daily beverage coverage 1 is below 12" in sparse.failures

    rehearsal_items = [
        item for item in items if item["order_id"] == 1 and item["product_name"] != "bread_29"
    ]
    rehearsal = validate_operational_product_coverage(
        [orders[0]],
        rehearsal_items,
        products,
        start=start,
        end=start,
    )
    assert rehearsal.passed
    assert rehearsal.observed["weekly_recurrence_evaluated"] is False


def test_business_events_require_matching_sales_and_discount_receipt_evidence():
    events = [
        {
            "id": 1,
            "event_type": "new_product_launch",
            "start_date": date(2026, 7, 3),
            "end_date": date(2026, 7, 5),
            "products": json.dumps(["bread_roll"]),
            "discount_pct": Decimal("12.0"),
            "active": 1,
        },
        {
            "id": 2,
            "event_type": "competitor_activity",
            "start_date": date(2026, 7, 17),
            "end_date": date(2026, 7, 19),
            "products": json.dumps(["stickbread"]),
            "discount_pct": Decimal("10.0"),
            "active": 1,
        },
    ]
    receipts = [
        {
            "receipt_id": "R-1",
            "created_at": datetime(2026, 7, 3, 8, 0),
            "items": json.dumps(
                [
                    {
                        "product_name": "bread_roll",
                        "discount_pct": 12,
                        "discount_source": "business_event",
                    }
                ]
            ),
        },
        {
            "receipt_id": "R-2",
            "created_at": datetime(2026, 7, 17, 8, 0),
            "items": json.dumps(
                [
                    {
                        "product_name": "stickbread",
                        "discount_pct": 10,
                        "discount_source": "business_event",
                    }
                ]
            ),
        },
    ]

    result = validate_business_event_evidence(events, receipts)
    assert result.passed
    assert result.observed["event_types"] == [
        "competitor_activity",
        "new_product_launch",
    ]

    missing = validate_business_event_evidence(events, receipts[:1])
    assert not missing.passed
    assert (
        "Business event 2 has no matching discounted receipt evidence for stickbread"
        in missing.failures
    )

    duplicate_events = events + [dict(events[0], id=3)]
    duplicate = validate_business_event_evidence(duplicate_events, receipts)
    assert not duplicate.passed
    assert (
        "Duplicate business event 3 matches event 1"
        in duplicate.failures
    )


def test_operational_checks_can_be_limited_to_one_rehearsal_day():
    day = date(2026, 6, 24)
    orders = [_order(1, day)]
    completeness = validate_operational_completeness(
        orders,
        _items(1),
        [_payment(1, day)],
        [_receipt(1, datetime(2026, 6, 24, 8, 31))],
        start=day,
        end=day,
    )
    material_checks = validate_weekly_material_checks(
        [{"material_name": "Flour", "check_date": day}],
        [{"material_name": "Flour", "track_inventory": 1}],
        start=day,
        end=day,
    )
    top3_events = validate_top3_events(
        [
            {
                "id": rank,
                "request_id": "R1",
                "operation_date": day,
                "shown_at": datetime(2026, 6, 24, 9, rank),
                "rank_position": rank,
                "selected_at": None,
                "purchased_order_id": None,
            }
            for rank in (1, 2, 3)
        ],
        orders,
        start=day,
        end=day,
    )

    assert completeness.passed
    assert material_checks.passed
    assert top3_events.passed
    assert tuple(completeness.observed) == (day.isoformat(),)
    assert tuple(top3_events.observed) == (day.isoformat(),)


def test_batch_equations_and_beverage_batch_rule_reconcile():
    batches = [
        {
            "batch_id": "B1",
            "product_name": "croissant",
            "quantity_initial": 10,
            "quantity_remaining": 2,
        },
        {
            "batch_id": "B2",
            "product_name": "latte",
            "quantity_initial": 1,
            "quantity_remaining": 1,
        },
    ]
    transactions = [
        {
            "batch_id": "B1",
            "receipt_id": "R1",
            "transaction_type": "outflow",
            "quantity": 7,
            "disposition": "sold",
        },
        {"batch_id": "B1", "transaction_type": "outflow", "quantity": 1, "disposition": "discarded"},
    ]
    products = [
        {"product_name": "croissant", "category": "bakery"},
        {"product_name": "latte", "category": "beverage"},
    ]

    result = validate_batch_equations(batches, transactions, products)

    assert result.failures == ("Beverage batch inventory found for B2",)


def test_batch_equations_reject_missing_and_untraceable_batch_ids():
    products = [{"product_name": "croissant", "category": "bakery"}]
    transactions = [
        {
            "id": 1,
            "batch_id": None,
            "product_name": "croissant",
            "transaction_type": "outflow",
            "quantity": 1,
            "disposition": "sold",
            "receipt_id": "T-1",
            "transaction_time": datetime(2026, 6, 24, 8, 30),
        },
        {
            "id": 2,
            "batch_id": "B404",
            "product_name": "croissant",
            "transaction_type": "outflow",
            "quantity": 1,
            "disposition": "discarded",
            "transaction_time": datetime(2026, 6, 24, 19, 0),
        },
    ]

    result = validate_batch_equations([], transactions, products)

    assert "Bakery inventory transaction 1 has no batch_id" in result.failures
    assert "Inventory transaction 2 references missing batch B404" in result.failures


def test_batch_lifecycle_scope_includes_prior_batch_and_future_day1_sale():
    batch = {
        "batch_id": "B-PREVIOUS",
        "product_name": "croissant",
        "production_time": datetime(2026, 6, 30, 13, 30),
        "quantity_initial": 2,
        "quantity_remaining": 0,
    }
    transactions = (
        {
            "id": 1,
            "batch_id": "B-PREVIOUS",
            "product_name": "croissant",
            "transaction_type": "outflow",
            "quantity": 1,
            "disposition": "sold",
            "receipt_id": "R-1",
            "transaction_time": datetime(2026, 7, 1, 9, 0),
        },
        {
            "id": 2,
            "batch_id": "B-PREVIOUS",
            "product_name": "croissant",
            "transaction_type": "outflow",
            "quantity": 1,
            "disposition": "sold",
            "receipt_id": "R-2",
            "transaction_time": datetime(2026, 7, 2, 9, 0),
        },
    )
    snapshot = DataSnapshot(
        available_tables=frozenset(),
        batch_inventory=(batch,),
        inventory_transactions=transactions,
    )

    batches, lifecycle_transactions = _batch_lifecycle_rows(
        snapshot,
        date(2026, 7, 1),
        date(2026, 7, 1),
    )
    result = validate_batch_equations(
        batches,
        lifecycle_transactions,
        [{"product_name": "croissant", "category": "bakery"}],
    )

    assert batches == [batch]
    assert lifecycle_transactions == list(transactions)
    assert result.passed


def test_material_scope_and_restock_timing_use_transaction_sources():
    products = [
        {"product_name": "croissant", "category": "bakery"},
        {"product_name": "latte", "category": "beverage"},
    ]
    recipes = [
        {"product_name": "croissant", "material_name": "Flour"},
        {"product_name": "latte", "material_name": "Milk"},
    ]
    raw_materials = [
        {"material_name": "Flour", "stock_quantity": Decimal("2"), "category": "baking"},
        {"material_name": "Milk", "stock_quantity": Decimal("1"), "category": "beverage"},
        {"material_name": "Cup Regular", "stock_quantity": Decimal("5"), "category": "packaging"},
    ]
    transactions = [
        {"id": 1, "material_name": "Flour", "transaction_type": "outflow", "quantity": Decimal("1"), "reference": "POS-1", "created_at": datetime(2026, 6, 24, 9)},
        {"id": 2, "material_name": "Milk", "transaction_type": "outflow", "quantity": Decimal("2"), "reference": "POS-1", "created_at": datetime(2026, 6, 24, 9)},
        {"id": 3, "material_name": "Milk", "transaction_type": "restock", "quantity": Decimal("2"), "reference": "manual_restock", "created_at": datetime(2026, 6, 24, 10)},
    ]

    scope = validate_material_outflow_scope(
        transactions,
        recipes,
        products,
        raw_materials,
        receipt_ids={"POS-1"},
    )
    assert scope.failures == ("Bakery material Flour has checkout outflow 1",)

    timing = validate_restock_timing(transactions, raw_materials)
    assert timing.failures == ("Material Milk became negative at transaction 2",)


def test_material_outflows_are_required_from_canonical_recipes_and_checkout():
    day = date(2026, 6, 24)
    products = [
        {"product_name": "croissant", "category": "bakery", "wastage_pct": Decimal("0.1")},
        {"product_name": "latte", "category": "beverage", "wastage_pct": Decimal("0.1")},
    ]
    recipes = [
        {"product_name": "croissant", "material_name": "Flour", "quantity_per_unit": Decimal("0.1")},
        {"product_name": "latte", "material_name": "Milk", "quantity_per_unit": Decimal("0.2")},
    ]
    raw_materials = [
        {"material_name": "Flour", "unit": "kg", "category": "baking", "track_inventory": 1},
        {"material_name": "Milk", "unit": "kg", "category": "beverage", "track_inventory": 1},
        {"material_name": "Cup Regular", "unit": "pcs", "category": "packaging", "track_inventory": 1},
        {"material_name": "Packaging Bag", "unit": "pcs", "category": "packaging", "track_inventory": 1},
    ]
    batch = {
        "batch_id": "B1",
        "product_name": "croissant",
        "quantity_initial": 10,
        "quantity_remaining": 10,
        "production_time": datetime(2026, 6, 24, 5, 0),
    }
    order = _order(1, day, ticket_id="T-1", dine_type="takeaway")
    items = [
        dict(_items(1)[0], coffee_size=None),
        dict(_items(1)[1], coffee_size="regular"),
    ]
    receipt = _receipt(1, datetime(2026, 6, 24, 8, 31))

    missing = validate_material_outflow_scope(
        [],
        recipes,
        products,
        raw_materials,
        batches=[batch],
        orders=[order],
        items=items,
        receipts=[receipt],
    )

    assert "Missing production material outflow production:20260624050000000000/Flour on 2026-06-24: expected 1.100000" in missing.failures
    assert "Missing checkout material outflow T-1/Milk on 2026-06-24: expected 0.220000" in missing.failures
    assert "Missing checkout material outflow T-1/Cup Regular on 2026-06-24: expected 1.000000" in missing.failures
    assert "Missing checkout material outflow T-1/Packaging Bag on 2026-06-24: expected 1.000000" in missing.failures

    cross_date = validate_material_outflow_scope(
        [
            {"id": 1, "material_name": "Flour", "transaction_type": "outflow", "quantity": Decimal("1.100000"), "unit": "kg", "reference": "production:20260624050000000000", "created_at": datetime(2026, 6, 24, 5, 0)},
            {"id": 2, "material_name": "Milk", "transaction_type": "outflow", "quantity": Decimal("0.220000"), "unit": "kg", "reference": "T-1", "created_at": datetime(2026, 6, 25, 8, 31)},
            {"id": 3, "material_name": "Cup Regular", "transaction_type": "outflow", "quantity": Decimal("1.000000"), "unit": "pcs", "reference": "T-1", "created_at": datetime(2026, 6, 24, 8, 31)},
            {"id": 4, "material_name": "Packaging Bag", "transaction_type": "outflow", "quantity": Decimal("1.000000"), "unit": "pcs", "reference": "T-1", "created_at": datetime(2026, 6, 24, 8, 31)},
        ],
        recipes,
        products,
        raw_materials,
        batches=[batch],
        orders=[order],
        items=items,
        receipts=[receipt],
    )
    assert "Material outflow 2 reference T-1 date 2026-06-25 does not match receipt date 2026-06-24" in cross_date.failures


def test_checkout_materials_use_canonical_null_wastage_fallback():
    day = date(2026, 6, 24)
    order = _order(
        1,
        day,
        ticket_id="T-1",
        item_count=1,
        total_amount="12.0",
        dine_type="dine_in",
    )
    item = {
        "id": 11,
        "order_id": 1,
        "product_name": "latte",
        "quantity": 1,
        "coffee_size": "regular",
    }
    receipt = _receipt(1, datetime(2026, 6, 24, 8, 31))
    transactions = [
        {
            "id": 1,
            "material_name": "Milk",
            "transaction_type": "outflow",
            "quantity": Decimal("0.206000"),
            "unit": "kg",
            "reference": "T-1",
            "created_at": datetime(2026, 6, 24, 8, 31),
        },
        {
            "id": 2,
            "material_name": "Cup Regular",
            "transaction_type": "outflow",
            "quantity": Decimal("1.000000"),
            "unit": "pcs",
            "reference": "T-1",
            "created_at": datetime(2026, 6, 24, 8, 31),
        },
    ]

    result = validate_material_outflow_scope(
        transactions,
        [
            {
                "product_name": "latte",
                "material_name": "Milk",
                "quantity_per_unit": Decimal("0.200000"),
            }
        ],
        [
            {
                "product_name": "latte",
                "category": "beverage",
                "wastage_pct": None,
            }
        ],
        [
            {
                "material_name": "Milk",
                "unit": "kg",
                "category": "beverage",
                "track_inventory": 1,
            },
            {
                "material_name": "Cup Regular",
                "unit": "pcs",
                "category": "packaging",
                "track_inventory": 1,
            },
        ],
        orders=[order],
        items=[item],
        receipts=[receipt],
    )

    assert result.passed


def test_weekly_material_checks_require_all_tracked_materials():
    tracked = [
        {"material_name": "Flour", "track_inventory": 1},
        {"material_name": "Milk", "track_inventory": 1},
    ]
    checks = [
        {"material_name": "Flour", "check_date": date(2026, 6, 24)},
    ]

    result = validate_weekly_material_checks(checks, tracked)

    assert result.failures == (
        "Full material check 2026-06-24 missing Milk",
        "Full material check 2026-07-01 missing Flour, Milk",
        "Full material check 2026-07-08 missing Flour, Milk",
        "Full material check 2026-07-15 missing Flour, Milk",
        "Full material check 2026-07-22 missing Flour, Milk",
    )

    empty_catalog = validate_weekly_material_checks([], [])
    assert empty_catalog.failures == ("No tracked raw materials found",)


def test_material_checks_require_small_nonzero_variance_after_baseline():
    checks = []
    for check_date in (date(2026, 7, 1), date(2026, 7, 8)):
        checks.extend(
            {
                "material_name": f"Material {index}",
                "check_date": check_date,
                "wastage_qty": Decimal("0.010") if index < 4 else Decimal("0"),
                "wastage_rate": Decimal("0.008") if index < 4 else Decimal("0"),
            }
            for index in range(8)
        )

    result = validate_material_check_variance(
        checks,
        start=date(2026, 7, 1),
        end=date(2026, 7, 8),
    )

    assert result.passed
    all_zero = [dict(row, wastage_qty=Decimal("0"), wastage_rate=Decimal("0")) for row in checks]
    failed = validate_material_check_variance(
        all_zero,
        start=date(2026, 7, 1),
        end=date(2026, 7, 8),
    )
    assert failed.failures == (
        "Material check 2026-07-01 has no recorded operational variance",
        "Material check 2026-07-08 has no recorded operational variance",
    )


def test_attendance_reuses_schedule_status_contract():
    schedules = [
        {
            "schedule_date": date(2026, 6, 24),
            "time_slot": "06:00-13:00",
            "employee_id": "E1",
            "employee_name": "Ada",
            "role": "bakery",
        }
    ]
    records = [
        {
            "emp_id": "E1",
            "date": date(2026, 6, 24),
            "punch_in": time(6, 5),
            "punch_out": time(13, 5),
            "status": "on_time",
        }
    ]

    assert validate_schedule_attendance(
        schedules,
        records,
        start=date(2026, 6, 24),
        end=date(2026, 6, 24),
    ).passed
    missing_day = validate_schedule_attendance(
        schedules,
        records,
        start=date(2026, 6, 24),
        end=date(2026, 6, 25),
    )
    assert missing_day.failures == ("Missing operational schedule on 2026-06-25",)
    status = validate_punch_status(schedules, records)
    assert status.failures == (
        "Attendance E1 on 2026-06-24 has status on_time, expected late",
    )


def test_attendance_treats_missing_punch_record_as_absence():
    day = date(2026, 6, 24)
    schedules = [
        {
            "schedule_date": day,
            "time_slot": "06:00-13:00",
            "employee_id": "E1",
            "employee_name": "Ada",
            "role": "bakery",
        }
    ]

    coverage = validate_schedule_attendance(
        schedules,
        [],
        start=day,
        end=day,
    )
    punch = validate_punch_status(schedules, [])

    assert coverage.passed
    assert punch.passed

    present = {
        "emp_id": "E1",
        "date": day,
        "punch_in": time(5, 55),
        "punch_out": None,
        "status": "present",
    }
    assert validate_schedule_attendance(
        schedules,
        [present],
        start=day,
        end=day,
    ).passed
    assert validate_punch_status(schedules, [present]).passed


def test_schedule_and_punch_reject_missing_canonical_window():
    day = date(2026, 6, 24)
    schedules = [
        {
            "schedule_date": day,
            "time_slot": "not-a-window",
            "employee_id": "E1",
            "employee_name": "Ada",
            "role": "bakery",
        }
    ]
    records = [
        {
            "emp_id": "E1",
            "date": day,
            "punch_in": time(6, 0),
            "punch_out": time(13, 0),
            "status": "on_time",
        }
    ]

    coverage = validate_schedule_attendance(
        schedules,
        records,
        start=day,
        end=day,
    )
    punch = validate_punch_status(schedules, records)

    failure = "Schedule E1 on 2026-06-24 has no canonical window"
    assert coverage.failures == (failure,)
    assert punch.failures == (failure,)


def test_top3_counts_allow_zero_conversion_but_reconcile_lifecycle():
    day = date(2026, 6, 24)
    events = [
        {"id": 1, "request_id": "R1", "operation_date": day, "rank_position": 1, "selected_at": None, "purchased_order_id": None},
        {"id": 2, "request_id": "R1", "operation_date": day, "rank_position": 2, "selected_at": None, "purchased_order_id": None},
        {"id": 3, "request_id": "R1", "operation_date": day, "rank_position": 3, "selected_at": None, "purchased_order_id": None},
    ]

    result = validate_top3_events(events, [])

    assert "Missing Top 3 exposure on 2026-06-25" in result.failures
    assert not any("selection" in failure.lower() for failure in result.failures)
    assert not any("purchase" in failure.lower() for failure in result.failures)

    events[0]["purchased_order_id"] = 999
    lifecycle = validate_top3_events(events, [])
    assert "Recommendation event 1 was purchased without selection" in lifecycle.failures


def test_top3_rejects_cross_date_and_reversed_lifecycle_timestamps():
    events = []
    event_id = 1
    for offset in range(31):
        operation_date = date(2026, 6, 24) + timedelta(days=offset)
        for rank in (1, 2, 3):
            shown_at = datetime.combine(operation_date, time(9, rank))
            selected_at = None
            if event_id == 1:
                shown_at = datetime(2021, 1, 1, 9, 0)
                selected_at = datetime(2020, 1, 1, 9, 0)
            events.append(
                {
                    "id": event_id,
                    "request_id": f"R-{operation_date.isoformat()}",
                    "operation_date": operation_date,
                    "shown_at": shown_at,
                    "rank_position": rank,
                    "selected_at": selected_at,
                    "purchased_order_id": None,
                }
            )
            event_id += 1

    result = validate_top3_events(events, [])

    assert "Recommendation event 1 shown_at 2021-01-01 does not match operation_date 2026-06-24" in result.failures
    assert "Recommendation event 1 selected_at 2020-01-01 does not match operation_date 2026-06-24" in result.failures
    assert "Recommendation event 1 selected_at precedes shown_at" in result.failures


def test_inventory_and_wastage_source_equations_are_explicit():
    inventory = validate_inventory_source_equations(
        [{"id": 1, "order_id": 10, "product_name": "croissant", "quantity": 2}],
        [{"id": 2, "product_name": "croissant", "quantity": 1, "receipt_id": "T-10", "transaction_type": "outflow", "disposition": "sold"}],
        [{"id": 10, "ticket_id": "T-10"}],
        [{"product_name": "croissant", "category": "bakery"}],
    )
    assert inventory.failures == (
        "Order 10 bakery quantity for croissant is 2, inventory sold outflow is 1",
    )

    extra = validate_inventory_source_equations(
        [],
        [{"id": 3, "product_name": "croissant", "quantity": 1, "receipt_id": "T-11", "transaction_type": "outflow", "disposition": "sold"}],
        [],
        [{"product_name": "croissant", "category": "bakery"}],
    )
    assert extra.failures == (
        "Inventory sold outflow for unknown bakery order T-11/croissant is 1",
    )

    wastage = validate_wastage_source_equations(
        [
            {
                "id": 1,
                "material_name": "Flour",
                "theoretical_stock": Decimal("8"),
                "actual_stock": Decimal("7"),
                "theoretical_consumed": Decimal("2"),
                "actual_consumed": Decimal("3"),
                "wastage_qty": Decimal("-1"),
                "wastage_rate": Decimal("-0.5"),
            }
        ]
    )
    assert "Negative material wastage in check 1" in wastage.failures


def test_wastage_validator_accepts_zero_consumption_precision_residue():
    result = validate_wastage_source_equations(
        [
            {
                "id": 1,
                "theoretical_stock": Decimal("5.601"),
                "actual_stock": Decimal("5.601"),
                "theoretical_consumed": Decimal("0"),
                "actual_consumed": Decimal("-0.001"),
                "wastage_qty": Decimal("0"),
                "wastage_rate": Decimal("0"),
            },
            {
                "id": 2,
                "theoretical_stock": Decimal("7.795"),
                "actual_stock": Decimal("7.795"),
                "theoretical_consumed": Decimal("0"),
                "actual_consumed": Decimal("0.001"),
                "wastage_qty": Decimal("0.001"),
                "wastage_rate": Decimal("0"),
            },
        ]
    )

    assert result.passed


def test_wastage_validator_accepts_stored_quantity_rounding_tolerance():
    result = validate_wastage_source_equations(
        [
            {
                "id": 3,
                "theoretical_stock": Decimal("19.595"),
                "actual_stock": Decimal("19.595"),
                "theoretical_consumed": Decimal("6.940"),
                "actual_consumed": Decimal("6.939"),
                "wastage_qty": Decimal("0"),
                "wastage_rate": Decimal("0"),
            }
        ]
    )

    assert result.passed


def test_wastage_validator_accepts_one_unit_legacy_wastage_rounding():
    result = validate_wastage_source_equations(
        [
            {
                "id": 4,
                "theoretical_stock": Decimal("7.260"),
                "actual_stock": Decimal("7.259"),
                "theoretical_consumed": Decimal("0.536"),
                "actual_consumed": Decimal("0.536"),
                "wastage_qty": Decimal("0"),
                "wastage_rate": Decimal("0"),
            }
        ]
    )

    assert result.passed


def test_discrete_material_quantities_reject_fractional_piece_counts():
    from scripts.validate_guangzhou_data_contract import (
        validate_discrete_material_quantities,
    )

    result = validate_discrete_material_quantities(
        [
            {
                "material_name": "Cups",
                "unit": "pcs",
                "stock_quantity": Decimal("10.5"),
                "reorder_point": Decimal("5"),
            }
        ],
        [
            {
                "id": 7,
                "material_name": "Cups",
                "quantity": Decimal("1.25"),
            }
        ],
        [
            {
                "id": 9,
                "material_name": "Cups",
                "theoretical_stock": Decimal("9"),
                "actual_stock": Decimal("8.5"),
                "theoretical_consumed": Decimal("1"),
                "actual_consumed": Decimal("1.5"),
                "wastage_qty": Decimal("0.5"),
            }
        ],
    )

    assert result.failures == (
        "Material Cups stock_quantity must be a whole piece count",
        "Material transaction 7 for Cups must use a whole piece quantity",
        "Material check 9 actual_stock for Cups must be a whole piece count",
        "Material check 9 actual_consumed for Cups must be a whole piece count",
        "Material check 9 wastage_qty for Cups must be a whole piece count",
    )


def test_inventory_source_equations_reject_cross_date_sales():
    order_day = date(2026, 6, 24)
    result = validate_inventory_source_equations(
        [{"id": 1, "order_id": 10, "product_name": "croissant", "quantity": 1}],
        [
            {
                "id": 2,
                "batch_id": "B1",
                "product_name": "croissant",
                "quantity": 1,
                "receipt_id": "T-10",
                "transaction_type": "outflow",
                "disposition": "sold",
                "transaction_time": datetime(2026, 7, 24, 9, 0),
            }
        ],
        [_order(10, order_day, ticket_id="T-10", item_count=1, total_amount="12.0")],
        [{"product_name": "croissant", "category": "bakery"}],
    )

    assert "Order 10 on 2026-06-24 bakery quantity for croissant is 1, inventory sold outflow is 0" in result.failures
    assert "Inventory sold outflow on 2026-07-24 for T-10/croissant has no matching bakery order" in result.failures


class ReadOnlyCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []
        self.closed = False

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self.connection.statements.append((normalized, tuple(params or ())))
        if "information_schema.tables" in normalized:
            self.rows = [{"TABLE_NAME": "orders"}]
        elif "FROM orders" in normalized:
            self.rows = []
        else:
            self.rows = []

    def fetchall(self):
        return deepcopy(self.rows)

    def close(self):
        self.closed = True


class ReadOnlyConnection:
    def __init__(self):
        self.statements = []
        self.rollback_count = 0
        self.commit_count = 0
        self.closed = False

    def cursor(self, dictionary=False):
        assert dictionary
        return ReadOnlyCursor(self)

    def rollback(self):
        self.rollback_count += 1

    def commit(self):
        self.commit_count += 1

    def close(self):
        self.closed = True


class SnapshotCursor:
    def __init__(self, connection, close_error=False):
        self.connection = connection
        self.close_error = close_error
        self.rows = []

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self.connection.statements.append((normalized, tuple(params or ())))
        if "information_schema.tables" in normalized:
            self.rows = [
                {"TABLE_NAME": table} for table in self.connection.table_rows
            ]
            return
        if normalized.startswith("SELECT * FROM "):
            table = normalized.split("SELECT * FROM ", 1)[1].split()[0]
            self.rows = deepcopy(self.connection.table_rows.get(table, []))
            return
        self.rows = []

    def fetchall(self):
        return deepcopy(self.rows)

    def close(self):
        if self.close_error:
            raise RuntimeError("cursor close failed")


class SnapshotConnection(ReadOnlyConnection):
    def __init__(self, table_rows, close_error=False):
        super().__init__()
        self.table_rows = table_rows
        self.close_error = close_error

    def cursor(self, dictionary=False):
        assert dictionary
        return SnapshotCursor(self, self.close_error)


def test_cli_is_read_only_reports_missing_tables_and_exits_nonzero(tmp_path, capsys):
    connection = ReadOnlyConnection()
    output = tmp_path / "validator.json"

    exit_code = main(
        ["--scope", "operational", "--output", str(output)],
        connect=lambda **kwargs: connection,
    )

    assert exit_code == 1
    assert connection.rollback_count == 1
    assert connection.commit_count == 0
    assert connection.closed
    statements = [sql for sql, _ in connection.statements]
    assert statements[0] == "START TRANSACTION READ ONLY"
    assert all(
        sql.startswith("SELECT") or sql == "START TRANSACTION READ ONLY"
        for sql in statements
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert not payload["passed"]
    assert "operational_required_table_attendance_records" in capsys.readouterr().out


def test_cli_accepts_only_a_bounded_operational_date_range():
    args = parse_args(
        [
            "--scope",
            "operational",
            "--start",
            "2026-06-24",
            "--end",
            "2026-06-24",
            "--output",
            "validator.json",
        ]
    )

    assert args.start == date(2026, 6, 24)
    assert args.end == date(2026, 6, 24)

    invalid_arguments = (
        ["--scope", "operational", "--start", "2026-06-24"],
        [
            "--scope",
            "historical",
            "--start",
            "2026-06-24",
            "--end",
            "2026-06-24",
        ],
        [
            "--scope",
            "operational",
            "--start",
            "2026-06-23",
            "--end",
            "2026-06-24",
        ],
        [
            "--scope",
            "operational",
            "--start",
            "2026-06-25",
            "--end",
            "2026-06-24",
        ],
    )
    for invalid in invalid_arguments:
        try:
            parse_args([*invalid, "--output", "validator.json"])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"Expected invalid date range to fail: {invalid}")


def test_collect_snapshot_rolls_back_when_cursor_close_fails():
    connection = SnapshotConnection({"orders": []}, close_error=True)

    try:
        collect_snapshot(connection, scope="historical")
    except RuntimeError as exc:
        assert str(exc) == "cursor close failed"
    else:
        raise AssertionError("Expected cursor close failure")

    assert connection.rollback_count == 1


def test_cli_historical_scope_reports_untrimmed_cutoff_leakage(tmp_path, capsys):
    start = date(2021, 1, 1)
    end = date(2026, 6, 23)
    orders = [
        _order(
            index + 1,
            start + timedelta(days=index),
            ticket_id=f"GZ-{start + timedelta(days=index):%Y%m%d}-0001",
            item_count=1,
            total_amount="12.0",
        )
        for index in range((end - start).days + 1)
    ]
    orders.append(
        _order(
            900002,
            date(2026, 6, 24),
            ticket_id="GZ-20260624-0001",
            item_count=1,
            total_amount="12.0",
        )
    )
    connection = SnapshotConnection(
        {
            "orders": orders,
            "order_items": [],
            "payments": [],
            "products": [],
        }
    )
    output = tmp_path / "historical.json"

    exit_code = main(
        ["--scope", "historical", "--output", str(output)],
        connect=lambda **_kwargs: connection,
    )

    assert exit_code == 1
    assert "historical_date_range" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    result = next(
        check for check in payload["checks"] if check["name"] == "historical_date_range"
    )
    assert "Historical order on or after operation cutoff: 2026-06-24" in result["failures"]


def test_cli_operational_scope_reports_unmatched_receipt(tmp_path, capsys):
    day = date(2026, 6, 24)
    tables = {
        "attendance_records": [],
        "batch_inventory": [],
        "business_events": [],
        "inventory_transactions": [],
        "material_transactions": [],
        "material_wastage_log": [],
        "order_items": _items(1),
        "orders": [_order(1, day)],
        "payments": [_payment(1, day)],
        "product_recipes": [],
        "products": [],
        "raw_materials": [],
        "receipts": [
            _receipt(1, datetime(2026, 6, 24, 8, 31)),
            dict(
                _receipt(99, datetime(2026, 6, 24, 9, 0)),
                receipt_id="EXTRA",
            ),
        ],
        "recommendation_events": [],
        "shift_schedule": [],
    }
    connection = SnapshotConnection(tables)
    output = tmp_path / "operational.json"

    exit_code = main(
        ["--scope", "operational", "--output", str(output)],
        connect=lambda **_kwargs: connection,
    )

    assert exit_code == 1
    assert "operational_normalized_relationships" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    result = next(
        check
        for check in payload["checks"]
        if check["name"] == "operational_normalized_relationships"
    )
    assert "Receipt EXTRA has no normalized order in operational scope" in result["failures"]


def test_run_checks_returns_check_result_for_each_validation_function():
    snapshot = DataSnapshot(available_tables=frozenset())

    results = run_checks(snapshot, scope="all")

    assert results
    assert all(isinstance(result, CheckResult) for result in results)
    assert len({result.name for result in results}) == len(results)
