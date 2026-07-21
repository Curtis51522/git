import csv
import copy
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError
from datetime import date, time
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from api.module4_frontend.beverage_options import beverage_capabilities, is_beverage
import scripts.rebuild_guangzhou_sales_history as rebuild
from scripts.rebuild_guangzhou_sales_history import (
    BASKET_SIZE_WEIGHTS,
    HISTORY_END,
    HISTORY_START,
    OPERATION_START,
    ORDER_RANGES,
    PAYMENT_RANGES,
    RANDOM_SEED,
    STORE_CLOSE,
    STORE_OPEN,
    TAKEAWAY_RANGE,
    BuildPaths,
    History,
    ItemRecord,
    OrderRecord,
    PaymentRecord,
    SourceRecord,
    WeatherRecord,
    _basket_sizes,
    _beverage_details,
    _cap_bakery_weights,
    _donor_profile,
    _index_source,
    _ticket_times,
    apply_history,
    build_history,
    create_product_cost_catalog,
    generate_outputs,
    load_product_costs,
    load_weather_csv,
    daily_order_target,
    daily_unit_target,
    money,
    parse_args,
    summarize_history,
    validate_apply_request,
)


BAKERY_PRODUCTS = (
    "croissant",
    "baguette",
    "cookie",
    "brownie",
    "donut",
    "bread_roll",
    "muffin",
    "sourdough",
)
BEVERAGE_PRODUCTS = ("latte", "lemonade")
PRICES = {
    "croissant": Decimal("8.58"),
    "baguette": Decimal("7.02"),
    "cookie": Decimal("6.24"),
    "brownie": Decimal("15.60"),
    "donut": Decimal("6.00"),
    "bread_roll": Decimal("6.50"),
    "muffin": Decimal("12.00"),
    "sourdough": Decimal("22.00"),
    "latte": Decimal("18.00"),
    "lemonade": Decimal("12.00"),
}
PRODUCT_COSTS = {name: Decimal("1.0") for name in PRICES}
APPLY_COSTS = {"croissant": Decimal("1.0")}


def _weekday_class(day):
    if day.weekday() >= 5:
        return "weekend"
    if day.weekday() == 4:
        return "friday"
    return "weekday"


def _first_date_for_class(year, month, target_class):
    day = date(year, month, 1)
    while day.month == month:
        if _weekday_class(day) == target_class:
            return day
        day = date.fromordinal(day.toordinal() + 1)
    raise AssertionError("No matching date")


@pytest.fixture(scope="module")
def source_fixture():
    rows = []
    ticket_number = 1
    classes = ("weekday", "friday", "weekend")
    for month in range(1, 13):
        for weekday_class in classes:
            source_day = _first_date_for_class(2023, month, weekday_class)
            quantities = (30, 25, 20, 10, 8, 7, 5, 3, 6 + month, 20 - month)
            for product_name, quantity in zip(
                BAKERY_PRODUCTS + BEVERAGE_PRODUCTS, quantities, strict=True
            ):
                rows.append(
                    SourceRecord(
                        order_date=source_day,
                        order_time=time(8, ticket_number % 60),
                        ticket_id=str(ticket_number),
                        product_name=product_name,
                        quantity=quantity,
                        unit_price=PRICES[product_name],
                        discount_rate=Decimal("0.00"),
                    )
                )
                ticket_number += 1
    rows.append(
        SourceRecord(
            order_date=date(2023, 12, 31),
            order_time=time(9, 0),
            ticket_id=str(ticket_number),
            product_name="croissant",
            quantity=1,
            unit_price=PRICES["croissant"],
            discount_rate=Decimal("0.00"),
        )
    )
    return tuple(rows)


@pytest.fixture(scope="module")
def history(source_fixture):
    return build_history(source_fixture, PRODUCT_COSTS, seed=RANDOM_SEED)


@pytest.fixture(scope="module")
def canonical_one_year_history():
    source_rows = rebuild.load_source_csv(
        Path("data/bakery_sales_raw_backup_cleaned.csv")
    )
    product_costs = load_product_costs(
        Path("data/reference/product_cost_catalog.json")
    )
    return build_history(source_rows, product_costs, seed=RANDOM_SEED)


def test_guangzhou_profile_contract():
    assert HISTORY_START == date(2025, 6, 24)
    assert HISTORY_END == date(2026, 6, 23)
    assert OPERATION_START == date(2026, 6, 24)
    assert STORE_OPEN == time(6, 0)
    assert STORE_CLOSE == time(19, 0)
    assert RANDOM_SEED == 20260718
    assert ORDER_RANGES == {
        "weekday": (110, 145),
        "friday": (120, 155),
        "weekend": (135, 175),
    }
    assert BASKET_SIZE_WEIGHTS == {1: 52, 2: 31, 3: 11, 4: 3, 5: 2, 6: 1}
    assert TAKEAWAY_RANGE == (0.75, 0.85)
    assert PAYMENT_RANGES == {
        "qr": (0.85, 0.92),
        "card": (0.05, 0.10),
        "cash": (0.02, 0.05),
    }


def test_cli_default_paths_are_locked():
    args = parse_args([])
    assert args.source == Path("data/bakery_sales_raw_backup_cleaned.csv")
    assert args.raw_output == Path("data/bakery_sales_raw.csv")
    assert args.manifest == Path("data/guangzhou_rebuild_manifest.json")


def test_script_entrypoint_runs_from_repository_root():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/rebuild_guangzhou_sales_history.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_history_apply_is_off_by_default():
    assert parse_args([]).apply_history is False


def test_history_apply_requires_exact_database_confirmation():
    with pytest.raises(SystemExit):
        validate_apply_request(True, None)
    with pytest.raises(SystemExit):
        validate_apply_request(True, "other")
    validate_apply_request(True, "bakery_ai")


def test_normalized_records_are_immutable():
    order = OrderRecord(
        ticket_id="GZ-20210101-0001",
        order_date=date(2021, 1, 1),
        order_time=time(8, 0),
        subtotal=Decimal("10.00"),
        discount_total=Decimal("0.00"),
        total_amount=Decimal("10.00"),
        total_profit=Decimal("0.00"),
        item_count=1,
        state="completed",
        dine_type="takeaway",
    )
    item = ItemRecord(
        ticket_id=order.ticket_id,
        product_name="croissant",
        quantity=1,
        unit_price=Decimal("10.00"),
        discount_rate=Decimal("0.00"),
        line_total=Decimal("10.00"),
        line_profit=Decimal("0.00"),
        freshness="Fresh",
    )
    payment = PaymentRecord(
        ticket_id=order.ticket_id,
        amount=Decimal("10.00"),
        payment_method="qr",
        payment_date=order.order_date,
    )
    assert isinstance(History((order,), (item,), (payment,)), History)
    for record in (order, item, payment):
        with pytest.raises(FrozenInstanceError):
            record.ticket_id = "changed"


def test_daily_order_target_uses_approved_ranges():
    cases = (
        (date(2026, 6, 22), False, "weekday"),
        (date(2026, 6, 19), False, "friday"),
        (date(2026, 6, 21), False, "weekend"),
        (date(2026, 6, 22), True, "weekend"),
    )
    for day, is_holiday, range_name in cases:
        low, high = ORDER_RANGES[range_name]
        values = {
            daily_order_target(day, random.Random(seed), is_holiday)
            for seed in range(100)
        }
        assert values
        assert min(values) >= low
        assert max(values) <= high


def test_daily_weather_changes_traffic_without_breaking_order_ranges():
    day = date(2026, 6, 22)
    clear = WeatherRecord(temp_mean=24.0, precipitation=0.0)
    heavy_rain = WeatherRecord(temp_mean=24.0, precipitation=30.0)
    clear_target = daily_order_target(day, random.Random(12), False, clear)
    rain_target = daily_order_target(day, random.Random(12), False, heavy_rain)
    assert ORDER_RANGES["weekday"][0] <= rain_target < clear_target
    assert clear_target <= ORDER_RANGES["weekday"][1]


def test_weather_loader_and_beverage_temperature_are_date_driven(tmp_path):
    weather_path = tmp_path / "weather.csv"
    weather_path.write_text(
        "date,temp_mean,temp_max,temp_min,precipitation\n"
        "2026-06-22,31.0,34.0,28.0,0.0\n"
        "2026-06-23,18.0,21.0,16.0,5.0\n",
        encoding="ascii",
    )
    weather = load_weather_csv(
        weather_path, date(2026, 6, 22), date(2026, 6, 23)
    )
    assert weather[date(2026, 6, 22)].is_rainy is False
    assert weather[date(2026, 6, 23)].is_rainy is True

    hot_day_results = Counter(
        _beverage_details("latte", random.Random(seed), weather[date(2026, 6, 22)])["beverage_temp"]
        for seed in range(200)
    )
    cool_day_results = Counter(
        _beverage_details("latte", random.Random(seed), WeatherRecord(12.0, 0.0))["beverage_temp"]
        for seed in range(200)
    )
    assert hot_day_results["cold"] > hot_day_results["hot"]
    assert cool_day_results["hot"] > cool_day_results["cold"]


def test_daily_unit_target_is_donor_derived_and_exactly_allocated():
    high_day = date(2023, 1, 2)
    low_day = date(2023, 1, 3)
    rows = (
        SourceRecord(
            high_day,
            time(8, 0),
            "high",
            "croissant",
            200,
            Decimal("8.58"),
            Decimal("0.00"),
        ),
        SourceRecord(
            low_day,
            time(8, 0),
            "low",
            "croissant",
            50,
            Decimal("8.58"),
            Decimal("0.00"),
        ),
    )
    index = _index_source(rows)
    high_target = daily_unit_target(100, _donor_profile(high_day, index), index)
    low_target = daily_unit_target(100, _donor_profile(low_day, index), index)
    assert high_target == 190
    assert low_target == 170
    high_baskets = _basket_sizes(100, high_target, random.Random(7))
    low_baskets = _basket_sizes(100, low_target, random.Random(7))
    assert sum(high_baskets) == high_target
    assert sum(low_baskets) == low_target
    assert min(high_baskets + low_baskets) >= 1


def test_donor_precedence_uses_disjoint_profiles():
    exact_day = date(2023, 1, 2)
    rows = (
        SourceRecord(
            exact_day,
            time(8, 0),
            "exact",
            "exact_product",
            1,
            Decimal("10.00"),
            Decimal("0.00"),
        ),
        SourceRecord(
            date(2023, 2, 6),
            time(8, 0),
            "seasonal",
            "seasonal_product",
            2,
            Decimal("11.00"),
            Decimal("0.00"),
        ),
        SourceRecord(
            date(2023, 3, 3),
            time(8, 0),
            "monthly",
            "monthly_product",
            3,
            Decimal("12.00"),
            Decimal("0.00"),
        ),
        SourceRecord(
            date(2023, 4, 4),
            time(8, 0),
            "full",
            "full_product",
            4,
            Decimal("13.00"),
            Decimal("0.00"),
        ),
    )
    index = _index_source(rows)

    exact = _donor_profile(exact_day, index)
    seasonal = _donor_profile(date(2024, 2, 5), index)
    monthly = _donor_profile(date(2024, 3, 4), index)
    full = _donor_profile(date(2024, 5, 6), index)

    assert (exact.precedence, exact.units, exact.source_days) == (
        "exact",
        Counter({"exact_product": 1}),
        1,
    )
    assert (seasonal.precedence, seasonal.units, seasonal.source_days) == (
        "month_weekday",
        Counter({"seasonal_product": 2}),
        1,
    )
    assert (monthly.precedence, monthly.units, monthly.source_days) == (
        "month",
        Counter({"monthly_product": 3}),
        1,
    )
    assert (full.precedence, full.units, full.source_days) == (
        "full",
        Counter(
            {
                "exact_product": 1,
                "seasonal_product": 2,
                "monthly_product": 3,
                "full_product": 4,
            }
        ),
        4,
    )


def test_source_returns_are_netted_before_profile_allocation():
    source_day = date(2025, 6, 16)
    correction_day = date(2025, 6, 23)
    rows = (
        SourceRecord(
            source_day,
            time(8, 0),
            "sale",
            "croissant",
            3,
            Decimal("8.58"),
            Decimal("0.00"),
        ),
        SourceRecord(
            source_day,
            time(8, 5),
            "return",
            "croissant",
            -1,
            Decimal("8.58"),
            Decimal("0.00"),
        ),
        SourceRecord(
            correction_day,
            time(8, 5),
            "later-return",
            "croissant",
            -1,
            Decimal("8.58"),
            Decimal("0.00"),
        ),
        SourceRecord(
            source_day,
            time(8, 10),
            "other",
            "baguette",
            2,
            Decimal("7.02"),
            Decimal("0.00"),
        ),
    )
    index = _index_source(rows)
    assert index.daily_units[source_day] == Counter(
        {"croissant": 2, "baguette": 2}
    )
    assert index.daily_units[correction_day] == Counter()
    expected_aggregate = Counter({"baguette": 2, "croissant": 1})
    assert index.month_class_units[(6, "weekday")] == expected_aggregate
    assert index.month_units[6] == expected_aggregate
    assert index.full_units == expected_aggregate
    assert index.trailing_units[date(2025, 6, 24)] == expected_aggregate


def test_reconstruction_is_deterministic(source_fixture):
    first = build_history(source_fixture, PRODUCT_COSTS, seed=20260718)
    second = build_history(source_fixture, PRODUCT_COSTS, seed=20260718)
    assert first.orders == second.orders
    assert first.items == second.items
    assert first.payments == second.payments


def test_reconstruction_stops_before_operation(source_fixture):
    result = build_history(source_fixture, PRODUCT_COSTS, seed=20260718)
    assert min(row.order_date for row in result.orders) == date(2025, 6, 24)
    assert max(row.order_date for row in result.orders) == date(2026, 6, 23)


def test_one_year_product_mix_avoids_excessive_long_tail(canonical_one_year_history):
    history = canonical_one_year_history
    order_dates = {order.ticket_id: order.order_date for order in history.orders}
    daily_units = defaultdict(Counter)
    annual_units = Counter()
    for item in history.items:
        daily_units[order_dates[item.ticket_id]][item.product_name] += item.quantity
        annual_units[item.product_name] += item.quantity

    product_names = sorted({item.product_name for item in history.items})
    bakery_products = tuple(name for name in product_names if not is_beverage(name))
    beverage_products = tuple(name for name in product_names if is_beverage(name))
    assert len(bakery_products) == 30
    assert len(beverage_products) == 15
    day_count = len(daily_units)
    assert day_count == 365
    bakery_coverage = [
        sum(daily_units[day][name] > 0 for name in bakery_products)
        for day in daily_units
    ]
    beverage_coverage = [
        sum(daily_units[day][name] > 0 for name in beverage_products)
        for day in daily_units
    ]
    assert min(bakery_coverage) >= 28
    assert min(beverage_coverage) >= 14
    assert 28.0 <= sum(bakery_coverage) / day_count <= 29.5
    assert 14.0 <= sum(beverage_coverage) / day_count <= 14.8
    assert max(bakery_coverage) == len(bakery_products)
    assert max(beverage_coverage) == len(beverage_products)
    assert min(bakery_coverage) < len(bakery_products)
    assert min(beverage_coverage) < len(beverage_products)

    assert min(annual_units[name] / day_count for name in bakery_products) >= 2.0
    assert min(annual_units[name] / day_count for name in beverage_products) >= 3.0
    bakery_total = sum(annual_units[name] for name in bakery_products)
    top_seven_share = sum(
        sorted((annual_units[name] for name in bakery_products), reverse=True)[:7]
    ) / bakery_total
    assert 0.55 <= top_seven_share <= 0.65


def test_baskets_and_allocations_are_exact(history, source_fixture):
    item_units = defaultdict(int)
    source_catalog = defaultdict(set)
    for row in source_fixture:
        source_catalog[row.product_name].add(row.unit_price)
    for row in history.items:
        item_units[row.ticket_id] += row.quantity
        assert row.product_name in source_catalog
        expected_prices = {money(value) for value in source_catalog[row.product_name]}
        if row.beverage_size == "large":
            expected_prices = {money(value + Decimal("3.0")) for value in expected_prices}
        assert row.unit_price in expected_prices
    daily_sizes = defaultdict(list)
    for order in history.orders:
        assert item_units[order.ticket_id] == order.item_count
        assert 1 <= order.item_count <= max(BASKET_SIZE_WEIGHTS)
        daily_sizes[order.order_date].append(order.item_count)
    for sizes in daily_sizes.values():
        mean_size = sum(sizes) / len(sizes)
        assert 1.70 <= mean_size <= 1.90


def test_bakery_concentration_uses_exact_integer_limits_on_bounded_history():
    source_dates = (date(2023, 1, 2), date(2023, 1, 6), date(2023, 1, 7))
    quantities = (30, 25, 20, 10, 8, 7, 5, 3)
    rows = []
    ticket_number = 1
    for source_day in source_dates:
        for product_name, quantity in zip(BAKERY_PRODUCTS, quantities, strict=True):
            rows.append(
                SourceRecord(
                    source_day,
                    time(8, 0),
                    str(ticket_number),
                    product_name,
                    quantity,
                    PRICES[product_name],
                    Decimal("0.00"),
                )
            )
            ticket_number += 1

    bounded = build_history(
        rows,
        PRODUCT_COSTS,
        seed=RANDOM_SEED,
        start_date=date(2025, 6, 24),
        end_date=date(2025, 7, 7),
    )
    ticket_days = {order.ticket_id: order.order_date for order in bounded.orders}
    daily_units = defaultdict(Counter)
    for item in bounded.items:
        daily_units[ticket_days[item.ticket_id]][item.product_name] += item.quantity

    assert len(daily_units) == 14
    for units in daily_units.values():
        bakery_units = sum(units.values())
        single_cap = 15 * bakery_units // 100
        top_three = sum(sorted(units.values(), reverse=True)[:3])
        assert max(units.values()) <= single_cap
        assert (35 * bakery_units + 99) // 100 <= top_three
        assert top_three <= 45 * bakery_units // 100


def test_top_three_minimum_depends_on_source_evidence():
    permitted = {
        "p1": 0.30,
        "p2": 0.25,
        "p3": 0.20,
        "p4": 0.10,
        "p5": 0.05,
        "p6": 0.04,
        "p7": 0.03,
        "p8": 0.03,
    }
    permitted_result = _cap_bakery_weights(permitted, permitted)
    permitted_leaders = sorted(
        permitted, key=lambda name: (-permitted[name], name)
    )[:3]
    assert max(permitted_result.values()) <= 0.15
    assert 0.35 <= sum(permitted_result[name] for name in permitted_leaders) <= 0.45

    not_permitted = {f"p{index}": 0.10 for index in range(10)}
    not_permitted_result = _cap_bakery_weights(not_permitted, not_permitted)
    not_permitted_leaders = sorted(not_permitted)[:3]
    assert max(not_permitted_result.values()) <= 0.15
    assert sum(not_permitted_result[name] for name in not_permitted_leaders) < 0.35

    uniform_rows = tuple(
        SourceRecord(
            date(2023, 1, 2),
            time(8, index),
            str(index),
            f"uniform_{index}",
            10,
            Decimal("10.00"),
            Decimal("0.00"),
        )
        for index in range(10)
    )
    uniform_history = build_history(
        uniform_rows,
        {f"uniform_{index}": Decimal("1.0") for index in range(10)},
        seed=RANDOM_SEED,
        start_date=date(2025, 6, 24),
        end_date=date(2025, 6, 24),
    )
    uniform_units = Counter()
    for item in uniform_history.items:
        uniform_units[item.product_name] += item.quantity
    uniform_total = sum(uniform_units.values())
    uniform_top_three = sum(sorted(uniform_units.values(), reverse=True)[:3])
    assert max(uniform_units.values()) <= 15 * uniform_total // 100
    assert uniform_top_three < (35 * uniform_total + 99) // 100


def test_ticket_times_use_distinct_minutes_and_both_peaks():
    times = _ticket_times(200, random.Random(20260718))
    assert len({(value.hour, value.minute) for value in times}) > 1
    assert any(6 <= value.hour <= 10 for value in times)
    assert any(11 <= value.hour <= 13 for value in times)


def test_daily_channel_and_payment_shares_stay_in_range(history):
    daily_orders = defaultdict(list)
    payment_by_ticket = {row.ticket_id: row for row in history.payments}
    for order in history.orders:
        daily_orders[order.order_date].append(order)
        assert STORE_OPEN <= order.order_time < STORE_CLOSE
        assert payment_by_ticket[order.ticket_id].amount == order.total_amount
        assert payment_by_ticket[order.ticket_id].payment_date == order.order_date
    assert len(payment_by_ticket) == len(history.orders)
    for orders in daily_orders.values():
        total = len(orders)
        takeaway_share = sum(row.dine_type == "takeaway" for row in orders) / total
        assert TAKEAWAY_RANGE[0] <= takeaway_share <= TAKEAWAY_RANGE[1]
        payment_counts = Counter(payment_by_ticket[row.ticket_id].payment_method for row in orders)
        for method, limits in PAYMENT_RANGES.items():
            share = payment_counts[method] / total
            assert limits[0] <= share <= limits[1]


def test_beverage_details_use_frontend_capabilities(history):
    beverage_items = [row for row in history.items if is_beverage(row.product_name)]
    assert beverage_items
    for row in beverage_items:
        capability = beverage_capabilities(row.product_name)
        assert row.beverage_size in capability["allowed_sizes"]
        assert row.beverage_temp in capability["allowed_temperatures"]
        assert row.beverage_sweetness in capability["allowed_sugar"]
        assert row.beverage_ice in capability["allowed_ice"]
        if row.beverage_temp == "hot":
            assert row.beverage_ice == "none"


def test_generated_profile_matches_approved_ranges(history):
    summary = summarize_history(history)
    assert 1.70 <= summary.mean_items_per_order <= 1.90
    assert 0.50 <= summary.one_item_share <= 0.54
    assert 0.29 <= summary.two_item_share <= 0.33
    assert 0.75 <= summary.takeaway_share <= 0.85
    assert 0.85 <= summary.qr_share <= 0.92
    assert summary.delivery_orders == 0
    assert summary.min_order_time >= time(6, 0)
    assert summary.max_order_time < time(19, 0)


def test_money_uses_cny_tenth_round_half_up():
    assert money(Decimal("1.04")) == Decimal("1.0")
    assert money(Decimal("1.05")) == Decimal("1.1")
    assert money(Decimal("2.99")) == Decimal("3.0")


def test_order_money_and_relations_reconcile(history):
    items_by_ticket = defaultdict(list)
    for item in history.items:
        items_by_ticket[item.ticket_id].append(item)
    payments_by_ticket = {row.ticket_id: row for row in history.payments}
    assert len(payments_by_ticket) == len(history.orders)
    for order in history.orders:
        items = items_by_ticket[order.ticket_id]
        assert order.item_count == sum(item.quantity for item in items)
        assert order.subtotal == money(
            sum(item.unit_price * item.quantity for item in items)
        )
        assert order.total_amount == money(order.subtotal - order.discount_total)
        assert payments_by_ticket[order.ticket_id].amount == order.total_amount
        assert order.total_profit == money(sum(item.line_profit for item in items))
        for value in (
            order.subtotal,
            order.discount_total,
            order.total_amount,
            order.total_profit,
        ):
            assert value.as_tuple().exponent == -1


def test_history_fails_with_sorted_missing_product_costs(source_fixture):
    costs = dict(PRODUCT_COSTS)
    del costs["latte"]
    del costs["croissant"]
    with pytest.raises(ValueError, match="croissant, latte"):
        build_history(source_fixture, costs, seed=RANDOM_SEED)


def test_large_beverage_surcharge_only_changes_large_items(history):
    beverage_items = [item for item in history.items if is_beverage(item.product_name)]
    assert any(item.beverage_size == "large" for item in beverage_items)
    assert any(item.beverage_size == "regular" for item in beverage_items)
    for item in beverage_items:
        expected = PRICES[item.product_name]
        if item.beverage_size == "large":
            expected += Decimal("3.0")
        assert item.unit_price == money(expected)


def test_sparse_legacy_signals_are_preserved_deterministically():
    source_rows = tuple(
        SourceRecord(
            order_date=date(2023, 1, index + 1),
            order_time=time(8, index),
            ticket_id=f"source-{index}",
            product_name="croissant",
            quantity=1,
            unit_price=Decimal("10.0"),
            discount_rate=Decimal("0.2") if index == 0 else Decimal("0.0"),
            is_day1=int(index == 0),
            is_top3=int(index == 0),
        )
        for index in range(20)
    )
    first = build_history(
        source_rows,
        {"croissant": Decimal("1.0")},
        seed=RANDOM_SEED,
        start_date=date(2025, 6, 24),
        end_date=date(2025, 6, 24),
    )
    second = build_history(
        source_rows,
        {"croissant": Decimal("1.0")},
        seed=RANDOM_SEED,
        start_date=date(2025, 6, 24),
        end_date=date(2025, 6, 24),
    )

    assert first == second
    top3_count = sum(item.is_top3 for item in first.items)
    day1_count = sum(item.is_day1 for item in first.items)
    assert 0 < top3_count < len(first.items)
    assert 0 < day1_count < len(first.items)
    assert all(
        item.discount_rate == (Decimal("0.2") if item.is_day1 else Decimal("0.0"))
        for item in first.items
    )


def test_beverages_never_receive_day1_state():
    source_rows = (
        SourceRecord(
            order_date=date(2023, 1, 1),
            order_time=time(8, 0),
            ticket_id="source-beverage",
            product_name="latte",
            quantity=1,
            unit_price=Decimal("18.0"),
            discount_rate=Decimal("0.2"),
            is_day1=1,
        ),
    )
    generated = build_history(
        source_rows,
        {"latte": Decimal("3.6")},
        seed=RANDOM_SEED,
        start_date=date(2025, 6, 24),
        end_date=date(2025, 6, 24),
    )

    assert generated.items
    assert all(item.is_day1 == 0 for item in generated.items)
    assert all(item.freshness == "Fresh" for item in generated.items)
    assert all(item.discount_rate == Decimal("0.0") for item in generated.items)


@pytest.mark.parametrize("invalid_rate", [Decimal("-0.1"), Decimal("1.1")])
def test_source_discount_rate_must_be_within_unit_interval(invalid_rate):
    source_rows = (
        SourceRecord(
            order_date=date(2023, 1, 1),
            order_time=time(8, 0),
            ticket_id="source-invalid-rate",
            product_name="croissant",
            quantity=1,
            unit_price=Decimal("10.0"),
            discount_rate=invalid_rate,
            is_day1=1,
        ),
    )

    with pytest.raises(ValueError, match="Discount rate must be between 0 and 1"):
        build_history(
            source_rows,
            {"croissant": Decimal("1.0")},
            seed=RANDOM_SEED,
            start_date=date(2025, 6, 24),
            end_date=date(2025, 6, 24),
        )


def test_physical_csv_revenue_matches_manifest_for_rare_discount_rate(tmp_path):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "output.csv"
    manifest_path = tmp_path / "manifest.json"
    with source_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "time",
                "ticket_id",
                "product_name",
                "quantity",
                "unit_price_cny",
                "is_rainy",
                "is_member_day",
                "is_competitor",
                "is_new_product",
                "is_day1",
                "is_top3",
                "discount_pct",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "date": "2023-01-01",
                "time": "08:00",
                "ticket_id": "source-rare-rate",
                "product_name": "croissant",
                "quantity": 1,
                "unit_price_cny": "10.0",
                "is_rainy": 0,
                "is_member_day": 0,
                "is_competitor": 0,
                "is_new_product": 0,
                "is_day1": 1,
                "is_top3": 0,
                "discount_pct": "0.23462511895188568",
            }
        )

    manifest = generate_outputs(
        BuildPaths(
            source_path,
            output_path,
            manifest_path,
            Path("data/reference/product_cost_catalog.json"),
        ),
        seed=RANDOM_SEED,
        start_date=date(2025, 6, 24),
        end_date=date(2025, 6, 24),
    )

    physical_revenue = Decimal("0.0")
    with output_path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    assert {Decimal(row["discount_pct"]) for row in rows} == {Decimal("0.2")}
    for row in rows:
        gross = money(Decimal(row["unit_price_cny"]) * int(row["quantity"]))
        discount = money(gross * Decimal(row["discount_pct"]))
        physical_revenue += money(gross - discount)
    assert money(physical_revenue) == Decimal(manifest["revenue_cny"])


def test_generate_outputs_writes_transaction_csv_and_fingerprinted_manifest(
    tmp_path, source_fixture
):
    source_path = tmp_path / "source.csv"
    raw_path = tmp_path / "raw.csv"
    manifest_path = tmp_path / "manifest.json"
    cost_path = tmp_path / "costs.json"
    fieldnames = [
        "date",
        "time",
        "ticket_id",
        "product_name",
        "quantity",
        "unit_price_cny",
        "is_rainy",
        "is_member_day",
        "is_competitor",
        "is_new_product",
        "is_day1",
        "is_top3",
        "discount_pct",
    ]
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in source_fixture:
            writer.writerow(
                {
                    "date": row.order_date.isoformat(),
                    "time": row.order_time.strftime("%H:%M"),
                    "ticket_id": row.ticket_id,
                    "product_name": row.product_name,
                    "quantity": row.quantity,
                    "unit_price_cny": row.unit_price,
                    "is_rainy": 0,
                    "is_member_day": 0,
                    "is_competitor": 0,
                    "is_new_product": 0,
                    "is_day1": 0,
                    "is_top3": int(row.product_name in BAKERY_PRODUCTS[:3]),
                    "discount_pct": row.discount_rate,
                }
            )
    cost_path.write_text(
        Path("data/reference/product_cost_catalog.json").read_text(encoding="ascii"),
        encoding="ascii",
    )

    result = generate_outputs(
        BuildPaths(source_path, raw_path, manifest_path, cost_path),
        seed=RANDOM_SEED,
        start_date=date(2025, 6, 24),
        end_date=date(2025, 6, 25),
    )

    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert handle.seekable()
    assert rows
    assert list(rows[0]) == [
        "date",
        "time",
        "ticket_id",
        "product_name",
        "quantity",
        "unit_price_cny",
        "category",
        "is_rainy",
        "is_member_day",
        "is_competitor",
        "is_new_product",
        "is_day1",
        "is_top3",
        "discount_pct",
        "beverage_size",
        "beverage_temp",
        "beverage_sweetness",
        "beverage_ice",
    ]
    assert {row["category"] for row in rows} == {"bakery", "beverage"}
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    assert result == manifest
    assert manifest["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert manifest["output_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert manifest["seed"] == RANDOM_SEED
    assert manifest["date_range"] == {"start": "2025-06-24", "end": "2025-06-25"}
    assert manifest["row_counts"]["output"] == len(rows)
    assert manifest["order_count"] > 0
    assert manifest["unit_count"] >= manifest["order_count"]
    assert set(manifest["basket_shares"]) == {"one_item", "two_item", "three_plus"}
    assert set(manifest["channel_shares"]) == {"takeaway", "dine_in", "delivery"}
    assert set(manifest["payment_shares"]) == set(PAYMENT_RANGES)
    assert manifest["product_concentration"]["top_products"]
    assert manifest["approved_ranges"] == {
        "daily_orders": {
            "weekday": [110, 145],
            "friday": [120, 155],
            "weekend": [135, 175],
        },
        "basket_mean": [1.7, 1.9],
        "takeaway_share": [0.75, 0.85],
        "payment_shares": {
            "qr": [0.85, 0.92],
            "card": [0.05, 0.1],
            "cash": [0.02, 0.05],
        },
        "bakery_single_product_share_max": 0.15,
        "bakery_top3_share": [0.35, 0.45],
        "bakery_top7_share": [0.55, 0.65],
        "minimum_bakery_product_daily_average": 2.0,
        "minimum_beverage_product_daily_average": 3.0,
    }
    assert manifest["created_at"].endswith("Z")
    assert "password" not in manifest_path.read_text(encoding="ascii").lower()


def test_generate_outputs_rejects_physical_writer_count_mismatch(
    tmp_path, source_fixture, monkeypatch
):
    source_path = tmp_path / "source.csv"
    raw_path = tmp_path / "raw.csv"
    manifest_path = tmp_path / "manifest.json"
    source_path.write_text("source fixture\n", encoding="ascii")
    original_writer = rebuild.write_raw_csv

    def wrong_count_writer(history, source_rows, output_path, weather_by_date=None):
        return original_writer(
            history, source_rows, output_path, weather_by_date
        ) - 1

    monkeypatch.setattr(rebuild, "load_source_csv", lambda path: source_fixture)
    monkeypatch.setattr(rebuild, "load_product_costs", lambda path: PRODUCT_COSTS)
    monkeypatch.setattr(rebuild, "write_raw_csv", wrong_count_writer)

    with pytest.raises(AssertionError, match="CSV writer row count"):
        generate_outputs(
            BuildPaths(source_path, raw_path, manifest_path, tmp_path / "costs.json"),
            seed=RANDOM_SEED,
            start_date=date(2025, 6, 24),
            end_date=date(2025, 6, 24),
        )


class RecordingCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def execute(self, sql, params=None):
        self.connection.statements.append(("execute", " ".join(sql.split()), params))
        if "SELECT id, ticket_id" in sql:
            self.rows = [(1, self.connection.ticket_id)]
        elif "mismatch_count" in sql:
            self.rows = [(self.connection.mismatch_count,)]
        else:
            self.rows = []

    def executemany(self, sql, params):
        params = list(params)
        self.connection.statements.append(("executemany", " ".join(sql.split()), params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0]

    def close(self):
        self.connection.cursor_closed = True


class RecordingConnection:
    def __init__(self, ticket_id, mismatch_count=0):
        self.ticket_id = ticket_id
        self.mismatch_count = mismatch_count
        self.statements = []
        self.started = False
        self.committed = False
        self.rolled_back = False
        self.cursor_closed = False

    def start_transaction(self):
        self.started = True

    def cursor(self):
        return RecordingCursor(self)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def _single_order_history(order_date=HISTORY_START):
    ticket_id = f"GZ-{order_date:%Y%m%d}-0001"
    item = ItemRecord(
        ticket_id=ticket_id,
        product_name="croissant",
        quantity=1,
        unit_price=Decimal("10.0"),
        discount_rate=Decimal("0.0"),
        line_total=Decimal("10.0"),
        line_profit=Decimal("9.0"),
        freshness="Fresh",
    )
    order = OrderRecord(
        ticket_id=ticket_id,
        order_date=order_date,
        order_time=time(8, 0),
        subtotal=Decimal("10.0"),
        discount_total=Decimal("0.0"),
        total_amount=Decimal("10.0"),
        total_profit=Decimal("9.0"),
        item_count=1,
        state="completed",
        dine_type="takeaway",
    )
    payment = PaymentRecord(ticket_id, Decimal("10.0"), "qr", order_date)
    return History((order,), (item,), (payment,))


def test_apply_history_uses_one_transaction_and_historical_identifier_deletes():
    history = _single_order_history()
    connection = RecordingConnection(history.orders[0].ticket_id)
    apply_history(history, connection, APPLY_COSTS, chunk_size=10)

    sql = "\n".join(statement[1] for statement in connection.statements)
    assert connection.started
    assert connection.committed
    assert not connection.rolled_back
    assert "DELETE p FROM payments p JOIN orders o ON p.order_id = o.id" in sql
    assert "DELETE oi FROM order_items oi JOIN orders o ON oi.order_id = o.id" in sql
    assert "DELETE FROM orders WHERE order_date <= %s" in sql
    assert sql.index("DELETE p FROM payments") < sql.index("DELETE FROM orders")
    assert sql.index("DELETE oi FROM order_items") < sql.index("DELETE FROM orders")
    assert sql.count("mismatch_count") >= 5
    assert "ROUND(oi.unit_price * oi.quantity, 1)" in sql
    assert (
        "ROUND(oi.line_total - cost_basis.unit_cost_cny * oi.quantity, 1)"
        in sql
    )


def test_apply_history_rolls_back_on_reconciliation_failure():
    history = _single_order_history()
    connection = RecordingConnection(history.orders[0].ticket_id, mismatch_count=1)
    with pytest.raises(ValueError, match="SQL reconciliation failed"):
        apply_history(history, connection, APPLY_COSTS)
    assert connection.started
    assert connection.rolled_back
    assert not connection.committed


def test_apply_history_rejects_operation_dates_before_starting_transaction():
    history = _single_order_history(OPERATION_START)
    connection = RecordingConnection(history.orders[0].ticket_id)
    with pytest.raises(ValueError, match="2026-06-24"):
        apply_history(history, connection, APPLY_COSTS)
    assert not connection.started


def test_apply_history_rejects_pre_reopening_dates_before_starting_transaction():
    history = _single_order_history(date(2025, 6, 23))
    connection = RecordingConnection(history.orders[0].ticket_id)
    with pytest.raises(ValueError, match="2025-06-23"):
        apply_history(history, connection, APPLY_COSTS)
    assert not connection.started


def test_apply_history_rejects_empty_history_before_starting_transaction():
    connection = RecordingConnection("unused")
    with pytest.raises(ValueError, match="must not be empty"):
        apply_history(History((), (), ()), connection, APPLY_COSTS)
    assert not connection.started


def test_apply_history_rejects_line_total_mismatch_before_starting_transaction():
    history = _single_order_history()
    bad_item = ItemRecord(
        **{
            **history.items[0].__dict__,
            "line_total": Decimal("9.0"),
        }
    )
    invalid = History(history.orders, (bad_item,), history.payments)
    connection = RecordingConnection(history.orders[0].ticket_id)
    with pytest.raises(ValueError, match="Invalid item line total"):
        apply_history(invalid, connection, APPLY_COSTS)
    assert not connection.started


def test_apply_history_rejects_aggregate_consistent_item_formula_mismatch():
    history = _single_order_history()
    item = ItemRecord(
        **{
            **history.items[0].__dict__,
            "line_total": Decimal("9.0"),
            "line_profit": Decimal("8.0"),
        }
    )
    order = OrderRecord(
        **{
            **history.orders[0].__dict__,
            "discount_total": Decimal("1.0"),
            "total_amount": Decimal("9.0"),
            "total_profit": Decimal("8.0"),
        }
    )
    payment = PaymentRecord(
        history.payments[0].ticket_id,
        Decimal("9.0"),
        history.payments[0].payment_method,
        history.payments[0].payment_date,
    )
    connection = RecordingConnection(order.ticket_id)

    with pytest.raises(ValueError, match="Invalid item line total"):
        apply_history(
            History((order,), (item,), (payment,)),
            connection,
            {"croissant": Decimal("1.0")},
        )
    assert not connection.started


def test_apply_history_rejects_aggregate_consistent_profit_formula_mismatch():
    history = _single_order_history()
    item = ItemRecord(
        **{**history.items[0].__dict__, "line_profit": Decimal("8.0")}
    )
    order = OrderRecord(
        **{**history.orders[0].__dict__, "total_profit": Decimal("8.0")}
    )
    connection = RecordingConnection(order.ticket_id)

    with pytest.raises(ValueError, match="Invalid item line profit"):
        apply_history(
            History((order,), (item,), history.payments),
            connection,
            {"croissant": Decimal("1.0")},
        )
    assert not connection.started


def test_apply_history_rejects_beverage_day1_before_transaction():
    ticket_id = "GZ-20250624-0001"
    item = ItemRecord(
        ticket_id=ticket_id,
        product_name="latte",
        quantity=1,
        unit_price=Decimal("18.0"),
        discount_rate=Decimal("0.0"),
        line_total=Decimal("18.0"),
        line_profit=Decimal("14.4"),
        freshness="Day-1",
        is_day1=1,
    )
    order = OrderRecord(
        ticket_id=ticket_id,
        order_date=HISTORY_START,
        order_time=time(8, 0),
        subtotal=Decimal("18.0"),
        discount_total=Decimal("0.0"),
        total_amount=Decimal("18.0"),
        total_profit=Decimal("14.4"),
        item_count=1,
        state="completed",
        dine_type="takeaway",
    )
    payment = PaymentRecord(ticket_id, Decimal("18.0"), "qr", order.order_date)
    connection = RecordingConnection(ticket_id)

    with pytest.raises(ValueError, match="Beverage items cannot be Day-1"):
        apply_history(
            History((order,), (item,), (payment,)),
            connection,
            {"latte": Decimal("3.6")},
        )
    assert not connection.started


class CostCatalogCursor:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def execute(self, sql):
        assert "FROM products p" in sql
        assert "product_recipes" in sql
        assert "raw_materials" in sql

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class CostCatalogConnection:
    def __init__(self, rows):
        self.catalog_cursor = CostCatalogCursor(rows)
        self.readonly = None
        self.rolled_back = False

    def start_transaction(self, readonly=False):
        self.readonly = readonly

    def cursor(self):
        return self.catalog_cursor

    def rollback(self):
        self.rolled_back = True


def test_product_cost_catalog_is_read_only_sorted_and_recipe_checked(tmp_path):
    recipe_path = tmp_path / "recipes.json"
    output_path = tmp_path / "costs.json"
    recipe_path.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "product_name": "bakery_a",
                        "ingredients": [
                            {
                                "material_name": "Flour",
                                "quantity_per_unit": "0.200000",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="ascii",
    )
    connection = CostCatalogConnection(
        [
            ("beverage_b", "beverage", "Tea", Decimal("0.100000"), Decimal("8.00")),
            ("bakery_a", "bakery", "Flour", Decimal("0.200000"), Decimal("10.00")),
        ]
    )

    payload = create_product_cost_catalog(
        connection,
        recipe_path,
        output_path,
        expected_product_count=2,
    )

    assert connection.readonly is True
    assert connection.rolled_back
    assert connection.catalog_cursor.closed
    assert [row["product_name"] for row in payload["products"]] == [
        "bakery_a",
        "beverage_b",
    ]
    assert [row["unit_cost_cny"] for row in payload["products"]] == ["2.0", "0.8"]
    assert json.loads(output_path.read_text(encoding="ascii")) == payload


def test_load_product_costs_fails_closed_on_catalog_contract(tmp_path):
    canonical = json.loads(
        Path("data/reference/product_cost_catalog.json").read_text(encoding="ascii")
    )
    catalog_path = tmp_path / "costs.json"
    catalog_path.write_text(json.dumps(canonical), encoding="ascii")
    assert len(load_product_costs(catalog_path)) == 45

    invalid_payloads = []
    for field, value in (
        ("schema_version", 2),
        ("currency", "USD"),
        ("money_step", "0.01"),
        ("product_count", 44),
        ("bakery_product_count", 29),
        ("beverage_product_count", 14),
    ):
        payload = copy.deepcopy(canonical)
        payload[field] = value
        invalid_payloads.append(payload)

    reversed_products = copy.deepcopy(canonical)
    reversed_products["products"].reverse()
    invalid_payloads.append(reversed_products)

    duplicate = copy.deepcopy(canonical)
    duplicate["products"][1]["product_name"] = duplicate["products"][0][
        "product_name"
    ]
    invalid_payloads.append(duplicate)

    missing_field = copy.deepcopy(canonical)
    del missing_field["products"][0]["category"]
    invalid_payloads.append(missing_field)

    wrong_category = copy.deepcopy(canonical)
    wrong_category["products"][0]["category"] = "bakery"
    invalid_payloads.append(wrong_category)

    wrong_precision = copy.deepcopy(canonical)
    wrong_precision["products"][0]["unit_cost_cny"] = "1.60"
    invalid_payloads.append(wrong_precision)

    for index, payload in enumerate(invalid_payloads):
        catalog_path.write_text(json.dumps(payload), encoding="ascii")
        with pytest.raises(ValueError, match="Product cost catalog"):
            load_product_costs(catalog_path)
