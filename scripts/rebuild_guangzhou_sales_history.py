from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.module4_frontend.beverage_options import (
    beverage_capabilities,
    beverage_unit_price,
    is_beverage,
)


HISTORY_START = date(2025, 6, 24)
HISTORY_END = date(2026, 6, 23)
OPERATION_START = date(2026, 6, 24)
STORE_OPEN = time(6, 0)
STORE_CLOSE = time(19, 0)
RANDOM_SEED = 20260718

ORDER_RANGES = {
    "weekday": (110, 145),
    "friday": (120, 155),
    "weekend": (135, 175),
}
BASKET_SIZE_WEIGHTS = {1: 52, 2: 31, 3: 11, 4: 3, 5: 2, 6: 1}
TAKEAWAY_RANGE = (0.75, 0.85)
PAYMENT_RANGES = {
    "qr": (0.85, 0.92),
    "card": (0.05, 0.10),
    "cash": (0.02, 0.05),
}


@dataclass(frozen=True)
class BuildPaths:
    source_csv: Path
    raw_output_csv: Path
    manifest_path: Path
    cost_catalog: Path
    weather_csv: Path = Path("data/guangzhou_weather.csv")


@dataclass(frozen=True)
class SourceRecord:
    order_date: date
    order_time: time
    ticket_id: str
    product_name: str
    quantity: int
    unit_price: Decimal
    discount_rate: Decimal
    is_rainy: int = 0
    is_member_day: int = 0
    is_competitor: int = 0
    is_new_product: int = 0
    is_day1: int = 0
    is_top3: int = 0


@dataclass(frozen=True)
class WeatherRecord:
    temp_mean: float
    precipitation: float

    @property
    def is_rainy(self) -> bool:
        return self.precipitation >= 1.0


@dataclass(frozen=True)
class OrderRecord:
    ticket_id: str
    order_date: date
    order_time: time
    subtotal: Decimal
    discount_total: Decimal
    total_amount: Decimal
    total_profit: Decimal
    item_count: int
    state: str
    dine_type: str


@dataclass(frozen=True)
class ItemRecord:
    ticket_id: str
    product_name: str
    quantity: int
    unit_price: Decimal
    discount_rate: Decimal
    line_total: Decimal
    line_profit: Decimal
    freshness: str
    beverage_size: str | None = None
    beverage_temp: str | None = None
    beverage_sweetness: str | None = None
    beverage_ice: str | None = None
    is_day1: int = 0
    is_top3: int = 0


@dataclass(frozen=True)
class PaymentRecord:
    ticket_id: str
    amount: Decimal
    payment_method: str
    payment_date: date


@dataclass(frozen=True)
class History:
    orders: tuple[OrderRecord, ...]
    items: tuple[ItemRecord, ...]
    payments: tuple[PaymentRecord, ...]


@dataclass(frozen=True)
class HistorySummary:
    mean_items_per_order: float
    one_item_share: float
    two_item_share: float
    takeaway_share: float
    qr_share: float
    delivery_orders: int
    min_order_time: time
    max_order_time: time


@dataclass(frozen=True)
class _SourceIndex:
    daily_units: dict[date, Counter[str]]
    trailing_units: dict[date, Counter[str]]
    month_class_units: dict[tuple[int, str], Counter[str]]
    month_class_days: dict[tuple[int, str], int]
    month_units: dict[int, Counter[str]]
    month_days: dict[int, int]
    full_units: Counter[str]
    source_days: int
    prices: dict[str, Decimal]
    day1_evidence: _BinaryFeatureEvidence
    top3_evidence: _BinaryFeatureEvidence
    discount_month_product: dict[tuple[int, str], Counter[Decimal]]
    discount_product: dict[str, Counter[Decimal]]
    positive_discounts: Counter[Decimal]
    products: tuple[str, ...]
    source_end: date


@dataclass(frozen=True)
class _BinaryFeatureEvidence:
    month_product: dict[tuple[int, str], tuple[int, int]]
    product: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class _DonorProfile:
    units: Counter[str]
    source_days: int
    precedence: str


MONEY_STEP = Decimal("0.1")
DISCOUNT_RATE_STEP = Decimal("0.1")
BAKERY_SHARE_CAP = 0.15
TOP_THREE_MIN = 0.35
TOP_THREE_MAX = 0.45
BAKERY_EVIDENCE_WEIGHT = 0.70
BEVERAGE_EVIDENCE_WEIGHT = 0.55
BAKERY_DAILY_ACTIVE_RANGE = (28, 30)
BEVERAGE_DAILY_ACTIVE_RANGE = (14, 15)
BAKERY_SHARE_CAP_PERCENT = 15
TOP_THREE_MIN_PERCENT = 35
TOP_THREE_MAX_PERCENT = 45
BASKET_MEAN_MIN = 1.70
BASKET_MEAN_MAX = 1.90
BASKET_WEIGHTED_MEAN = sum(
    size * weight for size, weight in BASKET_SIZE_WEIGHTS.items()
) / sum(BASKET_SIZE_WEIGHTS.values())


def daily_order_target(
    day: date,
    rng: random.Random,
    is_holiday: bool,
    weather: WeatherRecord | None = None,
) -> int:
    if is_holiday or day.weekday() >= 5:
        low, high = ORDER_RANGES["weekend"]
    elif day.weekday() == 4:
        low, high = ORDER_RANGES["friday"]
    else:
        low, high = ORDER_RANGES["weekday"]
    midpoint = (low + high) / 2
    weather_factor = 1.0
    if weather is not None:
        if weather.precipitation >= 20.0:
            weather_factor *= 0.90
        elif weather.precipitation >= 5.0:
            weather_factor *= 0.95
        elif weather.is_rainy:
            weather_factor *= 0.98
        if weather.temp_mean >= 30.0 or weather.temp_mean <= 12.0:
            weather_factor *= 0.97
    candidate = round(midpoint * rng.uniform(0.88, 1.12) * weather_factor)
    return min(max(candidate, low), high)


def _weekday_class(day: date) -> str:
    if day.weekday() >= 5:
        return "weekend"
    if day.weekday() == 4:
        return "friday"
    return "weekday"


def _date_rng(seed: int, day: date) -> random.Random:
    return random.Random((seed << 32) ^ day.toordinal())


def money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_STEP, rounding=ROUND_HALF_UP)


def historical_discount_rate(value: Decimal) -> Decimal:
    if not value.is_finite() or not Decimal("0.0") <= value <= Decimal("1.0"):
        raise ValueError("Discount rate must be between 0 and 1")
    return value.quantize(DISCOUNT_RATE_STEP, rounding=ROUND_HALF_UP)


def _floor_percent(total: int, percent: int) -> int:
    return total * percent // 100


def _ceil_percent(total: int, percent: int) -> int:
    return (total * percent + 99) // 100


def _primary_value(counts: Counter, label: str):
    if not counts:
        raise ValueError(f"Source catalog has no {label}")
    return min(counts, key=lambda value: (-counts[value], value))


def _positive_units(units: Counter[str]) -> Counter[str]:
    return Counter({product: quantity for product, quantity in units.items() if quantity > 0})


def _binary_feature_evidence(
    rows: Iterable[SourceRecord], field: str
) -> _BinaryFeatureEvidence:
    month_product = defaultdict(Counter)
    product = defaultdict(Counter)
    for row in rows:
        if row.quantity <= 0:
            continue
        value = int(bool(getattr(row, field)))
        month_product[(row.order_date.month, row.product_name)][value] += 1
        product[row.product_name][value] += 1

    def counts(values: Counter[int]) -> tuple[int, int]:
        return values[1], values[0] + values[1]

    return _BinaryFeatureEvidence(
        month_product={key: counts(values) for key, values in month_product.items()},
        product={key: counts(values) for key, values in product.items()},
    )


def _feature_value(
    evidence: _BinaryFeatureEvidence,
    day: date,
    ticket_id: str,
    product_name: str,
    seed: int,
    feature: str,
) -> int:
    positive, total = evidence.month_product.get(
        (day.month, product_name), evidence.product.get(product_name, (0, 0))
    )
    if positive <= 0 or total <= 0:
        return 0
    if positive >= total:
        return 1
    token = f"{seed}|{day.isoformat()}|{ticket_id}|{product_name}|{feature}"
    bucket = int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:8], "big")
    return int(bucket * total < positive * (1 << 64))


def _build_trailing_profiles(
    daily_units: dict[date, Counter[str]], source_end: date
) -> dict[date, Counter[str]]:
    first_cutoff = HISTORY_START - timedelta(days=1)
    rolling = Counter()
    for offset in range(89, -1, -1):
        rolling.update(daily_units.get(first_cutoff - timedelta(days=offset), {}))

    profiles = {}
    previous_cutoff = first_cutoff
    day = HISTORY_START
    while day <= HISTORY_END:
        cutoff = min(day - timedelta(days=1), source_end)
        while previous_cutoff < cutoff:
            previous_cutoff += timedelta(days=1)
            rolling.update(daily_units.get(previous_cutoff, {}))
            expired = previous_cutoff - timedelta(days=90)
            rolling.subtract(daily_units.get(expired, {}))
            for product in tuple(rolling):
                if rolling[product] == 0:
                    del rolling[product]
        profiles[day] = _positive_units(rolling)
        day += timedelta(days=1)
    return profiles


def _index_source(source_rows: Iterable[SourceRecord]) -> _SourceIndex:
    rows = tuple(source_rows)
    if not rows:
        raise ValueError("Source history must not be empty")

    daily_units = defaultdict(Counter)
    month_class_units = defaultdict(Counter)
    month_units = defaultdict(Counter)
    full_units = Counter()
    price_counts = defaultdict(Counter)
    discount_month_product = defaultdict(Counter)
    discount_product = defaultdict(Counter)
    positive_discounts = Counter()

    for row in rows:
        if row.quantity == 0:
            raise ValueError("Source quantities must be non-zero")
        if not row.product_name:
            raise ValueError("Source product names must not be empty")
        normalized_discount_rate = historical_discount_rate(row.discount_rate)
        daily_units[row.order_date][row.product_name] += row.quantity
        month_class_units[
            (row.order_date.month, _weekday_class(row.order_date))
        ][row.product_name] += row.quantity
        month_units[row.order_date.month][row.product_name] += row.quantity
        full_units[row.product_name] += row.quantity
        if row.quantity > 0:
            price_counts[row.product_name][row.unit_price] += row.quantity
            if normalized_discount_rate > 0:
                discount_month_product[
                    (row.order_date.month, row.product_name)
                ][normalized_discount_rate] += 1
                discount_product[row.product_name][normalized_discount_rate] += 1
                positive_discounts[normalized_discount_rate] += 1

    indexed_daily_units = {
        source_day: _positive_units(units)
        for source_day, units in daily_units.items()
    }
    indexed_month_class_units = {
        key: _positive_units(units) for key, units in month_class_units.items()
    }
    indexed_month_units = {
        month: _positive_units(units) for month, units in month_units.items()
    }
    full_units = _positive_units(full_units)
    products = tuple(sorted(full_units))
    if not products:
        raise ValueError("Source history has no positive product units")
    source_end = max(row.order_date for row in rows)
    source_dates = tuple(sorted(daily_units))
    month_class_days = Counter(
        (source_day.month, _weekday_class(source_day)) for source_day in source_dates
    )
    month_days = Counter(source_day.month for source_day in source_dates)
    return _SourceIndex(
        daily_units=indexed_daily_units,
        trailing_units=_build_trailing_profiles(dict(daily_units), source_end),
        month_class_units=indexed_month_class_units,
        month_class_days=dict(month_class_days),
        month_units=indexed_month_units,
        month_days=dict(month_days),
        full_units=full_units,
        source_days=len(source_dates),
        prices={
            product: _primary_value(price_counts[product], "price")
            for product in products
        },
        day1_evidence=_binary_feature_evidence(rows, "is_day1"),
        top3_evidence=_binary_feature_evidence(rows, "is_top3"),
        discount_month_product={
            key: values.copy() for key, values in discount_month_product.items()
        },
        discount_product={
            key: values.copy() for key, values in discount_product.items()
        },
        positive_discounts=positive_discounts,
        products=products,
        source_end=source_end,
    )


def _donor_profile(day: date, source: _SourceIndex) -> _DonorProfile:
    exact = source.daily_units.get(day)
    if exact:
        return _DonorProfile(exact.copy(), 1, "exact")
    month_class = (day.month, _weekday_class(day))
    seasonal = source.month_class_units.get(month_class)
    if seasonal:
        return _DonorProfile(
            seasonal.copy(), source.month_class_days[month_class], "month_weekday"
        )
    monthly = source.month_units.get(day.month)
    if monthly:
        return _DonorProfile(monthly.copy(), source.month_days[day.month], "month")
    return _DonorProfile(source.full_units.copy(), source.source_days, "full")


def daily_unit_target(
    order_count: int, donor: _DonorProfile, source: _SourceIndex
) -> int:
    if order_count < 1:
        raise ValueError("Daily order count must be positive")
    donor_daily_units = sum(donor.units.values()) / donor.source_days
    full_daily_units = sum(source.full_units.values()) / source.source_days
    donor_factor = donor_daily_units / full_daily_units
    target_mean = min(
        max(BASKET_WEIGHTED_MEAN * donor_factor, BASKET_MEAN_MIN),
        BASKET_MEAN_MAX,
    )
    minimum_units = math.ceil(BASKET_MEAN_MIN * order_count)
    maximum_units = math.floor(BASKET_MEAN_MAX * order_count)
    return min(max(round(order_count * target_mean), minimum_units), maximum_units)


def _trailing_units(day: date, source: _SourceIndex) -> Counter[str]:
    return source.trailing_units[day]


def _normalized(units: Counter[str], products: Iterable[str]) -> dict[str, float]:
    total = sum(units.values())
    if total <= 0:
        return {product: 0.0 for product in products}
    return {product: units[product] / total for product in products}


def _tier_shares(units: Counter[str], products: tuple[str, ...]) -> dict[str, float]:
    shares = _normalized(units, products)
    ranked = sorted(products, key=lambda product: (-shares[product], product))
    tiers = (ranked[:3], ranked[3:10], ranked[10:])
    result = {}
    for tier in tiers:
        if not tier:
            continue
        tier_share = sum(shares[product] for product in tier) / len(tier)
        result.update({product: tier_share for product in tier})
    return result


def _largest_remainder(
    weights: dict[str, float], total: int, caps: dict[str, int] | None = None
) -> dict[str, int]:
    names = tuple(sorted(weights))
    if total == 0:
        return {name: 0 for name in names}
    weight_total = sum(max(weights[name], 0.0) for name in names)
    if weight_total <= 0:
        normalized = {name: 1 / len(names) for name in names}
    else:
        normalized = {name: max(weights[name], 0.0) / weight_total for name in names}
    raw = {name: normalized[name] * total for name in names}
    allocation = {
        name: min(math.floor(raw[name]), caps[name] if caps else total)
        for name in names
    }
    remaining = total - sum(allocation.values())
    order = sorted(names, key=lambda name: (-(raw[name] - math.floor(raw[name])), name))
    while remaining:
        progressed = False
        for name in order:
            if caps is not None and allocation[name] >= caps[name]:
                continue
            allocation[name] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("Product caps cannot accommodate the basket units")
    return allocation


def _smooth_product_weights(
    weights: Mapping[str, float], evidence_weight: float
) -> dict[str, float]:
    if not weights:
        return {}
    if not 0.0 <= evidence_weight <= 1.0:
        raise ValueError("Evidence weight must be between 0 and 1")
    names = tuple(sorted(weights))
    positive_total = sum(max(float(weights[name]), 0.0) for name in names)
    if positive_total <= 0:
        return {name: 1.0 / len(names) for name in names}
    uniform_share = 1.0 / len(names)
    return {
        name: evidence_weight * max(float(weights[name]), 0.0) / positive_total
        + (1.0 - evidence_weight) * uniform_share
        for name in names
    }


def _select_active_products(
    products: tuple[str, ...],
    active_range: tuple[int, int],
    rng: random.Random,
) -> tuple[str, ...]:
    if not products:
        return ()
    lower = min(len(products), active_range[0])
    upper = min(len(products), active_range[1])
    active_count = rng.randint(lower, upper)
    if active_count == len(products):
        return products
    return tuple(sorted(rng.sample(products, active_count)))


def _cap_bakery_weights(
    weights: dict[str, float], evidence: dict[str, float]
) -> dict[str, float]:
    if len(weights) < math.ceil(1 / BAKERY_SHARE_CAP):
        return weights
    remaining_names = sorted(weights)
    result = {name: 0.0 for name in sorted(weights)}
    remaining_share = 1.0
    while remaining_names:
        weight_total = sum(weights[name] for name in remaining_names)
        proposed = {
            name: remaining_share * weights[name] / weight_total
            if weight_total
            else remaining_share / len(remaining_names)
            for name in remaining_names
        }
        capped = sorted(
            name for name, share in proposed.items() if share > BAKERY_SHARE_CAP
        )
        if not capped:
            result.update(proposed)
            break
        for name in capped:
            result[name] = BAKERY_SHARE_CAP
            remaining_names.remove(name)
            remaining_share -= BAKERY_SHARE_CAP

    leaders = sorted(evidence, key=lambda name: (-evidence[name], name))[:3]
    evidence_leader_share = sum(evidence[name] for name in leaders)
    current_leader_share = sum(result[name] for name in leaders)
    if evidence_leader_share >= TOP_THREE_MIN and current_leader_share < TOP_THREE_MIN:
        needed = TOP_THREE_MIN - current_leader_share
        donors = sorted(name for name in result if name not in leaders)
        donor_total = sum(result[name] for name in donors)
        capacity = sum(BAKERY_SHARE_CAP - result[name] for name in leaders)
        transfer = min(needed, donor_total, capacity)
        if transfer:
            for name in donors:
                result[name] -= transfer * result[name] / donor_total
            leader_capacity = sum(BAKERY_SHARE_CAP - result[name] for name in leaders)
            for name in leaders:
                result[name] += transfer * (
                    BAKERY_SHARE_CAP - result[name]
                ) / leader_capacity
    return result


def _constrain_bakery_allocation(
    allocation: dict[str, int],
    bakery_total: int,
    cap: int,
    evidence: dict[str, float],
) -> dict[str, int]:
    leaders = sorted(evidence, key=lambda name: (-evidence[name], name))[:3]
    evidence_leader_share = sum(evidence[name] for name in leaders)
    minimum_leader_units = _ceil_percent(bakery_total, TOP_THREE_MIN_PERCENT)
    if (
        evidence_leader_share < TOP_THREE_MIN
        or cap * len(leaders) < minimum_leader_units
    ):
        return allocation

    needed = minimum_leader_units - sum(allocation[name] for name in leaders)
    recipients = sorted(leaders, key=lambda name: (-evidence[name], name))
    donors = sorted(
        (name for name in allocation if name not in leaders),
        key=lambda name: (evidence[name], name),
    )
    for recipient in recipients:
        if needed <= 0:
            break
        capacity = cap - allocation[recipient]
        if capacity <= 0:
            continue
        for donor in donors:
            if needed <= 0 or capacity <= 0:
                break
            transfer = min(needed, capacity, allocation[donor])
            allocation[recipient] += transfer
            allocation[donor] -= transfer
            needed -= transfer
            capacity -= transfer
    if needed > 0:
        raise ValueError("Bakery allocation cannot satisfy top-three concentration")
    return allocation


def _product_allocation(
    basket_units: int,
    donor_units: Counter[str],
    trailing_units: Counter[str],
    source: _SourceIndex,
    weather: WeatherRecord | None = None,
    availability_rng: random.Random | None = None,
) -> dict[str, int]:
    donor_shares = _normalized(donor_units, source.products)
    tier_basis = trailing_units or donor_units
    tier_shares = _tier_shares(tier_basis, source.products)
    blended = {
        product: 0.8 * donor_shares[product] + 0.2 * tier_shares[product]
        for product in source.products
    }
    bakery_catalog = tuple(
        product for product in source.products if not is_beverage(product)
    )
    beverage_catalog = tuple(
        product for product in source.products if is_beverage(product)
    )
    selection_rng = availability_rng or random.Random(0)
    bakery = _select_active_products(
        bakery_catalog, BAKERY_DAILY_ACTIVE_RANGE, selection_rng
    )
    beverages = _select_active_products(
        beverage_catalog, BEVERAGE_DAILY_ACTIVE_RANGE, selection_rng
    )
    categories = {}
    if bakery:
        categories["bakery"] = sum(blended[product] for product in bakery)
    if beverages:
        categories["beverage"] = sum(blended[product] for product in beverages)
    category_units = _largest_remainder(categories, basket_units)

    allocation = {product: 0 for product in source.products}
    if bakery:
        bakery_weights = _smooth_product_weights(
            {product: blended[product] for product in bakery},
            BAKERY_EVIDENCE_WEIGHT,
        )
        evidence = _normalized(
            Counter({product: donor_units[product] for product in bakery}), bakery
        )
        bakery_weights = _cap_bakery_weights(bakery_weights, evidence)
        bakery_total = category_units["bakery"]
        caps = None
        if len(bakery) >= math.ceil(1 / BAKERY_SHARE_CAP):
            cap = _floor_percent(bakery_total, BAKERY_SHARE_CAP_PERCENT)
            if cap * len(bakery) >= bakery_total:
                caps = {product: cap for product in bakery}
        bakery_allocation = _largest_remainder(bakery_weights, bakery_total, caps)
        if caps is not None:
            bakery_allocation = _constrain_bakery_allocation(
                bakery_allocation, bakery_total, cap, evidence
            )
            if max(bakery_allocation.values()) > cap:
                raise AssertionError("Bakery allocation exceeded the single-product cap")
            top_three_units = sum(
                sorted(bakery_allocation.values(), reverse=True)[:3]
            )
            if top_three_units > _floor_percent(
                bakery_total, TOP_THREE_MAX_PERCENT
            ):
                raise AssertionError("Bakery allocation exceeded the top-three cap")
        allocation.update(bakery_allocation)
    if beverages:
        beverage_weights = _smooth_product_weights(
            {product: blended[product] for product in beverages},
            BEVERAGE_EVIDENCE_WEIGHT,
        )
        if weather is not None and (weather.temp_mean >= 28.0 or weather.temp_mean <= 15.0):
            prefer_cold = weather.temp_mean >= 28.0
            for product in beverages:
                temperatures = set(
                    beverage_capabilities(product)["allowed_temperatures"]
                )
                if temperatures == {"cold"}:
                    beverage_weights[product] *= 1.20 if prefer_cold else 0.80
                elif temperatures == {"hot"}:
                    beverage_weights[product] *= 0.80 if prefer_cold else 1.20
        allocation.update(
            _largest_remainder(beverage_weights, category_units["beverage"])
        )
    return allocation


def _basket_sizes(
    order_count: int, target_units: int, rng: random.Random
) -> list[int]:
    sizes = list(BASKET_SIZE_WEIGHTS)
    weights = [BASKET_SIZE_WEIGHTS[size] for size in sizes]
    baskets = rng.choices(sizes, weights=weights, k=order_count)
    minimum_units = math.ceil(BASKET_MEAN_MIN * order_count)
    maximum_units = math.floor(BASKET_MEAN_MAX * order_count)
    if not minimum_units <= target_units <= maximum_units:
        raise ValueError("Basket unit target is outside the approved mean range")
    current_units = sum(baskets)
    difference = target_units - current_units
    for index in range(len(baskets) - 1, -1, -1):
        if difference == 0:
            break
        if difference > 0:
            change = min(difference, max(sizes) - baskets[index])
        else:
            change = max(difference, 1 - baskets[index])
        baskets[index] += change
        difference -= change
    if difference:
        raise ValueError("Basket unit target cannot be allocated")
    return baskets


def _ticket_times(order_count: int, rng: random.Random) -> list[time]:
    result = []
    for _ in range(order_count):
        peak = rng.random()
        if peak < 0.50:
            hour = rng.randint(6, 10)
        elif peak < 0.80:
            hour = rng.randint(11, 13)
        else:
            hour = rng.randint(14, 18)
        result.append(time(hour, rng.randrange(60), rng.randrange(60)))
    return result


def _takeaway_types(order_count: int, rng: random.Random) -> list[str]:
    low = math.ceil(TAKEAWAY_RANGE[0] * order_count)
    high = math.floor(TAKEAWAY_RANGE[1] * order_count)
    count = min(max(round(rng.uniform(*TAKEAWAY_RANGE) * order_count), low), high)
    result = ["takeaway"] * count + ["dine_in"] * (order_count - count)
    rng.shuffle(result)
    return result


def _payment_methods(order_count: int, rng: random.Random) -> list[str]:
    for _ in range(100):
        raw = {method: rng.uniform(*limits) for method, limits in PAYMENT_RANGES.items()}
        total = sum(raw.values())
        shares = {method: value / total for method, value in raw.items()}
        if all(
            PAYMENT_RANGES[method][0] <= shares[method] <= PAYMENT_RANGES[method][1]
            for method in shares
        ):
            break
    else:
        raw = {method: sum(limits) / 2 for method, limits in PAYMENT_RANGES.items()}
        total = sum(raw.values())
        shares = {method: value / total for method, value in raw.items()}

    bounds = {
        method: (
            math.ceil(limits[0] * order_count),
            math.floor(limits[1] * order_count),
        )
        for method, limits in PAYMENT_RANGES.items()
    }
    best = None
    for qr_count in range(bounds["qr"][0], bounds["qr"][1] + 1):
        for card_count in range(bounds["card"][0], bounds["card"][1] + 1):
            cash_count = order_count - qr_count - card_count
            if not bounds["cash"][0] <= cash_count <= bounds["cash"][1]:
                continue
            counts = {"qr": qr_count, "card": card_count, "cash": cash_count}
            error = sum(
                (counts[method] / order_count - shares[method]) ** 2 for method in counts
            )
            candidate = (error, qr_count, card_count, cash_count)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("Payment ranges cannot accommodate the daily orders")
    methods = ["qr"] * best[1] + ["card"] * best[2] + ["cash"] * best[3]
    rng.shuffle(methods)
    return methods


def _beverage_details(
    product_name: str,
    rng: random.Random,
    weather: WeatherRecord | None = None,
) -> dict[str, str | None]:
    if not is_beverage(product_name):
        return {
            "beverage_size": None,
            "beverage_temp": None,
            "beverage_sweetness": None,
            "beverage_ice": None,
        }
    capability = beverage_capabilities(product_name)
    allowed_temperatures = tuple(capability["allowed_temperatures"])
    if weather is not None and {"hot", "cold"} <= set(allowed_temperatures):
        hot_probability = 0.15 + 0.70 / (
            1.0 + math.exp((weather.temp_mean - 22.0) / 4.0)
        )
        temperature = "hot" if rng.random() < hot_probability else "cold"
    else:
        temperature = rng.choice(allowed_temperatures)
    ice = "none" if temperature == "hot" else rng.choice(capability["allowed_ice"])
    return {
        "beverage_size": rng.choice(capability["allowed_sizes"]),
        "beverage_temp": temperature,
        "beverage_sweetness": rng.choice(capability["allowed_sugar"]),
        "beverage_ice": ice,
    }


def _day1_discount_rate(
    source: _SourceIndex, day: date, product_name: str
) -> Decimal:
    candidates = (
        source.discount_month_product.get((day.month, product_name)),
        source.discount_product.get(product_name),
        source.positive_discounts,
    )
    for counts in candidates:
        if counts:
            return historical_discount_rate(
                _primary_value(counts, "positive discount")
            )
    return Decimal("0.0")


def build_history(
    source_rows: Iterable[SourceRecord],
    product_costs: Mapping[str, Decimal],
    seed: int = RANDOM_SEED,
    holidays: Iterable[date] = (),
    start_date: date = HISTORY_START,
    end_date: date = HISTORY_END,
    weather_by_date: Mapping[date, WeatherRecord] | None = None,
) -> History:
    if not HISTORY_START <= start_date <= end_date <= HISTORY_END:
        raise ValueError("History range must stay within the approved boundaries")
    source = _index_source(source_rows)
    missing_costs = sorted(set(source.products) - set(product_costs))
    if missing_costs:
        raise ValueError(f"Missing product costs: {', '.join(missing_costs)}")
    holiday_dates = frozenset(holidays)
    orders = []
    items = []
    payments = []
    day = start_date
    while day <= end_date:
        rng = _date_rng(seed, day)
        weather = weather_by_date.get(day) if weather_by_date else None
        order_count = daily_order_target(day, rng, day in holiday_dates, weather)
        donor = _donor_profile(day, source)
        target_units = daily_unit_target(order_count, donor, source)
        baskets = _basket_sizes(order_count, target_units, rng)
        allocation = _product_allocation(
            target_units,
            donor.units,
            _trailing_units(day, source),
            source,
            weather,
            _date_rng(seed ^ 0xA5A5, day),
        )
        product_units = [
            product
            for product in source.products
            for _ in range(allocation[product])
        ]
        rng.shuffle(product_units)
        order_times = _ticket_times(order_count, rng)
        dine_types = _takeaway_types(order_count, rng)
        payment_methods = _payment_methods(order_count, rng)

        unit_offset = 0
        for ticket_index, basket_size in enumerate(baskets, start=1):
            ticket_id = f"GZ-{day:%Y%m%d}-{ticket_index:04d}"
            products = Counter(product_units[unit_offset : unit_offset + basket_size])
            unit_offset += basket_size
            ticket_items = []
            for product_name in sorted(products):
                quantity = products[product_name]
                beverage = is_beverage(product_name)
                details = _beverage_details(product_name, rng, weather)
                unit_price = money(source.prices[product_name])
                if beverage:
                    unit_price = beverage_unit_price(
                        unit_price, details["beverage_size"]
                    )
                is_day1 = 0
                if not beverage:
                    is_day1 = _feature_value(
                        source.day1_evidence,
                        day,
                        ticket_id,
                        product_name,
                        seed,
                        "is_day1",
                    )
                is_top3 = _feature_value(
                    source.top3_evidence,
                    day,
                    ticket_id,
                    product_name,
                    seed,
                    "is_top3",
                )
                discount_rate = Decimal("0.0")
                if is_day1:
                    discount_rate = _day1_discount_rate(source, day, product_name)
                gross = money(unit_price * quantity)
                discount = money(gross * discount_rate)
                line_total = money(gross - discount)
                line_profit = money(
                    line_total - money(product_costs[product_name]) * quantity
                )
                ticket_items.append(
                    ItemRecord(
                        ticket_id=ticket_id,
                        product_name=product_name,
                        quantity=quantity,
                        unit_price=unit_price,
                        discount_rate=discount_rate,
                        line_total=line_total,
                        line_profit=line_profit,
                        freshness="Day-1" if is_day1 else "Fresh",
                        is_day1=is_day1,
                        is_top3=is_top3,
                        **details,
                    )
                )
            subtotal = money(
                sum(
                    (row.unit_price * row.quantity for row in ticket_items),
                    start=Decimal("0.00"),
                )
            )
            total_amount = money(
                sum((row.line_total for row in ticket_items), start=Decimal("0.00"))
            )
            discount_total = money(subtotal - total_amount)
            total_profit = money(
                sum((row.line_profit for row in ticket_items), start=Decimal("0.0"))
            )
            orders.append(
                OrderRecord(
                    ticket_id=ticket_id,
                    order_date=day,
                    order_time=order_times[ticket_index - 1],
                    subtotal=subtotal,
                    discount_total=discount_total,
                    total_amount=total_amount,
                    total_profit=total_profit,
                    item_count=basket_size,
                    state="completed",
                    dine_type=dine_types[ticket_index - 1],
                )
            )
            items.extend(ticket_items)
            payments.append(
                PaymentRecord(
                    ticket_id=ticket_id,
                    amount=total_amount,
                    payment_method=payment_methods[ticket_index - 1],
                    payment_date=day,
                )
            )
        if unit_offset != len(product_units):
            raise AssertionError("Allocated product units do not match basket units")
        day += timedelta(days=1)
    return History(tuple(orders), tuple(items), tuple(payments))


def summarize_history(history: History) -> HistorySummary:
    if not history.orders:
        raise ValueError("History must contain orders")
    order_count = len(history.orders)
    payments = {payment.ticket_id: payment for payment in history.payments}
    return HistorySummary(
        mean_items_per_order=sum(order.item_count for order in history.orders)
        / order_count,
        one_item_share=sum(order.item_count == 1 for order in history.orders)
        / order_count,
        two_item_share=sum(order.item_count == 2 for order in history.orders)
        / order_count,
        takeaway_share=sum(order.dine_type == "takeaway" for order in history.orders)
        / order_count,
        qr_share=sum(
            payments[order.ticket_id].payment_method == "qr" for order in history.orders
        )
        / order_count,
        delivery_orders=sum(order.dine_type == "delivery" for order in history.orders),
        min_order_time=min(order.order_time for order in history.orders),
        max_order_time=max(order.order_time for order in history.orders),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/bakery_sales_raw_backup_cleaned.csv"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("data/bakery_sales_raw.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/guangzhou_rebuild_manifest.json"),
    )
    parser.add_argument(
        "--cost-catalog",
        type=Path,
        default=Path("data/reference/product_cost_catalog.json"),
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=Path("data/guangzhou_weather.csv"),
    )
    parser.add_argument("--apply-history", action="store_true")
    parser.add_argument("--confirm-database")
    return parser.parse_args(argv)


def validate_apply_request(apply_history: bool, confirmation: str | None) -> None:
    if apply_history and confirmation != "bakery_ai":
        raise SystemExit("Historical apply requires --confirm-database bakery_ai")
    if not apply_history and confirmation is not None:
        raise SystemExit("--confirm-database requires --apply-history")


RAW_OUTPUT_FIELDS = (
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
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decimal_text(value: Decimal) -> str:
    return format(value, ".1f")


def load_source_csv(path: Path) -> tuple[SourceRecord, ...]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for source in reader:
            rows.append(
                SourceRecord(
                    order_date=date.fromisoformat(source["date"]),
                    order_time=time.fromisoformat(source["time"]),
                    ticket_id=source["ticket_id"],
                    product_name=source["product_name"],
                    quantity=int(source["quantity"]),
                    unit_price=Decimal(source["unit_price_cny"]),
                    discount_rate=Decimal(source["discount_pct"]),
                    is_rainy=int(source["is_rainy"]),
                    is_member_day=int(source["is_member_day"]),
                    is_competitor=int(source["is_competitor"]),
                    is_new_product=int(source["is_new_product"]),
                    is_day1=int(source["is_day1"]),
                    is_top3=int(source["is_top3"]),
                )
            )
    return tuple(rows)


def load_weather_csv(
    path: Path, start_date: date, end_date: date
) -> dict[date, WeatherRecord]:
    if start_date > end_date:
        raise ValueError("Weather range start must not exceed end")
    records = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "temp_mean", "precipitation"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("Weather CSV is missing required columns")
        for row in reader:
            weather_date = date.fromisoformat(row["date"])
            if not start_date <= weather_date <= end_date:
                continue
            if weather_date in records:
                raise ValueError(f"Duplicate weather date: {weather_date.isoformat()}")
            temp_mean = float(row["temp_mean"])
            precipitation = float(row["precipitation"])
            if not math.isfinite(temp_mean) or not math.isfinite(precipitation):
                raise ValueError(f"Invalid weather value on {weather_date.isoformat()}")
            if precipitation < 0:
                raise ValueError(
                    f"Negative precipitation on {weather_date.isoformat()}"
                )
            records[weather_date] = WeatherRecord(temp_mean, precipitation)

    expected_dates = {
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    }
    missing = sorted(expected_dates - set(records))
    if missing:
        preview = ", ".join(day.isoformat() for day in missing[:5])
        raise ValueError(f"Weather CSV is missing active dates: {preview}")
    return records


def create_product_cost_catalog(
    connection,
    recipe_catalog_path: Path,
    output_path: Path,
    expected_product_count: int = 45,
) -> dict:
    cursor = None
    try:
        connection.start_transaction(readonly=True)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT p.product_name, p.category, pr.material_name, "
            "pr.quantity_per_unit, rm.unit_price FROM products p "
            "JOIN product_recipes pr ON pr.product_name = p.product_name "
            "JOIN raw_materials rm ON rm.material_name = pr.material_name "
            "ORDER BY p.product_name, pr.material_name"
        )
        rows = cursor.fetchall()
    finally:
        if cursor is not None:
            cursor.close()
        connection.rollback()

    recipes = defaultdict(dict)
    categories = {}
    material_prices = {}
    for product_name, category, material_name, quantity, unit_price in rows:
        quantity = Decimal(str(quantity))
        unit_price = Decimal(str(unit_price))
        if quantity <= 0 or unit_price < 0:
            raise ValueError(f"Invalid recipe cost basis: {product_name}")
        if material_name in recipes[product_name]:
            raise ValueError(f"Duplicate recipe material: {product_name}/{material_name}")
        recipes[product_name][material_name] = quantity
        categories[product_name] = category
        previous_price = material_prices.setdefault(material_name, unit_price)
        if previous_price != unit_price:
            raise ValueError(f"Conflicting material price: {material_name}")

    if len(recipes) != expected_product_count:
        raise ValueError(
            f"Expected {expected_product_count} recipe products, found {len(recipes)}"
        )

    versioned_payload = json.loads(recipe_catalog_path.read_text(encoding="utf-8"))
    versioned_recipes = {
        product["product_name"]: {
            ingredient["material_name"]: Decimal(ingredient["quantity_per_unit"])
            for ingredient in product["ingredients"]
        }
        for product in versioned_payload["products"]
    }
    mismatches = sorted(
        product_name
        for product_name, ingredients in versioned_recipes.items()
        if categories.get(product_name) != "bakery"
        or recipes.get(product_name) != ingredients
    )
    if mismatches:
        raise ValueError(f"Versioned bakery recipe mismatch: {', '.join(mismatches)}")

    products = []
    for product_name in sorted(recipes):
        recipe_cost = sum(
            (
                quantity * material_prices[material_name]
                for material_name, quantity in recipes[product_name].items()
            ),
            Decimal("0.0"),
        )
        products.append(
            {
                "product_name": product_name,
                "category": categories[product_name],
                "unit_cost_cny": _decimal_text(money(recipe_cost)),
            }
        )
    payload = {
        "schema_version": 1,
        "currency": "CNY",
        "money_step": _decimal_text(MONEY_STEP),
        "cost_basis": "recipe_quantity_per_unit_x_raw_material_unit_price",
        "product_count": len(products),
        "bakery_product_count": sum(row["category"] == "bakery" for row in products),
        "beverage_product_count": sum(
            row["category"] == "beverage" for row in products
        ),
        "products": products,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    return payload


def load_product_costs(path: Path) -> dict[str, Decimal]:
    payload = json.loads(path.read_text(encoding="ascii"))
    required_top_level = {
        "schema_version",
        "currency",
        "money_step",
        "cost_basis",
        "product_count",
        "bakery_product_count",
        "beverage_product_count",
        "products",
    }
    if not isinstance(payload, dict) or not required_top_level <= set(payload):
        raise ValueError("Product cost catalog is missing required fields")
    if payload["schema_version"] != 1:
        raise ValueError("Product cost catalog schema version must be 1")
    if payload["currency"] != "CNY":
        raise ValueError("Product cost catalog currency must be CNY")
    if payload["money_step"] != "0.1":
        raise ValueError("Product cost catalog money step must be 0.1")
    if payload["cost_basis"] != "recipe_quantity_per_unit_x_raw_material_unit_price":
        raise ValueError("Product cost catalog cost basis is invalid")
    if (
        payload["product_count"] != 45
        or payload["bakery_product_count"] != 30
        or payload["beverage_product_count"] != 15
    ):
        raise ValueError("Product cost catalog product counts are invalid")
    products = payload["products"]
    if not isinstance(products, list) or len(products) != 45:
        raise ValueError("Product cost catalog must contain exactly 45 products")

    costs = {}
    categories = Counter()
    names = []
    for row in products:
        if not isinstance(row, dict) or not {
            "product_name",
            "category",
            "unit_cost_cny",
        } <= set(row):
            raise ValueError("Product cost catalog product is missing required fields")
        name = row["product_name"]
        category = row["category"]
        raw_cost = row["unit_cost_cny"]
        if not isinstance(name, str) or not name:
            raise ValueError("Product cost catalog product name is invalid")
        if category not in {"bakery", "beverage"}:
            raise ValueError(f"Product cost catalog category is invalid: {name}")
        if not isinstance(raw_cost, str):
            raise ValueError(f"Product cost catalog cost is invalid: {name}")
        try:
            cost = Decimal(raw_cost)
        except (ArithmeticError, ValueError) as exc:
            raise ValueError(f"Product cost catalog cost is invalid: {name}") from exc
        if not cost.is_finite() or cost < 0 or cost.as_tuple().exponent != -1:
            raise ValueError(f"Product cost catalog cost precision is invalid: {name}")
        if name in costs:
            raise ValueError(f"Product cost catalog has duplicate product: {name}")
        costs[name] = cost
        categories[category] += 1
        names.append(name)
    if names != sorted(names):
        raise ValueError("Product cost catalog products must be sorted")
    if categories != Counter({"bakery": 30, "beverage": 15}):
        raise ValueError("Product cost catalog category counts are invalid")
    return costs


def _source_feature_catalog(
    source_rows: Iterable[SourceRecord],
) -> dict[str, dict[str, int]]:
    fields = (
        "is_rainy",
        "is_member_day",
        "is_competitor",
        "is_new_product",
    )
    counts = defaultdict(lambda: {field: Counter() for field in fields})
    for row in source_rows:
        if row.quantity <= 0:
            continue
        for field in fields:
            counts[row.product_name][field][getattr(row, field)] += row.quantity
    return {
        product: {
            field: _primary_value(field_counts, field)
            for field, field_counts in product_counts.items()
        }
        for product, product_counts in counts.items()
    }


def write_raw_csv(
    history: History,
    source_rows: Iterable[SourceRecord],
    output_path: Path,
    weather_by_date: Mapping[date, WeatherRecord] | None = None,
) -> int:
    features = _source_feature_catalog(source_rows)
    order_dates = {order.ticket_id: order for order in history.orders}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_row_count = 0
    with output_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_OUTPUT_FIELDS)
        writer.writeheader()
        for item in history.items:
            order = order_dates[item.ticket_id]
            product_features = features[item.product_name]
            weather = weather_by_date.get(order.order_date) if weather_by_date else None
            writer.writerow(
                {
                    "date": order.order_date.isoformat(),
                    "time": order.order_time.strftime("%H:%M:%S"),
                    "ticket_id": item.ticket_id,
                    "product_name": item.product_name,
                    "quantity": item.quantity,
                    "unit_price_cny": _decimal_text(item.unit_price),
                    "category": "beverage" if is_beverage(item.product_name) else "bakery",
                    "is_rainy": int(weather.is_rainy) if weather else product_features["is_rainy"],
                    "is_member_day": product_features["is_member_day"],
                    "is_competitor": product_features["is_competitor"],
                    "is_new_product": product_features["is_new_product"],
                    "is_day1": item.is_day1,
                    "is_top3": item.is_top3,
                    "discount_pct": _decimal_text(item.discount_rate),
                    "beverage_size": item.beverage_size or "",
                    "beverage_temp": item.beverage_temp or "",
                    "beverage_sweetness": item.beverage_sweetness or "",
                    "beverage_ice": item.beverage_ice or "",
                }
            )
            output_row_count += 1
    return output_row_count


def _share(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_manifest(
    history: History,
    source_rows: tuple[SourceRecord, ...],
    paths: BuildPaths,
    seed: int,
    output_row_count: int,
    weather_by_date: Mapping[date, WeatherRecord] | None = None,
) -> dict:
    order_count = len(history.orders)
    unit_count = sum(item.quantity for item in history.items)
    basket_counts = Counter(order.item_count for order in history.orders)
    channel_counts = Counter(order.dine_type for order in history.orders)
    payment_counts = Counter(payment.payment_method for payment in history.payments)
    product_units = Counter()
    for item in history.items:
        product_units[item.product_name] += item.quantity
    bakery_units = Counter(
        {name: units for name, units in product_units.items() if not is_beverage(name)}
    )
    bakery_total = sum(bakery_units.values())
    top_products = [
        {
            "product_name": name,
            "units": units,
            "share": _share(units, unit_count),
        }
        for name, units in sorted(
            product_units.items(), key=lambda row: (-row[1], row[0])
        )[:10]
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_sha256": _sha256(paths.source_csv),
        "weather_sha256": _sha256(paths.weather_csv),
        "output_sha256": _sha256(paths.raw_output_csv),
        "seed": seed,
        "approved_ranges": {
            "daily_orders": {
                name: list(limits) for name, limits in ORDER_RANGES.items()
            },
            "basket_mean": [BASKET_MEAN_MIN, BASKET_MEAN_MAX],
            "takeaway_share": list(TAKEAWAY_RANGE),
            "payment_shares": {
                name: list(limits) for name, limits in PAYMENT_RANGES.items()
            },
            "bakery_single_product_share_max": BAKERY_SHARE_CAP,
            "bakery_top3_share": [TOP_THREE_MIN, TOP_THREE_MAX],
            "bakery_top7_share": [0.55, 0.65],
            "minimum_bakery_product_daily_average": 2.0,
            "minimum_beverage_product_daily_average": 3.0,
        },
        "date_range": {
            "start": min(order.order_date for order in history.orders).isoformat(),
            "end": max(order.order_date for order in history.orders).isoformat(),
        },
        "row_counts": {"source": len(source_rows), "output": output_row_count},
        "order_count": order_count,
        "unit_count": unit_count,
        "revenue_cny": _decimal_text(
            money(sum((order.total_amount for order in history.orders), Decimal("0.0")))
        ),
        "basket_shares": {
            "one_item": _share(basket_counts[1], order_count),
            "two_item": _share(basket_counts[2], order_count),
            "three_plus": _share(
                sum(count for size, count in basket_counts.items() if size >= 3),
                order_count,
            ),
        },
        "channel_shares": {
            name: _share(channel_counts[name], order_count)
            for name in ("takeaway", "dine_in", "delivery")
        },
        "payment_shares": {
            name: _share(payment_counts[name], order_count)
            for name in ("qr", "card", "cash")
        },
        "product_concentration": {
            "bakery_max_share": _share(max(bakery_units.values()), bakery_total),
            "bakery_top3_share": _share(
                sum(sorted(bakery_units.values(), reverse=True)[:3]), bakery_total
            ),
            "beverage_share": _share(unit_count - bakery_total, unit_count),
            "top_products": top_products,
        },
        "weather": {
            "rain_threshold_mm": 1.0,
            "rainy_days": sum(
                record.is_rainy for record in (weather_by_date or {}).values()
            ),
            "minimum_temp_mean": min(
                (record.temp_mean for record in (weather_by_date or {}).values()),
                default=None,
            ),
            "maximum_temp_mean": max(
                (record.temp_mean for record in (weather_by_date or {}).values()),
                default=None,
            ),
        },
    }


def generate_outputs(
    paths: BuildPaths,
    seed: int = RANDOM_SEED,
    start_date: date = HISTORY_START,
    end_date: date = HISTORY_END,
) -> dict:
    source_rows = load_source_csv(paths.source_csv)
    product_costs = load_product_costs(paths.cost_catalog)
    weather_by_date = load_weather_csv(paths.weather_csv, start_date, end_date)
    history = build_history(
        source_rows,
        product_costs,
        seed=seed,
        start_date=start_date,
        end_date=end_date,
        weather_by_date=weather_by_date,
    )
    output_row_count = write_raw_csv(
        history, source_rows, paths.raw_output_csv, weather_by_date
    )
    if output_row_count != len(history.items):
        raise AssertionError(
            "CSV writer row count does not match generated history items"
        )
    manifest = build_manifest(
        history, source_rows, paths, seed, output_row_count, weather_by_date
    )
    paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    return manifest


def _validate_history(
    history: History, product_costs: Mapping[str, Decimal]
) -> None:
    if not history.orders:
        raise ValueError("History must not be empty")
    missing_costs = sorted(
        {item.product_name for item in history.items} - set(product_costs)
    )
    if missing_costs:
        raise ValueError(f"Missing product costs: {', '.join(missing_costs)}")
    items_by_ticket = defaultdict(list)
    for item in history.items:
        normalized_discount_rate = historical_discount_rate(item.discount_rate)
        if item.discount_rate != normalized_discount_rate:
            raise ValueError(
                f"Invalid persisted discount rate: {item.ticket_id}/{item.product_name}"
            )
        if is_beverage(item.product_name) and (
            item.is_day1 != 0
            or item.freshness != "Fresh"
            or item.discount_rate != Decimal("0.0")
        ):
            raise ValueError(
                f"Beverage items cannot be Day-1: {item.ticket_id}/{item.product_name}"
            )
        gross = money(item.unit_price * item.quantity)
        discount = money(gross * item.discount_rate)
        expected_line_total = money(gross - discount)
        if item.line_total != expected_line_total:
            raise ValueError(
                f"Invalid item line total: {item.ticket_id}/{item.product_name}"
            )
        expected_line_profit = money(
            expected_line_total
            - money(product_costs[item.product_name]) * item.quantity
        )
        if item.line_profit != expected_line_profit:
            raise ValueError(
                f"Invalid item line profit: {item.ticket_id}/{item.product_name}"
            )
        items_by_ticket[item.ticket_id].append(item)
    payments_by_ticket = defaultdict(list)
    for payment in history.payments:
        payments_by_ticket[payment.ticket_id].append(payment)
    for order in history.orders:
        items = items_by_ticket[order.ticket_id]
        payments = payments_by_ticket[order.ticket_id]
        if len(payments) != 1:
            raise ValueError(f"Invalid payment relation: {order.ticket_id}")
        if order.item_count != sum(item.quantity for item in items):
            raise ValueError(f"Invalid item count: {order.ticket_id}")
        if order.subtotal != money(
            sum((item.unit_price * item.quantity for item in items), Decimal("0.0"))
        ):
            raise ValueError(f"Invalid subtotal: {order.ticket_id}")
        if order.total_amount != money(order.subtotal - order.discount_total):
            raise ValueError(f"Invalid total: {order.ticket_id}")
        if order.total_amount != money(
            sum((item.line_total for item in items), Decimal("0.0"))
        ):
            raise ValueError(f"Invalid line total: {order.ticket_id}")
        if payments[0].amount != order.total_amount:
            raise ValueError(f"Invalid payment amount: {order.ticket_id}")
        if order.total_profit != money(
            sum((item.line_profit for item in items), Decimal("0.0"))
        ):
            raise ValueError(f"Invalid profit: {order.ticket_id}")


def _executemany_chunks(cursor, sql: str, rows: list[tuple], chunk_size: int) -> None:
    for offset in range(0, len(rows), chunk_size):
        cursor.executemany(sql, rows[offset : offset + chunk_size])


def _sql_reconciliation(
    cursor, history: History, product_costs: Mapping[str, Decimal]
) -> None:
    cost_rows = sorted(product_costs.items())
    cost_selects = ["SELECT %s AS product_name, %s AS unit_cost_cny"]
    cost_selects.extend("SELECT %s, %s" for _ in cost_rows[1:])
    cost_params = [value for row in cost_rows for value in row]
    cost_basis_sql = " UNION ALL ".join(cost_selects)
    checks = [
        (
            "SELECT ABS(COUNT(*) - %s) AS mismatch_count FROM orders "
            "WHERE order_date <= %s",
            (len(history.orders), HISTORY_END),
        ),
        (
            "SELECT ABS(COUNT(*) - %s) AS mismatch_count FROM order_items oi "
            "JOIN orders o ON oi.order_id = o.id WHERE o.order_date <= %s",
            (len(history.items), HISTORY_END),
        ),
        (
            "SELECT ABS(COUNT(*) - %s) AS mismatch_count FROM payments p "
            "JOIN orders o ON p.order_id = o.id WHERE o.order_date <= %s",
            (len(history.payments), HISTORY_END),
        ),
        (
            "SELECT COUNT(*) AS mismatch_count FROM ("
            "SELECT o.id FROM orders o LEFT JOIN order_items oi ON oi.order_id = o.id "
            "WHERE o.order_date <= %s GROUP BY o.id, o.item_count, o.subtotal, "
            "o.discount_total, o.total_amount, o.total_profit HAVING "
            "o.item_count <> COALESCE(SUM(oi.quantity), 0) OR "
            "o.subtotal <> COALESCE(SUM(oi.unit_price * oi.quantity), 0) OR "
            "o.total_amount <> o.subtotal - o.discount_total OR "
            "o.total_amount <> COALESCE(SUM(oi.line_total), 0) OR "
            "o.total_profit <> COALESCE(SUM(oi.line_profit), 0)) mismatches",
            (HISTORY_END,),
        ),
        (
            "SELECT COUNT(*) AS mismatch_count FROM ("
            "SELECT o.id FROM orders o LEFT JOIN payments p ON p.order_id = o.id "
            "WHERE o.order_date <= %s GROUP BY o.id, o.total_amount HAVING "
            "COUNT(p.id) <> 1 OR COALESCE(SUM(p.amount), 0) <> o.total_amount) mismatches",
            (HISTORY_END,),
        ),
    ]
    checks.append(
        (
            "SELECT COUNT(*) AS mismatch_count FROM order_items oi "
            "JOIN orders o ON oi.order_id = o.id "
            f"LEFT JOIN ({cost_basis_sql}) cost_basis "
            "ON cost_basis.product_name = oi.product_name "
            "WHERE o.order_date <= %s AND ("
            "cost_basis.product_name IS NULL OR "
            "oi.line_total <> ROUND("
            "ROUND(oi.unit_price * oi.quantity, 1) - "
            "ROUND(ROUND(oi.unit_price * oi.quantity, 1) * oi.discount_rate, 1), 1"
            ") OR oi.line_profit <> ROUND("
            "oi.line_total - cost_basis.unit_cost_cny * oi.quantity, 1"
            "))",
            tuple(cost_params + [HISTORY_END]),
        )
    )
    for sql, params in checks:
        cursor.execute(sql, params)
        if cursor.fetchone()[0] != 0:
            raise ValueError("SQL reconciliation failed")


def apply_history(
    history: History,
    connection,
    product_costs: Mapping[str, Decimal],
    chunk_size: int = 1000,
) -> None:
    invalid_dates = sorted(
        {
            order.order_date
            for order in history.orders
            if not HISTORY_START <= order.order_date <= HISTORY_END
        }
    )
    if invalid_dates:
        raise ValueError(
            "Historical apply date outside 2025-06-24 through 2026-06-23: "
            f"{invalid_dates[0].isoformat()}"
        )
    _validate_history(history, product_costs)
    cursor = None
    try:
        connection.start_transaction()
        cursor = connection.cursor()
        cursor.execute(
            "DELETE p FROM payments p JOIN orders o ON p.order_id = o.id "
            "WHERE o.order_date <= %s",
            (HISTORY_END,),
        )
        cursor.execute(
            "DELETE oi FROM order_items oi JOIN orders o ON oi.order_id = o.id "
            "WHERE o.order_date <= %s",
            (HISTORY_END,),
        )
        cursor.execute(
            "DELETE FROM orders WHERE order_date <= %s", (HISTORY_END,)
        )
        order_rows = [
            (
                order.ticket_id,
                order.order_date,
                order.order_time,
                order.subtotal,
                order.discount_total,
                order.total_amount,
                order.total_profit,
                order.item_count,
                order.state,
                order.dine_type,
            )
            for order in history.orders
        ]
        _executemany_chunks(
            cursor,
            "INSERT INTO orders (ticket_id, order_date, order_time, subtotal, "
            "discount_total, total_amount, total_profit, item_count, state, dine_type) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            order_rows,
            chunk_size,
        )
        cursor.execute(
            "SELECT id, ticket_id FROM orders WHERE order_date <= %s",
            (HISTORY_END,),
        )
        order_ids = {ticket_id: order_id for order_id, ticket_id in cursor.fetchall()}
        expected_tickets = {order.ticket_id for order in history.orders}
        if set(order_ids) != expected_tickets:
            raise ValueError("Inserted historical order identifiers do not match")
        item_rows = [
            (
                order_ids[item.ticket_id],
                item.product_name,
                item.quantity,
                item.unit_price,
                item.discount_rate,
                item.line_total,
                item.line_profit,
                item.freshness,
                item.beverage_size,
                item.beverage_temp,
                item.beverage_ice,
                item.beverage_sweetness,
            )
            for item in history.items
        ]
        _executemany_chunks(
            cursor,
            "INSERT INTO order_items (order_id, product_name, quantity, unit_price, "
            "discount_rate, line_total, line_profit, freshness, coffee_size, "
            "coffee_temp, coffee_ice, coffee_sugar) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            item_rows,
            chunk_size,
        )
        payment_rows = [
            (
                order_ids[payment.ticket_id],
                payment.payment_method,
                payment.amount,
                payment.payment_date,
            )
            for payment in history.payments
        ]
        _executemany_chunks(
            cursor,
            "INSERT INTO payments (order_id, payment_method, amount, payment_date) "
            "VALUES (%s,%s,%s,%s)",
            payment_rows,
            chunk_size,
        )
        _sql_reconciliation(cursor, history, product_costs)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_apply_request(args.apply_history, args.confirm_database)
    paths = BuildPaths(
        args.source,
        args.raw_output,
        args.manifest,
        args.cost_catalog,
        args.weather,
    )
    generate_outputs(paths)
    if args.apply_history:
        from db.mysql_client import get_db

        source_rows = load_source_csv(paths.source_csv)
        costs = load_product_costs(paths.cost_catalog)
        weather_by_date = load_weather_csv(paths.weather_csv, HISTORY_START, HISTORY_END)
        history = build_history(
            source_rows,
            costs,
            weather_by_date=weather_by_date,
        )
        connection = get_db(autocommit=False)
        try:
            apply_history(history, connection, costs)
        finally:
            connection.close()


if __name__ == "__main__":
    main()
