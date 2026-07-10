from __future__ import annotations

from statistics import median
from typing import Any, Iterable

from db.mysql_client import get_db


STRATEGY_DISCOUNT_PCT = {
    "clearance": 40,
    "amplify": 15,
    "margin": 25,
    "diversify": 12,
}


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _unique_products(products: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(product for product in products if product))


def build_discount_decisions(
    products: Iterable[str],
    inventory_rows: Iterable[dict[str, Any]],
    sales_rows: Iterable[dict[str, Any]],
    product_rows: Iterable[dict[str, Any]],
    priority_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    product_names = _unique_products(products)
    priorities = priority_map or {}
    inventory: dict[str, dict[str, Any]] = {}
    for row in inventory_rows:
        product = str(row.get("product_name") or "")
        if product not in product_names:
            continue
        current = inventory.setdefault(product, {"remaining": 0.0, "freshness": "Fresh"})
        current["remaining"] += _number(row.get("remaining"))
        if row.get("freshness_status") == "Day-1":
            current["freshness"] = "Day-1"

    sales = {product: {"latest": 0.0, "previous": 0.0} for product in product_names}
    for row in sales_rows:
        product = str(row.get("product_name") or "")
        period = str(row.get("period") or "")
        if product in sales and period in ("latest", "previous"):
            sales[product][period] += _number(row.get("quantity"))

    economics: dict[str, dict[str, float]] = {}
    for row in product_rows:
        product = str(row.get("product_name") or "")
        if product not in product_names:
            continue
        price = _number(row.get("selling_price"))
        cost = _number(row.get("material_cost"))
        wastage = _number(row.get("wastage_pct"))
        if wastage > 1:
            wastage /= 100
        adjusted_cost = cost * (1 + max(wastage, 0.0))
        margin = (price - adjusted_cost) / price if price > 0 else None
        economics[product] = {"price": price, "adjusted_cost": adjusted_cost, "margin": margin}

    coverage: dict[str, float | None] = {}
    for product in product_names:
        remaining = inventory.get(product, {}).get("remaining", 0.0)
        demand_basis = (sales[product]["latest"] + sales[product]["previous"]) / 2
        coverage[product] = remaining / demand_basis if remaining > 0 and demand_basis > 0 else None
    coverage_values = [value for value in coverage.values() if value is not None]
    coverage_benchmark = median(coverage_values) if coverage_values else None

    margin_values = [
        values["margin"]
        for product, values in economics.items()
        if inventory.get(product, {}).get("remaining", 0.0) > 0 and values["margin"] is not None
    ]
    margin_benchmark = median(margin_values) if margin_values else None

    decisions: dict[str, dict[str, Any]] = {}
    for product in product_names:
        remaining = _number(inventory.get(product, {}).get("remaining"))
        freshness = str(inventory.get(product, {}).get("freshness") or "Fresh")
        latest = sales[product]["latest"]
        previous = sales[product]["previous"]
        margin_value = economics.get(product, {}).get("margin")
        evidence = {
            "stock_remaining": round(remaining, 4),
            "latest_sales_units": round(latest, 4),
            "previous_sales_units": round(previous, 4),
            "stock_coverage": round(coverage[product], 4) if coverage[product] is not None else None,
            "coverage_benchmark": round(coverage_benchmark, 4) if coverage_benchmark is not None else None,
            "margin_pct": round(margin_value * 100, 4) if margin_value is not None else None,
            "margin_benchmark_pct": round(margin_benchmark * 100, 4) if margin_benchmark is not None else None,
        }

        strategy = ""
        reason = "No live discount trigger"
        source = "none"
        discount_pct = 0
        if remaining <= 0:
            reason = "No sellable finished-product stock"
        elif freshness == "Day-1":
            strategy = "clearance"
            reason = "Day-1 stock requires clearance before expiry"
            source = "freshness"
            discount_pct = STRATEGY_DISCOUNT_PCT[strategy]
        elif product in priorities:
            priority = priorities[product]
            strategy = str(priority.get("strategy") or "")
            reason = str(priority.get("reason") or "Cached revenue recommendation")
            source = "cached_revenue_analysis"
            discount_pct = int(priority.get("discount_pct") or STRATEGY_DISCOUNT_PCT.get(strategy, 0))
        elif coverage[product] is not None and coverage_benchmark is not None and coverage[product] > coverage_benchmark:
            strategy = "diversify"
            reason = "Stock coverage is above the current sellable-product benchmark"
            source = "live_policy"
            discount_pct = STRATEGY_DISCOUNT_PCT[strategy]
        elif latest > previous:
            strategy = "amplify"
            reason = "Latest trading-day unit sales are above the previous trading day"
            source = "live_policy"
            discount_pct = STRATEGY_DISCOUNT_PCT[strategy]
        elif margin_value is not None and margin_benchmark is not None and margin_value > margin_benchmark:
            strategy = "margin"
            reason = "Product margin is above the current sellable-product benchmark"
            source = "live_policy"
            discount_pct = STRATEGY_DISCOUNT_PCT[strategy]

        decisions[product] = {
            "discount_pct": min(max(discount_pct, 0), 50),
            "freshness": freshness,
            "strategy": strategy,
            "reason": reason,
            "source": source,
            "dynamic": discount_pct > 0,
            "evidence": evidence,
        }
    return decisions


def get_live_discounts(
    products: Iterable[str],
    priority_map: dict[str, dict[str, Any]] | None = None,
    db=None,
) -> dict[str, dict[str, Any]]:
    requested_products = _unique_products(products)
    if not requested_products:
        return {}

    owns_db = db is None
    active_db = db or get_db()
    cursor = active_db.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT DISTINCT product_name FROM batch_inventory WHERE quantity_remaining > 0"
        )
        benchmark_products = _unique_products(
            requested_products + [row["product_name"] for row in cursor.fetchall()]
        )
        placeholders = ",".join(["%s"] * len(benchmark_products))
        cursor.execute(
            f"""
            SELECT product_name, freshness_status, SUM(quantity_remaining) AS remaining
            FROM batch_inventory
            WHERE quantity_remaining > 0 AND product_name IN ({placeholders})
            GROUP BY product_name, freshness_status
            """,
            tuple(benchmark_products),
        )
        inventory_rows = cursor.fetchall()

        cursor.execute("SELECT DISTINCT order_date FROM orders ORDER BY order_date DESC LIMIT 2")
        trading_dates = [row["order_date"] for row in cursor.fetchall()]
        sales_rows = []
        if trading_dates:
            date_placeholders = ",".join(["%s"] * len(trading_dates))
            cursor.execute(
                f"""
                SELECT oi.product_name, o.order_date, SUM(oi.quantity) AS quantity
                FROM order_items oi
                JOIN orders o ON o.id = oi.order_id
                WHERE oi.product_name IN ({placeholders})
                  AND o.order_date IN ({date_placeholders})
                GROUP BY oi.product_name, o.order_date
                """,
                tuple(benchmark_products) + tuple(trading_dates),
            )
            date_period = {trading_dates[0]: "latest"}
            if len(trading_dates) > 1:
                date_period[trading_dates[1]] = "previous"
            for row in cursor.fetchall():
                sales_rows.append({
                    "product_name": row["product_name"],
                    "period": date_period.get(row["order_date"], ""),
                    "quantity": row["quantity"],
                })

        cursor.execute(
            f"""
            SELECT product_name, selling_price, material_cost, wastage_pct
            FROM products
            WHERE product_name IN ({placeholders})
            """,
            tuple(benchmark_products),
        )
        product_rows = cursor.fetchall()
    finally:
        cursor.close()
        if owns_db:
            active_db.close()

    decisions = build_discount_decisions(
        products=benchmark_products,
        inventory_rows=inventory_rows,
        sales_rows=sales_rows,
        product_rows=product_rows,
        priority_map=priority_map,
    )
    return {product: decisions[product] for product in requested_products}
