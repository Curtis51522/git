from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from s5_agent.s5_config.settings import THRESHOLDS
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation


def _data(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    value = raw.get("data", raw)
    if isinstance(value, dict) and isinstance(value.get("data"), dict):
        return value["data"]
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _avg(values: list[Any]) -> float:
    nums = [_float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else 0.0


def _series(data: dict[str, Any], key: str) -> list[Any]:
    trend = data.get("trend", {})
    if isinstance(trend, dict) and isinstance(trend.get(key), list):
        return trend[key]
    value = data.get(key, [])
    return value if isinstance(value, list) else []


def _pct_change(current: float, baseline: float) -> float:
    return round((current - baseline) / max(baseline, 1.0) * 100, 1)


def _total_revenue_series(data: dict[str, Any]) -> list[float]:
    orders = _series(data, "orders")
    average_order_values = _series(data, "avg_order")
    if orders and average_order_values and len(orders) == len(average_order_values):
        reconstructed = [
            round(_float(order_count) * _float(average_order_value), 2)
            for order_count, average_order_value in zip(orders, average_order_values)
        ]
        today_revenue = _float(data.get("today_revenue"))
        if not today_revenue or abs(reconstructed[-1] - today_revenue) <= max(1.0, today_revenue * 0.01):
            return reconstructed

    bread = _series(data, "bread")
    beverages = _series(data, "beverages")
    length = max(len(bread), len(beverages))
    values = []
    for index in range(length):
        values.append(
            _float(bread[index] if index < len(bread) else 0.0)
            + _float(beverages[index] if index < len(beverages) else 0.0)
        )
    return values


def _display_name(value: Any) -> str:
    return str(value or "").replace("_", " ")


async def _fetch_revenue_dashboard(params: dict[str, Any]) -> dict[str, Any]:
    date = str(params.get("date", "")) if isinstance(params, dict) else ""
    try:
        base_url = "http://127.0.0.1:8002/s4/revenue/daily"
        url = f"{base_url}?{urlencode({'date': date})}" if date else base_url
        with urlopen(url, timeout=10) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        pass
    return {"success": True, "data": {}, "tool": "revenue_dashboard_fallback"}


def _evidence(item_id: str, source: str, description: str, value: Any, **metadata: Any) -> EvidenceItem:
    return EvidenceItem(
        id=item_id,
        source=source,
        description=description,
        value=value,
        metadata=metadata,
    )


class RevenueTrendAgent:
    name = "RevenueTrendAgent"

    async def fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _fetch_revenue_dashboard(params)

    def analyze_for_graph(self, raw: dict[str, Any], params: dict[str, Any]) -> AgentOutput:
        data = _data(raw)
        bread = _series(data, "bread")
        orders = _series(data, "orders")
        average_orders = _series(data, "avg_order")
        total_revenue = _total_revenue_series(data)

        today_revenue = _float(data.get("today_revenue"))
        current_total = _float(total_revenue[-1]) if total_revenue else today_revenue
        baseline_total = _avg(total_revenue[:-1]) if len(total_revenue) > 1 else current_total
        recent_total_pct = _pct_change(current_total, baseline_total) if baseline_total else 0.0

        current_bread = _float(bread[-1]) if bread else current_total
        baseline_bread = _avg(bread[:-1]) if len(bread) > 1 else current_bread
        recent_bakery_pct = _pct_change(current_bread, baseline_bread) if baseline_bread else recent_total_pct

        current_orders = _int(orders[-1]) if orders else 0
        baseline_orders = _avg(orders[:-1]) if len(orders) > 1 else current_orders
        if "today_orders" in data:
            current_orders = _int(data.get("today_orders"))
        order_change_pct = (
            _float(data.get("orders_change"))
            if data.get("orders_change") is not None
            else (_pct_change(current_orders, baseline_orders) if baseline_orders else 0.0)
        )

        current_aov = _float(average_orders[-1]) if average_orders else 0.0
        baseline_aov = _avg(average_orders[:-1]) if len(average_orders) > 1 else current_aov
        aov_change_pct = (
            _float(data.get("avg_change"))
            if data.get("avg_change") is not None
            else (_pct_change(current_aov, baseline_aov) if baseline_aov else 0.0)
        )
        dashboard_revenue_change_pct = _float(data.get("revenue_change"))
        dashboard_profit_change_pct = _float(data.get("profit_change"))

        if dashboard_revenue_change_pct > 5:
            direction = "rising"
        elif dashboard_revenue_change_pct < -5:
            direction = "falling"
        else:
            direction = "stable"

        claim = (
            f"Revenue dashboard comparison is {direction}: total revenue moved "
            f"{dashboard_revenue_change_pct:+.1f}% vs yesterday, profit moved "
            f"{dashboard_profit_change_pct:+.1f}%, orders moved {order_change_pct:+.1f}%, "
            f"and average order value moved {aov_change_pct:+.1f}%."
        )

        low_sample_collapse = current_orders <= 3 and (dashboard_revenue_change_pct <= -50 or order_change_pct <= -50)
        risks = []
        recommendations = []
        if low_sample_collapse:
            risks.extend(["low_sample_size", "possible_data_gap"])
            recommendations.append(
                Recommendation(
                    id="revenue_data_completeness_check",
                    action=(
                        "Check whether the selected date has incomplete sales data, shortened "
                        "trading hours, or missing POS synchronization before treating the revenue drop as real demand collapse."
                    ),
                    urgency="high",
                    time_horizon="today",
                    rationale="Revenue and order volume collapsed, but the result is based on too few orders to interpret as a normal trading pattern.",
                    expected_impact="Prevents false operational decisions caused by incomplete or abnormal daily sales data.",
                    evidence_ids=[
                        "revenue",
                        "order_volume",
                        "revenue_trend_pct",
                        "order_change_pct",
                        "discount_rate_pct",
                    ],
                )
            )
        if dashboard_revenue_change_pct < -10:
            risks.append("revenue_decline")
            recommendations.append(
                Recommendation(
                    id="revenue_decline_review",
                    action="Review product-level sales drivers before setting next week's promotion plan.",
                    urgency="medium",
                    time_horizon="this_week",
                    rationale="Revenue dashboard data shows a material decline versus yesterday.",
                    expected_impact="Protects margin by targeting the products behind the decline.",
                    evidence_ids=["dashboard_revenue_change_pct"],
                )
            )
        return AgentOutput(
            agent_name=self.name,
            claim=claim,
            confidence=0.62 if low_sample_collapse else (0.8 if data else 0.45),
            metrics={
                "revenue_trend_pct": dashboard_revenue_change_pct,
                "dashboard_revenue_change_pct": dashboard_revenue_change_pct,
                "dashboard_profit_change_pct": dashboard_profit_change_pct,
                "order_change_pct": order_change_pct,
                "average_order_value_change_pct": aov_change_pct,
                "current_orders": current_orders,
                "recent_total_revenue_change_pct": recent_total_pct,
                "recent_bakery_revenue_change_pct": recent_bakery_pct,
                "trend_direction": direction,
            },
            evidence_items=[
                _evidence(
                    "dashboard_revenue_change_pct",
                    "revenue_trend",
                    "Dashboard total revenue change compared with yesterday",
                    dashboard_revenue_change_pct,
                    date=params.get("date", ""),
                ),
                _evidence(
                    "dashboard_profit_change_pct",
                    "revenue_trend",
                    "Dashboard profit change compared with yesterday",
                    dashboard_profit_change_pct,
                    date=params.get("date", ""),
                ),
                _evidence(
                    "order_change_pct",
                    "revenue_trend",
                    "Dashboard order count change compared with yesterday",
                    order_change_pct,
                    date=params.get("date", ""),
                ),
                _evidence(
                    "average_order_value_change_pct",
                    "revenue_trend",
                    "Dashboard average order value change compared with yesterday",
                    aov_change_pct,
                    date=params.get("date", ""),
                ),
                _evidence(
                    "revenue_trend_pct",
                    "revenue_trend",
                    "Dashboard total revenue change compared with yesterday",
                    dashboard_revenue_change_pct,
                    date=params.get("date", ""),
                ),
            ],
            risks=risks,
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness="fresh" if data else "unknown",
                completeness=1.0 if data else 0.4,
                source_status={"revenue_trend": "fresh" if data else "unknown"},
            ),
        )


class RevenueBenchmarkAgent:
    name = "RevenueBenchmarkAgent"

    async def fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _fetch_revenue_dashboard(params)

    def analyze_for_graph(self, raw: dict[str, Any], params: dict[str, Any]) -> AgentOutput:
        data = _data(raw)
        total_revenue = _total_revenue_series(data)
        current = _float(total_revenue[-1]) if total_revenue else _float(data.get("today_revenue"))
        recent_avg = _avg(total_revenue[:-1]) if len(total_revenue) > 1 else current
        vs_previous = _float(data.get("revenue_change"))
        vs_recent_avg = _pct_change(current, recent_avg) if recent_avg else 0.0

        if vs_recent_avg > 5:
            status = "above baseline"
        elif vs_recent_avg < -5:
            status = "below baseline"
        else:
            status = "near baseline"

        return AgentOutput(
            agent_name=self.name,
            claim=(
                f"Recent baseline check is {status}: dashboard total revenue is "
                f"{vs_recent_avg:+.1f}% against the recent total-revenue average."
            ),
            confidence=0.76 if data else 0.45,
            metrics={
                "revenue_vs_previous_day_pct": vs_previous,
                "revenue_vs_recent_avg_pct": vs_recent_avg,
                "dashboard_revenue_change_pct": vs_previous,
                "recent_total_revenue_change_pct": vs_recent_avg,
                "benchmark_status": status,
            },
            evidence_items=[
                _evidence(
                    "revenue_vs_previous_day_pct",
                    "revenue_benchmark",
                    "Dashboard total revenue change compared with yesterday",
                    vs_previous,
                    date=params.get("date", ""),
                ),
                _evidence(
                    "revenue_vs_recent_avg_pct",
                    "revenue_benchmark",
                    "Dashboard total revenue compared with the recent total-revenue average",
                    vs_recent_avg,
                    date=params.get("date", ""),
                ),
            ],
            risks=["revenue_benchmark_drop"] if vs_recent_avg < -15 else [],
            recommendations=[],
            data_quality=DataQuality(
                freshness="fresh" if data else "unknown",
                completeness=1.0 if data else 0.4,
                source_status={"revenue_benchmark": "fresh" if data else "unknown"},
            ),
        )


class OrderBehaviorAgent:
    name = "OrderBehaviorAgent"

    async def fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _fetch_revenue_dashboard(params)

    def analyze_for_graph(self, raw: dict[str, Any], params: dict[str, Any]) -> AgentOutput:
        data = _data(raw)
        orders = _series(data, "orders")
        average_orders = _series(data, "avg_order")
        current_orders = _int(orders[-1]) if orders else 0
        if "today_orders" in data:
            current_orders = _int(data.get("today_orders"))
        baseline_orders = _avg(orders[:-1]) if len(orders) > 1 else current_orders
        order_change_pct = (
            _float(data.get("orders_change"))
            if data.get("orders_change") is not None
            else (_pct_change(current_orders, baseline_orders) if baseline_orders else 0.0)
        )
        current_aov = _float(average_orders[-1]) if average_orders else 0.0
        baseline_aov = _avg(average_orders[:-1]) if len(average_orders) > 1 else current_aov
        aov_change_pct = (
            _float(data.get("avg_change"))
            if data.get("avg_change") is not None
            else (_pct_change(current_aov, baseline_aov) if baseline_aov else 0.0)
        )

        if order_change_pct > 5 and aov_change_pct < 5:
            driver = "volume-led"
        elif order_change_pct <= 5 and aov_change_pct > 5:
            driver = "basket-led"
        elif order_change_pct > 5 and aov_change_pct >= 5:
            driver = "volume-and-basket-led"
        else:
            driver = "stable"

        risks = []
        recommendations = []
        low_sample_collapse = current_orders <= 3 and order_change_pct <= -50
        if not low_sample_collapse and abs(order_change_pct - aov_change_pct) > 15:
            risks.append("order_value_shift")
            if order_change_pct < 0 < aov_change_pct:
                action = (
                    "Track traffic for 2-3 trading days before launching broad discounts; "
                    "if orders keep falling, use targeted bundles to restore visits while protecting average order value."
                )
                rationale = (
                    f"Orders moved {order_change_pct:+.1f}% on the revenue dashboard, "
                    f"while average order value moved {aov_change_pct:+.1f}%, so the day depends more on basket value than traffic."
                )
            else:
                action = (
                    "Track average order value for 2-3 trading days before launching broad discounts; "
                    "if basket value keeps weakening, use targeted bundles to lift spend per visit."
                )
                rationale = (
                    f"Orders are up {order_change_pct:.1f}% on the revenue dashboard, "
                    f"but average order value moved {aov_change_pct:+.1f}%, so revenue may be relying on more low-value orders."
                )
            recommendations.append(
                Recommendation(
                    id="average_order_value_watch",
                    action=action,
                    urgency="low",
                    time_horizon="this_week",
                    rationale=rationale,
                    expected_impact="Turns the traffic and basket-value split into a measured promotion decision instead of a broad discount reaction.",
                    evidence_ids=[
                        "profit_margin_pct",
                        "order_change_pct",
                        "average_order_value_change_pct",
                        "average_order_value",
                    ],
                )
            )

        return AgentOutput(
            agent_name=self.name,
            claim=(
                f"Revenue quality is {driver}: orders moved {order_change_pct:+.1f}% "
                f"while average order value moved {aov_change_pct:+.1f}%."
            ),
            confidence=0.77 if orders else 0.45,
            metrics={
                "order_volume_driver": driver,
                "order_change_pct": order_change_pct,
                "average_order_value_change_pct": aov_change_pct,
            },
            evidence_items=[
                _evidence(
                    "order_volume_driver",
                    "order_behavior",
                    "Revenue quality classification from order count and basket value movement",
                    driver,
                    date=params.get("date", ""),
                ),
                _evidence(
                    "order_change_pct",
                    "order_behavior",
                    "Order count compared with the recent baseline",
                    order_change_pct,
                    date=params.get("date", ""),
                ),
                _evidence(
                    "average_order_value_change_pct",
                    "order_behavior",
                    "Average order value compared with the recent baseline",
                    aov_change_pct,
                    date=params.get("date", ""),
                ),
            ],
            risks=risks,
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness="fresh" if data else "unknown",
                completeness=1.0 if data else 0.4,
                source_status={"order_behavior": "fresh" if data else "unknown"},
            ),
        )


class RevenueProductMixAgent:
    name = "RevenueProductMixAgent"

    async def fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _fetch_revenue_dashboard(params)

    def analyze_for_graph(self, raw: dict[str, Any], params: dict[str, Any]) -> AgentOutput:
        data = _data(raw)
        bread = data.get("bread_ranking", []) if isinstance(data.get("bread_ranking", []), list) else []
        beverages = data.get("beverage_ranking", []) if isinstance(data.get("beverage_ranking", []), list) else []
        category = data.get("category", {}) if isinstance(data.get("category", {}), dict) else {}

        bread_revenue = _float(category.get("Bread"))
        beverage_revenue = _float(category.get("Beverages")) + _float(category.get("Coffee"))
        total_revenue = bread_revenue + beverage_revenue
        if total_revenue <= 0:
            total_revenue = sum(_float(item.get("revenue")) for item in bread + beverages)

        top_product = bread[0] if bread else {}
        top_product_name = _display_name(top_product.get("name", ""))
        top_product_qty = _int(top_product.get("qty"))
        top_product_revenue = _float(top_product.get("revenue"))
        top_share_pct = round(top_product_revenue / max(total_revenue, 1.0) * 100, 1)
        top3_revenue = sum(_float(item.get("revenue")) for item in bread[:3])
        top3_share_pct = round(top3_revenue / max(total_revenue, 1.0) * 100, 1)
        beverage_share_pct = round(beverage_revenue / max(total_revenue, 1.0) * 100, 1)

        if top_product_name:
            claim = (
                f"Product mix is led by {top_product_name}, which sold {top_product_qty} units "
                f"and contributed {top_share_pct:.1f}% of tracked revenue; the top three bakery "
                f"items together contributed {top3_share_pct:.1f}%."
            )
        else:
            claim = "Product mix data is not available, so revenue concentration cannot be assessed."

        risks = []
        recommendations = []
        low_product_sample = top_product_qty <= 1 and total_revenue > 0
        if top3_share_pct > 60:
            risks.append("product_concentration")
            product_names = ", ".join(_display_name(item.get("name")) for item in bread[:3] if item.get("name"))
            recommendations.append(
                Recommendation(
                    id="product_concentration_review",
                    action=f"Review whether mid-tier bakery products can share demand with {product_names}.",
                    urgency="low",
                    time_horizon="this_week",
                    rationale="A large share of tracked revenue is concentrated in the leading bakery items.",
                    expected_impact="Reduces dependence on a narrow product group.",
                    evidence_ids=["top_product_revenue_share_pct"],
                )
            )

        return AgentOutput(
            agent_name=self.name,
            claim=claim,
            confidence=0.55 if low_product_sample else (0.76 if top_product_name else 0.4),
            metrics={
                "top_product": top_product_name,
                "top_product_units": top_product_qty,
                "top_product_revenue": top_product_revenue,
                "top_product_revenue_share_pct": top_share_pct,
                "top3_product_revenue_share_pct": top3_share_pct,
                "bread_revenue": bread_revenue,
                "beverage_revenue": beverage_revenue,
                "beverage_revenue_share_pct": beverage_share_pct,
            },
            evidence_items=[
                _evidence(
                    "top_product_revenue_share_pct",
                    "revenue_product_mix",
                    "Revenue share of the leading bakery product",
                    top_share_pct,
                    product=top_product_name,
                    quantity=top_product_qty,
                ),
                _evidence(
                    "top3_product_revenue_share_pct",
                    "revenue_product_mix",
                    "Revenue share of the leading three bakery products",
                    top3_share_pct,
                ),
                _evidence(
                    "category_revenue_split",
                    "revenue_product_mix",
                    "Revenue split between bakery and beverage categories",
                    {"bread": bread_revenue, "beverages": beverage_revenue},
                    beverage_share_pct=beverage_share_pct,
                ),
            ],
            risks=risks,
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness="fresh" if top_product_name else "unknown",
                completeness=1.0 if top_product_name else 0.4,
                source_status={"revenue_product_mix": "fresh" if top_product_name else "unknown"},
            ),
        )


class CategoryMixAgent:
    name = "CategoryMixAgent"

    async def fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _fetch_revenue_dashboard(params)

    def analyze_for_graph(self, raw: dict[str, Any], params: dict[str, Any]) -> AgentOutput:
        data = _data(raw)
        category = data.get("category", {}) if isinstance(data.get("category", {}), dict) else {}
        bread_revenue = _float(category.get("Bread"))
        beverage_revenue = _float(category.get("Beverages")) + _float(category.get("Coffee"))
        total_revenue = bread_revenue + beverage_revenue
        bread_share = round(bread_revenue / max(total_revenue, 1.0) * 100, 1)
        beverage_share = round(beverage_revenue / max(total_revenue, 1.0) * 100, 1)

        if beverage_share >= 20:
            balance = "mixed bakery-and-beverage"
        elif bread_share >= 80:
            balance = "bakery-led"
        else:
            balance = "balanced"

        return AgentOutput(
            agent_name=self.name,
            claim=(
                f"Category mix is {balance}: bakery revenue contributes {bread_share:.1f}% "
                f"and beverage revenue contributes {beverage_share:.1f}% of tracked revenue."
            ),
            confidence=0.74 if total_revenue > 0 else 0.4,
            metrics={
                "category_mix_balance": balance,
                "bread_revenue_share_pct": bread_share,
                "beverage_revenue_share_pct": beverage_share,
                "bread_revenue": bread_revenue,
                "beverage_revenue": beverage_revenue,
            },
            evidence_items=[
                _evidence(
                    "bread_revenue_share_pct",
                    "category_mix",
                    "Bakery revenue share of tracked category revenue",
                    bread_share,
                    date=params.get("date", ""),
                ),
                _evidence(
                    "beverage_revenue_share_pct",
                    "category_mix",
                    "Beverage revenue share of tracked category revenue",
                    beverage_share,
                    date=params.get("date", ""),
                ),
            ],
            risks=[],
            recommendations=[],
            data_quality=DataQuality(
                freshness="fresh" if total_revenue > 0 else "unknown",
                completeness=1.0 if total_revenue > 0 else 0.4,
                source_status={"category_mix": "fresh" if total_revenue > 0 else "unknown"},
            ),
        )


class HourlyRevenueAgent:
    name = "HourlyRevenueAgent"

    async def fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        date = str(params.get("date", "")) if isinstance(params, dict) else ""
        try:
            base_url = "http://127.0.0.1:8002/s4/revenue/hourly"
            url = f"{base_url}?{urlencode({'date': date})}" if date else base_url
            with urlopen(url, timeout=10) as response:
                if response.status == 200:
                    return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            pass
        return {"success": True, "data": {"hours": [], "bread": [], "beverages": []}, "tool": "hourly_revenue_fallback"}

    def analyze_for_graph(self, raw: dict[str, Any], params: dict[str, Any]) -> AgentOutput:
        data = _data(raw)
        hours = data.get("hours", []) if isinstance(data.get("hours", []), list) else []
        bread = data.get("bread", []) if isinstance(data.get("bread", []), list) else []
        beverages = data.get("beverages", []) if isinstance(data.get("beverages", []), list) else []
        revenue_values = data.get("revenue", []) if isinstance(data.get("revenue", []), list) else []
        profit_values = data.get("profit", []) if isinstance(data.get("profit", []), list) else []
        order_values = data.get("orders", []) if isinstance(data.get("orders", []), list) else []
        avg_order_values = data.get("avg_order", []) if isinstance(data.get("avg_order", []), list) else []
        margin_values = data.get("margin", []) if isinstance(data.get("margin", []), list) else []
        totals = []
        for index, hour in enumerate(hours):
            if revenue_values:
                total = _float(revenue_values[index] if index < len(revenue_values) else 0.0)
            else:
                total = _float(bread[index] if index < len(bread) else 0.0) + _float(beverages[index] if index < len(beverages) else 0.0)
            totals.append((str(hour), total))

        total_revenue = sum(value for _, value in totals)
        peak_hour = ""
        peak_revenue = 0.0
        peak_share_pct = 0.0
        low_hours = []
        if totals and total_revenue > 0:
            peak_hour, peak_revenue = max(totals, key=lambda item: item[1])
            peak_share_pct = round(peak_revenue / max(total_revenue, 1.0) * 100, 1)
            low_threshold = max(total_revenue / max(len(totals), 1) * 0.35, 20.0)
            low_hours = [hour for hour, value in totals if value <= low_threshold]

        profit_totals = []
        for index, hour in enumerate(hours):
            profit_totals.append((str(hour), _float(profit_values[index] if index < len(profit_values) else 0.0)))
        total_profit = sum(value for _, value in profit_totals)
        profit_available = bool(profit_values) and total_profit > 0
        peak_profit_hour = ""
        peak_profit = 0.0
        peak_profit_share_pct = 0.0
        peak_profit_margin_pct = 0.0
        peak_profit_orders = 0
        peak_profit_avg_order = 0.0
        if profit_available:
            peak_profit_hour, peak_profit = max(profit_totals, key=lambda item: item[1])
            peak_profit_share_pct = round(peak_profit / max(total_profit, 1.0) * 100, 1)
            peak_index = hours.index(peak_profit_hour) if peak_profit_hour in hours else -1
            if peak_index >= 0:
                peak_profit_margin_pct = _float(margin_values[peak_index] if peak_index < len(margin_values) else 0.0)
                peak_profit_orders = _int(order_values[peak_index] if peak_index < len(order_values) else 0)
                peak_profit_avg_order = _float(avg_order_values[peak_index] if peak_index < len(avg_order_values) else 0.0)

        if profit_available:
            claim = (
                f"Peak profit hour is {peak_profit_hour}, generating {chr(165)}{peak_profit:.0f} profit "
                f"and {peak_profit_share_pct:.1f}% of tracked hourly profit; peak revenue hour is {peak_hour}."
            )
        elif peak_hour:
            claim = (
                f"Peak revenue hour is {peak_hour}, contributing {peak_share_pct:.1f}% "
                "of tracked hourly revenue; hourly profit is not available, so revenue timing is used as a proxy."
            )
        else:
            claim = "Hourly revenue data is not available, so intraday profitability timing cannot be assessed."

        evidence_items = [
            _evidence(
                "peak_revenue_hour",
                "hourly_revenue",
                "Hour with the highest tracked revenue",
                peak_hour,
                date=params.get("date", ""),
                peak_revenue=round(peak_revenue, 2),
            ),
            _evidence(
                "hourly_peak_revenue_share_pct",
                "hourly_revenue",
                "Revenue share contributed by the peak hour",
                peak_share_pct,
                date=params.get("date", ""),
            ),
            _evidence(
                "low_revenue_hours",
                "hourly_revenue",
                "Hours with low tracked revenue relative to the trading day",
                low_hours[:3],
                date=params.get("date", ""),
            ),
        ]
        if profit_available:
            evidence_items.extend(
                [
                    _evidence(
                        "peak_profit_hour",
                        "hourly_profit",
                        "Hour with the highest tracked profit",
                        peak_profit_hour,
                        date=params.get("date", ""),
                        peak_profit=round(peak_profit, 2),
                    ),
                    _evidence(
                        "hourly_peak_profit_share_pct",
                        "hourly_profit",
                        "Profit share contributed by the peak profit hour",
                        peak_profit_share_pct,
                        date=params.get("date", ""),
                    ),
                ]
            )

        recommendations = []
        if profit_available and peak_profit_share_pct >= 18:
            recommendations.append(
                Recommendation(
                    id="peak_profit_window_protection",
                    action=(
                        f"Protect the {peak_profit_hour} profit window with enough stock, service coverage, "
                        "and high-margin pairing availability."
                    ),
                    urgency="low",
                    time_horizon="this_week",
                    rationale=(
                        f"The peak profit hour contributes {peak_profit_share_pct:.1f}% of tracked hourly profit "
                        f"with a {peak_profit_margin_pct:.1f}% hourly margin."
                    ),
                    expected_impact="Preserves the strongest intraday profit window without forcing broad discounts.",
                    evidence_ids=[
                        "peak_profit_hour",
                        "hourly_peak_profit_share_pct",
                        "peak_profit_margin_pct",
                    ],
                )
            )

        return AgentOutput(
            agent_name=self.name,
            claim=claim,
            confidence=0.72 if peak_hour else 0.35,
            metrics={
                "peak_revenue_hour": peak_hour,
                "peak_revenue": round(peak_revenue, 2),
                "hourly_peak_revenue_share_pct": peak_share_pct,
                "low_revenue_hours": low_hours[:3],
                "hourly_total_revenue": round(total_revenue, 2),
                "peak_profit_hour": peak_profit_hour,
                "peak_profit": round(peak_profit, 2),
                "hourly_peak_profit_share_pct": peak_profit_share_pct,
                "hourly_total_profit": round(total_profit, 2),
                "peak_profit_margin_pct": round(peak_profit_margin_pct, 1),
                "peak_profit_orders": peak_profit_orders,
                "peak_profit_avg_order": round(peak_profit_avg_order, 2),
            },
            evidence_items=evidence_items,
            risks=["hourly_profit_concentration" if profit_available else "hourly_revenue_concentration"]
            if (peak_profit_share_pct if profit_available else peak_share_pct) >= 40
            else [],
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness="fresh" if peak_hour else "unknown",
                completeness=1.0 if peak_hour and profit_available else 0.75 if peak_hour else 0.35,
                source_status={
                    "hourly_revenue": "fresh" if peak_hour else "unknown",
                    "hourly_profit": "fresh" if profit_available else "missing",
                },
                limitations=[] if profit_available else ["Hourly profit is not available; revenue timing is used as a proxy."],
            ),
        )


class DiscountImpactAgent:
    name = "DiscountImpactAgent"

    async def fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"success": True, "data": {}, "tool": "profit_metrics"}

    def analyze_for_graph(self, raw: dict[str, Any], params: dict[str, Any]) -> AgentOutput:
        data = _data(raw)
        revenue = _float(data.get("revenue", data.get("today_revenue")))
        discount_total = _float(data.get("discount_total", data.get("today_discount")))
        discount_rate_pct = round(discount_total / max(revenue, 1.0) * 100, 2)
        threshold_pct = THRESHOLDS["promo_high_discount_rate"] * 100

        if discount_rate_pct > threshold_pct:
            claim = (
                f"Discount pressure is elevated: discounts equal {discount_rate_pct:.1f}% "
                f"of revenue, above the {threshold_pct:.1f}% alert threshold."
            )
            risks = ["discount_margin_erosion"]
            recommendations = [
                Recommendation(
                    id="discount_margin_review",
                    action="Review discount rules before repeating the same promotion mix.",
                    urgency="medium",
                    time_horizon="this_week",
                    rationale="Discount exposure is above the configured alert threshold.",
                    expected_impact=round(discount_total * 0.5, 2),
                    evidence_ids=["discount_rate_pct"],
                )
            ]
        else:
            claim = (
                f"Discount exposure is controlled at {discount_rate_pct:.1f}% of revenue, "
                "so discount erosion is not visible in this revenue check."
            )
            risks = []
            recommendations = []

        return AgentOutput(
            agent_name=self.name,
            claim=claim,
            confidence=0.6 if 0 < revenue < 100 else (0.82 if revenue > 0 else 0.45),
            metrics={
                "discount_total": discount_total,
                "discount_rate_pct": discount_rate_pct,
                "discount_threshold_pct": threshold_pct,
            },
            evidence_items=[
                _evidence(
                    "discount_rate_pct",
                    "discount_impact",
                    "Discount amount as a percentage of revenue",
                    discount_rate_pct,
                    discount_total=discount_total,
                    revenue=revenue,
                )
            ],
            risks=risks,
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness="fresh" if revenue > 0 else "unknown",
                completeness=1.0 if revenue > 0 else 0.5,
                source_status={"discount_impact": "fresh" if revenue > 0 else "unknown"},
            ),
        )
