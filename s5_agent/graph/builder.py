from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from s5_agent.agents.forecast_accuracy import ForecastAccuracyAgent
from s5_agent.agents.forecast_overview import ForecastOverviewAgent
from s5_agent.agents.forecast_uncertainty import ForecastUncertaintyAgent
from s5_agent.agents.inventory import InventoryAgent
from s5_agent.agents.material_procurement import MaterialProcurementAgent
from s5_agent.agents.profit import ProfitAgent
from s5_agent.agents.production_plan import ProductionPlanAgent
from s5_agent.agents.revenue_analysis import (
    CategoryMixAgent,
    DiscountImpactAgent,
    HourlyRevenueAgent,
    OrderBehaviorAgent,
    RevenueBenchmarkAgent,
    RevenueProductMixAgent,
    RevenueTrendAgent,
)
from s5_agent.evidence.builder import build_evidence_graph
from s5_agent.graph.state import S5GraphState, S5Synthesis
from s5_agent.schemas.recommendation import Recommendation
from s5_agent.verifier.report_builder import verify_outputs

SUPPORTED_GRAPH_TEMPLATES = frozenset(
    {
        "inventory_diagnosis",
        "profit_root_cause",
        "production_advice",
    }
)


def _normalize_state(state: S5GraphState | dict[str, Any]) -> S5GraphState:
    if isinstance(state, S5GraphState):
        return state
    return S5GraphState.model_validate(state)


async def _inventory_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("inventory", {})
    agent = InventoryAgent()
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["inventory"] = output
    return {"agent_outputs": agent_outputs}


async def _profit_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("profit", {})
    agent = ProfitAgent("ProfitAgent")
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["profit"] = output
    return {"agent_outputs": agent_outputs}


async def _revenue_trend_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("revenue_trend", {})
    agent = RevenueTrendAgent()
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["revenue_trend"] = output
    return {"agent_outputs": agent_outputs}


async def _revenue_benchmark_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("revenue_benchmark", graph_state.raw_inputs.get("revenue_trend", {}))
    agent = RevenueBenchmarkAgent()
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["revenue_benchmark"] = output
    return {"agent_outputs": agent_outputs}


async def _order_behavior_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("order_behavior", graph_state.raw_inputs.get("revenue_trend", {}))
    agent = OrderBehaviorAgent()
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["order_behavior"] = output
    return {"agent_outputs": agent_outputs}


async def _revenue_product_mix_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("revenue_product_mix", {})
    agent = RevenueProductMixAgent()
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["revenue_product_mix"] = output
    return {"agent_outputs": agent_outputs}


async def _category_mix_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("category_mix", graph_state.raw_inputs.get("revenue_product_mix", {}))
    agent = CategoryMixAgent()
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["category_mix"] = output
    return {"agent_outputs": agent_outputs}


async def _hourly_revenue_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("hourly_revenue", {})
    agent = HourlyRevenueAgent()
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["hourly_revenue"] = output
    return {"agent_outputs": agent_outputs}


async def _discount_impact_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("discount_impact", {})
    if not raw:
        profit_output = graph_state.agent_outputs.get("profit")
        raw = {"data": profit_output.metrics if profit_output else {}}
    agent = DiscountImpactAgent()
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["discount_impact"] = output
    return {"agent_outputs": agent_outputs}


async def _production_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("production", {})
    agent = ProductionPlanAgent("ProductionPlanAgent")
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(
        raw,
        graph_state.request.params,
    )

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["production"] = output
    return {"agent_outputs": agent_outputs}


async def _forecast_overview_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("forecast_overview", {})
    agent = ForecastOverviewAgent("ForecastOverviewAgent")
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["forecast_overview"] = output
    return {"agent_outputs": agent_outputs}


async def _forecast_uncertainty_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("forecast_uncertainty", {})
    agent = ForecastUncertaintyAgent("ForecastUncertaintyAgent")
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["forecast_uncertainty"] = output
    return {"agent_outputs": agent_outputs}


async def _materials_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("materials", {})
    agent = MaterialProcurementAgent("MaterialProcurementAgent")
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["materials"] = output
    return {"agent_outputs": agent_outputs}


async def _forecast_accuracy_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("forecast_accuracy", {})
    agent = ForecastAccuracyAgent("ForecastAccuracyAgent")
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["forecast_accuracy"] = output
    return {"agent_outputs": agent_outputs}


def _evidence_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    return {"evidence_graph": build_evidence_graph(graph_state.agent_outputs)}


def _verify_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    return {"verification_report": verify_outputs(list(graph_state.agent_outputs.values()))}


def _synthesize_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    outputs = list(graph_state.agent_outputs.values())
    if not outputs:
        return {"synthesis": S5Synthesis(summary="No agent analysis was produced.")}

    if graph_state.template_id == "production_advice" and len(outputs) > 1:
        return {
            "synthesis": S5Synthesis(
                summary=_synthesize_forecast_summary(graph_state.agent_outputs),
                recommendations=_forecast_recommendations(graph_state.agent_outputs),
            )
        }
    if graph_state.template_id == "profit_root_cause" and len(outputs) > 1:
        return {
            "synthesis": S5Synthesis(
                summary=_synthesize_revenue_summary(graph_state.agent_outputs),
                recommendations=_merge_recommendations(graph_state.agent_outputs),
            )
        }

    return {
        "synthesis": S5Synthesis(
            summary=" ".join(output.claim for output in outputs),
            recommendations=[
                recommendation
                for output in outputs
                for recommendation in output.recommendations
            ],
        )
    }


def _money(value: Any) -> str:
    try:
        return f"{chr(165)}{float(value):.0f}"
    except (TypeError, ValueError):
        return f"{chr(165)}0"


def _number(value: Any) -> str:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "0"


def _names(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(str(value).replace("_", " ") for value in values[:5] if value)


def _int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural or f"{singular}s"


def _merge_recommendations(outputs: dict[str, Any]) -> list[Any]:
    recommendations = []
    seen = set()
    profit = outputs.get("profit")
    trend = outputs.get("revenue_trend")
    benchmark = outputs.get("revenue_benchmark")
    order_behavior = outputs.get("order_behavior")
    profit_metrics = profit.metrics if profit else {}
    trend_metrics = trend.metrics if trend else {}
    orders = _int_value(profit_metrics.get("orders"))
    revenue_trend_pct = _float_value(trend_metrics.get("revenue_trend_pct"))
    order_change_pct = _float_value(trend_metrics.get("order_change_pct"))
    low_sample_collapse = orders <= 3 and (revenue_trend_pct <= -50 or order_change_pct <= -50)
    for output in outputs.values():
        for recommendation in output.recommendations:
            if low_sample_collapse and recommendation.id == "profit_margin_recovery":
                continue
            if recommendation.id in seen:
                continue
            seen.add(recommendation.id)
            recommendations.append(recommendation)
    risks = [risk for output in outputs.values() for risk in output.risks]
    revenue = _float_value(profit_metrics.get("revenue"))
    if not recommendations and not risks and revenue > 0:
        recommendations.append(
            Recommendation(
                id="revenue_no_action_decision",
                action="No immediate revenue intervention is recommended.",
                urgency="low",
                time_horizon="ongoing",
                rationale="Margin, discount exposure, product concentration, and the recent revenue pattern are within acceptable ranges.",
                expected_impact="Avoids unnecessary promotions or pricing changes when evidence does not justify action.",
                evidence_ids=[
                    "profit_margin_pct",
                    "discount_rate_pct",
                    "top3_product_revenue_share_pct",
                    "revenue_trend_pct",
                ],
            )
        )
    recommendations.sort(key=lambda item: 0 if item.id == "revenue_data_completeness_check" else 1)
    return recommendations


def _synthesize_revenue_summary(outputs: dict[str, Any]) -> str:
    profit = outputs.get("profit")
    trend = outputs.get("revenue_trend")
    benchmark = outputs.get("revenue_benchmark")
    order_behavior = outputs.get("order_behavior")
    product_mix = outputs.get("revenue_product_mix")
    category_mix = outputs.get("category_mix")
    hourly_revenue = outputs.get("hourly_revenue")
    discount = outputs.get("discount_impact")

    profit_metrics = profit.metrics if profit else {}
    revenue = _float_value(profit_metrics.get("revenue"))
    profit_value = _float_value(profit_metrics.get("profit"))
    margin_pct = _float_value(profit_metrics.get("profit_margin_pct"))
    orders = _int_value(profit_metrics.get("orders"))
    average_order_value = _float_value(profit_metrics.get("average_order_value"))
    discount_rate_pct = _float_value(discount.metrics.get("discount_rate_pct")) if discount else 0.0
    trend_metrics = trend.metrics if trend else {}
    revenue_trend_pct = _float_value(trend_metrics.get("revenue_trend_pct"))
    order_change_pct = _float_value(trend_metrics.get("order_change_pct"))
    low_sample_collapse = orders <= 3 and (revenue_trend_pct <= -50 or order_change_pct <= -50)

    if low_sample_collapse:
        if margin_pct >= 20:
            opening = "This was not a healthy revenue day overall, even though margin remained strong"
        else:
            opening = "This was not a healthy revenue day overall"
    elif margin_pct >= 20:
        opening = "This was a healthy revenue day"
    else:
        opening = "This revenue day needs margin attention"

    if low_sample_collapse:
        if margin_pct >= 20:
            margin_note = "so the margin signal is positive but based on too little sales volume to judge normal performance"
        else:
            margin_note = "so margin cannot be judged reliably until sales data is verified"
        first_sentence = (
            f"{opening}: revenue reached {_money(revenue)} from {orders} {_plural(orders, 'order')}, "
            f"with {_money(profit_value)} profit and an {margin_pct:.1f}% margin, {margin_note}."
        )
    else:
        first_sentence = (
            f"{opening}: revenue reached {_money(revenue)} and profit reached {_money(profit_value)}, "
            f"producing an {margin_pct:.1f}% margin on {orders} {_plural(orders, 'order')} with average order value at "
            f"{chr(165)}{average_order_value:.2f}."
        )
    sentences = [
        first_sentence
    ]

    if trend:
        metrics = trend_metrics
        direction = metrics.get("trend_direction", "stable")
        sentences.append(
            f"Revenue dashboard comparison against the previous day is {direction}: total revenue moved "
            f"{_float_value(metrics.get('dashboard_revenue_change_pct', revenue_trend_pct)):+.1f}% vs yesterday, "
            f"profit moved {_float_value(metrics.get('dashboard_profit_change_pct')):+.1f}%, "
            f"order count moved {order_change_pct:+.1f}%, and average order value moved "
            f"{_float_value(metrics.get('average_order_value_change_pct')):+.1f}%."
        )

    if benchmark:
        metrics = benchmark.metrics
        sentences.append(
            f"The recent baseline check adds context: dashboard total revenue is "
            f"{_float_value(metrics.get('revenue_vs_recent_avg_pct')):+.1f}% against the recent total-revenue average."
        )

    if order_behavior:
        metrics = order_behavior.metrics
        driver = str(metrics.get("order_volume_driver") or "stable")
        sentences.append(
            f"Revenue quality is {driver}: orders moved "
            f"{_float_value(metrics.get('order_change_pct')):+.1f}% while average order value moved "
            f"{_float_value(metrics.get('average_order_value_change_pct')):+.1f}%, so the day should be read through both traffic and basket value."
        )

    if product_mix:
        metrics = product_mix.metrics
        top_product = str(metrics.get("top_product") or "")
        if top_product:
            sentences.append(
                f"Product mix was led by {top_product}, which sold "
                f"{_number(metrics.get('top_product_units'))} {_plural(_int_value(metrics.get('top_product_units')), 'unit')} and contributed "
                f"{_float_value(metrics.get('top_product_revenue_share_pct')):.1f}% of tracked revenue; "
                f"the leading three bakery products contributed "
                f"{_float_value(metrics.get('top3_product_revenue_share_pct')):.1f}%."
            )

    if category_mix:
        metrics = category_mix.metrics
        sentences.append(
            f"Category mix is {metrics.get('category_mix_balance')}: bakery contributes "
            f"{_float_value(metrics.get('bread_revenue_share_pct')):.1f}% and beverages contribute "
            f"{_float_value(metrics.get('beverage_revenue_share_pct')):.1f}% of tracked category revenue."
        )

    if hourly_revenue:
        metrics = hourly_revenue.metrics
        peak_hour = str(metrics.get("peak_revenue_hour") or "")
        if peak_hour:
            sentences.append(
                f"Peak revenue hour is {peak_hour}, contributing "
                f"{_float_value(metrics.get('hourly_peak_revenue_share_pct')):.1f}% of tracked hourly revenue; "
                "hourly profit is not available, so revenue timing is used as a proxy for intraday profitability."
            )

    if discount:
        threshold_pct = _float_value(discount.metrics.get("discount_threshold_pct"))
        if discount_rate_pct > threshold_pct and threshold_pct > 0:
            sentences.append(
                f"Discount exposure was {discount_rate_pct:.1f}% of revenue, above the {threshold_pct:.1f}% alert threshold, so discount rules should be reviewed before repeating the same promotion mix."
            )
        elif discount_rate_pct > 0:
            if low_sample_collapse:
                sentences.append(
                    f"Discount exposure was {discount_rate_pct:.1f}% of revenue, below the {threshold_pct:.1f}% alert threshold; with only {orders} {_plural(orders, 'order')}, it is a watch signal rather than evidence of margin erosion."
                )
            else:
                sentences.append(
                    f"Discount exposure was {discount_rate_pct:.1f}% of revenue, below the {threshold_pct:.1f}% alert threshold, so it is not evidence of margin erosion yet."
                )
        else:
            sentences.append("No discount erosion is visible in the revenue data.")

    if low_sample_collapse:
        sentences.append(
            "The first decision should be to verify data completeness or abnormal trading conditions before changing pricing, production, or promotion plans; waste impact is not included in this check."
        )
    else:
        if not discount:
            sentences.append("No discount erosion is visible in the revenue data.")
        sentences.append(
            "Waste impact is not included in this check."
            )
    return " ".join(sentences)


def _forecast_recommendations(outputs: dict[str, Any]) -> list[Any]:
    overview = outputs.get("forecast_overview")
    production = outputs.get("production")
    forecast_units = _int_value(overview.metrics.get("forecast_total_units")) if overview else 0
    total_bake = _int_value(production.metrics.get("total_bake")) if production else 0
    profit_gap = _int_value(production.metrics.get("scenario_profit_gap")) if production else 0
    base_units = int(total_bake * 0.85) if total_bake else 0
    contingency_units = int(profit_gap / 10) if profit_gap else 0

    recommendations = []
    for output in outputs.values():
        for recommendation in output.recommendations:
            if "base bake" in recommendation.action and forecast_units and base_units:
                action = (
                    f"Start with an 85% base bake of {base_units} units. Keep the remaining "
                    f"planned bake flexible, and treat the +{contingency_units} contingency units "
                    f"as capacity reserve toward the {forecast_units}-unit demand forecast, not an "
                    f"automatic bake; release it only if early sales confirm the expected demand path."
                )
                recommendations.append(recommendation.model_copy(update={"action": action}))
            else:
                recommendations.append(recommendation)
    return recommendations


def _synthesize_forecast_summary(outputs: dict[str, Any]) -> str:
    overview = outputs.get("forecast_overview")
    uncertainty = outputs.get("forecast_uncertainty")
    production = outputs.get("production")
    materials = outputs.get("materials")
    accuracy = outputs.get("forecast_accuracy")

    sentences = ["This week's plan is profitable, but it is deliberately conservative against the demand forecast."]
    forecast_units = _int_value(overview.metrics.get("forecast_total_units")) if overview else 0
    forecast_revenue = overview.metrics.get("forecast_total_revenue") if overview else 0
    total_bake = _int_value(production.metrics.get("total_bake")) if production else 0
    starting_stock = _int_value(production.metrics.get("day1_stock_total")) if production else 0
    total_available = total_bake + starting_stock
    supply_gap = max(forecast_units - total_available, 0)

    if overview:
        metrics = overview.metrics
        trend = metrics.get("forecast_trend", "unknown")
        top_products = _names(metrics.get("top_forecast_products", []))
        sentence = (
            f"Demand is forecast at {_number(forecast_units)} units and "
            f"{_money(forecast_revenue)} revenue"
        )
        if trend != "unknown":
            sentence += f", with a {trend} pattern across the week"
        if top_products:
            sentence += f"; the main demand drivers are {top_products}"
        sentences.append(sentence + ".")

    if production:
        metrics = production.metrics
        sentences.append(
            f"The production plan bakes {_number(total_bake)} units for "
            f"{_money(metrics.get('total_revenue'))} revenue and {_money(metrics.get('total_profit'))} profit, "
            f"while {starting_stock} starting-stock units give {total_available} units available, "
            f"leaving a {supply_gap}-unit gap if demand follows the forecast."
        )
        if metrics.get("waste_rate_pct") is not None:
            sentences.append(
                f"Expected-demand waste exposure is {metrics.get('waste_rate_pct')}%, so staged release remains the safer operating choice if early sales are weak."
            )

    if uncertainty:
        metrics = uncertainty.metrics
        uncertain_products = _names(metrics.get("top_uncertain_products", []))
        if uncertain_products:
            sentences.append(
                f"Uncertainty is concentrated in {uncertain_products}, with an average demand range of {_number(metrics.get('forecast_avg_interval_width'))} units."
            )

    if materials:
        metrics = materials.metrics
        low_count = int(metrics.get("low_material_count", 0) or 0)
        critical_count = int(metrics.get("critical_material_count", 0) or 0)
        total_order = metrics.get("material_total_order", 0)
        if low_count or critical_count:
            low_label = "item still needs" if low_count == 1 else "items still need"
            if critical_count:
                sentences.append(
                    f"Material readiness needs attention: {critical_count} critical and {low_count} low-stock {low_label} attention, with {_number(total_order)} units to order."
                )
            else:
                sentences.append(
                    f"No material is critical yet, but {low_count} low-stock {low_label} attention, with {_number(total_order)} units to order."
                )
        else:
            sentences.append("Material readiness does not show critical or low-stock blockers for this plan.")

    if accuracy:
        metrics = accuracy.metrics
        wape = metrics.get("forecast_wape")
        coverage = metrics.get("forecast_coverage")
        if wape or coverage:
            sentences.append(
                f"Recent forecast reliability shows {wape}% error and {coverage}% coverage, so the recommendation should stay evidence-led rather than committing the full plan upfront."
            )

    return " ".join(sentences)


def build_inventory_graph():
    graph = StateGraph(S5GraphState)
    graph.add_node("inventory", _inventory_node)
    graph.add_node("evidence", _evidence_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("inventory")
    graph.add_edge("inventory", "evidence")
    graph.add_edge("evidence", "verify")
    graph.add_edge("verify", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def build_profit_graph():
    graph = StateGraph(S5GraphState)
    graph.add_node("profit", _profit_node)
    graph.add_node("revenue_benchmark", _revenue_benchmark_node)
    graph.add_node("revenue_trend", _revenue_trend_node)
    graph.add_node("order_behavior", _order_behavior_node)
    graph.add_node("revenue_product_mix", _revenue_product_mix_node)
    graph.add_node("category_mix", _category_mix_node)
    graph.add_node("hourly_revenue", _hourly_revenue_node)
    graph.add_node("discount_impact", _discount_impact_node)
    graph.add_node("evidence", _evidence_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("profit")
    graph.add_edge("profit", "revenue_benchmark")
    graph.add_edge("revenue_benchmark", "revenue_trend")
    graph.add_edge("revenue_trend", "order_behavior")
    graph.add_edge("order_behavior", "revenue_product_mix")
    graph.add_edge("revenue_product_mix", "category_mix")
    graph.add_edge("category_mix", "hourly_revenue")
    graph.add_edge("hourly_revenue", "discount_impact")
    graph.add_edge("discount_impact", "evidence")
    graph.add_edge("evidence", "verify")
    graph.add_edge("verify", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def build_production_graph():
    graph = StateGraph(S5GraphState)
    graph.add_node("production", _production_node)
    graph.add_node("evidence", _evidence_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("production")
    graph.add_edge("production", "evidence")
    graph.add_edge("evidence", "verify")
    graph.add_edge("verify", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def build_forecast_graph():
    graph = StateGraph(S5GraphState)
    graph.add_node("forecast_overview", _forecast_overview_node)
    graph.add_node("forecast_uncertainty", _forecast_uncertainty_node)
    graph.add_node("production", _production_node)
    graph.add_node("materials", _materials_node)
    graph.add_node("forecast_accuracy", _forecast_accuracy_node)
    graph.add_node("evidence", _evidence_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("forecast_overview")
    graph.add_edge("forecast_overview", "forecast_uncertainty")
    graph.add_edge("forecast_uncertainty", "production")
    graph.add_edge("production", "materials")
    graph.add_edge("materials", "forecast_accuracy")
    graph.add_edge("forecast_accuracy", "evidence")
    graph.add_edge("evidence", "verify")
    graph.add_edge("verify", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def build_s5_graph(template_id: str):
    if template_id not in SUPPORTED_GRAPH_TEMPLATES:
        supported = ", ".join(sorted(SUPPORTED_GRAPH_TEMPLATES))
        raise ValueError(f"Unsupported S5 graph template: {template_id}. Supported: {supported}")
    if template_id == "profit_root_cause":
        return build_profit_graph()
    if template_id == "production_advice":
        return build_forecast_graph()
    return build_inventory_graph()
