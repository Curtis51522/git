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
from s5_agent.agents.wastage import WastageAgent
from s5_agent.agents.yield_agent import YieldAgent
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
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation
from s5_agent.verifier.report_builder import verify_outputs

SUPPORTED_GRAPH_TEMPLATES = frozenset(
    {
        "inventory_diagnosis",
        "profit_root_cause",
        "production_advice",
        "wastage_root_cause",
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
    if graph_state.template_id == "wastage_root_cause":
        output = output.model_copy(
            update={
                "risks": [
                    risk
                    for risk in output.risks
                    if str(risk).lower() not in {"low"}
                ]
            }
        )

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["inventory"] = output
    return {"agent_outputs": agent_outputs}


async def _wastage_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("wastage", {})
    agent = WastageAgent("WastageAgent")
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["wastage"] = output
    return {"agent_outputs": agent_outputs}


async def _yield_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("yield", {})
    agent = YieldAgent("YieldAgent")
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    opinion = agent.analyze(raw, graph_state.request.params)
    data = raw.get("data", {}) if isinstance(raw, dict) and "data" in raw else raw
    if not isinstance(data, dict):
        data = {}
    materials = data.get("materials", []) or []
    product_count = _int_value(data.get("product_count"))
    total_units = _int_value(data.get("total_units"))
    available = bool(materials and total_units)
    evidence = [
        EvidenceItem(
            id="yield_data_available",
            source="production_yield",
            description="Whether production yield or bake-run data is available for the selected date",
            value=available,
            metadata={"date": graph_state.request.params.get("date", "")},
        ),
        EvidenceItem(
            id="yield_total_units",
            source="production_yield",
            description="Total production or sold units visible to the yield check",
            value=total_units,
            metadata={"date": graph_state.request.params.get("date", "")},
        ),
    ]
    recommendations = []
    warnings = []
    limitations = []
    if not available:
        warnings.append("Production yield data is missing for the selected date.")
        limitations.append("Production-side waste and yield loss cannot be fully verified without bake-run data.")
        recommendations.append(
            Recommendation(
                id="yield_data_capture",
                action="Record production yield data before judging production-side waste or bake efficiency.",
                urgency="medium",
                time_horizon="today",
                rationale="Material wastage records alone cannot verify actual bake volumes, yield loss, or production-side waste.",
                expected_impact="Separates true zero material waste from missing production-run evidence.",
                evidence_ids=["yield_data_available"],
            )
        )

    output = AgentOutput(
        agent_name="YieldAgent",
        claim=opinion.opinion,
        confidence=float(opinion.confidence),
        metrics={
            "yield_data_available": available,
            "yield_total_units": total_units,
            "yield_material_count": len(materials),
            "yield_product_count": product_count,
        },
        evidence_items=evidence,
        risks=["yield_data_gap"] if not available else [],
        recommendations=recommendations,
        data_quality=DataQuality(
            freshness="fresh" if available else "missing",
            completeness=1.0 if available else 0.0,
            warnings=warnings,
            limitations=limitations,
            source_status={"production_yield": "fresh" if available else "missing"},
        ),
        limitations=limitations,
    )

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["yield"] = output
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
    overview = graph_state.agent_outputs.get("forecast_overview")
    forecast_units = _int_value(overview.metrics.get("forecast_total_units")) if overview else 0
    total_bake = _int_value(output.metrics.get("total_bake"))
    starting_stock = _int_value(output.metrics.get("day1_stock_total"))
    total_available = total_bake + starting_stock
    if forecast_units:
        supply_coverage_pct = round(total_available / max(forecast_units, 1) * 100, 1)
        demand_gap_units = max(forecast_units - total_available, 0)
        output = output.model_copy(
            update={
                "metrics": {
                    **output.metrics,
                    "supply_coverage_pct": supply_coverage_pct,
                    "demand_gap_units": demand_gap_units,
                    "total_available_units": total_available,
                },
                "evidence_items": [
                    *output.evidence_items,
                    EvidenceItem(
                        id="supply_coverage_pct",
                        source="production_plan",
                        description="Available bake plus starting stock as a share of forecast demand",
                        value=supply_coverage_pct,
                        metadata={
                            "forecast_units": forecast_units,
                            "total_bake": total_bake,
                            "starting_stock": starting_stock,
                        },
                    ),
                    EvidenceItem(
                        id="demand_gap_units",
                        source="production_plan",
                        description="Forecast demand units not covered by planned bake plus starting stock",
                        value=demand_gap_units,
                        metadata={
                            "forecast_units": forecast_units,
                            "total_available_units": total_available,
                        },
                    ),
                ],
            }
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
    if graph_state.template_id == "wastage_root_cause":
        return {
            "synthesis": S5Synthesis(
                summary=_synthesize_wastage_summary(graph_state.agent_outputs),
                recommendations=_merge_wastage_recommendations(graph_state.agent_outputs),
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


def _money_precise(value: Any) -> str:
    try:
        return f"{chr(165)}{float(value):.2f}"
    except (TypeError, ValueError):
        return f"{chr(165)}0.00"


def _number(value: Any) -> str:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "0"


def _names(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(str(value).replace("_", " ") for value in values[:5] if value)


def _natural_names(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    names = [str(value).replace("_", " ") for value in values[:5] if value]
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


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


def _wasted_material_names(materials: Any) -> str:
    if not isinstance(materials, list):
        return ""
    return _natural_names([item.get("name") for item in materials if isinstance(item, dict)])


def _wasted_material_cost_sentence(materials: Any) -> str:
    if not isinstance(materials, list) or not materials:
        return ""
    items = [item for item in materials if isinstance(item, dict) and item.get("name")]
    if not items:
        return ""
    first = items[0]
    first_text = f"{first.get('name')} caused the largest recorded loss at {_money_precise(first.get('waste_cost'))}"
    if len(items) == 1:
        return first_text + "."
    if len(items) == 2:
        second = items[1]
        return f"{first_text}, followed by {second.get('name')} at {_money_precise(second.get('waste_cost'))}."
    second = items[1]
    third = items[2]
    return (
        f"{first_text}, followed by {second.get('name')} at {_money_precise(second.get('waste_cost'))} "
        f"and {third.get('name')} at {_money_precise(third.get('waste_cost'))}."
    )


def _merge_recommendations(outputs: dict[str, Any]) -> list[Any]:
    recommendations = []
    seen = set()
    profit = outputs.get("profit")
    trend = outputs.get("revenue_trend")
    benchmark = outputs.get("revenue_benchmark")
    order_behavior = outputs.get("order_behavior")
    product_mix = outputs.get("revenue_product_mix")
    hourly_revenue = outputs.get("hourly_revenue")
    profit_metrics = profit.metrics if profit else {}
    trend_metrics = trend.metrics if trend else {}
    product_metrics = product_mix.metrics if product_mix else {}
    hourly_metrics = hourly_revenue.metrics if hourly_revenue else {}
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
            if recommendation.id == "peak_profit_window_protection":
                peak_hour = str(hourly_metrics.get("peak_profit_hour") or "")
                top_product = str(product_metrics.get("top_product") or "").title()
                product_phrase = top_product if top_product else "the leading high-margin item"
                recommendation = recommendation.model_copy(
                    update={
                        "action": (
                            f"Before {peak_hour}, pre-stage {product_phrase}, beverage pairing options, "
                            "and service coverage for the peak profit window."
                        ),
                        "expected_impact": (
                            "Protects the strongest intraday profit window from stock or queue delays "
                            "without forcing broad discounts."
                        ),
                    }
                )
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


def _merge_wastage_recommendations(outputs: dict[str, Any]) -> list[Any]:
    recommendations = []
    seen = set()
    for output in outputs.values():
        for recommendation in output.recommendations:
            if recommendation.id in seen:
                continue
            seen.add(recommendation.id)
            recommendations.append(recommendation)
    return recommendations


def _synthesize_wastage_summary(outputs: dict[str, Any]) -> str:
    wastage = outputs.get("wastage")
    yield_output = outputs.get("yield")
    inventory = outputs.get("inventory")
    if not wastage:
        return "No material wastage analysis was produced."

    waste_metrics = wastage.metrics
    material_count = _int_value(waste_metrics.get("material_count_checked"))
    wasted_count = _int_value(waste_metrics.get("wasted_material_count"))
    waste_cost = _float_value(waste_metrics.get("total_waste_cost"))
    top_materials = _natural_names(waste_metrics.get("top_consumed_materials", []))
    top_wasted_materials = waste_metrics.get("top_wasted_materials", [])
    requested_date = str(waste_metrics.get("requested_date") or "")
    latest_record_date = str(waste_metrics.get("latest_wastage_record_date") or "")
    has_selected_date_check = bool(waste_metrics.get("has_selected_date_wastage_check"))
    yield_available = bool(yield_output and yield_output.metrics.get("yield_data_available"))

    if material_count and wasted_count == 0:
        opening = (
            f"Material wastage records are clean for the selected day: {material_count} materials were checked, "
            f"0 materials logged waste, and recorded waste cost is {chr(165)}{waste_cost:.2f}."
        )
    elif material_count:
        opening = (
            "Today's material waste is limited in cost, but it should still be checked. "
            f"The system found waste in {wasted_count} of {material_count} checked materials, "
            f"with a total recorded waste cost of {chr(165)}{waste_cost:.2f}."
        )
    else:
        opening = "Material wastage cannot be verified because no material wastage records were available for the selected day."

    sentences = [opening]
    if requested_date and latest_record_date and not has_selected_date_check:
        sentences.append(
            f"No wastage check was submitted on {requested_date}; the latest available material wastage records up to this date are from {latest_record_date}."
        )
    if material_count and wasted_count > 0:
        sentences.append(
            "This is not a major financial loss yet, but repeated small losses in the same materials could become a process issue."
        )
        top_wasted_names = _wasted_material_names(top_wasted_materials)
        if top_wasted_names:
            sentences.append(f"The main items to review are {top_wasted_names}.")
        cost_sentence = _wasted_material_cost_sentence(top_wasted_materials)
        if cost_sentence:
            sentences.append(cost_sentence)
        if any(isinstance(item, dict) and not item.get("rate_available") for item in top_wasted_materials):
            sentences.append(
                "Their wastage rates cannot be calculated reliably because theoretical consumption is recorded as zero, so the issue may be either real handling waste or an incomplete production-consumption baseline."
            )
    elif top_materials:
        sentences.append(
            f"The highest-consumption materials in the check are {top_materials}, so they should remain the first audit trail if waste changes later."
        )

    if yield_available:
        total_units = _int_value(yield_output.metrics.get("yield_total_units"))
        material_rows = _int_value(yield_output.metrics.get("yield_material_count"))
        sentences.append(
            f"Production-yield data is available for this day, with {total_units} units and {material_rows} consumed-material rows. That means waste records can be cross-checked against production activity."
        )
    else:
        sentences.append(
            "Production-yield risk cannot be fully verified today because no bake-run or yield data was available; this limits any claim about actual production efficiency."
        )

    if inventory:
        inventory_units = _int_value(inventory.metrics.get("inventory"))
        day1_units = _int_value(inventory.metrics.get("day1_available"))
        sentences.append(
            f"Finished stock adds context: {inventory_units} units are recorded, including {day1_units} day-1 units, so finished stock should still be interpreted separately from material waste."
        )

    if material_count and wasted_count == 0:
        sentences.append(
            "The right conclusion is that material wastage records are clean, not that every material cost has already turned into sales."
        )
    return " ".join(sentences)


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
        if driver == "basket-led":
            quality_note = "so the day is being carried by larger baskets rather than more customers"
        elif driver == "volume-led":
            quality_note = "so the day is being carried mainly by customer traffic rather than basket expansion"
        elif driver == "volume-and-basket-led":
            quality_note = "so revenue quality is supported by both traffic and basket value"
        else:
            quality_note = "so the day should be read through both traffic and basket value"
        sentences.append(
            f"Revenue quality is {driver}: orders moved "
            f"{_float_value(metrics.get('order_change_pct')):+.1f}% while average order value moved "
            f"{_float_value(metrics.get('average_order_value_change_pct')):+.1f}%, {quality_note}."
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
        peak_profit_hour = str(metrics.get("peak_profit_hour") or "")
        peak_hour = str(metrics.get("peak_revenue_hour") or "")
        if peak_profit_hour:
            peak_profit_orders = _int_value(metrics.get("peak_profit_orders"))
            sentences.append(
                f"Peak profit hour is {peak_profit_hour}, generating {_money(metrics.get('peak_profit'))} profit across "
                f"{peak_profit_orders} {_plural(peak_profit_orders, 'order')} at "
                f"{_float_value(metrics.get('peak_profit_margin_pct')):.1f}% hourly margin and contributing "
                f"{_float_value(metrics.get('hourly_peak_profit_share_pct')):.1f}% of tracked hourly profit; "
                f"peak revenue hour is {peak_hour}, contributing "
                f"{_float_value(metrics.get('hourly_peak_revenue_share_pct')):.1f}% of tracked hourly revenue."
            )
        elif peak_hour:
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
    accuracy = outputs.get("forecast_accuracy")
    forecast_units = _int_value(overview.metrics.get("forecast_total_units")) if overview else 0
    total_bake = _int_value(production.metrics.get("total_bake")) if production else 0
    profit_gap = _int_value(production.metrics.get("scenario_profit_gap")) if production else 0
    base_units = int(total_bake * 0.85) if total_bake else 0
    contingency_units = int(profit_gap / 10) if profit_gap else 0
    top_products = overview.metrics.get("top_forecast_products", []) if overview else []
    top_driver = _names(top_products[:1]) if isinstance(top_products, list) else ""
    evidence_ids = ["scenario_profit_gap", "production_waste_rate_pct"]
    if production:
        evidence_ids.extend(["supply_coverage_pct", "demand_gap_units"])
    if accuracy:
        evidence_ids.extend(["forecast_wape", "forecast_coverage"])

    recommendations = []
    for output in outputs.values():
        for recommendation in output.recommendations:
            if "base bake" in recommendation.action and forecast_units and base_units:
                priority_sentence = (
                    f" Prioritize the top forecast driver first ({top_driver}) when releasing extra bake."
                    if top_driver
                    else " Prioritize the top forecast driver first when releasing extra bake."
                )
                action = (
                    f"Start with an 85% base bake of {base_units} units. Keep the remaining "
                    f"planned bake flexible, and treat the +{contingency_units} contingency units "
                    f"as capacity reserve toward the {forecast_units}-unit demand forecast, not an "
                    f"automatic bake. Use the first 1-2 trading days as the release gate: if actual "
                    f"sales track close to forecast, release extra capacity; if sales are weak, do not "
                    f"release the contingency bake automatically.{priority_sentence}"
                )
                recommendations.append(
                    recommendation.model_copy(
                        update={
                            "action": action,
                            "evidence_ids": list(dict.fromkeys(evidence_ids)),
                        }
                    )
                )
            else:
                recommendations.append(recommendation)
    return recommendations


def _synthesize_forecast_summary(outputs: dict[str, Any]) -> str:
    overview = outputs.get("forecast_overview")
    uncertainty = outputs.get("forecast_uncertainty")
    production = outputs.get("production")
    materials = outputs.get("materials")
    accuracy = outputs.get("forecast_accuracy")

    sentences = ["The 7-day production plan is economically positive, but it is deliberately conservative against the demand forecast."]
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
            f"{_money(forecast_revenue)} projected revenue"
        )
        if trend != "unknown":
            sentence += f", with a {trend} pattern across the week"
        if top_products:
            sentence += f"; the main demand drivers are {top_products}"
        sentences.append(sentence + ".")

    if production:
        metrics = production.metrics
        coverage_pct = round(total_available / max(forecast_units, 1) * 100, 1) if forecast_units else 0.0
        sentences.append(
            f"The production plan bakes {_number(total_bake)} units for "
            f"{_money(metrics.get('total_revenue'))} projected plan revenue and {_money(metrics.get('total_profit'))} planned profit, "
            f"while {starting_stock} starting-stock units give {total_available} units available, "
            f"covering {coverage_pct:.1f}% of forecast demand and "
            f"leaving a {supply_gap}-unit gap if demand follows the forecast."
        )
        if forecast_units and total_bake:
            base_units = int(total_bake * 0.85)
            top_products = _natural_names(overview.metrics.get("top_forecast_products", [])) if overview else ""
            material_watchlist = []
            if materials:
                material_watchlist = list(materials.metadata.get("critical_materials", [])) + list(
                    materials.metadata.get("low_materials", [])
                )
            sentences.append(
                f"Three production choices are visible: hold at {total_bake} planned units and accept that a {supply_gap}-unit supply gap remains, "
                f"expand toward {forecast_units} forecast units with higher waste exposure, or stage production from a {base_units}-unit base with conditional release capacity."
            )
            sentences.append(
                "The staged option is preferred because it preserves upside capacity while using early demand evidence to control waste; "
                "release extra bake only after the first 1-2 trading days confirm that sales are tracking close to forecast."
            )
            if top_products:
                sentences.append(
                    f"Product-level release priority should go to {top_products}, because these products drive the forecast and should be reviewed first when deciding where extra bake capacity is worth using."
                )
            if material_watchlist:
                sentences.append(
                    "Material constraints should shape the release order, so products depending on low-stock materials should not receive extra bake capacity until procurement is confirmed."
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
                f"Recent forecast reliability shows {wape}% error and {coverage}% coverage. "
                f"The {wape}% recent error means the week should not be locked in at once; "
                f"{coverage}% coverage is useful for release guardrails, so staged production and material readiness checks should guide extra bake releases."
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


def build_wastage_graph():
    graph = StateGraph(S5GraphState)
    graph.add_node("wastage", _wastage_node)
    graph.add_node("yield", _yield_node)
    graph.add_node("inventory", _inventory_node)
    graph.add_node("evidence", _evidence_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("wastage")
    graph.add_edge("wastage", "yield")
    graph.add_edge("yield", "inventory")
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
    if template_id == "wastage_root_cause":
        return build_wastage_graph()
    return build_inventory_graph()
