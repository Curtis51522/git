from __future__ import annotations

import math
from typing import Any

from langgraph.graph import END, StateGraph

from s5_agent.agents.forecast_accuracy import ForecastAccuracyAgent
from s5_agent.agents.forecast_overview import ForecastOverviewAgent
from s5_agent.agents.forecast_uncertainty import ForecastUncertaintyAgent
from s5_agent.agents.inventory import InventoryAgent
from s5_agent.agents.material_procurement import MaterialProcurementAgent
from s5_agent.agents.product_mix import ProductMixAgent
from s5_agent.agents.profit import ProfitAgent
from s5_agent.agents.production_plan import ProductionPlanAgent
from s5_agent.agents.promo import PromoAgent
from s5_agent.agents.wastage import WastageAgent
from s5_agent.agents.yield_agent import YieldAgent
from s5_agent.core.base import AgentOpinion
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
from s5_agent.s5_config.settings import THRESHOLDS
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
        "promotion_mix_analysis",
    }
)


def _normalize_state(state: S5GraphState | dict[str, Any]) -> S5GraphState:
    if isinstance(state, S5GraphState):
        return state
    return S5GraphState.model_validate(state)


def _agent_output_from_opinion(agent_name: str, opinion: AgentOpinion) -> AgentOutput:
    attribution = opinion.attribution if isinstance(opinion.attribution, dict) else {}
    evidence_source = opinion.evidence if isinstance(opinion.evidence, dict) else {}
    evidence_items = [
        EvidenceItem(
            id=f"{agent_name.lower()}_claim",
            source=agent_name,
            description=f"{agent_name} claim",
            value=opinion.opinion,
            metadata={key: value for key, value in attribution.items()},
        )
    ]
    for key, value in evidence_source.items():
        evidence_items.append(
            EvidenceItem(
                id=f"{agent_name.lower()}_{key}",
                source=agent_name,
                description=str(key).replace("_", " "),
                value=value,
            )
        )

    return AgentOutput(
        agent_name=agent_name,
        claim=opinion.opinion or f"{agent_name} completed analysis.",
        confidence=float(opinion.confidence),
        metrics={key: value for key, value in attribution.items()},
        evidence_items=evidence_items,
        risks=list(opinion.constraints),
        recommendations=[],
        data_quality=DataQuality(
            freshness="fresh",
            completeness=1.0,
            source_status={agent_name: "fresh"},
        ),
        limitations=[],
        errors=[],
        metadata={"source_agent": opinion.agent, "elapsed_ms": opinion.elapsed_ms},
    )


async def _inventory_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("inventory", {})
    agent = InventoryAgent()
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = agent.analyze_for_graph(raw, graph_state.request.params)
    output = output.model_copy(update={"agent_name": "FinishedStockAgent"})
    if graph_state.template_id == "wastage_root_cause":
        output = output.model_copy(
            update={
                "risks": [],
                "recommendations": [],
            }
        )

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["inventory"] = output
    return {"agent_outputs": agent_outputs}


async def _stock_data_quality_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    inventory = graph_state.agent_outputs.get("inventory")
    metrics = inventory.metrics if inventory else {}
    total_units = _int_value(metrics.get("inventory"))
    product_count = _int_value(metrics.get("product_count"))
    zero_count = _int_value(metrics.get("zero_stock_product_count"))
    low_count = _int_value(metrics.get("low_stock_product_count"))
    flow_balance_issue_count = _int_value(metrics.get("flow_balance_issue_count"))
    overdue_stock_total = _int_value(metrics.get("overdue_stock_total"))

    if not inventory or product_count == 0:
        status = "missing"
        claim = "Finished-product inventory records are missing for the selected scope."
        confidence = 0.3
        risks = ["inventory_data_gap"]
    elif total_units == 0 and zero_count == product_count:
        status = "all_zero"
        claim = "Finished-product inventory records show all-zero stock, so the result must be treated as stockout risk or inventory sync gap."
        confidence = 0.82
        risks = ["inventory_data_gap"]
    elif zero_count:
        status = "partial_zero"
        claim = f"Finished-product inventory records are available, but {zero_count} products have zero recorded stock."
        confidence = 0.86
        risks = ["stockout_risk"]
    else:
        status = "usable"
        claim = "Finished-product inventory records are usable for stock-risk review."
        confidence = 0.9
        risks = []
    if flow_balance_issue_count:
        record_phrase = (
            "record does" if flow_balance_issue_count == 1 else "records do"
        )
        claim += (
            f" {flow_balance_issue_count} baked-product flow {record_phrase} not reconcile."
        )
        confidence = min(confidence, 0.72)
        risks.append("inventory_flow_data_gap")
    if overdue_stock_total:
        claim += (
            f" {overdue_stock_total} expired finished-product units remain pending disposal verification."
        )
        confidence = min(confidence, 0.78)
        risks.append("expired_stock_pending_disposal_risk")

    output = AgentOutput(
        agent_name="StockDataQualityAgent",
        claim=claim,
        confidence=confidence,
        metrics={
            "inventory_record_status": status,
            "zero_stock_product_count": zero_count,
            "low_stock_product_count": low_count,
            "flow_balance_issue_count": flow_balance_issue_count,
            "overdue_stock_total": overdue_stock_total,
        },
        evidence_items=[
            EvidenceItem(
                id="inventory_record_status",
                source="inventory_quality",
                description="Finished-product inventory record status",
                value=status,
                metadata={"product_count": product_count, "total_units": total_units},
            ),
            EvidenceItem(
                id="zero_stock_product_count",
                source="inventory_quality",
                description="Number of products with zero recorded finished stock",
                value=zero_count,
                metadata={"product_count": product_count},
            ),
            EvidenceItem(
                id="flow_balance_issue_count",
                source="inventory_quality",
                description="Baked-product flow records that do not reconcile",
                value=flow_balance_issue_count,
            ),
            EvidenceItem(
                id="overdue_stock_total",
                source="inventory_quality",
                description="Expired finished-product units pending disposal verification",
                value=overdue_stock_total,
            ),
        ],
        risks=risks,
        data_quality=DataQuality(
            freshness="fresh" if status != "missing" else "missing",
            completeness=0.5 if status in {"missing", "all_zero"} else 1.0,
            limitations=(
                ["All-zero finished stock must be verified against batch records before operational decisions."]
                if status == "all_zero"
                else []
            ) + (
                ["Baked-product inflow and outflow records do not reconcile."]
                if flow_balance_issue_count
                else []
            ) + (
                ["Expired positive balances require disposal-record verification."]
                if overdue_stock_total
                else []
            ),
            source_status={
                "inventory_quality": "fresh" if status != "missing" else "missing",
                "inventory_flow": "unknown" if flow_balance_issue_count else "fresh",
            },
        ),
        limitations=(
            ["All-zero finished stock must be verified against batch records before operational decisions."]
            if status == "all_zero"
            else []
        ) + (
            ["Baked-product inflow and outflow records do not reconcile."]
            if flow_balance_issue_count
            else []
        ) + (
            ["Expired positive balances require disposal-record verification."]
            if overdue_stock_total
            else []
        ),
    )

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["stock_data_quality"] = output
    return {"agent_outputs": agent_outputs}


async def _inventory_recommendation_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    quality = graph_state.agent_outputs.get("stock_data_quality")
    quality_metrics = quality.metrics if quality else {}
    status = str(quality_metrics.get("inventory_record_status") or "missing")
    inventory = graph_state.agent_outputs.get("inventory")
    inventory_metrics = inventory.metrics if inventory else {}
    zero_products = inventory_metrics.get("zero_stock_products", []) or []
    low_products = inventory_metrics.get("low_stock_products", []) or []
    low_materials = inventory_metrics.get("low_stock_materials", []) or []
    critical_materials = inventory_metrics.get("critical_materials", []) or []
    high_sell_through_products = inventory_metrics.get("high_sell_through_products", []) or []
    slow_moving_products = inventory_metrics.get("slow_moving_products", []) or []
    overdue_stock_total = _int_value(
        inventory_metrics.get("overdue_stock_total")
    )
    overdue_stock_products = (
        inventory_metrics.get("overdue_stock_products", []) or []
    )
    flow_balance_issue_count = _int_value(
        inventory_metrics.get("flow_balance_issue_count")
    )
    recommendations = []
    if overdue_stock_total:
        product_scope = (
            _natural_names(overdue_stock_products)
            if overdue_stock_products
            else "the affected batches"
        )
        recommendations.append(
            Recommendation(
                id="inventory_expired_stock_audit",
                action=f"Remove the {overdue_stock_total} expired units for {product_scope} from physical stock and verify the disposal outflow records.",
                urgency="high",
                time_horizon="today",
                rationale="These units are older than Day-1 and are excluded from sellable finished-product stock.",
                expected_impact="Prevents expired products from being sold while preserving a complete stock audit trail.",
                evidence_ids=["overdue_stock_total"],
            )
        )
    if flow_balance_issue_count:
        recommendations.append(
            Recommendation(
                id="inventory_flow_record_audit",
                action="Verify the baked-product inflow and outflow records before using stock movement to change production.",
                urgency="high",
                time_horizon="today",
                rationale=f"{flow_balance_issue_count} baked-product flow record{'s' if flow_balance_issue_count != 1 else ''} do not reconcile.",
                expected_impact="Prevents an inconsistent stock ledger from driving production or replenishment decisions.",
                evidence_ids=["flow_balance_issue_count"],
            )
        )
    if status in {"all_zero", "missing"}:
        recommendations.append(
            Recommendation(
                id="inventory_stock_record_audit",
                action="Verify finished-product inventory records before treating the selected scope as truly out of stock.",
                urgency="high",
                time_horizon="today",
                rationale="Finished-product stock records are missing or all zero, which may indicate either a real stockout or an inventory sync gap.",
                expected_impact="Prevents production or sales decisions from being made on a potentially incomplete stock record.",
                evidence_ids=["inventory_total", "inventory_record_status"],
            )
        )
        claim = "Inventory action priority is to verify finished-product stock records before making production or sales decisions."
        confidence = 0.86
    else:
        if high_sell_through_products:
            high_sell_through_names = _natural_names(high_sell_through_products)
            if len(high_sell_through_products) == 1:
                high_sell_through_action = (
                    f"Prioritize {high_sell_through_names} in the next production review, "
                    "then confirm the quantity against forecast demand."
                )
            else:
                high_sell_through_action = (
                    f"Prioritize the {len(high_sell_through_products)} high-sell-through products "
                    f"in the next production review, led by {high_sell_through_names}, then confirm "
                    "the quantities against forecast demand."
                )
            recommendations.append(
                Recommendation(
                    id="inventory_high_sell_through_review",
                    action=high_sell_through_action,
                    urgency="high",
                    time_horizon="today",
                    rationale="These selected-date batches have high sell-through and no more than one unit left.",
                    expected_impact="Uses observed stock movement to protect availability without replacing the demand forecast.",
                    evidence_ids=["flow_baked_units", "flow_sell_through_pct"],
                )
            )
        if slow_moving_products:
            recommendations.append(
                Recommendation(
                    id="inventory_slow_moving_bake_review",
                    action=f"Stage or reduce the next bake for {_natural_names(slow_moving_products)} unless forecast demand justifies the remaining stock.",
                    urgency="medium",
                    time_horizon="today",
                    rationale="These selected-date batches have low sell-through and at least two units left.",
                    expected_impact="Reduces avoidable finished-product carryover while keeping the decision tied to forecast demand.",
                    evidence_ids=["flow_baked_units", "flow_sell_through_pct"],
                )
            )
        generic_zero_products = [
            name for name in zero_products if name not in high_sell_through_products
        ]
        if generic_zero_products:
            recommendations.append(
                Recommendation(
                    id="inventory_zero_stock_plan_check",
                    action=f"Check {_natural_names(generic_zero_products)} against the next production plan before the next trading period.",
                    urgency="high",
                    time_horizon="today",
                    rationale="These products have no finished stock recorded in the selected inventory snapshot.",
                    expected_impact="Prioritizes confirmed zero-stock products without treating the snapshot as proof of lost sales.",
                    evidence_ids=["inventory_total", "zero_stock_product_count"],
                )
            )
        generic_low_products = [
            name for name in low_products if name not in high_sell_through_products
        ]
        if generic_low_products:
            product_word = "product" if len(generic_low_products) == 1 else "products"
            remaining_word = (
                "remaining "
                if len(generic_low_products) < len(low_products)
                else ""
            )
            recommendations.append(
                Recommendation(
                    id="inventory_thin_stock_plan_check",
                    action=f"Review the {remaining_word}{len(generic_low_products)} {product_word} with only one unit against forecast demand and the next production plan.",
                    urgency="medium",
                    time_horizon="today",
                    rationale="One-unit stock across this product group leaves little operating buffer, but replenishment priority should follow expected demand rather than product-name order.",
                    expected_impact="Directs limited production capacity toward products with the strongest expected demand.",
                    evidence_ids=["low_stock_product_count", "thin_stock_product_share_pct"],
                )
            )
        if low_materials:
            recommendations.append(
                Recommendation(
                    id="inventory_material_restock_check",
                    action=f"Review raw-material replenishment for {_natural_names(low_materials)} before confirming the next production plan.",
                    urgency="high" if critical_materials else "medium",
                    time_horizon="today",
                    rationale="These materials are at or below their configured reorder points.",
                    expected_impact="Prevents planned production from being constrained by an avoidable material shortage.",
                    evidence_ids=["low_stock_material_count"],
                )
            )
        claim = (
            "Inventory action priority is to review the identified stock constraints before the next production plan."
            if recommendations
            else "Inventory action priority does not require an immediate stock intervention."
        )
        confidence = 0.75

    output = AgentOutput(
        agent_name="InventoryRecommendationAgent",
        claim=claim,
        confidence=confidence,
        metrics={"inventory_recommendation_count": len(recommendations)},
        evidence_items=[
            EvidenceItem(
                id="inventory_recommendation_basis",
                source="inventory_recommendation",
                description="Inventory recommendation trigger status",
                value=status,
                metadata={"recommendation_count": len(recommendations)},
            )
        ],
        recommendations=recommendations,
        data_quality=DataQuality(
            freshness="fresh",
            completeness=1.0,
            source_status={"inventory_recommendation": "fresh"},
        ),
    )

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["inventory_recommendation"] = output
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


async def _promotion_signal_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("promotion_signal", {})
    agent = PromoAgent("PromotionSignalAgent")
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = _agent_output_from_opinion(
        "PromotionSignalAgent",
        agent.analyze(raw, graph_state.request.params),
    )
    output = _attach_promotion_signal_metrics(output, raw)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["promotion_signal"] = output
    return {"agent_outputs": agent_outputs}


async def _promotion_product_mix_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("product_mix", {})
    agent = ProductMixAgent("PromotionProductMixAgent")
    if not raw:
        raw = await agent.fetch(graph_state.request.params)
    output = _agent_output_from_opinion(
        "PromotionProductMixAgent",
        agent.analyze(raw, graph_state.request.params),
    )
    output = _attach_promotion_product_mix_metrics(output, raw)

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["promotion_product_mix"] = output
    return {"agent_outputs": agent_outputs}


def _promotion_decision_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    recommendations = _promotion_mix_recommendations(graph_state.agent_outputs)
    evidence_ids = sorted(
        {
            evidence_id
            for recommendation in recommendations
            for evidence_id in recommendation.evidence_ids
        }
    )
    claim = (
        "Promotion decisions are ready for verifier review with evidence-backed recommendations."
        if recommendations
        else "Promotion data does not support a verified promotion recommendation."
    )
    output = AgentOutput(
        agent_name="PromotionDecisionAgent",
        claim=claim,
        confidence=0.75,
        metrics={"recommendation_count": len(recommendations)},
        evidence_items=[
            EvidenceItem(
                id="promotion_decision_basis",
                source="promotion_decision",
                description="Promotion decision evidence references",
                value=evidence_ids,
                metadata={"recommendation_count": len(recommendations)},
            )
        ],
        risks=[],
        recommendations=recommendations,
        data_quality=DataQuality(
            freshness="fresh",
            completeness=1.0,
            source_status={"promotion_decision": "fresh"},
        ),
    )

    agent_outputs = dict(graph_state.agent_outputs)
    agent_outputs["promotion_decision"] = output
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
    forecast_units = _int_value(overview.metrics.get("forecast_bakery_units")) if overview else 0
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
                        description="Available bake plus starting stock as a share of bakery forecast demand",
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
                        description="Bakery forecast demand units not covered by planned bake plus starting stock",
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
    report = verify_outputs(list(graph_state.agent_outputs.values()))
    if graph_state.template_id == "production_advice" and not _forecast_data_available(
        graph_state.agent_outputs.get("forecast_overview")
    ):
        warnings = list(report.data_quality_warnings)
        warning = "Forecast demand data is unavailable for the selected horizon."
        if warning not in warnings:
            warnings.append(warning)
        report = report.model_copy(
            update={
                "passed": False,
                "data_quality_warnings": warnings,
            }
        )
    return {"verification_report": report}


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
    if graph_state.template_id == "inventory_diagnosis":
        return {
            "synthesis": S5Synthesis(
                summary=_synthesize_inventory_summary(graph_state.agent_outputs),
                recommendations=_merge_wastage_recommendations(graph_state.agent_outputs),
            )
        }
    if graph_state.template_id == "promotion_mix_analysis":
        decision = graph_state.agent_outputs.get("promotion_decision")
        return {
            "synthesis": S5Synthesis(
                summary=_synthesize_promotion_mix_summary(graph_state.agent_outputs),
                recommendations=list(decision.recommendations) if decision else [],
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


def _display_name(value: Any) -> str:
    return str(value).replace("_", " ").title()


def _format_percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number.is_integer():
        return f"{int(number)}%"
    return f"{number:.1f}%"


def _event_label(event_type: Any) -> str:
    labels = {
        "new_product_launch": "New Product Launch",
        "competitor_activity": "Competitor Activity",
    }
    return labels.get(str(event_type), _display_name(event_type))


def _business_event_products(events: list[dict[str, Any]]) -> list[str]:
    products = []
    seen = set()
    for event in events:
        for product in event.get("products", []) or []:
            name = _display_name(product)
            if name and name not in seen:
                seen.add(name)
                products.append(name)
    return products


def _unique_business_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_events = []
    seen = set()
    for event in events:
        products = tuple(sorted(str(product) for product in (event.get("products", []) or [])))
        key = (
            str(event.get("event_type") or ""),
            str(event.get("start_date") or ""),
            str(event.get("end_date") or ""),
            products,
            str(event.get("discount_pct") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_events.append(event)
    return unique_events


def _business_event_summary_sentence(events: list[dict[str, Any]]) -> str:
    unique_events = _unique_business_events(events)
    if not unique_events:
        return ""
    count_labels = {1: "One", 2: "Two", 3: "Three", 4: "Four"}
    if len(unique_events) == 1:
        count_word = "One planned business event is"
    else:
        count_word = f"{count_labels.get(len(unique_events), str(len(unique_events)))} planned business events are"
    clauses = []
    for event in unique_events[:4]:
        label = _event_label(event.get("event_type"))
        products = _natural_names([_display_name(product) for product in event.get("products", []) or []])
        discount = _format_percent(event.get("discount_pct"))
        discount_text = f" with a {discount} planned discount" if discount else ""
        if event.get("event_type") == "new_product_launch":
            clauses.append(
                f"{label} applies to {products or 'selected products'}{discount_text}, so its demand should be monitored as a launch scenario with a weaker historical baseline"
            )
        elif event.get("event_type") == "competitor_activity":
            clauses.append(
                f"{label} applies to {products or 'selected products'}{discount_text}, so its demand should be treated as scenario-sensitive to competitor response"
            )
        else:
            clauses.append(
                f"{label} applies to {products or 'selected products'}{discount_text}"
            )
    if len(unique_events) == 1:
        scenario_sentence = (
            "This business event is a reserved scenario input, not part of the deployed 27-feature forecast model, "
            "so it should guide monitoring and staged release decisions rather than directly changing the forecast output."
        )
    else:
        scenario_sentence = (
            "These business events are reserved scenario inputs, not part of the deployed 27-feature forecast model, "
            "so they should guide monitoring and staged release decisions rather than directly changing the forecast output."
        )
    return f"{count_word} active in this forecast window. " + ". ".join(clauses) + f". {scenario_sentence}"


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


def _optional_float_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping_data(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    data = raw.get("data")
    if isinstance(data, dict):
        return data
    return raw


def _first_optional_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _optional_float_value(mapping.get(key))
        if value is not None:
            return value
    return None


def _normalize_rate_pct(value: Any) -> float | None:
    rate = _optional_float_value(value)
    if rate is None:
        return None
    if 0 <= rate <= 1:
        return round(rate * 100, 4)
    return round(rate, 4)


def _revenue_from_rows(rows: Any) -> float:
    if not isinstance(rows, list):
        return 0.0
    return sum(_float_value(row.get("revenue")) for row in rows if isinstance(row, dict))


def _category_value(category: dict[str, Any], *keys: str) -> float:
    lowered = {str(key).lower(): value for key, value in category.items()}
    return sum(_float_value(lowered.get(key.lower())) for key in keys)


def _attach_promotion_signal_metrics(output: AgentOutput, raw: Any) -> AgentOutput:
    data = _mapping_data(raw)
    metrics = dict(output.metrics)
    evidence_items = list(output.evidence_items)

    discount_total = _first_optional_float(data, "today_discount", "discount_total", "discount")
    revenue = _first_optional_float(data, "today_revenue", "revenue")
    discount_rate_pct = _normalize_rate_pct(data.get("discount_rate"))
    expired_cost = _first_optional_float(data, "expired_cost")
    expired_products = data.get("expired_products", [])
    basket_metrics = {
        "items_per_order": _first_optional_float(data, "items_per_order"),
        "items_per_order_change_pct": _first_optional_float(
            data,
            "items_per_order_change_pct",
            "items_per_order_change",
        ),
        "revenue_per_item": _first_optional_float(data, "revenue_per_item"),
        "revenue_per_item_change_pct": _first_optional_float(
            data,
            "revenue_per_item_change_pct",
            "revenue_per_item_change",
        ),
    }
    if not isinstance(expired_products, list):
        expired_products = []

    if discount_rate_pct is None and revenue and discount_total is not None:
        discount_rate_pct = round(discount_total / revenue * 100, 4)
    if discount_rate_pct is None:
        discount_rate_pct = _normalize_rate_pct(metrics.get("discount_rate_pct"))
    if discount_rate_pct is None:
        discount_rate_pct = _normalize_rate_pct(metrics.get("deviation"))

    if discount_total is not None:
        metrics["discount_total"] = discount_total
        evidence_items.append(
            EvidenceItem(
                id="discount_total",
                source="promotion_signal",
                description="Total discount amount tracked for the promotion period",
                value=discount_total,
                metadata={"revenue": revenue},
            )
        )
    if discount_rate_pct is not None:
        metrics["discount_rate_pct"] = discount_rate_pct
        evidence_items.append(
            EvidenceItem(
                id="discount_rate_pct",
                source="promotion_signal",
                description="Discount amount as a share of tracked revenue",
                value=discount_rate_pct,
                metadata={"discount_total": discount_total, "revenue": revenue},
            )
        )

    basket_descriptions = {
        "items_per_order": "Average number of products in each order",
        "items_per_order_change_pct": "Items per order compared with the previous trading day",
        "revenue_per_item": "Average revenue per sold product",
        "revenue_per_item_change_pct": "Revenue per item compared with the previous trading day",
    }
    for metric_id, value in basket_metrics.items():
        if value is None:
            continue
        metrics[metric_id] = value
        evidence_items.append(
            EvidenceItem(
                id=metric_id,
                source="promotion_signal",
                description=basket_descriptions[metric_id],
                value=value,
            )
        )

    expired_cost_revenue_pct = 0.0
    if expired_cost is not None and revenue:
        expired_cost_revenue_pct = round(expired_cost / revenue * 100, 4)
    if expired_cost is not None:
        metrics["expired_cost"] = expired_cost
        metrics["expired_cost_revenue_pct"] = expired_cost_revenue_pct
        evidence_items.extend(
            [
                EvidenceItem(
                    id="expired_cost",
                    source="promotion_signal",
                    description="Closing unsold-product cost from the revenue dashboard",
                    value=expired_cost,
                    metadata={"revenue": revenue},
                ),
                EvidenceItem(
                    id="expired_cost_revenue_pct",
                    source="promotion_signal",
                    description="Closing unsold-product cost as a share of revenue",
                    value=expired_cost_revenue_pct,
                    metadata={
                        "alert_threshold_pct": THRESHOLDS[
                            "profit_expired_cost_alert_pct"
                        ]
                    },
                ),
            ]
        )
    if expired_products:
        metrics["expired_products"] = expired_products
        evidence_items.append(
            EvidenceItem(
                id="expired_products",
                source="promotion_signal",
                description=(
                    "Products discarded at closing with loss, sales, margin, and "
                    "sell-through context"
                ),
                value=expired_products,
            )
        )

    risks = list(output.risks)
    if (
        expired_cost_revenue_pct
        >= THRESHOLDS["profit_expired_cost_alert_pct"]
    ):
        risks.append("unsold_product_loss")
    items_change_pct = basket_metrics["items_per_order_change_pct"]
    revenue_per_item_change_pct = basket_metrics["revenue_per_item_change_pct"]
    if (
        items_change_pct is not None
        and items_change_pct <= -10
        and revenue_per_item_change_pct is not None
        and revenue_per_item_change_pct >= -5
    ):
        risks.append("basket_size_weakness")

    return output.model_copy(
        update={
            "metrics": metrics,
            "evidence_items": evidence_items,
            "risks": list(dict.fromkeys(risks)),
        }
    )


def _product_mix_metrics(raw: Any) -> dict[str, Any]:
    data = _mapping_data(raw)
    bread_rows = data.get("bread_ranking", [])
    beverage_rows = data.get("beverage_ranking", [])
    category = data.get("category", {})
    category = category if isinstance(category, dict) else {}

    bread_revenue = _category_value(category, "Bread")
    beverage_revenue = _category_value(category, "Beverages", "Coffee")
    if bread_revenue <= 0:
        bread_revenue = _revenue_from_rows(bread_rows)
    if beverage_revenue <= 0:
        beverage_revenue = _revenue_from_rows(beverage_rows)

    total_tracked_revenue = bread_revenue + beverage_revenue
    bread_top3_revenue = 0.0
    top_product_name = None
    top_product_revenue = None
    top_product_units = None
    if isinstance(bread_rows, list) and bread_rows:
        bread_top3_revenue = sum(
            _float_value(row.get("revenue"))
            for row in bread_rows[:3]
            if isinstance(row, dict)
        )
        first_bread = bread_rows[0]
        if isinstance(first_bread, dict):
            top_product_name = first_bread.get("name")
            top_product_revenue = _optional_float_value(first_bread.get("revenue"))
            top_product_units = _optional_float_value(first_bread.get("qty"))

    metrics: dict[str, Any] = {}
    if top_product_name:
        metrics["top_product_name"] = top_product_name
    if top_product_units is not None:
        metrics["top_product_units"] = top_product_units
    if bread_revenue > 0:
        metrics["bread_revenue"] = round(bread_revenue, 4)
        metrics["top3_bread_revenue_share_pct"] = round(bread_top3_revenue / bread_revenue * 100, 4)
    if beverage_revenue > 0:
        metrics["beverage_revenue"] = round(beverage_revenue, 4)
    sold_bread_sku_count = _int_value(data.get("sold_bread_sku_count"))
    if sold_bread_sku_count:
        metrics["sold_bread_sku_count"] = sold_bread_sku_count
    total_bread_sku = _int_value(data.get("total_bread_sku"))
    if total_bread_sku:
        metrics["total_bread_sku"] = total_bread_sku
    if isinstance(beverage_rows, list) and beverage_rows:
        first_beverage = beverage_rows[0]
        if isinstance(first_beverage, dict):
            metrics["top_beverage_name"] = first_beverage.get("name")
            units = _optional_float_value(first_beverage.get("qty"))
            if units is not None:
                metrics["top_beverage_units"] = units
    if total_tracked_revenue > 0:
        if top_product_revenue is not None:
            metrics["top_product_revenue_share_pct"] = round(top_product_revenue / total_tracked_revenue * 100, 4)
        metrics["top3_product_revenue_share_pct"] = round(bread_top3_revenue / total_tracked_revenue * 100, 4)
        metrics["bread_revenue_share_pct"] = round(bread_revenue / total_tracked_revenue * 100, 4)
        metrics["beverage_revenue_share_pct"] = round(beverage_revenue / total_tracked_revenue * 100, 4)
    return metrics


def _attach_promotion_product_mix_metrics(output: AgentOutput, raw: Any) -> AgentOutput:
    metrics = {**output.metrics, **_product_mix_metrics(raw)}
    evidence_items = list(output.evidence_items)
    descriptions = {
        "top_product_revenue_share_pct": "Top bread product revenue as a share of total tracked product revenue",
        "top3_product_revenue_share_pct": "Top three bread products revenue as a share of total tracked product revenue",
        "top3_bread_revenue_share_pct": "Top three bread products revenue as a share of bread revenue",
        "bread_revenue_share_pct": "Bread revenue as a share of total tracked product revenue",
        "beverage_revenue_share_pct": "Beverage revenue as a share of total tracked product revenue",
        "top_product_name": "Top bread product name",
        "top_product_units": "Top bread product units sold",
        "bread_revenue": "Bread revenue",
        "beverage_revenue": "Beverage revenue",
        "sold_bread_sku_count": "Number of bread SKUs sold",
        "total_bread_sku": "Number of bread SKUs tracked",
        "top_beverage_name": "Top beverage product name",
        "top_beverage_units": "Top beverage units sold",
    }
    for metric_id, description in descriptions.items():
        if metric_id in metrics:
            evidence_items.append(
                EvidenceItem(
                    id=metric_id,
                    source="promotion_product_mix",
                    description=description,
                    value=metrics[metric_id],
                )
            )
    risks = list(output.risks)
    bread_share = _optional_float_value(metrics.get("bread_revenue_share_pct"))
    bread_top3_share = _optional_float_value(metrics.get("top3_bread_revenue_share_pct"))
    total_top3_share = _optional_float_value(metrics.get("top3_product_revenue_share_pct"))
    if (
        (bread_share is not None and bread_top3_share is not None and bread_share >= 60 and bread_top3_share >= 50)
        or (total_top3_share is not None and total_top3_share >= 50)
    ):
        risks.append("product_concentration")
    return output.model_copy(
        update={
            "metrics": metrics,
            "evidence_items": evidence_items,
            "risks": list(dict.fromkeys(risks)),
        }
    )


def _promotion_discount_rate_pct(promo: Any) -> float | None:
    if not promo:
        return None
    metrics = promo.metrics if isinstance(promo.metrics, dict) else {}
    discount_rate = _optional_float_value(metrics.get("discount_rate_pct"))
    if discount_rate is not None:
        return discount_rate
    return None


def _product_mix_top3_share_pct(product_mix: Any) -> float | None:
    if not product_mix:
        return None
    metrics = product_mix.metrics
    return _optional_float_value(metrics.get("top3_product_revenue_share_pct"))


def _promotion_loss_context(promo: Any) -> dict[str, Any]:
    metrics = promo.metrics if promo and isinstance(promo.metrics, dict) else {}
    expired_cost = _float_value(metrics.get("expired_cost"))
    expired_cost_revenue_pct = _float_value(
        metrics.get("expired_cost_revenue_pct")
    )
    products = metrics.get("expired_products", [])
    if not isinstance(products, list):
        products = []
    products = [item for item in products if isinstance(item, dict)]
    top_product = products[0] if products else {}
    is_material = (
        expired_cost_revenue_pct
        >= THRESHOLDS["profit_expired_cost_alert_pct"]
    )
    is_targetable = bool(
        is_material
        and top_product
        and _float_value(top_product.get("loss_share_pct"))
        >= THRESHOLDS["promotion_loss_concentration_pct"]
        and _float_value(top_product.get("margin_pct"))
        >= THRESHOLDS["promotion_target_margin_floor_pct"]
        and _float_value(top_product.get("sell_through_pct"))
        >= THRESHOLDS["promotion_target_sell_through_floor_pct"]
    )
    return {
        "expired_cost": expired_cost,
        "expired_cost_revenue_pct": expired_cost_revenue_pct,
        "products": products,
        "top_product": top_product,
        "is_material": is_material,
        "is_targetable": is_targetable,
    }


def _promotion_basket_context(promo: Any) -> dict[str, Any]:
    metrics = promo.metrics if promo and isinstance(promo.metrics, dict) else {}
    items_per_order = _optional_float_value(metrics.get("items_per_order"))
    items_change_pct = _optional_float_value(
        metrics.get("items_per_order_change_pct")
    )
    revenue_per_item = _optional_float_value(metrics.get("revenue_per_item"))
    revenue_per_item_change_pct = _optional_float_value(
        metrics.get("revenue_per_item_change_pct")
    )
    is_weak = bool(
        items_change_pct is not None
        and items_change_pct <= -10
        and revenue_per_item_change_pct is not None
        and revenue_per_item_change_pct >= -5
    )
    return {
        "items_per_order": items_per_order,
        "items_change_pct": items_change_pct,
        "revenue_per_item": revenue_per_item,
        "revenue_per_item_change_pct": revenue_per_item_change_pct,
        "is_weak": is_weak,
    }


def _synthesize_promotion_mix_summary(outputs: dict[str, Any]) -> str:
    promo = outputs.get("promotion_signal")
    product_mix = outputs.get("promotion_product_mix")
    loss_context = _promotion_loss_context(promo)
    basket_context = _promotion_basket_context(promo)
    sentences = []
    if promo:
        discount_rate = _promotion_discount_rate_pct(promo)
        if discount_rate is None:
            sentences.append("Discount exposure is not available in the promotion signal, so promotion decisions should not assume that pricing pressure is controlled.")
        elif discount_rate <= 5:
            sentences.append(f"Discount exposure is controlled at {discount_rate:.1f}%, so the current evidence does not justify a broad price cut.")
        else:
            sentences.append(f"Discount exposure is elevated at {discount_rate:.1f}%, so promotion impact should be checked before repeating the same campaign.")
        if basket_context["is_weak"]:
            revenue_per_item_change_pct = _float_value(
                basket_context["revenue_per_item_change_pct"]
            )
            if abs(revenue_per_item_change_pct) <= 5:
                item_value_text = (
                    f"revenue per item remained stable at "
                    f"{_money_precise(basket_context['revenue_per_item'])} "
                    f"({revenue_per_item_change_pct:+.1f}%)"
                )
            else:
                item_value_text = (
                    f"revenue per item moved {revenue_per_item_change_pct:+.1f}% to "
                    f"{_money_precise(basket_context['revenue_per_item'])}"
                )
            sentences.append(
                f"Items per order fell {abs(_float_value(basket_context['items_change_pct'])):.1f}% "
                f"to {_float_value(basket_context['items_per_order']):.2f}, while {item_value_text}. "
                "This points to a basket-size weakness rather than lower item value."
            )
        if loss_context["is_material"]:
            sentences.append(
                f"Products discarded at closing cost {_money_precise(loss_context['expired_cost'])}, "
                f"equal to {loss_context['expired_cost_revenue_pct']:.1f}% of revenue."
            )
            top_product = loss_context["top_product"]
            if top_product:
                top_name = str(top_product.get("name") or "the leading loss item")
                sentences.append(
                    f"{top_name} caused the largest recorded closing loss at "
                    f"{_money_precise(top_product.get('expired_cost'))}, with "
                    f"{_float_value(top_product.get('margin_pct')):.1f}% sold-product margin "
                    f"and {_float_value(top_product.get('sell_through_pct')):.1f}% sell-through."
                )
    if sentences and product_mix:
        sentences[-1] += "\n\n"
    if product_mix:
        metrics = product_mix.metrics
        bread_revenue = _float_value(metrics.get("bread_revenue"))
        beverage_revenue = _float_value(metrics.get("beverage_revenue"))
        beverage_share = _float_value(metrics.get("beverage_revenue_share_pct"))
        top_product = str(metrics.get("top_product_name") or "").replace("_", " ")
        top_product_units = _int_value(metrics.get("top_product_units"))
        top3_bread_share = _float_value(metrics.get("top3_bread_revenue_share_pct"))
        top3_total_share = _float_value(metrics.get("top3_product_revenue_share_pct"))
        sold_skus = _int_value(metrics.get("sold_bread_sku_count"))
        total_skus = _int_value(metrics.get("total_bread_sku"))
        top_beverage = str(metrics.get("top_beverage_name") or "").replace("_", " ")
        top_beverage_units = _int_value(metrics.get("top_beverage_units"))
        if bread_revenue or beverage_revenue:
            sentences.append(
                f"Bread generated {_money(bread_revenue)} and beverages generated {_money(beverage_revenue)}, with beverages contributing {beverage_share:.1f}% of tracked product revenue."
            )
        if top_product:
            if sold_skus and total_skus:
                sentences.append(
                    f"{sold_skus} of {total_skus} tracked bread SKUs recorded sales; "
                    f"{top_product} led the mix with {_number(top_product_units)} units sold."
                )
            else:
                sentences.append(
                    f"Within bread, {top_product} led the mix with {_number(top_product_units)} units sold."
                )
        if top3_bread_share:
            if "product_concentration" in product_mix.risks:
                sentences.append(
                    f"Bread revenue is concentrated: the top three bread products account for {top3_bread_share:.1f}% of bread revenue and {top3_total_share:.1f}% of total tracked product revenue."
                )
            else:
                sentences.append(
                    f"Bread product mix is distributed: the top three bread products account for {top3_bread_share:.1f}% of bread revenue and {top3_total_share:.1f}% of total tracked product revenue."
                )
        if top_beverage:
            sentences.append(
                f"The beverage side is led by {top_beverage} at {_number(top_beverage_units)} units, so beverage pairing can support targeted promotions without discounting the full bread range."
            )
    if not sentences:
        return "Promotion and product-mix analysis could not be completed because supporting dashboard data is missing."
    sentences[-1] += "\n\n"
    if loss_context["is_material"] and loss_context["is_targetable"]:
        sentences.append(
            "The practical decision is to keep broad discounts off the table and test a targeted late-day bundle for the leading closing-loss product."
        )
    elif loss_context["is_material"]:
        sentences.append(
            "The practical decision is to reduce or stage production for the affected products because the combined product evidence does not show that discounting would recover more value."
        )
    elif basket_context["is_weak"]:
        sentences.append(
            "The practical decision is to keep broad discounts off the table and run a narrow bread-and-beverage bundle test to rebuild basket size."
        )
    elif product_mix and "product_concentration" in product_mix.risks:
        sentences.append(
            "The practical decision is to keep broad discounts evidence-led while using product-level actions to reduce dependence on a small group of bread items."
        )
    else:
        sentences.append(
            "The practical decision is to keep broad discounts evidence-led and use targeted product-level tests only when traffic, margin, or closing-loss evidence justifies them."
        )
    return " ".join(sentences).replace("\n\n ", "\n\n")


def _promotion_reduction_rationale(top_product: dict[str, Any]) -> str:
    top_name = str(top_product.get("name") or "The leading closing-loss product")
    margin_pct = _float_value(top_product.get("margin_pct"))
    sell_through_pct = _float_value(top_product.get("sell_through_pct"))
    loss_share_pct = _float_value(top_product.get("loss_share_pct"))
    reasons = []

    if sell_through_pct < THRESHOLDS["promotion_target_sell_through_floor_pct"]:
        reasons.append(
            f"its {sell_through_pct:.1f}% sell-through does not show reliable demand for a targeted offer"
        )
    if loss_share_pct < THRESHOLDS["promotion_loss_concentration_pct"]:
        reasons.append(
            f"its {loss_share_pct:.1f}% share of closing loss is not concentrated enough to justify a product-specific discount"
        )

    if margin_pct >= THRESHOLDS["promotion_target_margin_floor_pct"]:
        opening = f"{top_name} retains {margin_pct:.1f}% sold-product margin"
        detail = f", but {_natural_names(reasons)}" if reasons else ""
    else:
        opening = (
            f"{top_name}'s {margin_pct:.1f}% sold-product margin leaves limited room "
            "for discounting"
        )
        detail = f", and {_natural_names(reasons)}" if reasons else ""

    return (
        f"{opening}{detail}. The available evidence does not show that discounting "
        "would recover more value than reducing or staging production."
    )


def _promotion_mix_recommendations(outputs: dict[str, Any]) -> list[Any]:
    recommendations = []
    promo = outputs.get("promotion_signal")
    product_mix = outputs.get("promotion_product_mix")
    discount_rate = _promotion_discount_rate_pct(promo)
    top3_share = _product_mix_top3_share_pct(product_mix)
    product_metrics = product_mix.metrics if product_mix else {}
    bread_share = _optional_float_value(product_metrics.get("bread_revenue_share_pct"))
    bread_top3_share = _optional_float_value(product_metrics.get("top3_bread_revenue_share_pct"))
    bread_concentration = (
        bread_share is not None
        and bread_top3_share is not None
        and bread_share >= 60
        and bread_top3_share >= 50
    )
    loss_context = _promotion_loss_context(promo)
    basket_context = _promotion_basket_context(promo)

    if loss_context["is_material"]:
        top_product = loss_context["top_product"]
        top_name = str(top_product.get("name") or "the affected products")
        evidence_ids = ["expired_cost", "expired_cost_revenue_pct"]
        if loss_context["products"]:
            evidence_ids.append("expired_products")
        if loss_context["is_targetable"]:
            recommendations.append(
                Recommendation(
                    id="promotion_targeted_closing_bundle",
                    action=(
                        f"Test a targeted late-day bundle for {top_name} before changing "
                        "menu-wide prices."
                    ),
                    urgency="high",
                    time_horizon="this_week",
                    rationale=(
                        f"{top_name} accounts for "
                        f"{_float_value(top_product.get('loss_share_pct')):.1f}% of closing loss, "
                        f"while sold units retain {_float_value(top_product.get('margin_pct')):.1f}% "
                        f"margin and {_float_value(top_product.get('sell_through_pct')):.1f}% sell-through."
                    ),
                    expected_impact=(
                        "Tests whether a narrow late-day offer can improve sell-through "
                        "without broad margin erosion."
                    ),
                    evidence_ids=evidence_ids,
                )
            )
        else:
            affected_names = _natural_names(
                [str(item.get("name") or "") for item in loss_context["products"][:3]]
            )
            action_target = affected_names or "the affected products"
            recommendations.append(
                Recommendation(
                    id="promotion_reduce_unsold_bake",
                    action=(
                        f"Reduce or stage the next bake for {action_target} before considering "
                        "a promotion."
                    ),
                    urgency="high",
                    time_horizon="this_week",
                    rationale=_promotion_reduction_rationale(top_product),
                    expected_impact=(
                        "Addresses excess production directly while avoiding discounts that "
                        "may not recover weak demand."
                    ),
                    evidence_ids=evidence_ids,
                )
            )

    if basket_context["is_weak"] and not loss_context["is_material"]:
        recommendations.append(
            Recommendation(
                id="promotion_basket_bundle_test",
                action=(
                    "Test one targeted bread-and-beverage bundle for 2-3 completed trading days, "
                    "then keep it only if items per order improves; do not use a broad discount."
                ),
                urgency="medium",
                time_horizon="this_week",
                rationale=(
                    f"Items per order fell {abs(_float_value(basket_context['items_change_pct'])):.1f}% to "
                    f"{_float_value(basket_context['items_per_order']):.2f}, while revenue per item moved "
                    f"{_float_value(basket_context['revenue_per_item_change_pct']):+.1f}%, so the evidence supports "
                    "a basket-size test rather than a menu-wide price cut."
                ),
                expected_impact=(
                    "Tests whether a narrow pairing can restore basket size while keeping discount exposure controlled."
                ),
                evidence_ids=[
                    "items_per_order",
                    "items_per_order_change_pct",
                    "revenue_per_item",
                    "revenue_per_item_change_pct",
                    "beverage_revenue_share_pct",
                ],
            )
        )

    if discount_rate is not None and discount_rate <= 5:
        no_broad_discount_action = (
            "Keep broad discounts off the table while the targeted basket test runs."
            if basket_context["is_weak"]
            else "Do not launch a broad discount unless traffic weakness persists."
        )
        recommendations.append(
            Recommendation(
                id="promotion_no_broad_discount",
                action=no_broad_discount_action,
                urgency="low",
                time_horizon="ongoing",
                rationale="Discount exposure is controlled, so a broad price cut is not justified by the current evidence.",
                expected_impact="Protects margin while keeping promotion decisions evidence-led.",
                evidence_ids=["discount_rate_pct"],
            )
        )
    if (top3_share is not None and top3_share >= 50) or bread_concentration:
        evidence_ids = ["top3_product_revenue_share_pct"]
        if bread_concentration:
            evidence_ids = ["top3_bread_revenue_share_pct", "bread_revenue_share_pct"]
        recommendations.append(
            Recommendation(
                id="promotion_mid_tier_bundle",
                action="Use targeted bundles or small rotation promotions to support mid-tier bread items instead of discounting the full menu.",
                urgency="medium",
                time_horizon="this_week",
                rationale="Revenue is concentrated in the leading bread products, so targeted bundles are an opportunity to support mid-tier items.",
                expected_impact="Improves product-mix balance while protecting margin from broad discount erosion.",
                evidence_ids=evidence_ids,
            )
        )
    return recommendations


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural or f"{singular}s"


def _order_quantities_text(quantities: Any) -> str:
    if not isinstance(quantities, dict):
        return ""
    parts = []
    for unit, raw_quantity in sorted(quantities.items()):
        quantity = _float_value(raw_quantity)
        formatted = f"{quantity:.0f}" if unit == "pcs" else f"{quantity:.2f}".rstrip("0").rstrip(".")
        parts.append(f"{formatted} {unit}")
    return _natural_names(parts)


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
    order_behavior = outputs.get("order_behavior")
    hourly_revenue = outputs.get("hourly_revenue")
    profit_metrics = profit.metrics if profit else {}
    trend_metrics = trend.metrics if trend else {}
    order_metrics = order_behavior.metrics if order_behavior else {}
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
            if (
                recommendation.id == "revenue_decline_review"
                and order_metrics.get("order_volume_driver")
                in {"volume-and-basket-contraction", "basket-contraction"}
            ):
                recommendation = recommendation.model_copy(
                    update={
                        "action": (
                            "Review basket composition and product availability for the affected day. "
                            "Use targeted bundles to rebuild items per order only after confirming that key products "
                            "were available; do not use a broad discount."
                        ),
                        "rationale": (
                            f"Items per order moved {_float_value(order_metrics.get('items_per_order_change_pct')):+.1f}%, "
                            f"while revenue per item moved {_float_value(order_metrics.get('revenue_per_item_change_pct')):+.1f}%, "
                            "so smaller baskets are a stronger explanation than lower item value."
                        ),
                        "expected_impact": (
                            "Targets the main basket-size weakness while checking availability before changing "
                            "pricing or production."
                        ),
                        "evidence_ids": [
                            "revenue_trend_pct",
                            "order_change_pct",
                            "average_order_value_change_pct",
                            "items_per_order",
                            "revenue_per_item",
                        ],
                    }
                )
            if recommendation.id == "peak_profit_window_protection":
                peak_hour = str(hourly_metrics.get("peak_profit_hour") or "")
                recommendation = recommendation.model_copy(
                    update={
                        "action": (
                            f"Before {peak_hour}, confirm stock for the products selling in that hour, "
                            "beverage pairing availability, and service coverage for the peak profit window."
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


def _synthesize_inventory_summary(outputs: dict[str, Any]) -> str:
    inventory = outputs.get("inventory")
    if not inventory:
        return "No finished-product inventory analysis was produced."

    metrics = inventory.metrics
    total_units = _int_value(metrics.get("inventory"))
    fresh_units = _int_value(metrics.get("fresh"))
    day1_units = _int_value(metrics.get("day1_available"))
    product_count = _int_value(metrics.get("product_count"))
    zero_count = _int_value(metrics.get("zero_stock_product_count"))
    zero_products = metrics.get("zero_stock_products", [])
    low_count = _int_value(metrics.get("low_stock_product_count"))
    thin_count = _int_value(metrics.get("thin_stock_product_count"))
    thin_share_pct = _float_value(metrics.get("thin_stock_product_share_pct"))
    units_per_product = _float_value(metrics.get("units_per_product"))
    day1_count = _int_value(metrics.get("day1_product_count"))
    day1_products = metrics.get("day1_products", [])
    raw_material_count = _int_value(metrics.get("raw_material_count"))
    low_materials = metrics.get("low_stock_materials", []) or []
    critical_materials = metrics.get("critical_materials", []) or []
    snapshot_date = str(metrics.get("snapshot_date") or "").strip()
    snapshot_basis = str(metrics.get("snapshot_basis") or "")
    flow_record_count = _int_value(metrics.get("flow_record_count"))
    flow_opening_units = _int_value(metrics.get("flow_opening_units"))
    flow_baked_units = _int_value(metrics.get("flow_baked_units"))
    flow_available_units = _int_value(metrics.get("flow_available_units"))
    flow_sold_units = _int_value(metrics.get("flow_sold_units"))
    flow_discarded_units = _int_value(metrics.get("flow_discarded_units"))
    flow_left_units = _int_value(metrics.get("flow_left_units"))
    flow_sell_through_pct = _float_value(metrics.get("flow_sell_through_pct"))
    flow_balance_issue_count = _int_value(metrics.get("flow_balance_issue_count"))
    flow_date = str(metrics.get("flow_date") or snapshot_date).strip()
    high_sell_through_products = metrics.get("high_sell_through_products", []) or []
    slow_moving_products = metrics.get("slow_moving_products", []) or []
    overdue_stock_total = _int_value(metrics.get("overdue_stock_total"))
    overdue_stock_products = metrics.get("overdue_stock_products", []) or []

    paragraphs = []
    if product_count == 0:
        paragraphs.append(
            "No finished-product inventory records are available for the selected scope. "
            "This is an inventory data gap, so stockout or clearance decisions should wait until batch records or the inventory feed are verified."
        )
    elif product_count and total_units == 0:
        opening = (
            f"No finished-product stock is recorded for the selected scope: {product_count} products show 0 units, with 0 fresh units and 0 day-1 units."
        )
        if zero_products:
            opening += f" The zero-stock products include {_natural_names(zero_products)}."
        paragraphs.append(opening)
        paragraphs.append(
            "This should be treated as a stockout risk or inventory sync gap until batch records are verified, not as proof that every shelf is physically empty."
        )
    else:
        if snapshot_basis == "selected_date_dashboard" and snapshot_date:
            opening = (
                f"For {snapshot_date}, finished-product stock totals {total_units} {_plural(total_units, 'unit')} across {product_count} tracked products, "
                f"averaging {units_per_product:.2f} units per product."
            )
        else:
            opening = (
                f"Current finished-product stock totals {total_units} {_plural(total_units, 'unit')} across {product_count} tracked products, "
                f"averaging {units_per_product:.2f} units per product."
            )
        if thin_count:
            opening += (
                f" Inventory is broadly thin: {thin_count} of {product_count} products "
                f"({thin_share_pct:.1f}%) have zero or one unit remaining."
            )
        paragraphs.append(opening)

        stock_risk = []
        if zero_count:
            stock_risk.append(
                f"{zero_count} products have no recorded finished stock, including {_natural_names(zero_products)}."
            )
        if low_count:
            product_word = "product has" if low_count == 1 else "products have"
            unit_word = "unit" if low_count == 1 else "unit each"
            additional = "additional " if zero_count else ""
            stock_risk.append(
                f"{low_count} {additional}{product_word} only 1 {unit_word}."
            )
        if stock_risk:
            stock_risk.append(
                "These figures should guide the next production review, but a point-in-time snapshot does not by itself prove that sales were lost."
            )
            paragraphs.append(" ".join(stock_risk))

        operations = [
            f"The snapshot contains {fresh_units} fresh {_plural(fresh_units, 'unit')} and "
            f"{day1_units} day-1 {_plural(day1_units, 'unit')}."
        ]
        if day1_count:
            operations.append(
                f"{day1_count} products carry day-1 stock, including {_natural_names(day1_products)}, so clearance risk should be checked before adding more bake volume."
            )
        elif product_count:
            operations.append(
                "No day-1 stock is recorded, so there is no immediate finished-product expiry pressure in this snapshot."
            )

        if raw_material_count:
            operations.append(
                f"Raw-material status covers the {raw_material_count} materials shown in the Inventory dashboard "
                "and uses current stock and configured reorder points."
            )
            critical_set = set(critical_materials)
            reorder_only = [name for name in low_materials if name not in critical_set]
            if critical_materials:
                verb = "is" if len(critical_materials) == 1 else "are"
                operations.append(f"{_natural_names(critical_materials)} {verb} out of stock.")
            if reorder_only:
                verb = "is" if len(reorder_only) == 1 else "are"
                operations.append(f"{_natural_names(reorder_only)} {verb} at or below its reorder point." if len(reorder_only) == 1 else f"{_natural_names(reorder_only)} {verb} at or below their reorder points.")
            if not low_materials:
                operations.append(
                    f"All {raw_material_count} dashboard materials are above their configured reorder points."
                )
        paragraphs.append(" ".join(operations))

    if overdue_stock_total:
        product_context = (
            f" for {_natural_names(overdue_stock_products)}"
            if overdue_stock_products
            else ""
        )
        unit_word = "unit" if overdue_stock_total == 1 else "units"
        paragraphs.append(
            f"Separately, {overdue_stock_total} expired {unit_word} remain pending disposal{product_context}. "
            "They are excluded from sellable stock and should be matched to disposal outflow records before the inventory review is closed."
        )

    if flow_record_count:
        flow_scope = f" for {flow_date}" if flow_date else ""
        movement = (
            f"Baked-product movement{flow_scope} covers {flow_record_count} recorded batch"
            f"{'es' if flow_record_count != 1 else ''}: {flow_opening_units} {_plural(flow_opening_units, 'unit')} "
            f"{'were' if flow_opening_units != 1 else 'was'} available at opening, "
            f"{flow_baked_units} {_plural(flow_baked_units, 'unit')} {'were' if flow_baked_units != 1 else 'was'} baked during the day, "
            f"giving {flow_available_units} {_plural(flow_available_units, 'unit')} available in total; "
            f"{flow_sold_units} {'were' if flow_sold_units != 1 else 'was'} sold, {flow_discarded_units} "
            f"{'were' if flow_discarded_units != 1 else 'was'} discarded, and {flow_left_units} "
            f"{'remain' if flow_left_units != 1 else 'remains'} in those batches. "
            f"That is {flow_sell_through_pct:.1f}% sell-through."
        )
        flow_findings = []
        if high_sell_through_products:
            names = _natural_names(high_sell_through_products)
            if len(high_sell_through_products) == 1:
                flow_findings.append(
                    f"{names} is close to selling through and should be checked first in the next production review."
                )
            else:
                flow_findings.append(
                    f"{len(high_sell_through_products)} products have high sell-through and no more than one unit left, "
                    f"led by {names}. They should be checked first in the next production review."
                )
        if slow_moving_products:
            names = _natural_names(slow_moving_products)
            verb = "have" if len(slow_moving_products) != 1 else "has"
            flow_findings.append(
                f"{names} {verb} comparatively slow movement and should be staged or reduced unless forecast demand supports the remaining stock."
            )
        if flow_balance_issue_count:
            record_word = "record does" if flow_balance_issue_count == 1 else "records do"
            flow_findings.append(
                f"{flow_balance_issue_count} baked-product flow {record_word} not reconcile, so its transaction trail should be verified before acting."
            )
        flow_findings.append(
            "This movement evidence explains how stock changed during the selected day, but it does not replace the demand forecast."
        )
        paragraphs.append(" ".join([movement, *flow_findings]))

    return "\n\n".join(paragraphs)


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
    additional_wasted_materials = waste_metrics.get("additional_wasted_materials", [])
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
        date_prefix = f"For {requested_date}, " if requested_date else ""
        opening = (
            f"{date_prefix}recorded material waste is limited in cost, but it should still be checked. "
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
        if isinstance(additional_wasted_materials, list) and additional_wasted_materials:
            extras = [
                item
                for item in additional_wasted_materials
                if isinstance(item, dict) and item.get("name")
            ]
            if extras:
                extra_names = _natural_names([item.get("name") for item in extras])
                extra_costs = _natural_names(
                    [_money_precise(item.get("waste_cost")) for item in extras]
                )
                entry_word = "entry" if len(extras) == 1 else "entries"
                suffix = "" if len(extras) == 1 else ", respectively"
                sentences.append(
                    f"{extra_names} also logged lower-priority waste {entry_word} at {extra_costs}{suffix}."
                )
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
            f"Production records for this date show {total_units} baked units and recorded consumption for {material_rows} materials. That means waste records can be cross-checked against production activity."
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
    expired_cost = _float_value(profit_metrics.get("expired_cost"))
    expired_cost_revenue_pct = _float_value(
        profit_metrics.get("expired_cost_revenue_pct")
    )
    profit_before_expiry = _float_value(
        profit_metrics.get("profit_before_expiry")
    )
    return_cost = _float_value(profit_metrics.get("non_sellable_return_cost"))
    is_material_unsold_loss = bool(
        profit and "unsold_product_loss" in profit.risks
    )
    discount_rate_pct = _float_value(discount.metrics.get("discount_rate_pct")) if discount else 0.0
    trend_metrics = trend.metrics if trend else {}
    revenue_trend_pct = _float_value(trend_metrics.get("revenue_trend_pct"))
    order_change_pct = _float_value(trend_metrics.get("order_change_pct"))
    low_sample_collapse = orders <= 3 and (revenue_trend_pct <= -50 or order_change_pct <= -50)
    material_revenue_decline = revenue_trend_pct <= -10

    if low_sample_collapse:
        if margin_pct >= 20:
            opening = "This was not a healthy revenue day overall, even though margin remained strong"
        else:
            opening = "This was not a healthy revenue day overall"
    elif is_material_unsold_loss:
        opening = "This revenue day was profitable, but closing product loss needs attention"
    elif material_revenue_decline and margin_pct >= 20:
        opening = "The day remained profitable, but revenue performance weakened materially"
    elif material_revenue_decline:
        opening = "Revenue and margin performance need attention"
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
            f"with {_money(profit_value)} profit and a margin of {margin_pct:.1f}%, {margin_note}."
        )
    else:
        first_sentence = (
            f"{opening}: revenue reached {_money(revenue)} and profit reached {_money(profit_value)}, "
            f"with a profit margin of {margin_pct:.1f}% on {orders} {_plural(orders, 'order')} and average order value at "
            f"{chr(165)}{average_order_value:.2f}."
        )
    sentences = [
        first_sentence
    ]

    if trend:
        metrics = trend_metrics
        direction = metrics.get("trend_direction", "stable")
        if metrics.get("previous_day_available", True):
            revenue_change_pct = _float_value(
                metrics.get("dashboard_revenue_change_pct", revenue_trend_pct)
            )
            profit_change_pct = _float_value(
                metrics.get("dashboard_profit_change_pct")
            )
            aov_change_pct = _float_value(
                metrics.get("average_order_value_change_pct")
            )
            if revenue_change_pct > 0 and profit_change_pct < 0:
                sentences.append(
                    "Compared with the previous day, revenue increased by "
                    f"{revenue_change_pct:.1f}% while profit decreased by "
                    f"{abs(profit_change_pct):.1f}%; order count moved "
                    f"{order_change_pct:+.1f}%, and average order value moved "
                    f"{aov_change_pct:+.1f}%."
                )
            elif revenue_change_pct < 0 and profit_change_pct > 0:
                sentences.append(
                    "Compared with the previous day, revenue decreased by "
                    f"{abs(revenue_change_pct):.1f}% while profit increased by "
                    f"{profit_change_pct:.1f}%; order count moved "
                    f"{order_change_pct:+.1f}%, and average order value moved "
                    f"{aov_change_pct:+.1f}%."
                )
            else:
                sentences.append(
                    f"Revenue dashboard comparison against the previous day is {direction}: total revenue moved "
                    f"{revenue_change_pct:+.1f}% vs yesterday, profit moved "
                    f"{profit_change_pct:+.1f}%, order count moved "
                    f"{order_change_pct:+.1f}%, and average order value moved "
                    f"{aov_change_pct:+.1f}%."
                )
        else:
            previous_date = str(
                metrics.get("previous_day_date") or "the previous day"
            )
            sentences.append(
                "Previous-day comparison is unavailable because no completed sales were "
                f"recorded on {previous_date}; recent completed-day baselines are used below."
            )

    if benchmark:
        metrics = benchmark.metrics
        baseline_day_count = _int_value(metrics.get("baseline_day_count"))
        sentences.append(
            f"The recent baseline check adds context: dashboard total revenue is "
            f"{_float_value(metrics.get('revenue_vs_recent_avg_pct')):+.1f}% against the previous "
            f"{baseline_day_count} completed trading-day average."
        )

    if sentences and any(
        (order_behavior, product_mix, category_mix, hourly_revenue, discount)
    ):
        sentences[-1] += "\n\n"

    if order_behavior:
        metrics = order_behavior.metrics
        driver = str(metrics.get("order_volume_driver") or "stable")
        comparison_basis = str(
            metrics.get("comparison_basis") or "previous_day"
        )
        if driver == "volume-and-basket-contraction":
            quality_prefix = "Revenue weakened through both fewer orders and smaller baskets"
            quality_note = "so both order volume and basket size contributed to the decline"
        elif driver == "volume-contraction":
            quality_prefix = "Revenue weakened mainly through fewer orders"
            quality_note = "so order volume was the main pressure while basket value was comparatively stable"
        elif driver == "basket-contraction":
            quality_prefix = "Revenue weakened mainly through smaller baskets"
            quality_note = "so basket value was the main pressure while order volume was comparatively stable"
        else:
            quality_prefix = (
                "Revenue quality against the recent baseline is"
                if comparison_basis == "recent_baseline"
                else "Revenue quality is"
            )
            quality_prefix = f"{quality_prefix} {driver}"
            if driver == "basket-led":
                quality_note = "so the day is being carried by larger baskets rather than higher order volume"
            elif driver == "volume-led":
                quality_note = "so the day is being carried mainly by higher order volume rather than basket expansion"
            elif driver == "volume-and-basket-led":
                quality_note = "so revenue quality is supported by both order volume and basket value"
            else:
                quality_note = "so the day should be read through both order volume and basket value"
        sentences.append(
            f"{quality_prefix}: orders moved "
            f"{_float_value(metrics.get('order_change_pct')):+.1f}% while average order value moved "
            f"{_float_value(metrics.get('average_order_value_change_pct')):+.1f}%, {quality_note}."
        )
        items_per_order = metrics.get("items_per_order")
        previous_items_per_order = metrics.get("previous_items_per_order")
        revenue_per_item = metrics.get("revenue_per_item")
        revenue_per_item_change_pct = metrics.get("revenue_per_item_change_pct")
        if (
            driver in {"volume-and-basket-contraction", "basket-contraction"}
            and items_per_order is not None
            and previous_items_per_order is not None
            and revenue_per_item is not None
            and revenue_per_item_change_pct is not None
        ):
            sentences.append(
                f"Items per order fell from {_float_value(previous_items_per_order):.2f} to "
                f"{_float_value(items_per_order):.2f}, while revenue per item remained broadly stable at "
                f"{chr(165)}{_float_value(revenue_per_item):.2f} "
                f"({_float_value(revenue_per_item_change_pct):+.1f}%); this indicates that smaller baskets, "
                "rather than lower item value, were the main pressure on average order value."
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
            "The first decision should be to verify data completeness or abnormal trading conditions before changing pricing, production, or promotion plans."
        )
    else:
        if not discount:
            sentences.append("No discount erosion is visible in the revenue data.")
    if sentences:
        sentences[-1] += "\n\n"
    if expired_cost > 0:
        if revenue > 0:
            sentences.append(
                f"Unsold products discarded at closing cost {chr(165)}{expired_cost:.2f}, equal to "
                f"{expired_cost_revenue_pct:.1f}% of revenue, and reduced the reported margin "
                f"from {profit_before_expiry / revenue * 100:.1f}% before this loss to "
                f"{margin_pct:.1f}% after it."
            )
        else:
            sentences.append(
                f"Unsold products discarded at closing cost {chr(165)}{expired_cost:.2f} and are already included in reported profit."
            )
    if return_cost > 0:
        sentences.append(
            f"Non-sellable return cost of {chr(165)}{return_cost:.2f} is already included in reported profit."
        )
    if expired_cost <= 0 and return_cost <= 0:
        sentences.append(
            "No expired-stock or non-sellable return cost was recorded."
        )
    sentences.append(
        "Separately recorded material-wastage variance is not deducted again."
    )
    return " ".join(sentences).replace("\n\n ", "\n\n")


def _forecast_data_available(overview: Any) -> bool:
    return bool(
        overview
        and overview.data_quality.freshness != "missing"
        and overview.data_quality.completeness > 0
    )


def _forecast_recommendations(outputs: dict[str, Any]) -> list[Any]:
    overview = outputs.get("forecast_overview")
    if not _forecast_data_available(overview):
        return [
            Recommendation(
                id="forecast_data_check",
                action="Verify the Forecast BI feed for the selected seven-day horizon, then rerun the production analysis.",
                urgency="high",
                time_horizon="today",
                rationale="Forecast demand is unavailable, so production quantities cannot be reconciled safely.",
                expected_impact="Prevents missing forecast data from being treated as zero demand.",
                evidence_ids=["forecast_total_units"],
            )
        ]
    production = outputs.get("production")
    accuracy = outputs.get("forecast_accuracy")
    forecast_units = _int_value(overview.metrics.get("forecast_bakery_units")) if overview else 0
    total_bake = _int_value(production.metrics.get("total_bake")) if production else 0
    wape = float(accuracy.metrics.get("forecast_wape", 0.0) or 0.0) if accuracy else 0.0
    planning_basis = min(total_bake, forecast_units) if total_bake and forecast_units else 0
    base_units = int(planning_basis * 0.85) if planning_basis else 0
    top_products = overview.metrics.get("top_bakery_products", []) if overview else []
    top_driver = _names(top_products[:1]) if isinstance(top_products, list) else ""

    recommendations = []
    day1_date = str(overview.metrics.get("forecast_day1_date", "") or "") if overview else ""
    day1_plan_date = str(production.metrics.get("production_day1_date", "") or "") if production else ""
    day1_forecast_units = _int_value(overview.metrics.get("forecast_day1_bakery_units")) if overview else 0
    day1_beverage_units = _int_value(overview.metrics.get("forecast_day1_beverage_units")) if overview else 0
    day1_plan_units = _int_value(production.metrics.get("production_day1_bake")) if production else 0
    day1_bakery_products = overview.metrics.get("forecast_day1_top_bakery_products", []) if overview else []
    day1_beverage_products = overview.metrics.get("forecast_day1_top_beverage_products", []) if overview else []
    if (
        day1_date
        and day1_date == day1_plan_date
        and day1_forecast_units > 0
        and day1_plan_units > 0
    ):
        execution_clauses = []
        bakery_priorities = _natural_names(day1_bakery_products[:3])
        beverage_priorities = _natural_names(day1_beverage_products[:3])
        if bakery_priorities:
            execution_clauses.append(f"Prioritize {bakery_priorities}")
        if day1_beverage_units > 0:
            beverage_clause = (
                f"prepare ingredients and service capacity for {day1_beverage_units} "
                "made-to-order beverage units"
            )
            if beverage_priorities:
                beverage_clause += f" led by {beverage_priorities}"
            execution_clauses.append(beverage_clause)
        execution_clauses.append(
            f"approve production above the {day1_plan_units}-unit plan only if early sales "
            "run ahead of the day's forecast pace"
        )
        if len(execution_clauses) == 1:
            execution_text = execution_clauses[0]
        else:
            execution_text = (
                ", ".join(execution_clauses[:-1])
                + ", and "
                + execution_clauses[-1]
            )
        recommendations.append(
            Recommendation(
                id="selected_day_production",
                action=(
                    f"For {day1_date}, use the planned bake of {day1_plan_units} bakery units "
                    f"against {day1_forecast_units} forecast bakery units. "
                    f"{execution_text}."
                ),
                urgency="high",
                time_horizon="today",
                rationale=(
                    "This operating advice uses the selected date's forecast and production plan "
                    "rather than a weekly average."
                ),
                expected_impact=(
                    "Turns the weekly strategy into a first-day operating checkpoint without "
                    "treating made-to-order beverages as pre-produced stock."
                ),
                evidence_ids=[
                    "forecast_day1_bakery_units",
                    "forecast_day1_beverage_units",
                    "production_day1_bake",
                ],
            )
        )
    if forecast_units and base_units and wape > 0:
        error_rate = wape / 100.0
        lower_guardrail = max(0, math.floor(forecast_units * (1.0 - error_rate)))
        upper_guardrail = math.ceil(forecast_units * (1.0 + error_rate))
        units_to_expected = max(forecast_units - base_units, 0)
        units_above_expected = max(upper_guardrail - forecast_units, 0)
        priority_sentence = (
            f" Prioritize the top forecast driver first ({top_driver}) when releasing extra bake."
            if top_driver
            else " Prioritize the top forecast driver first when releasing extra bake."
        )
        recommendations.append(
            Recommendation(
                id="historical_error_guardrail",
                action=(
                    f"Across the seven-day horizon, start with an 85% base bake of {base_units} units "
                    "against expected bakery demand of "
                    f"{forecast_units} units. Treat {lower_guardrail}-{upper_guardrail} bakery units as a "
                    f"historical-error operating guardrail, not a prediction interval. Keep the next "
                    f"{units_to_expected} units up to expected demand flexible, and release no more than "
                    f"{units_above_expected} units above expected demand only if the first 1-2 trading days "
                    f"track close to forecast.{priority_sentence}"
                ),
                urgency="high",
                time_horizon="this_week",
                rationale=(
                    f"Held-out historical error is {wape:.1f}%, so the operating guardrail is derived "
                    "from expected bakery demand instead of summed product-level Q10/Q90 bounds."
                ),
                expected_impact=(
                    f"Keeps the initial bake conservative while limiting above-expected release to "
                    f"{units_above_expected} units and requiring early-sales evidence."
                ),
                evidence_ids=[
                    "forecast_bakery_units",
                    "forecast_wape",
                    "production_total_bake",
                    "supply_coverage_pct",
                    "demand_gap_units",
                ],
            )
        )
    for output in outputs.values():
        for recommendation in output.recommendations:
            recommendations.append(recommendation)
    business_events = []
    if overview:
        business_events = [
            event for event in overview.metadata.get("business_events", [])
            if isinstance(event, dict) and event.get("active", True)
        ]
    event_products = _business_event_products(business_events)
    if event_products:
        recommendations.append(
            Recommendation(
                id="business_event_monitoring",
                action=(
                    f"Track {_natural_names(event_products)} separately during the first 1-2 trading days before releasing extra bake capacity."
                ),
                urgency="medium",
                time_horizon="this_week",
                rationale=(
                    "Active business events can change launch demand or competitor-response sensitivity without directly changing the deployed forecast model output."
                ),
                expected_impact=(
                    "Keeps scenario-driven products visible while preserving the baseline forecast and staged production discipline."
                ),
                evidence_ids=["business_events_active"],
            )
        )
    return recommendations


def _synthesize_forecast_summary(outputs: dict[str, Any]) -> str:
    overview = outputs.get("forecast_overview")
    uncertainty = outputs.get("forecast_uncertainty")
    production = outputs.get("production")
    materials = outputs.get("materials")
    accuracy = outputs.get("forecast_accuracy")

    if not _forecast_data_available(overview):
        return (
            "Forecast demand data is unavailable for the selected seven-day horizon. "
            "The production plan cannot be reconciled against demand, so quantitative "
            "bake, revenue, coverage, and procurement advice has been withheld. Verify "
            "the Forecast BI feed and rerun the analysis before locking the plan."
        )

    sentences = []
    forecast_units = _int_value(overview.metrics.get("forecast_total_units")) if overview else 0
    bakery_forecast_units = _int_value(overview.metrics.get("forecast_bakery_units")) if overview else 0
    beverage_forecast_units = _int_value(overview.metrics.get("forecast_beverage_units")) if overview else 0
    forecast_revenue = overview.metrics.get("forecast_total_revenue") if overview else 0
    total_bake = _int_value(production.metrics.get("total_bake")) if production else 0
    starting_stock = _int_value(production.metrics.get("day1_stock_total")) if production else 0
    total_available = total_bake + starting_stock
    supply_gap = max(bakery_forecast_units - total_available, 0)
    planned_profit = _float_value(production.metrics.get("total_profit")) if production else 0.0
    if planned_profit > 0 and supply_gap:
        sentences.append(
            "The seven-day outlook remains economically positive, although the production plan is intentionally conservative relative to expected bakery demand."
        )
    elif planned_profit > 0:
        sentences.append(
            "The seven-day outlook remains economically positive, and the production plan covers expected bakery demand; staged release is still appropriate because forecast uncertainty affects production timing."
        )
    else:
        sentences.append(
            "The seven-day outlook requires careful production control because the current plan does not show a positive expected profit after risk allowances."
        )

    if overview:
        metrics = overview.metrics
        trend = metrics.get("forecast_trend", "unknown")
        top_products = _names(metrics.get("top_forecast_products", []))
        sentence = (
            f"Demand is forecast at {_number(forecast_units)} units and "
            f"{_money(forecast_revenue)} projected revenue"
        )
        if bakery_forecast_units or beverage_forecast_units:
            sentence += (
                f", including {_number(bakery_forecast_units)} bakery units and "
                f"{_number(beverage_forecast_units)} made-to-order beverage units"
            )
        if trend != "unknown":
            sentence += f", with a {trend} pattern across the week"
        if top_products:
            sentence += f"; the main demand drivers are {top_products}"
        sentences.append(sentence + ".")
        business_events = [
            event for event in overview.metadata.get("business_events", [])
            if isinstance(event, dict) and event.get("active", True)
        ]
        event_sentence = _business_event_summary_sentence(business_events)
        if event_sentence:
            sentences.append(
                f"In addition, {event_sentence[0].lower()}{event_sentence[1:]}"
            )

    if production:
        metrics = production.metrics
        coverage_pct = (
            round(total_available / max(bakery_forecast_units, 1) * 100, 1)
            if bakery_forecast_units
            else 0.0
        )
        stock_clause = (
            f"{starting_stock} Day-1 carryover units give {total_available} units available"
            if starting_stock
            else f"with no Day-1 carryover stock, the plan provides {total_available} units available"
        )
        coverage_clause = (
            f"covering {coverage_pct:.1f}% of bakery forecast demand and leaving a "
            f"{supply_gap}-unit bakery gap if demand follows the forecast"
            if supply_gap
            else f"covering {coverage_pct:.1f}% of bakery forecast demand with no expected bakery gap"
        )
        sentences.append(
            f"Against this outlook, the production plan bakes {_number(total_bake)} units for "
            f"{_money(metrics.get('total_revenue'))} projected plan revenue and {_money(metrics.get('total_profit'))} "
            f"planned profit after waste and shortage risk allowances; {stock_clause}, {coverage_clause}."
        )
        if bakery_forecast_units and total_bake:
            top_products = _natural_names(overview.metrics.get("top_bakery_products", [])) if overview else ""
            material_watchlist = []
            if materials:
                material_watchlist = list(materials.metadata.get("critical_materials", [])) + list(
                    materials.metadata.get("low_materials", [])
                )
            decision = (
                "Because forecast error still affects how much demand will materialize, "
                "early sales should determine whether additional bake capacity is released"
            )
            if top_products:
                decision += f", while {top_products} should be reviewed first"
            if material_watchlist:
                decision += (
                    "; products depending on low-stock materials should wait until "
                    "procurement is confirmed"
                )
            sentences.append(decision + ".")
        if metrics.get("waste_rate_pct") is not None:
            waste_rate_pct = float(metrics.get("waste_rate_pct") or 0.0)
            if waste_rate_pct > 0:
                sentences.append(
                    f"Although expected-demand waste exposure is {waste_rate_pct}%, its effect can be limited by keeping later production conditional on observed sales."
                )
            else:
                sentences.append(
                    "Although the expected-demand scenario shows no planned waste, uncertainty still affects the timing of additional production."
                )

    if uncertainty:
        metrics = uncertainty.metrics
        uncertain_products = _names(metrics.get("top_uncertain_products", []))
        if uncertain_products:
            sentences.append(
                f"Meanwhile, uncertainty is concentrated in {uncertain_products}, with an average demand range of {_number(metrics.get('forecast_avg_interval_width'))} units across forecast products."
            )

    if materials:
        metrics = materials.metrics
        stock_data_available = bool(metrics.get("material_stock_data_available", True))
        low_count = int(metrics.get("low_material_count", 0) or 0)
        critical_count = int(metrics.get("critical_material_count", 0) or 0)
        order_quantities = _order_quantities_text(metrics.get("material_order_by_unit", {}))
        if not stock_data_available:
            sentences.append(
                "At the same time, material readiness could not be verified because current raw-material stock data was unavailable, so the inventory feed should be confirmed before the production plan is locked."
            )
        elif low_count or critical_count:
            status_parts = []
            if critical_count:
                status_parts.append(
                    f"{critical_count} critical {_plural(critical_count, 'material')}"
                )
            if low_count:
                status_parts.append(
                    f"{low_count} low-stock {_plural(low_count, 'material')}"
                )
            status_text = _natural_names(status_parts)
            order_clause = (
                f" Planned replenishment totals are {order_quantities}, with each quantity kept in its original measurement unit."
                if order_quantities
                else ""
            )
            sentences.append(
                f"At the same time, material readiness needs attention because {status_text} require procurement review.{order_clause}"
            )
        else:
            sentences.append(
                "At the same time, material readiness does not show critical or low-stock blockers for this plan."
            )

    if accuracy:
        metrics = accuracy.metrics
        wape = metrics.get("forecast_wape")
        coverage = metrics.get("forecast_coverage")
        if wape or coverage:
            sentences.append(
                f"Finally, the held-out historical evaluation shows {wape}% error and {coverage}% coverage; "
                "taken together, these results support staged production and material-readiness checks instead of locking the full week at once."
            )

    return " ".join(sentences)


def build_inventory_graph():
    graph = StateGraph(S5GraphState)
    graph.add_node("inventory", _inventory_node)
    graph.add_node("stock_data_quality", _stock_data_quality_node)
    graph.add_node("inventory_recommendation", _inventory_recommendation_node)
    graph.add_node("evidence", _evidence_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("inventory")
    graph.add_edge("inventory", "stock_data_quality")
    graph.add_edge("stock_data_quality", "inventory_recommendation")
    graph.add_edge("inventory_recommendation", "evidence")
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


def build_promotion_mix_graph():
    graph = StateGraph(S5GraphState)
    graph.add_node("promotion_signal", _promotion_signal_node)
    graph.add_node("promotion_product_mix", _promotion_product_mix_node)
    graph.add_node("promotion_decision", _promotion_decision_node)
    graph.add_node("evidence", _evidence_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("promotion_signal")
    graph.add_edge("promotion_signal", "promotion_product_mix")
    graph.add_edge("promotion_product_mix", "promotion_decision")
    graph.add_edge("promotion_decision", "evidence")
    graph.add_edge("evidence", "verify")
    graph.add_edge("verify", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def build_s5_graph(template_id: str):
    if template_id not in SUPPORTED_GRAPH_TEMPLATES:
        supported = ", ".join(sorted(SUPPORTED_GRAPH_TEMPLATES))
        raise ValueError(f"Unsupported S5 graph template: {template_id}. Supported: {supported}")
    if template_id == "promotion_mix_analysis":
        return build_promotion_mix_graph()
    if template_id == "profit_root_cause":
        return build_profit_graph()
    if template_id == "production_advice":
        return build_forecast_graph()
    if template_id == "wastage_root_cause":
        return build_wastage_graph()
    return build_inventory_graph()
