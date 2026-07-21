import os, sys, logging
from datetime import datetime as dt
from urllib.parse import urlencode
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.dashboard_api import fetch_dashboard_json
from s5_agent.core.tool import Tool
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation
logger = logging.getLogger("s5.agent.production_plan")

class ProductionPlanAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_production_plan",
            description="Get 7-day production plan",
            parameters={"date": "string"}, primary=True,
            _handler=self._get_plan))

    async def _get_plan(self, date: str = ""):
        return _query_plan(date)

    async def fetch(self, params):
        date_str = str(params.get("date", "")) if isinstance(params, dict) else ""
        data = _query_plan(date_str, params)
        return {"success": True, "data": data, "tool": "production_plan"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        summary = data.get("weekly_summary", {})
        grid = data.get("grid", [])
        dates = data.get("dates", [])
        buffer = data.get("buffer", 1.0)

        if not grid:
            return AgentOpinion(agent=self.name,
                opinion="No production plan data available.",
                confidence=0.3,
                attribution={"metric": "production_plan", "root_cause": "no_data", "deviation": 0})

        total_bake = summary.get("total_bake", 0)
        total_rev = summary.get("total_revenue", 0)
        total_profit = summary.get("total_profit", 0)
        scenarios = summary.get("scenarios", {})
        day1_stock = data.get("day1_stock_total", 0)
        total_available = total_bake + day1_stock

        # Top baked products from weekly summary
        top_baked = summary.get("top_products", [])[:5]
        if not top_baked:
            # Fallback: aggregate from plans if available
            top_baked = []
        top_str = "; ".join(f"{pn} ({qty}u)" for pn, qty in top_baked) if top_baked else "no product breakdown"

        scenario_note = ""
        q50_s = scenarios.get("q50", {})
        q50_waste = float(q50_s.get("waste", 0) or 0) if q50_s else 0.0
        waste_rate_pct = round(q50_waste / max(float(total_bake or 0), 1.0) * 100, 1)
        if q50_s:
            scenario_note = (
                f" In the expected-demand scenario, profit reaches {chr(165)}{q50_s.get('profit', 0):.0f}, "
                f"but {q50_s.get('waste', 0)} units may remain as waste, "
                f"equal to a {waste_rate_pct}% waste rate."
            )

        planning_days = len(dates) if len(dates) == 7 else 7
        stock_note = (
            f" A {buffer:.0%} buffer lifts total available supply to {total_available} units "
            f"after {day1_stock} starting-stock units."
            if day1_stock
            else f" A {buffer:.0%} buffer is applied to the production plan."
        )
        opinion = (
            f"The {planning_days}-day plan calls for {total_bake} bake units, with projected "
            f"revenue of {chr(165)}{total_rev:.0f} and profit of {chr(165)}{total_profit:.0f} "
            "after waste and shortage risk allowances."
            f"{stock_note} The main bake load is concentrated in {top_str}.{scenario_note}"
        )

        recommendations = []
        if buffer and buffer > 1.3:
            recommendations.append({
                "action": f"Reduce production buffer from {buffer:.0%} to 120% across all products",
                "urgency": "medium", "time_horizon": "this_week",
                "rationale": f"Buffer at {buffer:.0%} inflates bake quantities beyond forecast confidence level",
                "expected_impact": "Reduces over-production waste while maintaining sufficient stock for demand peaks"
            })
        if buffer and buffer < 1.0:
            recommendations.append({
                "action": f"Increase buffer to at least 110% to cover forecast uncertainty",
                "urgency": "high", "time_horizon": "tomorrow",
                "rationale": f"Buffer below 100% risks stockouts if actual demand exceeds forecast",
                "expected_impact": "Prevents lost sales from under-baking on peak days"
            })

        # Daily distribution check
        daily_bakes = {}
        for row in grid:
            d = row.get("date", row.get("day", ""))
            qty = (
                row.get("bake_total", 0)
                or row.get("bake_qty", 0)
                or row.get("qty", 0)
                or 0
            )
            daily_bakes[d] = daily_bakes.get(d, 0) + qty
        if daily_bakes:
            vals = list(daily_bakes.values())
            avg_daily = sum(vals) / len(vals)
            max_daily = max(vals)
            if avg_daily > 0 and max_daily / avg_daily > 1.8:
                peak_day = max(daily_bakes, key=daily_bakes.get)
                recommendations.append({
                    "action": f"Smooth production: shift 15% of {peak_day} bake volume to adjacent days to reduce peak-day strain",
                    "urgency": "low", "time_horizon": "this_week",
                    "rationale": f"{peak_day} bake is {max_daily/avg_daily:.0f}x the daily average, creating capacity bottleneck",
                    "expected_impact": "Evens out staff and oven utilization, reduces peak-day overtime"
                })

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.80,
            attribution={"metric": "production_plan", "root_cause": "plan_analysis",
                         "total_bake": total_bake, "total_rev": total_rev, "total_profit": total_profit},
            recommendations=recommendations)

    def analyze_for_graph(self, raw: dict, params: dict) -> AgentOutput:
        opinion = self.analyze(raw, params)
        data = raw.get("data", {}) if isinstance(raw, dict) and "data" in raw else raw
        if not isinstance(data, dict):
            data = {}

        summary = data.get("weekly_summary", {}) or {}
        total_bake = int(summary.get("total_bake", 0) or 0)
        total_revenue = float(summary.get("total_revenue", 0.0) or 0.0)
        total_profit = float(summary.get("total_profit", 0.0) or 0.0)
        profit_definition = str(
            summary.get(
                "profit_definition",
                "after_waste_and_shortage_risk_allowances",
            )
        )
        buffer = float(data.get("buffer", 1.0) or 1.0)
        day1_stock_total = int(data.get("day1_stock_total", 0) or 0)
        grid = data.get("grid", []) or []
        dates = data.get("dates", []) or []
        scenarios = summary.get("scenarios", {}) or {}
        q50_profit = float((scenarios.get("q50", {}) or {}).get("profit", 0.0) or 0.0)
        q10_profit = float((scenarios.get("q10", {}) or {}).get("profit", 0.0) or 0.0)
        q50_waste = float((scenarios.get("q50", {}) or {}).get("waste", 0.0) or 0.0)
        q90 = scenarios.get("q90", {}) or {}
        q90_shortage_units = int(q90.get("shortage", q90.get("shortage_units", 0)) or 0)
        scenario_profit_gap = round(q50_profit - q10_profit, 2)
        waste_rate_pct = round(q50_waste / max(total_bake, 1) * 100, 2)
        selected_date = str(params.get("date", "") or "")
        first_day_row = next(
            (
                row
                for row in grid
                if str(row.get("date", row.get("day", ""))) == selected_date
            ),
            grid[0] if grid else {},
        )
        first_day_date = str(
            first_day_row.get("date", first_day_row.get("day", "")) or ""
        )
        first_day_bake = int(
            first_day_row.get("bake_total", 0)
            or first_day_row.get("bake_qty", 0)
            or first_day_row.get("qty", 0)
            or 0
        )
        first_day_bake_plan = first_day_row.get("bake_plan", {}) or {}
        first_day_top_products = [
            name
            for name, _ in sorted(
                first_day_bake_plan.items(),
                key=lambda item: float(item[1] or 0),
                reverse=True,
            )[:5]
        ]

        risks = []
        if buffer > 1.3:
            risks.append("overproduction_risk")
        if buffer < 1.0:
            risks.append("stockout_risk")
        evidence_items = [
            EvidenceItem(
                id="production_total_bake",
                source="production_plan",
                description="Total planned bake units for the planning horizon",
                value=total_bake,
                metadata={"date": params.get("date", ""), "days": len(dates)},
            ),
            EvidenceItem(
                id="production_buffer",
                source="production_plan",
                description="Production buffer applied to forecast demand",
                value=buffer,
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="scenario_profit_gap",
                source="production_plan",
                description="Profit gap between median and downside production scenarios",
                value=scenario_profit_gap,
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="production_waste_rate_pct",
                source="production_plan",
                description="Expected-demand waste units as a percentage of planned bake units",
                value=waste_rate_pct,
                metadata={"date": params.get("date", ""), "expected_demand_waste_units": q50_waste},
            ),
            EvidenceItem(
                id="q90_shortage_units",
                source="production_plan",
                description="Bakery units not covered in the high-demand production scenario",
                value=q90_shortage_units,
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="production_day1_bake",
                source="production_plan",
                description="Planned bakery units for the first day of the selected horizon",
                value=first_day_bake,
                metadata={"date": first_day_date},
            ),
        ]

        recommendations = []
        for index, recommendation in enumerate(opinion.recommendations):
            action_text = recommendation.get("action", "")
            evidence_ids = (
                ["production_total_bake"]
                if action_text.startswith("Smooth production")
                else ["production_buffer"]
            )
            recommendations.append(
                Recommendation(
                    id=f"production_plan_action_{index + 1}",
                    action=recommendation.get("action", "Review production plan."),
                    urgency=recommendation.get("urgency", "medium"),
                    time_horizon=recommendation.get("time_horizon", "this_week"),
                    rationale=recommendation.get(
                        "rationale",
                        "Production plan metrics indicate an actionable planning risk.",
                    ),
                    expected_impact=recommendation.get("expected_impact"),
                    evidence_ids=evidence_ids,
                )
            )

        return AgentOutput(
            agent_name="ProductionPlanAgent",
            claim=opinion.opinion,
            confidence=float(opinion.confidence),
            metrics={
                "total_bake": total_bake,
                "total_revenue": total_revenue,
                "total_profit": total_profit,
                "profit_definition": profit_definition,
                "buffer": buffer,
                "day1_stock_total": day1_stock_total,
                "scenario_profit_gap": scenario_profit_gap,
                "waste_rate_pct": waste_rate_pct,
                "q90_shortage_units": q90_shortage_units,
                "production_day1_date": first_day_date,
                "production_day1_bake": first_day_bake,
                "production_day1_top_products": first_day_top_products,
            },
            evidence_items=evidence_items,
            risks=risks,
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness="fresh" if grid else "unknown",
                completeness=1.0 if total_bake > 0 else 0.5,
                source_status={"production_plan": "fresh" if grid else "unknown"},
            ),
            metadata={
                "planning_days": len(dates),
                "top_products": summary.get("top_products", [])[:5],
                "first_day_bake_plan": first_day_bake_plan,
            },
        )


def _query_plan(date_str="", params=None):
    try:
        if not date_str:
            date_str = dt.now().strftime("%Y-%m-%d")
        start_date = dt.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        url = (
            "http://127.0.0.1:8002/s3/plan/7day?"
            + urlencode({"date": start_date})
        )
        payload = fetch_dashboard_json(url, params, timeout=120)
        d7 = payload.get("dashboard_7day", {})
        ws = payload.get("weekly_summary", {})
        day1_total = int(
            d7.get("day1_stock_total", ws.get("day1_stock_total", 0)) or 0
        )
        return {
            "grid": d7.get("grid", []),
            "dates": d7.get("dates", []),
            "buffer": d7.get("buffer_applied", 1.0),
            "day1_stock_total": day1_total,
            "weekly_summary": {
                "total_bake": ws.get("total_bake", 0),
                "total_revenue": ws.get("total_revenue", 0),
                "total_profit": ws.get("total_profit", 0),
                "profit_definition": ws.get(
                    "profit_definition",
                    "after_waste_and_shortage_risk_allowances",
                ),
                "scenarios": ws.get("scenarios", {}),
                "top_products": ws.get("top_products", []),
            }
        }
    except Exception as e:
        logger.warning("ProductionPlan fetch failed: %s", e)
        return {"grid": [], "dates": [], "buffer": 1.0, "weekly_summary": {}}
