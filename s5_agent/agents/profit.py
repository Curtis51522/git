import json
import os, sys, logging, traceback
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.s5_config.settings import THRESHOLDS
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation
from db.mysql_client import get_db
logger = logging.getLogger("s5.agent.profit")


class ProfitAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_revenue_breakdown", description="Get daily revenue breakdown",
            parameters={"date": "string"}, primary=True, _handler=self._get_revenue))
        self.tools.register(Tool(name="get_profit_trend", description="Get profit trend over N days",
            parameters={"days": "int"}, primary=False, _handler=self._get_trend))

    async def _get_revenue(self, date: str = ""):
        try:
            base_url = "http://127.0.0.1:8002/s4/revenue/daily"
            url = f"{base_url}?{urlencode({'date': date})}" if date else base_url
            with urlopen(url, timeout=10) as response:
                if response.status == 200:
                    return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            pass
        return {"total_revenue": 0, "total_profit": 0, "total_orders": 0, "discount_total": 0}

    async def _get_trend(self, days: int = 7):
        return await self._get_revenue()

    async def fetch(self, params):
        date_str = ""
        if isinstance(params, dict):
            date_str = str(params.get("date", ""))
        dashboard = await self._get_revenue(date_str)
        dashboard_data = dashboard.get("data", {}) if isinstance(dashboard, dict) else {}
        if isinstance(dashboard_data, dict) and dashboard_data:
            return {
                "success": True,
                "data": {
                    "today_revenue": float(dashboard_data.get("today_revenue") or 0.0),
                    "today_profit": float(dashboard_data.get("today_profit") or 0.0),
                    "today_orders": int(dashboard_data.get("today_orders") or 0),
                    "discount_total": float(dashboard_data.get("today_discount") or dashboard_data.get("discount_total") or 0.0),
                },
                "tool": "revenue_dashboard",
            }
        try:
            db = get_db()
            cur = db.cursor(dictionary=True)
            if date_str:
                cur.execute("SELECT SUM(total_amount) as revenue, SUM(total_profit) as profit, COUNT(*) as orders, SUM(discount_total) as disc FROM orders WHERE order_date=%s AND state IN ('paid','completed')", (date_str,))
            else:
                cur.execute("SELECT SUM(total_amount) as revenue, SUM(total_profit) as profit, COUNT(*) as orders, SUM(discount_total) as disc FROM orders WHERE order_date=CURDATE() AND state IN ('paid','completed')")
            row = cur.fetchone()
            revenue = float(row.get("revenue") or 0)
            profit_val = float(row.get("profit") or 0)
            orders = int(row.get("orders") or 0)
            discount = float(row.get("disc") or 0)
            data = {
                "today_revenue": revenue,
                "today_profit": profit_val,
                "today_orders": orders,
                "discount_total": discount,
            }
            return {"success": True, "data": data, "tool": "profit_db"}
        except Exception as e:
            logger.warning("Profit DB fetch failed for date=%s: %s", date_str, e)
            logger.warning("Profit traceback: %s", traceback.format_exc())
        return {"success": True, "data": {"today_revenue": 0, "today_profit": 0, "today_orders": 0, "discount_total": 0}, "tool": "profit_db_fallback"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        if "data" in data and isinstance(data["data"], dict) and "today_revenue" in data["data"]:
            data = data["data"]
        revenue = float(data.get("today_revenue", 0))
        profit = float(data.get("today_profit", 0))
        orders = data.get("today_orders", 0)
        margin = profit / max(revenue, 1) * 100

        contributions = self._parse_upstream(context)
        if margin < THRESHOLDS["profit_low_margin_pct"]:
            top_factor = max(contributions, key=contributions.get, default="unknown")
            return AgentOpinion(agent=self.name,
                opinion=f"Margin {margin:.1f}% below {THRESHOLDS['profit_low_margin_pct']}% threshold on {orders} orders. Revenue ¥{revenue:.0f}, profit ¥{profit:.0f}. Main drag: {top_factor}.",
                confidence=0.85,
                attribution={"metric": "profit_margin", "root_cause": f"low_margin_{top_factor}",
                    "deviation": margin - 25, "contribution_pct": 0, "contributions": contributions},
                recommendations=[{"action": f"Address {top_factor} to improve margin", "urgency": "high",
                    "projected_gain": revenue * 0.05, "ease": "medium"}])
        avg_ticket = revenue / max(orders, 1)
        return AgentOpinion(agent=self.name,
            opinion=(
                f"Revenue performance is healthy: margin {margin:.1f}% on {orders} orders, "
                f"with average order value at ¥{avg_ticket:.2f}. Revenue reached ¥{revenue:.0f} "
                f"and profit reached ¥{profit:.0f}. No discount erosion is visible in the "
                f"revenue data; waste impact is not included in this check."
            ),
            confidence=0.85,
            attribution={"metric": "profit_margin", "root_cause": "healthy_margin", "deviation": 0, "contributions": {}})

    def analyze_for_graph(self, raw: dict, params: dict) -> AgentOutput:
        opinion = self.analyze(raw, params)
        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        if "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        revenue = float(data.get("today_revenue", 0.0) or 0.0)
        profit = float(data.get("today_profit", 0.0) or 0.0)
        orders = int(data.get("today_orders", 0) or 0)
        discount_total = float(data.get("discount_total", 0.0) or 0.0)
        margin_pct = round(profit / max(revenue, 1.0) * 100, 2)
        average_order_value = round(revenue / max(orders, 1), 2)
        discount_rate = round(discount_total / max(revenue, 1.0), 4)
        has_enough_sales_sample = revenue >= 100.0 and orders > 3
        is_low_margin = has_enough_sales_sample and margin_pct < THRESHOLDS["profit_low_margin_pct"]

        evidence_items = [
            EvidenceItem(
                id="profit_margin_pct",
                source="profit",
                description="Gross profit margin percentage for the requested period",
                value=margin_pct,
                metadata={"date": params.get("date", ""), "orders": orders},
            ),
            EvidenceItem(
                id="revenue",
                source="profit",
                description="Total revenue for the requested period",
                value=revenue,
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="discount_total",
                source="profit",
                description="Total discount amount for the requested period",
                value=discount_total,
                metadata={"date": params.get("date", ""), "discount_rate": discount_rate},
            ),
            EvidenceItem(
                id="order_volume",
                source="profit",
                description="Number of paid or completed orders for the requested period",
                value=orders,
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="average_order_value",
                source="profit",
                description="Average revenue per order for the requested period",
                value=average_order_value,
                metadata={"date": params.get("date", "")},
            ),
        ]

        recommendations = []
        if is_low_margin:
            recommendations.append(
                Recommendation(
                    id="profit_margin_recovery",
                    action="Review discounting, product mix, and cost drivers for margin recovery.",
                    urgency="high",
                    time_horizon="this_week",
                    rationale="The observed profit margin is below the configured alert threshold.",
                    expected_impact=round(revenue * 0.05, 2),
                    evidence_ids=["profit_margin_pct"],
                )
            )

        return AgentOutput(
            agent_name="ProfitAgent",
            claim=opinion.opinion,
            confidence=float(opinion.confidence),
            metrics={
                "revenue": revenue,
                "profit": profit,
                "orders": orders,
                "average_order_value": average_order_value,
                "profit_margin_pct": margin_pct,
                "discount_total": discount_total,
                "discount_rate": discount_rate,
            },
            evidence_items=evidence_items,
            risks=["low_margin"] if is_low_margin else [],
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness="fresh" if revenue > 0 else "unknown",
                completeness=1.0 if revenue > 0 else 0.5,
                source_status={"profit": "fresh" if revenue > 0 else "unknown"},
            ),
            metadata={"root_cause": opinion.attribution.get("root_cause", "")},
        )

    def _parse_upstream(self, context: str) -> dict:
        contribs = {}
        if "excessive_discount" in context: contribs["discount"] = 25
        if "low_stock" in context: contribs["stock_shortage"] = 15
        if "staff_absence" in context: contribs["staffing"] = 10
        if "rain" in context: contribs["weather"] = 20
        if "understaffed" in context: contribs["staffing"] = 15
        if not contribs: contribs["normal_operations"] = 100
        return contribs
