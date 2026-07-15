import json
import os, sys, logging, traceback
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.s5_config.settings import THRESHOLDS
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation
from s5_agent.core.dashboard_api import fetch_dashboard_json
from db.mysql_client import get_db
logger = logging.getLogger("s5.agent.profit")


def _profit_margin_pct(data, revenue, profit):
    dashboard_margin = data.get("profit_margin") if isinstance(data, dict) else None
    if dashboard_margin is not None:
        return float(dashboard_margin)
    return round(profit / max(revenue, 1.0) * 100, 2)


def _expired_cost_metrics(revenue, adjusted_profit, expired_cost):
    if revenue <= 0:
        return {
            "profit_before_expiry": adjusted_profit + expired_cost,
            "expired_cost_revenue_pct": 0.0,
            "profit_margin_before_expiry_pct": 0.0,
            "expired_margin_erosion_pct_points": 0.0,
        }
    expired_cost_revenue_pct = round(expired_cost / revenue * 100, 2)
    return {
        "profit_before_expiry": round(adjusted_profit + expired_cost, 2),
        "expired_cost_revenue_pct": expired_cost_revenue_pct,
        "profit_margin_before_expiry_pct": round(
            (adjusted_profit + expired_cost) / revenue * 100,
            2,
        ),
        "expired_margin_erosion_pct_points": expired_cost_revenue_pct,
    }


def _profit_cost_context(
    expired_cost,
    non_sellable_return_cost,
    revenue=0.0,
    adjusted_profit=0.0,
):
    sentences = []
    if expired_cost > 0:
        expired_metrics = _expired_cost_metrics(
            revenue,
            adjusted_profit,
            expired_cost,
        )
        if revenue > 0:
            sentences.append(
                f"Unsold products discarded at closing cost ¥{expired_cost:.2f}, equal to "
                f"{expired_metrics['expired_cost_revenue_pct']:.1f}% of revenue, and reduced "
                f"margin from {(adjusted_profit + expired_cost) / revenue * 100:.1f}% to "
                f"{adjusted_profit / revenue * 100:.1f}%."
            )
        else:
            sentences.append(
                f"Unsold products discarded at closing cost ¥{expired_cost:.2f} and are included in profit."
            )
    if non_sellable_return_cost > 0:
        sentences.append(
            "Non-sellable return cost of "
            f"¥{non_sellable_return_cost:.2f} is included in profit."
        )
    if not sentences:
        sentences.append(
            "No expired-stock or non-sellable return cost was recorded."
        )
    sentences.append(
        "Separately recorded material-wastage variance is not deducted again."
    )
    return " ".join(sentences)


class ProfitAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_revenue_breakdown", description="Get daily revenue breakdown",
            parameters={"date": "string"}, primary=True, _handler=self._get_revenue))
        self.tools.register(Tool(name="get_profit_trend", description="Get profit trend over N days",
            parameters={"days": "int"}, primary=False, _handler=self._get_trend))

    async def _get_revenue(self, date: str = "", authorization: str = ""):
        try:
            base_url = "http://127.0.0.1:8002/s4/revenue/daily"
            url = f"{base_url}?{urlencode({'date': date})}" if date else base_url
            payload = fetch_dashboard_json(
                url,
                {"_authorization": authorization},
            )
            if payload:
                return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            pass
        return {
            "total_revenue": 0,
            "total_profit": 0,
            "total_orders": 0,
            "discount_total": 0,
            "non_sellable_return_cost": 0,
        }

    async def _get_trend(self, days: int = 7):
        return await self._get_revenue()

    async def fetch(self, params):
        date_str = ""
        if isinstance(params, dict):
            date_str = str(params.get("date", ""))
        authorization = str(params.get("_authorization", "")) if isinstance(params, dict) else ""
        dashboard = await self._get_revenue(date_str, authorization)
        dashboard_data = dashboard.get("data", {}) if isinstance(dashboard, dict) else {}
        if isinstance(dashboard_data, dict) and dashboard_data:
            dashboard_margin = dashboard_data.get("profit_margin")
            return {
                "success": True,
                "data": {
                    "today_revenue": float(dashboard_data.get("today_revenue") or 0.0),
                    "today_profit": float(dashboard_data.get("today_profit") or 0.0),
                    "today_orders": int(dashboard_data.get("today_orders") or 0),
                    "profit_margin": (
                        float(dashboard_margin)
                        if dashboard_margin is not None
                        else None
                    ),
                    "discount_total": float(dashboard_data.get("today_discount") or dashboard_data.get("discount_total") or 0.0),
                    "expired_cost": float(dashboard_data.get("expired_cost") or 0.0),
                    "expired_products": dashboard_data.get("expired_products", []),
                    "non_sellable_return_cost": float(
                        dashboard_data.get("non_sellable_return_cost") or 0.0
                    ),
                },
                "tool": "revenue_dashboard",
            }
        db = None
        cur = None
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
            if date_str:
                cur.execute(
                    """
                    SELECT COALESCE(
                        SUM(
                            it.quantity
                            * p.material_cost
                            * (1 + COALESCE(p.wastage_pct, 0.03))
                        ),
                        0
                    ) AS return_cost
                    FROM inventory_transactions it
                    JOIN products p ON it.product_name = p.product_name
                    WHERE it.transaction_type = 'return'
                      AND it.disposition = 'non_sellable'
                      AND DATE(it.transaction_time) = %s
                    """,
                    (date_str,),
                )
            else:
                cur.execute(
                    """
                    SELECT COALESCE(
                        SUM(
                            it.quantity
                            * p.material_cost
                            * (1 + COALESCE(p.wastage_pct, 0.03))
                        ),
                        0
                    ) AS return_cost
                    FROM inventory_transactions it
                    JOIN products p ON it.product_name = p.product_name
                    WHERE it.transaction_type = 'return'
                      AND it.disposition = 'non_sellable'
                      AND DATE(it.transaction_time) = CURDATE()
                    """
                )
            return_row = cur.fetchone() or {}
            non_sellable_return_cost = float(return_row.get("return_cost") or 0)
            if date_str:
                cur.execute(
                    """
                    SELECT COALESCE(
                        SUM(
                            it.quantity
                            * COALESCE(NULLIF(it.unit_price, 0), p.material_cost)
                        ),
                        0
                    ) AS expired_cost
                    FROM inventory_transactions it
                    JOIN products p ON it.product_name = p.product_name
                    WHERE it.transaction_type = 'outflow'
                      AND it.freshness_status = 'Expired'
                      AND p.category = 'bakery'
                      AND DATE(it.transaction_time) = %s
                    """,
                    (date_str,),
                )
            else:
                cur.execute(
                    """
                    SELECT COALESCE(
                        SUM(
                            it.quantity
                            * COALESCE(NULLIF(it.unit_price, 0), p.material_cost)
                        ),
                        0
                    ) AS expired_cost
                    FROM inventory_transactions it
                    JOIN products p ON it.product_name = p.product_name
                    WHERE it.transaction_type = 'outflow'
                      AND it.freshness_status = 'Expired'
                      AND p.category = 'bakery'
                      AND DATE(it.transaction_time) = CURDATE()
                    """
                )
            expired_row = cur.fetchone() or {}
            expired_cost = float(expired_row.get("expired_cost") or 0)
            profit_val -= non_sellable_return_cost + expired_cost
            data = {
                "today_revenue": revenue,
                "today_profit": profit_val,
                "today_orders": orders,
                "discount_total": discount,
                "expired_cost": expired_cost,
                "non_sellable_return_cost": non_sellable_return_cost,
            }
            return {"success": True, "data": data, "tool": "profit_db"}
        except Exception as e:
            logger.warning("Profit DB fetch failed for date=%s: %s", date_str, e)
            logger.warning("Profit traceback: %s", traceback.format_exc())
        finally:
            if cur is not None:
                cur.close()
            if db is not None:
                db.close()
        return {
            "success": True,
            "data": {
                "today_revenue": 0,
                "today_profit": 0,
                "today_orders": 0,
                "discount_total": 0,
                "expired_cost": 0,
                "non_sellable_return_cost": 0,
            },
            "tool": "profit_db_fallback",
        }

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        if "data" in data and isinstance(data["data"], dict) and "today_revenue" in data["data"]:
            data = data["data"]
        revenue = float(data.get("today_revenue", 0))
        profit = float(data.get("today_profit", 0))
        orders = data.get("today_orders", 0)
        non_sellable_return_cost = float(
            data.get("non_sellable_return_cost", 0) or 0
        )
        expired_cost = float(data.get("expired_cost", 0) or 0)
        margin = _profit_margin_pct(data, revenue, profit)
        cost_context = _profit_cost_context(
            expired_cost,
            non_sellable_return_cost,
            revenue,
            profit,
        )
        healthy_cost_context = f". {cost_context}"
        expired_metrics = _expired_cost_metrics(revenue, profit, expired_cost)
        is_material_unsold_loss = (
            revenue >= 100.0
            and orders > 3
            and expired_metrics["expired_cost_revenue_pct"]
            >= THRESHOLDS["profit_expired_cost_alert_pct"]
        )

        contributions = self._parse_upstream(context)
        if is_material_unsold_loss:
            opening = (
                "Revenue remained profitable, but unsold product loss needs attention"
                if profit > 0
                else "Revenue performance and unsold product loss need attention"
            )
            return AgentOpinion(
                agent=self.name,
                opinion=(
                    f"{opening}: margin {margin:.1f}% on {orders} orders, with revenue at "
                    f"¥{revenue:.0f} and profit at ¥{profit:.0f}. {cost_context}"
                ),
                confidence=0.85,
                attribution={
                    "metric": "expired_cost_revenue_pct",
                    "root_cause": "unsold_product_loss",
                    "deviation": (
                        expired_metrics["expired_cost_revenue_pct"]
                        - THRESHOLDS["profit_expired_cost_alert_pct"]
                    ),
                    "contributions": {"expired_finished_products": 100},
                },
            )
        if margin < THRESHOLDS["profit_low_margin_pct"]:
            top_factor = max(contributions, key=contributions.get, default="unknown")
            return AgentOpinion(agent=self.name,
                opinion=(
                    f"Margin {margin:.1f}% below {THRESHOLDS['profit_low_margin_pct']}% "
                    f"threshold on {orders} orders. Revenue ¥{revenue:.0f}, profit "
                    f"¥{profit:.0f}. Main drag: {top_factor}. {cost_context}"
                ),
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
                f"revenue data{healthy_cost_context}"
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
        non_sellable_return_cost = float(
            data.get("non_sellable_return_cost", 0.0) or 0.0
        )
        expired_cost = float(data.get("expired_cost", 0.0) or 0.0)
        expired_products = data.get("expired_products", [])
        if not isinstance(expired_products, list):
            expired_products = []
        expired_products = sorted(
            (item for item in expired_products if isinstance(item, dict)),
            key=lambda item: float(item.get("expired_cost", 0.0) or 0.0),
            reverse=True,
        )
        margin_pct = _profit_margin_pct(data, revenue, profit)
        expired_metrics = _expired_cost_metrics(revenue, profit, expired_cost)
        average_order_value = round(revenue / max(orders, 1), 2)
        discount_rate = round(discount_total / max(revenue, 1.0), 4)
        has_enough_sales_sample = revenue >= 100.0 and orders > 3
        is_low_margin = has_enough_sales_sample and margin_pct < THRESHOLDS["profit_low_margin_pct"]
        is_material_unsold_loss = (
            has_enough_sales_sample
            and expired_metrics["expired_cost_revenue_pct"]
            >= THRESHOLDS["profit_expired_cost_alert_pct"]
        )

        evidence_items = [
            EvidenceItem(
                id="profit_margin_pct",
                source="profit",
                description="Adjusted profit margin percentage for the requested period",
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
            EvidenceItem(
                id="expired_cost",
                source="profit",
                description=(
                    "Cost of expired finished products recognized in profit for "
                    "the requested period"
                ),
                value=expired_cost,
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="expired_cost_revenue_pct",
                source="profit",
                description="Closing unsold-product loss as a percentage of revenue",
                value=expired_metrics["expired_cost_revenue_pct"],
                metadata={
                    "date": params.get("date", ""),
                    "alert_threshold_pct": THRESHOLDS[
                        "profit_expired_cost_alert_pct"
                    ],
                },
            ),
            EvidenceItem(
                id="profit_margin_before_expiry_pct",
                source="profit",
                description="Profit margin before closing unsold-product loss",
                value=expired_metrics["profit_margin_before_expiry_pct"],
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="expired_margin_erosion_pct_points",
                source="profit",
                description="Profit-margin percentage points lost to closing unsold products",
                value=expired_metrics["expired_margin_erosion_pct_points"],
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="non_sellable_return_cost",
                source="profit",
                description=(
                    "Cost of non-sellable returned products recognized in profit for "
                    "the requested period"
                ),
                value=non_sellable_return_cost,
                metadata={"date": params.get("date", "")},
            ),
        ]
        if expired_products:
            evidence_items.append(
                EvidenceItem(
                    id="expired_products",
                    source="profit",
                    description=(
                        "Products discarded at closing with recorded loss and "
                        "sell-through context"
                    ),
                    value=expired_products,
                    metadata={"date": params.get("date", "")},
                )
            )

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
        if is_material_unsold_loss:
            priority_products = expired_products[:3]
            priority_names = [
                str(item.get("name") or "").strip()
                for item in priority_products
                if str(item.get("name") or "").strip()
            ]
            if len(priority_names) > 1:
                priority_label = ", ".join(priority_names[:-1]) + ", and " + priority_names[-1]
            elif priority_names:
                priority_label = priority_names[0]
            else:
                priority_label = "the products discarded at closing"
            sell_through_rates = [
                f"{float(item.get('sell_through_pct', 0.0) or 0.0):.1f}%"
                for item in priority_products
            ]
            if len(sell_through_rates) > 1:
                sell_through_label = ", ".join(sell_through_rates[:-1]) + ", and " + sell_through_rates[-1]
            elif sell_through_rates:
                sell_through_label = sell_through_rates[0]
            else:
                sell_through_label = ""
            detail_rationale = (
                f" The highest recorded loss items were {priority_label}, with sell-through rates "
                f"of {sell_through_label}, respectively."
                if priority_names and sell_through_label
                else ""
            )
            evidence_ids = [
                "expired_cost",
                "expired_cost_revenue_pct",
                "profit_margin_before_expiry_pct",
                "profit_margin_pct",
            ]
            if expired_products:
                evidence_ids.append("expired_products")
            recommendations.append(
                Recommendation(
                    id="unsold_product_loss_reduction",
                    action=(
                        f"Review {priority_label} before the next production plan, "
                        "then reduce or stage bake quantities for items with repeated unsold loss."
                    ),
                    urgency="high",
                    time_horizon="this_week",
                    rationale=(
                        f"Closing unsold-product loss was ¥{expired_cost:.2f}, equal to "
                        f"{expired_metrics['expired_cost_revenue_pct']:.1f}% of revenue and "
                        f"reducing margin from {(profit + expired_cost) / revenue * 100:.1f}% to "
                        f"{margin_pct:.1f}%.{detail_rationale}"
                    ),
                    expected_impact=(
                        "Reduces avoidable finished-product loss while keeping production "
                        "changes tied to observed demand."
                    ),
                    evidence_ids=evidence_ids,
                )
            )

        risks = []
        if is_low_margin:
            risks.append("low_margin")
        if is_material_unsold_loss:
            risks.append("unsold_product_loss")

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
                "expired_cost": expired_cost,
                "expired_products": expired_products,
                **expired_metrics,
                "non_sellable_return_cost": non_sellable_return_cost,
            },
            evidence_items=evidence_items,
            risks=risks,
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
