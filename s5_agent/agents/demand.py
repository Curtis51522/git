import os, sys, logging, json, urllib.request
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.demand")

class DemandAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_forecast", description="Get S2 demand forecast",
            parameters={"date": "string"}, primary=True, _handler=self._get_forecast))
        self.tools.register(Tool(name="get_actual_sales", description="Get actual sales from orders DB",
            parameters={"date": "string"}, primary=False, fallback=True, _handler=self._get_actual_sales))

    async def _get_forecast(self, date: str = ""):
        return _fetch_demand_api(date)

    async def _get_actual_sales(self, date: str = ""):
        return await self._get_forecast(date)

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        Y = chr(165)
        data = raw.get("data", {}) if "data" in raw else raw
        orders = data.get("orders", data.get("today_orders", 0))
        revenue = data.get("today_revenue", 0)

        if orders == 0 and revenue == 0:
            date = ""
            if isinstance(params, dict):
                date = str(params.get("date", ""))
            if date:
                fresh = _fetch_demand_api(date)
                orders = fresh.get("orders", fresh.get("today_orders", 0))
                revenue = fresh.get("today_revenue", 0)
                data = fresh

        if orders == 0 and revenue == 0:
            return AgentOpinion(agent=self.name,
                opinion="0 orders today, avg {0}0.00/order, total revenue {0}0".format(Y),
                confidence=0.3,
                attribution={"metric": "demand", "root_cause": "no_data", "deviation": 0},
                evidence={"orders": 0, "avg_order_value": 0, "revenue": 0})

        avg_order_val = revenue / max(orders, 1)
        opinion = "{0} orders today, avg {1}{2:.2f}/order, total revenue {1}{3:.0f}".format(orders, Y, avg_order_val, revenue)

        root_cause = "normal_demand"
        deviation = 0
        recs = []
        weekly_avg_orders = data.get("weekly_avg_orders", 0)
        if weekly_avg_orders > 0:
            order_dev = (orders - weekly_avg_orders) / max(weekly_avg_orders, 1) * 100
            if order_dev < -15:
                root_cause = "demand_drop"
                deviation = round(order_dev, 1)
                opinion += " | Orders {0:+.1f}% vs weekly avg {1:.1f}".format(order_dev, weekly_avg_orders)
                recs.append({"action": "Investigate demand drop", "urgency": "high",
                    "rationale": "Orders {0:+.0f}% below weekly average".format(order_dev)})
            elif order_dev > 25:
                root_cause = "demand_surge"
                deviation = round(order_dev, 1)
                opinion += " | Orders {0:+.1f}% vs weekly avg {1:.1f}".format(order_dev, weekly_avg_orders)
                recs.append({"action": "Ensure production capacity", "urgency": "medium",
                    "rationale": "Orders {0:+.0f}% above weekly average".format(order_dev)})

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.80,
            attribution={"metric": "demand", "root_cause": root_cause, "deviation": deviation},
            evidence={"orders": orders, "avg_order_value": round(avg_order_val, 2), "revenue": revenue},
            recommendations=recs)


def _fetch_demand_api(date=""):
    url = "http://127.0.0.1:8002/s4/revenue/daily"
    if date:
        url += f"?date={date}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.loads(r.read())
            api_data = d.get("data", {})
            trend = api_data.get("trend", {})
            orders_list = trend.get("orders", [])
            weekly_avg = sum(orders_list[:-1]) / max(len(orders_list) - 1, 1) if len(orders_list) > 1 else 0
            return {
                "orders": api_data.get("today_orders", 0),
                "orders_change": api_data.get("orders_change", 0),
                "avg_order": api_data.get("avg_order", 0),
                "avg_change": api_data.get("avg_change", 0),
                "today_revenue": api_data.get("today_revenue", 0),
                "revenue_change": api_data.get("revenue_change", 0),
                "weekly_avg_orders": round(weekly_avg, 1),
            }
    except Exception as e:
        logger.warning("DEMAND_API_SYNC: error=%s", e)
    return {"orders": 0, "orders_change": 0, "avg_order": 0,
            "avg_change": 0, "today_revenue": 0, "revenue_change": 0, "weekly_avg_orders": 0}