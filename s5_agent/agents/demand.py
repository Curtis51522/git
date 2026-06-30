import os, sys, httpx, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.demand")

class DemandAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_forecast", description="Get S2 demand forecast",
            parameters={"date": "string"}, primary=True, _handler=self._get_forecast))
        self.tools.register(Tool(name="get_actual_sales", description="Get actual sales",
            parameters={"date": "string"}, primary=False, fallback=True, _handler=self._get_actual_sales))

    async def _get_forecast(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                url = f"http://127.0.0.1:8002/s4/revenue/daily?date={date}" if date else "http://127.0.0.1:8002/s4/revenue/daily"
                r = await c.get(url)
                if r.status_code == 200:
                    d = r.json()
                    return {"forecast": d.get("total_revenue", 0), "orders": d.get("total_orders", 0)}
        except Exception as e:
            logger.warning("Forecast fetch failed: %s", e)
        return {"forecast": 0, "orders": 0}

    async def _get_actual_sales(self, date: str = ""):
        return await self._get_forecast(date)

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        orders = data.get("orders", 0)
        return AgentOpinion(agent=self.name, opinion=f"Today: {orders} orders",
            confidence=0.7, attribution={"metric": "demand", "root_cause": "normal_demand", "deviation": 0},
            evidence={"orders": orders})
