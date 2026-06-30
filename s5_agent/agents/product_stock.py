import os, sys, httpx, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.s5_config.settings import THRESHOLDS
logger = logging.getLogger("s5.agent.product_stock")

class ProductStockAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_batch_inventory", description="Get finished goods inventory",
            parameters={"date": "string"}, primary=True, _handler=self._get_inventory))
        self.tools.register(Tool(name="get_freshness_ratio", description="Get fresh vs day-1 ratio",
            parameters={"date": "string"}, primary=False, fallback=True, _handler=self._get_freshness))

    async def _get_inventory(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("http://127.0.0.1:8002/s4/inventory/dashboard")
                if r.status_code == 200: return r.json()
        except: pass
        return {"products": []}

    async def _get_freshness(self, date: str = ""):
        return await self._get_inventory(date)

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        prods = data.get("products", [])
        day1_count = sum(1 for p in prods if p.get("freshness") == "day-1")
        if day1_count > len(prods) * THRESHOLDS["product_day1_ratio"]:
            return AgentOpinion(agent=self.name, opinion=f"High Day-1 stock: {day1_count}/{len(prods)} products aging",
                confidence=0.8, attribution={"metric": "product_stock", "root_cause": "high_day1_ratio", "deviation": day1_count},
                recommendations=[{"action": "Consider Day-1 discount or reprioritize baking", "urgency": "medium", "projected_gain": 80, "ease": "high"}])
        return AgentOpinion(agent=self.name, opinion="Product stock healthy", confidence=0.8,
            attribution={"metric": "product_stock", "root_cause": "healthy_stock", "deviation": 0})
