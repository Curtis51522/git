import os, sys, httpx, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.pricing")

class PricingAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_discount_policy", description="Get current discount settings",
            parameters={}, primary=True, _handler=self._get_discounts))
        self.tools.register(Tool(name="get_discount_history", description="Get historical discount usage",
            parameters={"days": "int"}, primary=False, _handler=self._get_history))

    async def _get_discounts(self):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("http://127.0.0.1:8002/s4/revenue/daily")
                if r.status_code == 200:
                    d = r.json()
                    return {"discount_total": d.get("discount_total", 0), "total": d.get("total_revenue", 1)}
        except: pass
        return {"discount_total": 0, "total": 1}

    async def _get_history(self, days: int = 7):
        return await self._get_discounts()

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        disc = float(data.get("discount_total", 0))
        total = float(data.get("total", 1))
        rate = disc / max(total, 1) * 100
        if rate > 15:
            return AgentOpinion(agent=self.name, opinion=f"High discount rate: {rate:.1f}%",
                confidence=0.8, attribution={"metric": "discount_rate", "root_cause": "excessive_discount", "deviation": rate},
                recommendations=[{"action": f"Reduce discounts below 10%", "urgency": "high", "projected_gain": disc * 0.5, "ease": "high"}])
        return AgentOpinion(agent=self.name, opinion=f"Discount rate normal: {rate:.1f}%",
            confidence=0.8, attribution={"metric": "discount_rate", "root_cause": "normal_discount", "deviation": rate})
