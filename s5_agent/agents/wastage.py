import os, sys, httpx, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.s5_config.settings import THRESHOLDS
logger = logging.getLogger("s5.agent.wastage")

class WastageAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_wastage_rates", description="Get current wastage rates",
            parameters={}, primary=True, _handler=self._get_wastage))
        self.tools.register(Tool(name="get_wastage_history", description="Get wastage history",
            parameters={"material_name": "string"}, primary=False, _handler=self._get_wastage_history))

    async def _get_wastage(self):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("http://127.0.0.1:8002/s4/inventory/wastage/summary")
                if r.status_code == 200: return r.json()
        except: pass
        return {"rates": []}

    async def _get_wastage_history(self, material_name: str = ""):
        return await self._get_wastage()

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        rates = data.get("rates", [])
        anomalies = [r for r in rates if abs(r.get("rate", 0)) > THRESHOLDS["wastage_abnormal_rate"]]
        if anomalies:
            names = [a.get("material_name", "?") for a in anomalies[:5]]
            return AgentOpinion(agent=self.name, opinion=f"Abnormal wastage: {', '.join(names)}",
                confidence=0.8, attribution={"metric": "wastage", "root_cause": "abnormal_wastage", "deviation": len(anomalies)},
                recommendations=[{"action": f"Check {n} wastage", "urgency": "medium", "projected_gain": 50, "ease": "medium"} for n in names])
        return AgentOpinion(agent=self.name, opinion="Wastage normal", confidence=0.8,
            attribution={"metric": "wastage", "root_cause": "normal_wastage", "deviation": 0})
