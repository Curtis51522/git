import os, sys, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.promo")

class PromoAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_active_promos", description="Get currently active promotions",
            parameters={}, primary=True, _handler=self._get_promos))
        self.tools.register(Tool(name="get_promo_performance", description="Get promo effectiveness history",
            parameters={"days": "int"}, primary=False, _handler=self._get_performance))

    async def _get_promos(self):
        return {"active": [], "note": "stub"}

    async def _get_performance(self, days: int = 30):
        return {"history": [], "note": "stub"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        active = data.get("active", [])
        if not active:
            return AgentOpinion(agent=self.name, opinion="No active promotions",
                confidence=0.9, attribution={"metric": "promo", "root_cause": "no_promos", "deviation": 0})
        return AgentOpinion(agent=self.name, opinion=f"{len(active)} active promotions",
            confidence=0.7, attribution={"metric": "promo", "root_cause": "promos_active", "deviation": 0})
