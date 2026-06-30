import os, sys, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.yield")

class YieldAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_actual_bake", description="Get S1 actual production",
            parameters={"date": "string"}, primary=True, _handler=self._get_actual))
        self.tools.register(Tool(name="get_planned_bake", description="Get S3 planned production",
            parameters={"date": "string"}, primary=False, fallback=True, _handler=self._get_planned))

    async def _get_actual(self, date: str = ""):
        return {"actual_units": 0, "products": [], "note": "stub - S1 batch_inventory"}

    async def _get_planned(self, date: str = ""):
        return {"planned_units": 0, "products": [], "note": "stub - S3 plan"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        actual = data.get("actual_units", 0)
        planned = data.get("planned_units", 0)
        if planned > 0:
            rate = actual / planned * 100
            if rate < 90:
                return AgentOpinion(agent=self.name, opinion=f"Yield {rate:.0f}% - underproduction",
                    confidence=0.7, attribution={"metric": "yield_rate", "root_cause": "underproduction", "deviation": rate - 100})
        return AgentOpinion(agent=self.name, opinion="Yield data unavailable",
            confidence=0.3, attribution={"metric": "yield_rate", "root_cause": "no_data", "deviation": 0})
