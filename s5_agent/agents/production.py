import os, sys, httpx, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.s5_config.settings import THRESHOLDS
logger = logging.getLogger("s5.agent.production")
BAKER_PER_OVEN = THRESHOLDS["production"]["units_per_baker_per_oven_per_hour"]

class ProductionAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_oven_capacity", description="Get oven capacity and baker count",
            parameters={}, primary=True, _handler=self._get_capacity))
        self.tools.register(Tool(name="get_s3_plan", description="Get S3 production plan",
            parameters={"date": "string"}, primary=False, _handler=self._get_s3_plan))

    async def _get_capacity(self):
        return {"ovens": THRESHOLDS["production"]["ovens"], "bakers": THRESHOLDS["production"]["bakers"], "capacity_per_hour": BAKER_PER_OVEN, "hours": THRESHOLDS["production"]["hours_per_shift"]}

    async def _get_s3_plan(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("http://127.0.0.1:8002/s4/inventory/dashboard")
                if r.status_code == 200: return r.json()
        except: pass
        return {"plan": []}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        ovens = data.get("ovens", THRESHOLDS["production"]["ovens"])
        bakers = data.get("bakers", THRESHOLDS["production"]["bakers"])
        max_cap = ovens * BAKER_PER_OVEN * 8
        demand_context = context or ""
        opinion = f"Capacity: {max_cap} units/day ({bakers} bakers x {ovens} ovens)"
        recs = []
        if "low" in demand_context.lower():
            recs.append({"action": "Reduce bake plan by 15%", "urgency": "medium", "projected_gain": 60, "ease": "high"})
        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.75,
            attribution={"metric": "production", "root_cause": "adequate_capacity", "deviation": 0},
            evidence={"max_capacity": max_cap, "bakers": bakers, "ovens": ovens},
            recommendations=recs)
