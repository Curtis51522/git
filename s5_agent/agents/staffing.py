import os, sys, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.staffing")

class StaffingAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_schedule", description="Get today schedule",
            parameters={"date": "string"}, primary=True, _handler=self._get_schedule))
        self.tools.register(Tool(name="get_attendance", description="Get attendance vs schedule",
            parameters={"date": "string"}, primary=False, fallback=True, _handler=self._get_attendance))

    async def _get_schedule(self, date: str = ""):
        return {"shifts": [], "total_staff": 0, "note": "stub"}

    async def _get_attendance(self, date: str = ""):
        return await self._get_schedule(date)

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        staff = data.get("total_staff", 0)
        if staff < 3:
            return AgentOpinion(agent=self.name, opinion=f"Understaffed: only {staff} on shift",
                confidence=0.85, attribution={"metric": "staffing", "root_cause": "understaffed", "deviation": 3 - staff},
                recommendations=[{"action": "Consider calling backup staff", "urgency": "high", "projected_gain": 120, "ease": "low"}])
        return AgentOpinion(agent=self.name, opinion=f"Staffing adequate: {staff} on shift",
            confidence=0.85, attribution={"metric": "staffing", "root_cause": "adequate_staffing", "deviation": 0})
