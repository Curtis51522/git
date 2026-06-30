import os, sys, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.attendance")

class AttendanceAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_attendance_log", description="Get today attendance records",
            parameters={"date": "string"}, primary=True, _handler=self._get_attendance))
        self.tools.register(Tool(name="get_punctuality", description="Get punctuality stats",
            parameters={"date": "string"}, primary=False, fallback=True, _handler=self._get_punctuality))

    async def _get_attendance(self, date: str = ""):
        return {"records": [], "on_time": 0, "late": 0, "absent": 0, "note": "stub"}

    async def _get_punctuality(self, date: str = ""):
        return await self._get_attendance(date)

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        absent = data.get("absent", 0)
        if absent > 0:
            return AgentOpinion(agent=self.name, opinion=f"{absent} employees absent today",
                confidence=0.9, attribution={"metric": "attendance", "root_cause": "staff_absence", "deviation": abs(absent)})
        return AgentOpinion(agent=self.name, opinion="All staff present", confidence=0.9,
            attribution={"metric": "attendance", "root_cause": "full_attendance", "deviation": 0})
