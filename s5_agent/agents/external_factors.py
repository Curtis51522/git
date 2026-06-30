import os, sys, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from datetime import date as dt_date
logger = logging.getLogger("s5.agent.external")

class ExternalFactorsAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_weather", description="Get weather for today",
            parameters={"date": "string"}, primary=True, _handler=self._get_weather))
        self.tools.register(Tool(name="get_calendar", description="Check holiday/weekend",
            parameters={"date": "string"}, primary=False, fallback=True, _handler=self._get_calendar))

    async def _get_weather(self, date: str = ""):
        return {"temperature": 28, "condition": "clear", "is_rainy": False, "note": "stub"}

    async def _get_calendar(self, date: str = ""):
        today = date or dt_date.today().isoformat()
        return {"date": today, "is_holiday": False, "is_weekend": dt_date.today().weekday() >= 5}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        factors = []
        if data.get("is_rainy"): factors.append("rain")
        if data.get("is_weekend"): factors.append("weekend")
        if data.get("is_holiday"): factors.append("holiday")
        opinion = f"External: {', '.join(factors) if factors else 'no significant factors'}"
        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.8,
            attribution={"metric": "external", "root_cause": "normal_day", "deviation": 0},
            evidence=data)
