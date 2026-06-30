import os, sys, logging, httpx
from datetime import date as dt_date
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.external")

class ExternalFactorsAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_weather", description="Get weather for today",
            parameters={"date": "string"}, primary=True, _handler=self._get_weather))
        self.tools.register(Tool(name="get_holiday", description="Check if today is holiday",
            parameters={"date": "string"}, primary=False, _handler=self._get_holiday))

    async def _get_weather(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                url = f"http://127.0.0.1:8002/s4/revenue/daily?date={date}" if date else "http://127.0.0.1:8002/s4/revenue/daily"
                r = await c.get(url)
                if r.status_code == 200:
                    d = r.json()
                    data = d.get("data", {})
                    # Extract weather-related fields from response or fallback
                    return {
                        "is_rainy": data.get("is_rainy", 0),
                        "temp_mean": data.get("temp_mean", 25),
                        "is_holiday": data.get("is_holiday", 0),
                        "is_weekend": data.get("is_weekend", 0)
                    }
        except Exception as e:
            logger.warning("External fetch failed: %s", e)
        return {"is_rainy": 0, "temp_mean": 25, "is_holiday": 0, "is_weekend": 0}

    async def _get_holiday(self, date: str = ""):
        w = await self._get_weather(date)
        return {"is_holiday": w.get("is_holiday", 0), "is_weekend": w.get("is_weekend", 0)}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        api_data = raw.get("data", {}) if "data" in raw else raw
        data = api_data.get("data", api_data) if isinstance(api_data, dict) else api_data
        is_rainy = data.get("is_rainy", 0)
        is_holiday = data.get("is_holiday", 0)
        factors = []
        if is_rainy: factors.append("rain")
        if is_holiday: factors.append("holiday")
        
        if factors:
            return AgentOpinion(agent=self.name,
                opinion=f"External factors: {', '.join(factors)}",
                confidence=0.8,
                attribution={"metric": "external", "root_cause": factors[0], "deviation": 1,
                    "factors": factors})
        return AgentOpinion(agent=self.name,
            opinion="No significant external factors (normal day)",
            confidence=0.8,
            attribution={"metric": "external", "root_cause": "normal_day", "deviation": 0})
