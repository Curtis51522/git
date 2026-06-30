import os, sys, logging, httpx
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.s5_config.settings import THRESHOLDS
logger = logging.getLogger("s5.agent.promo")

class PromoAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_active_promos", description="Get currently active promotions",
            parameters={}, primary=True, _handler=self._get_promos))
        self.tools.register(Tool(name="get_discount_impact", description="Get discount impact on revenue",
            parameters={"date": "string"}, primary=False, _handler=self._get_discount))

    async def _get_promos(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("http://127.0.0.1:8002/s4/revenue/daily" + (f"?date={date}" if date else ""))
                if r.status_code == 200:
                    d = r.json()
                    data = d.get("data", {})
                    disc = data.get("today_discount", 0)
                    rev = data.get("today_revenue", 1)
                    return {"active_promos": 1 if disc > 0 else 0, "total_discount": disc, "discount_rate": disc/max(rev,1)}
        except Exception as e:
            logger.warning("Promo fetch failed: %s", e)
        return {"active_promos": 0, "total_discount": 0, "discount_rate": 0}

    async def _get_discount(self, date: str = ""):
        return await self._get_promos(date)

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        api_data = raw.get("data", {}) if "data" in raw else raw
        data = api_data.get("data", api_data) if isinstance(api_data, dict) else api_data
        disc_rate = data.get("discount_rate", 0)
        if disc_rate > THRESHOLDS["promo_high_discount_rate"]:
            return AgentOpinion(agent=self.name,
                opinion=f"High discount rate: {disc_rate*100:.1f}%",
                confidence=0.8,
                attribution={"metric": "discount_rate", "root_cause": "high_discount", "deviation": disc_rate*100})
        return AgentOpinion(agent=self.name,
            opinion="Normal discount levels",
            confidence=0.8,
            attribution={"metric": "discount_rate", "root_cause": "normal_discount", "deviation": 0})
