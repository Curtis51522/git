import os, sys, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.dashboard_api import fetch_dashboard_json
from s5_agent.core.tool import Tool
from s5_agent.s5_config.settings import THRESHOLDS
logger = logging.getLogger("s5.agent.promo")

class PromoAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_active_promos", description="Get currently active promotions",
            parameters={}, primary=True, _handler=self._get_promos))
        self.tools.register(Tool(name="get_discount_impact", description="Get discount impact on revenue",
            parameters={"date": "string"}, primary=False, _handler=self._get_discount))

    async def _get_promos(self, date: str = "", authorization: str = ""):
        try:
            url = "http://127.0.0.1:8002/s4/revenue/daily" + (f"?date={date}" if date else "")
            payload = fetch_dashboard_json(
                url,
                {"_authorization": authorization},
            )
            data = payload.get("data", {})
            disc = data.get("today_discount", 0)
            rev = data.get("today_revenue", 1)
            return {"active_promos": 1 if disc > 0 else 0, "total_discount": disc, "discount_rate": disc/max(rev,1)}
        except Exception as e:
            logger.warning("Promo fetch failed: %s", e)
        return {"active_promos": 0, "total_discount": 0, "discount_rate": 0}

    async def fetch(self, params):
        date = str(params.get("date", "")) if isinstance(params, dict) else ""
        authorization = str(params.get("_authorization", "")) if isinstance(params, dict) else ""
        data = await self._get_promos(date, authorization)
        return {"success": True, "data": data, "tool": "revenue_dashboard"}

    async def _get_discount(self, date: str = ""):
        return await self._get_promos(date)

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        api_data = raw.get("data", {}) if isinstance(raw, dict) and "data" in raw else raw
        data = api_data.get("data", api_data) if isinstance(api_data, dict) else {}
        discount_total = float(data.get("today_discount", data.get("discount_total", 0)) or 0)
        revenue = float(data.get("today_revenue", data.get("revenue", 0)) or 0)
        discount_rate = float(data.get("discount_rate", 0) or 0)
        if not discount_rate and revenue > 0:
            discount_rate = discount_total / revenue
        deviation = discount_rate * 100
        if discount_rate > THRESHOLDS["promo_high_discount_rate"]:
            return AgentOpinion(agent=self.name,
                opinion=f"High discount rate: {deviation:.1f}%",
                confidence=0.8,
                attribution={"metric": "discount_rate", "root_cause": "high_discount", "deviation": deviation})
        return AgentOpinion(agent=self.name,
            opinion="Normal discount levels",
            confidence=0.8,
            attribution={"metric": "discount_rate", "root_cause": "normal_discount", "deviation": deviation})
