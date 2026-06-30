import os, sys, httpx, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.profit")

class ProfitAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_revenue_breakdown", description="Get daily revenue breakdown",
            parameters={"date": "string"}, primary=True, _handler=self._get_revenue))
        self.tools.register(Tool(name="get_profit_trend", description="Get profit trend over N days",
            parameters={"days": "int"}, primary=False, _handler=self._get_trend))

    async def _get_revenue(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                url = f"http://127.0.0.1:8002/s4/revenue/daily?date={date}" if date else "http://127.0.0.1:8002/s4/revenue/daily"
                r = await c.get(url)
                if r.status_code == 200: return r.json()
        except: pass
        return {"total_revenue": 0, "total_profit": 0, "total_orders": 0, "discount_total": 0}

    async def _get_trend(self, days: int = 7):
        return await self._get_revenue()

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        revenue = float(data.get("total_revenue", 0))
        profit = float(data.get("total_profit", 0))
        orders = data.get("total_orders", 0)
        margin = profit / max(revenue, 1) * 100

        contributions = self._parse_upstream(context)
        if margin < 20:
            top_factor = max(contributions, key=contributions.get, default="unknown")
            return AgentOpinion(agent=self.name,
                opinion=f"Profit margin {margin:.1f}% - below 20% threshold. Revenue: ${revenue:.0f}, Profit: ${profit:.0f}",
                confidence=0.85,
                attribution={"metric": "profit_margin", "root_cause": f"low_margin_{top_factor}",
                    "deviation": margin - 25, "contribution_pct": 0, "contributions": contributions},
                recommendations=[{"action": f"Address {top_factor} to improve margin", "urgency": "high",
                    "projected_gain": revenue * 0.05, "ease": "medium"}])
        return AgentOpinion(agent=self.name,
            opinion=f"Profit margin {margin:.1f}% - healthy. Revenue: ${revenue:.0f}, Profit: ${profit:.0f}",
            confidence=0.85,
            attribution={"metric": "profit_margin", "root_cause": "healthy_margin", "deviation": 0, "contributions": {}})

    def _parse_upstream(self, context: str) -> dict:
        contribs = {}
        if "excessive_discount" in context: contribs["discount"] = 25
        if "low_stock" in context: contribs["stock_shortage"] = 15
        if "staff_absence" in context: contribs["staffing"] = 10
        if "rain" in context: contribs["weather"] = 20
        if "understaffed" in context: contribs["staffing"] = 15
        if not contribs: contribs["normal_operations"] = 100
        return contribs
