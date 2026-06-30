import os, sys, logging, httpx
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.product_mix")

class ProductMixAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_product_ranking", description="Get bread and beverage sales ranking",
            parameters={"date": "string"}, primary=True, _handler=self._get_ranking))

    async def _get_ranking(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                url = "http://127.0.0.1:8002/s4/revenue/daily"
                if date: url += f"?date={date}"
                r = await c.get(url)
                if r.status_code == 200:
                    d = r.json().get("data", {})
                    return {
                        "bread_ranking": d.get("bread_ranking", []),
                        "beverage_ranking": d.get("beverage_ranking", []),
                        "category": d.get("category", {}),
                    }
        except Exception as e:
            logger.warning("Product mix fetch failed: %s", e)
        return {"bread_ranking": [], "beverage_ranking": [], "category": {}}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        bread = data.get("bread_ranking", [])
        bev = data.get("beverage_ranking", [])
        cat = data.get("category", {})

        if not bread and not bev:
            return AgentOpinion(agent=self.name,
                opinion="No product ranking data available.",
                confidence=0.3,
                attribution={"metric": "product_mix", "root_cause": "no_data", "deviation": 0})

        parts = []
        recs = []
        root_cause = "normal_mix"
        deviation = 0

        # Bread: top concentration risk
        if bread:
            total_bread_rev = sum(p.get("revenue", 0) for p in bread)
            top3_rev = sum(p.get("revenue", 0) for p in bread[:3])
            top3_pct = top3_rev / max(total_bread_rev, 1) * 100

            top_name = bread[0].get("name", "?")
            top_qty = bread[0].get("qty", 0)
            parts.append(f"Bread: {len(bread)} products, top is {top_name} ({top_qty} units, {chr(165)}{bread[0].get('revenue',0):.0f})")

            if top3_pct > 60:
                root_cause = "high_concentration"
                deviation = round(top3_pct - 50)
                parts.append(f"Top 3 breads = {top3_pct:.0f}% of revenue (concentration risk)")
                recs.append({
                    "action": f"Top 3 breads drive {top3_pct:.0f}% revenue. Promote mid-tier products to reduce risk.",
                    "urgency": "low",
                    "rationale": f"If {top_name} sells out or quality drops, {top3_pct:.0f}% of bread revenue is at risk"
                })

            # Low performers
            low = [p for p in bread if p.get("qty", 0) <= 3]
            if low:
                low_names = ", ".join(p["name"] for p in low[:3])
                parts.append(f"Low sellers: {low_names} (<=3 units)")

        # Beverages
        if bev:
            top_bev = bev[0].get("name", "?")
            top_bev_qty = bev[0].get("qty", 0)
            parts.append(f"Beverages: top is {top_bev} ({top_bev_qty} units)")

        # Category mix
        bread_rev = cat.get("Bread", 0)
        bev_rev = (cat.get("Beverages", 0) or 0) + (cat.get("Coffee", 0) or 0)
        total_cat = bread_rev + bev_rev
        if total_cat > 0:
            bev_pct = bev_rev / total_cat * 100
            parts.append(f"Mix: Bread {chr(165)}{bread_rev:.0f} vs Beverages {chr(165)}{bev_rev:.0f} ({bev_pct:.0f}% beverages)")

        opinion = " | ".join(parts)

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.75,
            attribution={"metric": "product_mix", "root_cause": root_cause, "deviation": deviation},
            recommendations=recs)
