import os, sys, logging
from datetime import datetime as dt
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from db.mysql_client import get_db
logger = logging.getLogger("s5.agent.production_plan")

class ProductionPlanAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_production_plan",
            description="Get 7-day production plan",
            parameters={"date": "string"}, primary=True,
            _handler=self._get_plan))

    async def _get_plan(self, date: str = ""):
        return _query_plan(date)

    async def fetch(self, params):
        date_str = str(params.get("date", "")) if isinstance(params, dict) else ""
        data = _query_plan(date_str)
        return {"success": True, "data": data, "tool": "production_plan"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        summary = data.get("weekly_summary", {})
        grid = data.get("grid", [])
        dates = data.get("dates", [])
        buffer = data.get("buffer", 1.0)

        if not grid:
            return AgentOpinion(agent=self.name,
                opinion="No production plan data available.",
                confidence=0.3,
                attribution={"metric": "production_plan", "root_cause": "no_data", "deviation": 0})

        total_bake = summary.get("total_bake", 0)
        total_rev = summary.get("total_revenue", 0)
        total_profit = summary.get("total_profit", 0)
        scenarios = summary.get("scenarios", {})

        # Top baked products
        by_product = {}
        for row in grid:
            pn = row.get("product_name", "")
            qty = row.get("bake_qty", 0) or row.get("qty", 0) or 0
            by_product[pn] = by_product.get(pn, 0) + qty
        top_baked = sorted(by_product.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = "; ".join(f"{pn} ({qty}u)" for pn, qty in top_baked)

        scenario_note = ""
        q50_s = scenarios.get("q50", {})
        if q50_s:
            scenario_note = (
                f" Q50 scenario: {chr(165)}{q50_s.get('profit', 0):.0f} profit, "
                f"{q50_s.get('waste', 0)} waste units."
            )

        opinion = (
            f"7-day production plan ({len(dates)} days): {total_bake} total bake units, "
            f"{chr(165)}{total_rev:.0f} revenue, {chr(165)}{total_profit:.0f} profit. "
            f"Buffer: {buffer:.0%}. "
            f"Top baked: {top_str}.{scenario_note}"
        )

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.80,
            attribution={"metric": "production_plan", "root_cause": "plan_analysis",
                         "total_bake": total_bake, "total_rev": total_rev, "total_profit": total_profit})


def _query_plan(date_str=""):
    try:
        from s3_scheduling.scheduler import Scheduler, generate_7day_s2_forecast
        if not date_str:
            date_str = dt.now().strftime("%Y-%m-%d")
        start_dt = dt.strptime(date_str, "%Y-%m-%d")
        if start_dt.weekday() != 0:
            from datetime import timedelta
            start_dt -= timedelta(days=start_dt.weekday())
        start_date = start_dt.strftime("%Y-%m-%d")
        s = Scheduler()
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT product_name, stock_day1 FROM products WHERE category='bakery'")
        day1_stock = {str(r[0]): int(r[1] or 0) for r in cur.fetchall()}
        for p in s.breads:
            if p not in day1_stock:
                day1_stock[p] = 0
        forecast = generate_7day_s2_forecast(start_date)
        result = s.generate_7day_plan(start_date, day1_stock, forecast)
        d7 = result["dashboard_7day"]
        ws = result["weekly_summary"]
        return {
            "grid": d7.get("grid", []),
            "dates": d7.get("dates", []),
            "buffer": d7.get("buffer_applied", 1.0),
            "weekly_summary": {
                "total_bake": ws.get("total_bake", 0),
                "total_revenue": ws.get("total_revenue", 0),
                "total_profit": ws.get("total_profit", 0),
                "scenarios": ws.get("scenarios", {}),
            }
        }
    except Exception as e:
        logger.warning("ProductionPlan fetch failed: %s", e)
        return {"grid": [], "dates": [], "buffer": 1.0, "weekly_summary": {}}
