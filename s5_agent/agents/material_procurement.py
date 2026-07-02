import os, sys, logging
from datetime import datetime as dt, timedelta
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from db.mysql_client import get_db
logger = logging.getLogger("s5.agent.material_procurement")

class MaterialProcurementAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_material_procurement",
            description="Get material procurement status",
            parameters={"date": "string"}, primary=True,
            _handler=self._get_materials))

    async def _get_materials(self, date: str = ""):
        return _query_materials(date)

    async def fetch(self, params):
        date_str = str(params.get("date", "")) if isinstance(params, dict) else ""
        data = _query_materials(date_str)
        return {"success": True, "data": data, "tool": "material_procurement"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        items = data.get("items", {})
        if not items:
            return AgentOpinion(agent=self.name,
                opinion="No material procurement data available.",
                confidence=0.3,
                attribution={"metric": "material_procurement", "root_cause": "no_data", "deviation": 0})

        critical = []
        low = []
        ok_count = 0
        total_order = 0
        biggest_gap = {"name": "", "gap": 0}

        for name, info in items.items():
            alert = str(info.get("alert", info.get("status", ""))).lower()
            need_raw = info.get("weekly_need")
            need = float(need_raw) if need_raw is not None else None
            stock = float(info.get("current_stock", 0))
            order = float(info.get("to_order", 0))
            total_order += order
            if need is not None:
                gap = need - stock
                if gap > biggest_gap["gap"]:
                    biggest_gap = {"name": name, "gap": gap}
            note = info.get("note", "")
            if note:
                ok_count += 1  # no estimate materials are informational only
            elif alert in ("critical", "urgent", "low") or (need is not None and stock < need * 0.5):
                if stock == 0 or (need is not None and stock < need * 0.3):
                    critical.append({"name": name, "need": need or 0, "stock": stock})
                else:
                    low.append({"name": name, "need": need or 0, "stock": stock})
            else:
                ok_count += 1

        parts = [f"{len(items)} materials monitored."]
        if critical:
            parts.append(f"Critical: " + ", ".join(
                f"{c['name']} (need {c['need']:.1f}, have {c['stock']:.1f})" for c in critical
            ))
        if low:
            parts.append(f"Low stock: " + ", ".join(
                f"{l['name']} (need {l['need']:.1f}, have {l['stock']:.1f})" for l in low
            ))
        parts.append(f"{ok_count} materials adequate.")
        parts.append(f"Total to order: {total_order:.0f} units.")
        if biggest_gap["name"]:
            parts.append(f"Largest gap: {biggest_gap['name']} (shortfall {biggest_gap['gap']:.1f}).")

        opinion = " ".join(parts)

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.80,
            attribution={"metric": "material_procurement", "root_cause": "materials_analysis",
                         "critical_count": len(critical), "low_count": len(low),
                         "total_order": total_order})


def _query_materials(date_str=""):
    try:
        from s3_scheduling.scheduler import Scheduler, generate_7day_s2_forecast
        if not date_str:
            date_str = dt.now().strftime("%Y-%m-%d")
        start_dt = dt.strptime(date_str, "%Y-%m-%d")
        if start_dt.weekday() != 0:
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
        dm = result["dashboard_materials"]
        return {"items": dm.get("items", {})}
    except Exception as e:
        logger.warning("MaterialProcurement fetch failed: %s", e)
        return {"items": {}}
