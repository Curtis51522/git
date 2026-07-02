import os, sys, logging
from datetime import datetime as dt, timedelta
from collections import defaultdict
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.wastage")

WASTAGE_SQL = """
SELECT
    mi.material_name,
    mi.current_stock,
    mi.unit,
    mi.cost_per_unit,
    COALESCE(wc.theoretical_consumed, 0) AS theoretical_consumed,
    COALESCE(wc.actual_consumed, 0) AS actual_consumed,
    COALESCE(wc.wastage_qty, 0) AS wastage_qty,
    COALESCE(wc.wastage_rate, 0) AS wastage_rate,
    wc.check_date
FROM material_inventory mi
LEFT JOIN (
    SELECT wc1.*
    FROM material_wastage_log wc1
    INNER JOIN (
        SELECT material_name, MAX(check_date) AS max_date
        FROM material_wastage_log
        WHERE check_date <= %s
        GROUP BY material_name
    ) wc2 ON wc1.material_name = wc2.material_name AND wc1.check_date = wc2.max_date
) wc ON mi.material_name = wc.material_name
ORDER BY COALESCE(wc.wastage_qty, 0) DESC
"""

TREND_SQL = """
SELECT
    material_name,
    check_date,
    COALESCE(wastage_qty, 0) AS wastage_qty,
    COALESCE(wastage_rate, 0) AS wastage_rate,
    COALESCE(theoretical_consumed, 0) AS theoretical_consumed,
    COALESCE(actual_consumed, 0) AS actual_consumed
FROM material_wastage_log
WHERE check_date >= %s AND check_date <= %s
ORDER BY material_name, check_date
"""

ROOT_CAUSE_MAP = {
    "Coffee Beans": "grinder calibration or brewing spillage",
    "Tea": "over-preparation or spillage during steeping",
    "Butter": "measurement errors or refrigeration issues",
    "Eggs": "cracking waste or over-preparation",
    "Milk": "over-pouring or refrigeration spoilage",
    "Bread Flour": "dough waste, scaling errors, or spillage",
    "Cake Flour": "scaling errors or prep spillage",
    "Sugar": "spillage during measuring or dough waste",
    "Yeast": "over-portioning or expired starter",
    "Chocolate": "portioning errors or temperature spoilage",
    "Cream": "over-whipping or refrigeration spoilage",
    "Cheese": "portioning errors or temperature spoilage",
    "Lids": "handling damage or over-dispensing",
    "Cups": "handling damage or over-dispensing",
    "Box": "handling damage or assembly waste",
    "Packaging Bag": "over-dispensing or tearing",
    "Bags": "over-dispensing or tearing",
    "Cup Regular": "handling damage or over-dispensing",
    "Cup Large": "handling damage or over-dispensing",
}


class WastageAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_wastage_trend",
            description="Get 14-day wastage trend per material with root cause analysis",
            parameters={"date": "string"}, primary=True,
            _handler=self._get_wastage))

    async def fetch(self, params):
        date = str(params.get("date", "")) if isinstance(params, dict) else ""
        if not date:
            date = dt.now().strftime("%Y-%m-%d")
        trend_data = _query_trend(date)
        latest_data = _query_wastage(date)
        return {"success": True, "data": {"trend": trend_data, "latest": latest_data}}

    async def _get_wastage(self, date: str = ""):
        return await self.fetch({"date": date})

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        trend_data = data.get("trend", {})
        latest_data = data.get("latest", {})
        trend_materials = trend_data.get("materials", [])
        latest_materials = latest_data.get("materials", [])

        if not trend_materials and not latest_materials:
            return AgentOpinion(agent=self.name,
                opinion="No wastage data available.",
                confidence=0.3,
                attribution={"metric": "wastage", "root_cause": "no_data"})

        query_date = str(params.get("date", "")) if isinstance(params, dict) else ""
        if not query_date:
            query_date = dt.now().strftime("%Y-%m-%d")

        # Group trend data by material
        material_days = defaultdict(list)
        for m in trend_materials:
            material_days[m["material_name"]].append(m)

        # Compute per-material trends
        material_analysis = []
        for name, days in sorted(material_days.items()):
            days.sort(key=lambda d: d["check_date"])
            today = [d for d in days if d["check_date"] == query_date]
            past_7 = [d for d in days if d["check_date"] < query_date][-7:]

            today_rate = today[0]["wastage_rate"] if today else 0
            today_qty = today[0]["wastage_qty"] if today else 0.0
            avg_7_rate = sum(d["wastage_rate"] for d in past_7) / len(past_7) if past_7 else today_rate
            avg_7_qty = sum(d["wastage_qty"] for d in past_7) / len(past_7) if past_7 else today_qty

            # Find matching latest material for cost
            latest_match = next((m for m in latest_materials if m["material_name"] == name), {})
            cost_per_unit = float(latest_match.get("cost_per_unit") or 0)
            unit = latest_match.get("unit", "kg")
            waste_cost_today = today_qty * cost_per_unit

            # Trend direction
            if len(past_7) >= 2 and today_rate > 0:
                delta = today_rate - avg_7_rate
                if delta > 0.01:
                    direction = "rising"
                elif delta < -0.01:
                    direction = "falling"
                else:
                    direction = "stable"
            else:
                direction = "stable"

            # Root cause match
            root_cause = ROOT_CAUSE_MAP.get(name, "general handling or measurement")

            material_analysis.append({
                "name": name,
                "unit": unit,
                "today_rate": today_rate,
                "today_qty": today_qty,
                "today_cost": round(waste_cost_today, 2),
                "avg_7_rate": round(avg_7_rate, 4),
                "avg_7_qty": round(avg_7_qty, 4),
                "direction": direction,
                "root_cause": root_cause,
            })

        # Compute totals
        materials_checked = len(latest_materials) if latest_materials else len(material_analysis)
        total_today_cost = sum(m["today_cost"] for m in material_analysis)
        rising = [m for m in material_analysis if m["direction"] == "rising" and m["today_rate"] >= 0.01]
        stable = [m for m in material_analysis if m["direction"] in ("stable", "falling")]
        has_any_wastage = any(m["today_rate"] > 0 for m in material_analysis)

        if not has_any_wastage:
            top_consumed = sorted(material_analysis, 
                key=lambda m: (sum(d["theoretical_consumed"] for d in material_days.get(m["name"], []) 
                               if d["check_date"] == query_date) or 0), reverse=True)[:3]
            top_names = ", ".join(f"{m['name']}" for m in top_consumed)
            opinion = (f"Wastage check completed: all {materials_checked} materials at 0% wastage. "
                      f"Top consumed today: {top_names}. Material costs are fully realized in sales.")
            confidence = 0.85
        elif rising:
            # Rising wastage items are the story
            rising_report = []
            for r in sorted(rising, key=lambda x: -x["today_rate"])[:5]:
                trend_note = f"up from {r['avg_7_rate']*100:.1f}% 7-day avg" if r["avg_7_rate"] > 0 else "new this period"
                rising_report.append(
                    f"{r['name']}: {r['today_rate']*100:.1f}% today ({trend_note}), "
                    f"{r['today_qty']:.3f}{r['unit']} lost = CNY{r['today_cost']:.2f}. "
                    f"Likely cause: {r['root_cause']}."
                )
            opinion = (
                f"{materials_checked} materials checked. {len(rising)} with rising wastage. "
                f"Total waste cost today: CNY{total_today_cost:.2f}. "
                + " ".join(rising_report)
            )
            confidence = 0.85
        else:
            # Wastage exists but stable/falling
            highest = max(material_analysis, key=lambda m: m["today_rate"]) if material_analysis else None
            opinion = (
                f"{materials_checked} materials checked. All within normal thresholds. "
                f"Highest: {highest['name']} at {highest['today_rate']*100:.1f}% (stable vs 7-day avg). "
                f"Total waste cost today: CNY{total_today_cost:.2f}. No concerning trends."
            )
            confidence = 0.85

        # Build recommendations
        recs = []
        for r in sorted(rising, key=lambda x: -x["today_rate"])[:3]:
            recs.append({
                "action": f"Investigate {r['name']} ({r['today_rate']*100:.0f}% rate, rising from {r['avg_7_rate']*100:.1f}% 7-day avg, CNY{r['today_cost']:.2f} today)",
                "urgency": "high" if r["today_rate"] > 0.03 else "medium",
                "rationale": f"Wastage rising above 7-day average. Likely cause: {r['root_cause']}.",
                "expected_impact": f"Returning {r['name']} to {r['avg_7_rate']*100:.1f}% average would save CNY{r['today_cost']*0.5:.2f}/day"
            })

        return AgentOpinion(agent=self.name,
            opinion=opinion,
            confidence=confidence,
            attribution={
                "metric": "wastage_trend",
                "materials_checked": materials_checked,
                "rising_count": len(rising),
                "total_waste_cost": round(total_today_cost, 2),
            },
            recommendations=recs)


def _query_wastage(date=""):
    if not date:
        date = dt.now().strftime("%Y-%m-%d")
    try:
        from db.mysql_client import get_db
        db = get_db()
        cursor = db.cursor()
        cursor.execute(WASTAGE_SQL, (date,))
        materials = []
        for row in cursor.fetchall():
            materials.append({
                "material_name": row[0],
                "current_stock": float(row[1]) if row[1] else 0,
                "unit": row[2] or "kg",
                "cost_per_unit": float(row[3]) if row[3] else None,
                "theoretical_consumed": float(row[4]) if row[4] else 0,
                "actual_consumed": float(row[5]) if row[5] else 0,
                "wastage_qty": float(row[6]) if row[6] else 0,
                "wastage_rate": float(row[7]) if row[7] else 0,
                "check_date": str(row[8]) if row[8] else None,
            })
        cursor.close()
        db.close()
        return {"materials": materials}
    except Exception as e:
        logger.warning("WASTAGE_DB: error=%s", e)
        return {"materials": []}


def _query_trend(date=""):
    if not date:
        date = dt.now().strftime("%Y-%m-%d")
    start_date_dt = dt.strptime(date, "%Y-%m-%d") - timedelta(days=14)
    start_date = start_date_dt.strftime("%Y-%m-%d")
    try:
        from db.mysql_client import get_db
        db = get_db()
        cursor = db.cursor()
        cursor.execute(TREND_SQL, (start_date, date))
        materials = []
        for row in cursor.fetchall():
            materials.append({
                "material_name": row[0],
                "check_date": str(row[1]) if row[1] else None,
                "wastage_qty": float(row[2]) if row[2] else 0,
                "wastage_rate": float(row[3]) if row[3] else 0,
                "theoretical_consumed": float(row[4]) if row[4] else 0,
                "actual_consumed": float(row[5]) if row[5] else 0,
            })
        cursor.close()
        db.close()
        return {"materials": materials}
    except Exception as e:
        logger.warning("WASTAGE_TREND_DB: error=%s", e)
        return {"materials": []}
