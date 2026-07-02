import os, sys, logging
from datetime import datetime as dt
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

class WastageAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_wastage_rates",
            description="Get detailed wastage rates per material",
            parameters={"date": "string"}, primary=True,
            _handler=self._get_wastage))

    async def fetch(self, params):
        date = str(params.get("date", "")) if isinstance(params, dict) else ""
        if not date:
            date = dt.now().strftime("%Y-%m-%d")
        data = _query_wastage(date)
        return {"success": True, "data": data}

    async def _get_wastage(self, date: str = ""):
        return await self.fetch({"date": date})

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        materials = data.get("materials", [])

        if not materials:
            return AgentOpinion(agent=self.name,
                opinion="No wastage data available for today.",
                confidence=0.3,
                attribution={"metric": "wastage", "root_cause": "no_data", "deviation": 0})

        # Check if any material has a check_date matching the query date
        query_date = str(params.get("date", "")) if isinstance(params, dict) else ""
        if not query_date:
            from datetime import datetime as _dt
            query_date = _dt.now().strftime("%Y-%m-%d")
        checked_today = any(str(m.get("check_date", "")) == query_date for m in materials)
        has_wastage = any((m.get("wastage_qty") or 0) > 0 for m in materials)

        if not checked_today:
            latest_date = max((str(m.get("check_date", "")) for m in materials if m.get("check_date")), default="unknown")
            return AgentOpinion(agent=self.name,
                opinion=f"No wastage check submitted for this date. Most recent check: {latest_date}. Submit a check via Inventory > New Material Check to enable wastage tracking.",
                confidence=0.5,
                attribution={"metric": "wastage", "root_cause": "no_check_today", "deviation": 0})

        if not has_wastage:
            top_by_consumed = sorted(materials, key=lambda m: m.get("theoretical_consumed", 0) or 0, reverse=True)[:3]
            top_names = ", ".join(f"{m['material_name']} ({m.get('theoretical_consumed',0):.1f}{m.get('unit','kg')})" for m in top_by_consumed)
            return AgentOpinion(agent=self.name,
                opinion=f"Wastage check completed: all {len(materials)} materials at 0% wastage. Top consumed: {top_names}.",
                confidence=0.85,
                attribution={"metric": "wastage", "root_cause": "zero_wastage", "deviation": 0})

        total_waste_cost = 0.0
        high_waste = []
        all_items = []
        for m in materials:
            qty = m.get("wastage_qty", 0) or 0
            rate = m.get("wastage_rate", 0) or 0
            cost = m.get("cost_per_unit") or 0
            waste_cost = qty * float(cost) if cost else 0
            total_waste_cost += waste_cost
            name = m.get("material_name", "?")
            unit = m.get("unit", "kg")
            all_items.append({"name": name, "qty": qty, "rate": rate, "cost": waste_cost, "unit": unit})
            if rate > 0.05:
                high_waste.append({"name": name, "qty": qty, "rate": rate, "cost": waste_cost, "unit": unit})

        all_items.sort(key=lambda x: -x["rate"])

        parts = [f"{len(materials)} materials checked."]
        if high_waste:
            details = "; ".join(
                f"{h['name']} ({h['rate']*100:.1f}% rate, {h['qty']:.2f}{h['unit']} lost, ~CNY{h['cost']:.0f})"
                for h in high_waste[:5]
            )
            parts.append(f"Elevated wastage: {details}.")
            parts.append(f"Total estimated wastage cost: CNY{total_waste_cost:.0f}.")
        else:
            parts.append("All materials within normal wastage thresholds (<5%).")
            if all_items:
                top = all_items[0]
                parts.append(f"Highest: {top['name']} at {top['rate']*100:.1f}%.")

        opinion = " ".join(parts)

        recs = []
        for h in high_waste[:3]:
            recs.append({
                "action": f"Investigate {h['name']} wastage ({h['rate']*100:.0f}% rate, CNY{h['cost']:.0f} lost)",
                "urgency": "high" if h["rate"] > 0.15 else "medium",
                "rationale": f"{h['name']} wastage rate of {h['rate']*100:.0f}% exceeds 5% threshold, costing ~CNY{h['cost']:.0f} today",
                "expected_impact": f"Reducing {h['name']} wastage to 5% could save CNY{h['cost']*0.7:.0f}/day"
            })

        return AgentOpinion(agent=self.name,
            opinion=opinion,
            confidence=0.85,
            attribution={"metric": "wastage", "root_cause": "wastage_analysis",
                         "elevated_count": len(high_waste), "total_waste_cost": round(total_waste_cost, 2)},
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
