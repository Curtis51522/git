import logging
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool

logger = logging.getLogger("s5.agent.yield")

YIELD_SQL = """
SELECT
    mt.material_name,
    SUM(mt.quantity) AS total_consumed,
    rm.stock_quantity,
    rm.unit,
    rm.reorder_point
FROM material_transactions mt
LEFT JOIN raw_materials rm ON rm.material_name = mt.material_name
WHERE DATE(mt.created_at) = %s
  AND mt.transaction_type = 'outflow'
  AND mt.reference LIKE 'production:%'
GROUP BY mt.material_name, rm.stock_quantity, rm.unit, rm.reorder_point
ORDER BY total_consumed DESC
"""

PRODUCT_COUNT_SQL = """
SELECT COUNT(DISTINCT it.product_name) AS product_count,
       COALESCE(SUM(it.quantity), 0) AS total_units
FROM inventory_transactions it
JOIN products p ON p.product_name = it.product_name
WHERE DATE(it.transaction_time) = %s
  AND it.transaction_type = 'inflow'
  AND p.category = 'bakery'
"""

class YieldAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_actual_bake", description="Get actual production intake and material usage",
            parameters={"date": "string"}, primary=True, _handler=self._get_bake))

    async def fetch(self, params):
        date = ""
        if isinstance(params, dict):
            date = str(params.get("date", "") or "")
        if not date:
            from datetime import date as _dt
            date = _dt.today().isoformat()
        logger.info("YIELD_FETCH: date=%s", repr(date))
        data = _fetch_yield_data(date)
        return {"success": True, "data": data}

    async def _get_bake(self, date: str = ""):
        return await self.fetch({"date": date})

    @staticmethod
    def _fmt_qty(qty, unit):
        """Format quantity: integer for pcs, 3 decimals for kg/L."""
        if unit == "pcs":
            return f"{qty:.0f}"
        return f"{qty:.3f}"
    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        materials = data.get("materials", [])
        product_count = data.get("product_count", 0)
        total_units = data.get("total_units", 0)

        if not materials:
            return AgentOpinion(agent=self.name,
                opinion="No production data available",
                confidence=0.3,
                attribution={"metric": "yield_rate", "root_cause": "no_data", "deviation": 0})

        total_materials = len(materials)
        top_mat = materials[0]
        top_name = top_mat.get("material_name", "?")
        top_consumed = top_mat.get("total_consumed", 0)
        top_unit = top_mat.get("unit", "kg")

        low_stock_materials = [m for m in materials
                               if m.get("current_stock") is not None
                               and m.get("threshold") is not None
                               and m.get("current_stock", 0) < m.get("threshold", 0)]

        opinion = f"PRODUCTION: {product_count} products baked, {total_units} total units. "
        opinion += f"{total_materials} raw materials consumed. Top material: {top_name} ({self._fmt_qty(top_consumed, top_unit)}{top_unit})."

        recs = []
        if low_stock_materials:
            names = ", ".join(m["material_name"] for m in low_stock_materials[:3])
            opinion += f" LOW STOCK: {names} below reorder threshold."
            recs.append({
                "action": f"Reorder low-stock materials: {names}",
                "urgency": "high",
                "rationale": f"Current stock below threshold after today's consumption"
            })

        return AgentOpinion(agent=self.name,
            opinion=opinion,
            confidence=0.85,
            attribution={"metric": "yield_rate", "root_cause": "normal_yield", "deviation": 0},
            evidence={"materials": materials, "product_count": product_count, "total_units": total_units},
            recommendations=recs)


def _fetch_yield_data(date=""):
    if not date:
        return {"materials": [], "product_count": 0, "total_units": 0, "note": "no_date"}
    db = None
    cursor = None
    try:
        from db.mysql_client import get_db

        db = get_db()
        cursor = db.cursor()
        cursor.execute(YIELD_SQL, (date,))
        materials = []
        for row in cursor.fetchall():
            materials.append({
                "material_name": row[0],
                "total_consumed": float(row[1]) if row[1] else 0,
                "current_stock": float(row[2]) if row[2] else None,
                "unit": row[3] or "kg",
                "threshold": float(row[4]) if row[4] else None,
            })
        cursor.execute(PRODUCT_COUNT_SQL, (date,))
        pc_row = cursor.fetchone()
        product_count = pc_row[0] if pc_row else 0
        total_units = int(pc_row[1]) if pc_row and pc_row[1] else 0
        return {
            "materials": materials,
            "product_count": product_count,
            "total_units": total_units,
        }
    except Exception as e:
        logger.warning("YIELD_DB: error=%s", e)
        return {
            "materials": [],
            "product_count": 0,
            "total_units": 0,
            "note": "api_unavailable",
        }
    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close()
