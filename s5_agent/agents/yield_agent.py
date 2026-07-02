import os, sys, logging, json, urllib.request
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.yield")

YIELD_SQL = """
SELECT 
    pr.material_name,
    SUM(oi.quantity * pr.quantity_per_unit) AS total_consumed,
    mi.current_stock,
    mi.unit,
    mi.threshold
FROM order_items oi
JOIN orders o ON oi.order_id = o.id
JOIN product_recipes pr ON oi.product_name = pr.product_name
LEFT JOIN material_inventory mi ON pr.material_name = mi.material_name
WHERE o.order_date = %s
  AND o.state != 'refunded'
GROUP BY pr.material_name, mi.current_stock, mi.unit, mi.threshold
ORDER BY total_consumed DESC
"""

PRODUCT_COUNT_SQL = """
SELECT COUNT(DISTINCT oi.product_name) as product_count,
       SUM(oi.quantity) as total_units
FROM order_items oi
JOIN orders o ON oi.order_id = o.id
WHERE o.order_date = %s
  AND o.state != 'refunded'
"""

class YieldAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_actual_bake", description="Get actual production from orders + recipes",
            parameters={"date": "string"}, primary=True, _handler=self._get_bake))
        self.tools.register(Tool(name="get_planned_bake", description="Get planned production",
            parameters={"date": "string"}, primary=False, _handler=self._get_planned))

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

    async def _get_planned(self, date: str = ""):
        return await self._get_bake(date)


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

        Y = chr(165)
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
        cursor.close()
        db.close()
        _populate_batch_inventory(date)
        return {
            "materials": materials,
            "product_count": product_count,
            "total_units": total_units,
        }
    except Exception as e:
        logger.warning("YIELD_DB: error=%s", e)
    return {"materials": [], "product_count": 0, "total_units": 0, "note": "api_unavailable"}


def _populate_batch_inventory(date):
    try:
        from db.mysql_client import get_db
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM batch_inventory WHERE DATE(production_time) = %s", (date,))
        count = cursor.fetchone()[0]
        if count > 0:
            cursor.close()
            db.close()
            return
        ins_sql = (
            "INSERT INTO batch_inventory "
            "(batch_id, product_name, quantity, production_time, tray_color, freshness_status, quantity_initial, quantity_remaining, sales_area) "
            "SELECT "
            "CONCAT('BTH-', DATE_FORMAT(o.order_date, '%%Y%%m%%d'), '-', oi.product_name) as batch_id, "
            "oi.product_name, "
            "SUM(oi.quantity) as quantity, "
            "TIMESTAMP(o.order_date, '06:00:00') as production_time, "
            "'brown' as tray_color, "
            "'fresh' as freshness_status, "
            "SUM(oi.quantity) as quantity_initial, "
            "0 as quantity_remaining, "
            "'front' as sales_area "
            "FROM order_items oi "
            "JOIN orders o ON oi.order_id = o.id "
            "WHERE o.order_date = %s AND o.state != 'refunded' "
            "GROUP BY oi.product_name "
            "ON DUPLICATE KEY UPDATE quantity = VALUES(quantity)"
        )
        cursor.execute(ins_sql, (date,))
        db.commit()
        logger.info("YIELD: populated batch_inventory for %s, %s rows", date, cursor.rowcount)
        cursor.close()
        db.close()
    except Exception as e:
        logger.warning("YIELD: batch_inventory populate failed: %s", e)