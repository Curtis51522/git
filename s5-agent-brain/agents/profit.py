# Profit Agent - revenue, cost, margin calculations
import httpx, logging
from typing import Dict, Any
from .base import BaseAgent

logger = logging.getLogger("s5.agent.profit")


class ProfitAgent(BaseAgent):
    def __init__(self):
        super().__init__("profit")

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:8002/s1/inventory_transactions?limit=5000", timeout=10)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning("Profit fetch failed: %s", e)
            return {"transactions": []}

    def _get_product_costs(self) -> Dict[str, float]:
        """Query product costs from MySQL directly."""
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
            from db.mysql_client import get_db
            db = get_db()
            cur = db.cursor()
            cur.execute("SELECT product_name, cost_price FROM products WHERE category='bakery'")
            rows = cur.fetchall()
            cur.close()
            return {r[0]: float(r[1]) for r in rows}
        except Exception as e:
            logger.warning("Failed to fetch product costs: %s", e)
            return {}

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        transactions = raw.get("transactions", [])
        cost_map = self._get_product_costs()

        total_revenue = 0.0
        total_cost = 0.0
        by_product = {}

        for t in transactions:
            pname = t.get("product_name", "unknown")
            qty = t.get("quantity", 0) or 1
            price = float(t.get("unit_price", 0) or 0)
            if price <= 0:
                continue
            unit_cost = cost_map.get(pname, 0)
            revenue = qty * price
            cost_total = qty * unit_cost
            total_revenue += revenue
            total_cost += cost_total

            if pname not in by_product:
                by_product[pname] = {"revenue": 0, "cost": 0, "qty": 0}
            by_product[pname]["revenue"] += revenue
            by_product[pname]["cost"] += cost_total
            by_product[pname]["qty"] += qty

        gross_profit = total_revenue - total_cost
        margin_pct = round((gross_profit / total_revenue * 100), 1) if total_revenue > 0 else 0

        return {
            "opinion": f"Revenue RM{total_revenue:.0f}, Cost RM{total_cost:.0f}, Profit RM{gross_profit:.0f} ({margin_pct}% margin)",
            "confidence": 0.9,
            "constraints": [],
            "data": {
                "total_revenue": total_revenue,
                "total_cost": total_cost,
                "gross_profit": gross_profit,
                "margin_pct": margin_pct,
                "by_product": by_product,
            },
        }
