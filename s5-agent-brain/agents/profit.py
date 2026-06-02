# Profit Agent - revenue + cost + margin from MySQL
# Phase 3: dynamic confidence based on query results.
# Phase 4: per-product filtering to avoid total-store data for product-specific queries.
import logging, sys, os
from typing import Dict, Any
from .base import BaseAgent

_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

logger = logging.getLogger("s5.agent.profit")


class ProfitAgent(BaseAgent):
    def __init__(self):
        super().__init__("profit")
        self._fetch_ok = False

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._fetch_ok = False
        try:
            from db.mysql_client import get_db
            db = get_db()
            cur = db.cursor()

            product = params.get("product", "all")
            target_products = None
            if product and product not in ("all", "-"):
                target_products = set(p.strip() for p in product.split(","))

            # Total revenue from transactions (filtered by product if needed)
            if target_products:
                placeholders = ",".join(["%s"] * len(target_products))
                cur.execute(
                    f"SELECT COALESCE(SUM(quantity * unit_price), 0) FROM inventory_transactions WHERE transaction_type = 'outflow' AND product_name IN ({placeholders})",
                    tuple(target_products))
            else:
                cur.execute("SELECT COALESCE(SUM(quantity * unit_price), 0) FROM inventory_transactions WHERE transaction_type = 'outflow'")
            total_revenue = float(cur.fetchone()[0] or 0)

            # Total cost from products cost_price x transactions (filtered)
            if target_products:
                placeholders = ",".join(["%s"] * len(target_products))
                cur.execute(f"""
                    SELECT COALESCE(SUM(t.quantity * COALESCE(p.cost_price, 3.0)), 0)
                    FROM inventory_transactions t
                    LEFT JOIN products p ON t.product_name = p.product_name
                    WHERE t.transaction_type = 'outflow' AND t.product_name IN ({placeholders})
                """, tuple(target_products))
            else:
                cur.execute("""
                    SELECT COALESCE(SUM(t.quantity * COALESCE(p.cost_price, 3.0)), 0)
                    FROM inventory_transactions t
                    LEFT JOIN products p ON t.product_name = p.product_name
                    WHERE t.transaction_type = 'outflow'
                """)
            total_cost = float(cur.fetchone()[0] or 0)

            # Per-product breakdown (filtered)
            if target_products:
                placeholders = ",".join(["%s"] * len(target_products))
                cur.execute(
                    f"SELECT product_name, SUM(quantity) FROM inventory_transactions WHERE transaction_type = 'outflow' AND product_name IN ({placeholders}) GROUP BY product_name",
                    tuple(target_products))
            else:
                cur.execute("SELECT product_name, SUM(quantity) FROM inventory_transactions WHERE transaction_type = 'outflow' GROUP BY product_name")
            by_product = {}
            for row in cur.fetchall():
                by_product[row[0]] = int(row[1] or 0)

            # Per-product margin: revenue - cost per product
            if target_products:
                placeholders = ",".join(["%s"] * len(target_products))
                cur.execute(f"""
                    SELECT t.product_name,
                           COALESCE(SUM(t.quantity * t.unit_price), 0),
                           COALESCE(SUM(t.quantity * COALESCE(p.cost_price, 3.0)), 0)
                    FROM inventory_transactions t
                    LEFT JOIN products p ON t.product_name = p.product_name
                    WHERE t.transaction_type = 'outflow' AND t.product_name IN ({placeholders})
                    GROUP BY t.product_name""", tuple(target_products))
            else:
                cur.execute("""
                    SELECT t.product_name,
                           COALESCE(SUM(t.quantity * t.unit_price), 0),
                           COALESCE(SUM(t.quantity * COALESCE(p.cost_price, 3.0)), 0)
                    FROM inventory_transactions t
                    LEFT JOIN products p ON t.product_name = p.product_name
                    WHERE t.transaction_type = 'outflow'
                    GROUP BY t.product_name""")
            by_product_margin = {}
            for row in cur.fetchall():
                pname = row[0]
                rev = float(row[1] or 0)
                cost = float(row[2] or 0)
                margin = round((rev - cost) / max(rev, 1), 2)
                by_product_margin[pname] = margin

            cur.close()
            self._fetch_ok = True
            return {
                "total_revenue": total_revenue,
                "total_cost": total_cost,
                "by_product": by_product,
                "by_product_margin": by_product_margin,
                "target_products": list(target_products) if target_products else ["all"],
            }
        except Exception as e:
            logger.warning("Profit fetch failed: %s", e)
            return {"total_revenue": 0, "total_cost": 0, "by_product": {}}

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any],
                history: str = "", key_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        total_revenue = raw.get("total_revenue", 0)
        total_cost = raw.get("total_cost", 0)
        gross_profit = total_revenue - total_cost
        margin_pct = round((gross_profit / max(total_revenue, 1)) * 100, 1)

        # Dynamic confidence
        if self._fetch_ok and total_revenue > 0:
            confidence = 0.90
        elif self._fetch_ok:
            confidence = 0.50
        else:
            confidence = 0.05

        # Per-product breakdown for opinion
        by_product = raw.get("by_product", {})
        product = params.get("product", "all")
        if product not in ("all", "-") and by_product:
            detail_parts = []
            for pname, qty in sorted(by_product.items()):
                detail_parts.append(f"{pname}: {qty} sold")
            detail_str = "; ".join(detail_parts)
            opinion = f"Revenue RM{total_revenue:.0f}, Cost RM{total_cost:.0f}, Profit RM{gross_profit:.0f} ({margin_pct}% margin) [{detail_str}]"
        else:
            opinion = f"Revenue RM{total_revenue:.0f}, Cost RM{total_cost:.0f}, Profit RM{gross_profit:.0f} ({margin_pct}% margin)"

        return {
            "opinion": opinion,
            "confidence": round(confidence, 2),
            "constraints": [],
            "data": {
                "total_revenue": total_revenue,
                "total_cost": total_cost,
                "gross_profit": gross_profit,
                "margin_pct": margin_pct,
                "by_product": raw.get("by_product", {}),
            },
        }
