# Inventory Agent - stock levels + freshness + waste risk
# Phase 4: direct DB query for authoritative freshness data (no heuristic guessing).
import httpx, logging
from typing import Dict, Any
from .base import BaseAgent
from s5_agent.s5_config.settings import S1_INVENTORY_URL, THRESHOLDS

logger = logging.getLogger("s5.agent.inventory")


def _format_opinion(total_qty, fresh, day1, waste_risk, per_product, params, product_str):
    """Format inventory opinion. Per-product for comparison, aggregated otherwise."""
    intent = params.get("intent", "")
    if intent == "comparison_analysis" and len(per_product) >= 2:
        parts = []
        for pname, pdata in sorted(per_product.items()):
            parts.append(f"{pname}: stock={pdata['qty']} (fresh={pdata['fresh']}, day-1={pdata['day1']})")
        return " | ".join(parts) + f", waste_risk={waste_risk}"
    if product_str == "all" and per_product:
        details = ", ".join(f"{k}:{v['qty']}" for k, v in sorted(per_product.items()))
        return f"Stock {total_qty} (fresh={fresh}, day-1={day1}) across {len(per_product)} products [{details}]"
    return f"Stock {total_qty} (fresh={fresh}, day-1={day1}), waste_risk={waste_risk}"


class InventoryAgent(BaseAgent):
    def __init__(self):
        super().__init__("inventory")
        self._fetch_ok = False

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._fetch_ok = False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(S1_INVENTORY_URL, timeout=10)
                resp.raise_for_status()
                self._fetch_ok = True
                return resp.json()
        except Exception as e:
            logger.warning("Inventory fetch failed: %s", e)
            return {}

    def _query_db_freshness(self, product_filter=None):
        """Query batch_inventory directly for authoritative freshness data."""
        try:
            from memory_store import _get_db
            db = _get_db()
            cur = db.cursor()
            if product_filter:
                placeholders = ",".join(["%s"] * len(product_filter))
                cur.execute(
                    f"SELECT product_name, freshness_status, SUM(quantity) as total "
                    f"FROM batch_inventory WHERE product_name IN ({placeholders}) "
                    f"GROUP BY product_name, freshness_status", list(product_filter))
            else:
                cur.execute(
                    "SELECT product_name, freshness_status, SUM(quantity) as total "
                    "FROM batch_inventory GROUP BY product_name, freshness_status")
            rows = cur.fetchall()
            cur.close()
            result = {}
            for pname, status, qty in rows:
                if pname not in result:
                    result[pname] = {"Fresh": 0, "Day-1": 0, "qty": 0}
                status_clean = status.strip()
                result[pname][status_clean] = int(qty)
                result[pname]["qty"] += int(qty)
            # Fetch selling prices
            cur = db.cursor()
            cur.execute("SELECT product_name, selling_price FROM products")
            prices = {r[0]: r[1] for r in cur.fetchall()}
            cur.close()
            for pname in result:
                result[pname]["selling_price"] = prices.get(pname, THRESHOLDS["inventory_default_price"])
            return result
        except Exception as e:
            logger.warning("DB freshness query failed: %s", e)
            return None

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any],
                history: str = "", key_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        product = params.get("product", "croissant")
        total_qty = 0
        freshness_counts = {"Fresh": 0, "Day-1": 0}
        per_product = {}

        target_products = None
        if "," in product:
            target_products = set(p.strip() for p in product.split(","))
        elif product == "all":
            target_products = None
        else:
            target_products = {product}

        # Primary: direct DB query for authoritative freshness data
        db_data = self._query_db_freshness(target_products)
        if db_data:
            for pname, pdata in db_data.items():
                total_qty += pdata["qty"]
                freshness_counts["Fresh"] += pdata.get("Fresh", 0)
                freshness_counts["Day-1"] += pdata.get("Day-1", 0)
                per_product[pname] = {
                    "qty": pdata["qty"],
                    "fresh": pdata.get("Fresh", 0),
                    "day1": pdata.get("Day-1", 0),
                    "selling_price": pdata.get("selling_price", THRESHOLDS["inventory_default_price"]),
                }
        else:
            # Fallback: S1 API + heuristic guess (legacy)
            inventory_list = raw.get("inventory", [])
            for item in inventory_list:
                pname = item.get("product_name", "unknown")
                if target_products is None or pname in target_products:
                    pqty = item.get("total_quantity", 0)
                    pbatches = item.get("batches", 0)
                    total_qty += pqty
                    p_fresh = pqty if pbatches <= 1 else max(0, pqty // 2)
                    p_day1 = 0 if pbatches <= 1 else pqty - p_fresh
                    pselling = item.get("selling_price", THRESHOLDS["inventory_default_price"])
                    per_product[pname] = {"qty": pqty, "batches": pbatches, "fresh": p_fresh, "day1": p_day1, "selling_price": pselling}
                    if pbatches <= 1:
                        freshness_counts["Fresh"] += pqty
                    else:
                        freshness_counts["Fresh"] += max(0, pqty // 2)
                        freshness_counts["Day-1"] += pqty - max(0, pqty // 2)

        fresh = freshness_counts.get("Fresh", 0)
        day1 = freshness_counts.get("Day-1", 0)
        waste_risk = "high" if (total_qty > THRESHOLDS["inventory_total_high"] and fresh < THRESHOLDS["inventory_fresh_low"]) else "low" if day1 == 0 else "medium"

        # Confidence: DB direct query is authoritative (0.95)
        matched = len(per_product) > 0
        if db_data and matched:
            confidence = 0.95
        elif self._fetch_ok and matched:
            confidence = 0.75
        elif matched:
            confidence = 0.50
        else:
            confidence = 0.10

        opinion = _format_opinion(total_qty, fresh, day1, waste_risk, per_product, params, product) if matched else f"No stock data for {product}"

        constraints = []
        if total_qty == 0 and matched:
            constraints.append("no stock at all - emergency restock needed")

        return {
            "opinion": opinion, "confidence": round(confidence, 2), "constraints": constraints,
            "data": {
                "inventory": total_qty, "fresh": fresh, "day1_available": day1,
                "waste_risk": waste_risk, "freshness_breakdown": freshness_counts,
                "per_product": per_product,
                "unit_price": per_product.get(product, {}).get("selling_price", THRESHOLDS["inventory_default_price"]) if target_products and len(target_products) == 1 else 5.90,
            },
        }