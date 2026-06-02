# Inventory Agent - stock levels + freshness + waste risk
# Phase 3: dynamic confidence based on API health and data completeness.
import httpx, logging
from typing import Dict, Any
from .base import BaseAgent
from s5_config.settings import S1_INVENTORY_URL, PRODUCT_NAMES

logger = logging.getLogger("s5.agent.inventory")


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

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any],
                history: str = "", key_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        product = params.get("product", "croissant")
        inventory_list = raw.get("inventory", [])
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

        for item in inventory_list:
            pname = item.get("product_name", "unknown")
            if target_products is None or pname in target_products:
                pqty = item.get("total_quantity", 0)
                pbatches = item.get("batches", 0)
                total_qty += pqty
                per_product[pname] = {"qty": pqty, "batches": pbatches}
                if pbatches <= 1:
                    freshness_counts["Fresh"] += pqty
                else:
                    freshness_counts["Fresh"] += max(0, pqty // 2)
                    freshness_counts["Day-1"] += pqty - max(0, pqty // 2)

        fresh = freshness_counts.get("Fresh", 0)
        day1 = freshness_counts.get("Day-1", 0)
        waste_risk = "high" if (total_qty > 60 and fresh < 10) else "low" if day1 == 0 else "medium"

        # Dynamic confidence
        has_data = len(inventory_list) > 0
        matched = len(per_product) > 0
        expected = len(target_products) if target_products else len(PRODUCT_NAMES)
        data_ratio = min(1.0, len(per_product) / max(expected, 1)) if matched else 0

        if self._fetch_ok and matched and data_ratio >= 0.8:
            confidence = 0.85 + 0.10 * data_ratio  # 0.85-0.95
        elif self._fetch_ok and matched:
            confidence = 0.70
        elif self._fetch_ok:
            confidence = 0.40  # API ok but no matching products
        else:
            confidence = 0.10  # API failed

        if product == "all" and per_product:
            details = ", ".join(f"{k}:{v['qty']}" for k, v in sorted(per_product.items()))
            opinion = f"Stock {total_qty} (fresh={fresh}, day-1={day1}) across {len(per_product)} products [{details}]"
        elif matched:
            opinion = f"Stock {total_qty} (fresh={fresh}, day-1={day1}), waste_risk={waste_risk}"
        else:
            opinion = f"No stock data for {product}"

        constraints = []
        if total_qty == 0 and matched:
            constraints.append("no stock at all - emergency restock needed")

        return {
            "opinion": opinion, "confidence": round(confidence, 2), "constraints": constraints,
            "data": {
                "inventory": total_qty, "fresh": fresh, "day1_available": day1,
                "waste_risk": waste_risk, "freshness_breakdown": freshness_counts,
                "per_product": per_product,
            },
        }