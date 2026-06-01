# Inventory Agent - stock levels + freshness + waste risk
import httpx, logging
from typing import Dict, Any
from .base import BaseAgent
from s5_config.settings import S1_INVENTORY_URL, PRODUCT_NAMES

logger = logging.getLogger("s5.agent.inventory")


class InventoryAgent(BaseAgent):
    def __init__(self):
        super().__init__("inventory")

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(S1_INVENTORY_URL, timeout=10)
            resp.raise_for_status()
            return resp.json()

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        product = params.get("product", "croissant")
        inventory_list = raw.get("inventory", [])
        total_qty = 0
        freshness_counts = {"Fresh": 0, "Day-1": 0}
        per_product = {}

        for item in inventory_list:
            if product == "all" or item.get("product_name") == product:
                pname = item.get("product_name", "unknown")
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

        if product == "all" and per_product:
            details = ", ".join(f"{k}:{v['qty']}" for k, v in sorted(per_product.items()))
            opinion = f"Stock {total_qty} (fresh={fresh}, day-1={day1}) across {len(per_product)} products [{details}]"
        else:
            opinion = f"Stock {total_qty} (fresh={fresh}, day-1={day1}), waste_risk={waste_risk}"

        constraints = []
        if total_qty == 0:
            constraints.append("no stock at all - emergency restock needed")

        return {
            "opinion": opinion,
            "confidence": 0.95,
            "constraints": constraints,
            "data": {
                "inventory": total_qty,
                "fresh": fresh,
                "day1_available": day1,
                "waste_risk": waste_risk,
                "freshness_breakdown": freshness_counts,
                "per_product": per_product,
            },
        }
