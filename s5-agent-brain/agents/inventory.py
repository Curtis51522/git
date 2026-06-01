# Inventory Agent — stock levels + freshness + waste risk
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
            data = resp.json()
        return data

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        product = params.get("product", "croissant")
        inventory_list = raw.get("inventory", [])
        total_qty = 0
        freshness_counts = {}
        for item in inventory_list:
            if item.get("product_name") == product:
                total_qty += item.get("total_quantity", 0)
                for batch in item.get("batches", []):
                    status = batch.get("freshness_status", "Fresh")
                    qty = batch.get("quantity", 0)
                    freshness_counts[status] = freshness_counts.get(status, 0) + qty

        fresh = freshness_counts.get("Fresh", 0)
        day1 = freshness_counts.get("Day-1", 0)
        waste_risk = "high" if (total_qty > 60 and fresh < 10) else "low" if day1 == 0 else "medium"

        constraints = []
        if total_qty == 0:
            constraints.append("no stock at all — emergency restock needed")

        return {
            "opinion": f"Stock {total_qty} (fresh={fresh}, day-1={day1}), waste_risk={waste_risk}",
            "confidence": 0.95,
            "constraints": constraints,
            "data": {
                "inventory": total_qty,
                "fresh": fresh,
                "day1_available": day1,
                "waste_risk": waste_risk,
                "freshness_breakdown": freshness_counts,
            },
        }
