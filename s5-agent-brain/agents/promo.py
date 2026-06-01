# Promo Agent — dynamic discount optimization + bundle suggestions
# Computes optimal discount based on overstock severity + freshness + combo potential
import logging
from typing import Dict, Any
from .base import BaseAgent

logger = logging.getLogger("s5.agent.promo")

COFFEE_PRODUCTS = ["latte", "americano", "cappuccino", "cold_brew", "iced_americano", "mocha"]

class PromoAgent(BaseAgent):
    def __init__(self):
        super().__init__("promo")

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        inventory_data = params.get("_inventory", {})
        demand_data = params.get("_demand", {})

        stock = inventory_data.get("inventory", 0)
        day1 = inventory_data.get("day1_available", 0)
        forecast = demand_data.get("forecast", 0)

        suggestions = []
        constraints = []
        discount_pct = 0

        # Day-1 discount: severity-based (only if day-1 stock exists)
        if day1 > 0 and forecast > 0:
            overstock_ratio = stock / max(forecast, 1)
            if overstock_ratio > 1.5:
                discount_pct = min(50, int(20 + (overstock_ratio - 1.5) * 40))
                suggestions.append(
                    f"{discount_pct}% off day-1 ({day1} units, {overstock_ratio:.1f}x overstock)"
                )
            elif overstock_ratio > 1.2:
                discount_pct = 25
                suggestions.append(
                    f"{discount_pct}% off day-1 ({day1} units, moderately overstocked)"
                )
            else:
                discount_pct = 10
                suggestions.append(
                    f"{discount_pct}% off day-1 ({day1} units, near forecast)"
                )

        # Bundle combo: pair overstock bakery with coffee
        surplus = max(0, stock - forecast)
        if surplus > 5:
            for coffee in COFFEE_PRODUCTS[:3]:
                suggestions.append(
                    f"Bundle: bakery + {coffee} combo — clear {surplus} surplus units"
                )

        # No action needed
        if not suggestions:
            suggestions.append("No discount needed — stock matches demand")

        return {
            "opinion": "; ".join(suggestions),
            "confidence": 0.7,
            "constraints": constraints,
            "data": {
                "discount_pct": discount_pct,
                "surplus": surplus,
                "suggestions": suggestions,
            },
        }
