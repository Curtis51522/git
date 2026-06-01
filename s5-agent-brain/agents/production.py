# Production Agent — capacity constraints + baking recommendation
# Cross-references Demand forecast with Schedule staffing
import httpx, logging
from typing import Dict, Any
from .base import BaseAgent

logger = logging.getLogger("s5.agent.production")

BAKER_UNITS_PER_HOUR = 15  # conservative: one baker can produce ~15 units/hour

class ProductionAgent(BaseAgent):
    def __init__(self):
        super().__init__("production")

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}  # computes from Demand + Staffing data, no external fetch

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        demand_data = params.get("_demand", {})
        staffing_data = params.get("_staffing", {})

        forecast = demand_data.get("forecast", 0)
        baker_count = staffing_data.get("bakers", 1)
        baker_hours = staffing_data.get("baker_hours", 6)

        max_capacity = baker_count * baker_hours * BAKER_UNITS_PER_HOUR
        recommended = min(forecast, max_capacity)
        capped = forecast > max_capacity

        constraints = []
        if capped:
            constraints.append(f"demand {forecast} exceeds capacity {max_capacity} — capped to {recommended}")

        return {
            "opinion": f"Capacity {max_capacity} ({baker_count} bakers × {baker_hours}h), bake {recommended}",
            "confidence": 0.85,
            "constraints": constraints,
            "data": {
                "max_capacity": max_capacity,
                "recommended": recommended,
                "bakers": baker_count,
                "baker_hours": baker_hours,
                "is_capped": capped,
            },
        }
