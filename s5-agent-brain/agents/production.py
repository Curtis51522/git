# Production Agent - capacity constraints + baking recommendation
# Phase 4: oven-physics-based capacity with baker-per-oven constraint.
import httpx, logging, sys, os
from typing import Dict, Any
from .base import BaseAgent

_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

logger = logging.getLogger("s5.agent.production")


class ProductionAgent(BaseAgent):
    def __init__(self):
        super().__init__("production")
        self._cfg = None

    def _get_config(self):
        if self._cfg is not None:
            return self._cfg
        try:
            from optimizer import BakeryConfig
            self._cfg = BakeryConfig()
        except Exception:
            self._cfg = type("Cfg", (), {
                "oven_layers": 2, "oven_count": 2, "capacity_per_layer": 12,
                "baking_time_min": 18, "baking_window_hours": 4.5,
                "max_units_per_hour": 160, "batch_size": 1,
            })()
        return self._cfg

    def _oven_rate_per_oven(self):
        cfg = self._get_config()
        return cfg.max_units_per_hour / max(cfg.oven_count, 1)

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any],
                history: str = "", key_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        demand_data = params.get("_demand", {})
        staffing_data = params.get("_staffing", {})

        forecast = demand_data.get("forecast", 0)
        baker_count = staffing_data.get("bakers", 1)
        baker_hours = staffing_data.get("baker_hours", 7)
        cfg = self._get_config()

        # 1 baker operates 1 oven; capacity is per-oven rate
        ovens_used = min(baker_count, cfg.oven_count)
        per_oven_rate = self._oven_rate_per_oven()
        effective_hours = min(baker_hours, cfg.baking_window_hours)
        max_capacity = int(ovens_used * effective_hours * per_oven_rate)
        capped = forecast > max_capacity

        has_demand = "forecast" in demand_data and forecast >= 0
        has_staffing = "bakers" in staffing_data
        if has_demand and has_staffing and baker_count > 0:
            confidence = 0.85
        elif has_demand and has_staffing:
            confidence = 0.60
        elif has_demand:
            confidence = 0.50
        else:
            confidence = 0.20

        constraints = []
        if capped:
            constraints.append(f"demand {forecast} exceeds capacity {max_capacity} - capped")
        if baker_count == 0:
            constraints.append("no bakers available")

        recommended = min(forecast, max_capacity) if forecast > 0 else 0

        return {
            "opinion": f"Capacity {max_capacity} ({baker_count} bakers x {ovens_used} ovens, {effective_hours:.1f}h eff, {per_oven_rate:.0f}/hr/oven, window={cfg.baking_window_hours:.1f}h), bake {recommended}",
            "confidence": round(confidence, 2),
            "constraints": constraints,
            "data": {
                "max_capacity": max_capacity, "recommended": recommended,
                "bakers": baker_count, "baker_hours": baker_hours,
                "ovens_used": ovens_used, "oven_rate": per_oven_rate,
                "is_capped": capped, "effective_hours": effective_hours,
            },
        }
