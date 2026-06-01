# Arbitrator  cross-agent health audit + final decision
# Merges Health Agent and Arbitrator into one.
import logging, time
from typing import Dict, Any, List

logger = logging.getLogger("s5.arbitrator")

class Arbitrator:
    """
    Receives outputs from all agents, performs health audit (conflict detection),
    and produces the final decision with reasoning trace.
    """

    def __init__(self):
        self.agent_results: Dict[str, Dict[str, Any]] = {}

    def audit(self, results: Dict[str, Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Health audit: cross-check all agents for conflicts."""
        conflicts = []
        warnings = []

        # Extract key values
        demand = results.get("demand", {}).get("data", {})
        inventory = results.get("inventory", {}).get("data", {})
        production = results.get("production", {}).get("data", {})
        staffing = results.get("staffing", {}).get("data", {})
        promo = results.get("promo", {}).get("data", {})

        forecast = demand.get("forecast", 0)
        stock = inventory.get("inventory", 0)
        max_cap = production.get("max_capacity", 0)
        bakers = staffing.get("bakers", 0)
        waste_risk = inventory.get("waste_risk", "low")

        # Conflict 1: Demand vs Inventory
        if stock == 0:
            conflicts.append("STOCKOUT: zero inventory, cannot meet any demand")
        elif forecast > stock * 1.5:
            conflicts.append(f"UNDERSTOCK: demand ({forecast}) >> stock ({stock}), will stock out")

        # Conflict 2: Demand vs Production capacity
        if max_cap > 0 and forecast > max_cap:
            conflicts.append(f"CAPACITY_GAP: demand ({forecast}) exceeds max production ({max_cap})")

        # Conflict 3: Staffing gap
        if bakers == 0 and forecast > 0:
            conflicts.append("NO_BAKERS: production needed but no bakers scheduled")

        # Warning: waste risk
        if waste_risk == "high":
            warnings.append(f"WASTE_RISK: high waste risk  {stock} inventory, only {forecast} demand")

        # Warning: overstock
        if stock > forecast * 2 and forecast > 0:
            warnings.append(f"OVERSTOCK: {stock} units vs {forecast} forecast ({stock/forecast:.1f}x)")

        # Collect all agent constraints
        all_constraints = []
        for name, r in results.items():
            all_constraints.extend(r.get("constraints", []))

        return {
            "conflicts": conflicts,
            "warnings": warnings,
            "all_constraints": all_constraints,
            "healthy": len(conflicts) == 0,
        }

    def decide(self, results: Dict[str, Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Final decision synthesizing all agent opinions + audit results."""
        t0 = time.perf_counter()

        audit = self.audit(results, params)

        demand = results.get("demand", {}).get("data", {})
        production = results.get("production", {}).get("data", {})
        inventory = results.get("inventory", {}).get("data", {})
        promo = results.get("promo", {}).get("data", {})

        forecast = demand.get("forecast", 0)
        recommended = production.get("recommended", 0)
        max_cap = production.get("max_capacity", 0)
        stock = inventory.get("inventory", 0)
        day1 = inventory.get("day1_available", 0)

        # Decision logic
        if audit["healthy"]:
            action = f"Bake {recommended} units (match forecast {forecast}, within capacity {max_cap})"
            priority = "normal"
        elif any("STOCKOUT" in c for c in audit["conflicts"]):
            action = f"URGENT: only {stock} in stock for {forecast} demand. {(forecast - stock)} unit gap."
            priority = "critical"
        elif any("CAPACITY_GAP" in c for c in audit["conflicts"]):
            action = f"Capped: demand {forecast} but max capacity {max_cap}. Bake {recommended}. Gap {(forecast - recommended)} units."
            priority = "warning"
        elif any("NO_BAKERS" in c for c in audit["conflicts"]):
            action = f"Cannot produce: no bakers scheduled. Demand {forecast} unmet."
            priority = "critical"
        else:
            action = f"Bake {recommended} units. {len(audit['warnings'])} warning(s) to review."
            priority = "normal"

        # Build reasoning trace
        trace = []
        for name, r in results.items():
            opinion = r.get("opinion", "")
            confidence = r.get("confidence", 0)
            constraints = r.get("constraints", [])
            trace.append({
                "agent": name,
                "opinion": opinion,
                "confidence": confidence,
                "constraints": constraints,
            })

        return {
            "action": action,
            "priority": priority,
            "reasoning_trace": trace,
            "audit": audit,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
