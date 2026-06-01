# Arbitrator - cross-agent health audit + final decision
# Merges Health Agent and Arbitrator into one.
import logging, time
from optimizer import optimize_single, optimize_multi, ProductState, CostParams
from typing import Dict, Any, List

logger = logging.getLogger("s5.arbitrator")


class Arbitrator:
    """Receives outputs from active agents, performs health audit (conflict detection),
    and produces the final decision with reasoning trace."""

    def __init__(self):
        pass

    def audit(self, results: Dict[str, Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Health audit: cross-check only agents that actually ran."""
        conflicts = []
        warnings = []

        demand = results.get("demand", {}).get("data", {})
        inventory = results.get("inventory", {}).get("data", {})
        production = results.get("production", {}).get("data", {})
        staffing = results.get("staffing", {}).get("data", {})

        forecast = demand.get("forecast", 0)
        stock = inventory.get("inventory", 0)
        max_cap = production.get("max_capacity", 0)
        bakers = staffing.get("bakers", 0)
        waste_risk = inventory.get("waste_risk", "low")

        has_demand = "demand" in results and forecast > 0
        has_inventory = "inventory" in results
        has_production = "production" in results and max_cap > 0
        has_staffing = "staffing" in results

        # Conflict 1: Demand vs Inventory (only if both present)
        if has_inventory and has_demand:
            if stock == 0:
                conflicts.append("STOCKOUT: zero inventory, cannot meet any demand")
            elif forecast > stock * 1.5:
                conflicts.append(f"UNDERSTOCK: demand ({forecast}) >> stock ({stock}), will stock out")

        # Conflict 2: Demand vs Production capacity
        if has_production and has_demand and forecast > max_cap:
            conflicts.append(f"CAPACITY_GAP: demand ({forecast}) exceeds max production ({max_cap})")

        # Conflict 3: Staffing gap (only if staffing agent ran)
        if has_staffing and bakers == 0 and has_demand:
            conflicts.append("NO_BAKERS: production needed but no bakers scheduled")
        if has_staffing and staffing.get("cashiers", 0) == 0:
            conflicts.append("NO_CASHIERS: cannot open without cashier")

        # Warnings
        if waste_risk == "high":
            warnings.append(f"WASTE_RISK: high waste risk - {stock} inventory, only {forecast} demand")
        if has_inventory and has_demand and stock > forecast * 2 and forecast > 0:
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
        """Final decision: adapts to which agents were active."""
        t0 = time.perf_counter()
        intent = params.get("intent", "stock_query")
        audit = self.audit(results, params)

        demand = results.get("demand", {}).get("data", {})
        production = results.get("production", {}).get("data", {})
        inventory = results.get("inventory", {}).get("data", {})
        promo = results.get("promo", {}).get("data", {})

        forecast = demand.get("forecast", 0)
        recommended = production.get("recommended", 0)
        max_cap = production.get("max_capacity", 0)
        stock = inventory.get("inventory", 0)
        waste_risk = inventory.get("waste_risk", "low")

        # Intent-aware decision
        if intent == "waste_analysis":
            surplus = max(0, stock - forecast)
            ratio = stock / max(forecast, 1)
            if ratio > 1.5:
                products_detail = inventory.get("freshness_breakdown", {})
                fresh_stock = products_detail.get("Fresh", 0)
                day1_stock = products_detail.get("Day-1", 0)
                action = (
                    f"WASTE RISK: {stock} total inventory vs {forecast} demand ({ratio:.1f}x). "
                    f"Day-1 stock: {day1_stock} units at risk. "
                    f"Root cause: production outpacing demand, {surplus} surplus units will go stale."
                )
                priority = "warning"
            elif ratio > 1.2:
                action = (
                    f"Moderate overstock: {stock} inventory vs {forecast} forecast ({ratio:.1f}x). "
                    f"Day-1 units ({inventory.get('day1_available', 0)}) should be prioritized for sale."
                )
                priority = "normal"
            else:
                action = f"Healthy balance: {stock} in stock matches {forecast} demand."
                priority = "normal"

        elif intent == "schedule_audit":
            bakers = results.get("staffing", {}).get("data", {}).get("bakers", 0)
            if audit["healthy"]:
                action = f"Schedule looks good: {bakers} bakers, no anomalies."
                priority = "normal"
            else:
                action = f"Schedule issues: {', '.join(audit['conflicts'])}"
                priority = "warning"

        elif intent == "promo_eval":
            discount = promo.get("discount_pct", 0)
            surplus = promo.get("surplus", 0)
            if surplus > 5:
                action = f"Promo recommended: {discount}% off to clear {surplus} surplus units."
                priority = "normal"
            else:
                action = f"No promo needed: surplus only {surplus} units."
                priority = "normal"

        elif intent == "profit_analysis":
            action = f"Profit check: {forecast} forecast demand at {stock} inventory."
            priority = "normal"

        else:
            # stock_query / cross_source_audit / multi_product
            # Multi-objective optimizer replaces simple if/else
            cap = max_cap if max_cap > 0 else 75
            stocks_val = inventory.get("inventory", 0)
            per_product_inv = inventory.get("per_product", {})
            per_product_demand = demand.get("per_product", {})

            if per_product_inv and per_product_demand and len(per_product_inv) > 1:
                prod_states = []
                for pname, pdata in per_product_inv.items():
                    p_demand = per_product_demand.get(pname, {}).get("forecast", 0)
                    prod_states.append(ProductState(pname, demand=p_demand, stock=pdata["qty"]))
                opt = optimize_multi(prod_states, cap)
                action = opt["rationale"]
                priority = "warning" if opt.get("shortage_units", 0) > 0 else "normal"
            else:
                opt = optimize_single(forecast, stocks_val, cap)
                action = opt["rationale"]
                priority = "warning" if opt["shortage_units"] > 0 else "normal"
        # Reasoning trace
        trace = []
        for name, r in results.items():
            trace.append({
                "agent": name,
                "opinion": r.get("opinion", ""),
                "confidence": r.get("confidence", 0),
                "constraints": r.get("constraints", []),
            })

        return {
            "action": action,
            "priority": priority,
            "reasoning_trace": trace,
            "audit": audit,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }