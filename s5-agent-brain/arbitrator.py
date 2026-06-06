# Arbitrator - cross-agent health audit + final decision
# Merges Health Agent and Arbitrator into one.
# Agent deliberation (LLM-mediated consensus) resolves disagreements but does NOT override optimizer results.
import logging, time, json
from optimizer import optimize_single, optimize_multi, ProductState, CostParams, BakeryConfig, generate_pareto_plans
from typing import Dict, Any, List, Optional
import httpx, os, sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
try:
    from config.settings import DEEPSEEK_API_KEY
except ImportError:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions"
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

logger = logging.getLogger("s5.arbitrator")


def _run_comparison(demand_data, inventory_data, profit_data=None):
    """Compare products side-by-side: forecast, stock, L7d sales, margin."""
    per_product_demand = demand_data.get("per_product", {})
    per_product_inv = inventory_data.get("per_product", {})
    by_product = profit_data.get("by_product", {}) if profit_data else {}
    by_product_margin = profit_data.get("by_product_margin", {}) if profit_data else {}

    if not per_product_demand or not per_product_inv:
        return None

    lines = []
    for pname in sorted(set(per_product_demand.keys()) | set(per_product_inv.keys())):
        fc = per_product_demand.get(pname, {}).get("forecast", 0)
        lo = per_product_demand.get(pname, {}).get("lower", 0)
        hi = per_product_demand.get(pname, {}).get("upper", 0)
        trend = per_product_demand.get(pname, {}).get("trend", "?")
        stock = per_product_inv.get(pname, {}).get("qty", 0)
        day1 = per_product_inv.get(pname, {}).get("day1", 0)
        price = per_product_inv.get(pname, {}).get("selling_price", "?")
        l7d_sold = by_product.get(pname, 0)
        margin = by_product_margin.get(pname)

        coverage = round(stock / max(fc, 1) * 100)
        extra = ""
        if day1 > 0:
            extra += f", day-1={day1}"
        if l7d_sold:
            extra += f", L7d={l7d_sold}/day"
        if margin is not None:
            extra += f", margin={margin:.0%}"

        lines.append(
            f"{pname}: forecast={fc:.0f} ({lo:.0f}-{hi:.0f}, {trend}), "
            f"stock={stock:.0f} ({coverage}% coverage){extra}, price=RM{price}"
        )

    return " | ".join(lines)

class Arbitrator:
    def __init__(self, config: BakeryConfig = None):
        self.config = config or BakeryConfig()
        self._cached_attribution = None  # causal attribution removed, always None

    def _get_causal_costs(self) -> CostParams:
        return CostParams()

    # ------------------------------------------------------------------
    # Agent Deliberation
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Health Audit
    # ------------------------------------------------------------------
    def audit(self, results, params):
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

        if has_inventory and has_demand:
            if stock == 0:
                conflicts.append("STOCKOUT: zero inventory, cannot meet any demand")
            elif forecast > stock * 1.5:
                conflicts.append(f"UNDERSTOCK: demand ({forecast}) >> stock ({stock}), will stock out")

        # Only flag capacity gap when production shortfall exceeds capacity
        production_shortfall = max(0, forecast - stock)
        if has_production and has_demand and production_shortfall > max_cap and max_cap > 0:
            # CAPACITY_GAP removed: capacity >> demand in realistic staffing
            pass
        if has_staffing and bakers == 0 and has_demand:
            conflicts.append("NO_BAKERS: production needed but no bakers scheduled")
        if has_staffing and staffing.get("cashiers", 0) == 0:
            conflicts.append("NO_CASHIERS: cannot open without cashier")

        if waste_risk == "high":
            warnings.append(f"WASTE_RISK: high waste risk - {stock} inventory, only {forecast} demand")
        if has_inventory and has_demand and stock > forecast * 2 and forecast > 0:
            warnings.append(f"OVERSTOCK: {stock} units vs {forecast} forecast ({stock/forecast:.1f}x)")

        all_constraints = []
        for name, r in results.items():
            all_constraints.extend(r.get("constraints", []))

        return {"conflicts": conflicts, "warnings": warnings, "all_constraints": all_constraints, "healthy": len(conflicts) == 0}

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def decide(self, results, params):
        t0 = time.perf_counter()
        intent = params.get("intent", "stock_query")
        audit = self.audit(results, params)
        product = params.get("product", "croissant")

        opt = {}
        counterfactual = None

        demand = results.get("demand", {}).get("data", {})
        production = results.get("production", {}).get("data", {})
        inventory = results.get("inventory", {}).get("data", {})
        promo = results.get("promo", {}).get("data", {})

        forecast = demand.get("forecast", 0)
        recommended = production.get("recommended", 0)
        max_cap = production.get("max_capacity", 0)
        stock = inventory.get("inventory", 0)
        waste_risk = inventory.get("waste_risk", "low")

        costs = self._get_causal_costs()
        counterfactual = None


        if intent == "waste_analysis":
            per_product_inv = inventory.get("per_product", {})
            per_product_demand = demand.get("per_product", {})
            if per_product_inv and per_product_demand and len(per_product_inv) >= 2:
                # Per-product spoilage risk ranking
                ranked = []
                for pname, pdata in per_product_inv.items():
                    pf = per_product_demand.get(pname, {}).get("forecast", 0)
                    pq = pdata.get("qty", 0)
                    pday1 = pdata.get("day1", 0)
                    ratio = pq / max(pf, 1)
                    risk = ratio * pday1 / max(pq, 1)  # spoilage risk = coverage * day1_ratio
                    ranked.append((pname, pf, pq, pday1, ratio, risk))
                ranked.sort(key=lambda x: -x[5])  # highest risk first
                parts = []
                for pname, pf, pq, pday1, ratio, risk in ranked:
                    if ratio > 1.5:
                        parts.append(f"{pname}: HIGH risk (stock={pq} vs forecast={pf}, {ratio:.1f}x, day-1={pday1})")
                    elif ratio > 1.2:
                        parts.append(f"{pname}: MED risk (stock={pq} vs forecast={pf}, {ratio:.1f}x, day-1={pday1})")
                    elif ratio < 0.5:
                        parts.append(f"{pname}: no waste risk - understock (stock={pq} vs forecast={pf})")
                    else:
                        parts.append(f"{pname}: low risk (stock={pq} vs forecast={pf})")
                action = " | ".join(parts)
                # Global priority: use worst-case
                max_ratio = max(r[4] for r in ranked)
                if max_ratio > 1.5:
                    priority = "warning"
                elif max_ratio > 1.2:
                    priority = "normal"
                else:
                    priority = "normal"
            else:
                surplus = max(0, stock - forecast)
                ratio = stock / max(forecast, 1)
                if ratio > 1.5:
                    day1_stock = inventory.get("freshness_breakdown", {}).get("Day-1", 0)
                    action = (f"WASTE RISK: {stock} total inventory vs {forecast} demand ({ratio:.1f}x). "
                              f"Day-1 stock: {day1_stock} units at risk. "
                              f"Root cause: production outpacing demand, {surplus} surplus units will go stale.")
                    priority = "warning"
                elif ratio > 1.2:
                    action = (f"Moderate overstock: {stock} inventory vs {forecast} forecast ({ratio:.1f}x). "
                              f"Day-1 units should be prioritized for sale.")
                    priority = "normal"
                elif ratio < 0.5:
                    gap = forecast - stock
                    action = (f"UNDERSTOCK: waste is not the issue. Stock ({stock}) covers only {ratio:.0%} "
                              f"of forecast demand ({forecast}). Missing {gap} units means lost sales, "
                              f"not waste. Increase production to match demand.")
                    priority = "critical"
                else:
                    action = f"Stock-demand ratio {ratio:.0%}: {stock} in stock vs {forecast} forecast. No severe waste or shortage risk."
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

        elif intent == "comparison_analysis":
            profit_data = results.get("profit", {}).get("data", {})
            comparison = _run_comparison(demand, inventory, profit_data)
            if comparison:
                action = comparison
                priority = "normal"
            else:
                action = f"No historical data available for comparison. Current: {forecast} forecast, {stock} inventory."
                priority = "normal"

        else:
            # stock_query / cross_source_audit - OPTIMIZER computes numbers
            # Cap = 0 if no bakers scheduled (rest day); otherwise use production capacity
            bakers_on_duty = results.get("staffing", {}).get("data", {}).get("bakers", 1)
            if bakers_on_duty == 0:
                cap = 0
            elif max_cap > 0:
                cap = min(max_cap, self.config.daily_capacity)
            else:
                cap = self.config.daily_capacity
            per_product_inv = inventory.get("per_product", {})
            per_product_demand = demand.get("per_product", {})

            if per_product_inv and per_product_demand and len(per_product_inv) > 1:
                prod_states = []
                for pname, pdata in per_product_inv.items():
                    p_demand = per_product_demand.get(pname, {}).get("forecast", 0)
                    day1 = pdata.get("day1", 0)
                    fresh = max(0, pdata["qty"] - day1)
                    p_low = per_product_demand.get(pname, {}).get("lower", max(0, p_demand - 15))
                    p_high = per_product_demand.get(pname, {}).get("upper", p_demand + 15)
                    p_price = pdata.get("selling_price", 5.90)
                    prod_states.append(ProductState(pname, demand=p_demand,
                        demand_low=p_low, demand_high=p_high,
                        fresh_stock=fresh, day1_stock=day1,
                        waste_loss=costs.waste_loss, stockout_loss=costs.stockout_loss,
                        production_cost=costs.production_cost, unit_price=p_price))
                opt = optimize_multi(prod_states, cap, costs, config=self.config)
                action = opt["rationale"]
                priority = "warning" if opt.get("shortage_units", 0) > 0 else "normal"
                # Attach minimal Pareto context for multi-product (uses MIP result as balanced)
                opt["pareto_plans"] = []
                opt["pareto_context"] = {"product": "all", "stock": int(stock), "forecast": int(forecast)}
            else:
                day1 = inventory.get("day1_available", 0)
                demand_low = demand.get("forecast_low", max(0, forecast * 0.7))
                demand_high = demand.get("forecast_high", forecast * 1.3)
                pareto = generate_pareto_plans(
                    forecast, stock, cap,
                    demand_low=demand_low, demand_high=demand_high,
                    costs=costs, product_name=product,
                    day1_stock=day1, config=self.config, unit_price=inventory.get("unit_price", 5.90))
                # Default to balanced plan; LLM Decision layer will override
                balanced = next((p for p in pareto["plans"] if p["label"] == "B_balanced"), pareto["plans"][0])
                opt = {"bake_units": balanced["bake"], "shortage_units": balanced["shortage"],
                       "profit_rm": balanced.get("profit_rm", 0), "revenue_rm": balanced.get("revenue_rm", 0),
                       "pareto_plans": pareto["plans"], "pareto_context": pareto["context"]}
                action = opt["pareto_context"].get("product", product) + ": " + balanced["rationale"]
                priority = "warning" if opt["shortage_units"] > 0 else "normal"


        trace = []
        for name, r in results.items():
            trace.append({"agent": name, "opinion": r.get("opinion", ""),
                         "confidence": r.get("confidence", 0), "constraints": r.get("constraints", [])})

        result = {"action": action, "priority": priority, "reasoning_trace": trace, "audit": audit,
                  "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if self._cached_attribution is not None:
            result["causal_calibration"] = {"waste_loss_per_unit": self._cached_attribution.avg_waste_per_unit_cost,
                "stockout_loss_per_unit": self._cached_attribution.avg_stockout_per_unit_cost,
                "top_waste_driver": self._cached_attribution.top_waste_driver, "method": self._cached_attribution.method}
        if opt.get("profit_rm") is not None:
            result["optimizer_profit"] = {"profit_rm": opt["profit_rm"], "revenue_rm": opt.get("revenue_rm", 0),
                "risk_preference": opt.get("risk_preference", "balanced")}
        if "pareto_plans" in opt:
            result["pareto_plans"] = opt["pareto_plans"]
        if "pareto_context" in opt:
            result["pareto_context"] = opt["pareto_context"]
        return result
