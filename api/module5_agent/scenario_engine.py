"""Scenario Engine -- B2 What-if Simulator core.

Runs Plan A (baseline), Plan B (user scenario), Plan C (grid-search optimal)
and produces a comparison report with attribution.

All computation is pure Python -- no LLM calls in the engine itself.
LLM is only used downstream by Composer for natural-language summary.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from api.module5_agent.elasticity import get_estimator, ElasticityEstimator

logger = logging.getLogger("s5.scenario")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PlanResult:
    """Result of a single scenario plan."""
    label: str
    production: int
    projected_sales: float
    waste: float
    revenue: float
    discount_cost: float
    profit: float
    capacity_util_pct: float
    params: Dict = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Full Plan A/B/C comparison."""
    plan_a: PlanResult
    plan_b: PlanResult
    plan_c: Optional[PlanResult]
    attribution: Dict
    recommendation: str

# ---------------------------------------------------------------------------
# Product prices & costs (RM) -- synced with DB products table
# ---------------------------------------------------------------------------
PRODUCT_DATA = {
    "donut":               {"price": 4.50, "cost": 1.35},
    "croissant":           {"price": 5.50, "cost": 1.90},
    "bread_coconut":       {"price": 3.50, "cost": 1.05},
    "bread_roll":          {"price": 3.50, "cost": 1.05},
    "chiffon":             {"price": 5.00, "cost": 1.75},
    "croissant_chocolate": {"price": 5.50, "cost": 1.90},
}

def _get_product_price(product: str) -> float:
    return PRODUCT_DATA.get(product, {}).get("price", 3.50)

def _get_product_cost(product: str) -> float:
    return PRODUCT_DATA.get(product, {}).get("cost", 1.05)


class ScenarioEngine:
    """Computes Plan A/B/C for what-if scenarios."""

    def __init__(self, estimator: Optional[ElasticityEstimator] = None):
        self.estimator = estimator or get_estimator()

    # ------------------------------------------------------------------
    def compare(
        self,
        product: str,
        forecast: float,
        inventory: int,
        capacity: int,
        base_price: Optional[float] = None,
        scenario_type: str = "discount",
        adjustments: Optional[Dict] = None,
        forecast_low: Optional[float] = None,
        forecast_high: Optional[float] = None,
    ) -> ComparisonReport:
        """Main entry point. Returns Plan A/B/C comparison."""
        adjustments = adjustments or {}
        price = base_price or _get_product_price(product)
        coeffs = self.estimator.get_coefficients()

        # ---- Plan A: Baseline (current forecast) ----
        plan_a = self._compute_baseline(forecast, inventory, capacity, price, product)

        # ---- Plan B: User scenario ----
        plan_b = self._compute_scenario(
            forecast, inventory, capacity, price,
            scenario_type, adjustments, coeffs, product,
        )

        # ---- Plan C: Optimistic (uses forecast_high when available) ----
        fc_opt = forecast_high if forecast_high is not None else forecast
        plan_c = self._grid_search_optimal(
            fc_opt, inventory, capacity, price,
            scenario_type, coeffs, product,
        )
        if plan_c:
            plan_c.label = "Plan C (Optimistic)" if forecast_high is not None else "Plan C (Optimal)"

        # ---- Sensitivity: worst/best case profit range ----
        sensitivity = {}
        if forecast_low is not None and forecast_high is not None:
            plan_a_low = self._compute_baseline(forecast_low, inventory, capacity, price, product)
            plan_a_high = self._compute_baseline(forecast_high, inventory, capacity, price, product)
            sensitivity = {
                "forecast_range": f"{forecast_low}-{forecast_high}",
                "profit_worst": round(plan_a_low.profit, 2),
                "profit_expected": round(plan_a.profit, 2),
                "profit_best": round(plan_a_high.profit, 2),
                "revenue_worst": round(plan_a_low.revenue, 2),
                "revenue_best": round(plan_a_high.revenue, 2),
                "waste_worst": round(plan_a_low.waste, 1),
                "waste_best": round(plan_a_high.waste, 1),
            }

        # ---- Attribution ----
        attribution = self._attribute(plan_a, plan_b)
        attribution["sensitivity"] = sensitivity

        # ---- Recommendation ----
        best = max(
            [p for p in [plan_a, plan_b, plan_c] if p is not None],
            key=lambda p: p.profit,
        )

        return ComparisonReport(
            plan_a=plan_a,
            plan_b=plan_b,
            plan_c=plan_c,
            attribution=attribution,
            recommendation=f"Plan '{best.label}' yields highest profit (RM{best.profit:.2f}).",
        )

    # ------------------------------------------------------------------
    # Plan A: Baseline
    # ------------------------------------------------------------------
    def _compute_baseline(self, forecast: float, inventory: int,
                          capacity: int, price: float, product: str = "croissant") -> PlanResult:
        needed = max(int(forecast - inventory + 0.5), 0)
        production = min(needed, capacity)
        sales = min(forecast, production + inventory)
        surplus = max(0, production + inventory - forecast)
        revenue = sales * price
        cost = production * _get_product_cost(product)
        # Day-1 salvage: at most 30% of fresh demand can be sold at 90% price
        day1_cap = int(forecast * 0.3)
        salvageable = min(int(surplus), max(day1_cap, 0))
        real_waste = int(surplus) - salvageable
        cost_ratio = _get_product_cost(product) / max(_get_product_price(product), 0.01)
        salvage = salvageable * price * 0.9 * (1 - cost_ratio)
        profit = revenue + salvage - cost
        cap_pct = (production / capacity * 100) if capacity > 0 else 0

        return PlanResult(
            label="Plan A (Current)",
            production=production,
            projected_sales=sales,
            waste=real_waste,
            revenue=round(revenue, 2),
            discount_cost=0,
            profit=round(profit, 2),
            capacity_util_pct=round(cap_pct, 1),
            params={"forecast": forecast, "inventory": inventory, "price": price, "surplus_day1": surplus},
        )

    # ------------------------------------------------------------------
    # Plan B: User scenario
    # ------------------------------------------------------------------
    def _compute_scenario(self, forecast: float, inventory: int,
                          capacity: int, price: float,
                          scenario_type: str, adjustments: Dict,
                          coeffs: Dict, product: str = "croissant") -> PlanResult:
        if scenario_type == "discount":
            return self._scenario_discount(forecast, inventory, capacity, price, adjustments, coeffs, product)
        elif scenario_type == "staffing":
            return self._scenario_staffing(forecast, inventory, capacity, price, adjustments, coeffs, product)
        elif scenario_type == "production":
            return self._scenario_production(forecast, inventory, capacity, price, adjustments, coeffs, product)
        else:
            # Default: treat as production adjustment
            return self._scenario_production(forecast, inventory, capacity, price, adjustments, coeffs, product)

    def _scenario_discount(self, forecast, inventory, capacity, price, adj, coeffs, product: str = "croissant"):
        discount_pct = float(adj.get("discount_pct", 20))
        elasticity = coeffs.get("discount_volume_elasticity", 0.15)
        volume_mult = 1.0 + elasticity * (discount_pct / 10.0)
        projected_sales = forecast * volume_mult
        needed = max(int(projected_sales - inventory + 0.5), 0)
        production = min(needed, capacity)
        sales = min(projected_sales, production + inventory)
        waste = max(0, production + inventory - projected_sales)
        # Revenue: sales at discounted price
        discounted_price = price * (1 - discount_pct / 100)
        revenue = sales * discounted_price
        discount_cost = sales * price * (discount_pct / 100)
        cost = production * _get_product_cost(product)
        profit = revenue - cost

        return PlanResult(
            label=f"Plan B ({discount_pct:.0f}% off)",
            production=production,
            projected_sales=round(sales, 1),
            waste=round(waste, 1),
            revenue=round(revenue, 2),
            discount_cost=round(discount_cost, 2),
            profit=round(profit, 2),
            capacity_util_pct=round(production/capacity*100, 1) if capacity > 0 else 0,
            params={"discount_pct": discount_pct, "volume_mult": round(volume_mult, 2)},
        )

    def _scenario_staffing(self, forecast, inventory, capacity, price, adj, coeffs, product: str = "croissant"):
        headcount_delta = int(adj.get("headcount_delta", 1))
        tph = coeffs.get("staff_throughput_per_headcount", 8.0)
        # More staff means can handle more customers → less waste from slow service
        extra_capacity = headcount_delta * tph * 8  # 8-hour shift
        new_capacity = capacity + extra_capacity
        needed = max(int(forecast - inventory + 0.5), 0)
        production = min(needed, new_capacity)
        sales = min(forecast, production + inventory)
        waste = max(0, production + inventory - forecast)
        revenue = sales * price
        cost = production * _get_product_cost(product)
        profit = revenue - cost

        return PlanResult(
            label=f"Plan B (+{headcount_delta} staff)",
            production=production,
            projected_sales=round(sales, 1),
            waste=round(waste, 1),
            revenue=round(revenue, 2),
            discount_cost=0,
            profit=round(profit, 2),
            capacity_util_pct=round(production/new_capacity*100, 1) if new_capacity > 0 else 0,
            params={"headcount_delta": headcount_delta, "new_capacity": new_capacity},
        )

    def _scenario_production(self, forecast, inventory, capacity, price, adj, coeffs, product: str = "croissant"):
        target = int(adj.get("production_target", forecast))
        if target > capacity:
            production = capacity
        else:
            production = target
        sales = min(forecast, production + inventory)
        surplus = max(0, production + inventory - forecast)
        revenue = sales * price
        cost = production * _get_product_cost(product)
        # Day-1 salvage: at most 30% of fresh demand can be sold at 90% price
        day1_cap = int(forecast * 0.3)
        salvageable = min(int(surplus), max(day1_cap, 0))
        real_waste = int(surplus) - salvageable
        cost_ratio = _get_product_cost(product) / max(_get_product_price(product), 0.01)
        salvage = salvageable * price * 0.9 * (1 - cost_ratio)
        profit = revenue + salvage - cost

        return PlanResult(
            label=f"Plan B (produce {production})",
            production=production,
            projected_sales=round(sales, 1),
            waste=real_waste,
            revenue=round(revenue, 2),
            discount_cost=0,
            profit=round(profit, 2),
            capacity_util_pct=round(production/capacity*100, 1) if capacity > 0 else 0,
            params={"production_target": target, "capacity": capacity},
        )

    # ------------------------------------------------------------------
    # Plan C: Grid-search optimal
    # ------------------------------------------------------------------
    def _grid_search_optimal(self, forecast, inventory, capacity, price,
                             scenario_type, coeffs, product: str = "croissant") -> Optional[PlanResult]:
        """Search a small grid of parameters to find the profit-maximizing plan."""
        best = None
        best_profit = -999999

        if scenario_type == "discount":
            for disc in [0, 10, 20, 30, 40, 50]:
                adj = {"discount_pct": disc}
                result = self._scenario_discount(forecast, inventory, capacity, price, adj, coeffs, product)
                if result.profit > best_profit:
                    best_profit = result.profit
                    best = result
        elif scenario_type == "staffing":
            for hc in [0, 1, 2, 3]:
                adj = {"headcount_delta": hc}
                result = self._scenario_staffing(forecast, inventory, capacity, price, adj, coeffs)
                if result.profit > best_profit:
                    best_profit = result.profit
                    best = result
        elif scenario_type == "production":
            needed = max(1, int(forecast - inventory + 0.5))
            # Pure production candidates
            candidates = sorted(set([
                max(1, int(forecast * 0.5)),
                max(1, int(forecast * 0.8)),
                needed,
                max(1, int(forecast * 1.0)),
                max(1, int(forecast * 1.2)),
                max(1, int(forecast * 1.5)),
                min(capacity, max(1, int(forecast * 2.0))),
            ]))
            for target in candidates:
                adj = {"production_target": target}
                result = self._scenario_production(forecast, inventory, capacity, price, adj, coeffs, product)
                if result.profit > best_profit:
                    best_profit = result.profit
                    best = result
            # Also search discount scenarios -- discounting may unlock higher profit
            for disc in [10, 20, 30, 40, 50]:
                adj = {"discount_pct": disc}
                result = self._scenario_discount(forecast, inventory, capacity, price, adj, coeffs, product)
                if result.profit > best_profit:
                    best_profit = result.profit
                    best = result

        if best:
            best.label = "Plan C (Optimal)"
        return best

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------
    def _attribute(self, plan_a: PlanResult, plan_b: PlanResult) -> Dict:
        """Decompose profit delta into components."""
        delta_profit = round(plan_b.profit - plan_a.profit, 2)
        delta_revenue = round(plan_b.revenue - plan_a.revenue, 2)
        delta_waste = round(plan_b.waste - plan_a.waste, 1)
        delta_sales = round(plan_b.projected_sales - plan_a.projected_sales, 1)

        components = []

        if plan_b.discount_cost > 0:
            components.append({
                "factor": "Discount cost",
                "impact": round(-plan_b.discount_cost, 2),
                "explanation": f"RM{plan_b.discount_cost:.2f} lost from {plan_b.params.get('discount_pct', 0)}% discount"
            })

        if delta_sales != 0:
            sign = "gain" if delta_sales > 0 else "loss"
            components.append({
                "factor": f"Sales volume {sign}",
                "impact": round(delta_sales, 1),
                "explanation": f"{delta_sales:+.1f} units vs baseline"
            })

        if delta_waste > 0:
            components.append({
                "factor": "Extra waste",
                "impact": round(-delta_waste, 1),
                "explanation": f"{delta_waste:+.1f} units wasted vs baseline"
            })

        return {
            "delta_profit": delta_profit,
            "delta_revenue": delta_revenue,
            "delta_waste": delta_waste,
            "delta_sales": delta_sales,
            "components": components,
        }
