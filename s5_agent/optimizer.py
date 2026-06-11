# Multi-objective production optimizer with bakery constraints
# Phase 2: MIP with integer batch sizes + shelf-life + day-1 prioritization.
# Phase 4: Profit-aware objective + uncertainty-aware (robust) optimization.
#
# Uses scipy.optimize.milp (HiGHS solver, 2024 MIP competition winner).
#
# Formulation (MIP - Phase 4):
#   maximize profit = revenue - production_cost - waste_cost - stockout_cost - shelf_penalty
#   s.t.      sum_i b_i <= total_capacity
#             b_i + fresh_i + usable_day1_i + shortage_i = demand_i + waste_i
#             shelf_penalty_i >= day1_i - demand_i   (unsold day-1 becomes waste)
#             b_i in {0, batch, 2*batch, ...}         (integer batch constraint)
#             b_i, waste_i, shortage_i, shelf_penalty_i >= 0
#
# Uncertainty-aware: uses forecast CI to compute robust demand scenarios.
# Risk preference alpha in [0,1]:
#   0 = conservative (minimize waste, use lower bound)
#   0.5 = balanced (use predicted)
#   1 = aggressive (minimize stockout, use upper bound)

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import logging, math
import numpy as np

logger = logging.getLogger("s5.optimizer")

# HiGHS MIP solver (scipy >= 1.9)
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
    HAS_LP = True
except Exception:
    HAS_MILP = False
    try:
        from scipy.optimize import linprog
        HAS_LP = True
    except Exception:
        HAS_LP = False

if HAS_MILP:
    logger.info("MILP solver available (scipy.milp + HiGHS)")
elif HAS_LP:
    logger.warning("MILP not available, falling back to LP + batch rounding")
else:
    logger.warning("scipy not installed, falling back to analytical single-product mode")


# ---------------------------------------------------------------------------
# Configurable bakery parameters (industry-standard defaults)
# ---------------------------------------------------------------------------
@dataclass
@dataclass
class BakeryConfig:
    """Physical constraints of a typical community bakery.
    All values are configurable; defaults drawn from industry benchmarks."""
    oven_layers: int = 2
    capacity_per_layer: int = 12
    batch_size: int = 1
    baking_time_min: int = 18
    # baking_window_hours = shop_open - baker_start - setup (computed below)
    oven_layers: int = 2        # trays per oven
    oven_count: int = 2          # number of ovens
    shop_open_hour: int = 9     # 9:00 AM opening
    baker_start_hour: int = 4   # baker arrives at 4:00 AM, 5h before opening
    setup_minutes: int = 30     # oven preheat + cleanup (not baking)
    # Phase 4: risk preference (0=conservative, 0.5=balanced, 1=aggressive)
    risk_preference: float = 0.5
    # Default per-unit revenue (can be overridden per product)
    default_unit_price: float = 5.90

    @property
    def baking_window_hours(self) -> float:
        """Net hours available for baking before shop opens."""
        raw = self.shop_open_hour - self.baker_start_hour - self.setup_minutes / 60.0
        return max(0.0, raw)

    @property
    def max_batches_per_hour(self) -> float:
        return 60.0 / self.baking_time_min

    @property
    def max_units_per_hour(self) -> float:
        return self.max_batches_per_hour * self.oven_layers * self.oven_count * self.capacity_per_layer

    @property
    def daily_capacity(self) -> int:
        """Total units bakeable before shop opens."""
        return int(self.max_units_per_hour * self.baking_window_hours)

    @property
    def batch_count(self) -> int:
        return self.daily_capacity // self.batch_size
@dataclass
class CostParams:
    waste_loss: float = 0.59
    stockout_loss: float = 5.50
    production_cost: float = 0.50
    shelf_penalty: float = 0.80

    def update_from_causal(self, attribution) -> None:
        if attribution is not None:
            self.waste_loss = getattr(attribution, "avg_waste_per_unit_cost", self.waste_loss)
            self.stockout_loss = getattr(attribution, "avg_stockout_per_unit_cost", self.stockout_loss)


@dataclass
class ProductState:
    name: str
    demand: float = 0
    demand_low: float = 0
    demand_high: float = 0
    fresh_stock: float = 0
    day1_stock: float = 0
    unit_price: float = 5.90
    stockout_loss: float = 5.50
    waste_loss: float = 0.59
    production_cost: float = 0.50
    shelf_penalty: float = 0.80

    @property
    def total_stock(self) -> float:
        return self.fresh_stock + self.day1_stock

    def robust_demand(self, alpha: float) -> float:
        """Compute risk-adjusted demand: alpha=0 -> low, 0.5 -> mid, 1 -> high."""
        alpha = max(0.0, min(1.0, alpha))
        if self.demand_low > 0 and self.demand_high > 0:
            return self.demand_low + alpha * (self.demand_high - self.demand_low)
        return self.demand


# ---------------------------------------------------------------------------
# Single-product optimization (analytical + batch rounding + day-1 priority)
# ---------------------------------------------------------------------------
def optimize_single(demand: float, stock: float, max_capacity: float,
                    costs: CostParams = None, product_name: str = "",
                    day1_stock: float = 0, config: BakeryConfig = None,
                    unit_price: float = 5.90) -> Dict[str, Any]:
    """
    Phase 4: Profit-aware single product optimization.
    Day-1 prioritization: sell oldest stock first before baking fresh.
    Integer batch rounding: bake must be in multiples of batch_size.
    """
    if costs is None:
        costs = CostParams()
    if config is None:
        config = BakeryConfig()

    fresh_stock = max(0, stock - day1_stock)

    # Step 1: Clear day-1 stock first (sell before expiry)
    day1_sold = min(day1_stock, demand)
    remaining_demand = demand - day1_sold
    day1_wasted = day1_stock - day1_sold

    # Step 2: Use fresh stock
    fresh_used = min(fresh_stock, remaining_demand)
    gap = remaining_demand - fresh_used

    # Step 3: Compute optimal bake (profit-aware)
    # Profit from baking 1 unit: revenue - production_cost
    # But baking also incurs waste risk if overbaked
    bake_raw = max(0, min(gap, max_capacity))

    # Batch rounding
    batch = config.batch_size
    bake = int(math.ceil(bake_raw / batch) * batch) if bake_raw > 0 else 0
    bake = min(bake, int(max_capacity))

    # Step 4: Compute outcomes
    total_available = bake + fresh_stock + day1_sold
    sold = min(total_available, demand)
    shortage = max(0, demand - sold)
    waste = max(0, total_available - demand) + day1_wasted

    # Profit calculation
    revenue = float(sold * unit_price)
    total_cost = (bake * costs.production_cost +
                  waste * costs.waste_loss +
                  shortage * costs.stockout_loss)
    profit = revenue - total_cost

    prefix = f"{product_name}: " if product_name else ""
    if bake == 0 and stock >= demand:
        reason = f"{prefix}Overstocked: {stock:.0f} vs demand {demand:.0f}. Bake 0."
        profit_note = f" (profit RM{profit:.0f} from existing stock)"
    elif bake == 0:
        reason = f"{prefix}No bakers available. demand {demand:.0f} unmet."
        profit_note = ""
    else:
        reason = f"{prefix}Optimal: bake {bake} (batched) + stock {stock:.0f} = demand {demand:.0f}."
        profit_note = f" (profit RM{profit:.0f})"

    return {
        "bake_units": bake,
        "waste_units": int(waste),
        "shortage_units": int(shortage),
        "profit_rm": round(profit, 2),
        "revenue_rm": round(revenue, 2),
        "rationale": reason,
        "profit_note": profit_note,
        "method": "analytical single-product (profit-aware)",
    }


# ---------------------------------------------------------------------------
# Multi-product MIP optimization (Phase 4: profit-aware + robust)
# ---------------------------------------------------------------------------
def optimize_multi(products: List[ProductState], total_capacity: float,
                   costs: CostParams = None, config: BakeryConfig = None) -> Dict[str, Any]:
    """
    Phase 4: Multi-product profit-aware MIP with robust demand.
    
    Objective: maximize sum_i [ revenue_i - production_cost_i - waste_cost_i - stockout_cost_i ]
    where revenue_i = float(min(available_i, demand_i) * unit_price_i)
    
    Uses HiGHS branch-and-bound MIP solver via scipy.milp.
    Falls back to LP + rounding if MIP unavailable.
    """
    if not HAS_MILP:
        if HAS_LP:
            return _optimize_multi_lp_rounded(products, total_capacity, costs, config)
        return _optimize_multi_fallback(products, total_capacity, config)

    if costs is None:
        costs = CostParams()
    if config is None:
        config = BakeryConfig()

    n = len(products)
    if n == 0:
        return {"bake_units": 0, "per_product": {}, "rationale": "No products", "method": "none"}

    # Use robust demand based on risk preference
    alpha = config.risk_preference
    demands = [p.robust_demand(alpha) for p in products]

    # -----------------------------------------------------------------------
    # Decision variables per product i:
    #   b_i    = bake quantity (integer, batch-aligned)
    #   w_i    = waste (excess beyond demand)
    #   s_i    = shortage (unmet demand)
    #   rev_i  = actual revenue (capped at demand * price)
    # Total: 4 * n variables
    # -----------------------------------------------------------------------
    nvars = 4 * n
    # Indices: b[0..n-1], w[n..2n-1], s[2n..3n-1], rev[3n..4n-1]

    # Objective: maximize sum(rev_i - production_cost*b_i - waste_cost*w_i - stockout_cost*s_i)
    # milp minimizes, so multiply by -1
    c_obj = [0.0] * nvars
    for i, p in enumerate(products):
        c_obj[i] = costs.production_cost          # production cost
        c_obj[n + i] = costs.waste_loss            # waste cost
        c_obj[2 * n + i] = costs.stockout_loss     # stockout cost
        c_obj[3 * n + i] = -float(p.unit_price)           # revenue (negative to maximize)

    # Integrality: bake quantities are integers
    integrality = np.zeros(nvars, dtype=int)
    for i in range(n):
        integrality[i] = 1  # b_i integer

    # Constraint 1: sum(b_i) <= total_capacity
    A_ub = np.zeros((1, nvars))
    for i in range(n):
        A_ub[0, i] = 1.0
    b_ub = np.array([total_capacity])

    # Equality + inequality constraints per product
    A_eq_rows, b_eq_rows = [], []
    A_ineq_rows, b_ineq_rows = [], []

    for i, p in enumerate(products):
        d = demands[i]
        effective_day1 = min(p.day1_stock, d)

        # Flow balance: b_i + fresh_i + day1_usable_i + s_i = d + w_i
        # => b_i - w_i + s_i = d - fresh_i - effective_day1
        row = np.zeros(nvars)
        row[i] = 1.0
        row[n + i] = -1.0
        row[2 * n + i] = 1.0
        A_eq_rows.append(row)
        b_eq_rows.append(d - p.fresh_stock - effective_day1)

        # Revenue cap: rev_i <= d * price_i
        row2 = np.zeros(nvars)
        row2[3 * n + i] = 1.0
        A_ineq_rows.append(row2)
        b_ineq_rows.append(d * float(p.unit_price))

        # Revenue cannot exceed available * price
        # rev_i <= (b_i + fresh_i + day1_i) * price_i
        row3 = np.zeros(nvars)
        row3[i] = -float(p.unit_price)
        row3[3 * n + i] = 1.0
        A_ineq_rows.append(row3)
        b_ineq_rows.append((p.fresh_stock + effective_day1) * float(p.unit_price))

        # Revenue >= 0
        # already handled by bounds below

        # Day-1 waste: w_i >= day1_i - d (pre-existing surplus)
        day1_surplus = p.day1_stock - effective_day1
        if day1_surplus > 0:
            row4 = np.zeros(nvars)
            row4[n + i] = 1.0
            A_ineq_rows.append(row4)
            b_ineq_rows.append(day1_surplus)

        # If stock already exceeds demand, allow zero bake with full waste
        if p.total_stock >= d:
            row5 = np.zeros(nvars)
            row5[i] = 1.0
            A_ineq_rows.append(row5)
            b_ineq_rows.append(0.0)

    # Combine constraints
    constraints = []

    # Capacity constraint
    from scipy.optimize import LinearConstraint
    if A_ub.shape[0] > 0:
        constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))

    # Equality constraints
    if A_eq_rows:
        A_eq = np.array(A_eq_rows)
        constraints.append(LinearConstraint(A_eq, np.array(b_eq_rows), np.array(b_eq_rows)))

    # Inequality constraints
    if A_ineq_rows:
        A_ineq = np.array(A_ineq_rows)
        constraints.append(LinearConstraint(A_ineq, -np.inf, np.array(b_ineq_rows)))

    # Bounds
    bounds = Bounds(0, np.inf)

    # Batch alignment: b_i must be multiple of batch_size
    # Use integrality + post-hoc rounding for batch alignment
    # (HiGHS supports integer variables but not modulo constraints directly)

    try:
        result = milp(c_obj, constraints=constraints, bounds=bounds,
                      integrality=integrality,
                      options={"disp": False, "time_limit": 5})
    except Exception as e:
        logger.warning("MILP solver failed: %s", e)
        if HAS_LP:
            return _optimize_multi_lp_rounded(products, total_capacity, costs, config)
        return _optimize_multi_fallback(products, total_capacity, config)

    if not result.success:
        logger.warning("MILP did not converge: %s", result.message)
        if HAS_LP:
            return _optimize_multi_lp_rounded(products, total_capacity, costs, config)
        return _optimize_multi_fallback(products, total_capacity, config)

    # Extract solution with batch rounding
    batch = config.batch_size
    per_product = {}
    total_bake = 0
    total_waste = 0
    total_short = 0
    total_profit = 0
    total_revenue = 0

    for i, p in enumerate(products):
        d = demands[i]
        raw_b = result.x[i]
        # Round to batch size
        b = int(math.ceil(raw_b / batch) * batch) if raw_b > 0 else 0
        b = min(b, int(total_capacity) - total_bake)
        total_bake += b
        w = int(max(0, result.x[n + i]))
        s = int(max(0, result.x[2 * n + i]))
        rev = result.x[3 * n + i]
        day1_wasted = max(0, p.day1_stock - min(p.day1_stock, d))
        total_waste += w + int(day1_wasted)
        total_short += s
        total_revenue += rev
        total_profit += rev - (b * costs.production_cost + w * costs.waste_loss + s * costs.stockout_loss)
        per_product[p.name] = {
            "bake": b, "waste": w + int(day1_wasted), "shortage": s,
            "revenue": round(rev, 2), "profit": round(rev - b * costs.production_cost - w * costs.waste_loss - s * costs.stockout_loss, 2),
        }

    # Build rationale
    risk_label = {0.0: "conservative (low CI)", 0.5: "balanced (predicted)", 1.0: "aggressive (high CI)"}.get(
        alpha, f"alpha={alpha:.1f}")
    pp_parts = []
    for i, p in enumerate(products):
        d = demands[i]
        pd = per_product[p.name]
        coverage = round((p.total_stock + pd["bake"]) / max(d, 1) * 100)
        if pd["bake"] > 0:
            pp_parts.append(f"{p.name}: bake {pd['bake']} (stock {int(p.total_stock)}+{pd['bake']}={int(p.total_stock+pd['bake'])}/{int(d)}, {coverage}%)")
        elif pd["shortage"] > 0:
            pp_parts.append(f"{p.name}: shortage {pd['shortage']} (stock {int(p.total_stock)}/{int(d)}, {coverage}%)")
        else:
            pp_parts.append(f"{p.name}: ok (stock {int(p.total_stock)}/{int(d)}, {coverage}%)")

    return {
        "bake_units": total_bake,
        "waste_units": total_waste,
        "shortage_units": total_short,
        "total_cost_rm": round(-result.fun, 2) if result.fun else 0,
        "profit_rm": round(total_profit, 2),
        "revenue_rm": round(total_revenue, 2),
        "per_product": per_product,
        "risk_preference": risk_label,
        "rationale": f"MIP ({risk_label}) across {n} products (batch={batch}, profit RM{total_profit:.2f}): {'; '.join(pp_parts)}",
        "method": "scipy milp (HiGHS MIP, profit-aware, robust demand)",
    }


# ---------------------------------------------------------------------------
# Fallback methods (unchanged from Phase 2)
# ---------------------------------------------------------------------------
def _optimize_multi_lp_rounded(products: List[ProductState], cap: float,
                                 costs: CostParams, cfg: BakeryConfig) -> Dict[str, Any]:
    """LP fallback: solve continuous LP then round to nearest batch."""
    if costs is None:
        costs = CostParams()
    if cfg is None:
        cfg = BakeryConfig()
    from scipy.optimize import linprog
    n = len(products)
    nvars = 3 * n

    c_obj = [costs.production_cost] * n + [costs.waste_loss] * n + [costs.stockout_loss] * n

    A_ub = [[0.0] * nvars]
    for i in range(n):
        A_ub[0][i] = 1.0
    b_ub = [cap]

    A_eq = []
    b_eq = []
    for i, p in enumerate(products):
        effective_day1 = min(p.day1_stock, p.demand)
        rhs = p.demand - p.fresh_stock - effective_day1
        row = [0.0] * nvars
        row[i] = 1.0
        row[n + i] = -1.0
        row[2 * n + i] = 1.0
        A_eq.append(row)
        b_eq.append(rhs)

    bounds = [(0, None) for _ in range(nvars)]

    try:
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")
    except Exception as e:
        return _optimize_multi_fallback(products, cap, cfg)

    if not result.success:
        return _optimize_multi_fallback(products, cap, cfg)

    batch = cfg.batch_size
    per_product = {}
    total_bake = 0
    total_waste = 0
    total_short = 0

    for i, p in enumerate(products):
        raw_b = result.x[i]
        b = int(math.ceil(raw_b / batch) * batch) if raw_b > 0 else 0
        b = min(b, int(cap))
        w = int(round(result.x[n + i]))
        s = int(round(result.x[2 * n + i]))
        day1_wasted = max(0, p.day1_stock - min(p.day1_stock, p.demand))
        total_bake += b
        total_waste += w + int(day1_wasted)
        total_short += s
        per_product[p.name] = {"bake": b, "waste": w + int(day1_wasted), "shortage": s}

    pp_parts = []
    for i, p in enumerate(products):
        b = per_product[p.name]["bake"]
        s = per_product[p.name]["shortage"]
        coverage = round((p.total_stock + b) / max(p.demand, 1) * 100)
        if b > 0:
            pp_parts.append(f"{p.name}: bake {b} (batch-rounded, {coverage}%)")
        elif s > 0:
            pp_parts.append(f"{p.name}: shortage {s} ({coverage}%)")
        else:
            pp_parts.append(f"{p.name}: ok ({coverage}%)")

    return {
        "bake_units": total_bake,
        "waste_units": total_waste,
        "shortage_units": total_short,
        "per_product": per_product,
        "rationale": f"LP + batch rounding across {n} products (batch={batch}): {'; '.join(pp_parts)}",
        "method": "scipy linprog + post-hoc batch rounding",
    }


def _optimize_multi_fallback(products: List[ProductState], cap: float,
                              cfg: BakeryConfig = None) -> Dict[str, Any]:
    """Proportional allocation fallback."""
    if cfg is None:
        cfg = BakeryConfig()
    total_gap = sum(max(0, p.demand - p.total_stock) for p in products)
    per_product = {}
    total_bake = 0
    for p in products:
        gap = max(0, p.demand - p.total_stock)
        alloc = int(min(gap, cap * gap / total_gap)) if total_gap > 0 else 0
        total_bake += alloc
        per_product[p.name] = {"bake": alloc, "waste": 0, "shortage": max(0, int(gap - alloc))}
    return {
        "bake_units": min(total_bake, int(cap)),
        "waste_units": 0,
        "per_product": per_product,
        "rationale": "Proportional fallback (scipy unavailable)",
        "method": "proportional fallback",
    }


# ---------------------------------------------------------------------------
# Pareto Plan Generator (Plan A: LLM-constrained decision)
# ---------------------------------------------------------------------------
def generate_pareto_plans(demand: float, stock: float, max_capacity: float,
                          demand_low: float = None, demand_high: float = None,
                          costs: CostParams = None, product_name: str = "",
                          day1_stock: float = 0, config: BakeryConfig = None,
                          unit_price: float = 5.90) -> Dict[str, Any]:
    """Generate 4 Pareto plans with cross-scenario payoff matrix.

    Each plan is evaluated under all 3 demand scenarios (low, predicted, high)
    to expose real risk/reward tradeoffs for LLM decision-making.

    Plans:
      A - Aggressive:  bake for upper bound (minimize stockout)
      B - Balanced:    bake for predicted demand
      C - Conservative: bake for lower bound (minimize waste)
      D - Baseline:    exact gap fill, zero margin

    Returns {plans: [...], context: {...}} with cross-scenario outcomes.
    """
    if costs is None:
        costs = CostParams()
    if config is None:
        config = BakeryConfig()
    if demand_low is None:
        demand_low = max(0, demand * 0.7)
    if demand_high is None:
        demand_high = demand * 1.3

    # Bake decisions: how much each plan bakes
    bake_targets = {
        "A_aggressive": demand_high,
        "B_balanced": demand,
        "C_conservative": demand_low,
        "D_baseline": max(0, demand - stock),
    }

    # Demand scenarios for cross-evaluation
    demand_scenarios = [
        ("low_demand", demand_low, "worst-case: demand hits lower bound"),
        ("predicted", demand, "expected: demand matches forecast"),
        ("high_demand", demand_high, "best-case: demand hits upper bound"),
    ]

    plans = []
    for label, bake_target in bake_targets.items():
        result = optimize_single(bake_target, stock, max_capacity, costs,
                                 product_name, day1_stock, config, unit_price)
        bake_qty = result["bake_units"]

        # Cross-evaluate: what happens under each demand scenario with this bake?
        scenario_outcomes = {}
        total_available = bake_qty + stock
        for s_label, s_demand, s_desc in demand_scenarios:
            sold = min(total_available, s_demand)
            waste = max(0, total_available - s_demand)
            shortage = max(0, s_demand - sold)
            profit = float(sold * unit_price) - (bake_qty * costs.production_cost +
                       waste * costs.waste_loss + shortage * costs.stockout_loss)
            scenario_outcomes[s_label] = {
                "demand": round(s_demand),
                "sold": int(sold),
                "waste": int(waste),
                "shortage": int(shortage),
                "profit_rm": round(profit, 2),
            }

        # Primary outcome (under predicted demand)
        primary = scenario_outcomes["predicted"]

        # Risk metrics
        worst_profit = scenario_outcomes["low_demand"]["profit_rm"]
        best_profit = scenario_outcomes["high_demand"]["profit_rm"]
        profit_swing = round(best_profit - worst_profit, 2)
        max_waste_exposure = scenario_outcomes["low_demand"]["waste"]
        max_shortage_exposure = scenario_outcomes["high_demand"]["shortage"]

        risk_label = label.split("_")[1] if "_" in label else label

        plan = {
            "label": label,
            "bake": bake_qty,
            "demand_used": round(bake_target),
            "profit_rm": primary["profit_rm"],
            "waste": primary["waste"],
            "shortage": primary["shortage"],
            "rationale": result["rationale"],
            "risk": risk_label,
            # Risk-reward profile
            "profit_swing_rm": profit_swing,
            "worst_case_profit": worst_profit,
            "best_case_profit": best_profit,
            "max_waste_exposure": max_waste_exposure,
            "max_shortage_exposure": max_shortage_exposure,
            # Full scenario matrix
            "scenarios": scenario_outcomes,
        }

        # Human-readable description
        if label == "A_aggressive":
            plan["desc"] = (f"Bake {bake_qty} for upper bound. "
                           f"Upside: RM{best_profit:.0f} if demand surges. "
                           f"Downside: {max_waste_exposure} waste units if demand drops.")
        elif label == "B_balanced":
            plan["desc"] = (f"Bake {bake_qty} for forecast. "
                           f"Balanced: +/- RM{abs(profit_swing):.0f} swing. "
                           f"{max_shortage_exposure} shortage if demand spikes.")
        elif label == "C_conservative":
            plan["desc"] = (f"Bake {bake_qty} for lower bound. "
                           f"Safest: {max_waste_exposure} max waste. "
                           f"But {max_shortage_exposure} lost sales if busy day.")
        else:
            plan["desc"] = (f"Exact gap ({bake_qty}). "
                           f"Zero margin. {max_shortage_exposure} shortage if any spike.")

        plans.append(plan)

    context = {
        "product": product_name or "all",
        "stock": round(stock),
        "day1_stock": round(day1_stock),
        "max_capacity": round(max_capacity),
        "forecast": round(demand),
        "forecast_range": f"{round(demand_low)}-{round(demand_high)}",
        "unit_price": unit_price,
    }

    return {"plans": plans, "context": context}

# ---------------------------------------------------------------------------
# Multi-Period Projection: 7-day rolling simulation
# ---------------------------------------------------------------------------
def project_multi_period(daily_forecasts: list, initial_fresh: float, initial_day1: float,
                         bake_today: float, unit_price: float = 5.90,
                         costs=None) -> dict:
    """Simulate 7-day stock evolution from today's decision.

    Args:
        daily_forecasts: list of 7 daily demand forecasts [d0, d1, ..., d6]
        initial_fresh: current fresh stock
        initial_day1: current day-1 (yesterday's) stock
        bake_today: units baked today (from selected plan)
        unit_price: per-unit selling price
        costs: CostParams or None for defaults

    Returns:
        {"days": [...], "cumulative_waste": int, "cumulative_shortage": int,
         "final_stock": int, "avg_stock_coverage_pct": float, "risk_trend": str}
    """
    if costs is None:
        costs = CostParams()

    days = []
    fresh = initial_fresh + bake_today
    day1 = initial_day1
    cum_waste = 0
    cum_shortage = 0
    coverages = []

    for i, demand in enumerate(daily_forecasts[:7]):
        demand = max(0, demand)
        total_avail = fresh + day1

        # Sell day-1 first (FIFO), then fresh
        sold_day1 = min(day1, demand)
        remaining_demand = demand - sold_day1
        sold_fresh = min(fresh, remaining_demand)
        total_sold = sold_day1 + sold_fresh
        shortage = max(0, demand - total_sold)

        # Day-1 that wasn't sold becomes waste
        wasted = max(0, day1 - sold_day1)
        cum_waste += wasted

        # Unsold fresh becomes next day's day-1
        next_day1 = max(0, fresh - sold_fresh)

        # Profit for the day
        revenue = float(total_sold * unit_price)
        cost = wasted * costs.waste_loss + shortage * costs.stockout_loss + bake_today * costs.production_cost
        day_profit = round(revenue - cost, 2)

        # Coverage
        coverage = round((total_avail / demand * 100) if demand > 0 else 999, 1)
        coverages.append(coverage)

        days.append({
            "day": i,
            "demand": round(demand),
            "fresh_start": round(fresh),
            "day1_start": round(day1),
            "sold": int(total_sold),
            "waste": int(wasted),
            "shortage": int(shortage),
            "day_profit_rm": day_profit,
            "coverage_pct": coverage,
        })

        # Roll forward: bake tomorrow = max(0, tomorrow_demand - day1_carryover)
        # Uses next day's demand so bake matches what's actually needed
        if i + 1 < len(daily_forecasts):
            next_demand = max(0, daily_forecasts[i + 1])
        else:
            next_demand = demand  # last day: estimate based on current
        target_bake = max(0, next_demand - next_day1)
        fresh = target_bake
        day1 = next_day1
        cum_shortage += shortage

    # Trend analysis
    avg_coverage = round(sum(coverages) / len(coverages), 1) if coverages else 0
    if avg_coverage >= 100:
        risk = "healthy"
    elif avg_coverage >= 70:
        risk = "tight"
    else:
        risk = "critical"

    return {
        "days": days,
        "cumulative_waste": cum_waste,
        "cumulative_shortage": cum_shortage,
        "final_stock": int(day1),
        "avg_stock_coverage_pct": avg_coverage,
        "risk_trend": risk,
    }