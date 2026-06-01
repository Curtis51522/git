# Multi-objective production optimizer
# Replaces simple if/else in arbitrator with cost-minimizing optimization.
#
# Two modes:
#   1. Single-product: analytical solution (for stock_query)
#   2. Multi-product: scipy linprog allocates shared baker capacity across 6 products
#
# Thesis contribution: "We formulate bakery production as a constrained
# cost-minimization problem, replacing rule-based arbitration with
# optimal allocation under uncertainty."

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("s5.optimizer")

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed, falling back to analytical single-product mode")


@dataclass
class CostParams:
    DEFAULT_WASTE_LOSS = 0.59
    DEFAULT_STOCKOUT_LOSS = 5.50
    DEFAULT_PRODUCTION_COST = 0.50
DEFAULT_PRODUCTION_COST = 0.50  # RM/unit


class CostParams:
    """Per-unit costs (RM). Calibrated from historical data via causal inference."""
    waste_loss: float = DEFAULT_WASTE_LOSS
    stockout_loss: float = DEFAULT_STOCKOUT_LOSS
    production_cost: float = DEFAULT_PRODUCTION_COST


@dataclass
class ProductState:
    """Per-product state for multi-product optimization."""
    name: str
    demand: float = 0
    stock: float = 0          # total (fresh + day1)
    stockout_loss: float = DEFAULT_STOCKOUT_LOSS
    waste_loss: float = DEFAULT_WASTE_LOSS
    production_cost: float = DEFAULT_PRODUCTION_COST


def optimize_single(demand: float, stock: float, max_capacity: float,
                    costs: CostParams = None) -> Dict[str, Any]:
    """
    Single-product: analytical optimum = clamp(demand - stock, 0, capacity).
    """
    if costs is None:
        costs = CostParams()
    unconstrained = demand - stock
    bake = max(0.0, min(unconstrained, max_capacity))
    available = bake + stock
    waste = max(0.0, available - demand)
    shortage = max(0.0, demand - available)
    total_cost = costs.waste_loss * waste + costs.stockout_loss * shortage + costs.production_cost * bake

    if unconstrained <= 0:
        reason = f"Overstocked: {stock:.0f} vs demand {demand:.0f}. Bake 0 to avoid waste."
    elif unconstrained > max_capacity:
        reason = f"Gap {unconstrained:.0f} > capacity {max_capacity:.0f}. Bake full ({max_capacity:.0f}). Shortage {shortage:.0f} unavoidable."
    else:
        reason = f"Optimal: bake {bake:.0f} + stock {stock:.0f} = demand {demand:.0f}."

    return {
        "bake_units": int(bake), "available": int(available),
        "waste_units": int(waste), "shortage_units": int(shortage),
        "total_cost_rm": round(total_cost, 2),
        "rationale": reason, "method": "analytical single-product",
    }


def optimize_multi(products: List[ProductState], total_capacity: float,
                   costs: CostParams = None) -> Dict[str, Any]:
    """
    Multi-product: allocate shared baker capacity across products to
    minimize total expected cost (waste + shortage + production).

    Formulation:
      minimize sum_i [ w_i * waste_i + s_i * shortage_i + c_i * b_i ]
      s.t.      sum_i b_i <= total_capacity
                b_i >= 0 for all i

    Uses scipy.optimize.linprog (Simplex).
    Falls back to proportional allocation if scipy unavailable.
    """
    if costs is None:
        costs = CostParams()

    n = len(products)
    if n == 0:
        return {"bake_units": 0, "per_product": {}, "rationale": "No products", "method": "none"}

    if HAS_SCIPY:
        return _optimize_multi_scipy(products, total_capacity)
    else:
        return _optimize_multi_fallback(products, total_capacity)


def _optimize_multi_scipy(products: List[ProductState], cap: float) -> Dict[str, Any]:
    """Solve multi-product allocation via linear programming."""
    n = len(products)

    # Decision variables: b_0 ... b_{n-1} (bake units), waste_0 ... waste_{n-1}, short_0 ... short_{n-1}
    # Total: 3n variables
    # Objective: min sum(waste_loss_i * waste_i + stockout_loss_i * short_i + prod_cost_i * b_i)
    c = []
    for p in products:
        c.extend([p.production_cost, p.waste_loss, p.stockout_loss])

    # Constraints:
    # (1) sum b_i <= cap
    # (2) b_i + waste_i - short_i = demand_i - stock_i  for each i
    # (3) b_i, waste_i, short_i >= 0

    A_ub = [[0.0] * (3 * n)]
    for i in range(n):
        A_ub[0][3 * i] = 1.0  # coefficient for b_i
    b_ub = [cap]

    A_eq = []
    b_eq = []
    for i, p in enumerate(products):
        row = [0.0] * (3 * n)
        row[3 * i] = 1.0      # b_i
        row[3 * i + 1] = 1.0  # waste_i
        row[3 * i + 2] = -1.0  # short_i
        A_eq.append(row)
        b_eq.append(max(0.0, p.demand - p.stock))

    bounds = [(0, None) for _ in range(3 * n)]

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
    except Exception as e:
        logger.warning("linprog failed: %s, using fallback", e)
        return _optimize_multi_fallback(products, cap)

    if not result.success:
        logger.warning("linprog not optimal: %s, using fallback", result.message)
        return _optimize_multi_fallback(products, cap)

    per_product = {}
    total_bake = 0
    total_waste = 0
    total_short = 0
    for i, p in enumerate(products):
        b = round(result.x[3 * i])
        w = round(result.x[3 * i + 1])
        s = round(result.x[3 * i + 2])
        total_bake += b
        total_waste += w
        total_short += s
        per_product[p.name] = {"bake": b, "waste": w, "shortage": s}

    return {
        "bake_units": total_bake,
        "waste_units": total_waste,
        "shortage_units": total_short,
        "total_cost_rm": round(result.fun, 2) if result.fun else 0,
        "per_product": per_product,
        "rationale": f"Optimal allocation across {n} products (LP Simplex, cost RM{result.fun:.2f})",
        "method": "scipy linprog (multi-product LP)",
    }


def _optimize_multi_fallback(products: List[ProductState], cap: float) -> Dict[str, Any]:
    """Proportional allocation fallback when scipy unavailable."""
    total_gap = sum(max(0, p.demand - p.stock) for p in products)
    per_product = {}
    total_bake = 0
    for p in products:
        gap = max(0, p.demand - p.stock)
        alloc = int(min(gap, cap * gap / total_gap)) if total_gap > 0 else 0
        total_bake += alloc
        per_product[p.name] = {"bake": alloc, "waste": 0, "shortage": max(0, int(gap - alloc))}
    return {
        "bake_units": min(total_bake, int(cap)),
        "per_product": per_product,
        "rationale": f"Proportional fallback (scipy unavailable)",
        "method": "proportional fallback",
    }