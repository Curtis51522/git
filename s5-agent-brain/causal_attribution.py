# Causal Attribution Engine
# Uses econml CausalForestDML to estimate treatment effects and calibrate
# cost parameters for the production optimizer.

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("s5.causal")

try:
    from econml.dml import CausalForestDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    logger.warning("econml not installed, causal attribution disabled")


@dataclass
class WasteAttribution:
    avg_waste_per_unit_cost: float
    avg_stockout_per_unit_cost: float
    top_waste_driver: str
    driver_effects: Dict[str, float]
    sample_size: int
    method: str


def synthesize_training_data(num_samples: int = 200):
    np.random.seed(42)
    n = num_samples
    day_of_week = np.random.randint(0, 7, n)
    is_weekend = (day_of_week >= 5).astype(float)
    weather_score = np.random.normal(0, 1, n)
    true_demand = 40 + 15 * is_weekend + 8 * weather_score + np.random.normal(0, 8, n)
    true_demand = np.maximum(5, true_demand)
    overbake_bias = np.where(is_weekend, 10, 5)
    bake_qty = true_demand + overbake_bias + np.random.normal(0, 5, n)
    bake_qty = np.maximum(0, bake_qty)
    sold = np.minimum(bake_qty, true_demand + np.random.exponential(3, n))
    waste_qty = np.maximum(0, bake_qty - sold)
    X = np.column_stack([day_of_week, is_weekend, weather_score, true_demand])
    return X, bake_qty.reshape(-1, 1), waste_qty


def calibrate_costs(X, T, Y):
    if not HAS_ECONML:
        return _heuristic_calibration(X, T, Y)
    try:
        cf = CausalForestDML(n_estimators=100, min_samples_leaf=10, max_depth=5, random_state=42)
        cf.fit(Y=Y.ravel(), T=T.ravel(), X=X)
        ate = cf.ate(X=X)
        waste_loss = float(max(0, ate.item() if hasattr(ate, "item") else ate))
        stockout_loss = 5.50
        true_demand = X[:, 3]
        bake_qty_arr = T.ravel()
        shortage = np.maximum(0, true_demand - bake_qty_arr)
        feature_names = ["day_of_week", "is_weekend", "weather_score", "true_demand"]
        driver_effects = {}
        for i, name in enumerate(feature_names):
            Xp = X.copy()
            Xp[:, i] += 1.0
            driver_effects[name] = round(float(cf.ate(X=Xp)[0] - ate[0]), 4)
        top_driver = max(driver_effects, key=lambda k: abs(driver_effects[k]))
        return WasteAttribution(
            avg_waste_per_unit_cost=round(waste_loss, 4),
            avg_stockout_per_unit_cost=round(stockout_loss, 2),
            top_waste_driver=top_driver,
            driver_effects=driver_effects,
            sample_size=len(Y),
            method="CausalForestDML (econml)",
        )
    except Exception as e:
        logger.warning("CausalForestDML failed: %s", e)
        return _heuristic_calibration(X, T, Y)


def _heuristic_calibration(X, T, Y):
    bake_qty_arr = T.ravel()
    waste_qty = Y
    true_demand = X[:, 3]
    overbake = np.maximum(0, bake_qty_arr - true_demand)
    mask = overbake > 0
    waste_loss = float(np.mean(waste_qty[mask] / overbake[mask])) if mask.any() else 1.80
    stockout_loss = 5.50
    driver_effects = {
        "day_of_week": float(np.corrcoef(X[:, 0], waste_qty)[0, 1]) if len(waste_qty) > 1 else 0,
        "is_weekend": float(np.corrcoef(X[:, 1], waste_qty)[0, 1]) if len(waste_qty) > 1 else 0,
        "weather_score": float(np.corrcoef(X[:, 2], waste_qty)[0, 1]) if len(waste_qty) > 1 else 0,
        "true_demand": float(np.corrcoef(X[:, 3], waste_qty)[0, 1]) if len(waste_qty) > 1 else 0,
    }
    top_driver = max(driver_effects, key=lambda k: abs(driver_effects[k])) if driver_effects else "is_weekend"
    return WasteAttribution(
        avg_waste_per_unit_cost=round(waste_loss, 4),
        avg_stockout_per_unit_cost=round(stockout_loss, 2),
        top_waste_driver=top_driver,
        driver_effects={k: round(v, 4) for k, v in driver_effects.items()},
        sample_size=len(Y),
        method="heuristic (correlation-based)",
    )


def counterfactual_analysis(bake_qty, stock, demand, attribution):
    scenarios = {}
    for label, b in [("bake_0", 0), ("bake_ideal", max(0, demand - stock)), ("bake_actual", bake_qty)]:
        avail = b + stock
        waste = max(0, avail - demand)
        short = max(0, demand - avail)
        cost = attribution.avg_waste_per_unit_cost * waste + attribution.avg_stockout_per_unit_cost * short
        scenarios[label] = {"bake": int(b), "waste": int(waste), "shortage": int(short), "cost_rm": round(cost, 2)}
    return {
        "scenarios": scenarios,
        "waste_loss_per_unit": attribution.avg_waste_per_unit_cost,
        "stockout_loss_per_unit": attribution.avg_stockout_per_unit_cost,
        "top_driver": attribution.top_waste_driver,
        "method": attribution.method,
    }
