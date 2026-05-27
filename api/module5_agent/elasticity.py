"""Elasticity Estimator for B2 What-if Simulator.

Two-phase strategy:
- Cold-start: DeepSeek LLM synthesizes domain-informed elasticity coefficients
- Warm-start (30+ days of real data): OLS regression on historical pairs

Coefficients are persisted to models/elasticity_cache.json for crash recovery.
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger("s5.elasticity")

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
)
CACHE_PATH = os.path.join(MODEL_DIR, "elasticity_cache.json")

# ---------------------------------------------------------------------------
# Default heuristics (used when LLM is unavailable)
# ---------------------------------------------------------------------------
DEFAULT_ELASTICITY = {
    "discount_volume_elasticity":     0.15,   # +15% volume per 10% discount
    "staff_throughput_per_headcount": 8.0,    # +8 txns/hr per extra person
    "production_waste_rate":          0.05,   # +5% waste per 10% over-forecast
    "cross_elasticity_bread_coffee":  0.12,   # +12% coffee sales when bread on discount
    "freshness_discount_attraction":  0.08,   # +8% more day-1 sold per 10% extra discount
    "peak_hour_multiplier":           1.3,    # 30% more volume in peak hours
    "source": "default_heuristic",
}


class ElasticityEstimator:
    """Two-phase elasticity estimator for counterfactual simulation."""

    def __init__(self):
        self.coefficients: Dict = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_coefficients(self) -> Dict:
        """Return current coefficients. If cold-start and empty, synthesize."""
        if not self.coefficients:
            self.synthesize_from_llm()
        return self.coefficients or DEFAULT_ELASTICITY

    def synthesize_from_llm(self) -> Dict:
        """Cold-start: ask DeepSeek to estimate bakery-specific elasticities."""
        try:
            from api.module5_agent.llm_client import call_deepseek

            prompt = """You are a bakery operations expert in Malaysia. Estimate realistic
elasticity coefficients for a small bakery-cafe (6 products, 4 staff, 50 capacity/day).

Return ONLY valid JSON with these keys:
{
  "discount_volume_elasticity": <float, % volume increase per 10% discount, e.g. 0.15 = 15%>,
  "staff_throughput_per_headcount": <float, extra transactions per hour per additional staff, e.g. 8.0>,
  "production_waste_rate": <float, % extra waste per 10% overproduction, e.g. 0.05 = 5%>,
  "cross_elasticity_bread_coffee": <float, % coffee sales increase when bread on discount, e.g. 0.12>,
  "freshness_discount_attraction": <float, % more day-old items sold per 10% extra discount, e.g. 0.08>,
  "peak_hour_multiplier": <float, volume multiplier during peak hours (8-10am, 12-2pm), e.g. 1.3>
}

Be conservative -- Malaysian bakeries have thin margins. Small shop, RM pricing.
Return ONLY the JSON object, no explanation."""

            system = "You are a bakery economics expert. Return only JSON. Be realistic."
            response = call_deepseek(prompt, system, max_tokens=300)

            if response:
                response = response.strip()
                if response.startswith("```"):
                    parts = response.split("```")
                    response = parts[1] if len(parts) > 1 else response
                    if response.startswith("json"):
                        response = response[4:]
                coeffs = json.loads(response)
                coeffs["source"] = "llm_synthesized"
                self.coefficients = coeffs
                self._save()
                logger.info("Elasticity synthesized via LLM: %s", coeffs)
                return coeffs
        except Exception as e:
            logger.warning("LLM elasticity synthesis failed: %s -- using defaults", e)

        self.coefficients = dict(DEFAULT_ELASTICITY)
        self._save()
        return self.coefficients

    def fit_from_history(self, historical_data: list) -> Dict:
        """Warm-start: fit OLS regression on real historical data.

        historical_data: list of dicts with keys:
          {discount_pct, volume, headcount, throughput, ...}

        Requires at least 30 data points.
        """
        if len(historical_data) < 30:
            logger.info("Not enough data for OLS fit (%d < 30)", len(historical_data))
            return self.get_coefficients()

        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression

            # Fit discount elasticity: volume ~ discount_pct
            X_disc = np.array([[d.get("discount_pct", 0)] for d in historical_data])
            y_vol = np.array([d.get("volume", 0) for d in historical_data])
            if len(X_disc) > 0 and np.std(X_disc) > 0:
                lr = LinearRegression()
                lr.fit(X_disc, y_vol)
                base_vol = np.mean(y_vol[X_disc.flatten() == 0]) if any(X_disc.flatten() == 0) else np.mean(y_vol)
                if base_vol > 0:
                    self.coefficients["discount_volume_elasticity"] = round(
                        float(lr.coef_[0] / base_vol), 4
                    )

            # Fit staff throughput: throughput ~ headcount
            X_staff = np.array([[d.get("headcount", 1)] for d in historical_data])
            y_tput = np.array([d.get("throughput", 0) for d in historical_data])
            if len(X_staff) > 0 and np.std(X_staff) > 0:
                lr2 = LinearRegression()
                lr2.fit(X_staff, y_tput)
                self.coefficients["staff_throughput_per_headcount"] = round(
                    float(lr2.coef_[0]), 2
                )

            self.coefficients["source"] = "ols_fitted"
            self._save()
            logger.info("Elasticity fitted via OLS on %d data points", len(historical_data))
        except Exception as e:
            logger.warning("OLS fit failed: %s", e)

        return self.coefficients

    # ------------------------------------------------------------------
    # Apply elasticity to counterfactual
    # ------------------------------------------------------------------
    def apply_discount_effect(self, base_volume: float, discount_pct: float) -> Tuple[float, float]:
        """Return (projected_volume, additional_revenue_loss_from_discount).

        additional_revenue_loss = discount_pct * projected_volume * avg_price
        (approximate -- caller should adjust with actual price)
        """
        coeffs = self.get_coefficients()
        elasticity = coeffs.get("discount_volume_elasticity", 0.15)
        # Normalize: effect per 10% discount
        volume_boost = 1.0 + elasticity * (discount_pct / 10.0)
        projected = base_volume * volume_boost
        return round(projected, 1), round(projected * (discount_pct / 100), 2)

    def apply_staff_effect(self, base_throughput: float, headcount_delta: int) -> float:
        """Return projected throughput with headcount change."""
        coeffs = self.get_coefficients()
        tph = coeffs.get("staff_throughput_per_headcount", 8.0)
        return max(0, base_throughput + headcount_delta * tph)

    def apply_production_effect(self, base_waste: float, overproduction_pct: float) -> float:
        """Return projected waste with overproduction."""
        coeffs = self.get_coefficients()
        rate = coeffs.get("production_waste_rate", 0.05)
        extra_waste = base_waste + base_waste * rate * (overproduction_pct / 10.0)
        return round(extra_waste, 1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self):
        try:
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, "r") as f:
                    self.coefficients = json.load(f)
                logger.info("Loaded elasticity from cache: source=%s",
                           self.coefficients.get("source", "unknown"))
        except Exception:
            self.coefficients = {}

    def _save(self):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(CACHE_PATH, "w") as f:
                json.dump(self.coefficients, f, default=str, indent=2)
        except Exception as e:
            logger.warning("Failed to save elasticity cache: %s", e)


# Singleton
_estimator: Optional[ElasticityEstimator] = None

def get_estimator() -> ElasticityEstimator:
    global _estimator
    if _estimator is None:
        _estimator = ElasticityEstimator()
    return _estimator
