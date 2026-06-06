# Promo Agent - dynamic discount optimization + bundle suggestions
# 5-dimension scoring engine: freshness, surplus, margin, trend, S4 pairing
import asyncio
import logging
import time
import httpx
from typing import Dict, Any
from .base import BaseAgent
from s5_config.settings import S4_COMBO_URL

logger = logging.getLogger("s5.agent.promo")


class PromoAgent(BaseAgent):
    def __init__(self):
        super().__init__("promo")
        self._fetch_ok = False

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        inventory_data = params.get("_inventory", {})
        demand_data = params.get("_demand", {})
        per_product_inv = inventory_data.get("per_product", {})
        per_product_demand = demand_data.get("per_product", {})

        combo_results = {}
        self._fetch_ok = False

        for pname, pdata in per_product_inv.items():
            stock = pdata.get("qty", 0)
            forecast = per_product_demand.get(pname, {}).get("forecast", 0)
            surplus = max(0, stock - forecast)
            if surplus <= 0:
                continue

            # Calculate preliminary discount (4 dims, no Pairing) to pass to S4
            batches = pdata.get("batches", 0)
            local_day1_est = (stock // 2) if batches >= 2 else 0
            freshr_est = local_day1_est / max(stock, 1)
            trend_est = per_product_demand.get(pname, {}).get("trend", "stable")
            from s5_config.settings import PRODUCT_NAMES
            margin_est = 0.62  # fallback
            surplus_score_est = min(1.0, surplus / max(forecast, 1))
            prelim_urgency = (
                freshr_est * 0.25 +
                surplus_score_est * 0.25 +
                margin_est * 0.20 +
                (1.0 if trend_est == "declining" else 0.5 if trend_est == "stable" else 0.0) * 0.15
            ) / 0.85  # normalize for 4-dims (no Pairing 15%)
            prelim_discount = min(50, max(0, round(prelim_urgency * 50)))
            if freshr_est > 0 and prelim_discount < 20:
                prelim_discount = 20
            order = {"items": [{"product_name": pname, "quantity": int(surplus)}],
                     "discount_overrides": {pname: prelim_discount}}
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(S4_COMBO_URL, json=order)
                    resp.raise_for_status()
                    data = resp.json()
                    recs = data.get("recommendations", [])
                    freshness = data.get("freshness", {})
                    if recs:
                        combo_results[pname] = {
                            "surplus": int(surplus), "stock": int(stock),
                            "forecast": int(forecast), "pairings": recs[:3],
                            "freshness": freshness.get(pname, {}),
                        }
            except Exception as e:
                logger.warning("S4 combo failed for %s: %s", pname, e)

        self._fetch_ok = len(combo_results) > 0
        return {"combo_results": combo_results}

    # ---- 5-Dimension Dynamic Discount Engine ----
    _PRODUCT_MARGINS: Dict[str, float] = {
        "croissant": 0.64, "donut": 0.62, "chiffon": 0.65,
        "bread_roll": 0.58, "bread_coconut": 0.60, "croissant_chocolate": 0.63,
    }
    _TREND_MAP: Dict[str, float] = {"declining": 1.0, "stable": 0.5, "rising": 0.0}

    def _calc_dynamic_discount(self, pname: str, stock: int, forecast: int,
                                freshness_ratio: float, trend: str,
                                pairing_score: float,
                                dynamic_margin: float = None) -> Dict[str, Any]:
        """Compute discount percentage from 5 weighted dimensions.
        
        Dimensions:
          freshness (25%): Day-1 ratio - older stock needs deeper discount
          surplus (25%):   Surplus/forecast ratio
          margin (20%):    Product margin - high-margin can discount deeper
          trend (15%):     Demand trend - declining needs stimulation
          pairing (15%):   S4 bundle score - good pairing reduces discount need
        """
        surplus = max(0, stock - forecast)
        surplus_score = min(1.0, surplus / max(forecast, 1))
        margin_factor = dynamic_margin if dynamic_margin is not None else self._PRODUCT_MARGINS.get(pname, 0.60)
        trend_score = self._TREND_MAP.get(trend, 0.5)
        pairing_inverse = 1.0 - min(1.0, pairing_score)

        urgency = (
            freshness_ratio * 0.25 +
            surplus_score * 0.25 +
            margin_factor * 0.20 +
            trend_score * 0.15 +
            pairing_inverse * 0.15
        )
        discount_pct = min(50, max(0, round(urgency * 50)))
        # Floor: Day-1 items already have 20% standalone discount
        if freshness_ratio > 0 and discount_pct < 20:
            discount_pct = 20
        return {
            "discount_pct": discount_pct,
            "urgency": round(urgency, 3),
            "breakdown": {
                "freshness": round(freshness_ratio, 2),
                "surplus": round(surplus_score, 2),
                "margin": round(margin_factor, 2),
                "trend": round(trend_score, 2),
                "pairing": round(pairing_inverse, 2),
            },
        }

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any],
                history: str = "", key_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        inventory_data = params.get("_inventory", {})
        demand_data = params.get("_demand", {})
        per_product_inv = inventory_data.get("per_product", {})
        per_product_demand = demand_data.get("per_product", {})
        combo_results = raw.get("combo_results", {})
        profit_data = params.get("_profit", {})
        by_product_margin = profit_data.get("by_product_margin", {})
        global_trend = demand_data.get("trend", "stable")

        opinions = []
        constraints = []
        overall_discount = 0
        total_surplus = 0
        discount_details = {}

        for pname, pdata in per_product_inv.items():
            stock = pdata.get("qty", 0)
            forecast = per_product_demand.get(pname, {}).get("forecast", 0)
            surplus = max(0, stock - forecast)
            if surplus <= 0:
                continue

            total_surplus += surplus

            # Freshness ratio: use real batch_inventory data from S4
            combo_info = combo_results.get(pname, {})
            freshness_data = combo_info.get("freshness", {})
            if freshness_data:
                freshness_ratio = freshness_data.get("day1_ratio", 0)
            else:
                # Fallback: estimate from batches
                batches = pdata.get("batches", 0)
                local_day1 = (stock // 2) if batches >= 2 else 0
                freshness_ratio = local_day1 / max(stock, 1)

            # Per-product trend or global fallback
            trend = per_product_demand.get(pname, {}).get("trend", global_trend)

            # S4 pairing score (combo_info already fetched above)
            pairings = combo_info.get("pairings", [])
            top_pair = pairings[0] if pairings else {}
            pairing_score = top_pair.get("total_score", 0)

            # Use dynamic margin from ProfitAgent if available
            dyn_margin = by_product_margin.get(pname)
            dd = self._calc_dynamic_discount(
                pname, stock, forecast, freshness_ratio, trend, pairing_score,
                dynamic_margin=dyn_margin)
            discount_pct = dd["discount_pct"]
            if discount_pct > overall_discount:
                overall_discount = discount_pct

            discount_details[pname] = dd

            # Build opinion with dimension breakdown
            bd = dd["breakdown"]
            dim_str = "F={} S={} M={} T={} P={}".format(
                bd["freshness"], bd["surplus"], bd["margin"],
                bd["trend"], bd["pairing"])

            # Urgency level label
            u = dd["urgency"]
            if u >= 0.7:
                level = "HIGH"
            elif u >= 0.4:
                level = "MEDIUM"
            else:
                level = "LOW"

            if pairings:
                pair_name = top_pair.get("products", top_pair.get("coffee_name", "?"))
                pair_score = pairing_score
                opinions.append(
                    "{}% off {} ({} surplus, stock {} vs forecast {}) "
                    "[{} urgency {}: {}]; Top pairing: {} (score {:.0f})".format(
                        discount_pct, pname, surplus, stock, forecast,
                        level, dd["urgency"], dim_str, pair_name, pair_score)
                )
            else:
                opinions.append(
                    "{}% off {} ({} surplus, stock {} vs forecast {}) "
                    "[{} urgency {}: {}]".format(
                        discount_pct, pname, surplus, stock, forecast,
                        dd["urgency"], dim_str)
                )

        if not opinions:
            # Determine why no discount: understock vs adequately stocked
            has_shortage = False
            for pname, pdata in per_product_inv.items():
                stock = pdata.get("qty", 0)
                forecast = per_product_demand.get(pname, {}).get("forecast", 0)
                if stock < forecast:
                    has_shortage = True
                    break
            if has_shortage:
                opinions.append("No discount needed - stock is below demand, prioritize restocking")
            else:
                opinions.append("No discount needed - stock matches demand")

        # Confidence: more dimensions engaged = higher reliability
        if len(discount_details) > 0 and len(combo_results) > 0:
            confidence = 0.85
        elif len(discount_details) > 0:
            confidence = 0.75
        elif total_surplus > 0:
            confidence = 0.55
        else:
            confidence = 0.40

        return {
            "opinion": "; ".join(opinions),
            "confidence": round(confidence, 2),
            "constraints": constraints,
            "data": {
                "discount_pct": overall_discount,
                "surplus": total_surplus,
                "per_product": combo_results,
                "opinions": opinions,
                "discount_details": discount_details,
            },
        }
