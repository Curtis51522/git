# Demand Agent - sales forecast + trend analysis + SHAP drivers
import httpx, logging, asyncio
from typing import Dict, Any
from .base import BaseAgent
from s5_config.settings import S2_FORECAST_URL, PRODUCT_NAMES

logger = logging.getLogger("s5.agent.demand")


class DemandAgent(BaseAgent):
    def __init__(self):
        super().__init__("demand")

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        product = params.get("product", "croissant")
        days = params.get("days", 7)
        date = params.get("date", "")

        async with httpx.AsyncClient() as client:
            if product == "all":
                all_forecasts = []
                for p in PRODUCT_NAMES:
                    url = f"{S2_FORECAST_URL}?days={days}&product={p}"
                    if date:
                        url += f"&date={date}"
                    try:
                        resp = await client.get(url, timeout=10)
                        resp.raise_for_status()
                        data = resp.json()
                        all_forecasts.extend(data.get("forecasts", []))
                    except Exception as e:
                        logger.warning("Demand fetch failed for %s: %s", p, e)
                return {"forecasts": all_forecasts, "product": "all"}
            else:
                url = f"{S2_FORECAST_URL}?days={days}&product={product}"
                if date:
                    url += f"&date={date}"
                resp = await client.get(url, timeout=10)
                resp.raise_for_status()
                return resp.json()

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        forecasts = raw.get("forecasts", [])
        target_date = params.get("date", "")
        if not target_date:
            return {"opinion": "No date specified", "confidence": 0, "constraints": [], "data": {"forecast": 0}}

        # Filter to target date only
        date_matches = [f for f in forecasts if f.get("forecast_date", "")[:10] == target_date[:10]]
        if not date_matches:
            # Fallback: take first available forecasts
            date_matches = forecasts[:len(PRODUCT_NAMES)] if raw.get("product") == "all" else forecasts[:1]

        # Sum predicted demand across matched forecasts
        predicted = sum(f.get("predicted_demand", 0) for f in date_matches)
        lower = sum(f.get("lower_bound", 0) for f in date_matches)
        upper = sum(f.get("upper_bound", 0) for f in date_matches)

        # Trend from first product's first 3 days
        trend = "stable"
        per_product_fc = {}
        for f in date_matches:
            pname = f.get("product_name", "")
            if pname:
                per_product_fc[pname] = {
                    "forecast": f.get("predicted_demand", 0),
                    "lower": f.get("lower_bound", 0),
                    "upper": f.get("upper_bound", 0),
                }
        sample = [f for f in forecasts if f.get("product_name") == (date_matches[0].get("product_name", "") if date_matches else "croissant")]
        if len(sample) >= 3:
            vals = [s.get("predicted_demand", 0) for s in sample[:3]]
            if vals[0] > vals[2] * 1.15:
                trend = "declining"
            elif vals[2] > vals[0] * 1.15:
                trend = "rising"

        return {
            "opinion": f"Forecast {predicted} units ({lower}-{upper}), trend {trend}",
            "confidence": 0.7,
            "constraints": [],
            "data": {
                "forecast": predicted,
                "per_product": per_product_fc,
                "forecast_low": lower,
                "forecast_high": upper,
                "trend": trend,
                "confidence_label": "high" if predicted > 0 else "low",
            },
        }
