# Demand Agent — sales forecast + trend analysis + SHAP drivers
import httpx, logging
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
        url = f"{S2_FORECAST_URL}?days={days}&product={product}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        return data

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        forecasts = raw.get("forecasts", [])
        today_fc = forecasts[0] if forecasts else {}
        tomorrow_fc = forecasts[1] if len(forecasts) > 1 else {}

        predicted = tomorrow_fc.get("predicted_demand", 0)
        lower = tomorrow_fc.get("lower_bound", 0)
        upper = tomorrow_fc.get("upper_bound", 0)
        confidence = tomorrow_fc.get("confidence", "medium")

        trend = "stable"
        if len(forecasts) >= 3:
            vals = [f.get("predicted_demand", 0) for f in forecasts[:3]]
            if vals[0] > vals[2] * 1.15: trend = "declining"
            elif vals[2] > vals[0] * 1.15: trend = "rising"

        return {
            "opinion": f"Forecast {predicted} units ({lower}-{upper}), trend {trend}",
            "confidence": 0.7 if confidence == "high" else 0.5 if confidence == "medium" else 0.3,
            "constraints": [],
            "data": {
                "forecast": predicted,
                "forecast_low": lower,
                "forecast_high": upper,
                "trend": trend,
                "confidence_label": confidence,
            },
        }
