# Demand Agent - sales forecast + trend analysis + SHAP drivers
# Phase 3: dynamic confidence based on data quality.
# Phase 4: request-level snapshot cache for deterministic intra-query results.
import httpx, logging, asyncio
from typing import Dict, Any
from .base import BaseAgent
from s5_config.settings import S2_FORECAST_URL, PRODUCT_NAMES

logger = logging.getLogger("s5.agent.demand")



def _format_opinion(predicted, lower, upper, trend, memory_note, per_product_fc, params):
    """Format demand opinion. Per-product for comparison, aggregated otherwise."""
    intent = params.get("intent", "")
    if intent == "comparison_analysis" and len(per_product_fc) >= 2:
        parts = []
        for pname, pdata in sorted(per_product_fc.items()):
            parts.append(f"{pname}: {pdata['forecast']:.0f} ({pdata['lower']:.0f}-{pdata['upper']:.0f}, {pdata.get('trend', 'stable')})")
        return " | ".join(parts) + memory_note
    return f"Forecast {predicted} units ({lower}-{upper}), trend {trend}{memory_note}"


class DemandAgent(BaseAgent):
    def __init__(self):
        super().__init__("demand")
        self._fetch_ok = False
        self._products_fetched = 0

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        product = params.get("product", "croissant")
        days = params.get("days", 7)
        date = params.get("date", "")
        self._fetch_ok = False
        self._products_fetched = 0

        async with httpx.AsyncClient() as client:
            target_products = []
            if "," in product:
                target_products = [p.strip() for p in product.split(",") if p.strip() in PRODUCT_NAMES]
            elif product == "all":
                target_products = list(PRODUCT_NAMES)
            else:
                target_products = [product] if product in PRODUCT_NAMES else ["croissant"]

            # Request-level snapshot: deduplicate S2 calls within one fetch cycle
            _fetch_snapshot: Dict[str, list] = {}
            all_forecasts = []
            for p in target_products:
                cache_key = f"{p}:{days}:{date}"
                if cache_key in _fetch_snapshot:
                    all_forecasts.extend(_fetch_snapshot[cache_key])
                    self._products_fetched += 1
                    continue
                url = f"{S2_FORECAST_URL}?days={days}&product={p}"
                if date:
                    url += f"&date={date}"
                try:
                    resp = await client.get(url, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    forecasts = data.get("forecasts", [])
                    _fetch_snapshot[cache_key] = forecasts
                    all_forecasts.extend(forecasts)
                    self._products_fetched += 1
                except Exception as e:
                    logger.warning("Demand fetch failed for %s: %s", p, e)
            self._fetch_ok = self._products_fetched > 0
            return {"forecasts": all_forecasts, "product": product}

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any],
                history: str = "", key_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        forecasts = raw.get("forecasts", [])
        target_date = params.get("date", "")
        product = params.get("product", "croissant")

        if not target_date:
            return {"opinion": "No date specified", "confidence": 0.1, "constraints": [], "data": {"forecast": 0}}

        date_matches = [f for f in forecasts if f.get("forecast_date", "")[:10] == target_date[:10]]
        if not date_matches:
            date_matches = forecasts[:len(PRODUCT_NAMES)] if product in ("all", "") or "," in product else forecasts[:1]

        predicted = sum(f.get("predicted_demand", 0) for f in date_matches)
        lower = sum(f.get("lower_bound", 0) for f in date_matches)
        upper = sum(f.get("upper_bound", 0) for f in date_matches)

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
        # Per-product trend (not global - avoids cross-product contamination)
        from collections import defaultdict
        _by_product = defaultdict(list)
        for f in forecasts:
            pn = f.get("product_name", "")
            if pn:
                _by_product[pn].append(f.get("predicted_demand", 0))
        for pn in per_product_fc:
            vals = _by_product.get(pn, [])
            if len(vals) >= 3:
                if vals[0] > vals[2] * 1.15:
                    per_product_fc[pn]["trend"] = "declining"
                elif vals[2] > vals[0] * 1.15:
                    per_product_fc[pn]["trend"] = "rising"
                else:
                    per_product_fc[pn]["trend"] = "stable"
            else:
                per_product_fc[pn]["trend"] = "stable"
        # Global trend as fallback (first product's trend)
        trend = per_product_fc[list(per_product_fc.keys())[0]]["trend"] if per_product_fc else "stable"

        # Dynamic confidence
        expected_count = self._products_fetched if self._products_fetched > 0 else (len(date_matches) if date_matches else 1)
        actual_count = len(date_matches)
        data_ratio = min(1.0, actual_count / max(expected_count, 1))
        has_forecast = predicted > 0

        if self._fetch_ok and has_forecast and data_ratio >= 0.8:
            confidence = 0.70 + 0.20 * data_ratio  # 0.70-0.90
        elif self._fetch_ok and has_forecast:
            confidence = 0.55
        elif self._fetch_ok:
            confidence = 0.35  # fetched but no forecast (maybe Monday)
        else:
            confidence = 0.15  # API failed

        memory_note = ""
        if key_metrics:
            prev_forecasts = key_metrics.get("forecast_history", [])
            prev_scopes = key_metrics.get("product_scopes", [])
            # Walk backwards to find the most recent turn with matching scope
            for i in range(len(prev_forecasts) - 1, -1, -1):
                prev_val = prev_forecasts[i]
                prev_scope = prev_scopes[i] if i < len(prev_scopes) else ""
                scope_match = (prev_scope == product) or (prev_scope in ("all", "") and product in ("all", ""))
                if prev_val > 0 and scope_match:
                    delta = predicted - prev_val
                    pct = (delta / prev_val) * 100 if prev_val != 0 else 0
                    direction = "up" if delta > 0 else "down" if delta < 0 else "unchanged"
                    memory_note = f" | vs last query: {direction} {abs(pct):.0f}% (was {prev_val})"
                    break

        return {
            "opinion": _format_opinion(predicted, lower, upper, trend, memory_note, per_product_fc, params),
            "confidence": round(confidence, 2),
            "constraints": [],
            "data": {
                "forecast": predicted, "per_product": per_product_fc,
                "forecast_low": lower, "forecast_high": upper,
                "trend": trend, "confidence_label": "high" if confidence >= 0.7 else "mid" if confidence >= 0.4 else "low",
                "_raw_forecasts": forecasts,
            },
        }
