import os, sys, logging, httpx
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.feature_sensitivity")

# Human-readable feature group descriptions
FEATURE_LABELS = {
    "is_rainy": "rainy weather",
    "is_weekend": "weekend",
    "is_holiday": "holiday",
    "is_member_day": "member day",
    "is_day1": "day-1 product launch promo",
    "is_top3": "top-3 bestseller",
    "discount_pct": "discount percentage",
    "daily_tickets": "daily footfall (ticket count)",
    "lag_1": "yesterday sales",
    "lag_7_avg": "7-day rolling average sales",
    "lag_30_avg": "30-day rolling average sales",
    "day_of_week": "day of week",
    "month": "month of year",
    "product_id": "product identity",
}

HIGH_IMPACT_FEATURES = {"is_rainy", "is_weekend", "is_holiday",
                        "is_member_day", "discount_pct", "daily_tickets",
                        "lag_7_avg", "lag_1", "is_day1"}

RESERVED_SCENARIO_LABELS = {
    "is_new_product": "new product launch",
    "is_competitor": "competitor activity",
}

class FeatureSensitivityAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_feature_importance", description="Get XGBoost feature importance scores",
            parameters={"date": "string"}, primary=True, _handler=self._get_importance))
        self.tools.register(Tool(name="get_today_features", description="Get today's actual feature values",
            parameters={"date": "string"}, _handler=self._get_today_features))

    async def fetch(self, params):
        """Override to fetch both importance and today's features."""
        date = params.get("date", "")
        imp_result = await self.tools.execute_with_fallback("get_feature_importance", {"date": date})
        today_result = await self.tools.execute_with_fallback("get_today_features", {"date": date})
        return {
            "success": imp_result.success or today_result.success,
            "data": {
                "get_feature_importance": imp_result.data if imp_result.success else {},
                "get_today_features": today_result.data if today_result.success else {},
            },
            "tool": "get_feature_importance+get_today_features",
            "fallback_used": imp_result.fallback_used or today_result.fallback_used,
            "latency_ms": (imp_result.latency_ms or 0) + (today_result.latency_ms or 0),
            "error": ""
        }

    async def _get_importance(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("http://127.0.0.1:8002/s2/features/importance")
                if r.status_code == 200:
                    return r.json()
        except Exception as e:
            logger.warning("Feature importance fetch failed: %s", e)
        return {"status": "error", "ranked": [], "grouped": {}}

    async def _get_today_features(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                url = "http://127.0.0.1:8002/s2/features/today"
                if date:
                    url += f"?date={date}"
                r = await c.get(url)
                if r.status_code == 200:
                    return r.json()
        except Exception as e:
            logger.warning("Today features fetch failed: %s", e)
        return {"status": "error", "features": {}, "interpretations": {}}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        importance_data = raw.get("data", {}) if "data" in raw else raw
        if "get_feature_importance" in importance_data:
            importance_data = importance_data["get_feature_importance"]
        ranked = importance_data.get("ranked", [])
        if not ranked:
            return AgentOpinion(agent=self.name,
                opinion="Feature importance data unavailable; cannot assess model sensitivity.",
                confidence=0.2,
                attribution={"metric": "feature_sensitivity", "root_cause": "no_data", "deviation": 0})

        # Get today's features from the second tool result
        today = {}
        reserved_scenarios = {}
        if isinstance(raw, dict):
            raw_data = raw.get("data", {}) if isinstance(raw.get("data", {}), dict) else {}
            today_raw = raw.get("get_today_features", {}) or raw_data.get("get_today_features", {})
            today = today_raw.get("features", {})
            reserved_scenarios = today_raw.get("reserved_scenario_features", {})

        # Build importance lookup
        imp_map = {item["feature"]: item["importance"] for item in ranked}

        # Find activated high-impact features
        activated = []
        for feat in HIGH_IMPACT_FEATURES:
            feat_val = today.get(feat, 0)
            imp = imp_map.get(feat, 0)
            if feat_val and feat_val > 0 and imp > 0:
                label = FEATURE_LABELS.get(feat, feat)
                activated.append({"feature": feat, "label": label, "importance": imp, "value": feat_val})

        activated.sort(key=lambda x: x["importance"], reverse=True)
        scenario_context = []
        for feat, meta in reserved_scenarios.items():
            if isinstance(meta, dict) and (meta.get("active") or meta.get("value")):
                scenario_context.append({
                    "feature": feat,
                    "label": RESERVED_SCENARIO_LABELS.get(feat, meta.get("label", feat)),
                    "importance": None,
                    "value": meta.get("value", 1),
                    "description": meta.get("description", ""),
                })

        # Top 5 features by importance
        top5 = [{"feature": item["feature"], "label": FEATURE_LABELS.get(item["feature"], item["feature"]),
                 "importance": item["importance"]} for item in ranked[:5]]

        # Build opinion
        if activated:
            active_str = "; ".join(
                f"{chr(10)}{chr(9)}- {a['label']} (importance {a['importance']*100:.1f}%, active={'yes' if a['value'] else 'no'})"
                for a in activated[:5]
            )
            top_activated = activated[0]
            opinion = (
                f"Demand sensitivity: top-5 drivers are "
                + ", ".join(f"{t['label']} ({t['importance']*100:.1f}%)" for t in top5)
                + f". Today's activated high-impact features:{active_str}. "
                + f"The strongest active driver is {top_activated['label']} "
                + f"at {top_activated['importance']*100:.1f}% importance "
                + f"({'active' if top_activated['value'] else 'inactive'} today), "
                + f"which {'contributes to' if top_activated['value'] else 'would normally drive'} "
                + f"demand shifts."
            )
        else:
            opinion = (
                f"Demand drivers (top-5): "
                + ", ".join(f"{t['label']} ({t['importance']*100:.1f}%)" for t in top5)
                + ". No high-impact external features are active today; "
                + "demand is primarily driven by temporal and lag patterns."
            )
        if scenario_context:
            scenario_str = "; ".join(f"{s['label']} active" for s in scenario_context)
            opinion += (
                f" Reserved scenario context is also active: {scenario_str}. "
                "This is a business event context, not a deployed model feature weight."
            )

        # Recommendations based on active features
        recs = []
        for scenario in scenario_context:
            if scenario["feature"] == "is_new_product":
                recs.append({
                    "action": "For the next 2 weeks: Monitor new product sales daily and compare demand against the forecast window.",
                    "urgency": "medium",
                    "rationale": "New product launch is active as business scenario context, not as a deployed model feature weight."
                })
            elif scenario["feature"] == "is_competitor":
                recs.append({
                    "action": "For this week: Review competitor-response bundles or targeted discounts before changing broad production levels.",
                    "urgency": "high",
                    "rationale": "Competitor activity is active as business scenario context, not as a deployed model feature weight."
                })
        for a in activated[:3]:
            if a["feature"] == "is_rainy" and a["value"]:
                recs.append({
                    "action": "For tomorrow: Increase hot items and promote delivery options",
                    "urgency": "medium",
                    "rationale": f"Rain is a {a['importance']*100:.1f}%-weighted demand driver active today"
                })
            elif a["feature"] == "is_weekend" and a["value"]:
                recs.append({
                    "action": "For tomorrow: Adjust staffing for weekend demand pattern",
                    "urgency": "low",
                    "rationale": f"Weekend pattern at {a['importance']*100:.1f}% importance is active today"
                })
            elif a["feature"] == "is_holiday" and a["value"]:
                recs.append({
                    "action": "For tomorrow: Holiday demand pattern active, verify production quantities",
                    "urgency": "medium",
                    "rationale": f"Holiday at {a['importance']*100:.1f}% importance is active today"
                })

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.75,
            attribution={"metric": "feature_sensitivity", "root_cause": "model_drivers",
                         "deviation": sum(a["importance"] for a in activated) * 100 if activated else 0,
                         "top_features": top5, "activated": activated,
                         "scenario_context": scenario_context},
            recommendations=recs)
