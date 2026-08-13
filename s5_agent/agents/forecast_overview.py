import logging
from datetime import datetime as dt
from urllib.parse import urlencode
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.dashboard_api import fetch_dashboard_json
from s5_agent.core.tool import Tool
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.s5_config.settings import api_url
logger = logging.getLogger("s5.agent.forecast_overview")

class ForecastOverviewAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_forecast_overview",
            description="Get 7-day demand forecast overview",
            parameters={"date": "string"}, primary=True,
            _handler=self._get_overview))

    async def _get_overview(self, date: str = ""):
        return _query_forecast_overview(date)

    async def fetch(self, params):
        date_str = str(params.get("date", "")) if isinstance(params, dict) else ""
        data = _query_forecast_overview(date_str, params)
        return {"success": True, "data": data, "tool": "forecast_overview"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        entries = data.get("entries", [])
        if not entries:
            return AgentOpinion(agent=self.name,
                opinion="No forecast data available for this date.",
                confidence=0.3,
                attribution={"metric": "forecast_overview", "root_cause": "no_data", "deviation": 0})

        by_day = {}
        by_product = {}
        total_qty = 0
        total_rev = 0
        for e in entries:
            dd = e["forecast_date"]
            pn = e["product_name"]
            qty = e["predicted_qty"]
            price = e.get("unit_price", 0)
            rev = qty * price
            by_day.setdefault(dd, {"qty": 0, "rev": 0, "products": 0})
            by_day[dd]["qty"] += qty
            by_day[dd]["rev"] += rev
            by_day[dd]["products"] += 1
            by_product.setdefault(pn, {"qty": 0, "rev": 0, "price": price})
            by_product[pn]["qty"] += qty
            by_product[pn]["rev"] += rev
            total_qty += qty
            total_rev += rev

        sorted_days = sorted(by_day.items())
        direction = _classify_revenue_trend(sorted_days, "rev")

        peak_day = max(sorted_days, key=lambda x: x[1]["rev"])

        sorted_prods = sorted(by_product.items(), key=lambda x: x[1]["qty"], reverse=True)
        top5 = sorted_prods[:5]
        top5_str = "; ".join(
            f"{pn} ({v['qty']:.0f}u, {chr(165)}{v['rev']:.0f})"
            for pn, v in top5
        )

        # Bread vs beverage split
        bev_names = _get_beverage_names()
        bread_qty = sum(v["qty"] for pn, v in by_product.items() if pn not in bev_names)
        bev_qty = sum(v["qty"] for pn, v in by_product.items() if pn in bev_names)
        total = bread_qty + bev_qty
        bread_pct = bread_qty / max(total, 1) * 100

        opinion = (
            f"7-day forecast ({sorted_days[0][0]} to {sorted_days[-1][0]}): "
            f"{total_qty:.0f} total units, {chr(165)}{total_rev:.0f} projected revenue. "
            f"Trend: {direction}. "
            f"Peak day: {peak_day[0]} ({chr(165)}{peak_day[1]['rev']:.0f}). "
            f"Top products: {top5_str}. "
            f"Bread {bread_pct:.0f}% / Beverage {100-bread_pct:.0f}%."
        )

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.80,
            attribution={"metric": "forecast_overview", "root_cause": "forecast_analysis",
                         "deviation": 0, "direction": direction,
                         "total_qty": total_qty, "total_rev": total_rev})

    def analyze_for_graph(self, raw: dict, params: dict) -> AgentOutput:
        opinion = self.analyze(raw, params)
        data = raw.get("data", {}) if isinstance(raw, dict) and "data" in raw else raw
        if not isinstance(data, dict):
            data = {}

        entries = data.get("entries", []) or []
        by_day = {}
        by_product = {}
        total_units = 0.0
        total_revenue = 0.0
        beverage_names = _get_beverage_names()
        for entry in entries:
            day = entry.get("forecast_date", "")
            product = entry.get("product_name", "")
            category = str(entry.get("category", "")).strip().lower()
            if category not in {"bakery", "beverage"}:
                category = "beverage" if product in beverage_names else "bakery"
            quantity = float(entry.get("predicted_qty", 0.0) or 0.0)
            price = float(entry.get("unit_price", 0.0) or 0.0)
            revenue = quantity * price
            by_day.setdefault(
                day,
                {
                    "units": 0.0,
                    "revenue": 0.0,
                    "bakery_units": 0.0,
                    "beverage_units": 0.0,
                    "products": {},
                },
            )
            by_day[day]["units"] += quantity
            by_day[day]["revenue"] += revenue
            by_day[day][f"{category}_units"] += quantity
            by_day[day]["products"].setdefault(
                product,
                {"units": 0.0, "category": category},
            )
            by_day[day]["products"][product]["units"] += quantity
            by_product.setdefault(
                product,
                {"units": 0.0, "revenue": 0.0, "category": category},
            )
            by_product[product]["units"] += quantity
            by_product[product]["revenue"] += revenue
            total_units += quantity
            total_revenue += revenue

        sorted_days = sorted(by_day.items())
        trend = _classify_revenue_trend(sorted_days, "revenue")

        peak_day = max(sorted_days, key=lambda item: item[1]["revenue"])[0] if sorted_days else ""
        top_products = sorted(
            by_product.items(),
            key=lambda item: item[1]["units"],
            reverse=True,
        )[:5]
        top_product_names = [name for name, _ in top_products if name]
        bakery_products = {
            name: values
            for name, values in by_product.items()
            if values["category"] == "bakery"
        }
        beverage_products = {
            name: values
            for name, values in by_product.items()
            if values["category"] == "beverage"
        }
        bakery_units = sum(values["units"] for values in bakery_products.values())
        bakery_revenue = sum(values["revenue"] for values in bakery_products.values())
        beverage_units = sum(values["units"] for values in beverage_products.values())
        beverage_revenue = sum(values["revenue"] for values in beverage_products.values())
        top_bakery_products = [
            name
            for name, _ in sorted(
                bakery_products.items(),
                key=lambda item: item[1]["units"],
                reverse=True,
            )[:5]
        ]
        top_beverage_products = [
            name
            for name, _ in sorted(
                beverage_products.items(),
                key=lambda item: item[1]["units"],
                reverse=True,
            )[:5]
        ]
        first_day = sorted_days[0][0] if sorted_days else ""
        first_day_values = by_day.get(first_day, {})
        first_day_products = first_day_values.get("products", {})
        first_day_top_bakery_products = [
            name
            for name, values in sorted(
                first_day_products.items(),
                key=lambda item: item[1]["units"],
                reverse=True,
            )
            if values["category"] == "bakery"
        ][:5]
        first_day_top_beverage_products = [
            name
            for name, values in sorted(
                first_day_products.items(),
                key=lambda item: item[1]["units"],
                reverse=True,
            )
            if values["category"] == "beverage"
        ][:5]
        first_day_bakery_units = float(first_day_values.get("bakery_units", 0.0) or 0.0)
        first_day_beverage_units = float(first_day_values.get("beverage_units", 0.0) or 0.0)
        business_events = [
            event for event in (data.get("business_events", []) or [])
            if isinstance(event, dict) and event.get("active", True)
        ]
        reserved_scenario_features = data.get("reserved_scenario_features", {}) or {}

        evidence_items = [
            EvidenceItem(
                id="forecast_total_units",
                source="forecast_overview",
                description="Total forecast demand units for the planning horizon",
                value=round(total_units, 2),
                metadata={"date": params.get("date", ""), "entry_count": len(entries)},
            ),
            EvidenceItem(
                id="forecast_total_revenue",
                source="forecast_overview",
                description="Total forecast revenue for the planning horizon",
                value=round(total_revenue, 2),
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="forecast_peak_day",
                source="forecast_overview",
                description="Highest-revenue forecast day",
                value=peak_day,
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="forecast_bakery_units",
                source="forecast_overview",
                description="Forecast demand units for baked products in the planning horizon",
                value=round(bakery_units, 2),
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="forecast_beverage_units",
                source="forecast_overview",
                description="Forecast demand units for made-to-order beverages in the planning horizon",
                value=round(beverage_units, 2),
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="forecast_day1_bakery_units",
                source="forecast_overview",
                description="Forecast bakery demand units for the first day of the selected horizon",
                value=round(first_day_bakery_units, 2),
                metadata={"date": first_day},
            ),
            EvidenceItem(
                id="forecast_day1_beverage_units",
                source="forecast_overview",
                description="Forecast made-to-order beverage units for the first day of the selected horizon",
                value=round(first_day_beverage_units, 2),
                metadata={"date": first_day},
            ),
        ]
        if business_events:
            evidence_items.append(
                EvidenceItem(
                    id="business_events_active",
                    source="business_events",
                    description="Active planned business events attached to the selected forecast date",
                    value=len(business_events),
                    metadata={
                        "date": params.get("date", ""),
                        "events": business_events,
                        "reserved_scenario_features": reserved_scenario_features,
                    },
                )
            )

        return AgentOutput(
            agent_name="ForecastOverviewAgent",
            claim=opinion.opinion,
            confidence=float(opinion.confidence),
            metrics={
                "forecast_total_units": round(total_units, 2),
                "forecast_total_revenue": round(total_revenue, 2),
                "forecast_bakery_units": round(bakery_units, 2),
                "forecast_bakery_revenue": round(bakery_revenue, 2),
                "forecast_beverage_units": round(beverage_units, 2),
                "forecast_beverage_revenue": round(beverage_revenue, 2),
                "forecast_day1_date": first_day,
                "forecast_day1_bakery_units": round(first_day_bakery_units, 2),
                "forecast_day1_beverage_units": round(first_day_beverage_units, 2),
                "forecast_day1_top_bakery_products": first_day_top_bakery_products,
                "forecast_day1_top_beverage_products": first_day_top_beverage_products,
                "forecast_trend": trend,
                "forecast_peak_day": peak_day,
                "top_forecast_products": top_product_names,
                "top_bakery_products": top_bakery_products,
                "top_beverage_products": top_beverage_products,
                "business_event_count": len(business_events),
            },
            evidence_items=evidence_items,
            risks=[],
            recommendations=[],
            data_quality=DataQuality(
                freshness="fresh" if entries else "missing",
                completeness=1.0 if entries else 0.0,
                warnings=(
                    ["Forecast demand data is unavailable for the selected horizon."]
                    if not entries
                    else []
                ),
                source_status={
                    "forecast_overview": "fresh" if entries else "missing",
                    "business_events": "fresh" if business_events else "unknown",
                },
            ),
            metadata={
                "top_products": top_product_names,
                "business_events": business_events,
                "reserved_scenario_features": reserved_scenario_features,
            },
        )


def _get_beverage_names():
    return {"latte","americano","cappuccino","mocha","espresso","flat_white",
            "caramel_macchiato","cold_brew","hot_chocolate","matcha_latte",
            "milk_tea","chai_latte","earl_grey","english_breakfast","lemonade"}


def _classify_revenue_trend(sorted_days, revenue_key):
    if len(sorted_days) < 2:
        return "unknown"

    window_size = min(3, len(sorted_days) // 2)
    first_average = sum(
        values[revenue_key] for _, values in sorted_days[:window_size]
    ) / window_size
    last_average = sum(
        values[revenue_key] for _, values in sorted_days[-window_size:]
    ) / window_size

    if last_average > first_average * 1.05:
        return "rising"
    if last_average < first_average * 0.95:
        return "falling"
    return "stable"


def _query_forecast_overview(date_str="", params=None):
    try:
        if not date_str:
            date_str = dt.now().strftime("%Y-%m-%d")
        forecast_url = api_url("s2/forecast") + "?" + urlencode(
            {"days": 7, "date": date_str}
        )
        forecast_payload = fetch_dashboard_json(
            forecast_url,
            params,
            timeout=120,
        )
        forecasts = forecast_payload.get("forecasts", [])
        beverage_names = _get_beverage_names()
        entries = []
        for forecast in forecasts:
            product_name = str(forecast.get("product_name", ""))
            entries.append(
                {
                    "forecast_date": forecast.get("forecast_date", ""),
                    "product_name": product_name,
                    "predicted_qty": float(forecast.get("predicted_demand", 0) or 0),
                    "lower_bound": float(forecast.get("lower_bound", 0) or 0),
                    "upper_bound": float(forecast.get("upper_bound", 0) or 0),
                    "unit_price": float(forecast.get("unit_price", 0) or 0),
                    "category": (
                        "beverage" if product_name in beverage_names else "bakery"
                    ),
                }
            )

        business_events = []
        reserved_scenario_features = {}
        try:
            event_url = api_url("s2/business-events") + "?" + urlencode(
                {"date": date_str}
            )
            event_payload = fetch_dashboard_json(event_url, params, timeout=60)
            business_events = event_payload.get("events", [])
            reserved_scenario_features = event_payload.get(
                "reserved_scenario_features",
                {},
            )
        except Exception as event_error:
            logger.warning("Business event context fetch failed: %s", event_error)

        return {
            "entries": entries,
            "cached": forecast_payload.get("cached", False),
            "business_events": business_events,
            "reserved_scenario_features": reserved_scenario_features,
        }
    except Exception as error:
        logger.warning("ForecastOverview fetch failed: %s", error)
        return {"entries": []}
