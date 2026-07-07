import os, sys, logging
from datetime import datetime as dt, timedelta
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from db.mysql_client import get_db
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
        data = _query_forecast_overview(date_str)
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
        if len(sorted_days) >= 2:
            first3 = sum(v["rev"] for _, v in sorted_days[:3])
            last3 = sum(v["rev"] for _, v in sorted_days[3:]) if len(sorted_days) >= 6 else first3
            if last3 > first3 * 1.05:
                direction = "rising"
            elif last3 < first3 * 0.95:
                direction = "falling"
            else:
                direction = "stable"
        else:
            direction = "unknown"

        peak_day = max(sorted_days, key=lambda x: x[1]["rev"])
        valley_day = min(sorted_days, key=lambda x: x[1]["rev"])

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
        for entry in entries:
            day = entry.get("forecast_date", "")
            product = entry.get("product_name", "")
            quantity = float(entry.get("predicted_qty", 0.0) or 0.0)
            price = float(entry.get("unit_price", 0.0) or 0.0)
            revenue = quantity * price
            by_day.setdefault(day, {"units": 0.0, "revenue": 0.0})
            by_day[day]["units"] += quantity
            by_day[day]["revenue"] += revenue
            by_product.setdefault(product, {"units": 0.0, "revenue": 0.0})
            by_product[product]["units"] += quantity
            by_product[product]["revenue"] += revenue
            total_units += quantity
            total_revenue += revenue

        sorted_days = sorted(by_day.items())
        trend = "unknown"
        if len(sorted_days) >= 2:
            first_half = sum(value["revenue"] for _, value in sorted_days[: max(len(sorted_days) // 2, 1)])
            second_half = sum(value["revenue"] for _, value in sorted_days[max(len(sorted_days) // 2, 1):])
            if second_half > first_half:
                trend = "rising"
            elif second_half < first_half:
                trend = "falling"
            else:
                trend = "stable"

        peak_day = max(sorted_days, key=lambda item: item[1]["revenue"])[0] if sorted_days else ""
        top_products = sorted(
            by_product.items(),
            key=lambda item: item[1]["units"],
            reverse=True,
        )[:5]
        top_product_names = [name for name, _ in top_products if name]

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
        ]

        return AgentOutput(
            agent_name="ForecastOverviewAgent",
            claim=opinion.opinion,
            confidence=float(opinion.confidence),
            metrics={
                "forecast_total_units": round(total_units, 2),
                "forecast_total_revenue": round(total_revenue, 2),
                "forecast_trend": trend,
                "forecast_peak_day": peak_day,
                "top_forecast_products": top_product_names,
            },
            evidence_items=evidence_items,
            risks=[],
            recommendations=[],
            data_quality=DataQuality(
                freshness="fresh" if entries else "missing",
                completeness=1.0 if entries else 0.0,
                source_status={"forecast_overview": "fresh" if entries else "missing"},
            ),
            metadata={"top_products": top_product_names},
        )


def _get_beverage_names():
    return {"latte","americano","cappuccino","mocha","espresso","flat_white",
            "caramel_macchiato","cold_brew","hot_chocolate","matcha_latte",
            "milk_tea","chai_latte","earl_grey","english_breakfast","lemonade"}


def _query_forecast_overview(date_str=""):
    try:
        from api.module2_forecast import _do_forecast
        if not date_str:
            date_str = dt.now().strftime("%Y-%m-%d")
        f = _do_forecast(None, 7, use_cache=True, start_date=date_str)
        forecasts = f.get("forecasts", [])

        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT product_name, unit_price FROM products")
        prices = {str(r[0]): float(r[1] or 0) for r in cur.fetchall()}

        entries = []
        for fc in forecasts:
            pn = fc.get("product_name", "")
            entries.append({
                "forecast_date": fc.get("forecast_date", ""),
                "product_name": pn,
                "predicted_qty": float(fc.get("predicted_demand", 0)),
                "lower_bound": float(fc.get("lower_bound", 0)),
                "upper_bound": float(fc.get("upper_bound", 0)),
                "unit_price": prices.get(pn, 0),
            })
        return {"entries": entries, "cached": f.get("cached", False)}
    except Exception as e:
        logger.warning("ForecastOverview fetch failed: %s", e)
        return {"entries": []}
