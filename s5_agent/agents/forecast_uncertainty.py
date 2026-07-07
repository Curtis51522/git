import os, sys, logging
from datetime import datetime as dt
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
logger = logging.getLogger("s5.agent.forecast_uncertainty")

class ForecastUncertaintyAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_forecast_uncertainty",
            description="Get prediction interval widths per product",
            parameters={"date": "string"}, primary=True,
            _handler=self._get_uncertainty))

    async def _get_uncertainty(self, date: str = ""):
        return _query_uncertainty(date)

    async def fetch(self, params):
        date_str = str(params.get("date", "")) if isinstance(params, dict) else ""
        data = _query_uncertainty(date_str)
        return {"success": True, "data": data, "tool": "forecast_uncertainty"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        products = data.get("products", [])
        if not products:
            return AgentOpinion(agent=self.name,
                opinion="No forecast uncertainty data available.",
                confidence=0.3,
                attribution={"metric": "forecast_uncertainty", "root_cause": "no_data", "deviation": 0})

        prods_sorted = sorted(products, key=lambda x: x["avg_width"], reverse=True)
        top_uncertain = prods_sorted[:5]
        avg_width = sum(p["avg_width"] for p in products) / max(len(products), 1)
        top_str = "; ".join(
            f"{p['name']} ({chr(165)}{p['avg_price']:.0f}, {chr(165)}{p['avg_qty']:.0f} predicted, {chr(165)}{p['avg_width']:.0f} range)"
            for p in top_uncertain
        )

        high_risk = [p for p in top_uncertain if p["avg_qty"] > avg_width * 2]
        risk_note = ""
        if high_risk:
            risk_note = f" High-risk (demand > 2x uncertainty): " + ", ".join(p["name"] for p in high_risk)

        opinion = (
            f"Forecast uncertainty: avg interval width {chr(165)}{avg_width:.0f}. "
            f"Most uncertain products: {top_str}.{risk_note}"
        )

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.75,
            attribution={"metric": "forecast_uncertainty", "deviation": round(avg_width, 1),
                         "avg_width": avg_width, "top_uncertain": [p["name"] for p in top_uncertain]})

    def analyze_for_graph(self, raw: dict, params: dict) -> AgentOutput:
        opinion = self.analyze(raw, params)
        data = raw.get("data", {}) if isinstance(raw, dict) and "data" in raw else raw
        if not isinstance(data, dict):
            data = {}

        products = data.get("products", []) or []
        sorted_products = sorted(
            products,
            key=lambda product: float(product.get("avg_width", 0.0) or 0.0),
            reverse=True,
        )
        top_uncertain = sorted_products[:5]
        avg_width = (
            sum(float(product.get("avg_width", 0.0) or 0.0) for product in products)
            / max(len(products), 1)
        )
        widest_product = top_uncertain[0].get("name", "") if top_uncertain else ""
        top_names = [str(product.get("name", "")) for product in top_uncertain if product.get("name")]

        evidence_items = [
            EvidenceItem(
                id="forecast_avg_interval_width",
                source="forecast_uncertainty",
                description="Average forecast interval width across products",
                value=round(avg_width, 2),
                metadata={"date": params.get("date", ""), "product_count": len(products)},
            ),
            EvidenceItem(
                id="forecast_uncertain_products",
                source="forecast_uncertainty",
                description="Products with the widest forecast intervals",
                value=top_names,
                metadata={"date": params.get("date", "")},
            ),
        ]

        return AgentOutput(
            agent_name="ForecastUncertaintyAgent",
            claim=opinion.opinion,
            confidence=float(opinion.confidence),
            metrics={
                "forecast_avg_interval_width": round(avg_width, 2),
                "widest_uncertainty_product": widest_product,
                "top_uncertain_products": top_names,
            },
            evidence_items=evidence_items,
            risks=["forecast_uncertainty_hotspots"] if top_names else [],
            recommendations=[],
            data_quality=DataQuality(
                freshness="fresh" if products else "missing",
                completeness=1.0 if products else 0.0,
                source_status={"forecast_uncertainty": "fresh" if products else "missing"},
            ),
            metadata={"top_uncertain_products": top_names},
        )


def _query_uncertainty(date_str=""):
    try:
        from api.module2_forecast import _do_forecast
        if not date_str:
            date_str = dt.now().strftime("%Y-%m-%d")
        f = _do_forecast(None, 7, use_cache=True, start_date=date_str)
        forecasts = f.get("forecasts", [])
        by_product = {}
        for fc in forecasts:
            pn = fc.get("product_name", "")
            lo = float(fc.get("lower_bound", 0))
            hi = float(fc.get("upper_bound", 0))
            qty = float(fc.get("predicted_demand", 0))
            by_product.setdefault(pn, {"widths": [], "qties": [], "prices": []})
            by_product[pn]["widths"].append(hi - lo)
            by_product[pn]["qties"].append(qty)
        products = []
        for pn, v in by_product.items():
            if v["widths"]:
                products.append({
                    "name": pn,
                    "avg_width": round(sum(v["widths"]) / len(v["widths"]), 1),
                    "avg_qty": round(sum(v["qties"]) / len(v["qties"]), 1),
                    "avg_price": 0,
                })
        return {"products": products}
    except Exception as e:
        logger.warning("ForecastUncertainty fetch failed: %s", e)
        return {"products": []}
