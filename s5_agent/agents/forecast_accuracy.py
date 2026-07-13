import os, sys, logging

_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.core.dashboard_api import fetch_dashboard_json
logger = logging.getLogger("s5.agent.forecast_accuracy")


class ForecastAccuracyAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_forecast_accuracy",
            description="Get forecast accuracy metrics",
            parameters={}, primary=True,
            _handler=self._get_accuracy))

    async def _get_accuracy(self):
        return _query_accuracy()

    async def fetch(self, params):
        data = _query_accuracy(params)
        return {"success": True, "data": data, "tool": "forecast_accuracy"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        overall = data.get("overall", {})
        if not overall:
            return AgentOpinion(agent=self.name,
                opinion="No accuracy metrics available yet.",
                confidence=0.3,
                attribution={"metric": "forecast_accuracy", "root_cause": "no_data", "deviation": 0})

        wape = overall.get("WAPE", 0)
        coverage = overall.get("conformal_coverage_80", 0)
        avg_width = overall.get("conformal_avg_width", 0)

        opinion = (
            f"Forecast reliability: recent error {wape:.1f}%, "
            f"coverage {coverage:.1f}%, average demand range {avg_width:.1f} units."
        )

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.80,
            attribution={"metric": "forecast_accuracy", "root_cause": "accuracy_report",
                         "WAPE": wape, "coverage": coverage})

    def analyze_for_graph(self, raw: dict, params: dict) -> AgentOutput:
        opinion = self.analyze(raw, params)
        data = raw.get("data", {}) if isinstance(raw, dict) and "data" in raw else raw
        if not isinstance(data, dict):
            data = {}

        overall = data.get("overall", {}) or {}
        wape = float(overall.get("WAPE", 0.0) or 0.0)
        coverage = float(overall.get("conformal_coverage_80", 0.0) or 0.0)
        avg_width = float(overall.get("conformal_avg_width", 0.0) or 0.0)

        evidence_items = [
            EvidenceItem(
                id="forecast_wape",
                source="forecast_accuracy",
                description="Weighted absolute percentage error for recent forecasts",
                value=round(wape, 2),
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="forecast_coverage",
                source="forecast_accuracy",
                description="Recent forecast interval coverage rate",
                value=round(coverage, 2),
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="forecast_accuracy_interval_width",
                source="forecast_accuracy",
                description="Average recent forecast interval width",
                value=round(avg_width, 2),
                metadata={"date": params.get("date", "")},
            ),
        ]

        return AgentOutput(
            agent_name="ForecastAccuracyAgent",
            claim=opinion.opinion,
            confidence=float(opinion.confidence),
            metrics={
                "forecast_wape": round(wape, 2),
                "forecast_coverage": round(coverage, 2),
                "forecast_accuracy_interval_width": round(avg_width, 2),
            },
            evidence_items=evidence_items,
            risks=[],
            recommendations=[],
            data_quality=DataQuality(
                freshness="fresh" if overall else "missing",
                completeness=1.0 if overall else 0.0,
                source_status={"forecast_accuracy": "fresh" if overall else "missing"},
            ),
        )


def _query_accuracy(params=None):
    try:
        payload = fetch_dashboard_json(
            "http://127.0.0.1:8002/s2/accuracy",
            params or {},
        )
        return payload.get("metrics", {})
    except (OSError, TimeoutError, ValueError) as e:
        logger.warning("ForecastAccuracy fetch failed: %s", e)
        return {"overall": {}}
