import os, sys, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
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
        data = _query_accuracy()
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
            f"Model accuracy: WAPE {wape:.1f}% (lower is better). "
            f"Conformal 80% coverage: {coverage:.1f}% (target 80%). "
            f"Avg prediction interval: {chr(165)}{avg_width:.1f}."
        )

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.80,
            attribution={"metric": "forecast_accuracy", "root_cause": "accuracy_report",
                         "WAPE": wape, "coverage": coverage})


def _query_accuracy():
    try:
        import httpx
        import asyncio
        async def _fetch():
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("http://127.0.0.1:8002/s2/accuracy")
                if r.status_code == 200:
                    d = r.json()
                    return d.get("metrics", {})
                return {}
        return asyncio.run(_fetch())
    except Exception as e:
        logger.warning("ForecastAccuracy fetch failed: %s", e)
        return {"overall": {}}
