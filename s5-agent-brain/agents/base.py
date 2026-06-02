# Base agent class - every agent inherits from this
import asyncio, httpx, logging, time
from typing import Dict, Any, Optional

logger = logging.getLogger("s5.agent")


class BaseAgent:
    """Each agent: fetch data -> analyze -> return structured opinion.
    Phase 3: accepts optional history for memory-aware analysis."""

    def __init__(self, name: str):
        self.name = name

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch raw data from external services (S1/S2/S3). Override per agent."""
        return {}

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any],
                history: str = "", key_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Transform raw data into agent opinion. Override per agent.
        history: compact text summary of prior turns.
        key_metrics: extracted numerical trends from recent history."""
        return {"opinion": "", "confidence": 0.0, "constraints": [], "data": raw}

    async def run(self, params: Dict[str, Any],
                  history: str = "", key_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Full pipeline: fetch -> analyze, with timeout."""
        t0 = time.perf_counter()
        try:
            raw = await asyncio.wait_for(self.fetch(params), timeout=15)
        except asyncio.TimeoutError:
            logger.warning("%s fetch timed out, using empty data", self.name)
            raw = {}
        except Exception as e:
            logger.warning("%s fetch failed: %s", self.name, e)
            raw = {}
        result = self.analyze(raw, params, history, key_metrics)
        result["agent"] = self.name
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result