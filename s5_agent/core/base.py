# s5_agent/core/base.py
import asyncio, logging, time, os, sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from s5_agent.core.tool import Tool, ToolResult, ToolRegistry

logger = logging.getLogger("s5.agent")

@dataclass
class AgentOpinion:
    agent: str
    opinion: str = ""
    confidence: float = 0.0
    attribution: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    tool_calls: List[Dict] = field(default_factory=list)

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.tools = ToolRegistry()
        self._setup_tools()

    def _setup_tools(self):
        """Override in subclass to register tools via self.tools.register()"""
        pass

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data using primary tool. Override for custom fetch logic."""
        primary = self.tools.get_primary()
        if not primary:
            logger.warning("%s has no primary tools", self.name)
            return {}
        tool = primary[0]
        result = await self.tools.execute_with_fallback(tool.name, params)
        return {"success": result.success, "data": result.data if result.success else {},
                "tool": result.tool_name, "fallback_used": result.fallback_used,
                "latency_ms": result.latency_ms, "error": result.error if not result.success else ""}

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any],
                context: str = "", history: str = "", key_metrics: Dict = None) -> AgentOpinion:
        """Override in subclass. raw = fetch() output, context = upstream agent opinions."""
        return AgentOpinion(agent=self.name, opinion="", confidence=0.0)

    async def run(self, params: Dict[str, Any], context: str = "",
                  history: str = "", key_metrics: Dict = None) -> AgentOpinion:
        t0 = time.perf_counter()
        try:
            raw = await asyncio.wait_for(self.fetch(params), timeout=15)
        except asyncio.TimeoutError:
            logger.warning("%s fetch timed out", self.name)
            raw = {"success": False, "data": {}, "error": "timeout"}
        except Exception as e:
            logger.warning("%s fetch failed: %s", self.name, e)
            raw = {"success": False, "data": {}, "error": str(e)}
        result = self.analyze(raw, params, context, history, key_metrics)
        result.agent = self.name
        result.data = raw
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result
