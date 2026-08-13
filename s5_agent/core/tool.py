# s5_agent/core/tool.py
import logging, time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field

logger = logging.getLogger("s5.tool")

@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str = ""
    latency_ms: float = 0.0
    tool_name: str = ""
    fallback_used: bool = False

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str] = field(default_factory=dict)
    primary: bool = True
    fallback: bool = False
    _handler: Optional[Callable] = field(default=None, repr=False)

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        try:
            if self._handler:
                result = self._handler(**params)
                import asyncio as _asyncio
                if _asyncio.iscoroutine(result):
                    result = await result
                data = result
            else:
                data = await self._http_call(params)
            return ToolResult(success=True, data=data, tool_name=self.name,
                            latency_ms=round((time.perf_counter()-t0)*1000,1))
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name,
                            latency_ms=round((time.perf_counter()-t0)*1000,1))

    async def _http_call(self, params: Dict) -> Any:
        raise NotImplementedError("Override _handler or _http_call")

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get_primary(self) -> List[Tool]:
        return [t for t in self._tools.values() if t.primary]

    def get_fallbacks(self) -> List[Tool]:
        return [t for t in self._tools.values() if t.fallback]

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    async def execute_with_fallback(self, tool_name: str, params: Dict) -> ToolResult:
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Tool {tool_name} not found")
        result = await tool.execute(params)
        if result.success:
            return result
        logger.warning("Tool %s failed: %s, trying fallbacks", tool_name, result.error)
        for fb in self.get_fallbacks():
            if fb.name != tool_name:
                r = await fb.execute(params)
                if r.success:
                    r.fallback_used = True
                    return r
        return result
