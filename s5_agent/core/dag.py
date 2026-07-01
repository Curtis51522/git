# s5_agent/core/dag.py
import asyncio, logging, time
from typing import Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("s5.dag")

@dataclass
class DAGNode:
    agent_name: str
    phase: int
    dependencies: List[str] = field(default_factory=list)

@dataclass
class DAGTemplate:
    intent: str
    nodes: List[DAGNode]
    description: str = ""

class DAGExecutor:
    def __init__(self, agents: Dict[str, Any], memory=None):
        self.agents = agents
        self.memory = memory
        self.max_phases = 4
        self.total_timeout = 120
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 200

    async def execute(self, template: DAGTemplate, params: Dict,
                      query: str = "", intent: str = "") -> Dict[str, Any]:
        # Check cache
        cache_key = self._make_cache_key(intent, params)
        if cache_key in self._cache:
            logger.info("DAG cache HIT for %s", cache_key)
            cached = self._cache[cache_key]
            cached["cache_hit"] = True
            return cached

        t0 = time.perf_counter()
        results: Dict[str, Any] = {}
        audit_trail: List[Dict] = []
        max_phase = max(n.phase for n in template.nodes)

        for phase in range(1, max_phase + 1):
            phase_nodes = [n for n in template.nodes if n.phase == phase]
            if not phase_nodes:
                continue
            logger.info("DAG Phase %d: %d agents", phase, len(phase_nodes))
            phase_tasks = {}
            for node in phase_nodes:
                if node.agent_name not in self.agents:
                    logger.warning("Agent %s not registered, skipping", node.agent_name)
                    continue
                upstream_context = self._build_context(node, results)
                history = ""
                if self.memory:
                    history = self.memory.get_context_for_agent(node.agent_name, params)
                task = self.agents[node.agent_name].run(
                    params=params, context=upstream_context, history=history
                )
                phase_tasks[node.agent_name] = task

            if phase_tasks:
                phase_results = await asyncio.gather(*phase_tasks.values(), return_exceptions=True)
                for agent_name, result in zip(phase_tasks.keys(), phase_results):
                    if isinstance(result, Exception):
                        logger.error("Agent %s crashed: %s", agent_name, result)
                        results[agent_name] = {"agent": agent_name, "opinion": "", "confidence": 0.0,
                                               "error": str(result), "data": {}}
                    else:
                        results[agent_name] = result.__dict__ if hasattr(result, '__dict__') else result
                    audit_trail.append({"agent": agent_name, "phase": phase, "elapsed_ms": results[agent_name].get("elapsed_ms", 0)})

            if time.perf_counter() - t0 > self.total_timeout:
                logger.warning("DAG total timeout reached at phase %d", phase)
                break

        output = {
            "results": results,
            "audit_trail": audit_trail,
            "total_elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "phases_executed": max_phase,
            "intent": intent,
            "query": query,
            "cache_hit": False
        }
        # Write to cache
        self._cache[cache_key] = output
        if len(self._cache) > self._cache_max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return output

    def _make_cache_key(self, intent: str, params: Dict) -> str:
        date = str(params.get("date", "")) if params else ""
        module = str(params.get("module", "")) if params else ""
        return f"{intent}:{date}:{module}"

    def _build_context(self, node: DAGNode, results: Dict) -> str:
        if not node.dependencies:
            return ""
        parts = []
        for dep in node.dependencies:
            if dep in results:
                r = results[dep]
                opinion = r.get("opinion", "") if isinstance(r, dict) else getattr(r, "opinion", "")
                confidence = r.get("confidence", 0) if isinstance(r, dict) else getattr(r, "confidence", 0)
                if opinion:
                    parts.append(f"[{dep} (confidence={confidence:.2f})]: {opinion}")
        return "\n".join(parts) if parts else ""
