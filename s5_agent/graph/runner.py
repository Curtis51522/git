from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from s5_agent.graph.builder import build_s5_graph
from s5_agent.graph.state import S5GraphState, S5Request
from s5_agent.schemas.response import S5AnalysisResponse


def _normalize_state(state: S5GraphState | dict[str, Any]) -> S5GraphState:
    if isinstance(state, S5GraphState):
        return state
    return S5GraphState.model_validate(state)


def state_to_response(
    state: S5GraphState | dict[str, Any],
    elapsed_ms: int | float,
) -> S5AnalysisResponse:
    graph_state = _normalize_state(state)
    warnings = list(graph_state.errors)
    warnings.extend(graph_state.verification_report.data_quality_warnings)

    return S5AnalysisResponse(
        summary=graph_state.synthesis.summary or "No summary available.",
        agent_outputs=list(graph_state.agent_outputs.values()),
        evidence_graph=graph_state.evidence_graph,
        verification_report=graph_state.verification_report,
        recommendations=list(graph_state.synthesis.recommendations),
        warnings=warnings,
        metadata={
            "intent": graph_state.template_id,
            "template": graph_state.template_id,
            "total_elapsed_ms": elapsed_ms,
        },
    )


async def run_s5_graph(
    template_id: str,
    request: S5Request,
    raw_inputs: dict[str, dict] | None = None,
) -> S5AnalysisResponse:
    graph = build_s5_graph(template_id)
    graph_request = (
        request if isinstance(request, S5Request) else S5Request.model_validate(request)
    )
    initial_state = S5GraphState(
        request=graph_request,
        template_id=template_id,
        run_id=str(uuid4()),
        raw_inputs=raw_inputs or {},
    )

    started_at = time.perf_counter()
    final_state = await graph.ainvoke(initial_state)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
    return state_to_response(final_state, elapsed_ms)
