from __future__ import annotations

from typing import Any

from s5_agent.schemas.evidence import EvidenceGraph


def serialize_evidence_graph(graph: EvidenceGraph) -> dict[str, Any]:
    return graph.model_dump()
