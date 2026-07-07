from __future__ import annotations

from s5_agent.schemas.agent_output import AgentOutput
from s5_agent.schemas.evidence import EvidenceEdge, EvidenceGraph, EvidenceNode


def build_evidence_graph(outputs: dict[str, AgentOutput]) -> EvidenceGraph:
    nodes: list[EvidenceNode] = []
    edges: list[EvidenceEdge] = []
    node_ids: set[str] = set()

    def add_node(node: EvidenceNode) -> None:
        if node.id in node_ids:
            return
        nodes.append(node)
        node_ids.add(node.id)

    for output in outputs.values():
        claim_id = f"claim:{output.agent_name}"
        add_node(
            EvidenceNode(
                id=claim_id,
                type="claim",
                label=output.claim,
                value=output.confidence,
                metadata={"agent_name": output.agent_name},
            )
        )

        for item in output.evidence_items:
            source_id = f"source:{item.source}"
            metric_id = f"metric:{item.id}"

            add_node(
                EvidenceNode(
                    id=source_id,
                    type="data_source",
                    label=item.source,
                )
            )
            add_node(
                EvidenceNode(
                    id=metric_id,
                    type="metric",
                    label=item.description,
                    value=item.value,
                    metadata={"source": item.source, **item.metadata},
                )
            )
            edges.append(
                EvidenceEdge(
                    source_id=source_id,
                    target_id=metric_id,
                    type="supports",
                )
            )
            edges.append(
                EvidenceEdge(
                    source_id=metric_id,
                    target_id=claim_id,
                    type="supports",
                    confidence=output.confidence,
                )
            )

        for index, risk in enumerate(output.risks):
            risk_id = f"risk:{output.agent_name}:{index}"
            add_node(
                EvidenceNode(
                    id=risk_id,
                    type="risk",
                    label=risk,
                    metadata={"agent_name": output.agent_name},
                )
            )
            edges.append(
                EvidenceEdge(
                    source_id=claim_id,
                    target_id=risk_id,
                    type="causes",
                    confidence=output.confidence,
                )
            )

        for recommendation in output.recommendations:
            recommendation_id = f"recommendation:{recommendation.id}"
            add_node(
                EvidenceNode(
                    id=recommendation_id,
                    type="recommendation",
                    label=recommendation.action,
                    value=recommendation.expected_impact,
                    metadata={
                        "urgency": recommendation.urgency,
                        "time_horizon": recommendation.time_horizon,
                        "rationale": recommendation.rationale,
                        "claim_ids": recommendation.claim_ids,
                        "evidence_ids": recommendation.evidence_ids,
                    },
                )
            )
            for evidence_id in recommendation.evidence_ids:
                edges.append(
                    EvidenceEdge(
                        source_id=f"metric:{evidence_id}",
                        target_id=recommendation_id,
                        type="justifies",
                    )
                )

    return EvidenceGraph(nodes=nodes, edges=edges)
