from __future__ import annotations

from s5_agent.schemas.agent_output import AgentOutput


def find_unsupported_recommendations(outputs: list[AgentOutput]) -> list[str]:
    evidence_ids = {
        evidence.id
        for output in outputs
        for evidence in output.evidence_items
    }
    unsupported: list[str] = []

    for output in outputs:
        for recommendation in output.recommendations:
            if not recommendation.evidence_ids:
                unsupported.append(recommendation.id)
                continue
            if not any(evidence_id in evidence_ids for evidence_id in recommendation.evidence_ids):
                unsupported.append(recommendation.id)

    return unsupported


def find_missing_evidence(outputs: list[AgentOutput]) -> list[str]:
    return [
        output.agent_name
        for output in outputs
        if output.claim.strip() and not output.evidence_items
    ]
