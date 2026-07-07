from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from s5_agent.schemas.agent_output import AgentOutput
from s5_agent.schemas.evidence import EvidenceGraph
from s5_agent.schemas.recommendation import Recommendation
from s5_agent.schemas.verification import VerificationReport


class S5AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=1)
    agent_outputs: list[AgentOutput] = Field(default_factory=list)
    evidence_graph: EvidenceGraph = Field(default_factory=EvidenceGraph)
    verification_report: VerificationReport
    recommendations: list[Recommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
