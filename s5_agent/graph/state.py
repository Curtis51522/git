from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from s5_agent.schemas.agent_output import AgentOutput
from s5_agent.schemas.evidence import EvidenceGraph
from s5_agent.schemas.recommendation import Recommendation
from s5_agent.schemas.verification import VerificationReport


class S5Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    module: str = ""
    params: dict = Field(default_factory=dict)
    lang: str = "en"
    force_refresh: bool = False


class S5Synthesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    recommendations: list[Recommendation] = Field(default_factory=list)


class S5GraphState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: S5Request
    template_id: str
    run_id: str = ""
    raw_inputs: dict[str, dict] = Field(default_factory=dict)
    agent_outputs: dict[str, AgentOutput] = Field(default_factory=dict)
    evidence_graph: EvidenceGraph = Field(default_factory=EvidenceGraph)
    verification_report: VerificationReport = Field(
        default_factory=lambda: VerificationReport(passed=False)
    )
    synthesis: S5Synthesis = Field(default_factory=S5Synthesis)
    errors: list[str] = Field(default_factory=list)
    timings: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, Any] = Field(default_factory=dict)
