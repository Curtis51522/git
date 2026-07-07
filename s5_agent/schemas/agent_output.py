from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from s5_agent.schemas.evidence import EvidenceItem, EvidenceValue
from s5_agent.schemas.recommendation import Recommendation


DataFreshness = Literal["fresh", "stale", "missing", "unknown"]


class DataQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    freshness: DataFreshness = "unknown"
    completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_status: dict[str, DataFreshness] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str = Field(..., min_length=1)
    claim: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    metrics: dict[str, EvidenceValue] = Field(default_factory=dict)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    data_quality: DataQuality = Field(default_factory=DataQuality)
    limitations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
