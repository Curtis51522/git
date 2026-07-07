from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EvidenceValue = str | int | float | bool | None | list[Any] | dict[str, Any]
EvidenceNodeType = Literal[
    "data_source",
    "metric",
    "claim",
    "risk",
    "recommendation",
    "constraint",
    "data_quality_issue",
]
EvidenceEdgeType = Literal[
    "supports",
    "contradicts",
    "depends_on",
    "causes",
    "justifies",
    "limited_by",
]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    value: EvidenceValue = None
    metadata: dict[str, EvidenceValue] = Field(default_factory=dict)


class EvidenceNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    type: EvidenceNodeType
    label: str = Field(..., min_length=1)
    value: EvidenceValue = None
    metadata: dict[str, EvidenceValue] = Field(default_factory=dict)


class EvidenceEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    type: EvidenceEdgeType
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, EvidenceValue] = Field(default_factory=dict)


class EvidenceGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[EvidenceNode] = Field(default_factory=list)
    edges: list[EvidenceEdge] = Field(default_factory=list)
