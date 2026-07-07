from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Urgency = Literal["high", "medium", "low"]
TimeHorizon = Literal[
    "today",
    "tomorrow",
    "this_week",
    "next_7_days",
    "next_30_days",
    "ongoing",
]


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    urgency: Urgency
    time_horizon: TimeHorizon
    rationale: str = Field(..., min_length=1)
    expected_impact: str | int | float | None = None
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
