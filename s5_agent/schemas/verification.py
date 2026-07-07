from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    unsupported_recommendations: list[str] = Field(default_factory=list)
    conflicting_claims: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    data_quality_warnings: list[str] = Field(default_factory=list)
    confidence_adjustments: dict[str, float] = Field(default_factory=dict)
