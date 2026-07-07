from __future__ import annotations

from s5_agent.schemas.agent_output import AgentOutput
from s5_agent.schemas.verification import VerificationReport
from s5_agent.verifier.conflict_checker import find_conflicting_claims
from s5_agent.verifier.evidence_checker import (
    find_missing_evidence,
    find_unsupported_recommendations,
)


def verify_outputs(outputs: list[AgentOutput]) -> VerificationReport:
    unsupported_recommendations = find_unsupported_recommendations(outputs)
    missing_evidence = find_missing_evidence(outputs)
    conflicting_claims = find_conflicting_claims(outputs)
    data_quality_warnings = [
        warning
        for output in outputs
        for warning in output.data_quality.warnings
    ]

    passed = not (
        unsupported_recommendations
        or missing_evidence
        or conflicting_claims
    )

    return VerificationReport(
        passed=passed,
        unsupported_claims=[],
        unsupported_recommendations=unsupported_recommendations,
        conflicting_claims=conflicting_claims,
        missing_evidence=missing_evidence,
        data_quality_warnings=data_quality_warnings,
        confidence_adjustments={},
    )
