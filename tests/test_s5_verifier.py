from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation
from s5_agent.verifier.evidence_checker import (
    find_missing_evidence,
    find_unsupported_recommendations,
)
from s5_agent.verifier.report_builder import verify_outputs


def _recommendation(rec_id: str, evidence_ids: list[str]) -> Recommendation:
    return Recommendation(
        id=rec_id,
        action=f"Take action for {rec_id}.",
        urgency="high",
        time_horizon="today",
        rationale="The action should be supported by linked evidence.",
        expected_impact="Improves operating reliability.",
        claim_ids=["claim_one"],
        evidence_ids=evidence_ids,
    )


def _evidence(evidence_id: str, source: str = "InventoryAgent") -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        source=source,
        description=f"Evidence item {evidence_id}.",
        value="observed",
        metadata={},
    )


def _output(
    agent_name: str,
    claim: str,
    evidence_items: list[EvidenceItem] | None = None,
    recommendations: list[Recommendation] | None = None,
    data_quality: DataQuality | None = None,
) -> AgentOutput:
    return AgentOutput(
        agent_name=agent_name,
        claim=claim,
        confidence=0.8,
        metrics={},
        evidence_items=evidence_items or [],
        recommendations=recommendations or [],
        data_quality=data_quality or DataQuality(),
    )


def test_find_unsupported_recommendations_flags_empty_and_unknown_evidence_ids():
    supported_evidence = _evidence("ev_stock")
    inventory_output = _output(
        "InventoryAgent",
        "Stock coverage is limited.",
        evidence_items=[supported_evidence],
        recommendations=[
            _recommendation("rec_supported", ["ev_stock"]),
            _recommendation("rec_empty", []),
            _recommendation("rec_unknown", ["ev_missing"]),
        ],
    )

    unsupported = find_unsupported_recommendations([inventory_output])

    assert unsupported == ["rec_empty", "rec_unknown"]


def test_find_missing_evidence_flags_agents_with_claims_but_no_evidence():
    outputs = [
        _output("RevenueTrendAgent", "Revenue trend is declining.", evidence_items=[]),
        _output("InventoryAgent", "Stock coverage is limited.", evidence_items=[_evidence("ev_stock")]),
    ]

    missing = find_missing_evidence(outputs)

    assert missing == ["RevenueTrendAgent"]


def test_verify_outputs_builds_report_from_verifier_checks_and_quality_warnings():
    data_quality = DataQuality(
        freshness="stale",
        completeness=0.5,
        warnings=["Revenue data is stale."],
        limitations=[],
        source_status={"revenue": "stale"},
    )
    outputs = [
        _output(
            "RevenueTrendAgent",
            "Revenue trend is declining.",
            evidence_items=[],
            recommendations=[_recommendation("rec_review_revenue", [])],
            data_quality=data_quality,
        ),
        _output("InventoryAgent", "Stock coverage is limited.", evidence_items=[_evidence("ev_stock")]),
    ]

    report = verify_outputs(outputs)

    assert report.passed is False
    assert report.unsupported_claims == []
    assert report.unsupported_recommendations == ["rec_review_revenue"]
    assert report.missing_evidence == ["RevenueTrendAgent"]
    assert report.conflicting_claims == []
    assert report.data_quality_warnings == ["Revenue data is stale."]
    assert report.confidence_adjustments == {}
