from s5_agent.evaluation.metrics import (
    conflict_rate,
    evidence_coverage,
    evaluate_response,
    mean_agent_confidence,
    unsupported_recommendation_rate,
    verification_pass_rate,
)
from s5_agent.evaluation.baselines.rule_only import inventory_rule_only


def test_evidence_coverage_returns_one_for_empty_recommendations():
    assert evidence_coverage([]) == 1.0


def test_evidence_coverage_counts_truthy_evidence_ids():
    recommendations = [
        {"id": "a", "evidence_ids": ["ev1"]},
        {"id": "b", "evidence_ids": []},
        {"id": "c"},
    ]

    assert evidence_coverage(recommendations) == 0.3333


def test_unsupported_recommendation_rate_returns_zero_for_empty_recommendations():
    assert unsupported_recommendation_rate([]) == 0.0


def test_unsupported_recommendation_rate_counts_missing_or_empty_evidence_ids():
    recommendations = [
        {"id": "a", "evidence_ids": ["ev1"]},
        {"id": "b", "evidence_ids": []},
        {"id": "c"},
        {"id": "d", "evidence_ids": ["ev2"]},
    ]

    assert unsupported_recommendation_rate(recommendations) == 0.5


def test_inventory_rule_only_returns_response_like_shape():
    result = inventory_rule_only(
        {
            "id": "inventory_shortage",
            "product": "croissant",
            "inventory": {"day1": 12, "day2": 4, "day3": 0},
            "day1_available": 12,
        }
    )

    assert isinstance(result["summary"], str)
    assert "croissant" in result["summary"]
    assert result["recommendations"] == [
        {
            "id": "rule_clear_day1",
            "action": "Apply Day-1 clearance promotion.",
            "evidence_ids": ["scenario_day1_available"],
        }
    ]


def test_evaluate_response_returns_research_metrics():
    response = {
        "recommendations": [
            {"id": "a", "evidence_ids": ["ev1"]},
            {"id": "b", "evidence_ids": []},
        ],
        "agent_outputs": [
            {"agent_name": "A", "confidence": 0.8},
            {"agent_name": "B", "confidence": 0.6},
        ],
        "verification_report": {
            "passed": False,
            "conflicting_claims": [{"a": "A", "b": "B"}],
            "data_quality_warnings": ["stale source"],
        },
    }

    metrics = evaluate_response(response)

    assert metrics == {
        "evidence_coverage": 0.5,
        "unsupported_recommendation_rate": 0.5,
        "verification_passed": 0.0,
        "conflict_present": 1.0,
        "data_quality_warning_count": 1,
        "mean_agent_confidence": 0.7,
        "decision_quality_score": 0.34,
    }


def test_batch_rates_support_ablation_tables():
    responses = [
        {
            "agent_outputs": [{"confidence": 0.9}],
            "verification_report": {"passed": True, "conflicting_claims": []},
        },
        {
            "agent_outputs": [{"confidence": 0.5}],
            "verification_report": {"passed": False, "conflicting_claims": [{"id": "c"}]},
        },
    ]

    assert verification_pass_rate(responses) == 0.5
    assert conflict_rate(responses) == 0.5
    assert mean_agent_confidence(responses) == 0.7
