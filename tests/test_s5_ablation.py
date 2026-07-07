from s5_agent.evaluation.ablation import compare_ablation_variants, evaluate_variant_outputs


def test_evaluate_variant_outputs_averages_response_metrics():
    responses = [
        {
            "recommendations": [{"id": "a", "evidence_ids": ["ev1"]}],
            "agent_outputs": [{"confidence": 0.8}],
            "verification_report": {"passed": True, "conflicting_claims": []},
        },
        {
            "recommendations": [{"id": "b", "evidence_ids": []}],
            "agent_outputs": [{"confidence": 0.6}],
            "verification_report": {"passed": False, "conflicting_claims": [{"id": "c"}]},
        },
    ]

    result = evaluate_variant_outputs("without_verifier", responses)

    assert result["variant"] == "without_verifier"
    assert result["num_cases"] == 2
    assert result["evidence_coverage"] == 0.5
    assert result["unsupported_recommendation_rate"] == 0.5
    assert result["verification_pass_rate"] == 0.5
    assert result["conflict_rate"] == 0.5
    assert result["mean_agent_confidence"] == 0.7


def test_compare_ablation_variants_reports_delta_from_baseline():
    variants = {
        "rule_only": [
            {
                "recommendations": [{"id": "a", "evidence_ids": []}],
                "agent_outputs": [{"confidence": 0.4}],
                "verification_report": {"passed": False, "conflicting_claims": []},
            }
        ],
        "proposed": [
            {
                "recommendations": [{"id": "b", "evidence_ids": ["ev1"]}],
                "agent_outputs": [{"confidence": 0.8}],
                "verification_report": {"passed": True, "conflicting_claims": []},
            }
        ],
    }

    rows = compare_ablation_variants(variants, baseline_name="rule_only")

    assert [row["variant"] for row in rows] == ["rule_only", "proposed"]
    assert rows[0]["decision_quality_delta"] == 0.0
    assert rows[1]["decision_quality_delta"] > 0.0
