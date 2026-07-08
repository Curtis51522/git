from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation
from s5_agent.graph.state import S5GraphState, S5Request


def test_agent_output_accepts_structured_schema_fields():
    evidence = EvidenceItem(
        id="ev_revenue_trend",
        source="RevenueTrendAgent",
        description="Revenue increased compared with the previous period.",
        value="up",
        metadata={"period": "7d", "change": 12.5},
    )
    recommendation = Recommendation(
        id="rec_focus_bundle",
        action="Prioritize a bundle for croissant_chocolate tomorrow.",
        urgency="high",
        time_horizon="tomorrow",
        rationale="The recommendation is linked to the revenue trend evidence.",
        expected_impact="Higher attachment rate",
        claim_ids=["claim_revenue_trend"],
        evidence_ids=[evidence.id],
    )
    data_quality = DataQuality(
        freshness="fresh",
        completeness=0.95,
        warnings=[],
        limitations=["No competitor pricing data was available."],
        source_status={"orders": "fresh"},
    )

    output = AgentOutput(
        agent_name="RevenueTrendAgent",
        claim="Revenue momentum is positive.",
        confidence=0.86,
        metrics={
            "revenue_change_pct": 12.5,
            "trend_direction": "up",
            "sample_days": 7,
        },
        evidence_items=[evidence],
        recommendations=[recommendation],
        data_quality=data_quality,
    )

    assert output.claim == "Revenue momentum is positive."
    assert output.metrics["revenue_change_pct"] == 12.5
    assert output.metrics["trend_direction"] == "up"
    assert output.evidence_items[0].id == "ev_revenue_trend"
    assert output.recommendations[0].evidence_ids == ["ev_revenue_trend"]
    assert output.data_quality.freshness == "fresh"


def test_recommendation_links_to_evidence_ids():
    recommendation = Recommendation(
        id="rec_material_order",
        action="Order flour for this week.",
        urgency="medium",
        time_horizon="this_week",
        rationale="The linked evidence shows limited stock coverage.",
        expected_impact=240.0,
        claim_ids=["claim_low_coverage"],
        evidence_ids=["ev_stock_days", "ev_weekly_consumption"],
    )

    assert recommendation.evidence_ids == ["ev_stock_days", "ev_weekly_consumption"]


def test_graph_state_defaults_include_raw_inputs_and_runtime_containers():
    state = S5GraphState(
        request=S5Request(
            query="Check inventory",
            params={"product": "croissant"},
            lang="en",
        ),
        template_id="inventory_diagnosis",
    )

    assert state.request.query == "Check inventory"
    assert state.request.params == {"product": "croissant"}
    assert state.request.lang == "en"
    assert state.template_id == "inventory_diagnosis"
    assert state.raw_inputs == {}
    assert state.agent_outputs == {}
    assert state.errors == []
