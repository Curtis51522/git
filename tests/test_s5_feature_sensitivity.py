from s5_agent.agents.feature_sensitivity import FeatureSensitivityAgent


def test_feature_sensitivity_treats_reserved_scenario_as_context_not_model_feature():
    agent = FeatureSensitivityAgent("FeatureSensitivityAgent")
    raw = {
        "data": {
            "get_feature_importance": {
                "status": "ok",
                "model": "quantile_model_q50",
                "ranked": [
                    {"feature": "lag_7_avg", "importance": 0.30},
                    {"feature": "is_rainy", "importance": 0.20},
                    {"feature": "daily_tickets", "importance": 0.10},
                ],
            },
            "get_today_features": {
                "status": "ok",
                "features": {
                    "lag_7_avg": 12.0,
                    "is_rainy": 0,
                    "daily_tickets": 48,
                },
                "reserved_scenario_features": {
                    "is_new_product": {"active": False},
                    "is_competitor": {
                        "active": True,
                        "label": "Nearby competitor opening or promotion response window",
                        "description": "A nearby competitor opening or promotion response is active.",
                    },
                },
                "interpretations": {},
            },
        }
    }

    result = agent.analyze(raw, {"date": "2026-07-07"})

    assert "reserved scenario context" in result.opinion.lower()
    assert "competitor activity" in result.opinion
    assert result.attribution["scenario_context"][0]["feature"] == "is_competitor"
    assert result.attribution["scenario_context"][0]["importance"] is None
    assert result.recommendations[0]["action"].startswith("For this week")
    assert "feature importance" not in result.recommendations[0]["rationale"].lower()


def test_feature_sensitivity_reads_business_events_from_today_features():
    from s5_agent.agents.feature_sensitivity import FeatureSensitivityAgent

    agent = FeatureSensitivityAgent("FeatureSensitivityAgent")
    raw = {
        "data": {
            "get_feature_importance": {
                "status": "ok",
                "model": "quantile_model_q50",
                "ranked": [
                    {"feature": "lag_7_avg", "importance": 0.30},
                    {"feature": "is_top3", "importance": 0.20},
                    {"feature": "lag_1", "importance": 0.10},
                ],
            },
            "get_today_features": {
                "status": "ok",
                "features": {"lag_7_avg": 12.0, "is_top3": 0, "lag_1": 10.0},
                "business_events": [
                    {
                        "id": 1,
                        "event_type": "competitor_activity",
                        "label": "Competitor activity",
                        "start_date": "2026-07-01",
                        "end_date": "2026-07-14",
                        "products": ["croissant"],
                        "discount_pct": 10.0,
                        "note": "Nearby store opening promotion",
                        "active": True,
                    }
                ],
                "reserved_scenario_features": {
                    "is_new_product": {"active": False, "model_input": False},
                    "is_competitor": {
                        "active": True,
                        "value": 1,
                        "model_input": False,
                        "events": [
                            {
                                "id": 1,
                                "event_type": "competitor_activity",
                                "active": True,
                            }
                        ],
                    },
                },
                "interpretations": {},
            },
        }
    }

    result = agent.analyze(raw, {"date": "2026-07-07"})

    assert result.attribution["scenario_context"][0]["feature"] == "is_competitor"
    assert result.attribution["scenario_context"][0]["importance"] is None
    assert "not a deployed model feature weight" in result.opinion
    assert result.recommendations
