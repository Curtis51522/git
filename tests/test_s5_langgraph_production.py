import asyncio

from s5_agent.agents.forecast_accuracy import ForecastAccuracyAgent
from s5_agent.agents.forecast_overview import ForecastOverviewAgent
from s5_agent.agents.forecast_uncertainty import ForecastUncertaintyAgent
from s5_agent.agents.material_procurement import MaterialProcurementAgent
from s5_agent.agents.production_plan import ProductionPlanAgent
from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request


def test_production_advice_uses_full_forecast_l1_agent_outputs():
    raw = {
        "forecast_overview": {
            "data": {
                "entries": [
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "croissant",
                        "predicted_qty": 500.0,
                        "lower_bound": 70.0,
                        "upper_bound": 130.0,
                        "unit_price": 12.0,
                    },
                    {
                        "forecast_date": "2026-07-08",
                        "product_name": "baguette",
                        "predicted_qty": 400.0,
                        "lower_bound": 60.0,
                        "upper_bound": 100.0,
                        "unit_price": 10.0,
                    },
                ]
            }
        },
        "forecast_uncertainty": {
            "data": {
                "products": [
                    {"name": "croissant", "avg_width": 60.0, "avg_qty": 100.0, "avg_price": 12.0},
                    {"name": "baguette", "avg_width": 40.0, "avg_qty": 80.0, "avg_price": 10.0},
                ]
            }
        },
        "production": {
            "data": {
                "grid": [{"date": "2026-07-07", "bake_qty": 180}],
                "dates": ["2026-07-07"],
                "buffer": 1.4,
                "day1_stock_total": 25,
                "weekly_summary": {
                    "total_bake": 420,
                    "total_revenue": 2600.0,
                    "total_profit": 780.0,
                    "scenarios": {
                        "q50": {"profit": 780.0, "waste": 20},
                        "q10": {"profit": 100.0, "waste": 60},
                    },
                    "top_products": [("croissant", 180)],
                },
            }
        },
        "materials": {
            "data": {
                "items": {
                    "flour": {
                        "weekly_need": 100.0,
                        "current_stock": 40.0,
                        "to_order": 60.0,
                        "unit": "kg",
                        "alert": "low",
                    }
                }
            }
        },
        "forecast_accuracy": {
            "data": {
                "overall": {
                    "WAPE": 18.5,
                    "conformal_coverage_80": 82.0,
                    "conformal_avg_width": 22.0,
                }
            }
        },
    }
    request = S5Request(
        query="Generate production advice",
        module="forecast",
        params={"date": "2026-07-07", "product": "all"},
    )

    result = asyncio.run(
        run_s5_graph(
            "production_advice",
            request,
            raw_inputs=raw,
        )
    )

    assert result.metadata["template"] == "production_advice"
    assert [output.agent_name for output in result.agent_outputs] == [
        "ForecastOverviewAgent",
        "ForecastUncertaintyAgent",
        "ProductionPlanAgent",
        "MaterialProcurementAgent",
        "ForecastAccuracyAgent",
    ]
    metrics_by_agent = {output.agent_name: output.metrics for output in result.agent_outputs}
    assert metrics_by_agent["ForecastOverviewAgent"]["forecast_total_units"] == 900.0
    assert metrics_by_agent["ProductionPlanAgent"]["total_bake"] == 420
    assert metrics_by_agent["ProductionPlanAgent"]["supply_coverage_pct"] == 49.4
    assert metrics_by_agent["ProductionPlanAgent"]["demand_gap_units"] == 455
    assert metrics_by_agent["ProductionPlanAgent"]["total_available_units"] == 445
    assert metrics_by_agent["MaterialProcurementAgent"]["low_material_count"] == 1
    assert result.summary.startswith("The 7-day production plan is economically positive, but it is deliberately conservative against the demand forecast.")
    assert "900 units" in result.summary
    assert "445 units available" in result.summary
    assert "covering 49.4% of forecast demand" in result.summary
    assert "leaving a 455-unit gap" in result.summary
    assert "Three production choices are visible" in result.summary
    assert "hold at 420 planned units" in result.summary
    assert "expand toward 900 forecast units" in result.summary
    assert "stage production from a 357-unit base" in result.summary
    assert "The staged option is preferred" in result.summary
    assert "release extra bake only after the first 1-2 trading days" in result.summary
    assert "priority should go to croissant and baguette" in result.summary
    assert "Material constraints should shape the release order" in result.summary
    assert "No material is critical yet, but 1 low-stock item still needs attention" in result.summary
    assert "multi-source forecast check" not in result.summary
    assert "croissant, baguette" in result.summary
    assert "Q10" not in result.summary
    assert "Q50" not in result.summary
    assert any(node.id == "claim:ProductionPlanAgent" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:production_total_bake" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:supply_coverage_pct" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:demand_gap_units" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:forecast_total_units" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:material_total_order" for node in result.evidence_graph.nodes)


def test_forecast_graph_fetches_data_when_raw_inputs_missing(monkeypatch):
    async def fake_overview_fetch(self, params):
        return {
            "success": True,
            "data": {
                "entries": [
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "croissant",
                        "predicted_qty": 100.0,
                        "lower_bound": 70.0,
                        "upper_bound": 130.0,
                        "unit_price": 12.0,
                    }
                ]
            },
            "tool": "forecast_overview_test",
        }

    async def fake_uncertainty_fetch(self, params):
        return {
            "success": True,
            "data": {
                "products": [
                    {"name": "croissant", "avg_width": 60.0, "avg_qty": 100.0, "avg_price": 12.0}
                ]
            },
            "tool": "forecast_uncertainty_test",
        }

    async def fake_production_fetch(self, params):
        return {
            "success": True,
            "data": {
                "grid": [{"date": "2026-07-07", "bake_qty": 180}],
                "dates": ["2026-07-07"],
                "buffer": 1.4,
                "day1_stock_total": 25,
                "weekly_summary": {
                    "total_bake": 420,
                    "total_revenue": 2600.0,
                    "total_profit": 780.0,
                    "scenarios": {
                        "q50": {"profit": 780.0, "waste": 20},
                        "q10": {"profit": 100.0, "waste": 60},
                    },
                    "top_products": [("croissant", 180)],
                },
            },
            "tool": "production_plan_test",
        }

    async def fake_material_fetch(self, params):
        return {
            "success": True,
            "data": {
                "items": {
                    "flour": {
                        "weekly_need": 100.0,
                        "current_stock": 40.0,
                        "to_order": 60.0,
                        "unit": "kg",
                        "alert": "low",
                    }
                }
            },
            "tool": "material_procurement_test",
        }

    async def fake_accuracy_fetch(self, params):
        return {
            "success": True,
            "data": {
                "overall": {
                    "WAPE": 18.5,
                    "conformal_coverage_80": 82.0,
                    "conformal_avg_width": 22.0,
                }
            },
            "tool": "forecast_accuracy_test",
        }

    monkeypatch.setattr(ForecastOverviewAgent, "fetch", fake_overview_fetch)
    monkeypatch.setattr(ForecastUncertaintyAgent, "fetch", fake_uncertainty_fetch)
    monkeypatch.setattr(ProductionPlanAgent, "fetch", fake_production_fetch)
    monkeypatch.setattr(MaterialProcurementAgent, "fetch", fake_material_fetch)
    monkeypatch.setattr(ForecastAccuracyAgent, "fetch", fake_accuracy_fetch)
    request = S5Request(
        query="Generate production advice",
        module="forecast",
        params={"date": "2026-07-07", "product": "all"},
    )

    result = asyncio.run(run_s5_graph("production_advice", request))

    assert result.summary != "No production plan data available."
    assert len(result.agent_outputs) == 5
    metrics_by_agent = {output.agent_name: output.metrics for output in result.agent_outputs}
    assert metrics_by_agent["ProductionPlanAgent"]["total_bake"] == 420
    assert metrics_by_agent["ProductionPlanAgent"]["supply_coverage_pct"] == 445.0
    assert metrics_by_agent["ProductionPlanAgent"]["demand_gap_units"] == 0
    assert metrics_by_agent["ForecastOverviewAgent"]["forecast_total_units"] == 100.0
    hedge = next(
        recommendation
        for recommendation in result.recommendations
        if "85% base bake" in recommendation.action
    )
    assert "Keep the remaining planned bake flexible" in hedge.action
    assert "100-unit demand forecast" in hedge.action
    assert "Use the first 1-2 trading days as the release gate" in hedge.action
    assert "do not release the contingency bake automatically" in hedge.action
    assert "risk-adjusted capacity signal" in hedge.action
    assert "contingency units" not in hedge.action
    assert "Prioritize the top forecast driver first" in hedge.action
    assert hedge.evidence_ids == [
        "scenario_profit_gap",
        "production_waste_rate_pct",
        "supply_coverage_pct",
        "demand_gap_units",
        "forecast_wape",
        "forecast_coverage",
    ]


def test_forecast_summary_translates_accuracy_into_production_strategy():
    raw = {
        "forecast_overview": {
            "data": {
                "entries": [
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "croissant",
                        "predicted_qty": 500.0,
                        "lower_bound": 70.0,
                        "upper_bound": 130.0,
                        "unit_price": 12.0,
                    }
                ]
            }
        },
        "production": {
            "data": {
                "grid": [{"date": "2026-07-07", "bake_qty": 300}],
                "dates": ["2026-07-07"],
                "buffer": 1.05,
                "day1_stock_total": 100,
                "weekly_summary": {
                    "total_bake": 300,
                    "total_revenue": 3600.0,
                    "total_profit": 1800.0,
                    "scenarios": {
                        "q50": {"profit": 1800.0, "waste": 45},
                        "q10": {"profit": 900.0, "waste": 90},
                    },
                    "top_products": [("croissant", 300)],
                },
            }
        },
        "forecast_accuracy": {
            "data": {
                "overall": {
                    "WAPE": 30.0,
                    "conformal_coverage_80": 78.5,
                    "conformal_avg_width": 5.0,
                }
            }
        },
    }
    request = S5Request(
        query="Generate production advice",
        module="forecast",
        params={"date": "2026-07-07", "product": "all"},
    )

    result = asyncio.run(run_s5_graph("production_advice", request, raw_inputs=raw))

    assert "covering 80.0% of forecast demand" in result.summary
    metrics_by_agent = {output.agent_name: output.metrics for output in result.agent_outputs}
    assert metrics_by_agent["ProductionPlanAgent"]["supply_coverage_pct"] == 80.0
    assert metrics_by_agent["ProductionPlanAgent"]["demand_gap_units"] == 100
    assert "Three production choices are visible" in result.summary
    assert "stage production from a 255-unit base" in result.summary
    assert "a 100-unit supply gap remains" in result.summary
    assert "30.0% recent error means the week should not be locked in at once" in result.summary
    assert "78.5% coverage is useful for release guardrails" in result.summary
    assert "staged production and material readiness checks should guide extra bake releases" in result.summary


def test_forecast_summary_includes_reserved_business_events():
    raw = {
        "forecast_overview": {
            "data": {
                "entries": [
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "bread_roll",
                        "predicted_qty": 120.0,
                        "lower_bound": 90.0,
                        "upper_bound": 150.0,
                        "unit_price": 8.0,
                    },
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "stickbread",
                        "predicted_qty": 80.0,
                        "lower_bound": 60.0,
                        "upper_bound": 110.0,
                        "unit_price": 7.0,
                    },
                ],
                "business_events": [
                    {
                        "id": 6,
                        "event_type": "new_product_launch",
                        "label": "New product launch",
                        "start_date": "2026-07-07",
                        "end_date": "2026-07-07",
                        "products": ["bread_roll"],
                        "discount_pct": 25.0,
                        "note": "Launch campaign",
                        "active": True,
                    },
                    {
                        "id": 7,
                        "event_type": "competitor_activity",
                        "label": "Competitor activity",
                        "start_date": "2026-07-07",
                        "end_date": "2026-07-07",
                        "products": ["stickbread"],
                        "discount_pct": 25.0,
                        "note": "Competitor response",
                        "active": True,
                    },
                    {
                        "id": 8,
                        "event_type": "competitor_activity",
                        "label": "Competitor activity",
                        "start_date": "2026-07-07",
                        "end_date": "2026-07-07",
                        "products": ["stickbread"],
                        "discount_pct": 25.0,
                        "note": "Duplicate competitor response",
                        "active": True,
                    },
                ],
                "reserved_scenario_features": {
                    "is_new_product": {
                        "active": True,
                        "value": 1,
                        "model_input": False,
                    },
                    "is_competitor": {
                        "active": True,
                        "value": 1,
                        "model_input": False,
                    },
                },
            }
        },
        "production": {
            "data": {
                "grid": [{"date": "2026-07-07", "bake_qty": 140}],
                "dates": ["2026-07-07"],
                "buffer": 1.05,
                "day1_stock_total": 20,
                "weekly_summary": {
                    "total_bake": 160,
                    "total_revenue": 1200.0,
                    "total_profit": 700.0,
                    "scenarios": {
                        "q50": {"profit": 700.0, "waste": 10},
                        "q10": {"profit": 200.0, "waste": 30},
                    },
                    "top_products": [("bread_roll", 90), ("stickbread", 70)],
                },
            }
        },
    }
    request = S5Request(
        query="Generate production advice",
        module="forecast",
        params={"date": "2026-07-07", "product": "all"},
    )

    result = asyncio.run(run_s5_graph("production_advice", request, raw_inputs=raw))
    metrics_by_agent = {output.agent_name: output.metrics for output in result.agent_outputs}
    overview_metrics = metrics_by_agent["ForecastOverviewAgent"]

    assert overview_metrics["business_event_count"] == 3
    assert "Two planned business events are active" in result.summary
    assert "New Product Launch" in result.summary
    assert "Bread Roll" in result.summary
    assert "Competitor Activity" in result.summary
    assert "Stickbread" in result.summary
    assert "25% planned discount" in result.summary
    assert result.summary.count("Competitor Activity applies to Stickbread with a 25% planned discount") == 1
    assert "reserved scenario inputs" in result.summary
    assert "not part of the deployed 27-feature forecast model" in result.summary
    event_recommendation = next(
        recommendation
        for recommendation in result.recommendations
        if recommendation.id == "business_event_monitoring"
    )
    assert "Bread Roll and Stickbread" in event_recommendation.action
    assert "first 1-2 trading days" in event_recommendation.action
    assert event_recommendation.evidence_ids == ["business_events_active"]


def test_single_business_event_summary_uses_singular_reserved_scenario_wording():
    from s5_agent.graph.builder import _business_event_summary_sentence

    summary = _business_event_summary_sentence(
        [
            {
                "event_type": "competitor_activity",
                "start_date": "2026-07-08",
                "end_date": "2026-07-10",
                "products": ["sourdough"],
                "discount_pct": 25.0,
                "active": True,
            }
        ]
    )

    assert "One planned business event is active" in summary
    assert "This business event is a reserved scenario input" in summary
    assert "These business events are reserved scenario inputs" not in summary
