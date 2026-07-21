import asyncio

from s5_agent.agents import forecast_overview as forecast_overview_module
from s5_agent.agents import forecast_uncertainty as forecast_uncertainty_module
from s5_agent.agents import material_procurement as material_procurement_module
from s5_agent.agents import production_plan as production_plan_module
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
                        "category": "bakery",
                    },
                    {
                        "forecast_date": "2026-07-08",
                        "product_name": "baguette",
                        "predicted_qty": 400.0,
                        "lower_bound": 60.0,
                        "upper_bound": 100.0,
                        "unit_price": 10.0,
                        "category": "bakery",
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
                        "q90": {"profit": 900.0, "waste": 0, "shortage": 200},
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
    assert result.summary.startswith(
        "The seven-day outlook remains economically positive, although the production plan is intentionally conservative relative to forecast demand."
    )
    assert "900 units" in result.summary
    assert "25 Day-1 carryover units give 445 units available" in result.summary
    assert "445 units available" in result.summary
    assert "covering 49.4% of bakery forecast demand" in result.summary
    assert "leaving a 455-unit bakery gap" in result.summary
    assert "Against this outlook" in result.summary
    assert "Because forecast error still affects how much demand will materialize" in result.summary
    assert "early sales should determine whether additional bake capacity is released" in result.summary
    assert "croissant and baguette should be reviewed first" in result.summary
    assert "procurement is confirmed" in result.summary
    assert "no material is critical yet, but 1 low-stock item still needs attention" in result.summary
    assert "the held-out historical evaluation shows 18.5% error and 82.0% coverage" in result.summary
    assert "Three production choices are visible" not in result.summary
    assert "The staged option is preferred" not in result.summary
    assert "release extra bake only after the first 1-2 trading days" not in result.summary
    assert "recent error" not in result.summary.lower()
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


def test_forecast_overview_compares_equal_length_windows_for_trend():
    daily_units = [100, 101, 102, 97, 98, 99, 100]
    entries = [
        {
            "forecast_date": f"2026-07-{15 + index:02d}",
            "product_name": "croissant",
            "predicted_qty": units,
            "unit_price": 1.0,
            "category": "bakery",
        }
        for index, units in enumerate(daily_units)
    ]

    output = ForecastOverviewAgent("ForecastOverviewAgent").analyze_for_graph(
        {"data": {"entries": entries}},
        {"date": "2026-07-15"},
    )

    assert output.metrics["forecast_trend"] == "stable"
    assert "Trend: stable." in output.claim


def test_forecast_agents_read_selected_date_dashboard_sources(monkeypatch):
    calls = []

    def fake_dashboard_fetch(url, params=None, timeout=10):
        calls.append((url, dict(params or {}), timeout))
        if "/s2/forecast" in url:
            return {
                "status": "ok",
                "forecasts": [
                    {
                        "forecast_date": "2026-06-30",
                        "product_name": "croissant",
                        "predicted_demand": 10,
                        "lower_bound": 8,
                        "upper_bound": 13,
                        "unit_price": 12.0,
                    }
                ],
            }
        if "/s2/business-events" in url:
            return {
                "status": "ok",
                "events": [],
                "reserved_scenario_features": {},
            }
        if "/s3/plan/7day" in url:
            return {
                "status": "ok",
                "dashboard_7day": {
                    "dates": ["2026-06-30"],
                    "grid": [{"date": "2026-06-30", "bake_qty": 9}],
                    "buffer_applied": 1.05,
                    "day1_stock_total": 1,
                },
                "weekly_summary": {
                    "total_bake": 9,
                    "total_revenue": 108.0,
                    "total_profit": 70.0,
                    "scenarios": {},
                    "top_products": [["croissant", 9]],
                },
            }
        if "/s3/materials" in url:
            return {
                "status": "ok",
                "dashboard_materials": {
                    "stock_data_available": True,
                    "items": {"Bread Flour": {"alert": "ok"}},
                },
            }
        raise AssertionError(f"Unexpected dashboard URL: {url}")

    for module in (
        forecast_overview_module,
        forecast_uncertainty_module,
        production_plan_module,
        material_procurement_module,
    ):
        monkeypatch.setattr(
            module,
            "fetch_dashboard_json",
            fake_dashboard_fetch,
            raising=False,
        )

    params = {
        "date": "2026-06-30",
        "_authorization": "Bearer manager-token",
    }
    overview = asyncio.run(ForecastOverviewAgent("ForecastOverviewAgent").fetch(params))["data"]
    uncertainty = asyncio.run(
        ForecastUncertaintyAgent("ForecastUncertaintyAgent").fetch(params)
    )["data"]
    production = asyncio.run(ProductionPlanAgent("ProductionPlanAgent").fetch(params))["data"]
    materials = asyncio.run(
        MaterialProcurementAgent("MaterialProcurementAgent").fetch(params)
    )["data"]

    assert overview["entries"][0]["predicted_qty"] == 10.0
    assert uncertainty["products"][0]["avg_width"] == 5.0
    assert production["weekly_summary"]["total_bake"] == 9
    assert production["day1_stock_total"] == 1
    assert materials["items"] == {"Bread Flour": {"alert": "ok"}}
    assert all(call[1]["_authorization"] == "Bearer manager-token" for call in calls)
    assert all(call[2] >= 60 for call in calls)
    assert any("/s2/forecast?days=7&date=2026-06-30" in call[0] for call in calls)
    assert any("/s3/plan/7day?date=2026-06-30" in call[0] for call in calls)
    assert any("/s3/materials?date=2026-06-30" in call[0] for call in calls)


def test_missing_forecast_data_blocks_quantitative_production_advice():
    raw = {
        "forecast_overview": {"data": {"entries": []}},
        "forecast_uncertainty": {"data": {"products": []}},
        "production": {
            "data": {
                "dates": ["2026-06-30"],
                "grid": [{"date": "2026-06-30", "bake_qty": 100}],
                "weekly_summary": {
                    "total_bake": 100,
                    "total_revenue": 1200.0,
                    "total_profit": 800.0,
                    "scenarios": {},
                    "top_products": [["croissant", 100]],
                },
            }
        },
        "materials": {
            "data": {
                "stock_data_available": True,
                "items": {"Bread Flour": {"alert": "ok"}},
            }
        },
        "forecast_accuracy": {
            "data": {"wape": 30.0, "coverage": 78.8, "interval_width": 5.3}
        },
    }
    request = S5Request(
        query="Generate production advice",
        module="forecast",
        params={"date": "2026-06-30", "module": "forecast"},
        lang="en",
        force_refresh=True,
    )

    result = asyncio.run(
        run_s5_graph("production_advice", request, raw_inputs=raw)
    )

    assert "Forecast demand data is unavailable" in result.summary
    assert result.verification_report.passed is False
    assert all(
        recommendation.id != "production_plan_action_1"
        for recommendation in result.recommendations
    )


def test_forecast_recommendation_uses_historical_error_guardrail_not_summed_q90():
    raw = {
        "forecast_overview": {
            "data": {
                "entries": [
                    {
                        "forecast_date": "2026-06-30",
                        "product_name": "bread_coconut",
                        "predicted_qty": 433.0,
                        "lower_bound": 1.0,
                        "upper_bound": 1447.0,
                        "unit_price": 12.0,
                        "category": "bakery",
                    }
                ]
            }
        },
        "forecast_uncertainty": {"data": {"products": []}},
        "production": {
            "data": {
                "grid": [{"date": "2026-06-30", "bake_qty": 433}],
                "dates": ["2026-06-30"],
                "buffer": 1.05,
                "day1_stock_total": 0,
                "weekly_summary": {
                    "total_bake": 433,
                    "total_revenue": 4751.5,
                    "total_profit": 3326.05,
                    "scenarios": {
                        "q10": {"profit": -2124.35, "waste": 432, "shortage": 0},
                        "q50": {"profit": 3326.05, "waste": 0, "shortage": 0},
                        "q90": {"profit": 446.05, "waste": 0, "shortage": 1041},
                    },
                    "top_products": [("bread_coconut", 65)],
                },
            }
        },
        "materials": {"data": {"items": {}, "stock_data_available": True}},
        "forecast_accuracy": {
            "data": {
                "overall": {
                    "WAPE": 30.0,
                    "conformal_coverage_80": 78.8,
                    "conformal_avg_width": 5.0,
                }
            }
        },
    }
    request = S5Request(
        query="Generate production advice",
        module="forecast",
        params={"date": "2026-06-30", "product": "all"},
    )

    result = asyncio.run(run_s5_graph("production_advice", request, raw_inputs=raw))
    guardrail = next(
        recommendation
        for recommendation in result.recommendations
        if recommendation.id == "historical_error_guardrail"
    )
    production = next(
        output
        for output in result.agent_outputs
        if output.agent_name == "ProductionPlanAgent"
    )

    assert "85% base bake of 368 units" in guardrail.action
    assert "303-563 bakery units" in guardrail.action
    assert "not a prediction interval" in guardrail.action
    assert "next 65 units up to expected demand flexible" in guardrail.action
    assert "no more than 130 units above expected demand" in guardrail.action
    assert "1041" not in guardrail.action
    assert "Q90" not in guardrail.action
    assert "summed product-level" in guardrail.rationale
    assert guardrail.evidence_ids == [
        "forecast_bakery_units",
        "forecast_wape",
        "production_total_bake",
        "supply_coverage_pct",
        "demand_gap_units",
    ]
    assert "forecast_volatility_risk" not in production.risks
    assert all("1041" not in rec.action for rec in result.recommendations)


def test_forecast_recommendations_start_with_selected_day_operating_advice():
    raw = {
        "forecast_overview": {
            "data": {
                "entries": [
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "bread_coconut",
                        "predicted_qty": 30.0,
                        "unit_price": 12.0,
                        "category": "bakery",
                    },
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "macaron",
                        "predicted_qty": 25.0,
                        "unit_price": 10.0,
                        "category": "bakery",
                    },
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "mantequilla",
                        "predicted_qty": 20.0,
                        "unit_price": 9.0,
                        "category": "bakery",
                    },
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "croissant_chocolate",
                        "predicted_qty": 15.0,
                        "unit_price": 12.0,
                        "category": "bakery",
                    },
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "apple_pie",
                        "predicted_qty": 10.0,
                        "unit_price": 14.0,
                        "category": "bakery",
                    },
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "latte",
                        "predicted_qty": 8.0,
                        "unit_price": 16.0,
                        "category": "beverage",
                    },
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "cappuccino",
                        "predicted_qty": 5.0,
                        "unit_price": 18.0,
                        "category": "beverage",
                    },
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "cold_brew",
                        "predicted_qty": 4.0,
                        "unit_price": 18.0,
                        "category": "beverage",
                    },
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "espresso",
                        "predicted_qty": 2.0,
                        "unit_price": 12.0,
                        "category": "beverage",
                    },
                    {
                        "forecast_date": "2026-07-07",
                        "product_name": "americano",
                        "predicted_qty": 1.0,
                        "unit_price": 14.0,
                        "category": "beverage",
                    },
                    {
                        "forecast_date": "2026-07-08",
                        "product_name": "croissant",
                        "predicted_qty": 100.0,
                        "unit_price": 12.0,
                        "category": "bakery",
                    },
                ]
            }
        },
        "forecast_uncertainty": {"data": {"products": []}},
        "production": {
            "data": {
                "grid": [
                    {
                        "date": "2026-07-07",
                        "bake_total": 90,
                        "bake_plan": {"bread_coconut": 55, "macaron": 35},
                    },
                    {
                        "date": "2026-07-08",
                        "bake_total": 110,
                        "bake_plan": {"croissant": 110},
                    },
                ],
                "dates": ["2026-07-07", "2026-07-08"],
                "buffer": 1.05,
                "day1_stock_total": 0,
                "weekly_summary": {
                    "total_bake": 200,
                    "total_revenue": 2320.0,
                    "total_profit": 1500.0,
                    "scenarios": {
                        "q10": {"profit": 900.0, "waste": 20, "shortage": 0},
                        "q50": {"profit": 1500.0, "waste": 0, "shortage": 0},
                        "q90": {"profit": 1700.0, "waste": 0, "shortage": 20},
                    },
                    "top_products": [
                        ("croissant", 110),
                        ("bread_coconut", 55),
                        ("macaron", 35),
                    ],
                },
            }
        },
        "materials": {"data": {"items": {}, "stock_data_available": True}},
        "forecast_accuracy": {
            "data": {
                "overall": {
                    "WAPE": 20.0,
                    "conformal_coverage_80": 80.0,
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
    metrics_by_agent = {
        output.agent_name: output.metrics for output in result.agent_outputs
    }
    today = result.recommendations[0]

    assert metrics_by_agent["ForecastOverviewAgent"]["forecast_day1_date"] == "2026-07-07"
    assert metrics_by_agent["ForecastOverviewAgent"]["forecast_day1_bakery_units"] == 100.0
    assert metrics_by_agent["ForecastOverviewAgent"]["forecast_day1_beverage_units"] == 20.0
    assert metrics_by_agent["ProductionPlanAgent"]["production_day1_bake"] == 90
    assert today.id == "selected_day_production"
    assert today.time_horizon == "today"
    assert "For 2026-07-07" in today.action
    assert "planned bake of 90 bakery units against 100 forecast bakery units" in today.action
    assert "Prioritize bread coconut, macaron, and mantequilla" in today.action
    assert "croissant chocolate" not in today.action
    assert "apple pie" not in today.action
    assert "20 made-to-order beverage units led by latte, cappuccino, and cold brew" in today.action
    assert "espresso" not in today.action
    assert "americano" not in today.action
    assert "approve production above the 90-unit plan only if early sales run ahead" in today.action
    assert "selected date's forecast and production plan" in today.rationale
    assert today.evidence_ids == [
        "forecast_day1_bakery_units",
        "forecast_day1_beverage_units",
        "production_day1_bake",
    ]
    assert any(
        recommendation.id == "historical_error_guardrail"
        for recommendation in result.recommendations
    )
    weekly = next(
        recommendation
        for recommendation in result.recommendations
        if recommendation.id == "historical_error_guardrail"
    )
    assert weekly.action.startswith("Across the seven-day horizon")


def test_forecast_accuracy_identifies_held_out_historical_evaluation():
    output = ForecastAccuracyAgent("ForecastAccuracyAgent").analyze_for_graph(
        {
            "data": {
                "overall": {
                    "WAPE": 29.9,
                    "conformal_coverage_80": 78.9,
                    "conformal_avg_width": 5.0,
                }
            }
        },
        {"date": "2026-07-15"},
    )

    assert "Held-out historical evaluation" in output.claim
    assert "recent" not in output.claim.lower()
    assert all("recent" not in item.description.lower() for item in output.evidence_items)


def test_forecast_uncertainty_uses_demand_units_in_claim():
    output = ForecastUncertaintyAgent("ForecastUncertaintyAgent").analyze_for_graph(
        {
            "data": {
                "products": [
                    {
                        "name": "macaron",
                        "avg_width": 48.0,
                        "avg_qty": 17.0,
                        "avg_price": 10.0,
                    }
                ]
            }
        },
        {"date": "2026-07-15"},
    )

    assert "average interval width 48 units" in output.claim
    assert "17 units predicted" in output.claim
    assert "48-unit range" in output.claim
    assert "\u00a5" not in output.claim


def test_material_procurement_claim_matches_required_stock_threshold():
    output = MaterialProcurementAgent("MaterialProcurementAgent").analyze_for_graph(
        {
            "data": {
                "items": {
                    "Bread Flour": {
                        "weekly_need": 100.0,
                        "current_stock": 120.0,
                        "to_order": 0.0,
                        "unit": "kg",
                        "alert": "ok",
                    },
                    "Butter": {
                        "weekly_need": 50.0,
                        "current_stock": 60.0,
                        "to_order": 0.0,
                        "unit": "kg",
                        "alert": "ok",
                    },
                    "Lids": {
                        "weekly_need": 100.0,
                        "current_stock": 80.0,
                        "to_order": 20.0,
                        "unit": "pcs",
                        "alert": "ok",
                    },
                }
            }
        },
        {"date": "2026-07-15"},
    )

    assert output.metrics["low_material_count"] == 1
    assert "2 materials adequate" in output.claim
    assert "1 material below required stock" in output.claim
    assert "3 materials adequate" not in output.claim


def test_material_procurement_reports_unavailable_stock_data_without_false_shortages():
    from s5_agent.graph.builder import _synthesize_forecast_summary

    overview = ForecastOverviewAgent("ForecastOverviewAgent").analyze_for_graph(
        {
            "data": {
                "entries": [
                    {
                        "forecast_date": "2026-07-15",
                        "product_name": "croissant",
                        "predicted_qty": 10,
                        "unit_price": 12.0,
                        "category": "bakery",
                    }
                ]
            }
        },
        {"date": "2026-07-15"},
    )
    output = MaterialProcurementAgent("MaterialProcurementAgent").analyze_for_graph(
        {
            "data": {
                "items": {},
                "stock_data_available": False,
                "error": "raw_material_stock_unavailable",
            }
        },
        {"date": "2026-07-15"},
    )
    summary = _synthesize_forecast_summary(
        {"forecast_overview": overview, "materials": output}
    )

    assert output.metrics["material_stock_data_available"] is False
    assert output.metrics["critical_material_count"] == 0
    assert output.metrics["low_material_count"] == 0
    assert output.metrics["material_total_order"] == 0.0
    assert output.risks == ["material_data_gap"]
    assert output.data_quality.freshness == "missing"
    assert "Current material stock could not be verified" in output.claim
    assert "material readiness could not be verified" in summary
    assert "does not show critical or low-stock blockers" not in summary
    assert output.recommendations[0].id == "material_stock_data_check"


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
                        "category": "bakery",
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
                        "q90": {"profit": 900.0, "waste": 0, "shortage": 70},
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
    assert "against expected bakery demand of 100 units" in hedge.action
    assert "85% base bake of 85 units" in hedge.action
    assert "81-119 bakery units" in hedge.action
    assert "next 15 units up to expected demand flexible" in hedge.action
    assert "no more than 19 units above expected demand" in hedge.action
    assert "first 1-2 trading days" in hedge.action
    assert "not a prediction interval" in hedge.action
    assert "high-demand scenario" not in hedge.action
    assert "contingency" not in hedge.action
    assert "Prioritize the top forecast driver first" in hedge.action
    assert hedge.evidence_ids == [
        "forecast_bakery_units",
        "forecast_wape",
        "production_total_bake",
        "supply_coverage_pct",
        "demand_gap_units",
    ]


def test_forecast_production_scope_separates_bakery_from_made_to_order_beverages():
    raw = {
        "forecast_overview": {
            "data": {
                "entries": [
                    {
                        "forecast_date": "2026-07-15",
                        "product_name": "croissant",
                        "predicted_qty": 100.0,
                        "lower_bound": 90.0,
                        "upper_bound": 110.0,
                        "unit_price": 12.0,
                        "category": "bakery",
                    },
                    {
                        "forecast_date": "2026-07-15",
                        "product_name": "espresso",
                        "predicted_qty": 50.0,
                        "lower_bound": 40.0,
                        "upper_bound": 60.0,
                        "unit_price": 15.0,
                        "category": "beverage",
                    },
                ]
            }
        },
        "forecast_uncertainty": {"data": {"products": []}},
        "production": {
            "data": {
                "grid": [{"date": "2026-07-15", "bake_qty": 80}],
                "dates": ["2026-07-15"],
                "buffer": 1.05,
                "day1_stock_total": 0,
                "weekly_summary": {
                    "total_bake": 80,
                    "total_revenue": 960.0,
                    "total_profit": 500.0,
                    "scenarios": {
                        "q10": {"profit": -100.0, "waste": 20, "shortage": 0},
                        "q50": {"profit": 500.0, "waste": 0, "shortage": 0},
                        "q90": {"profit": 650.0, "waste": 0, "shortage": 20},
                    },
                    "top_products": [("croissant", 80)],
                },
            }
        },
        "materials": {"data": {"items": {}, "stock_data_available": True}},
        "forecast_accuracy": {
            "data": {
                "overall": {
                    "WAPE": 20.0,
                    "conformal_coverage_80": 80.0,
                    "conformal_avg_width": 5.0,
                }
            }
        },
    }
    request = S5Request(
        query="Generate production advice",
        module="forecast",
        params={"date": "2026-07-15", "product": "all"},
    )

    result = asyncio.run(run_s5_graph("production_advice", request, raw_inputs=raw))
    metrics_by_agent = {output.agent_name: output.metrics for output in result.agent_outputs}

    assert metrics_by_agent["ForecastOverviewAgent"]["forecast_total_units"] == 150.0
    assert metrics_by_agent["ForecastOverviewAgent"]["forecast_bakery_units"] == 100.0
    assert metrics_by_agent["ForecastOverviewAgent"]["forecast_beverage_units"] == 50.0
    assert metrics_by_agent["ProductionPlanAgent"]["supply_coverage_pct"] == 80.0
    assert metrics_by_agent["ProductionPlanAgent"]["demand_gap_units"] == 20
    assert "including 100 bakery units and 50 made-to-order beverage units" in result.summary
    assert "with no Day-1 carryover stock, the plan provides 80 units available" in result.summary
    assert "with no starting stock" not in result.summary
    assert "early sales should determine whether additional bake capacity is released" in result.summary
    assert "croissant should be reviewed first" in result.summary
    assert "croissant and espresso should be reviewed first" not in result.summary
    assert "the expected-demand scenario shows no planned waste" in result.summary
    hedge = next(rec for rec in result.recommendations if "85% base bake" in rec.action)
    assert "85% base bake of 68 units" in hedge.action
    assert "80-120 bakery units" in hedge.action
    assert "next 32 units up to expected demand flexible" in hedge.action
    assert "no more than 20 units above expected demand" in hedge.action


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

    assert "covering 80.0% of bakery forecast demand" in result.summary
    metrics_by_agent = {output.agent_name: output.metrics for output in result.agent_outputs}
    assert metrics_by_agent["ProductionPlanAgent"]["supply_coverage_pct"] == 80.0
    assert metrics_by_agent["ProductionPlanAgent"]["demand_gap_units"] == 100
    assert "a 100-unit bakery gap" in result.summary
    assert "Because forecast error still affects how much demand will materialize" in result.summary
    assert "the held-out historical evaluation shows 30.0% error and 78.5% coverage" in result.summary
    assert "these results support staged production and material-readiness checks" in result.summary
    assert "Three production choices are visible" not in result.summary
    assert "stage production from a 255-unit base" not in result.summary


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
    assert "In addition, two planned business events are active" in result.summary
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
