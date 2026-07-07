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
    assert metrics_by_agent["MaterialProcurementAgent"]["low_material_count"] == 1
    assert result.summary.startswith("This week's plan is profitable, but it is deliberately conservative against the demand forecast.")
    assert "900 units" in result.summary
    assert "445 units available" in result.summary
    assert "leaving a 455-unit gap" in result.summary
    assert "No material is critical yet, but 1 low-stock item still needs attention" in result.summary
    assert "multi-source forecast check" not in result.summary
    assert "croissant, baguette" in result.summary
    assert "Q10" not in result.summary
    assert "Q50" not in result.summary
    assert any(node.id == "claim:ProductionPlanAgent" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:production_total_bake" for node in result.evidence_graph.nodes)
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
    assert metrics_by_agent["ForecastOverviewAgent"]["forecast_total_units"] == 100.0
    hedge = next(
        recommendation
        for recommendation in result.recommendations
        if "85% base bake" in recommendation.action
    )
    assert "Keep the remaining planned bake flexible" in hedge.action
    assert "100-unit demand forecast" in hedge.action
    assert hedge.evidence_ids == [
        "scenario_profit_gap",
        "production_waste_rate_pct",
    ]
