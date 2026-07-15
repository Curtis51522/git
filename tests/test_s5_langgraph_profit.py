import asyncio

from s5_agent.agents.revenue_analysis import (
    HourlyRevenueAgent,
    OrderBehaviorAgent,
    RevenueBenchmarkAgent,
    RevenueTrendAgent,
)
from s5_agent.agents.profit import ProfitAgent
from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request


def _request(date="2026-07-02"):
    return S5Request(
        query="Analyze profit",
        module="revenue",
        params={"date": date, "product": "all"},
    )


def test_profit_root_cause_uses_revenue_multi_agent_output():
    raw = {
        "profit": {
            "data": {
                "today_revenue": 1000.0,
                "today_profit": 150.0,
                "today_orders": 20,
                "discount_total": 80.0,
            }
        },
        "revenue_trend": {
            "data": {
                "dates": ["07-01", "07-02", "07-03"],
                "bread": [800.0, 900.0, 1000.0],
                "orders": [16, 18, 20],
                "avg_order": [50.0, 50.0, 50.0],
                "today_revenue": 1000.0,
                "revenue_change": 11.1,
                "profit_change": -14.8,
                "orders_change": 6.1,
                "avg_change": -0.8,
                "previous_day_available": True,
            }
        },
        "revenue_product_mix": {
            "data": {
                "bread_ranking": [
                    {"name": "croissant", "qty": 18, "revenue": 360.0, "profit": 250.0},
                    {"name": "baguette", "qty": 12, "revenue": 240.0, "profit": 170.0},
                ],
                "beverage_ranking": [
                    {"name": "latte", "qty": 10, "revenue": 200.0, "profit": 150.0},
                ],
                "category": {"Bread": 700.0, "Beverages": 300.0},
                "total_bread_sku": 8,
                "total_bev_sku": 5,
            }
        },
        "hourly_revenue": {
            "data": {
                "hours": ["09:00", "10:00", "11:00", "12:00"],
                "bread": [80.0, 120.0, 360.0, 140.0],
                "beverages": [20.0, 60.0, 180.0, 40.0],
                "revenue": [100.0, 180.0, 540.0, 180.0],
                "profit": [70.0, 140.0, 410.0, 130.0],
                "orders": [2, 3, 6, 3],
                "avg_order": [50.0, 60.0, 90.0, 60.0],
                "margin": [70.0, 77.8, 75.9, 72.2],
            }
        },
    }

    result = asyncio.run(run_s5_graph("profit_root_cause", _request("2026-07-07"), raw_inputs=raw))

    assert result.metadata["template"] == "profit_root_cause"
    assert [output.agent_name for output in result.agent_outputs] == [
        "ProfitAgent",
        "RevenueBenchmarkAgent",
        "RevenueTrendAgent",
        "OrderBehaviorAgent",
        "RevenueProductMixAgent",
        "CategoryMixAgent",
        "HourlyRevenueAgent",
        "DiscountImpactAgent",
    ]
    assert result.agent_outputs[0].metrics["profit_margin_pct"] == 15.0
    assert any(node.id == "claim:ProfitAgent" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:profit_margin_pct" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:revenue_trend_pct" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:revenue_vs_previous_day_pct" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:revenue_vs_recent_avg_pct" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:order_volume_driver" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:top_product_revenue_share_pct" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:bread_revenue_share_pct" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:peak_revenue_hour" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:peak_profit_hour" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:hourly_peak_revenue_share_pct" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:discount_rate_pct" for node in result.evidence_graph.nodes)
    assert "Peak profit hour" in result.summary
    assert "generating ¥410 profit" in result.summary
    assert "6 orders" in result.summary
    assert "75.9% hourly margin" in result.summary
    assert "revenue timing is used as a proxy" not in result.summary
    assert "revenue increased by 11.1% while profit decreased by 14.8%" in result.summary
    assert "higher order volume rather than basket expansion" in result.summary
    assert "customer traffic" not in result.summary
    assert result.summary.count("\n\n") == 2
    summary_paragraphs = result.summary.split("\n\n")
    assert "Compared with the previous day" in summary_paragraphs[0]
    assert "Revenue quality" in summary_paragraphs[1]
    assert "Separately recorded material-wastage variance" in summary_paragraphs[2]


def test_hourly_revenue_agent_prefers_profitability_metrics_when_available():
    agent = HourlyRevenueAgent()
    raw = {
        "data": {
            "hours": ["09:00", "10:00", "11:00"],
            "bread": [200.0, 300.0, 260.0],
            "beverages": [80.0, 90.0, 70.0],
            "revenue": [280.0, 420.0, 330.0],
            "profit": [210.0, 330.0, 264.0],
            "orders": [4, 5, 3],
            "avg_order": [70.0, 78.0, 110.0],
            "margin": [75.0, 84.6, 80.0],
        }
    }

    result = agent.analyze_for_graph(raw, {"date": "2026-06-30"})

    assert "Peak profit hour is 10:00" in result.claim
    assert "revenue timing is used as a proxy" not in result.claim
    assert result.metrics["peak_profit_hour"] == "10:00"
    assert result.metrics["peak_profit"] == 330.0
    assert result.metrics["peak_revenue"] == 420.0
    assert result.metrics["hourly_total_revenue"] == 1030.0
    assert result.metrics["hourly_total_profit"] == 804.0
    assert result.metrics["peak_profit_margin_pct"] == 84.6
    assert result.recommendations[0].id == "peak_profit_window_protection"
    assert result.recommendations[0].time_horizon == "this_week"
    assert result.recommendations[0].evidence_ids == [
        "peak_profit_hour",
        "hourly_peak_profit_share_pct",
        "peak_profit_margin_pct",
    ]
    assert not result.data_quality.limitations


def test_hourly_revenue_agent_excludes_closing_adjustment_from_hourly_shares():
    agent = HourlyRevenueAgent()
    raw = {
        "data": {
            "hours": ["12:00", "13:00", "Closing adjustment"],
            "bread": [400.0, 300.0, 0.0],
            "beverages": [147.0, 100.0, 0.0],
            "revenue": [547.0, 400.0, 0.0],
            "profit": [476.0, 300.0, -698.4],
            "orders": [7, 5, 0],
            "avg_order": [78.14, 80.0, 0.0],
            "margin": [87.0, 75.0, 0.0],
        }
    }

    result = agent.analyze_for_graph(raw, {"date": "2026-07-14"})

    assert result.metrics["peak_profit_hour"] == "12:00"
    assert result.metrics["hourly_total_profit"] == 776.0
    assert result.metrics["hourly_peak_profit_share_pct"] == 61.3
    assert "Closing adjustment" not in result.metrics["low_revenue_hours"]


def test_revenue_agents_use_recent_baseline_when_previous_day_is_unavailable():
    raw = {
        "data": {
            "today_revenue": 1200.0,
            "today_orders": 12,
            "avg_order": 100.0,
            "revenue_change": None,
            "profit_change": None,
            "orders_change": None,
            "avg_change": None,
            "previous_day_available": False,
            "previous_day_date": "2026-07-13",
            "recent_baseline": {
                "day_count": 7,
                "start_date": "2026-07-06",
                "end_date": "2026-07-12",
                "avg_revenue": 1000.0,
                "avg_orders": 10.0,
                "avg_order_value": 100.0,
            },
            "trend": {
                "dates": ["07-08", "07-09", "07-10", "07-11", "07-12", "07-13", "07-14"],
                "bread": [700.0, 700.0, 700.0, 700.0, 700.0, 0.0, 900.0],
                "beverages": [300.0, 300.0, 300.0, 300.0, 300.0, 0.0, 300.0],
                "orders": [10, 10, 10, 10, 10, 0, 12],
                "avg_order": [100.0, 100.0, 100.0, 100.0, 100.0, 0.0, 100.0],
            },
        }
    }

    trend = RevenueTrendAgent().analyze_for_graph(raw, {"date": "2026-07-14"})
    benchmark = RevenueBenchmarkAgent().analyze_for_graph(raw, {"date": "2026-07-14"})
    behavior = OrderBehaviorAgent().analyze_for_graph(raw, {"date": "2026-07-14"})

    assert trend.metrics["previous_day_available"] is False
    assert trend.metrics["dashboard_revenue_change_pct"] is None
    assert trend.metrics["recent_total_revenue_change_pct"] == 20.0
    assert trend.metrics["trend_direction"] == "unavailable"
    assert "no completed sales were recorded on 2026-07-13" in trend.claim
    assert benchmark.metrics["revenue_vs_recent_avg_pct"] == 20.0
    assert benchmark.metrics["baseline_day_count"] == 7
    assert behavior.metrics["comparison_basis"] == "recent_baseline"
    assert behavior.metrics["order_change_pct"] == 20.0
    assert behavior.metrics["average_order_value_change_pct"] == 0.0


def test_profit_graph_fetches_data_when_raw_inputs_missing(monkeypatch):
    async def fake_fetch(self, params):
        return {
            "success": True,
            "data": {
                "today_revenue": 1000.0,
                "today_profit": 150.0,
                "today_orders": 20,
                "discount_total": 80.0,
            },
            "tool": "profit_test",
        }

    monkeypatch.setattr(ProfitAgent, "fetch", fake_fetch)

    result = asyncio.run(run_s5_graph("profit_root_cause", _request("2026-07-07")))

    assert result.agent_outputs[0].metrics["revenue"] == 1000.0
    assert result.agent_outputs[0].metrics["profit_margin_pct"] == 15.0


def test_profit_root_cause_healthy_no_risk_gets_no_action_decision():
    raw = {
        "profit": {
            "data": {
                "today_revenue": 2963.0,
                "today_profit": 2370.4,
                "today_orders": 47,
                "discount_total": 0.0,
            }
        },
        "revenue_trend": {
            "data": {
                "dates": ["06-26", "06-27", "06-28", "06-29", "06-30", "07-01", "07-02"],
                "bread": [2050.0, 2140.0, 2190.0, 2250.0, 2310.0, 2380.0, 2460.0],
                "orders": [39, 40, 42, 43, 44, 45, 47],
                "avg_order": [59.0, 60.0, 60.5, 61.0, 61.5, 62.0, 63.04],
                "today_revenue": 2963.0,
                "revenue_change": 6.3,
            }
        },
        "revenue_product_mix": {
            "data": {
                "bread_ranking": [
                    {"name": "croissant_chocolate", "qty": 19, "revenue": 494.0, "profit": 395.2},
                    {"name": "croissant", "qty": 17, "revenue": 425.0, "profit": 340.0},
                    {"name": "baguette", "qty": 14, "revenue": 350.0, "profit": 280.0},
                ],
                "beverage_ranking": [
                    {"name": "latte", "qty": 12, "revenue": 336.0, "profit": 268.8},
                ],
                "category": {"Bread": 2180.0, "Beverages": 783.0},
                "total_bread_sku": 9,
                "total_bev_sku": 6,
            }
        },
        "hourly_revenue": {
            "data": {
                "hours": ["09:00", "10:00", "11:00", "12:00", "13:00"],
                "bread": [320.0, 410.0, 780.0, 520.0, 150.0],
                "beverages": [90.0, 110.0, 260.0, 210.0, 113.0],
            }
        },
    }

    result = asyncio.run(run_s5_graph("profit_root_cause", _request(), raw_inputs=raw))

    assert "This was a healthy revenue day" in result.summary
    assert "47 orders" in result.summary
    assert "average order value at ¥63.04" in result.summary
    assert "against the previous day" in result.summary
    assert "Revenue quality is volume-led" in result.summary
    assert "croissant chocolate" in result.summary
    assert "Category mix" in result.summary
    assert "Peak revenue hour" in result.summary
    assert "No expired-stock or non-sellable return cost was recorded" in result.summary
    assert [recommendation.id for recommendation in result.recommendations] == ["revenue_no_action_decision"]
    assert result.recommendations[0].time_horizon == "ongoing"
    assert result.recommendations[0].evidence_ids == [
        "profit_margin_pct",
        "discount_rate_pct",
        "top3_product_revenue_share_pct",
        "revenue_trend_pct",
    ]
    assert any(node.id == "metric:order_volume" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:average_order_value" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:revenue_trend_pct" for node in result.evidence_graph.nodes)
    assert any(node.id == "metric:top_product_revenue_share_pct" for node in result.evidence_graph.nodes)


def test_profit_root_cause_explains_expired_cost_and_uses_correct_margin_article():
    raw = {
        "profit": {
            "data": {
                "today_revenue": 1000.0,
                "today_profit": 694.0,
                "today_orders": 10,
                "discount_total": 0.0,
                "expired_cost": 25.0,
            }
        }
    }

    result = asyncio.run(run_s5_graph("profit_root_cause", _request(), raw_inputs=raw))

    assert "with a profit margin of 69.4%" in result.summary
    assert "2.5% of revenue" in result.summary
    assert "71.9% before this loss to 69.4% after it" in result.summary
    assert "Waste impact is not included" not in result.summary


def test_profit_root_cause_turns_material_unsold_loss_into_action():
    raw = {
        "profit": {
            "data": {
                "today_revenue": 4179.6,
                "today_profit": 2902.13,
                "today_orders": 52,
                "discount_total": 63.8,
                "expired_cost": 698.4,
            }
        }
    }

    result = asyncio.run(
        run_s5_graph(
            "profit_root_cause",
            _request("2026-07-14"),
            raw_inputs=raw,
        )
    )
    profit_output = next(
        output for output in result.agent_outputs if output.agent_name == "ProfitAgent"
    )

    assert "unsold_product_loss" in profit_output.risks
    assert result.summary.startswith(
        "This revenue day was profitable, but closing product loss needs attention"
    )
    assert "16.7% of revenue" in result.summary
    assert "86.1% before this loss to 69.4% after it" in result.summary
    assert [item.id for item in result.recommendations] == [
        "unsold_product_loss_reduction"
    ]
    assert "No immediate revenue intervention" not in result.recommendations[0].action


def test_profit_root_cause_order_value_shift_gets_watch_recommendation():
    raw = {
        "profit": {
            "data": {
                "today_revenue": 2963.0,
                "today_profit": 2370.4,
                "today_orders": 47,
                "discount_total": 0.0,
            }
        },
        "revenue_trend": {
            "data": {
                "dates": ["06-26", "06-27", "06-28", "06-29", "06-30", "07-01", "07-02"],
                "bread": [2900.0, 2940.0, 2920.0, 2960.0, 2930.0, 2940.0, 2963.0],
                "orders": [35, 36, 36, 36, 36, 36, 47],
                "avg_order": [65.0, 66.0, 65.5, 66.5, 65.0, 66.0, 63.04],
                "today_revenue": 2963.0,
                "revenue_change": 1.1,
            }
        },
        "revenue_product_mix": {
            "data": {
                "bread_ranking": [
                    {"name": "macaron", "qty": 37, "revenue": 554.0, "profit": 440.0},
                    {"name": "croissant", "qty": 18, "revenue": 425.0, "profit": 340.0},
                    {"name": "baguette", "qty": 15, "revenue": 330.0, "profit": 265.0},
                ],
                "beverage_ranking": [],
                "category": {"Bread": 2963.0, "Beverages": 0.0},
                "total_bread_sku": 9,
                "total_bev_sku": 6,
            }
        },
    }

    result = asyncio.run(run_s5_graph("profit_root_cause", _request(), raw_inputs=raw))

    assert "Order value shift risk" not in result.summary
    assert "order count moved +31.2%" in result.summary
    assert result.recommendations[0].id == "average_order_value_watch"
    assert "2-3 trading days" in result.recommendations[0].action
    assert "targeted bundles" in result.recommendations[0].action
    assert "broad discounts" in result.recommendations[0].action
    assert result.recommendations[0].time_horizon == "this_week"
    assert result.recommendations[0].evidence_ids == [
        "profit_margin_pct",
        "order_change_pct",
        "average_order_value_change_pct",
        "average_order_value",
    ]
    assert "average order value" in result.recommendations[0].action


def test_profit_root_cause_explains_joint_order_and_basket_contraction():
    dashboard = {
        "data": {
            "date": "2026-07-15",
            "today_revenue": 2637.5,
            "today_profit": 2306.86,
            "today_orders": 45,
            "avg_order": 58.61,
            "today_discount": 92.1,
            "expired_cost": 8.66,
            "revenue_change": -36.9,
            "profit_change": -35.9,
            "orders_change": -13.5,
            "avg_change": -27.1,
            "previous_day_available": True,
            "recent_baseline": {
                "day_count": 7,
                "avg_revenue": 3952.59,
                "avg_orders": 49.29,
                "avg_order_value": 80.2,
            },
            "today_items": 215,
            "items_per_order": 4.78,
            "items_per_order_change": -27.8,
            "revenue_per_item": 12.27,
            "revenue_per_item_change": 1.0,
            "trend": {
                "dates": ["07-09", "07-10", "07-11", "07-12", "07-13", "07-14", "07-15"],
                "bread": [2800.0, 2900.0, 2700.0, 3000.0, 2800.0, 3024.7, 1833.4],
                "beverages": [1068.8, 1095.6, 1044.0, 1179.6, 1170.8, 1148.0, 799.0],
                "orders": [48, 50, 47, 52, 49, 52, 45],
                "avg_order": [80.6, 79.9, 79.7, 80.4, 81.0, 80.38, 58.61],
            },
        }
    }
    raw = {
        "profit": dashboard,
        "revenue_trend": dashboard,
        "revenue_benchmark": dashboard,
        "order_behavior": dashboard,
    }

    result = asyncio.run(
        run_s5_graph(
            "profit_root_cause",
            _request("2026-07-15"),
            raw_inputs=raw,
        )
    )

    behavior = next(
        output
        for output in result.agent_outputs
        if output.agent_name == "OrderBehaviorAgent"
    )
    benchmark = next(
        output
        for output in result.agent_outputs
        if output.agent_name == "RevenueBenchmarkAgent"
    )

    assert result.summary.startswith(
        "The day remained profitable, but revenue performance weakened materially"
    )
    assert "with a profit margin of 87.5%" in result.summary
    assert "healthy revenue day" not in result.summary
    assert behavior.metrics["order_volume_driver"] == "volume-and-basket-contraction"
    assert "Revenue weakened through both fewer orders and smaller baskets" in result.summary
    assert "Items per order fell from 6.62 to 4.78" in result.summary
    assert "revenue per item remained broadly stable at \u00a512.27" in result.summary
    assert "Revenue quality is stable" not in result.summary
    assert benchmark.risks == []
    assert result.recommendations[0].id == "revenue_decline_review"
    assert "targeted bundles" in result.recommendations[0].action
    assert "product availability" in result.recommendations[0].action
    assert "broad discount" in result.recommendations[0].action
    assert "items_per_order" in result.recommendations[0].evidence_ids
    assert "revenue_per_item" in result.recommendations[0].evidence_ids


def test_profit_root_cause_uses_revenue_dashboard_comparison_as_primary_source():
    raw = {
        "profit": {
            "data": {
                "today_revenue": 2984.0,
                "today_profit": 2436.73,
                "today_orders": 41,
                "discount_total": 0.0,
            }
        },
        "revenue_trend": {
            "data": {
                "date": "2026-06-30",
                "today_revenue": 2984.0,
                "today_profit": 2436.73,
                "today_orders": 41,
                "avg_order": 72.78,
                "revenue_change": 42.6,
                "profit_change": 40.9,
                "orders_change": -6.8,
                "avg_change": 53.0,
                "trend": {
                    "dates": ["06-24", "06-25", "06-26", "06-27", "06-28", "06-29", "06-30"],
                    "bread": [1841.0, 2001.0, 2997.0, 4600.0, 1944.0, 1559.0, 2326.0],
                    "beverages": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "orders": [41, 41, 43, 45, 42, 44, 41],
                    "avg_order": [60.71, 61.59, 88.49, 124.71, 60.62, 47.57, 72.78],
                },
            }
        },
        "revenue_product_mix": {
            "data": {
                "bread_ranking": [
                    {"name": "macaron", "qty": 59, "revenue": 590.0, "profit": 413.0},
                    {"name": "mantequilla", "qty": 37, "revenue": 333.0, "profit": 288.6},
                    {"name": "bread_coconut", "qty": 32, "revenue": 320.0, "profit": 274.56},
                ],
                "beverage_ranking": [
                    {"name": "cold_brew", "qty": 8, "revenue": 144.0, "profit": 120.0},
                ],
                "category": {"Bread": 2326.0, "Beverages": 0.0, "Coffee": 658.0},
                "total_bread_sku": 9,
                "total_bev_sku": 6,
            }
        },
        "hourly_revenue": {
            "data": {
                "hours": ["09:00", "10:00", "11:00"],
                "bread": [500.0, 400.0, 300.0],
                "beverages": [126.0, 100.0, 80.0],
            }
        },
    }

    result = asyncio.run(run_s5_graph("profit_root_cause", _request("2026-06-30"), raw_inputs=raw))

    assert "Revenue dashboard comparison" in result.summary
    assert "total revenue moved +42.6% vs yesterday" in result.summary
    assert "profit moved +40.9%" in result.summary
    assert "order count moved -6.8%" in result.summary
    assert "average order value moved +53.0%" in result.summary
    assert "larger baskets rather than higher order volume" in result.summary
    assert "+49.2% against the previous day" not in result.summary
    trend_output = next(output for output in result.agent_outputs if output.agent_name == "RevenueTrendAgent")
    assert trend_output.metrics["dashboard_revenue_change_pct"] == 42.6
    assert trend_output.metrics["dashboard_profit_change_pct"] == 40.9
    assert trend_output.metrics["order_change_pct"] == -6.8
    assert trend_output.metrics["average_order_value_change_pct"] == 53.0
    assert any(node.id == "metric:dashboard_revenue_change_pct" for node in result.evidence_graph.nodes)
    benchmark_output = next(output for output in result.agent_outputs if output.agent_name == "RevenueBenchmarkAgent")
    assert benchmark_output.metrics["revenue_vs_recent_avg_pct"] == -6.1


def test_profit_root_cause_peak_profit_recommendation_is_operationally_specific():
    raw = {
        "profit": {
            "data": {
                "today_revenue": 2984.0,
                "today_profit": 2436.73,
                "today_orders": 41,
                "discount_total": 0.0,
            }
        },
        "revenue_trend": {
            "data": {
                "today_revenue": 2984.0,
                "today_profit": 2436.73,
                "today_orders": 41,
                "avg_order": 72.78,
                "revenue_change": 42.6,
                "profit_change": 40.9,
                "orders_change": -6.8,
                "avg_change": 53.0,
            }
        },
        "revenue_product_mix": {
            "data": {
                "bread_ranking": [
                    {"name": "macaron", "qty": 59, "revenue": 590.0, "profit": 413.0},
                    {"name": "mantequilla", "qty": 37, "revenue": 333.0, "profit": 288.6},
                    {"name": "bread_coconut", "qty": 32, "revenue": 320.0, "profit": 274.56},
                ],
                "beverage_ranking": [
                    {"name": "cold_brew", "qty": 8, "revenue": 144.0, "profit": 120.0},
                ],
                "category": {"Bread": 2326.0, "Beverages": 0.0, "Coffee": 658.0},
            }
        },
        "hourly_revenue": {
            "data": {
                "hours": ["09:00", "10:00", "11:00"],
                "revenue": [626.0, 420.0, 330.0],
                "profit": [446.36, 330.0, 264.0],
                "orders": [2, 5, 3],
                "avg_order": [313.0, 84.0, 110.0],
                "margin": [71.3, 78.6, 80.0],
            }
        },
    }

    result = asyncio.run(run_s5_graph("profit_root_cause", _request("2026-06-30"), raw_inputs=raw))
    peak_recommendation = next(
        recommendation
        for recommendation in result.recommendations
        if recommendation.id == "peak_profit_window_protection"
    )

    assert "Before 09:00" in peak_recommendation.action
    assert "Macaron" in peak_recommendation.action
    assert "beverage pairing" in peak_recommendation.action
    assert "service coverage" in peak_recommendation.action
    assert "stock or queue delays" in peak_recommendation.expected_impact


def test_profit_root_cause_low_order_collapse_prioritizes_data_check_recommendation():
    raw = {
        "profit": {
            "data": {
                "today_revenue": 30.8,
                "today_profit": 26.18,
                "today_orders": 1,
                "discount_total": 3.2,
            }
        },
        "revenue_trend": {
            "data": {
                "dates": ["06-26", "06-27", "06-28", "06-29", "06-30", "07-01", "07-02"],
                "bread": [5100.0, 5200.0, 5000.0, 5300.0, 5150.0, 5250.0, 30.8],
                "orders": [36, 38, 35, 37, 39, 37, 1],
                "avg_order": [61.0, 60.0, 62.0, 60.5, 61.5, 62.0, 30.8],
                "today_revenue": 30.8,
                "revenue_change": -99.4,
            }
        },
        "revenue_product_mix": {
            "data": {
                "bread_ranking": [
                    {"name": "brioche", "qty": 1, "revenue": 12.8, "profit": 10.5},
                ],
                "beverage_ranking": [],
                "category": {"Bread": 30.8, "Beverages": 0.0},
                "total_bread_sku": 9,
                "total_bev_sku": 6,
            }
        },
    }

    result = asyncio.run(run_s5_graph("profit_root_cause", _request(), raw_inputs=raw))

    assert "not a healthy revenue day overall" in result.summary
    assert "1 order" in result.summary
    assert "1 orders" not in result.summary
    assert "1 unit" in result.summary
    assert "1 units" not in result.summary
    assert "margin signal is positive" in result.summary
    assert "No discount erosion is visible" not in result.summary
    assert "clean on discounts" not in result.summary
    assert result.recommendations[0].id == "revenue_data_completeness_check"
    assert result.recommendations[0].time_horizon == "today"
    assert "incomplete sales data" in result.recommendations[0].action
    assert result.recommendations[0].evidence_ids == [
        "revenue",
        "order_volume",
        "revenue_trend_pct",
        "order_change_pct",
        "discount_rate_pct",
    ]
