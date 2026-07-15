from s5_agent.agents.production_plan import ProductionPlanAgent


def test_production_plan_agent_returns_graph_output_with_evidence():
    agent = ProductionPlanAgent("ProductionPlanAgent")
    raw = {
        "data": {
            "grid": [
                {"date": "2026-07-07", "bake_qty": 180},
                {"date": "2026-07-08", "bake_qty": 120},
            ],
            "dates": ["2026-07-07", "2026-07-08"],
            "buffer": 1.4,
            "day1_stock_total": 25,
            "weekly_summary": {
                "total_bake": 420,
                "total_revenue": 2600.0,
                "total_profit": 780.0,
                "profit_definition": "after_waste_and_shortage_risk_allowances",
                "scenarios": {
                    "q50": {"profit": 780.0, "waste": 20},
                    "q10": {"profit": 100.0, "waste": 60},
                    "q90": {"profit": 700.0, "waste": 0, "shortage": 75},
                },
                "top_products": [("croissant", 180), ("bagel", 120)],
            },
        }
    }

    output = agent.analyze_for_graph(raw, {"date": "2026-07-07", "product": "all"})

    assert output.agent_name == "ProductionPlanAgent"
    assert output.metrics["total_bake"] == 420
    assert output.metrics["total_revenue"] == 2600.0
    assert output.metrics["total_profit"] == 780.0
    assert output.metrics["profit_definition"] == (
        "after_waste_and_shortage_risk_allowances"
    )
    assert "after waste and shortage risk allowances" in output.claim
    assert output.metrics["buffer"] == 1.4
    assert output.metrics["waste_rate_pct"] == 4.76
    assert output.metrics["q90_shortage_units"] == 75
    assert {item.id for item in output.evidence_items} >= {
        "production_total_bake",
        "production_buffer",
        "production_waste_rate_pct",
        "q90_shortage_units",
    }
    assert "overproduction_risk" in output.risks
    assert output.recommendations[0].evidence_ids == ["production_buffer"]


def test_production_plan_hedge_recommendation_explains_base_bake_and_contingency():
    agent = ProductionPlanAgent("ProductionPlanAgent")
    raw = {
        "data": {
            "grid": [{"date": "2026-07-07", "bake_qty": 180}],
            "dates": ["2026-07-07"],
            "buffer": 1.05,
            "day1_stock_total": 79,
            "weekly_summary": {
                "total_bake": 1750,
                "total_revenue": 19337.0,
                "total_profit": 12626.0,
                "scenarios": {
                    "q50": {"profit": 9650.0, "waste": 553},
                    "q10": {"profit": 2103.0, "waste": 900},
                    "q90": {"profit": 12000.0, "waste": 0, "shortage": 754},
                },
                "top_products": [
                    ("croissant_chocolate", 164),
                    ("croissant", 161),
                ],
            },
        }
    }

    output = agent.analyze_for_graph(raw, {"date": "2026-07-07", "product": "all"})
    hedge = output.recommendations[0]

    assert output.metrics["waste_rate_pct"] == 31.6
    assert "The 7-day plan calls for 1750 bake units" in output.claim
    assert "A 105% buffer lifts total available supply" in output.claim
    assert "equal to a 31.6% waste rate" in output.claim
    assert "Q50" not in output.claim
    assert "Q10" not in output.claim
    assert "Start with an 85% base bake of 1487 units" in hedge.action
    assert "Keep the remaining planned bake flexible" in hedge.action
    assert "release additional capacity" in hedge.action
    assert "expected demand path" in hedge.action
    assert "Q50" not in hedge.action
    assert "Q10" not in hedge.action
    assert "staged production" in hedge.rationale
    assert "committing the full 1750 units upfront" in hedge.rationale
    assert f"limits exposure to the downside scenario, which still projects {chr(165)}2103 profit" in str(hedge.expected_impact)
    assert "keeps the 31.6% expected-demand waste exposure visible" in str(hedge.expected_impact)
    assert f"protects against the {chr(165)}2103 downside case" not in str(hedge.expected_impact)
    assert "Q50" not in hedge.rationale
    assert "Q10" not in hedge.rationale
    assert hedge.evidence_ids == ["scenario_profit_gap", "production_waste_rate_pct"]
