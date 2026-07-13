from s5_agent.agents.profit import ProfitAgent


def test_profit_agent_returns_graph_output_with_evidence():
    agent = ProfitAgent("ProfitAgent")
    raw = {
        "data": {
            "today_revenue": 1000.0,
            "today_profit": 150.0,
            "today_orders": 20,
            "discount_total": 80.0,
        }
    }

    output = agent.analyze_for_graph(raw, {"date": "2026-07-07", "product": "all"})

    assert output.agent_name == "ProfitAgent"
    assert output.metrics["revenue"] == 1000.0
    assert output.metrics["profit"] == 150.0
    assert output.metrics["profit_margin_pct"] == 15.0
    assert output.metrics["discount_total"] == 80.0
    assert output.evidence_items[0].id == "profit_margin_pct"
    assert "low_margin" in output.risks
    assert output.recommendations[0].evidence_ids == ["profit_margin_pct"]


def test_profit_agent_healthy_day_explains_revenue_without_unsupported_waste_claim():
    agent = ProfitAgent("ProfitAgent")
    raw = {
        "data": {
            "today_revenue": 2963.0,
            "today_profit": 2370.4,
            "today_orders": 47,
            "discount_total": 0.0,
        }
    }

    output = agent.analyze_for_graph(raw, {"date": "2026-07-02", "product": "all"})
    evidence_ids = {item.id for item in output.evidence_items}

    assert "Revenue performance is healthy" in output.claim
    assert "average order value at ¥63.04" in output.claim
    assert "No discount erosion is visible in the revenue data" in output.claim
    assert "waste impact is not included in this check" in output.claim
    assert "waste drag detected" not in output.claim
    assert output.metrics["average_order_value"] == 63.04
    assert evidence_ids >= {
        "profit_margin_pct",
        "revenue",
        "discount_total",
        "order_volume",
        "average_order_value",
    }


def test_profit_agent_exposes_non_sellable_return_cost_in_profit_evidence():
    agent = ProfitAgent("ProfitAgent")
    raw = {
        "data": {
            "today_revenue": 1000.0,
            "today_profit": 140.0,
            "today_orders": 20,
            "discount_total": 0.0,
            "non_sellable_return_cost": 10.0,
        }
    }

    output = agent.analyze_for_graph(raw, {"date": "2026-07-07", "product": "all"})
    evidence_ids = {item.id for item in output.evidence_items}

    assert output.metrics["non_sellable_return_cost"] == 10.0
    assert "non_sellable_return_cost" in evidence_ids
    assert "Non-sellable return cost of ¥10.00 is included in profit" in output.claim
    assert "broader production waste is not included" in output.claim
