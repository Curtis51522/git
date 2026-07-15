import asyncio

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
    assert output.evidence_items[0].description == (
        "Adjusted profit margin percentage for the requested period"
    )
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
    assert "No expired-stock or non-sellable return cost was recorded" in output.claim
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
    assert "Separately recorded material-wastage variance is not deducted again" in output.claim


def test_profit_agent_exposes_expired_stock_cost_in_profit_evidence():
    agent = ProfitAgent("ProfitAgent")
    raw = {
        "data": {
            "today_revenue": 1000.0,
            "today_profit": 125.0,
            "today_orders": 20,
            "discount_total": 0.0,
            "expired_cost": 25.0,
        }
    }

    output = agent.analyze_for_graph(raw, {"date": "2026-07-07", "product": "all"})
    evidence_ids = {item.id for item in output.evidence_items}

    assert output.metrics["expired_cost"] == 25.0
    assert output.metrics["expired_cost_revenue_pct"] == 2.5
    assert output.metrics["profit_margin_before_expiry_pct"] == 15.0
    assert output.metrics["expired_margin_erosion_pct_points"] == 2.5
    assert evidence_ids >= {
        "expired_cost",
        "expired_cost_revenue_pct",
        "profit_margin_before_expiry_pct",
        "expired_margin_erosion_pct_points",
    }
    assert "2.5% of revenue" in output.claim
    assert "15.0% to 12.5%" in output.claim
    assert "unsold_product_loss" not in output.risks
    assert "Waste impact is not included" not in output.claim


def test_profit_agent_flags_material_unsold_product_loss():
    agent = ProfitAgent("ProfitAgent")
    raw = {
        "data": {
            "today_revenue": 4179.6,
            "today_profit": 2902.13,
            "today_orders": 52,
            "discount_total": 63.8,
            "expired_cost": 698.4,
            "expired_products": [
                {
                    "name": "Cream Horn",
                    "expired_cost": 73.6,
                    "sell_through_pct": 20.0,
                },
                {
                    "name": "Apple Pie",
                    "expired_cost": 61.6,
                    "sell_through_pct": 33.33,
                },
                {
                    "name": "Chocopie",
                    "expired_cost": 56.6,
                    "sell_through_pct": 25.93,
                },
            ],
        }
    }

    output = agent.analyze_for_graph(raw, {"date": "2026-07-14", "product": "all"})

    assert output.metrics["expired_cost_revenue_pct"] == 16.71
    assert output.metrics["profit_margin_before_expiry_pct"] == 86.15
    assert output.metrics["expired_margin_erosion_pct_points"] == 16.71
    assert "unsold_product_loss" in output.risks
    assert "unsold product loss needs attention" in output.claim
    assert "Revenue performance is healthy" not in output.claim
    assert [item.id for item in output.recommendations] == [
        "unsold_product_loss_reduction"
    ]
    recommendation = output.recommendations[0]
    assert recommendation.urgency == "high"
    assert recommendation.time_horizon == "this_week"
    assert recommendation.evidence_ids == [
        "expired_cost",
        "expired_cost_revenue_pct",
        "profit_margin_before_expiry_pct",
        "profit_margin_pct",
        "expired_products",
    ]
    assert "Cream Horn, Apple Pie, and Chocopie" in recommendation.action
    assert "20.0%, 33.3%, and 25.9%" in recommendation.rationale
    assert "16.7% of revenue" in recommendation.rationale
    assert "86.1% to 69.4%" in recommendation.rationale


def test_profit_fetch_preserves_expired_product_breakdown(monkeypatch):
    agent = ProfitAgent("ProfitAgent")
    expired_products = [
        {
            "name": "Cream Horn",
            "expired_cost": 73.6,
            "sell_through_pct": 20.0,
        }
    ]

    async def fake_revenue(_date, _authorization):
        return {
            "data": {
                "today_revenue": 4179.6,
                "today_profit": 2902.13,
                "today_orders": 52,
                "today_discount": 63.8,
                "expired_cost": 698.4,
                "expired_products": expired_products,
            }
        }

    monkeypatch.setattr(agent, "_get_revenue", fake_revenue)

    result = asyncio.run(agent.fetch({"date": "2026-07-14"}))

    assert result["data"]["expired_products"] == expired_products


def test_profit_agent_uses_dashboard_profit_margin_as_source_of_truth(monkeypatch):
    agent = ProfitAgent("ProfitAgent")

    async def fake_revenue(_date, _authorization):
        return {
            "data": {
                "today_revenue": 4179.6,
                "today_profit": 3600.53,
                "today_orders": 52,
                "today_discount": 63.8,
                "profit_margin": 86.1,
            }
        }

    monkeypatch.setattr(agent, "_get_revenue", fake_revenue)

    raw = asyncio.run(agent.fetch({"date": "2026-07-14"}))
    output = agent.analyze_for_graph(raw, {"date": "2026-07-14"})

    assert raw["data"]["profit_margin"] == 86.1
    assert output.metrics["profit_margin_pct"] == 86.1
    assert "margin 86.1%" in output.claim
