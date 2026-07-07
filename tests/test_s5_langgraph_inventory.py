import asyncio

from s5_agent.agents.inventory import InventoryAgent
from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request


def test_run_s5_graph_returns_inventory_diagnosis_response(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)
    raw = {
        "inventory": [
            {
                "product_name": "croissant",
                "total_quantity": 80,
                "batches": 2,
                "selling_price": 5.9,
            }
        ]
    }
    request = S5Request(query="Check croissant inventory", params={"product": "croissant"})

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={"inventory": raw},
        )
    )

    assert result.summary
    assert result.metadata["template"] == "inventory_diagnosis"
    assert result.evidence_graph.nodes
    assert result.verification_report.passed is True
    assert [output.agent_name for output in result.agent_outputs] == [
        "FinishedStockAgent",
        "StockDataQualityAgent",
        "InventoryRecommendationAgent",
    ]


def test_inventory_graph_fetches_data_when_raw_inputs_missing(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)

    async def fake_fetch(self, params):
        return {
            "inventory": [
                {
                    "product_name": "croissant",
                    "total_quantity": 80,
                    "batches": 2,
                    "selling_price": 5.9,
                }
            ]
        }

    monkeypatch.setattr(InventoryAgent, "fetch", fake_fetch)
    request = S5Request(query="Check inventory", module="inventory", params={"product": "croissant"})

    result = asyncio.run(run_s5_graph("inventory_diagnosis", request))

    assert result.summary != "No stock data for croissant"
    assert result.agent_outputs[0].metrics["inventory"] == 80


def test_inventory_summary_treats_all_zero_stock_as_data_gap_or_stockout_risk(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)
    raw = {
        "inventory": [
            {"product_name": "croissant", "total_quantity": 0, "batches": 1, "selling_price": 10.0},
            {"product_name": "bread_roll", "total_quantity": 0, "batches": 1, "selling_price": 6.5},
            {"product_name": "stickbread", "total_quantity": 0, "batches": 1, "selling_price": 6.0},
        ]
    }
    request = S5Request(query="Check inventory", module="inventory", params={"product": "all"})

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={"inventory": raw},
        )
    )

    outputs = {output.agent_name: output for output in result.agent_outputs}
    output = outputs["FinishedStockAgent"]
    data_quality = outputs["StockDataQualityAgent"]
    recommendation_agent = outputs["InventoryRecommendationAgent"]

    assert "No finished-product stock is recorded for the selected scope" in result.summary
    assert "This should be treated as a stockout risk or inventory sync gap" in result.summary
    assert "bread roll, croissant, and stickbread" in result.summary
    assert "stockout_risk" in output.risks
    assert "inventory_data_gap" in output.risks
    assert "inventory_data_gap" in data_quality.risks
    assert data_quality.metrics["inventory_record_status"] == "all_zero"
    assert recommendation_agent.recommendations[0].id == "inventory_stock_record_audit"
    assert output.metrics["zero_stock_product_count"] == 3
    recommendation = result.recommendations[0]
    assert recommendation.id == "inventory_stock_record_audit"
    assert "Verify finished-product inventory records" in recommendation.action


def test_inventory_summary_flags_missing_finished_stock_records(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)
    request = S5Request(query="Check inventory", module="inventory", params={"product": "all"})

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={"inventory": {"inventory": []}},
        )
    )
    outputs = {output.agent_name: output for output in result.agent_outputs}

    assert "No finished-product inventory records are available for the selected scope" in result.summary
    assert "inventory_data_gap" in outputs["FinishedStockAgent"].risks
    assert outputs["StockDataQualityAgent"].metrics["inventory_record_status"] == "missing"
    assert result.recommendations[0].id == "inventory_stock_record_audit"
