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
