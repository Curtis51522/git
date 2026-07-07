from s5_agent.evidence.builder import build_evidence_graph
from s5_agent.evidence.serializer import serialize_evidence_graph
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation


def test_build_evidence_graph_links_sources_metrics_claims_risks_and_recommendations():
    inventory_evidence = EvidenceItem(
        id="ev_stock_days",
        source="inventory_db",
        description="Stock coverage in days",
        value=1.5,
        metadata={"product": "croissant"},
    )
    sales_evidence = EvidenceItem(
        id="ev_sales_rate",
        source="sales_history",
        description="Average daily units sold",
        value=32,
        metadata={"window_days": 7},
    )
    recommendation = Recommendation(
        id="rec_order_croissants",
        action="Order additional croissant ingredients.",
        urgency="high",
        time_horizon="tomorrow",
        rationale="Stock coverage is below the target threshold.",
        expected_impact="Reduced stockout risk",
        claim_ids=["claim:InventoryAgent"],
        evidence_ids=["ev_stock_days"],
    )
    output = AgentOutput(
        agent_name="InventoryAgent",
        claim="Croissant inventory coverage is low.",
        confidence=0.91,
        metrics={"stock_days": 1.5},
        evidence_items=[inventory_evidence, sales_evidence],
        risks=["Croissants may stock out before the next delivery."],
        recommendations=[recommendation],
        data_quality=DataQuality(freshness="fresh", completeness=0.9),
    )

    graph = build_evidence_graph({"InventoryAgent": output})

    nodes_by_id = {node.id: node for node in graph.nodes}
    edges = {(edge.source_id, edge.target_id, edge.type) for edge in graph.edges}

    assert nodes_by_id["claim:InventoryAgent"].type == "claim"
    assert nodes_by_id["claim:InventoryAgent"].label == "Croissant inventory coverage is low."
    assert nodes_by_id["source:inventory_db"].type == "data_source"
    assert nodes_by_id["source:inventory_db"].label == "inventory_db"
    assert nodes_by_id["metric:ev_stock_days"].type == "metric"
    assert nodes_by_id["metric:ev_stock_days"].label == "Stock coverage in days"
    assert nodes_by_id["metric:ev_stock_days"].value == 1.5
    assert nodes_by_id["metric:ev_stock_days"].metadata == {
        "source": "inventory_db",
        "product": "croissant",
    }
    assert nodes_by_id["risk:InventoryAgent:0"].type == "risk"
    assert nodes_by_id["risk:InventoryAgent:0"].label == (
        "Croissants may stock out before the next delivery."
    )
    assert nodes_by_id["recommendation:rec_order_croissants"].type == "recommendation"
    assert nodes_by_id["recommendation:rec_order_croissants"].label == (
        "Order additional croissant ingredients."
    )

    assert ("source:inventory_db", "metric:ev_stock_days", "supports") in edges
    assert ("source:sales_history", "metric:ev_sales_rate", "supports") in edges
    assert ("metric:ev_stock_days", "claim:InventoryAgent", "supports") in edges
    assert ("metric:ev_sales_rate", "claim:InventoryAgent", "supports") in edges
    assert ("claim:InventoryAgent", "risk:InventoryAgent:0", "causes") in edges
    assert (
        "metric:ev_stock_days",
        "recommendation:rec_order_croissants",
        "justifies",
    ) in edges


def test_build_evidence_graph_creates_one_data_source_node_per_unique_source():
    first = AgentOutput(
        agent_name="InventoryAgent",
        claim="Inventory is constrained.",
        confidence=0.8,
        evidence_items=[
            EvidenceItem(
                id="ev_stock_days",
                source="inventory_db",
                description="Stock coverage in days",
                value=1.5,
            )
        ],
    )
    second = AgentOutput(
        agent_name="DemandAgent",
        claim="Demand remains elevated.",
        confidence=0.76,
        evidence_items=[
            EvidenceItem(
                id="ev_reorder_gap",
                source="inventory_db",
                description="Gap to reorder point",
                value=12,
            )
        ],
    )

    graph = build_evidence_graph(
        {"InventoryAgent": first, "DemandAgent": second}
    )

    source_nodes = [
        node
        for node in graph.nodes
        if node.type == "data_source" and node.label == "inventory_db"
    ]

    assert len(source_nodes) == 1


def test_serialize_evidence_graph_returns_model_dump_dict():
    output = AgentOutput(
        agent_name="DemandAgent",
        claim="Demand is stable.",
        confidence=0.72,
        evidence_items=[
            EvidenceItem(
                id="ev_demand_index",
                source="forecast_model",
                description="Demand index",
                value=0.67,
            )
        ],
    )

    serialized = serialize_evidence_graph(
        build_evidence_graph({"DemandAgent": output})
    )

    assert isinstance(serialized, dict)
    assert serialized["nodes"][0]["id"] == "claim:DemandAgent"
    assert {
        "source_id": "source:forecast_model",
        "target_id": "metric:ev_demand_index",
        "type": "supports",
        "confidence": None,
        "metadata": {},
    } in serialized["edges"]
