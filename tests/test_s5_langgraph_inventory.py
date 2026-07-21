import asyncio

from s5_agent.agents import inventory as inventory_module
from s5_agent.agents.inventory import InventoryAgent
from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request


def test_inventory_flow_reconciliation_includes_opening_stock():
    result = inventory_module._summarize_inflow_history(
        {
            "inflow_history": {
                "date": "2026-07-15",
                "records": [
                    {
                        "product_name": "bagel",
                        "quantity_opening": 2,
                        "quantity_baked": 3,
                        "quantity_sold": 4,
                        "quantity_discarded": 0,
                        "quantity_other_outflow": 0,
                        "quantity_left": 1,
                    }
                ],
            }
        },
        target_products=None,
    )

    assert result["flow_opening_units"] == 2
    assert result["flow_baked_units"] == 3
    assert result["flow_left_units"] == 1
    assert result["flow_balance_issue_count"] == 0
    assert result["flow_per_product"]["bagel"]["opening"] == 2


def test_inventory_snapshot_includes_flow_products_that_closed_at_zero():
    raw = {
        "status": "ok",
        "bread_stock": [
            {
                "product_name": "apple_pie",
                "fresh_qty": 1,
                "day1_qty": 0,
                "total_qty": 1,
            }
        ],
        "baking_materials": [],
        "beverage_materials": [],
        "packaging_materials": [],
        "inflow_history": {
            "status": "ok",
            "date": "2026-07-16",
            "records": [
                {
                    "product_name": "apple_pie",
                    "quantity_opening": 0,
                    "quantity_baked": 6,
                    "quantity_sold": 5,
                    "quantity_discarded": 0,
                    "quantity_other_outflow": 0,
                    "quantity_left": 1,
                    "data_quality_issue": False,
                },
                {
                    "product_name": "bagel",
                    "quantity_opening": 0,
                    "quantity_baked": 5,
                    "quantity_sold": 5,
                    "quantity_discarded": 0,
                    "quantity_other_outflow": 0,
                    "quantity_left": 0,
                    "data_quality_issue": False,
                },
                {
                    "product_name": "croissant",
                    "quantity_opening": 1,
                    "quantity_baked": 4,
                    "quantity_sold": 5,
                    "quantity_discarded": 0,
                    "quantity_other_outflow": 0,
                    "quantity_left": 0,
                    "data_quality_issue": False,
                },
            ],
        },
    }

    result = InventoryAgent().analyze(
        raw,
        {"product": "all", "date": "2026-07-16"},
    )

    assert result["data"]["product_count"] == 3
    assert result["data"]["zero_stock_products"] == ["bagel", "croissant"]
    assert result["data"]["per_product"]["bagel"]["qty"] == 0
    assert result["data"]["per_product"]["croissant"]["qty"] == 0


def test_inventory_summary_flags_expired_stock_pending_disposal():
    raw = {
        "status": "ok",
        "bread_stock": [
            {
                "product_name": "bagel",
                "fresh_qty": 3,
                "day1_qty": 0,
                "total_qty": 3,
            }
        ],
        "overdue_total": 4,
        "overdue_stock": [
            {
                "product_name": "brownie",
                "overdue_qty": 4,
                "oldest_production_date": "2026-06-30",
            }
        ],
        "baking_materials": [],
        "beverage_materials": [],
        "packaging_materials": [],
    }
    request = S5Request(
        query="Check inventory",
        module="inventory",
        params={"product": "all", "date": "2026-07-02"},
    )

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={"inventory": raw},
        )
    )

    finished_stock = next(
        output
        for output in result.agent_outputs
        if output.agent_name == "FinishedStockAgent"
    )
    assert finished_stock.metrics["overdue_stock_total"] == 4
    assert finished_stock.metrics["overdue_stock_products"] == ["brownie"]
    assert "expired_stock_pending_disposal_risk" in finished_stock.risks
    assert "4 expired units remain pending disposal" in result.summary
    recommendation_ids = [item.id for item in result.recommendations]
    assert "inventory_expired_stock_audit" in recommendation_ids


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


def test_historical_inventory_fetch_does_not_fall_back_to_current_stock(monkeypatch):
    calls = []

    def fake_fetch(url, params, timeout=10):
        calls.append(url)
        return {}

    monkeypatch.setattr(inventory_module, "fetch_dashboard_json", fake_fetch)

    agent = InventoryAgent()
    params = {
        "date": "2026-06-30",
        "module": "inventory",
        "product": "all",
    }
    raw = asyncio.run(agent.fetch(params))
    output = agent.analyze_for_graph(raw, params)

    assert len(calls) == 1
    assert calls[0].endswith("/s4/inventory/dashboard?date=2026-06-30")
    assert raw["bread_stock"] == []
    assert output.metrics["snapshot_date"] == "2026-06-30"
    assert output.metrics["product_count"] == 0
    assert "inventory_data_gap" in output.risks


def test_inventory_fetch_adds_selected_date_inflow_history(monkeypatch):
    calls = []

    def fake_fetch(url, params, timeout=10):
        calls.append((url, params.get("_authorization"), timeout))
        if "/s1/inflow/history" in url:
            return {
                "status": "ok",
                "date": "2026-07-15",
                "remaining_label": "Left Now",
                "records": [{"product_name": "croissant", "quantity_baked": 8}],
            }
        return {"status": "ok", "bread_stock": []}

    monkeypatch.setattr(inventory_module, "fetch_dashboard_json", fake_fetch)
    monkeypatch.setattr(
        inventory_module,
        "S4_DASHBOARD_URL",
        "http://127.0.0.1:8002/s4/inventory/dashboard",
    )
    monkeypatch.setattr(
        inventory_module,
        "S1_INFLOW_HISTORY_URL",
        "http://127.0.0.1:8002/s1/inflow/history",
        raising=False,
    )

    result = asyncio.run(
        InventoryAgent().fetch(
            {
                "date": "2026-07-15",
                "product": "all",
                "_authorization": "Bearer manager-token",
            }
        )
    )

    assert result["inflow_history"]["records"][0]["quantity_baked"] == 8
    assert calls == [
        (
            "http://127.0.0.1:8002/s4/inventory/dashboard?date=2026-07-15",
            "Bearer manager-token",
            10,
        ),
        (
            "http://127.0.0.1:8002/s1/inflow/history?date=2026-07-15",
            "Bearer manager-token",
            10,
        ),
    ]


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


def test_empty_dashboard_snapshot_does_not_fall_back_to_current_batch_stock(monkeypatch):
    def fail_current_stock_query(self, product_filter=None):
        raise AssertionError("Historical dashboard data must not be replaced by current stock")

    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", fail_current_stock_query)
    request = S5Request(
        query="Check inventory",
        module="inventory",
        params={"product": "all", "date": "2026-07-01"},
    )

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={
                "inventory": {
                    "status": "ok",
                    "bread_stock": [],
                    "baking_materials": [],
                    "coffee_materials": [],
                }
            },
        )
    )

    assert "No finished-product inventory records are available" in result.summary
    assert result.recommendations[0].id == "inventory_stock_record_audit"


def test_inventory_summary_reports_thin_stock_without_low_risk_label(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)
    raw = {
        "status": "ok",
        "bread_stock": [
            {"product_name": "bread_roll", "fresh_qty": 0, "day1_qty": 0, "total_qty": 0},
            {"product_name": "cornbread", "fresh_qty": 0, "day1_qty": 0, "total_qty": 0},
            {"product_name": "macaron", "fresh_qty": 1, "day1_qty": 0, "total_qty": 1},
            {"product_name": "croissant", "fresh_qty": 2, "day1_qty": 0, "total_qty": 2},
        ],
        "baking_materials": [],
        "coffee_materials": [],
    }
    request = S5Request(
        query="Check inventory",
        module="inventory",
        params={"product": "all", "date": "2026-07-15"},
    )

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={"inventory": raw},
        )
    )
    outputs = {output.agent_name: output for output in result.agent_outputs}
    finished_stock = outputs["FinishedStockAgent"]

    assert "low" not in finished_stock.risks
    assert "stockout_risk" in finished_stock.risks
    assert "widespread_low_stock_risk" in finished_stock.risks
    assert finished_stock.metrics["low_stock_product_count"] == 1
    assert finished_stock.metrics["thin_stock_product_share_pct"] == 75.0
    assert finished_stock.metrics["units_per_product"] == 0.75
    assert "For 2026-07-15, finished-product stock totals 3 units across 4 tracked products" in result.summary
    assert "3 of 4 products (75.0%) have zero or one unit remaining" in result.summary
    assert "2 products have no recorded finished stock" in result.summary
    assert "1 additional product has only 1 unit" in result.summary
    assert "\n\n" in result.summary
    recommendation_ids = [rec.id for rec in result.recommendations]
    assert "inventory_zero_stock_plan_check" in recommendation_ids
    assert "inventory_thin_stock_plan_check" in recommendation_ids
    zero_stock_rec = next(rec for rec in result.recommendations if rec.id == "inventory_zero_stock_plan_check")
    thin_stock_rec = next(rec for rec in result.recommendations if rec.id == "inventory_thin_stock_plan_check")
    assert "bread roll and cornbread" in zero_stock_rec.action
    assert "macaron" not in zero_stock_rec.action
    assert "1 product with only one unit" in thin_stock_rec.action


def test_inventory_summary_omits_additional_when_no_product_has_zero_stock(
    monkeypatch,
):
    raw = {
        "status": "ok",
        "bread_stock": [
            {"product_name": "bagel", "fresh_qty": 1, "day1_qty": 0, "total_qty": 1},
            {"product_name": "donut", "fresh_qty": 1, "day1_qty": 0, "total_qty": 1},
            {"product_name": "muffin", "fresh_qty": 2, "day1_qty": 0, "total_qty": 2},
        ],
        "baking_materials": [],
        "beverage_materials": [],
        "packaging_materials": [],
    }
    request = S5Request(
        query="Check inventory",
        module="inventory",
        params={"product": "all", "date": "2026-06-30"},
    )

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={"inventory": raw},
        )
    )

    assert "2 products have only 1 unit each" in result.summary
    assert "2 additional products" not in result.summary


def test_inventory_summary_explains_stock_with_inflow_movements(monkeypatch):
    raw = {
        "status": "ok",
        "bread_stock": [
            {"product_name": "croissant", "fresh_qty": 1, "day1_qty": 0, "total_qty": 1},
            {"product_name": "muffin", "fresh_qty": 7, "day1_qty": 0, "total_qty": 7},
        ],
        "baking_materials": [],
        "coffee_materials": [],
        "inflow_history": {
            "status": "ok",
            "date": "2026-07-15",
            "remaining_label": "Left Now",
            "records": [
                {
                    "product_name": "croissant",
                    "quantity_baked": 10,
                    "quantity_sold": 9,
                    "quantity_discarded": 0,
                    "quantity_other_outflow": 0,
                    "quantity_left": 1,
                    "data_quality_issue": False,
                },
                {
                    "product_name": "muffin",
                    "quantity_baked": 10,
                    "quantity_sold": 2,
                    "quantity_discarded": 1,
                    "quantity_other_outflow": 0,
                    "quantity_left": 7,
                    "data_quality_issue": False,
                },
            ],
        },
    }
    request = S5Request(
        query="Check inventory",
        module="inventory",
        params={"product": "all", "date": "2026-07-15"},
    )

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={"inventory": raw},
        )
    )
    outputs = {output.agent_name: output for output in result.agent_outputs}
    finished_stock = outputs["FinishedStockAgent"]
    recommendation_ids = [item.id for item in result.recommendations]

    assert finished_stock.metrics["flow_record_count"] == 2
    assert finished_stock.metrics["flow_baked_units"] == 20
    assert finished_stock.metrics["flow_sold_units"] == 11
    assert finished_stock.metrics["flow_discarded_units"] == 1
    assert finished_stock.metrics["flow_left_units"] == 8
    assert finished_stock.metrics["flow_sell_through_pct"] == 55.0
    assert finished_stock.metrics["high_sell_through_products"] == ["croissant"]
    assert finished_stock.metrics["slow_moving_products"] == ["muffin"]
    assert "high_sell_through_stock_risk" in finished_stock.risks
    assert "overproduction_risk" in finished_stock.risks
    assert "inventory_high_sell_through_review" in recommendation_ids
    assert "inventory_slow_moving_bake_review" in recommendation_ids
    assert "20 units were baked, 11 were sold, 1 was discarded, and 8 remain" in result.summary
    assert "55.0% sell-through" in result.summary
    assert "croissant is close to selling through" in result.summary
    assert "muffin has comparatively slow movement" in result.summary
    assert "does not replace the demand forecast" in result.summary


def test_inventory_high_sell_through_products_are_ranked_and_split_from_remaining_thin_stock(monkeypatch):
    raw = {
        "status": "ok",
        "bread_stock": [
            {"product_name": "apple_pie", "fresh_qty": 1, "day1_qty": 0, "total_qty": 1},
            {"product_name": "baguette", "fresh_qty": 1, "day1_qty": 0, "total_qty": 1},
            {"product_name": "brownie", "fresh_qty": 1, "day1_qty": 0, "total_qty": 1},
            {"product_name": "croissant", "fresh_qty": 1, "day1_qty": 0, "total_qty": 1},
        ],
        "baking_materials": [],
        "coffee_materials": [],
        "inflow_history": {
            "status": "ok",
            "date": "2026-07-15",
            "remaining_label": "Left Now",
            "records": [
                {
                    "product_name": "apple_pie",
                    "quantity_baked": 6,
                    "quantity_sold": 5,
                    "quantity_discarded": 0,
                    "quantity_other_outflow": 0,
                    "quantity_left": 1,
                    "data_quality_issue": False,
                },
                {
                    "product_name": "baguette",
                    "quantity_baked": 4,
                    "quantity_sold": 3,
                    "quantity_discarded": 0,
                    "quantity_other_outflow": 0,
                    "quantity_left": 1,
                    "data_quality_issue": False,
                },
                {
                    "product_name": "brownie",
                    "quantity_baked": 7,
                    "quantity_sold": 6,
                    "quantity_discarded": 0,
                    "quantity_other_outflow": 0,
                    "quantity_left": 1,
                    "data_quality_issue": False,
                },
                {
                    "product_name": "croissant",
                    "quantity_baked": 7,
                    "quantity_sold": 6,
                    "quantity_discarded": 0,
                    "quantity_other_outflow": 0,
                    "quantity_left": 1,
                    "data_quality_issue": False,
                },
            ],
        },
    }
    request = S5Request(
        query="Check inventory",
        module="inventory",
        params={"product": "all", "date": "2026-07-15"},
    )

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={"inventory": raw},
        )
    )
    finished_stock = next(
        output for output in result.agent_outputs
        if output.agent_name == "FinishedStockAgent"
    )
    high_sell_through_rec = next(
        item for item in result.recommendations
        if item.id == "inventory_high_sell_through_review"
    )
    thin_stock_rec = next(
        item for item in result.recommendations
        if item.id == "inventory_thin_stock_plan_check"
    )

    assert finished_stock.metrics["high_sell_through_products"] == [
        "brownie",
        "croissant",
        "apple_pie",
    ]
    assert "3 products have high sell-through and no more than one unit left" in result.summary
    assert "led by brownie, croissant, and apple pie" in result.summary
    assert "Prioritize the 3 high-sell-through products" in high_sell_through_rec.action
    assert "led by brownie, croissant, and apple pie" in high_sell_through_rec.action
    assert "Review the remaining 1 product with only one unit" in thin_stock_rec.action


def test_inventory_flow_balance_issue_is_reported(monkeypatch):
    raw = {
        "status": "ok",
        "bread_stock": [
            {"product_name": "bagel", "fresh_qty": 0, "day1_qty": 0, "total_qty": 0},
        ],
        "baking_materials": [],
        "coffee_materials": [],
        "inflow_history": {
            "status": "ok",
            "date": "2026-07-15",
            "remaining_label": "Left Now",
            "records": [
                {
                    "product_name": "bagel",
                    "quantity_baked": 4,
                    "quantity_sold": 3,
                    "quantity_discarded": 2,
                    "quantity_other_outflow": 0,
                    "quantity_left": 0,
                    "data_quality_issue": True,
                },
            ],
        },
    }
    request = S5Request(
        query="Check inventory",
        module="inventory",
        params={"product": "all", "date": "2026-07-15"},
    )

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={"inventory": raw},
        )
    )
    outputs = {output.agent_name: output for output in result.agent_outputs}

    assert outputs["FinishedStockAgent"].metrics["flow_balance_issue_count"] == 1
    assert "inventory_flow_data_gap" in outputs["FinishedStockAgent"].risks
    assert "inventory_flow_record_audit" in [item.id for item in result.recommendations]
    assert "does not reconcile" in result.summary


def test_inventory_summary_includes_raw_material_reorder_risk(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)
    raw = {
        "status": "ok",
        "bread_stock": [
            {"product_name": "macaron", "fresh_qty": 4, "day1_qty": 0, "total_qty": 4},
        ],
        "baking_materials": [
            {"material_name": "Bread Flour", "stock": 8.0, "reorder_point": 10.0, "unit": "kg"},
            {"material_name": "Butter", "stock": 0.0, "reorder_point": 2.0, "unit": "kg"},
        ],
        "coffee_materials": [
            {"material_name": "Coffee Beans", "stock": 4.0, "reorder_point": 2.0, "unit": "kg"},
        ],
    }
    request = S5Request(
        query="Check inventory",
        module="inventory",
        params={"product": "all", "date": "2026-07-15"},
    )

    result = asyncio.run(
        run_s5_graph(
            "inventory_diagnosis",
            request,
            raw_inputs={"inventory": raw},
        )
    )
    outputs = {output.agent_name: output for output in result.agent_outputs}
    finished_stock = outputs["FinishedStockAgent"]

    assert finished_stock.metrics["raw_material_count"] == 3
    assert finished_stock.metrics["low_stock_material_count"] == 2
    assert finished_stock.metrics["critical_material_count"] == 1
    assert "Raw-material status covers the 3 materials shown in the Inventory dashboard" in result.summary
    assert "Butter is out of stock" in result.summary
    assert "Bread Flour is at or below its reorder point" in result.summary
    assert "material_shortage_risk" in finished_stock.risks
    assert any(rec.id == "inventory_material_restock_check" for rec in result.recommendations)


def test_inventory_agent_deduplicates_shared_material_groups(monkeypatch):
    monkeypatch.setattr(
        InventoryAgent,
        "_query_db_freshness",
        lambda self, product_filter=None: None,
    )
    raw = {
        "status": "ok",
        "bread_stock": [
            {
                "product_name": "macaron",
                "fresh_qty": 2,
                "day1_qty": 0,
                "total_qty": 2,
            }
        ],
        "baking_materials": [
            {
                "material_name": "Milk",
                "stock": 10.0,
                "reorder_point": 2.0,
                "unit": "L",
            }
        ],
        "beverage_materials": [
            {
                "material_name": "Milk",
                "stock": 10.0,
                "reorder_point": 2.0,
                "unit": "L",
            },
            {
                "material_name": "Coffee Beans",
                "stock": 4.0,
                "reorder_point": 2.0,
                "unit": "kg",
            },
        ],
        "coffee_materials": [
            {
                "material_name": "Legacy Duplicate",
                "stock": 1.0,
                "reorder_point": 1.0,
                "unit": "kg",
            }
        ],
        "packaging_materials": [
            {
                "material_name": "Packaging Bag",
                "stock": 20.0,
                "reorder_point": 5.0,
                "unit": "pcs",
            }
        ],
    }

    result = InventoryAgent().analyze(raw, {"product": "all"})

    assert [
        item["material_name"] for item in result["data"]["raw_materials"]
    ] == ["Coffee Beans", "Milk", "Packaging Bag"]
