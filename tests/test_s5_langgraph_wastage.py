import asyncio

from s5_agent.agents import yield_agent
from s5_agent.agents.inventory import InventoryAgent
from s5_agent.agents.wastage import TREND_SQL, WASTAGE_SQL
from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request


def test_wastage_queries_exclude_untracked_materials():
    assert "WHERE rm.track_inventory = 1" in WASTAGE_SQL
    assert "JOIN raw_materials rm ON rm.material_name = mw.material_name" in TREND_SQL
    assert "rm.track_inventory = 1" in TREND_SQL


def test_yield_queries_use_actual_production_transactions():
    assert "FROM material_transactions mt" in yield_agent.YIELD_SQL
    assert "mt.reference LIKE 'production:%'" in yield_agent.YIELD_SQL
    assert "FROM inventory_transactions it" in yield_agent.PRODUCT_COUNT_SQL
    assert "it.transaction_type = 'inflow'" in yield_agent.PRODUCT_COUNT_SQL
    assert "p.category = 'bakery'" in yield_agent.PRODUCT_COUNT_SQL
    assert not hasattr(yield_agent, "_populate_batch_inventory")


def test_yield_fetch_is_read_only_and_returns_actual_production(monkeypatch):
    executed_sql = []

    class FakeCursor:
        def __init__(self):
            self.rows = []

        def execute(self, sql, params):
            executed_sql.append(sql)
            if "FROM material_transactions mt" in sql:
                self.rows = [
                    ("Bread Flour", 2.3587, 74.555341, "kg", 12.0),
                    ("Butter", 1.077302, 19.91897, "kg", 5.0),
                ]
            elif "FROM inventory_transactions it" in sql:
                self.rows = [(30, 158)]
            else:
                raise AssertionError(f"Unexpected yield query: {sql}")

        def fetchall(self):
            return list(self.rows)

        def fetchone(self):
            return self.rows[0] if self.rows else None

        def close(self):
            return None

    class FakeDatabase:
        def __init__(self):
            self.cursor_instance = FakeCursor()
            self.closed = False

        def cursor(self):
            return self.cursor_instance

        def close(self):
            self.closed = True

    database = FakeDatabase()
    connection_count = 0

    def fake_get_db():
        nonlocal connection_count
        connection_count += 1
        return database

    monkeypatch.setattr("db.mysql_client.get_db", fake_get_db)

    result = yield_agent._fetch_yield_data("2026-07-15")

    assert result == {
        "materials": [
            {
                "material_name": "Bread Flour",
                "total_consumed": 2.3587,
                "current_stock": 74.555341,
                "unit": "kg",
                "threshold": 12.0,
            },
            {
                "material_name": "Butter",
                "total_consumed": 1.077302,
                "current_stock": 19.91897,
                "unit": "kg",
                "threshold": 5.0,
            },
        ],
        "product_count": 30,
        "total_units": 158,
    }
    assert connection_count == 1
    assert len(executed_sql) == 2
    assert all(sql.lstrip().upper().startswith("SELECT") for sql in executed_sql)
    assert database.closed is True


def _wastage_material(
    name,
    *,
    qty=0.0,
    rate=0.0,
    cost_per_unit=1.0,
    consumed=0.0,
    material_id=1,
):
    return {
        "material_name": name,
        "id": material_id,
        "current_stock": 20.0,
        "unit": "kg",
        "cost_per_unit": cost_per_unit,
        "theoretical_consumed": consumed,
        "actual_consumed": consumed + qty,
        "wastage_qty": qty,
        "wastage_rate": rate,
        "check_date": "2026-06-30",
    }


def test_wastage_graph_zero_waste_stays_evidence_limited(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)
    raw = {
        "wastage": {
            "data": {
                "trend": {
                    "materials": [
                        {
                            "material_name": "Bread Flour",
                            "check_date": "2026-07-02",
                            "wastage_qty": 0.0,
                            "wastage_rate": 0.0,
                            "theoretical_consumed": 12.0,
                            "actual_consumed": 12.0,
                        },
                        {
                            "material_name": "Butter",
                            "check_date": "2026-07-02",
                            "wastage_qty": 0.0,
                            "wastage_rate": 0.0,
                            "theoretical_consumed": 3.0,
                            "actual_consumed": 3.0,
                        },
                    ]
                },
                "latest": {
                    "materials": [
                        {
                            "material_name": "Bread Flour",
                            "current_stock": 80.0,
                            "unit": "kg",
                            "cost_per_unit": 4.0,
                            "theoretical_consumed": 12.0,
                            "actual_consumed": 12.0,
                            "wastage_qty": 0.0,
                            "wastage_rate": 0.0,
                            "check_date": "2026-07-02",
                        },
                        {
                            "material_name": "Butter",
                            "current_stock": 35.0,
                            "unit": "kg",
                            "cost_per_unit": 18.0,
                            "theoretical_consumed": 3.0,
                            "actual_consumed": 3.0,
                            "wastage_qty": 0.0,
                            "wastage_rate": 0.0,
                            "check_date": "2026-07-02",
                        },
                    ]
                },
            }
        },
        "yield": {"data": {"materials": [], "product_count": 0, "total_units": 0}},
        "inventory": {
            "inventory": [
                {"product_name": "bread_roll", "total_quantity": 0, "batches": 1, "selling_price": 6.5},
                {"product_name": "cornbread", "total_quantity": 0, "batches": 1, "selling_price": 8.0},
                {"product_name": "macaron", "total_quantity": 1, "batches": 1, "selling_price": 10.0},
                {"product_name": "croissant", "total_quantity": 2, "batches": 1, "selling_price": 12.0},
            ]
        },
    }
    request = S5Request(
        query="Check wastage",
        module="wastage",
        params={"date": "2026-07-07", "product": "all"},
    )

    result = asyncio.run(run_s5_graph("wastage_root_cause", request, raw_inputs=raw))

    assert result.metadata["template"] == "wastage_root_cause"
    assert result.verification_report.passed is True
    assert result.warnings
    assert "Material wastage records are clean" in result.summary
    assert "No wastage check was submitted on 2026-07-07" in result.summary
    assert "latest available material wastage records up to this date are from 2026-07-02" in result.summary
    assert "Production-yield risk cannot be fully verified today" in result.summary
    assert "production-yield risk cannot" not in result.summary
    assert "all costs flowed straight into sales" not in result.summary
    assert "working perfectly" not in result.summary
    assert "completely clean" not in result.summary
    outputs_by_agent = {output.agent_name: output for output in result.agent_outputs}
    assert outputs_by_agent["WastageAgent"].metrics["material_count_checked"] == 2
    assert outputs_by_agent["WastageAgent"].metrics["wasted_material_count"] == 0
    assert outputs_by_agent["WastageAgent"].metrics["total_waste_cost"] == 0.0
    assert outputs_by_agent["WastageAgent"].metrics["latest_wastage_record_date"] == "2026-07-02"
    assert outputs_by_agent["YieldAgent"].metrics["yield_data_available"] is False
    assert outputs_by_agent["FinishedStockAgent"].risks == []
    assert "low" not in {
        risk
        for output in result.agent_outputs
        for risk in output.risks
    }
    evidence_ids = {
        evidence.id
        for output in result.agent_outputs
        for evidence in output.evidence_items
    }
    assert {
        "material_count_checked",
        "wasted_material_count",
        "total_waste_cost",
        "latest_wastage_record_date",
        "yield_data_available",
        "inventory_total",
    } <= evidence_ids
    recommendation_text = " ".join(recommendation.action for recommendation in result.recommendations)
    assert "Record production yield data" in recommendation_text
    assert "Verify zero-waste material entries" in recommendation_text


def test_wastage_summary_reads_like_business_analysis_when_loss_is_small(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)
    names = [
        "Butter",
        "Coffee Beans",
        "Bread Flour",
        "Sugar",
        "Baking Powder",
        "Box",
        "Milk",
        "Eggs",
        "Yeast",
        "Chocolate",
        "Cream",
        "Cheese",
        "Lids",
        "Cups",
        "Packaging Bag",
        "Tea",
        "Cake Flour",
        "Cup Regular",
    ]
    material_overrides = {
        "Butter": {"qty": 0.010, "cost_per_unit": 60.0},
        "Coffee Beans": {"qty": 0.006, "cost_per_unit": 80.0},
        "Bread Flour": {"qty": 0.010, "cost_per_unit": 8.0, "consumed": 50.0},
        "Eggs": {"qty": 0.007, "cost_per_unit": 12.0},
        "Baking Powder": {"consumed": 100.0},
        "Box": {"consumed": 80.0},
        "Milk": {"consumed": 40.0},
    }
    materials = [
        _wastage_material(name, material_id=index + 1, **material_overrides.get(name, {}))
        for index, name in enumerate(names)
    ]
    raw = {
        "wastage": {
            "data": {
                "trend": {"materials": materials},
                "latest": {"materials": materials},
            }
        },
        "yield": {
            "data": {
                "materials": [{"material_name": f"Material {index}"} for index in range(10)],
                "product_count": 5,
                "total_units": 261,
            }
        },
        "inventory": {"inventory": []},
    }
    request = S5Request(
        query="Check wastage",
        module="wastage",
        params={"date": "2026-06-30", "product": "all"},
    )

    result = asyncio.run(run_s5_graph("wastage_root_cause", request, raw_inputs=raw))
    summary = result.summary
    currency = chr(165)

    assert summary.startswith("For 2026-06-30, recorded material waste is limited in cost, but it should still be checked.")
    assert f"The system found waste in 4 of 18 checked materials, with a total recorded waste cost of {currency}1.24." in summary
    assert "This is not a major financial loss yet" in summary
    assert "repeated small losses in the same materials could become a process issue" in summary
    assert "The main items to review are Butter, Coffee Beans, and Bread Flour." in summary
    assert f"Eggs also logged a small waste entry at {currency}0.08, but it is lower priority than the top three losses." in summary
    assert f"Butter caused the largest recorded loss at {currency}0.60, followed by Coffee Beans at {currency}0.48 and Bread Flour at {currency}0.08." in summary
    assert "Their wastage rates cannot be calculated reliably because theoretical consumption is recorded as zero" in summary
    assert "Production records for this date show 261 baked units and recorded consumption for 10 materials." in summary
    assert "finished stock should still be interpreted separately from material waste" in summary


def test_wastage_recommendations_keep_small_cost_precision(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)
    raw = {
        "wastage": {
            "data": {
                "trend": {
                    "materials": [
                        {
                            "material_name": "Coffee Beans",
                            "id": 1,
                            "check_date": "2026-07-07",
                            "wastage_qty": 0.0,
                            "wastage_rate": 0.0,
                            "theoretical_consumed": 0.5,
                            "actual_consumed": 0.5,
                        },
                        {
                            "material_name": "Coffee Beans",
                            "id": 2,
                            "check_date": "2026-07-07",
                            "wastage_qty": 0.006,
                            "wastage_rate": 0.0119,
                            "theoretical_consumed": 0.5,
                            "actual_consumed": 0.506,
                        }
                    ]
                },
                "latest": {
                    "materials": [
                        {
                            "material_name": "Coffee Beans",
                            "current_stock": 8.0,
                            "unit": "kg",
                            "cost_per_unit": 20.0,
                            "theoretical_consumed": 0.5,
                            "actual_consumed": 0.506,
                            "wastage_qty": 0.006,
                            "wastage_rate": 0.0119,
                            "check_date": "2026-07-07",
                        }
                    ]
                },
            }
        },
        "yield": {"data": {"materials": [], "product_count": 0, "total_units": 0}},
        "inventory": {"inventory": []},
    }
    request = S5Request(
        query="Check wastage",
        module="wastage",
        params={"date": "2026-07-07", "product": "all"},
    )

    result = asyncio.run(run_s5_graph("wastage_root_cause", request, raw_inputs=raw))
    recommendation_text = " ".join(recommendation.rationale for recommendation in result.recommendations)
    outputs_by_agent = {output.agent_name: output for output in result.agent_outputs}

    currency = chr(165)

    assert outputs_by_agent["WastageAgent"].metrics["wasted_material_count"] == 1
    assert f"Coffee Beans logged material waste at 1.2% with {currency}0.12 recorded cost." in recommendation_text
    assert f"with {currency}0 recorded cost" not in recommendation_text
