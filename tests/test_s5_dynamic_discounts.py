from fastapi.testclient import TestClient


def test_live_policy_uses_relative_inventory_pressure_without_cached_analysis():
    from s5_agent.discount_policy import build_discount_decisions

    decisions = build_discount_decisions(
        products=["bread_coconut", "brownie"],
        inventory_rows=[
            {"product_name": "bread_coconut", "freshness_status": "Fresh", "remaining": 10},
            {"product_name": "brownie", "freshness_status": "Fresh", "remaining": 30},
        ],
        sales_rows=[
            {"product_name": "bread_coconut", "period": "latest", "quantity": 0},
            {"product_name": "bread_coconut", "period": "previous", "quantity": 3},
            {"product_name": "brownie", "period": "latest", "quantity": 0},
            {"product_name": "brownie", "period": "previous", "quantity": 1},
        ],
        product_rows=[
            {"product_name": "bread_coconut", "selling_price": 9, "material_cost": 1.27, "wastage_pct": 0.05},
            {"product_name": "brownie", "selling_price": 16, "material_cost": 5.59, "wastage_pct": 0.05},
        ],
    )

    assert decisions["brownie"]["discount_pct"] == 12
    assert decisions["brownie"]["strategy"] == "diversify"
    assert decisions["brownie"]["source"] == "live_policy"
    assert decisions["brownie"]["dynamic"] is True
    assert decisions["brownie"]["evidence"]["stock_coverage"] > decisions["brownie"]["evidence"]["coverage_benchmark"]


def test_live_policy_keeps_day1_clearance_and_cached_priority_precedence():
    from s5_agent.discount_policy import build_discount_decisions

    decisions = build_discount_decisions(
        products=["croissant", "baguette"],
        inventory_rows=[
            {"product_name": "croissant", "freshness_status": "Day-1", "remaining": 4},
            {"product_name": "baguette", "freshness_status": "Fresh", "remaining": 4},
        ],
        sales_rows=[],
        product_rows=[],
        priority_map={
            "baguette": {
                "strategy": "amplify",
                "discount_pct": 15,
                "reason": "Current revenue momentum signal",
            }
        },
    )

    assert decisions["croissant"]["discount_pct"] == 40
    assert decisions["croissant"]["strategy"] == "clearance"
    assert decisions["baguette"]["discount_pct"] == 15
    assert decisions["baguette"]["strategy"] == "amplify"


def test_discount_endpoint_uses_live_policy_when_revenue_cache_is_empty(monkeypatch):
    from s5_agent import server

    monkeypatch.setattr(server, "_latest_cached_synthesis", lambda _intent: {})
    monkeypatch.setattr(
        server,
        "get_live_discounts",
        lambda products, priority_map=None: {
            product: {
                "discount_pct": 12,
                "freshness": "Fresh",
                "strategy": "diversify",
                "reason": "Live inventory pressure",
                "source": "live_policy",
                "dynamic": True,
                "evidence": {"stock_remaining": 30},
            }
            for product in products
        },
    )

    response = TestClient(server.app).post("/discounts", json={"products": ["brownie"]})

    assert response.status_code == 200
    body = response.json()["discounts"]["brownie"]
    assert body["discount_pct"] == 12
    assert body["source"] == "live_policy"
    assert body["evidence"]["stock_remaining"] == 30


def test_single_product_validation_uses_all_sellable_products_as_benchmark():
    from s5_agent.discount_policy import get_live_discounts

    class PolicyCursor:
        def __init__(self):
            self.rows = []

        def execute(self, sql, params=None):
            if "SELECT DISTINCT product_name FROM batch_inventory" in sql:
                self.rows = [
                    {"product_name": "bread_coconut"},
                    {"product_name": "brownie"},
                ]
            elif "FROM batch_inventory" in sql:
                self.rows = [
                    {"product_name": "bread_coconut", "freshness_status": "Fresh", "remaining": 10},
                    {"product_name": "brownie", "freshness_status": "Fresh", "remaining": 30},
                ]
            elif "SELECT DISTINCT order_date" in sql:
                self.rows = [{"order_date": "2026-07-03"}, {"order_date": "2026-07-02"}]
            elif "FROM order_items" in sql:
                self.rows = [
                    {"product_name": "bread_coconut", "order_date": "2026-07-02", "quantity": 3},
                    {"product_name": "brownie", "order_date": "2026-07-02", "quantity": 1},
                ]
            elif "FROM products" in sql:
                self.rows = [
                    {"product_name": "bread_coconut", "selling_price": 9, "material_cost": 1.27, "wastage_pct": 0.05},
                    {"product_name": "brownie", "selling_price": 16, "material_cost": 5.59, "wastage_pct": 0.05},
                ]
            else:
                raise AssertionError(f"Unexpected SQL: {sql}")

        def fetchall(self):
            return self.rows

        def close(self):
            return None

    class PolicyDB:
        def cursor(self, dictionary=False):
            return PolicyCursor()

    decisions = get_live_discounts(["bread_coconut"], db=PolicyDB())

    assert list(decisions) == ["bread_coconut"]
    assert decisions["bread_coconut"]["discount_pct"] == 25
    assert decisions["bread_coconut"]["evidence"]["margin_benchmark_pct"] < decisions["bread_coconut"]["evidence"]["margin_pct"]
