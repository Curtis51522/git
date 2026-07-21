import asyncio

from s3_scheduling import scheduler as scheduler_module
from s3_scheduling.scheduler import Scheduler, BREAD_CAPACITY, DEMAND_BUFFER


def make_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.products = {
        "croissant": {"price": 10.0, "is_drink": False},
        "baguette": {"price": 8.0, "is_drink": False},
        "latte": {"price": 12.0, "is_drink": True},
    }
    scheduler.breads = {
        name: info
        for name, info in scheduler.products.items()
        if not info["is_drink"]
    }
    scheduler.drinks = {
        name: info
        for name, info in scheduler.products.items()
        if info["is_drink"]
    }
    return scheduler


def test_generate_7day_forecast_uses_one_continuous_s2_request(monkeypatch):
    from api import module2_forecast

    calls = []

    def fake_forecast(product, days, use_cache, start_date):
        calls.append((product, days, use_cache, start_date))
        return {
            "status": "ok",
            "forecasts": [
                {
                    "forecast_date": "2026-06-30",
                    "product_name": "croissant",
                    "predicted_demand": 10,
                    "lower_bound": 8,
                    "upper_bound": 13,
                },
                {
                    "forecast_date": "2026-07-01",
                    "product_name": "croissant",
                    "predicted_demand": 12,
                    "lower_bound": 9,
                    "upper_bound": 15,
                },
            ],
        }

    monkeypatch.setattr(module2_forecast, "_do_forecast", fake_forecast)
    result = scheduler_module.generate_7day_s2_forecast("2026-06-30")

    assert calls == [(None, 7, True, "2026-06-30")]
    assert len(result) == 7
    assert result["2026-06-30"]["croissant"] == {
        "q10": 8,
        "q50": 10,
        "q90": 13,
    }
    assert result["2026-07-01"]["croissant"]["q50"] == 12
    assert result["2026-07-06"] == {}


def test_actual_replay_carries_only_unsold_fresh_stock_forward():
    scheduler = make_scheduler()

    outcome = scheduler.replay_actual_outcome(
        bake_plan={"croissant": 5},
        day1_stock={"croissant": 3},
        actual_demand={"croissant": 2},
    )

    assert outcome["day1_sold"]["croissant"] == 2
    assert outcome["fresh_sold"]["croissant"] == 0
    assert outcome["waste"]["croissant"] == 1
    assert outcome["next_day1_stock"]["croissant"] == 5
    assert outcome["shortage"]["croissant"] == 0
    assert outcome["sales_units"] == 2
    assert outcome["waste_units"] == 1


def test_actual_replay_reports_shortage_when_stock_and_bake_are_insufficient():
    scheduler = make_scheduler()

    outcome = scheduler.replay_actual_outcome(
        bake_plan={"croissant": 2},
        day1_stock={"croissant": 1},
        actual_demand={"croissant": 6},
    )

    assert outcome["day1_sold"]["croissant"] == 1
    assert outcome["fresh_sold"]["croissant"] == 2
    assert outcome["shortage"]["croissant"] == 3
    assert outcome["next_day1_stock"]["croissant"] == 0
    assert outcome["sales_units"] == 3
    assert outcome["fill_rate"] == 0.5


def test_q50_baseline_plan_applies_buffer_and_capacity_limit():
    scheduler = make_scheduler()
    forecast = {
        "croissant": {"q50": BREAD_CAPACITY},
        "baguette": {"q50": BREAD_CAPACITY},
        "latte": {"q50": BREAD_CAPACITY},
    }

    plan = scheduler.build_q50_baseline_plan(forecast, capacity=BREAD_CAPACITY)

    assert "latte" not in plan
    assert sum(plan.values()) <= BREAD_CAPACITY
    assert plan["croissant"] > 0
    assert plan["baguette"] > 0


def test_q50_baseline_plan_uses_buffer_when_capacity_allows():
    scheduler = make_scheduler()
    forecast = {"croissant": {"q50": 10}}

    plan = scheduler.build_q50_baseline_plan(forecast, capacity=BREAD_CAPACITY)

    assert plan["croissant"] == int(10 * DEMAND_BUFFER)


def test_material_dashboard_does_not_convert_database_failure_to_zero_stock(monkeypatch):
    from db import mysql_client

    scheduler = make_scheduler()

    def fail_database_connection():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(mysql_client, "get_db", fail_database_connection)

    result = scheduler.dashboard_format_materials(
        {"materials": {"Bread Flour": 10.0}}
    )

    assert result["stock_data_available"] is False
    assert result["items"] == {}
    assert result["error"] == "raw_material_stock_unavailable"


def test_material_estimate_uses_only_direct_database_recipe(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "load_recipe_from_db",
        lambda product_name: {
            "Bread Flour": 0.06,
            "Water": 0.04,
            "Salt": 0.0012,
        }
        if product_name == "baguette"
        else {},
    )

    result = scheduler_module.estimate_raw_materials({"baguette": 10}, {})

    assert result == {
        "Bread Flour": 0.6,
        "Water": 0.4,
        "Salt": 0.012,
    }
    assert "Baking Powder" not in result
    assert "Yeast" not in result


def test_material_dashboard_excludes_explicitly_untracked_recipe_materials(monkeypatch):
    from db import mysql_client

    class Rows:
        data = [
            {
                "material_name": "Bread Flour",
                "stock_quantity": 20.0,
                "unit": "kg",
                "track_inventory": True,
            },
            {
                "material_name": "Water",
                "stock_quantity": 0.0,
                "unit": "L",
                "track_inventory": False,
            },
        ]

    class Query:
        def select(self, columns):
            assert "track_inventory" in columns
            return self

        def execute(self):
            return Rows()

    class Database:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    database = Database()
    monkeypatch.setattr(mysql_client, "get_db", lambda: database)
    monkeypatch.setattr(mysql_client, "q", lambda *_args: Query())
    scheduler = make_scheduler()

    result = scheduler.dashboard_format_materials(
        {
            "week_start": "2026-07-15",
            "week_end": "2026-07-21",
            "materials": {
                "Bread Flour": 4.0,
                "Water": 15.0,
                "Unknown Ingredient": 2.0,
            },
        }
    )

    assert "Bread Flour" in result["items"]
    assert "Water" not in result["items"]
    assert result["items"]["Unknown Ingredient"]["alert"] == "urgent"
    assert database.closed


def test_solver_sells_day1_stock_before_fresh_bake(monkeypatch):
    monkeypatch.setattr(scheduler_module, "estimate_raw_materials", lambda *_: {})
    scheduler = make_scheduler()

    result = scheduler.solve(
        day1_stock={"croissant": 5},
        demand={"croissant": 5},
        capacity=10,
    )

    assert result["day1_sold"]["croissant"] == 5
    assert result["fresh_sold"]["croissant"] == 0
    assert result["waste"]["croissant"] == result["bake_plan"]["croissant"]


def test_rolling_plan_carries_unsold_fresh_but_not_expired_day1_stock():
    scheduler = make_scheduler()
    observed_stock = []

    def fake_solve_scenarios(stock, *_):
        observed_stock.append(dict(stock))
        return {
            "bake_plan": {"croissant": 5},
            "fresh_sold": {"croissant": 1},
            "day1_sold": {"croissant": 1},
            "waste": {"croissant": 2},
            "shortage": {"croissant": 0},
            "profit": 10.0,
            "revenue": 20.0,
            "total_bake": 5,
            "status": "OPTIMAL",
            "capacity_used_pct": 0.6,
            "materials": {},
            "scenario_q10": {"profit": 0, "waste_units": 0, "shortage_units": 0},
            "scenario_q50": {"profit": 0, "waste_units": 0, "shortage_units": 0},
            "scenario_q90": {"profit": 0, "waste_units": 0, "shortage_units": 0},
        }

    scheduler.solve_scenarios = fake_solve_scenarios
    forecast = {
        "2026-07-01": {"croissant": {"q10": 1, "q50": 2, "q90": 3}},
        "2026-07-02": {"croissant": {"q10": 1, "q50": 2, "q90": 3}},
    }

    result = scheduler.generate_7day_plan(
        "2026-07-01",
        {"croissant": 3},
        forecast,
    )

    assert observed_stock[0]["croissant"] == 3
    assert observed_stock[1]["croissant"] == 4
    assert result["weekly_summary"]["profit_definition"] == (
        "after_waste_and_shortage_risk_allowances"
    )


def test_day1_loader_reconstructs_selected_date_opening_stock(monkeypatch):
    from api import module3_scheduling

    class Cursor:
        def __init__(self):
            self.sql = ""
            self.params = ()
            self.closed = False

        def execute(self, sql, params):
            self.sql = " ".join(sql.split())
            self.params = params

        def fetchall(self):
            return [("croissant", 3), ("latte", 20)]

        def close(self):
            self.closed = True

    class Database:
        def __init__(self):
            self.cursor_instance = Cursor()
            self.closed = False

        def cursor(self):
            return self.cursor_instance

        def close(self):
            self.closed = True

    database = Database()
    monkeypatch.setattr(module3_scheduling, "get_db", lambda: database)

    result = module3_scheduling._load_day1_stock(
        {"croissant": {}, "baguette": {}},
        "2026-07-15",
    )

    assert result == {"croissant": 3, "baguette": 0}
    assert "JOIN products" in database.cursor_instance.sql
    assert "LEFT JOIN inventory_transactions" in database.cursor_instance.sql
    assert "p.category = 'bakery'" in database.cursor_instance.sql
    assert "it.transaction_time < %s" in database.cursor_instance.sql
    assert "it.transaction_type = 'inflow' THEN ABS(it.quantity)" in database.cursor_instance.sql
    assert "it.transaction_type = 'outflow' THEN -ABS(it.quantity)" in database.cursor_instance.sql
    assert database.cursor_instance.params == (
        "2026-07-15 00:00:00",
        "2026-07-15",
    )
    assert database.cursor_instance.closed
    assert database.closed


def test_7day_plan_starts_on_selected_date(monkeypatch):
    from api import module3_scheduling

    captured = {}

    class FakeScheduler:
        breads = {"croissant": {}}

        def generate_7day_plan(self, start_date, day1_stock, forecast):
            captured["start_date"] = start_date
            captured["day1_stock"] = day1_stock
            captured["forecast"] = forecast
            return {
                "dashboard_7day": {"dates": [start_date], "grid": []},
                "weekly_summary": {
                    "total_bake": 0,
                    "total_profit": 0.0,
                    "total_revenue": 0.0,
                    "daily_profits": [],
                    "scenarios": {},
                    "top_products": [],
                },
            }

    monkeypatch.setattr(scheduler_module, "Scheduler", FakeScheduler)
    monkeypatch.setattr(
        scheduler_module,
        "generate_7day_s2_forecast",
        lambda start_date: {start_date: {}},
    )
    monkeypatch.setattr(
        module3_scheduling,
        "_load_day1_stock",
        lambda breads, start_date: {"croissant": 3},
        raising=False,
    )

    result = asyncio.run(
        module3_scheduling.get_7day_production_plan(date="2026-07-15")
    )

    assert captured["start_date"] == "2026-07-15"
    assert captured["day1_stock"] == {"croissant": 3}
    assert result["dashboard_7day"]["dates"] == ["2026-07-15"]
    assert result["weekly_summary"]["day1_stock_total"] == 3
