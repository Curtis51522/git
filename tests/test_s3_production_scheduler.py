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

    scheduler.generate_7day_plan(
        "2026-07-01",
        {"croissant": 3},
        forecast,
    )

    assert observed_stock[0]["croissant"] == 3
    assert observed_stock[1]["croissant"] == 4
