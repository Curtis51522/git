import asyncio
from collections import Counter
from datetime import date, datetime, timedelta

from api import module3_scheduling as s3
from api.operation_clock import operation_time_scope


def normal_week_forecast(start_date="2026-07-06"):
    dates = [
        "2026-07-06",
        "2026-07-07",
        "2026-07-08",
        "2026-07-09",
        "2026-07-10",
        "2026-07-11",
        "2026-07-12",
    ]
    return {
        day: {
            "total_units": 300,
            "baker_units": 300,
            "coffee_units": 180,
            "demand_level": "normal",
        }
        for day in dates
    }


class FakeCursor:
    def execute(self, *_args, **_kwargs):
        return None


class FakeDb:
    def cursor(self):
        return FakeCursor()


class FakeQuery:
    def __init__(self, rows=None):
        self.data = rows or []

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def update(self, *_args):
        return self

    def execute(self):
        return self


def test_default_employees_can_solve_standard_seven_day_week():
    result = s3.solve_shift_schedule(
        s3.DEFAULT_EMPLOYEES,
        "2026-07-06",
        7,
        demand_forecast=normal_week_forecast(),
    )

    assert result
    assert len(result) > 0


def test_forecast_aggregation_separates_bakery_and_beverage_products():
    forecasts = [
        {
            "forecast_date": "2026-06-29",
            "product_name": "croissant",
            "freshness_status": "Total",
            "predicted_demand": 12,
        },
        {
            "forecast_date": "2026-06-29",
            "product_name": "latte",
            "freshness_status": "Total",
            "predicted_demand": 7,
        },
    ]

    result = s3._aggregate_forecast_rows(
        forecasts,
        "2026-06-29",
        {"croissant": "bakery", "latte": "beverage"},
    )

    assert result == {
        "2026-06-29": {
            "total_units": 19,
            "baker_units": 12,
            "coffee_units": 7,
        }
    }


def test_high_demand_week_is_feasible_for_small_shop_staffing():
    forecasts = {
        (date(2026, 6, 29) + timedelta(days=offset)).isoformat(): {
            "total_units": 320,
            "baker_units": 220,
            "coffee_units": 100,
            "demand_level": "high",
        }
        for offset in range(7)
    }

    result = s3.solve_shift_schedule(
        s3.DEFAULT_EMPLOYEES,
        "2026-06-29",
        7,
        demand_forecast=forecasts,
    )

    assert result
    for day in forecasts:
        for time_slot in s3.TIME_SLOTS:
            role_counts = Counter(
                shift.role
                for shift in result
                if shift.date == day and shift.time_slot == time_slot
            )
            assert role_counts == Counter({"baker": 3, "cashier": 1, "barista": 1})


def test_open_schedule_dates_include_monday():
    start = datetime(2026, 6, 29)

    assert s3._open_schedule_dates(start, 7) == {
        (start + timedelta(days=offset)).date().isoformat()
        for offset in range(7)
    }


def test_single_day_solver_uses_exact_role_coverage_without_overstaffing():
    forecast = {
        "2026-06-24": {
            "total_units": 200,
            "baker_units": 170,
            "demand_level": "normal",
        }
    }

    result = s3.solve_shift_schedule(
        s3.DEFAULT_EMPLOYEES,
        "2026-06-24",
        1,
        demand_forecast=forecast,
    )

    assert len(result) == 8
    for time_slot in s3.TIME_SLOTS:
        role_counts = Counter(
            shift.role for shift in result if shift.time_slot == time_slot
        )
        assert role_counts == Counter({"baker": 2, "cashier": 1, "barista": 1})


def test_seed_for_start_date_is_stable_across_calls():
    assert s3._stable_seed("2026-07-06") == s3._stable_seed("2026-07-06")
    assert s3._stable_seed("2026-07-06") != s3._stable_seed("2026-07-07")


def test_solver_worker_uses_selected_operation_date(monkeypatch):
    selected = datetime(2026, 6, 25, 9, 0)

    def inspect_operation_clock(payload):
        return {
            "operation_date": s3.operation_now().date().isoformat(),
            "is_past": s3._is_past_date(payload["start_date"]),
        }

    monkeypatch.setattr(s3, "_solve_impl", inspect_operation_clock)

    with operation_time_scope(selected):
        result = asyncio.run(s3.solve_schedule({"start_date": "2026-06-25", "days": 1}))

    assert result == {"operation_date": "2026-06-25", "is_past": False}


def test_swap_fetches_shifts_before_skill_validation(monkeypatch):
    future_date = (date.today() + timedelta(days=1)).isoformat()
    employees = [
        s3.Employee(id="E001", name="Alice", role="baker"),
        s3.Employee(id="E002", name="Bob", role="baker"),
    ]
    rows = [
        {
            "id": 1,
            "schedule_date": future_date,
            "time_slot": "06:00-13:00",
            "employee_id": "E001",
            "employee_name": "Alice",
            "role": "baker",
        },
        {
            "id": 2,
            "schedule_date": future_date,
            "time_slot": "12:00-19:00",
            "employee_id": "E002",
            "employee_name": "Bob",
            "role": "baker",
        },
    ]

    monkeypatch.setattr(s3, "load_employees", lambda: employees)
    monkeypatch.setattr(s3, "get_db", lambda: FakeDb())
    monkeypatch.setattr(s3, "q", lambda *_args, **_kwargs: FakeQuery(rows))

    result = asyncio.run(s3.swap_employees({
        "date": future_date,
        "time_slot": "06:00-13:00",
        "from_employee_id": "E001",
        "to_employee_id": "E002",
        "to_date": future_date,
        "to_time_slot": "12:00-19:00",
    }))

    assert result["status"] == "ok"


def test_swap_rejects_past_schedule_date():
    past_date = (date.today() - timedelta(days=1)).isoformat()

    result = asyncio.run(s3.swap_employees({
        "date": past_date,
        "time_slot": "06:00-13:00",
        "from_employee_id": "E001",
        "to_employee_id": "E002",
        "to_date": past_date,
        "to_time_slot": "12:00-19:00",
    }))

    assert result == {
        "status": "error",
        "message": "Cannot modify past schedules. Select today or a future date.",
    }
