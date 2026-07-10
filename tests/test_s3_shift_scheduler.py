import asyncio
from datetime import date, timedelta

from api import module3_scheduling as s3


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


def test_seed_for_start_date_is_stable_across_calls():
    assert s3._stable_seed("2026-07-06") == s3._stable_seed("2026-07-06")
    assert s3._stable_seed("2026-07-06") != s3._stable_seed("2026-07-07")


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
