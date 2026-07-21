import asyncio
from datetime import datetime, timedelta

import pytest

from api import module3_scheduling as s3
from kpi.attendance import (
    AttendanceSystem,
    build_schedule_windows,
    calculate_period_metrics,
    derive_attendance_status,
    format_time_hhmm,
)


def test_build_schedule_windows_merges_two_daily_slots():
    rows = [
        {
            "date": "2026-07-15",
            "employee_id": "E009",
            "employee_name": "Lin Yue",
            "role": "cashier",
            "time_slot": "06:00-13:00",
        },
        {
            "date": "2026-07-15",
            "employee_id": "E009",
            "employee_name": "Lin Yue",
            "role": "cashier",
            "time_slot": "12:00-19:00",
        },
    ]

    windows = build_schedule_windows(rows, "2026-07-15")

    assert windows["E009"]["shift_start"] == "06:00"
    assert windows["E009"]["shift_end"] == "19:00"
    assert windows["E009"]["time_slot"] == "06:00-19:00"


def test_format_time_hhmm_zero_pads_mysql_timedelta():
    assert format_time_hhmm(timedelta(hours=6, minutes=4)) == "06:04"


@pytest.mark.parametrize(
    ("punch_in", "punch_out", "expected"),
    [
        ("05:55", "19:03", "on_time"),
        ("06:05", "19:03", "late"),
        ("05:55", "18:50", "early_leave"),
        ("06:05", "18:50", "late_and_early_leave"),
    ],
)
def test_derive_attendance_status_uses_both_shift_boundaries(
    punch_in, punch_out, expected
):
    window = {
        "shift_start": "06:00",
        "shift_end": "19:00",
        "start_minutes": 360,
        "end_minutes": 1140,
    }
    record = {"punch_in": punch_in, "punch_out": punch_out}

    assert derive_attendance_status(record, window) == expected


class ScriptedCursor:
    def __init__(self, database):
        self.database = database
        self.rows = []
        self.lastrowid = 99

    def execute(self, sql, _params=None):
        normalized = " ".join(sql.split()).lower()
        self.database.statements.append(normalized)
        if "from employees" in normalized:
            self.rows = list(self.database.employees)
        elif "from shift_schedule" in normalized:
            self.rows = list(self.database.schedules)
        elif "from attendance_records" in normalized:
            self.rows = list(self.database.attendance)
        else:
            self.rows = []

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class ScriptedDatabase:
    def __init__(self, employees=None, schedules=None, attendance=None):
        self.employees = employees or []
        self.schedules = schedules or []
        self.attendance = attendance or []
        self.statements = []
        self.commits = 0

    def cursor(self, **_kwargs):
        return ScriptedCursor(self)

    def commit(self):
        self.commits += 1


class CorrectionCursor:
    def __init__(self, database):
        self.database = database
        self.rows = []
        self.lastrowid = 41

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split()).lower()
        self.database.statements.append((normalized, params))
        if "from employees" in normalized:
            self.rows = [self.database.employee]
        elif "from shift_schedule" in normalized:
            self.rows = list(self.database.schedules)
        elif "select * from attendance_records" in normalized:
            self.rows = list(self.database.attendance)
        elif normalized.startswith("insert into attendance_correction_log"):
            self.database.audit_params = params
            self.rows = []
        else:
            self.rows = []

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        return None


class CorrectionDatabase:
    def __init__(self):
        self.employee = {"id": "E001", "name": "Zhang Wei", "role": "baker"}
        self.schedules = [
            {
                "schedule_date": "2026-07-15",
                "time_slot": "06:00-13:00",
                "employee_id": "E001",
                "employee_name": "Zhang Wei",
                "role": "baker",
            }
        ]
        self.attendance = []
        self.statements = []
        self.audit_params = None
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, **_kwargs):
        return CorrectionCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_historical_attendance_contains_scheduled_employees_only():
    database = ScriptedDatabase(
        employees=[
            {"id": "E001", "name": "Zhang Wei", "role": "baker"},
            {"id": "E002", "name": "Li Na", "role": "barista"},
            {"id": "E003", "name": "Wang Lei", "role": "cashier"},
        ],
        attendance=[
            {
                "emp_id": "E001",
                "punch_in": "05:55:00",
                "punch_out": "13:03:00",
                "status": "on_time",
            },
            {
                "emp_id": "E003",
                "punch_in": "05:58:00",
                "punch_out": "13:04:00",
                "status": "on_time",
            },
        ],
    )
    schedule = [
        {
            "date": "2026-07-15",
            "employee_id": "E001",
            "employee_name": "Zhang Wei",
            "role": "baker",
            "time_slot": "06:00-13:00",
        },
        {
            "date": "2026-07-15",
            "employee_id": "E002",
            "employee_name": "Li Na",
            "role": "barista",
            "time_slot": "12:00-19:00",
        },
    ]
    system = AttendanceSystem()
    system._get_db = lambda: database

    result = system.get_date_attendance("2026-07-15", schedule)

    assert [row["id"] for row in result] == ["E001", "E002"]
    assert result[0]["status"] == "on_time"
    assert result[0]["punch_in"] == "05:55"
    assert result[1]["status"] == "absent"


def test_punch_rejects_employee_without_today_schedule():
    database = ScriptedDatabase(
        employees=[
            {
                "id": "E003",
                "name": "Wang Lei",
                "role": "cashier",
                "pin": "1234",
            }
        ]
    )
    system = AttendanceSystem()
    system._get_db = lambda: database

    success, message, record = system.punch("E003", "1234")

    assert success is False
    assert message == "No scheduled shift for this employee today"
    assert record is None
    assert database.commits == 0


def test_manager_correction_requires_a_reason():
    system = AttendanceSystem()
    system._get_db = lambda: CorrectionDatabase()

    success, message, record = system.correct_punch(
        "E001",
        "2026-07-15",
        "05:58",
        "13:04",
        "",
        "manager",
    )

    assert success is False
    assert message == "Correction reason is required"
    assert record is None


def test_manager_correction_derives_status_and_writes_audit_record():
    database = CorrectionDatabase()
    system = AttendanceSystem()
    system._get_db = lambda: database

    success, message, record = system.correct_punch(
        "E001",
        "2026-07-15",
        "06:05",
        "13:04",
        "Forgot to punch in",
        "manager",
    )

    assert success is True
    assert message == "Attendance corrected for E001 on 2026-07-15"
    assert record["status"] == "late"
    assert record["corrected"] is True
    assert record["correction_reason"] == "Forgot to punch in"
    assert database.commits == 1
    assert database.rollbacks == 0
    assert database.audit_params is not None
    assert database.audit_params[7] == "late"
    assert database.audit_params[8] == "Forgot to punch in"
    assert database.audit_params[9] == "manager"


def test_manager_correction_rejects_employee_without_selected_date_schedule():
    database = CorrectionDatabase()
    database.schedules = []
    system = AttendanceSystem()
    system._get_db = lambda: database

    success, message, record = system.correct_punch(
        "E001",
        "2026-07-15",
        "05:58",
        "13:04",
        "Device issue",
        "manager",
    )

    assert success is False
    assert message == "No scheduled shift for this employee on 2026-07-15"
    assert record is None


def test_period_metrics_use_scheduled_days_and_both_punch_boundaries():
    employees = [{"id": "E001", "name": "Zhang Wei", "role": "baker"}]
    schedules = [
        {
            "schedule_date": day,
            "employee_id": "E001",
            "employee_name": "Zhang Wei",
            "role": "baker",
            "time_slot": "06:00-13:00",
        }
        for day in ("2026-07-01", "2026-07-02", "2026-07-03")
    ]
    records = [
        {
            "emp_id": "E001",
            "date": "2026-07-01",
            "punch_in": "05:55:00",
            "punch_out": "13:05:00",
        },
        {
            "emp_id": "E001",
            "date": "2026-07-02",
            "punch_in": "06:05:00",
            "punch_out": "13:05:00",
        },
        {
            "emp_id": "E001",
            "date": "2026-07-03",
            "punch_in": "05:55:00",
            "punch_out": "12:50:00",
        },
    ]

    metrics = calculate_period_metrics(
        employees,
        schedules,
        records,
        datetime(2026, 7, 3, 23, 59),
    )["E001"]

    assert metrics["scheduled_days"] == 3
    assert metrics["attended_days"] == 3
    assert metrics["attendance_rate"] == 100.0
    assert metrics["punctuality"] == 66.7
    assert metrics["shift_completion"] == 66.7
    assert metrics["late_count"] == 1
    assert metrics["early_leave_count"] == 1
    assert metrics["absent_days"] == 0
    assert metrics["work_hours"] == 20.75


def test_period_metrics_ignore_days_before_system_attendance_start():
    employees = [{"id": "E001", "name": "Zhang Wei", "role": "baker"}]
    schedules = [
        {
            "schedule_date": day,
            "employee_id": "E001",
            "employee_name": "Zhang Wei",
            "role": "baker",
            "time_slot": "06:00-13:00",
        }
        for day in ("2026-06-23", "2026-06-24")
    ]
    records = [
        {
            "emp_id": "E001",
            "date": day,
            "punch_in": "05:55:00",
            "punch_out": "13:05:00",
        }
        for day in ("2026-06-23", "2026-06-24")
    ]

    metrics = calculate_period_metrics(
        employees,
        schedules,
        records,
        datetime(2026, 6, 24, 23, 59),
    )["E001"]

    assert metrics["scheduled_days"] == 1
    assert metrics["attended_days"] == 1
    assert metrics["attendance_display"] == "1 / 1 (100.0%)"


def test_period_metrics_exclude_unfinished_future_shift():
    employees = [{"id": "E001", "name": "Zhang Wei", "role": "baker"}]
    schedules = [
        {
            "schedule_date": "2026-07-04",
            "employee_id": "E001",
            "employee_name": "Zhang Wei",
            "role": "baker",
            "time_slot": "12:00-19:00",
        }
    ]

    metrics = calculate_period_metrics(
        employees,
        schedules,
        [],
        datetime(2026, 7, 4, 18, 59),
    )["E001"]

    assert metrics["scheduled_days"] == 0
    assert metrics["attendance_rate"] == 0.0


def test_period_metrics_include_completed_record_after_clock_reset():
    employees = [{"id": "E001", "name": "Zhang Wei", "role": "baker"}]
    schedules = [
        {
            "schedule_date": "2026-06-24",
            "employee_id": "E001",
            "employee_name": "Zhang Wei",
            "role": "baker",
            "time_slot": "06:00-13:00",
        },
        {
            "schedule_date": "2026-06-25",
            "employee_id": "E001",
            "employee_name": "Zhang Wei",
            "role": "baker",
            "time_slot": "06:00-13:00",
        },
    ]
    records = [
        {
            "emp_id": "E001",
            "date": "2026-06-24",
            "punch_in": "05:55:00",
            "punch_out": "13:03:00",
        },
        {
            "emp_id": "E001",
            "date": "2026-06-25",
            "punch_in": "05:55:00",
            "punch_out": "13:03:00",
        },
    ]

    metrics = calculate_period_metrics(
        employees,
        schedules,
        records,
        datetime(2026, 6, 24, 5, 0),
    )["E001"]

    assert metrics["scheduled_days"] == 1
    assert metrics["attended_days"] == 1
    assert metrics["attendance_display"] == "1 / 1 (100.0%)"


def test_period_metrics_credit_only_overlap_with_scheduled_window():
    employees = [{"id": "E001", "name": "Zhang Wei", "role": "baker"}]
    schedules = [
        {
            "schedule_date": "2026-07-05",
            "employee_id": "E001",
            "employee_name": "Zhang Wei",
            "role": "baker",
            "time_slot": "06:00-19:00",
        }
    ]
    records = [
        {
            "emp_id": "E001",
            "date": "2026-07-05",
            "punch_in": "05:30:00",
            "punch_out": "19:30:00",
        }
    ]

    metrics = calculate_period_metrics(
        employees,
        schedules,
        records,
        datetime(2026, 7, 5, 23, 59),
    )["E001"]

    assert metrics["work_hours"] == 13.0


class FakeApiAttendance:
    def get_date_attendance(self, _date, _schedule):
        return [
            {
                "id": "E001",
                "name": "Zhang Wei",
                "role": "baker",
                "status": "late_and_early_leave",
                "punch_in": "06:05",
                "punch_out": "12:50",
                "time_slot": "06:00-13:00",
            },
            {
                "id": "E002",
                "name": "Li Na",
                "role": "barista",
                "status": "on_time",
                "punch_in": "11:55",
                "punch_out": "19:05",
                "time_slot": "12:00-19:00",
            },
        ]

    def get_monthly_metrics(self, _year, _month, period_end=None):
        assert period_end == datetime(2026, 7, 15, 23, 59, 59)
        return {
            "E001": {
                "attendance_rate": 85.7,
                "attendance_display": "12 / 14 (85.7%)",
                "punctuality_rate": 83.3,
                "shift_completion_rate": 91.7,
                "late_count": 2,
                "early_leave_count": 1,
            },
            "E002": {
                "attendance_rate": 100.0,
                "attendance_display": "14 / 14 (100.0%)",
                "punctuality_rate": 100.0,
                "shift_completion_rate": 100.0,
                "late_count": 0,
                "early_leave_count": 0,
            },
        }


def test_attendance_history_returns_period_metrics_and_early_leave_summary(
    monkeypatch,
):
    async def fake_schedule(date, days):
        assert date == "2026-07-15"
        assert days == 1
        return {"schedule": [{"date": date}]}

    monkeypatch.setattr(s3, "_get_attendance", lambda: FakeApiAttendance())
    monkeypatch.setattr(s3, "get_schedule", fake_schedule)

    response = asyncio.run(s3.get_attendance_history("2026-07-15"))

    assert response["present"] == 2
    assert response["late"] == 1
    assert response["early_leave"] == 1
    assert response["absent"] == 0
    assert response["employees"][0]["attendance_rate"] == 85.7
    assert response["employees"][0]["attendance_display"] == "12 / 14 (85.7%)"
    assert response["employees"][0]["shift_completion_rate"] == 91.7


def test_today_attendance_uses_operation_time_as_metric_cutoff(monkeypatch):
    selected_time = datetime(2026, 6, 24, 19, 5)

    class FakeTodayAttendance:
        def get_today_attendance(self, _schedule):
            return [
                {
                    "id": "E001",
                    "name": "Zhang Wei",
                    "role": "baker",
                    "status": "on_time",
                }
            ]

        def get_monthly_metrics(self, year, month, period_end=None):
            assert (year, month) == (2026, 6)
            assert period_end == selected_time
            return {
                "E001": {
                    "scheduled_days": 1,
                    "attended_days": 1,
                    "attendance_rate": 100.0,
                    "attendance_display": "1 / 1 (100.0%)",
                }
            }

    async def fake_schedule(date, days):
        assert date == "2026-06-24"
        assert days == 1
        return {"schedule": [{"date": date}]}

    monkeypatch.setattr(s3, "operation_now", lambda: selected_time)
    monkeypatch.setattr(s3, "_get_attendance", lambda: FakeTodayAttendance())
    monkeypatch.setattr(s3, "get_schedule", fake_schedule)

    response = asyncio.run(s3.get_attendance_dashboard())

    employee = response["today_attendance"]["employees"][0]
    assert employee["attendance_display"] == "1 / 1 (100.0%)"
