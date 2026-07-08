import asyncio
from datetime import date, timedelta

from api import module3_scheduling as s3
from kpi.calculator import KPICalculator
from kpi.collector import KPIDataCollector
from kpi import config as kpi_config


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, *_args, **_kwargs):
        return None

    def fetchall(self):
        return self.rows

    def close(self):
        return None


class FakeDb:
    def __init__(self, result_sets):
        self.result_sets = list(result_sets)

    def cursor(self, **_kwargs):
        return FakeCursor(self.result_sets.pop(0))


class FakeAttendance:
    def punch(self, *_args, **_kwargs):
        return True, "ok", {"id": 1}


def test_collect_attendance_uses_real_punch_hours_and_caps_attendance_rate():
    collector = KPIDataCollector()
    records = [
        {
            "emp_id": "E001",
            "status": "on_time",
            "date": date(2026, 6, 1),
            "punch_in": timedelta(hours=6),
            "punch_out": timedelta(hours=14),
        },
        {
            "emp_id": "E001",
            "status": "late",
            "date": date(2026, 6, 2),
            "punch_in": timedelta(hours=6, minutes=30),
            "punch_out": timedelta(hours=13),
        },
    ]
    scheduled_days = [{"employee_id": "E001", "sched_days": 1}]
    collector._get_db = lambda: FakeDb([records, scheduled_days])

    result = collector._collect_attendance(
        [{"id": "E001", "name": "Alice", "role": "baker"}],
        "2026-06",
    )

    assert result["E001"]["work_hours"] == 14.5
    assert result["E001"]["attendance_rate"] == 100.0


def test_successful_attendance_punch_invalidates_kpi_cache(monkeypatch):
    s3._kpi_cache = {"cache_key": "2026-06", "data": {"stale": True}}
    monkeypatch.setattr(s3, "_get_attendance", lambda: FakeAttendance())

    result = asyncio.run(s3.punch_attendance(emp_id="E001", pin="1234"))

    assert result["status"] == "ok"
    assert s3._kpi_cache is None


def test_kpi_config_documents_actual_revenue_metric_name():
    doc = kpi_config.__doc__ or ""

    assert "revenue_contribution" in doc
    assert "revenue_hr" not in doc


def test_kpi_robust_z_scores_are_clipped_for_small_sample_outliers():
    employees = [
        {"id": "E001", "name": "A", "role": "baker", "kpis": {"attendance_rate": 110}},
        {"id": "E002", "name": "B", "role": "baker", "kpis": {"attendance_rate": 95}},
        {"id": "E003", "name": "C", "role": "baker", "kpis": {"attendance_rate": 95}},
        {"id": "E004", "name": "D", "role": "baker", "kpis": {"attendance_rate": 96}},
        {"id": "E005", "name": "E", "role": "baker", "kpis": {"attendance_rate": 96}},
    ]

    result = KPICalculator().normalize_within_role(employees)

    assert result[0]["z_scores"]["attendance_rate"] == 5.0
