"""
KPI Attendance System ? MySQL-backed Punch Card + Attendance Metrics
====================================================================
Real-time punch-in/out tracking with PIN verification.
Computes schedule-based attendance, punctuality, shift completion, and work hours.
"""

import os
import sys
from datetime import date, datetime, time, timedelta

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from api.operation_clock import operation_now

SYSTEM_ATTENDANCE_START_DATE = date(2026, 6, 24)


def time_to_minutes(value):
    """Convert database and API time values to whole minutes."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, timedelta):
        return int(value.total_seconds() // 60)
    if isinstance(value, (datetime, time)):
        return value.hour * 60 + value.minute

    text = str(value).strip()
    parts = text.split(":")
    if len(parts) < 2:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
    except ValueError:
        return None
    if hours < 0 or minutes < 0 or minutes > 59:
        return None
    return hours * 60 + minutes


def format_time_hhmm(value):
    """Format a supported time value as zero-padded HH:MM."""
    minutes = time_to_minutes(value)
    if minutes is None:
        return None
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _schedule_date(row):
    value = row.get("date", row.get("schedule_date"))
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")


def build_schedule_windows(rows, target_date):
    """Merge an employee's scheduled slots into one daily working window."""
    grouped = {}
    for row in rows or []:
        if _schedule_date(row) != target_date:
            continue
        employee_id = str(row.get("employee_id") or "")
        parts = str(row.get("time_slot") or "").split("-")
        if not employee_id or len(parts) != 2:
            continue
        start = time_to_minutes(parts[0])
        end = time_to_minutes(parts[1])
        if start is None or end is None or end <= start:
            continue

        current = grouped.setdefault(
            employee_id,
            {
                "employee_id": employee_id,
                "employee_name": row.get("employee_name", ""),
                "role": row.get("role", ""),
                "start_minutes": start,
                "end_minutes": end,
            },
        )
        current["start_minutes"] = min(current["start_minutes"], start)
        current["end_minutes"] = max(current["end_minutes"], end)

    for window in grouped.values():
        window["shift_start"] = format_time_hhmm(window["start_minutes"])
        window["shift_end"] = format_time_hhmm(window["end_minutes"])
        window["time_slot"] = f'{window["shift_start"]}-{window["shift_end"]}'
    return grouped


def derive_attendance_status(record, window):
    """Derive status from both punch boundaries and the scheduled window."""
    punch_in = time_to_minutes(record.get("punch_in"))
    punch_out = time_to_minutes(record.get("punch_out"))
    if punch_in is None:
        return "absent"

    shift_start = window.get("start_minutes")
    shift_end = window.get("end_minutes")
    if shift_start is None:
        shift_start = time_to_minutes(window.get("shift_start"))
    if shift_end is None:
        shift_end = time_to_minutes(window.get("shift_end"))
    if shift_start is None or shift_end is None:
        return "present"

    late = punch_in > shift_start
    if punch_out is None:
        return "late" if late else "present"

    early_leave = punch_out < shift_end
    if late and early_leave:
        return "late_and_early_leave"
    if late:
        return "late"
    if early_leave:
        return "early_leave"
    return "on_time"


def _record_date(record):
    value = record.get("date")
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")


def resolve_month_period_end(year, month, records, now=None):
    """Resolve a stable month cutoff shared by attendance and KPI reports."""
    import calendar

    now = now or datetime.now()
    month_start = datetime(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    month_end = datetime(year, month, last_day, 23, 59, 59)

    latest_record_end = None
    record_dates = sorted(
        value for value in (_record_date(record) for record in records) if value
    )
    if record_dates:
        try:
            latest_record_end = datetime.strptime(
                record_dates[-1], "%Y-%m-%d"
            ) + timedelta(days=1, seconds=-1)
        except ValueError:
            latest_record_end = None

    requested_month = (year, month)
    current_month = (now.year, now.month)
    if requested_month < current_month:
        period_end = month_end
    elif requested_month == current_month:
        period_end = max(now, latest_record_end or month_start)
    else:
        period_end = latest_record_end or month_start
    return min(period_end, month_end)


def _percentage(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return min(100.0, round(numerator / denominator * 100, 1))


def calculate_period_metrics(employees, schedules, records, period_end):
    """Calculate schedule-based attendance metrics up to a precise cutoff."""
    employee_map = {row["id"]: row for row in employees}
    counters = {
        employee_id: {
            "scheduled_days": 0,
            "attended_days": 0,
            "on_time_arrivals": 0,
            "full_checkouts": 0,
            "late_count": 0,
            "early_leave_count": 0,
            "credited_minutes": 0,
        }
        for employee_id in employee_map
    }
    record_map = {
        (_record_date(record), record.get("emp_id")): record
        for record in records
    }
    schedule_dates = sorted({_schedule_date(row) for row in schedules if _schedule_date(row)})

    for schedule_date in schedule_dates:
        try:
            day_start = datetime.strptime(schedule_date, "%Y-%m-%d")
        except ValueError:
            continue
        if day_start.date() < SYSTEM_ATTENDANCE_START_DATE:
            continue
        windows = build_schedule_windows(schedules, schedule_date)
        for employee_id, window in windows.items():
            if employee_id not in counters:
                continue
            shift_start = day_start + timedelta(minutes=window["start_minutes"])
            shift_end = day_start + timedelta(minutes=window["end_minutes"])
            record = record_map.get((schedule_date, employee_id))
            punch_in = time_to_minutes(record.get("punch_in")) if record else None
            punch_out = time_to_minutes(record.get("punch_out")) if record else None
            completed_record = (
                day_start.date() == period_end.date()
                and punch_in is not None
                and punch_out is not None
            )
            if shift_end > period_end and not completed_record:
                continue

            values = counters[employee_id]
            values["scheduled_days"] += 1
            if punch_in is None:
                continue

            values["attended_days"] += 1
            if punch_in <= window["start_minutes"]:
                values["on_time_arrivals"] += 1
            else:
                values["late_count"] += 1

            if punch_out is None:
                continue
            if punch_out >= window["end_minutes"]:
                values["full_checkouts"] += 1
            else:
                values["early_leave_count"] += 1

            overlap_start = max(punch_in, window["start_minutes"])
            overlap_end = min(punch_out, window["end_minutes"])
            values["credited_minutes"] += max(0, overlap_end - overlap_start)

    metrics = {}
    for employee_id, values in counters.items():
        employee = employee_map[employee_id]
        scheduled_days = values["scheduled_days"]
        attended_days = values["attended_days"]
        punctuality = _percentage(values["on_time_arrivals"], attended_days)
        shift_completion = _percentage(values["full_checkouts"], attended_days)
        attendance_rate = _percentage(attended_days, scheduled_days)
        metrics[employee_id] = {
            "name": employee.get("name", ""),
            "role": employee.get("role", ""),
            "scheduled_days": scheduled_days,
            "attended_days": attended_days,
            "attendance_rate": attendance_rate,
            "attendance_display": (
                f"{attended_days} / {scheduled_days} ({attendance_rate:.1f}%)"
            ),
            "punctuality": punctuality,
            "punctuality_rate": punctuality,
            "shift_completion": shift_completion,
            "shift_completion_rate": shift_completion,
            "late_count": values["late_count"],
            "early_leave_count": values["early_leave_count"],
            "absent_days": max(0, scheduled_days - attended_days),
            "work_hours": round(values["credited_minutes"] / 60, 2),
        }
    return metrics


class AttendanceSystem:
    """Employee punch-card system with MySQL-backed attendance metrics."""

    def __init__(self, store_path=None):
        self.store_path = store_path  # kept for compatibility

    def _get_db(self):
        from db.mysql_client import get_db
        return get_db()

    def register_employee(self, emp_id, name, role, pin):
        """Register an employee with PIN (upsert)."""
        db = self._get_db()
        cur = db.cursor()
        cur.execute("""
            INSERT INTO employees (id, name, role, pin)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE name=VALUES(name), role=VALUES(role), pin=VALUES(pin)
        """, (emp_id, name, role, pin))
        db.commit()

    def punch(self, emp_id, pin):
        """
        Record a punch event. Toggles between punch-in and punch-out.
        Returns: (success: bool, message: str, record: dict or None)
        """
        db = self._get_db()
        cur = db.cursor(dictionary=True)

        # Verify employee
        cur.execute("SELECT id, name, role, pin FROM employees WHERE id=%s", (emp_id,))
        emp = cur.fetchone()
        if not emp:
            return False, f"Employee {emp_id} not found", None
        if emp["pin"] != pin:
            return False, "Incorrect PIN", None

        current_time = operation_now()
        today = current_time.strftime("%Y-%m-%d")
        now = current_time.strftime("%H:%M:%S")

        cur.execute(
            """
            SELECT schedule_date, time_slot, employee_id, employee_name, role
            FROM shift_schedule
            WHERE employee_id=%s AND schedule_date=%s
            ORDER BY time_slot
            """,
            (emp_id, today),
        )
        schedule_rows = cur.fetchall()
        window = build_schedule_windows(schedule_rows, today).get(emp_id)
        if not window:
            return False, "No scheduled shift for this employee today", None

        # Check existing record for today
        cur.execute("SELECT * FROM attendance_records WHERE emp_id=%s AND date=%s", (emp_id, today))
        record = cur.fetchone()

        if record and record.get("punch_in") and not record.get("punch_out"):
            # Punch out
            updated_record = dict(record)
            updated_record["punch_out"] = now
            status = derive_attendance_status(updated_record, window)
            cur.execute(
                "UPDATE attendance_records SET punch_out=%s, status=%s WHERE id=%s",
                (now, status, record["id"])
            )
            db.commit()
            record["punch_out"] = now
            record["status"] = status
            return True, f"Punched OUT at {now}", record
        elif record:
            # Already punched in and out today
            return False, "Already completed punch for today", record
        else:
            # New punch-in
            status = derive_attendance_status(
                {"punch_in": now, "punch_out": None},
                window,
            )
            cur.execute(
                "INSERT INTO attendance_records (emp_id, emp_name, emp_role, date, punch_in, status) VALUES (%s, %s, %s, %s, %s, %s)",
                (emp_id, emp["name"], emp["role"], today, now, status)
            )
            db.commit()
            new_record = {
                "id": cur.lastrowid, "emp_id": emp_id, "emp_name": emp["name"],
                "emp_role": emp["role"], "date": today, "punch_in": now,
                "punch_out": None, "status": status
            }
            return True, f"Punched IN at {now}", new_record

    def correct_punch(
        self,
        emp_id,
        attendance_date,
        punch_in,
        punch_out,
        reason,
        corrected_by,
    ):
        """Correct one scheduled attendance record and append an audit entry."""
        reason = str(reason or "").strip()
        if not reason:
            return False, "Correction reason is required", None
        if len(reason) > 255:
            return False, "Correction reason must not exceed 255 characters", None
        try:
            datetime.strptime(str(attendance_date), "%Y-%m-%d")
        except ValueError:
            return False, "Attendance date must use YYYY-MM-DD", None

        punch_in_minutes = time_to_minutes(punch_in)
        punch_out_minutes = time_to_minutes(punch_out)
        if punch_in_minutes is None or punch_out_minutes is None:
            return False, "Punch times must use HH:MM", None
        if punch_out_minutes <= punch_in_minutes:
            return False, "Punch-out must be later than punch-in", None

        normalized_in = f"{format_time_hhmm(punch_in)}:00"
        normalized_out = f"{format_time_hhmm(punch_out)}:00"
        db = self._get_db()
        cur = db.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT id, name, role FROM employees WHERE id=%s",
                (emp_id,),
            )
            employee = cur.fetchone()
            if not employee:
                return False, f"Employee {emp_id} not found", None

            cur.execute(
                """
                SELECT schedule_date, time_slot, employee_id, employee_name, role
                FROM shift_schedule
                WHERE employee_id=%s AND schedule_date=%s
                ORDER BY time_slot
                """,
                (emp_id, attendance_date),
            )
            schedule_rows = cur.fetchall()
            window = build_schedule_windows(schedule_rows, attendance_date).get(emp_id)
            if not window:
                return (
                    False,
                    f"No scheduled shift for this employee on {attendance_date}",
                    None,
                )

            status = derive_attendance_status(
                {"punch_in": normalized_in, "punch_out": normalized_out},
                window,
            )
            cur.execute(
                "SELECT * FROM attendance_records WHERE emp_id=%s AND date=%s",
                (emp_id, attendance_date),
            )
            previous = cur.fetchone() or {}
            cur.execute("START TRANSACTION")
            cur.execute(
                """
                INSERT INTO attendance_records
                    (emp_id, emp_name, emp_role, date, punch_in, punch_out, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    emp_name=VALUES(emp_name),
                    emp_role=VALUES(emp_role),
                    punch_in=VALUES(punch_in),
                    punch_out=VALUES(punch_out),
                    status=VALUES(status)
                """,
                (
                    emp_id,
                    employee["name"],
                    employee["role"],
                    attendance_date,
                    normalized_in,
                    normalized_out,
                    status,
                ),
            )
            cur.execute(
                """
                INSERT INTO attendance_correction_log
                    (emp_id, attendance_date, previous_punch_in,
                     previous_punch_out, previous_status, corrected_punch_in,
                     corrected_punch_out, corrected_status, reason, corrected_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    emp_id,
                    attendance_date,
                    previous.get("punch_in"),
                    previous.get("punch_out"),
                    previous.get("status"),
                    normalized_in,
                    normalized_out,
                    status,
                    reason,
                    str(corrected_by or "manager"),
                ),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            if hasattr(cur, "close"):
                cur.close()

        record = {
            "emp_id": emp_id,
            "emp_name": employee["name"],
            "emp_role": employee["role"],
            "date": attendance_date,
            "punch_in": format_time_hhmm(normalized_in),
            "punch_out": format_time_hhmm(normalized_out),
            "status": status,
            "corrected": True,
            "correction_reason": reason,
            "corrected_by": str(corrected_by or "manager"),
        }
        return (
            True,
            f"Attendance corrected for {emp_id} on {attendance_date}",
            record,
        )

    def get_today_attendance(self, schedule=None):
        """Return today's attendance for scheduled employees only."""
        now = operation_now()
        today = now.strftime("%Y-%m-%d")
        return self._get_attendance_for_date(today, schedule or [], now)

    def get_date_attendance(self, date_str, schedule=None):
        """Return attendance for employees scheduled on the selected date."""
        return self._get_attendance_for_date(date_str, schedule or [], operation_now())

    def _get_attendance_for_date(self, date_str, schedule, reference_time):
        db = self._get_db()
        cur = db.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT id, name, role FROM employees WHERE available=1 AND role != %s",
                ("manager",),
            )
            employees = {row["id"]: row for row in cur.fetchall()}
            cur.execute("SELECT * FROM attendance_records WHERE date=%s", (date_str,))
            records = {row["emp_id"]: row for row in cur.fetchall()}
            cur.execute(
                """
                SELECT emp_id, reason, corrected_by, created_at
                FROM attendance_correction_log
                WHERE attendance_date=%s
                ORDER BY id
                """,
                (date_str,),
            )
            corrections = {row["emp_id"]: row for row in cur.fetchall()}
        finally:
            if hasattr(cur, "close"):
                cur.close()

        windows = build_schedule_windows(schedule, date_str)
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return []

        result = []
        for employee_id in sorted(windows):
            window = windows[employee_id]
            employee = employees.get(employee_id, {})
            record = records.get(employee_id)
            correction = corrections.get(employee_id)
            if record:
                status = derive_attendance_status(record, window)
            else:
                shift_start = target_date + timedelta(minutes=window["start_minutes"])
                status = "scheduled" if reference_time < shift_start else "absent"

            result.append(
                {
                    "id": employee_id,
                    "name": employee.get("name") or window["employee_name"],
                    "role": employee.get("role") or window["role"],
                    "status": status,
                    "punch_in": format_time_hhmm(record.get("punch_in")) if record else None,
                    "punch_out": format_time_hhmm(record.get("punch_out")) if record else None,
                    "shift_start": window["shift_start"],
                    "shift_end": window["shift_end"],
                    "time_slot": window["time_slot"],
                    "corrected": bool(correction),
                    "correction_reason": correction.get("reason") if correction else None,
                    "corrected_by": correction.get("corrected_by") if correction else None,
                }
            )
        return result

    def get_monthly_metrics(self, year=None, month=None, period_end=None):
        """Return schedule-based metrics for a month up to a precise cutoff."""
        now = datetime.now()
        if year is None:
            year = now.year
        if month is None:
            month = now.month
        month_str = f"{year}-{month:02d}"

        db = self._get_db()
        cur = db.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT id, name, role FROM employees "
                "WHERE available=1 AND role != %s",
                ("manager",),
            )
            employees = cur.fetchall()
            cur.execute(
                "SELECT * FROM attendance_records WHERE date LIKE %s",
                (month_str + "%",),
            )
            records = cur.fetchall()
            cur.execute(
                """
                SELECT schedule_date, time_slot, employee_id, employee_name, role
                FROM shift_schedule
                WHERE schedule_date LIKE %s
                ORDER BY schedule_date, time_slot, employee_id
                """,
                (month_str + "%",),
            )
            schedules = cur.fetchall()
        finally:
            if hasattr(cur, "close"):
                cur.close()

        if period_end is None:
            period_end = resolve_month_period_end(year, month, records, now=now)

        return calculate_period_metrics(
            employees,
            schedules,
            records,
            period_end,
        )
