"""
KPI Attendance System — Punch Card + Attendance Metrics
=========================================================
Real-time punch-in/out tracking with PIN verification.
Computes: attendance_rate, punctuality_rate, late_count, overtime_hours.

Integrates with: [[dashboard-designs]] — Shift+KPI Dashboard, Panel 1
"""

from datetime import datetime, timedelta
from collections import defaultdict
import json, os


class AttendanceSystem:
    """Employee punch-card system with attendance metrics."""

    def __init__(self, store_path=None):
        self.store_path = store_path
        self.records = []          # list of {id, name, date, punch_in, punch_out, status}
        self.employees = {}        # {id: {name, role, pin}}
        self._load()

    def register_employee(self, emp_id, name, role, pin):
        """Register an employee with PIN for punch card."""
        self.employees[emp_id] = {"name": name, "role": role, "pin": pin}

    def punch(self, emp_id, pin):
        """
        Record a punch event. If no open punch today, it's a punch-in.
        If already punched in, it's a punch-out.

        Returns: (success: bool, message: str, record: dict or None)
        """
        if emp_id not in self.employees:
            return False, f"Employee {emp_id} not found", None

        if self.employees[emp_id]["pin"] != pin:
            return False, "Incorrect PIN", None

        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H:%M")

        # Check if already punched in today (without punch-out)
        for r in self.records:
            if r["id"] == emp_id and r["date"] == today and r.get("punch_out") is None:
                # Punch out
                r["punch_out"] = now
                r["status"] = self._check_status(r)
                self._save()
                return True, f"Punched OUT at {now}", r

        # New punch-in
        record = {
            "id": emp_id,
            "name": self.employees[emp_id]["name"],
            "role": self.employees[emp_id]["role"],
            "date": today,
            "punch_in": now,
            "punch_out": None,
            "status": "present",
        }
        self.records.append(record)
        self._save()
        return True, f"Punched IN at {now}", record

    def _check_status(self, record, shift_start="08:00", late_threshold_min=15):
        """Determine attendance status based on punch times."""
        punch_in = datetime.strptime(record["punch_in"], "%H:%M")
        shift = datetime.strptime(shift_start, "%H:%M")
        late_by = (punch_in - shift).total_seconds() / 60

        if late_by <= 0:
            return "on_time"
        elif late_by <= late_threshold_min:
            return "late"
        elif record.get("punch_out") is None:
            return "present"  # still working
        else:
            return "absent" if late_by > 120 else "late"

    def get_today_attendance(self):
        """Get today's attendance status for all registered employees."""
        today = datetime.now().strftime("%Y-%m-%d")
        result = []

        for emp_id, emp in self.employees.items():
            today_records = [r for r in self.records 
                           if r["id"] == emp_id and r["date"] == today]
            
            if not today_records:
                status = "absent"
                punch_in = None
                punch_out = None
            else:
                latest = today_records[-1]
                status = latest.get("status", "present")
                punch_in = latest.get("punch_in")
                punch_out = latest.get("punch_out")

            result.append({
                "id": emp_id,
                "name": emp["name"],
                "role": emp["role"],
                "status": status,
                "punch_in": punch_in,
                "punch_out": punch_out,
            })

        return result

    def get_monthly_metrics(self, year=None, month=None):
        """
        Compute monthly attendance KPIs for all employees.

        Returns:
            dict: {emp_id: {attendance_rate, punctuality_rate, late_count, absent_days}}
        """
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month

        month_str = f"{year}-{month:02d}"
        month_records = [r for r in self.records if r["date"].startswith(month_str)]

        # Working days in month (exclude weekends, simplified)
        import calendar
        working_days = sum(
            1 for d in range(1, calendar.monthrange(year, month)[1] + 1)
            if datetime(year, month, d).weekday() < 6
        )

        metrics = {}
        for emp_id in self.employees:
            emp_records = [r for r in month_records if r["id"] == emp_id]
            days_present = len(set(r["date"] for r in emp_records if r.get("punch_in")))
            late_count = sum(1 for r in emp_records if r.get("status") == "late")
            on_time = sum(1 for r in emp_records if r.get("status") == "on_time")
            absent = working_days - days_present

            metrics[emp_id] = {
                "name": self.employees[emp_id]["name"],
                "role": self.employees[emp_id]["role"],
                "attendance_rate": round(days_present / working_days * 100, 1) if working_days > 0 else 0,
                "punctuality_rate": round(on_time / max(days_present, 1) * 100, 1),
                "late_count": late_count,
                "absent_days": max(0, absent),
                "working_days": working_days,
                "days_present": days_present,
            }

        return metrics

    def get_weekly_attendance(self):
        """Get attendance for the current week (Mon-Sun)."""
        today = datetime.now()
        monday = today - timedelta(days=today.weekday())
        week_dates = [(monday + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

        result = {}
        for emp_id, emp in self.employees.items():
            row = {"id": emp_id, "name": emp["name"], "role": emp["role"]}
            for d in week_dates:
                day_records = [r for r in self.records if r["id"] == emp_id and r["date"] == d]
                if day_records:
                    latest = day_records[-1]
                    row[d] = {
                        "status": latest.get("status", "present"),
                        "in": latest.get("punch_in"),
                        "out": latest.get("punch_out"),
                    }
                else:
                    row[d] = {"status": "no_record", "in": None, "out": None}
            result[emp_id] = row

        return {
            "week_start": week_dates[0],
            "week_end": week_dates[-1],
            "employees": result,
        }

    def dashboard_format(self):
        """Output matching [[dashboard-designs]] Panel 1: Today Attendance."""
        today_data = self.get_today_attendance()
        weekly_data = self.get_weekly_attendance()
        monthly = self.get_monthly_metrics()

        return {
            "today_attendance": {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total": len(today_data),
                "present": sum(1 for e in today_data if e["status"] in ("on_time", "late", "present")),
                "absent": sum(1 for e in today_data if e["status"] == "absent"),
                "late": sum(1 for e in today_data if e["status"] == "late"),
                "employees": today_data,
            },
            "weekly_shift": weekly_data,
            "monthly_summary": {
                emp_id: {
                    "name": m["name"],
                    "role": m["role"],
                    "attendance_rate": m["attendance_rate"],
                    "punctuality_rate": m["punctuality_rate"],
                    "absent_days": m["absent_days"],
                }
                for emp_id, m in monthly.items()
            },
        }

    def _save(self):
        if self.store_path:
            data = {"records": self.records, "employees": self.employees}
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            with open(self.store_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

    def _load(self):
        if self.store_path and os.path.exists(self.store_path):
            with open(self.store_path) as f:
                data = json.load(f)
            self.records = data.get("records", [])
            self.employees = data.get("employees", {})
