"""
KPI Shift Schedule — Weekly Roster
====================================
7-day shift grid with role-based scheduling.

Integrates with: [[dashboard-designs]] — Shift+KPI Dashboard, Panel 2

Shift types:
  OPEN  (06:00-14:00)  — baker + barista
  MID   (10:00-18:00)  — barista + cashier
  CLOSE (14:00-22:00)  — barista + cashier
  FULL  (08:00-18:00)  — manager
"""

from datetime import datetime, timedelta
from collections import defaultdict


SHIFT_TYPES = {
    "OPEN":  {"start": "06:00", "end": "14:00", "roles": ["baker", "barista"]},
    "MID":   {"start": "10:00", "end": "18:00", "roles": ["barista", "cashier"]},
    "CLOSE": {"start": "14:00", "end": "22:00", "roles": ["barista", "cashier"]},
    "FULL":  {"start": "08:00", "end": "18:00", "roles": ["manager"]},
}

# Minimum staff per shift
MIN_STAFF = {
    "OPEN":  {"baker": 2, "barista": 1},
    "MID":   {"barista": 1, "cashier": 1},
    "CLOSE": {"barista": 1, "cashier": 1},
    "FULL":  {"manager": 1},
}


class ShiftScheduler:
    def __init__(self):
        self.employees = {}    # {id: {name, role, max_shifts_per_week}}
        self.schedule = {}     # {date: {shift_type: [emp_ids]}}
        self.day_off = {}      # {emp_id: [dates]}

    def add_employee(self, emp_id, name, role, max_shifts=5):
        self.employees[emp_id] = {
            "name": name, "role": role, "max_shifts": max_shifts
        }

    def auto_schedule(self, start_date=None):
        """
        Auto-generate a 7-day shift schedule ensuring minimum coverage.
        Simple round-robin with role constraints.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=datetime.now().weekday())
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        self.schedule = {}
        emp_shift_count = defaultdict(int)

        for day_offset in range(7):
            date = (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            self.schedule[date] = {}

            for shift_type, shift_info in SHIFT_TYPES.items():
                assigned = []
                for role, min_count in MIN_STAFF.get(shift_type, {}).items():
                    # Find available employees for this role
                    candidates = [
                        eid for eid, einfo in self.employees.items()
                        if einfo["role"] == role
                        and emp_shift_count[eid] < einfo["max_shifts"]
                        and date not in self.day_off.get(eid, [])
                    ]
                    # Sort by current shift count (balance workload)
                    candidates.sort(key=lambda eid: emp_shift_count[eid])
                    for eid in candidates[:min_count]:
                        assigned.append(eid)
                        emp_shift_count[eid] += 1

                if assigned:
                    self.schedule[date][shift_type] = assigned

        return self.schedule

    def set_day_off(self, emp_id, date_str):
        """Mark a day off for an employee."""
        if emp_id not in self.day_off:
            self.day_off[emp_id] = []
        if date_str not in self.day_off[emp_id]:
            self.day_off[emp_id].append(date_str)

    def get_weekly_grid(self, start_date=None):
        """
        Return a 7-day shift grid for dashboard display.
        Rows = employees, Columns = days, Cells = shift_type or OFF.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=datetime.now().weekday())
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        if not self.schedule:
            self.auto_schedule(start_date)

        dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        grid = []

        for emp_id, einfo in self.employees.items():
            row = {
                "id": emp_id,
                "name": einfo["name"],
                "role": einfo["role"],
                "shifts": {},
            }
            for date_str in dates:
                shift_found = "OFF"
                for shift_type, assigned in self.schedule.get(date_str, {}).items():
                    if emp_id in assigned:
                        shift_found = shift_type
                        break
                if date_str in self.day_off.get(emp_id, []):
                    shift_found = "OFF"
                row["shifts"][date_str] = shift_found
            grid.append(row)

        return {
            "week_start": dates[0],
            "week_end": dates[-1],
            "dates": dates,
            "grid": grid,
            "coverage": self._check_coverage(dates),
        }

    def _check_coverage(self, dates):
        """Check if all shifts meet minimum staffing."""
        issues = []
        for date_str in dates:
            for shift_type, min_staff in MIN_STAFF.items():
                assigned = self.schedule.get(date_str, {}).get(shift_type, [])
                assigned_roles = defaultdict(int)
                for eid in assigned:
                    role = self.employees.get(eid, {}).get("role", "")
                    assigned_roles[role] += 1
                for role, min_count in min_staff.items():
                    if assigned_roles[role] < min_count:
                        issues.append({
                            "date": date_str,
                            "shift": shift_type,
                            "role": role,
                            "assigned": assigned_roles[role],
                            "required": min_count,
                        })
        return {"complete": len(issues) == 0, "issues": issues}

    def dashboard_format(self):
        """Output matching [[dashboard-designs]] Panel 2: Weekly Shift."""
        grid = self.get_weekly_grid()
        return {
            "weekly_shift": {
                "week_start": grid["week_start"],
                "week_end": grid["week_end"],
                "dates": grid["dates"],
                "employees": [
                    {
                        "name": e["name"],
                        "role": e["role"],
                        "schedule": e["shifts"],
                    }
                    for e in grid["grid"]
                ],
                "coverage_ok": grid["coverage"]["complete"],
                "coverage_issues": grid["coverage"]["issues"],
            }
        }
