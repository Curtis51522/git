"""
KPI Attendance System ? MySQL-backed Punch Card + Attendance Metrics
====================================================================
Real-time punch-in/out tracking with PIN verification.
Computes: attendance_rate, punctuality_rate, late_count, overtime_hours.
"""

from datetime import datetime, timedelta
from collections import defaultdict
import json, os, sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


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

        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H:%M:%S")

        # Check existing record for today
        cur.execute("SELECT * FROM attendance_records WHERE emp_id=%s AND date=%s", (emp_id, today))
        record = cur.fetchone()

        if record and record.get("punch_in") and not record.get("punch_out"):
            # Punch out
            cur.execute(
                "UPDATE attendance_records SET punch_out=%s, status=%s WHERE id=%s",
                (now, "on_time" if record.get("status") != "late" else record["status"], record["id"])
            )
            db.commit()
            record["punch_out"] = now
            return True, f"Punched OUT at {now}", record
        elif record:
            # Already punched in and out today
            return False, "Already completed punch for today", record
        else:
            # New punch-in
            cur.execute(
                "INSERT INTO attendance_records (emp_id, emp_name, emp_role, date, punch_in, status) VALUES (%s, %s, %s, %s, %s, %s)",
                (emp_id, emp["name"], emp["role"], today, now, "present")
            )
            db.commit()
            new_record = {
                "id": cur.lastrowid, "emp_id": emp_id, "emp_name": emp["name"],
                "emp_role": emp["role"], "date": today, "punch_in": now,
                "punch_out": None, "status": "present"
            }
            return True, f"Punched IN at {now}", new_record

    def get_today_attendance(self, schedule=None):
        """Get today's attendance status for all employees.
        
        Args:
            schedule: optional list of dicts from S3 schedule API
                      [{employee_id, date, time_slot, ...}]
                      If provided, status is checked against shift start time.
        """
        db = self._get_db()
        cur = db.cursor(dictionary=True)
        today = datetime.now().strftime("%Y-%m-%d")

        cur.execute("SELECT id, name, role FROM employees WHERE available=1 AND role != %s", ("manager",))
        employees = cur.fetchall()

        cur.execute("SELECT * FROM attendance_records WHERE date=%s", (today,))
        records = {r["emp_id"]: r for r in cur.fetchall()}

        # Build schedule map: emp_id -> {shift_start, shift_end, time_slot}
        sched_map = {}
        if schedule:
            for s in schedule:
                eid = s.get("employee_id", "")
                if eid and s.get("date") == today:
                    slot = s.get("time_slot", "06:00-13:00")
                    parts = slot.split("-") if "-" in slot else ["06:00", "13:00"]
                    shift_start = parts[0]
                    shift_end = parts[1] if len(parts) > 1 else "13:00"
                    sched_map[eid] = {
                        "shift_start": shift_start,
                        "shift_end": shift_end,
                        "time_slot": slot,
                    }
        result = []
        for emp in employees:
            eid = emp["id"]
            rec = records.get(eid)
            sched_info = sched_map.get(eid)

            if rec:
                status = self._check_status_with_schedule(rec, sched_info["shift_start"] if sched_info else None)
                result.append({
                    "id": eid, "name": emp["name"], "role": emp["role"],
                    "status": status,
                    "punch_in": str(rec.get("punch_in", ""))[:5] if rec.get("punch_in") else None,
                    "punch_out": str(rec.get("punch_out", ""))[:5] if rec.get("punch_out") else None,
                    "shift_start": sched_info["shift_start"] if sched_info else None,
                    "shift_end": sched_info["shift_end"] if sched_info else None,
                    "time_slot": sched_info["time_slot"] if sched_info else None,
                })
            else:
                # No punch record
                if sched_info:
                    # Scheduled but no punch = absent
                    result.append({
                        "id": eid, "name": emp["name"], "role": emp["role"],
                        "status": "absent", "punch_in": None, "punch_out": None,
                        "shift_start": sched_info["shift_start"],
                        "shift_end": sched_info["shift_end"],
                        "time_slot": sched_info["time_slot"],
                    })
                else:
                    # Not scheduled and no punch = off_day
                    result.append({
                        "id": eid, "name": emp["name"], "role": emp["role"],
                        "status": "off_day", "punch_in": None, "punch_out": None,
                        "shift_start": None, "shift_end": None, "time_slot": None,
                    })
        return result

    def get_date_attendance(self, date_str, schedule=None):
        db = self._get_db()
        cur = db.cursor(dictionary=True)
        cur.execute('SELECT id, name, role FROM employees WHERE available=1 AND role != %s', ('manager',))
        employees = cur.fetchall()
        cur.execute('SELECT * FROM attendance_records WHERE date=%s', (date_str,))
        records = {r['emp_id']: r for r in cur.fetchall()}
        sched_map = {}
        if schedule:
            for s in schedule:
                eid = s.get('employee_id', '')
                sd = str(s.get('date', ''))
                if eid and sd == date_str:
                    slot = s.get('time_slot', '06:00-13:00')
                    parts = slot.split('-') if '-' in slot else ['06:00', '13:00']
                    sched_map[eid] = {
                        'shift_start': parts[0],
                        'shift_end': parts[1] if len(parts) > 1 else '13:00',
                        'time_slot': slot,
                    }
        result = []
        for emp in employees:
            eid = emp['id']
            rec = records.get(eid)
            sched_info = sched_map.get(eid)
            if rec:
                status = self._check_status_with_schedule(rec, sched_info['shift_start'] if sched_info else None)
                result.append({
                    'id': eid, 'name': emp['name'], 'role': emp['role'],
                    'status': status,
                    'punch_in': str(rec.get('punch_in', ''))[:5] if rec.get('punch_in') else None,
                    'punch_out': str(rec.get('punch_out', ''))[:5] if rec.get('punch_out') else None,
                    'shift_start': sched_info['shift_start'] if sched_info else None,
                    'shift_end': sched_info['shift_end'] if sched_info else None,
                    'time_slot': sched_info['time_slot'] if sched_info else None,
                })
            else:
                if sched_info:
                    result.append({
                        'id': eid, 'name': emp['name'], 'role': emp['role'],
                        'status': 'absent', 'punch_in': None, 'punch_out': None,
                        'shift_start': sched_info['shift_start'],
                        'shift_end': sched_info['shift_end'],
                        'time_slot': sched_info['time_slot'],
                    })
                else:
                    result.append({
                        'id': eid, 'name': emp['name'], 'role': emp['role'],
                        'status': 'off_day', 'punch_in': None, 'punch_out': None,
                        'shift_start': None, 'shift_end': None, 'time_slot': None,
                    })
        return result

    def _check_status_with_schedule(self, record, shift_start=None, late_min=15):
        """Check if punch-in is on time based on shift schedule."""
        if not shift_start:
            shift_start = "08:00"
        punch_in_str = str(record.get("punch_in", ""))
        if not punch_in_str:
            return "absent"
        
        # Parse times for comparison
        try:
            punch_h, punch_m = punch_in_str.split(":")[0], punch_in_str.split(":")[1]
            shift_h, shift_m = shift_start.split(":")
            punch_minutes = int(punch_h) * 60 + int(punch_m)
            shift_minutes = int(shift_h) * 60 + int(shift_m)
            late_by = punch_minutes - shift_minutes
        except (ValueError, IndexError):
            return "present"

        if late_by <= 0:
            return "on_time"
        elif late_by <= late_min:
            return "late"
        else:
            return "late"


    def get_monthly_metrics(self, year=None, month=None):
        if year is None: year = datetime.now().year
        if month is None: month = datetime.now().month
        import calendar
        working_days = sum(1 for d in range(1, calendar.monthrange(year, month)[1] + 1)
                          if datetime(year, month, d).weekday() < 7)

        db = self._get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT id, name, role FROM employees WHERE available=1 AND role != %s", ("manager",))
        employees = cur.fetchall()

        month_str = f"{year}-{month:02d}"
        cur.execute("SELECT * FROM attendance_records WHERE date LIKE %s", (month_str + "%",))
        all_records = cur.fetchall()

        metrics = {}
        for emp in employees:
            emp_recs = [r for r in all_records if r["emp_id"] == emp["id"]]
            days_present = len(set(r["date"].strftime("%Y-%m-%d") if hasattr(r["date"], "strftime") else str(r["date"]) for r in emp_recs if r.get("punch_in") or r.get("status") in ("on_time", "late", "present")))
            late_count = sum(1 for r in emp_recs if r.get("status") == "late")
            on_time = sum(1 for r in emp_recs if r.get("status") == "on_time")
            absent = working_days - days_present
            metrics[emp["id"]] = {
                "name": emp["name"], "role": emp["role"],
                "attendance_rate": round(days_present / working_days * 100, 1) if working_days > 0 else 0,
                "punctuality_rate": round(on_time / max(days_present, 1) * 100, 1),
                "late_count": late_count, "absent_days": max(0, absent),
                "working_days": working_days, "days_present": days_present,
            }
        return metrics

    def get_weekly_attendance(self):
        today = datetime.now()
        monday = today - timedelta(days=today.weekday())
        week_dates = [(monday + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

        db = self._get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT id, name, role FROM employees WHERE available=1 AND role != %s", ("manager",))
        employees = cur.fetchall()
        cur.execute("SELECT * FROM attendance_records WHERE date BETWEEN %s AND %s",
                    (week_dates[0], week_dates[-1]))
        all_records = cur.fetchall()

        result = {}
        for emp in employees:
            row = {"id": emp["id"], "name": emp["name"], "role": emp["role"]}
            for d in week_dates:
                day_recs = [r for r in all_records if r["emp_id"] == emp["id"] and str(r["date"]) == d]
                if day_recs:
                    latest = day_recs[-1]
                    row[d] = {
                        "status": latest.get("status", "present"),
                        "in": str(latest.get("punch_in", ""))[:5] if latest.get("punch_in") else None,
                        "out": str(latest.get("punch_out", ""))[:5] if latest.get("punch_out") else None,
                    }
                else:
                    row[d] = {"status": "no_record", "in": None, "out": None}
            result[emp["id"]] = row
        return {"week_start": week_dates[0], "week_end": week_dates[-1], "employees": result}

    def dashboard_format(self):
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
                    "name": m["name"], "role": m["role"],
                    "attendance_rate": m["attendance_rate"],
                    "punctuality_rate": m["punctuality_rate"],
                    "absent_days": m["absent_days"],
                } for emp_id, m in monthly.items()
            },
        }
