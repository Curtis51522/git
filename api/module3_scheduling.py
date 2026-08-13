"""
S3 Shift Scheduling -- Demand-driven CP-SAT solver.

Connects to S2 forecast to determine required staff per shift.
- Baker: 2 per slot for routine demand, 3 per slot for high demand
- Cashier: 1 per slot
- Barista: 2 baristas, 1 per slot
- Manager: 1 manager, optional per slot

Model: 9 operational employees plus 1 manager, with primary and approved
secondary roles.
Two overlapping 7-hour shifts cover 06:00-19:00.
"""

import asyncio
import concurrent.futures
import hashlib
import json
import logging
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from contextvars import copy_context
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from uuid import uuid4

from db.mysql_client import get_db, q
from api.auth import get_current_user, require_manager
from api.operation_clock import operation_now

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/s3", tags=["Module 3 - Shift Scheduling"])
_s3_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


# ======================================================================
# Data models
# ======================================================================

class Employee(BaseModel):
    id: str
    name: str
    role: str = Field(..., description="Single role: baker, cashier, barista, manager")
    min_hours_per_week: float = 14.0
    max_hours_per_week: float = 42.0
    available: bool = True
    unavailable_dates: List[str] = Field(default_factory=list)
    is_deputy: bool = False  # deputy manager (baker who covers manager off-shift)
    secondary_roles: List[str] = Field(default_factory=list)  # e.g. barista who can also cashier


class ShiftResult(BaseModel):
    date: str
    time_slot: str
    role: str
    employee_id: str
    employee_name: str
    demand_level: str = "normal"
    production_target: Optional[int] = None
    is_deputy: bool = False


# ======================================================================
# Default employees -- 9 operational employees plus 1 manager
# ======================================================================

DEFAULT_EMPLOYEES = [
    Employee(id="E001", name="Zhang Wei",  role="baker",    min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E002", name="Li Na",      role="cashier",  min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E003", name="Wang Lei",   role="barista",  min_hours_per_week=14, max_hours_per_week=56, secondary_roles=["cashier"]),
    Employee(id="E004", name="Liu Yang",   role="baker",    min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E005", name="Chen Hao",   role="baker",    min_hours_per_week=14, max_hours_per_week=56, is_deputy=True),
    Employee(id="E006", name="Zhao Min",   role="baker",    min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E007", name="Huang Jian", role="cashier",  min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E008", name="Wu Tao",     role="baker",    min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E009", name="Lin Yue",    role="barista",  min_hours_per_week=14, max_hours_per_week=56, secondary_roles=["cashier"]),
    Employee(id="E010", name="Sun Jie",    role="manager",  min_hours_per_week=14, max_hours_per_week=42),
]

TIME_SLOTS = ["06:00-13:00", "12:00-19:00"]
ROLES = ["baker", "cashier", "barista", "manager"]
SLOT_HOURS = {"06:00-13:00": 7, "12:00-19:00": 7}


_SCHEDULE_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schedule_history (
  id BIGINT NOT NULL AUTO_INCREMENT,
  change_group VARCHAR(36) NOT NULL,
  snapshot_type VARCHAR(24) NOT NULL,
  change_type VARCHAR(32) NOT NULL,
  change_reason VARCHAR(255) NOT NULL,
  source_schedule_id INT DEFAULT NULL,
  schedule_date DATE NOT NULL,
  time_slot VARCHAR(20) NOT NULL,
  employee_id VARCHAR(10) DEFAULT NULL,
  employee_name VARCHAR(50) DEFAULT NULL,
  role VARCHAR(30) DEFAULT NULL,
  staff_count INT NOT NULL DEFAULT 1,
  demand_level VARCHAR(10) DEFAULT 'normal',
  production_target INT DEFAULT NULL,
  recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_schedule_history_date (schedule_date, recorded_at),
  KEY idx_schedule_history_baseline (snapshot_type, schedule_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def _schedule_date_text(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _ensure_schedule_history_table(db) -> None:
    cursor = db.cursor()
    try:
        cursor.execute(_SCHEDULE_HISTORY_TABLE_SQL)
    finally:
        cursor.close()
    if not getattr(db, "autocommit", True):
        db.commit()


def _schedule_snapshot_payloads(
    rows: list,
    *,
    snapshot_type: str,
    change_type: str,
    reason: str,
    change_group: str,
) -> list:
    payloads = []
    for row in rows:
        payloads.append({
            "change_group": change_group,
            "snapshot_type": snapshot_type,
            "change_type": change_type,
            "change_reason": reason,
            "source_schedule_id": row.get("id"),
            "schedule_date": _schedule_date_text(row.get("schedule_date", "")),
            "time_slot": row.get("time_slot", ""),
            "employee_id": row.get("employee_id"),
            "employee_name": row.get("employee_name"),
            "role": row.get("role"),
            "staff_count": row.get("staff_count", 1),
            "demand_level": row.get("demand_level", "normal"),
            "production_target": row.get("production_target"),
        })
    return payloads


def _baseline_shift_counts(history_rows: list, schedule_dates: set) -> tuple:
    baseline_dates = {
        _schedule_date_text(row.get("schedule_date", ""))
        for row in history_rows
    }
    complete = bool(schedule_dates) and schedule_dates.issubset(baseline_dates)
    if not complete:
        return {}, False

    counts = {}
    for row in history_rows:
        employee_id = row.get("employee_id")
        if employee_id:
            counts[employee_id] = counts.get(employee_id, 0) + 1
    return counts, True


def _insert_schedule_snapshot(
    db,
    rows: list,
    *,
    snapshot_type: str,
    change_type: str,
    reason: str,
    change_group: str,
) -> None:
    payloads = _schedule_snapshot_payloads(
        rows,
        snapshot_type=snapshot_type,
        change_type=change_type,
        reason=reason,
        change_group=change_group,
    )
    if payloads:
        q(db, "schedule_history").insert(payloads).execute()


def _ensure_schedule_baseline(
    db,
    rows: list,
    *,
    change_type: str,
    reason: str,
    change_group: str,
) -> None:
    if not rows:
        return
    _ensure_schedule_history_table(db)
    row_dates = {_schedule_date_text(row.get("schedule_date", "")) for row in rows}
    existing = q(db, "schedule_history").select("schedule_date")\
        .eq("snapshot_type", "baseline")\
        .gte("schedule_date", min(row_dates))\
        .lte("schedule_date", max(row_dates)).execute()
    existing_dates = {
        _schedule_date_text(row.get("schedule_date", ""))
        for row in (existing.data or [])
    }
    missing_dates = row_dates - existing_dates
    if not missing_dates:
        return
    baseline_rows = [
        row for row in rows
        if _schedule_date_text(row.get("schedule_date", "")) in missing_dates
    ]
    _insert_schedule_snapshot(
        db,
        baseline_rows,
        snapshot_type="baseline",
        change_type=change_type,
        reason=reason,
        change_group=change_group,
    )


def _capture_schedule_change(db, rows: list, change_type: str, reason: str) -> None:
    if not rows:
        return
    change_group = str(uuid4())
    _ensure_schedule_baseline(
        db,
        rows,
        change_type=change_type,
        reason=reason,
        change_group=change_group,
    )
    _insert_schedule_snapshot(
        db,
        rows,
        snapshot_type="before_change",
        change_type=change_type,
        reason=reason,
        change_group=change_group,
    )


def _load_schedule_baseline(db, start_date: str, end_date: str, schedule_dates: set) -> tuple:
    _ensure_schedule_history_table(db)
    result = q(db, "schedule_history").select("schedule_date,employee_id")\
        .eq("snapshot_type", "baseline")\
        .gte("schedule_date", start_date)\
        .lte("schedule_date", end_date).execute()
    return _baseline_shift_counts(result.data or [], schedule_dates)


def _coverage_values(actual: int, baseline: Optional[int]) -> dict:
    """Return coverage values without treating a missing baseline as zero."""
    if baseline is None:
        return {
            "expected": actual,
            "rate_pct": 100.0,
            "extra_shifts": None,
            "baseline_available": False,
        }

    expected = max(0, baseline)
    if expected == 0:
        rate = 100.0
    else:
        rate = min(100.0, round(actual / expected * 100, 1))
    return {
        "expected": expected,
        "rate_pct": rate,
        "extra_shifts": max(0, actual - expected),
        "baseline_available": True,
    }


def _schedule_compliance_violations(rows: list, employees: List[Employee], days: int) -> list:
    """Evaluate hour, availability, and consecutive-day schedule constraints."""
    employee_lookup = {employee.id: employee for employee in employees}
    hours_by_employee = {}
    dates_by_employee = {}
    violations = []

    for row in rows:
        employee_id = row.get("employee_id")
        if not employee_id:
            continue
        hours_by_employee[employee_id] = hours_by_employee.get(employee_id, 0) + SLOT_HOURS.get(
            row.get("time_slot", ""), 7
        )
        schedule_date = _schedule_date_text(row.get("schedule_date", ""))
        dates_by_employee.setdefault(employee_id, set()).add(schedule_date)

        employee = employee_lookup.get(employee_id)
        if employee and schedule_date in employee.unavailable_dates:
            violations.append({
                "employee": employee.name,
                "type": f"Scheduled on unavailable date {schedule_date}",
            })

    covered_weeks = max(1, (days + 6) // 7)
    complete_weeks = days // 7
    for employee_id, employee in employee_lookup.items():
        hours = hours_by_employee.get(employee_id, 0)
        maximum = employee.max_hours_per_week * covered_weeks
        minimum = employee.min_hours_per_week * complete_weeks
        if hours > maximum:
            violations.append({
                "employee": employee.name,
                "type": f"Scheduled {hours:g}h above the {maximum:g}h period limit",
            })
        if complete_weeks and employee.available and employee.role != "manager" and hours < minimum:
            violations.append({
                "employee": employee.name,
                "type": f"Scheduled {hours:g}h below the {minimum:g}h period minimum",
            })

        consecutive_days = 0
        previous_date = None
        for date_text in sorted(dates_by_employee.get(employee_id, set())):
            current_date = datetime.strptime(date_text, "%Y-%m-%d").date()
            consecutive_days = consecutive_days + 1 if previous_date and (current_date - previous_date).days == 1 else 1
            previous_date = current_date
            if consecutive_days > 5:
                violations.append({
                    "employee": employee.name,
                    "type": "Scheduled for more than 5 consecutive days",
                })
                break

    return violations


def _stable_seed(value: str) -> int:
    """Return a process-stable integer seed for CP-SAT."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


# ======================================================================
# Employee loading
# ======================================================================

def load_employees() -> List[Employee]:
    try:
        db = get_db()
        r = q(db, "employees").select("*").execute()
        if r.data:
            results = []
            for e in r.data:
                unavailable = e.get("unavailable_dates", "[]")
                if isinstance(unavailable, str):
                    unavailable = json.loads(unavailable)
                # Look up is_deputy from defaults
                is_dep = False
                sec_roles = []
                for demp in DEFAULT_EMPLOYEES:
                    if demp.id == e["id"]:
                        is_dep = demp.is_deputy
                        sec_roles = demp.secondary_roles
                        break
                results.append(Employee(
                    id=e["id"],
                    name=e["name"],
                    role=e.get("role", "baker"),
                    min_hours_per_week=float(e.get("min_hours_per_week", 14)),
                    max_hours_per_week=float(e.get("max_hours_per_week", 42)),
                    available=bool(e.get("available", True)),
                    unavailable_dates=unavailable,
                    is_deputy=is_dep,
                    secondary_roles=sec_roles,
                ))
            return results
    except Exception:
        pass
    return DEFAULT_EMPLOYEES


# ======================================================================
# S2 forecast helper
# ======================================================================

def _classify_demand_level(total_units: int, weekday_baseline: Optional[float] = None) -> str:
    if total_units <= 0:
        return "low"
    if weekday_baseline and weekday_baseline > 0:
        demand_index = total_units / weekday_baseline
        if demand_index > 1.15:
            return "high"
        if demand_index < 0.85:
            return "low"
        return "normal"
    if total_units >= 400:
        return "high"
    if total_units >= 200:
        return "normal"
    return "low"


def _fetch_weekday_demand_baselines(start_date: str, lookback_days: int = 180) -> Dict[int, float]:
    try:
        base = datetime.strptime(start_date, "%Y-%m-%d").date()
        history_start = (base - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        db = get_db()
        c = db.cursor(dictionary=True)
        c.execute(
            "SELECT WEEKDAY(dt) AS weekday, AVG(day_units) AS avg_units "
            "FROM ("
            "  SELECT o.order_date AS dt, SUM(oi.quantity) AS day_units "
            "  FROM orders o "
            "  JOIN order_items oi ON oi.order_id = o.id "
            "  WHERE o.order_date >= %s "
            "    AND o.order_date < %s "
            "    AND LOWER(COALESCE(o.state, 'completed')) NOT IN ('refunded', 'cancelled') "
            "  GROUP BY o.order_date"
            ") daily "
            "GROUP BY WEEKDAY(dt)",
            (history_start, start_date),
        )
        return {
            int(row["weekday"]): float(row["avg_units"])
            for row in c.fetchall()
            if row.get("weekday") is not None and row.get("avg_units") is not None
        }
    except Exception as e:
        logger.warning("S3 weekday demand baseline query failed: %s", e)
        return {}


def _fetch_product_categories() -> Dict[str, str]:
    db = get_db()
    c = db.cursor(dictionary=True)
    c.execute("SELECT product_name, category FROM products")
    return {
        str(row["product_name"]).strip().lower(): str(row["category"]).strip().lower()
        for row in c.fetchall()
        if row.get("product_name") and row.get("category")
    }


def _aggregate_forecast_rows(
    forecasts: List[dict],
    start_date: str,
    product_categories: Dict[str, str],
) -> Dict[str, dict]:
    daily = {}
    for forecast in forecasts:
        forecast_date = forecast.get("forecast_date", "")
        if forecast_date < start_date:
            continue

        freshness = forecast.get("freshness_status", "Fresh")
        if freshness not in ("Fresh", "Total"):
            continue

        if forecast_date not in daily:
            daily[forecast_date] = {
                "total_units": 0,
                "baker_units": 0,
                "coffee_units": 0,
            }

        demand = int(forecast.get("predicted_demand", 0))
        product_name = str(forecast.get("product_name", "")).strip().lower()
        category = product_categories.get(product_name)
        if category == "beverage":
            daily[forecast_date]["coffee_units"] += demand
        elif category == "bakery":
            daily[forecast_date]["baker_units"] += demand
        else:
            logger.warning("S3 forecast product has no category mapping: %s", product_name)
        daily[forecast_date]["total_units"] += demand

    return daily


def _fetch_demand_forecast(start_date: str, days: int = 7) -> Dict[str, dict]:
    """Fetch S2 forecast, aggregate by date, and compute data-driven demand levels.

    Within-week relative ranking via S2 forecast: top 1/3 = high, bottom 1/3 = low, middle = normal.

    Returns: {date: {"total_units": int, "baker_units": int,
    "coffee_units": int, "demand_level": str}}
    """
    try:
        from api.module2_forecast import _do_forecast
        forecast_data = _do_forecast(None, days, start_date=start_date)
        forecasts = forecast_data.get("forecasts", [])
        daily = _aggregate_forecast_rows(
            forecasts,
            start_date,
            _fetch_product_categories(),
        )

        weekday_baselines = _fetch_weekday_demand_baselines(start_date)
        for d, item in daily.items():
            total = item["total_units"]
            weekday = datetime.strptime(d, "%Y-%m-%d").weekday()
            baseline = weekday_baselines.get(weekday)
            daily[d]["demand_level"] = _classify_demand_level(total, baseline)
            daily[d]["demand_baseline"] = round(baseline, 1) if baseline else None
            daily[d]["demand_index"] = round(total / baseline, 3) if baseline else None

        return daily

    except Exception as e:
        logger.error("S2 forecast fetch failed: %s", e, exc_info=True)
        return {}



def solve_shift_schedule(
    employees: List[Employee],
    start_date: str,
    num_days: int = 7,
    demand_forecast: Optional[Dict[str, dict]] = None,
    shop_closed_weekdays: Optional[set] = None,
) -> List[ShiftResult]:
    """CP-SAT shift scheduling with hard constraints and hour-balancing objective.

    Hard constraints:
      1. Employee only works primary or secondary roles
      2. At most 1 role per time slot
      3. At most 2 shifts per day
      4. No same-role double-shift unless demand forces it (low/normal days)
      5. Maximum 5 consecutive working days
      6. Weekly hours: 7h min, 56h max
      7. Unavailable dates honoured
      8. Per-slot staffing matches the demand requirement

    Soft objective: minimize (max_weekly_hours - min_weekly_hours) spread.
    """
    if shop_closed_weekdays is None:
        shop_closed_weekdays = set()
    if demand_forecast is None:
        demand_forecast = {}

    base = datetime.strptime(start_date, "%Y-%m-%d")

    if not demand_forecast:
        raise ValueError("A complete S2 demand forecast is required")

    emp_list = [e for e in employees if e.available and e.role != "manager"]
    if not emp_list:
        return []

    N = len(emp_list)
    S = len(TIME_SLOTS)
    R = len(ROLES)
    emp_of = {i: emp_list[i] for i in range(N)}

    # Allowed roles per employee
    allowed_roles = {}
    for i in range(N):
        emp = emp_list[i]
        roles = {emp.role}
        sec = getattr(emp, "secondary_roles", []) or []
        roles.update(sec)
        allowed_roles[i] = {ROLES.index(r) for r in roles if r in ROLES}

    baker_r = ROLES.index("baker")
    cashier_r = ROLES.index("cashier")
    barista_r = ROLES.index("barista")
    baker_count = sum(1 for i in range(N) if emp_of[i].role == "baker")
    # ---------- Build daily demand ----------
    high_day_count = 0
    daily_demand = {}
    for d in range(num_days):
        dt = base + timedelta(days=d)
        date_str = dt.strftime("%Y-%m-%d")

        if dt.weekday() in shop_closed_weekdays:
            daily_demand[d] = {"_level": "closed", "baker": 0, "cashier": 0, "barista": 0}
            continue

        fc = demand_forecast.get(date_str, {})
        level = fc.get("demand_level", "normal")

        if level == "high":
            req = {"baker": 3, "cashier": 1, "barista": 1, "_level": "high"}
            high_day_count += 1
        elif level == "low":
            req = {"baker": 2, "cashier": 1, "barista": 1, "_level": "low"}
        else:
            req = {"baker": 2, "cashier": 1, "barista": 1, "_level": "normal"}

        daily_demand[d] = req

    # ---------- CP-SAT model ----------
    model = cp_model.CpModel()

    shift = {}
    for e in range(N):
        for d in range(num_days):
            for s in range(S):
                for r in range(R):
                    shift[(e, d, s, r)] = model.NewBoolVar(f"x_e{e}_d{d}_s{s}_r{r}")

    # ---- C1: Only allowed roles, at most 1 role per slot ----
    for e in range(N):
        for d in range(num_days):
            for s in range(S):
                for r in range(R):
                    if r not in allowed_roles[e]:
                        model.Add(shift[(e, d, s, r)] == 0)
                model.Add(sum(shift[(e, d, s, r)] for r in range(R)) <= 1)

    # ---- C2: Per-slot coverage ----
    for d in range(num_days):
        req = daily_demand[d]
        if req.get("_level") == "closed":
            for e in range(N):
                for s in range(S):
                    for r in range(R):
                        model.Add(shift[(e, d, s, r)] == 0)
            continue

        bn = req.get("baker", 0)
        cn = req.get("cashier", 0)
        barn = req.get("barista", 0)

        for s in range(S):
            # Baker coverage
            if bn > 0:
                model.Add(sum(shift[(e, d, s, baker_r)] for e in range(N)) == bn)
            # Cashier coverage, including employees assigned through a secondary role
            if cn > 0:
                model.Add(sum(shift[(e, d, s, cashier_r)] for e in range(N)) == cn)
            # Barista coverage
            if barn > 0:
                model.Add(sum(shift[(e, d, s, barista_r)] for e in range(N)) == barn)

    # ---- C3: At most 2 shifts per day ----
    for e in range(N):
        for d in range(num_days):
            model.Add(sum(shift[(e, d, s, r)] for s in range(S) for r in range(R)) <= 2)

    # ---- C4: No same-role double-shift unless demand forces it ----
    for e in range(N):
        for d in range(num_days):
            bn = daily_demand[d].get("baker", 0)
            for r in range(R):
                rname = ROLES[r]
                # Forbid same-role double when both slots can be covered single-shift
                if rname == "baker" and bn * 2 <= baker_count:
                    model.Add(shift[(e, d, 0, r)] + shift[(e, d, 1, r)] <= 1)

    # ---- C5: Max 5 consecutive working days ----
    works = {}
    for e in range(N):
        for d in range(num_days):
            works[(e, d)] = model.NewBoolVar(f"w_e{e}_d{d}")
            day_total = sum(shift[(e, d, s, r)] for s in range(S) for r in range(R))
            model.Add(day_total >= 1).OnlyEnforceIf(works[(e, d)])
            model.Add(day_total == 0).OnlyEnforceIf(works[(e, d)].Not())
    for e in range(N):
        for d in range(num_days - 5):
            model.Add(sum(works[(e, d + k)] for k in range(6)) <= 5)

    # ---- C6: Weekly hours 7-56h for a full-week schedule ----
    slot_hours = [SLOT_HOURS.get(TIME_SLOTS[s], 7) for s in range(S)]
    sick_by_role = {}
    for e in range(N):
        emp = emp_of[e]
        if emp.unavailable_dates:
            sick_by_role[emp.role] = sick_by_role.get(emp.role, 0) + 1
    role_members = {}
    for role in ROLES:
        role_members[role] = sum(1 for e in range(N) if emp_of[e].role == role)

    role_required_hours = {role: 0 for role in ROLES}
    for d in range(num_days):
        for role in ("baker", "cashier", "barista"):
            role_required_hours[role] += daily_demand[d].get(role, 0) * sum(slot_hours)

    for e in range(N):
        emp = emp_of[e]
        weekly_hours = sum(
            shift[(e, d, s, r)] * slot_hours[s]
            for d in range(num_days) for s in range(S) for r in range(R)
        )
        model.Add(weekly_hours >= (7 if num_days >= 7 else 0))
        max_h = 56
        sick_in_role = sick_by_role.get(emp.role, 0)
        if sick_in_role > 0:
            total = role_members.get(emp.role, 1)
            available = max(1, total - sick_in_role)
            max_h = min(63, max_h * total // available)
        role_capacity = role_members.get(emp.role, 0) * max_h
        if role_required_hours.get(emp.role, 0) > role_capacity:
            max_h = 63
        if high_day_count >= 2:
            max_h = min(63, max_h + high_day_count * 7)
        model.Add(weekly_hours <= max_h)

    # ---- C7: Unavailable dates ----
    for e in range(N):
        emp = emp_of[e]
        for d in range(num_days):
            dt = base + timedelta(days=d)
            if dt.strftime("%Y-%m-%d") in emp.unavailable_dates:
                for s in range(S):
                    for r in range(R):
                        model.Add(shift[(e, d, s, r)] == 0)

    # ---- Objective: Minimize hour spread (max - min) ----
    max_hours = model.NewIntVar(0, 63, "max_hours")
    min_hours = model.NewIntVar(0, 63, "min_hours")
    for e in range(N):
        emp_hours = sum(
            shift[(e, d, s, r)] * slot_hours[s]
            for d in range(num_days) for s in range(S) for r in range(R)
        )
        model.Add(emp_hours <= max_hours)
        model.Add(emp_hours >= min_hours)
    spread = model.NewIntVar(0, 56, "spread")
    model.Add(spread == max_hours - min_hours)
    model.Minimize(spread)

    # ---- Solve ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    # Deterministic: same week should produce the same schedule across processes.
    solver.parameters.random_seed = _stable_seed(start_date)
    solver.parameters.num_search_workers = 1
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger = logging.getLogger("s3.solver")
        logger.error("Solver status: %s", solver.StatusName(status))
        return []

    # ---- Build results ----
    results = []
    for d in range(num_days):
        dt = base + timedelta(days=d)
        if dt.weekday() in shop_closed_weekdays:
            continue
        date_str = dt.strftime("%Y-%m-%d")
        fc = demand_forecast.get(date_str, {})
        level = daily_demand[d].get("_level", "normal")

        for s in range(S):
            for r_idx, role_name in enumerate(ROLES):
                for e in range(N):
                    if solver.Value(shift[(e, d, s, r_idx)]) == 1:
                        emp = emp_of[e]
                        prod_target = None
                        if role_name == "baker" and s == 0:
                            baker_units = fc.get("baker_units", 0)
                            baker_count = daily_demand.get(d, {}).get("baker", 1)
                            if baker_count > 0:
                                prod_target = baker_units // baker_count
                        results.append(ShiftResult(
                            date=date_str,
                            time_slot=TIME_SLOTS[s],
                            role=role_name,
                            employee_id=emp.id,
                            employee_name=emp.name,
                            demand_level=level,
                            production_target=prod_target,
                            is_deputy=getattr(emp, "is_deputy", False),
                        ))

    results.sort(key=lambda x: (x.date, x.time_slot, x.role))
    return results



# ======================================================================
# GET /s3/schedule
# ======================================================================
@router.get("/schedule", dependencies=[Depends(require_manager)])
async def get_schedule(
    date: str = Query(None),
    days: int = Query(7, ge=1, le=14),
):
    try:
        db = get_db()
        if date:
            end_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=days-1)).strftime("%Y-%m-%d")
            r = q(db, "shift_schedule").select("*").gte("schedule_date", date).lte("schedule_date", end_date).order("schedule_date,time_slot").execute()
        else:
            r = q(db, "shift_schedule").select("*").order("schedule_date,time_slot").execute()
        rows = r.data if r.data else []
    except Exception as e:
        return {"status": "ok", "schedule": [], "message": str(e)}

    all_emps_db = {e.id: e for e in load_employees()}
    schedule = []
    for row in rows:
        d = row["schedule_date"]
        if hasattr(d, "strftime"):
            d = d.strftime("%Y-%m-%d")
        emp = all_emps_db.get(row["employee_id"])
        schedule.append(ShiftResult(
            date=d,
            time_slot=row["time_slot"],
            role=row.get("role", ""),
            employee_id=row["employee_id"],
            employee_name=row["employee_name"],
            demand_level=row.get("demand_level", "normal"),
            production_target=row.get("production_target"),
            is_deputy=emp.is_deputy if emp else False,
        ))

    all_emps = {e.id: e for e in load_employees()}
    emp_summary = {}
    for s in schedule:
        eid = s.employee_id
        if eid not in emp_summary:
            emp = all_emps.get(eid)
            sec = emp.secondary_roles if emp else []
            emp_summary[eid] = {"name": s.employee_name, "hours": 0, "role": s.role, "secondary_roles": sec}
        emp_summary[eid]["hours"] += SLOT_HOURS.get(s.time_slot, 7)

    return {
        "status": "ok",
        "date": date,
        "schedule": [s.model_dump() for s in schedule],
        "employee_summary": emp_summary,
    }


@router.get("/schedule/history", dependencies=[Depends(require_manager)])
async def get_schedule_history(
    date: str = Query(...),
    days: int = Query(1, ge=1, le=14),
):
    """Return recorded schedule baselines and pre-change snapshots."""
    base = datetime.strptime(date, "%Y-%m-%d")
    end_date = (base + timedelta(days=days - 1)).strftime("%Y-%m-%d")
    db = get_db()
    _ensure_schedule_history_table(db)
    result = q(db, "schedule_history").select("*")\
        .gte("schedule_date", date)\
        .lte("schedule_date", end_date)\
        .order("recorded_at,id").execute()
    return {
        "status": "ok",
        "period": {"start": date, "end": end_date, "days": days},
        "history": result.data or [],
    }


# ======================================================================
# POST /s3/solve -- Demand-driven generation
# ======================================================================

def _save_kpi_snapshot(week_start, kpi_data, snapshot_type, trigger_event):
    """Save KPI snapshot to DB."""
    try:
        db = get_db()
        c = db.cursor()
        c.execute(
            "INSERT INTO kpi_snapshots (week_start, snapshot_type, trigger_event, data) VALUES (%s, %s, %s, %s)",
            (week_start, snapshot_type, trigger_event, json.dumps(kpi_data, default=str))
        )
        db.commit()
    except Exception:
        pass

def _log_sick_leave(employee_id, leave_date, action):
    """Log sick leave to audit table."""
    try:
        db = get_db()
        c = db.cursor()
        c.execute(
            "INSERT INTO sick_leave_log (employee_id, leave_date, action) VALUES (%s, %s, %s)",
            (employee_id, leave_date, action)
        )
        db.commit()
    except Exception:
        pass

def _is_past_date(date_str):
    """Check if a date is in the past (before today)."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return d < operation_now().date()
    except Exception:
        return False


async def _run_s3_task(task, payload):
    loop = asyncio.get_running_loop()
    context = copy_context()
    return await loop.run_in_executor(_s3_executor, context.run, task, payload)

def _persist_schedule(results, base, num_days, change_type):
    db = get_db()
    start_date = base.strftime("%Y-%m-%d")
    end_date = (base + timedelta(days=num_days - 1)).strftime("%Y-%m-%d")
    existing = q(db, "shift_schedule").select("*")\
        .gte("schedule_date", start_date)\
        .lte("schedule_date", end_date).execute()
    _capture_schedule_change(
        db,
        existing.data or [],
        change_type,
        f"Schedule replaced for {start_date} through {end_date}",
    )
    for i in range(num_days):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        q(db, "shift_schedule").delete().eq("schedule_date", d).execute()
    for r in results:
        q(db, "shift_schedule").insert({
            "schedule_date": r.date,
            "time_slot": r.time_slot,
            "employee_id": r.employee_id,
            "employee_name": r.employee_name,
            "role": r.role,
            "staff_count": 1,
            "demand_level": r.demand_level,
            "production_target": r.production_target,
        }).execute()
    current = q(db, "shift_schedule").select("*")\
        .gte("schedule_date", start_date)\
        .lte("schedule_date", end_date).execute()
    _ensure_schedule_baseline(
        db,
        current.data or [],
        change_type=change_type,
        reason=f"Initial schedule recorded for {start_date} through {end_date}",
        change_group=str(uuid4()),
    )


def _rebuild_from_employees(start_date, num_days, employees, change_type):
    base = datetime.strptime(start_date, "%Y-%m-%d")
    demand_forecast = _fetch_demand_forecast(start_date, num_days)
    expected_dates = {
        (base + timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(num_days)
    }
    missing_dates = sorted(expected_dates - set(demand_forecast))
    if missing_dates:
        raise RuntimeError(
            "S2 forecast is unavailable for: " + ", ".join(missing_dates)
        )
    results = solve_shift_schedule(
        employees, start_date, num_days,
        demand_forecast=demand_forecast,
        shop_closed_weekdays=set(),
    )
    requested_end = base + timedelta(days=num_days)
    results = [r for r in results if r.date >= start_date and r.date < requested_end.strftime("%Y-%m-%d")]
    try:
        _persist_schedule(results, base, num_days, change_type)
    except Exception:
        logger.exception("Failed to persist schedule")
        raise
    return results
def _build_schedule_response(results):
    all_emps = {e.id: e for e in load_employees()}
    emp_summary = {}
    for s in results:
        eid = s.employee_id
        if eid not in emp_summary:
            emp = all_emps.get(eid)
            sec = emp.secondary_roles if emp else []
            emp_summary[eid] = {"name": s.employee_name, "hours": 0, "role": s.role, "secondary_roles": sec}
        emp_summary[eid]["hours"] += SLOT_HOURS.get(s.time_slot, 7)
    return {
        "status": "ok",
        "total_shifts": len(results),
        "schedule": [r.model_dump() for r in results],
        "employee_summary": emp_summary,
    }


def _open_schedule_dates(base: datetime, days: int) -> set:
    return {
        (base + timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(days)
    }

def _solve_impl(payload: dict) -> dict:
    start_date = payload.get("start_date", operation_now().strftime("%Y-%m-%d"))
    if _is_past_date(start_date):
        return {"status": "error", "message": "Cannot modify past schedules. Select today or a future date."}
    num_days = min(payload.get("days", 7), 14)
    unavailable_map = payload.get("unavailable", {})

    employees = load_employees()
    for e in employees:
        if e.id in unavailable_map:
            e.unavailable_dates = unavailable_map[e.id]

    _rebuild_from_employees(start_date, num_days, employees, "solve")
    # Auto-replace sick employees after generation
    base = datetime.strptime(start_date, "%Y-%m-%d")
    db2 = get_db()
    for e in employees:
        if e.unavailable_dates:
            for d in e.unavailable_dates:
                try:
                    d_date = datetime.strptime(d, "%Y-%m-%d").date()
                    if base.date() <= d_date < (base + timedelta(days=num_days)).date():
                        _replace_sick_shifts(db2, e.id, d, e.role, employees)
                except ValueError:
                    pass
    # Re-read schedule from DB to include replacements
    r = q(db2, "shift_schedule").select("*").gte("schedule_date", start_date).lte("schedule_date", (base + timedelta(days=num_days - 1)).strftime("%Y-%m-%d")).order("schedule_date").order("time_slot").execute()
    schedule_rows = r.data or []
    schedule_list = []
    for row in schedule_rows:
        sd = row.get("schedule_date")
        if hasattr(sd, "strftime"):
            sd = sd.strftime("%Y-%m-%d")
        schedule_list.append(ShiftResult(
            date=str(sd),
            time_slot=row.get("time_slot", ""),
            role=row.get("role", ""),
            employee_id=row.get("employee_id", ""),
            employee_name=row.get("employee_name", ""),
            demand_level=row.get("demand_level", "normal"),
            production_target=row.get("production_target"),
        ))
    return _build_schedule_response(schedule_list)


@router.post("/solve", dependencies=[Depends(require_manager)])
async def solve_schedule(payload: dict):
    return await _run_s3_task(_solve_impl, payload)




# ======================================================================
# Helpers
# ======================================================================
def _date_str(v):
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    return str(v)

# ======================================================================
# POST /s3/swap -- Same-role only, cross-date supported
# ======================================================================
@router.post("/swap", dependencies=[Depends(require_manager)])
async def swap_employees(payload: dict):
    date = payload.get("date", "")
    time_slot = payload.get("time_slot", "")
    from_id = payload.get("from_employee_id", "")
    to_id = payload.get("to_employee_id", "")
    to_date = payload.get("to_date", date)
    to_time_slot = payload.get("to_time_slot", "")

    if not all([date, time_slot, from_id, to_id]):
        return {"status": "error", "message": "Missing required fields"}
    if _is_past_date(date):
        return {"status": "error", "message": "Cannot modify past schedules. Select today or a future date."}
    if from_id == to_id and date == to_date and time_slot == to_time_slot:
        return {"status": "error", "message": "Cannot swap with yourself"}

    employees = {e.id: e for e in load_employees()}
    if from_id not in employees or to_id not in employees:
        return {"status": "error", "message": "Unknown employee ID"}

    from_emp = employees[from_id]
    to_emp = employees[to_id]

    try:
        db = get_db()
        r1 = q(db, "shift_schedule").select("*").eq("schedule_date", date).execute()
        all_shifts = r1.data if r1.data else []
        if to_date != date:
            r2 = q(db, "shift_schedule").select("*").eq("schedule_date", to_date).execute()
            all_shifts += (r2.data if r2.data else [])
    except Exception:
        return {"status": "error", "message": "Could not fetch schedule"}

    from_shift = next((s for s in all_shifts if s.get("employee_id") == from_id and str(s.get("time_slot","")) == time_slot and _date_str(s.get("schedule_date","")) == date), None)
    if not from_shift:
        return {"status": "error", "message": f"{from_emp.name} has no shift on {date} {time_slot}"}

    to_shift = next((s for s in all_shifts if s.get("employee_id") == to_id and _date_str(s.get("schedule_date","")) == to_date and (not to_time_slot or str(s.get("time_slot","")) == to_time_slot)), None)
    if not to_shift:
        return {"status": "error", "message": f"{to_emp.name} has no shift on {to_date}" + (f" {to_time_slot}" if to_time_slot else "")}

    # Skill-based swap check: each employee must be able to cover the other's assigned role.
    from_skills = {from_emp.role} | set(getattr(from_emp, "secondary_roles", []) or [])
    to_skills = {to_emp.role} | set(getattr(to_emp, "secondary_roles", []) or [])
    from_role = from_shift.get("role", "")
    to_role = to_shift.get("role", "")
    if to_role not in from_skills:
        return {"status": "rejected", "reason": f"{from_emp.name} cannot take {to_emp.name}'s {to_role} shift (skills: {sorted(from_skills)})"}
    if from_role not in to_skills:
        return {"status": "rejected", "reason": f"{to_emp.name} cannot take {from_emp.name}'s {from_role} shift (skills: {sorted(to_skills)})"}

    if to_date in to_emp.unavailable_dates:
        return {"status": "rejected", "reason": f"{to_emp.name} is unavailable on {to_date}"}

    try:
        db = get_db()
        _capture_schedule_change(
            db,
            all_shifts,
            "swap",
            f"{from_id} swapped with {to_id}",
        )
        cur = db.cursor()
        cur.execute("START TRANSACTION")
        try:
            q(db, "shift_schedule").update({
                "employee_id": to_id, "employee_name": to_emp.name,
            }).eq("id", from_shift["id"]).execute()

            q(db, "shift_schedule").update({
                "employee_id": from_id, "employee_name": from_emp.name,
            }).eq("id", to_shift["id"]).execute()

            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

        return {
            "status": "ok",
            "message": f"Swapped: {from_emp.name} ({date} {time_slot}) <-> {to_emp.name} ({to_date} {to_shift.get('time_slot','')})",
        }
    except Exception as e:
        logger.error("Swap transaction failed: %s", e, exc_info=True)
        return {"status": "error", "message": str(e)}


# ======================================================================
# Sick leave persistence helpers
# ======================================================================

def _add_sick_date(employee_id: str, date: str):
    """Add a date to an employee's unavailable_dates in the DB."""
    try:
        db = get_db()
        r = q(db, "employees").select("unavailable_dates").eq("id", employee_id).execute()
        if r.data:
            current = r.data[0].get("unavailable_dates", "[]")
            if isinstance(current, str):
                current = json.loads(current)
            if not isinstance(current, list):
                current = []
            if date not in current:
                current.append(date)
            q(db, "employees").update({"unavailable_dates": json.dumps(current)}).eq("id", employee_id).execute()
    except Exception:
        logger.error("Failed to add sick date for employee %s", employee_id, exc_info=True)

def _remove_sick_date(employee_id: str, date: str):
    """Remove a date from an employee's unavailable_dates in the DB."""
    try:
        db = get_db()
        r = q(db, "employees").select("unavailable_dates").eq("id", employee_id).execute()
        if r.data:
            current = r.data[0].get("unavailable_dates", "[]")
            if isinstance(current, str):
                current = json.loads(current)
            if not isinstance(current, list):
                current = []
            if date in current:
                current.remove(date)
            q(db, "employees").update({"unavailable_dates": json.dumps(current)}).eq("id", employee_id).execute()
    except Exception:
        logger.error("Failed to remove sick date for employee %s", employee_id, exc_info=True)

def _clear_all_sick_dates():
    """Clear all employees' unavailable_dates."""
    try:
        db = get_db()
        q(db, "employees").update({"unavailable_dates": "[]"}).neq("id", "").execute()
    except Exception:
        pass

# ======================================================================
# POST /s3/resync
# ======================================================================
def _resync_impl(payload: dict) -> dict:
    start_date = payload.get("start_date", operation_now().strftime("%Y-%m-%d"))
    if _is_past_date(start_date):
        return {"status": "error", "message": "Cannot modify past schedules. Select today or a future date."}
    num_days = min(payload.get("days", 7), 14)
    _clear_all_sick_dates()
    employees = load_employees()
    _rebuild_from_employees(start_date, num_days, employees, "resync")
    # Auto-replace sick employees after generation
    base = datetime.strptime(start_date, "%Y-%m-%d")
    db2 = get_db()
    for e in employees:
        if e.unavailable_dates:
            for d in e.unavailable_dates:
                try:
                    d_date = datetime.strptime(d, "%Y-%m-%d").date()
                    if base.date() <= d_date < (base + timedelta(days=num_days)).date():
                        _replace_sick_shifts(db2, e.id, d, e.role, employees)
                except ValueError:
                    pass
    # Re-read schedule from DB to include replacements
    r = q(db2, "shift_schedule").select("*").gte("schedule_date", start_date).lte("schedule_date", (base + timedelta(days=num_days - 1)).strftime("%Y-%m-%d")).order("schedule_date").order("time_slot").execute()
    schedule_rows = r.data or []
    schedule_list = []
    for row in schedule_rows:
        sd = row.get("schedule_date")
        if hasattr(sd, "strftime"):
            sd = sd.strftime("%Y-%m-%d")
        schedule_list.append(ShiftResult(
            date=str(sd),
            time_slot=row.get("time_slot", ""),
            role=row.get("role", ""),
            employee_id=row.get("employee_id", ""),
            employee_name=row.get("employee_name", ""),
            demand_level=row.get("demand_level", "normal"),
            production_target=row.get("production_target"),
        ))
    return _build_schedule_response(schedule_list)


@router.post("/resync", dependencies=[Depends(require_manager)])
async def resync_schedule(payload: dict):
    return await _run_s3_task(_resync_impl, payload)


# ======================================================================
# POST /s3/sick -- Persist sick leave + resync
# ======================================================================

def _replace_sick_shifts(db, employee_id: str, date: str, role: str, employees: list, time_slot: str = None) -> int:
    """Remove sick employee's shifts for a date and find same-role replacements.
    Returns number of shifts replaced."""
    query = q(db, "shift_schedule").select("*").eq("employee_id", employee_id).eq("schedule_date", date)
    if time_slot:
        query = query.eq("time_slot", time_slot)
    sick_shifts = query.execute()
    if sick_shifts.data:
        day_rows = q(db, "shift_schedule").select("*").eq("schedule_date", date).execute()
        _capture_schedule_change(
            db,
            day_rows.data or [],
            "sick_leave",
            f"Sick leave recorded for {employee_id}",
        )
    removed_slots = []
    for row in (sick_shifts.data or []):
        # Save original to sick_replacements for undo
        try:
            q(db, "sick_replacements").insert({
                "original_employee_id": employee_id,
                "original_employee_name": row.get("employee_name", ""),
                "schedule_date": date,
                "time_slot": row["time_slot"],
                "role": row.get("role", role),
                "demand_level": row.get("demand_level", "normal"),
                "production_target": row.get("production_target"),
                "replaced_at": operation_now().strftime("%Y-%m-%d %H:%M:%S"),
                "replacement_employee_id": "",
                "is_undone": 0,
            }).execute()
        except Exception:
            pass
        removed_slots.append({
            "id": row["id"],
            "time_slot": row["time_slot"],
            "role": row.get("role", role),
            "demand_level": row.get("demand_level", "normal"),
            "production_target": row.get("production_target"),
        })
        q(db, "shift_schedule").delete().eq("id", row["id"]).execute()
    replaced = 0
    for slot in removed_slots:
        candidates = [e for e in employees
                      if e.role == slot["role"] and e.id != employee_id
                      and date not in e.unavailable_dates]
        for candidate in candidates:
            existing = q(db, "shift_schedule").select("id").eq("employee_id", candidate.id).eq("schedule_date", date).eq("time_slot", slot["time_slot"]).execute()
            if not existing.data:
                q(db, "shift_schedule").insert({
                    "schedule_date": date,
                    "time_slot": slot["time_slot"],
                    "employee_id": candidate.id,
                    "employee_name": candidate.name,
                    "role": slot["role"],
                    "staff_count": 1,
                    "demand_level": slot.get("demand_level", "normal"),
                    "production_target": slot.get("production_target"),
                }).execute()
                # Record replacement in sick_replacements
                try:
                    q(db, "sick_replacements").update({"replacement_employee_id": candidate.id}).eq("original_employee_id", employee_id).eq("schedule_date", date).eq("time_slot", slot["time_slot"]).eq("is_undone", 0).execute()
                except Exception:
                    pass
                replaced += 1
                break
    return replaced

def _sick_impl(payload: dict) -> dict:
    employee_id = payload.get("employee_id", "")
    date = payload.get("date", "")
    start_date = payload.get("start_date", date or operation_now().strftime("%Y-%m-%d"))
    if not employee_id or not date:
        return {"status": "error", "message": "employee_id and date required"}
    if _is_past_date(date):
        return {"status": "error", "message": "Cannot modify past schedules. Select today or a future date."}
    _log_sick_leave(employee_id, date, "added")
    _add_sick_date(employee_id, date)
    time_slot = payload.get("time_slot", "")
    # Local replacement: only fix the sick employee's shifts on the sick date
    employees = load_employees()
    emp_lookup = {e.id: e for e in employees}
    sick_emp = emp_lookup.get(employee_id)
    if not sick_emp:
        return {"status": "error", "message": f"Employee {employee_id} not found"}
    db = get_db()
    _replace_sick_shifts(db, employee_id, date, sick_emp.role, employees, time_slot)
    # 3. Return current schedule from DB
    num_days = min(payload.get("days", 7), 14)
    base = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = base + timedelta(days=num_days)
    r = q(db, "shift_schedule").select("*").gte("schedule_date", start_date).lte("schedule_date", (end_date - timedelta(days=1)).strftime("%Y-%m-%d")).order("schedule_date").order("time_slot").execute()
    schedule_rows = r.data or []
    schedule_list = []
    for row in schedule_rows:
        sd = row.get("schedule_date")
        if hasattr(sd, "strftime"):
            sd = sd.strftime("%Y-%m-%d")
        schedule_list.append(ShiftResult(
            date=str(sd),
            time_slot=row.get("time_slot", ""),
            role=row.get("role", ""),
            employee_id=row.get("employee_id", ""),
            employee_name=row.get("employee_name", ""),
            demand_level=row.get("demand_level", "normal"),
            production_target=row.get("production_target"),
        ))
    return _build_schedule_response(schedule_list)


@router.post("/sick", dependencies=[Depends(require_manager)])
async def mark_sick(payload: dict):
    return await _run_s3_task(_sick_impl, payload)


# ======================================================================
# POST /s3/unsick -- Remove sick leave + resync
# ======================================================================
def _unsick_impl(payload: dict) -> dict:
    employee_id = payload.get("employee_id", "")
    date = payload.get("date", "")
    start_date = payload.get("start_date", date or operation_now().strftime("%Y-%m-%d"))
    if not employee_id or not date:
        return {"status": "error", "message": "employee_id and date required"}
    if _is_past_date(date):
        return {"status": "error", "message": "Cannot modify past schedules. Select today or a future date."}
    _log_sick_leave(employee_id, date, "removed")
    _remove_sick_date(employee_id, date)
    
    db = get_db()
    # Restore original shifts from sick_replacements
    try:
        replacements = q(db, "sick_replacements").select("*").eq("original_employee_id", employee_id).eq("schedule_date", date).eq("is_undone", 0).execute()
        if replacements.data:
            day_rows = q(db, "shift_schedule").select("*").eq("schedule_date", date).execute()
            _capture_schedule_change(
                db,
                day_rows.data or [],
                "sick_leave_removed",
                f"Sick leave removed for {employee_id}",
            )
        for rep in (replacements.data or []):
            rep_id = rep.get("replacement_employee_id", "")
            # Delete replacement shift
            if rep_id:
                q(db, "shift_schedule").delete().eq("employee_id", rep_id).eq("schedule_date", date).eq("time_slot", rep["time_slot"]).execute()
            # Restore original employee's shift
            q(db, "shift_schedule").insert({
                "schedule_date": date,
                "time_slot": rep["time_slot"],
                "employee_id": employee_id,
                "employee_name": rep["original_employee_name"],
                "role": rep["role"],
                "staff_count": 1,
                "demand_level": rep.get("demand_level", "normal"),
                "production_target": rep.get("production_target"),
            }).execute()
            # Mark as undone
            q(db, "sick_replacements").update({"is_undone": 1}).eq("id", rep["id"]).execute()
    except Exception as e:
        logger.error("Failed to restore from sick_replacements: %s", e, exc_info=True)
    
    # Re-read and return current schedule
    num_days = min(payload.get("days", 7), 14)
    base = datetime.strptime(start_date, "%Y-%m-%d")
    r = q(db, "shift_schedule").select("*").gte("schedule_date", start_date).lte("schedule_date", (base + timedelta(days=num_days - 1)).strftime("%Y-%m-%d")).order("schedule_date").order("time_slot").execute()
    schedule_rows = r.data or []
    schedule_list = []
    for row in schedule_rows:
        sd = row.get("schedule_date")
        if hasattr(sd, "strftime"):
            sd = sd.strftime("%Y-%m-%d")
        schedule_list.append(ShiftResult(
            date=str(sd),
            time_slot=row.get("time_slot", ""),
            role=row.get("role", ""),
            employee_id=row.get("employee_id", ""),
            employee_name=row.get("employee_name", ""),
            demand_level=row.get("demand_level", "normal"),
            production_target=row.get("production_target"),
        ))
    _save_kpi_snapshot(start_date, {}, "adjustment", "unsick")
    return _build_schedule_response(schedule_list)


@router.post("/unsick", dependencies=[Depends(require_manager)])
async def unmark_sick(payload: dict):
    return await _run_s3_task(_unsick_impl, payload)

# ======================================================================
# GET /s3/kpi -- Scheduling KPIs
# ======================================================================

@router.get("/kpi", dependencies=[Depends(require_manager)])
async def get_kpi(
    start_date: str = Query(None),
    days: int = Query(7, ge=1, le=14),
):
    """Compute coverage, fairness, and compliance KPIs for the schedule."""
    try:
        if not start_date:
            start_date = (operation_now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        base = datetime.strptime(start_date, "%Y-%m-%d")
        # Allow KPI queries across the retained historical date range.
        end_date = (base + timedelta(days=days - 1)).strftime("%Y-%m-%d")

        db = get_db()
        r = q(db, "shift_schedule").select("*")\
            .gte("schedule_date", start_date)\
            .lte("schedule_date", end_date)\
            .order("schedule_date,time_slot").execute()
        rows = r.data if r.data else []

        if not rows:
            n_working = len(_open_schedule_dates(base, days))
            return {
                "status": "ok",
                "period": {"start": start_date, "end": end_date, "days": days, "working_days": n_working},
                "summary": {"total_shifts": 0, "total_hours": 0, "avg_hours_per_emp": 0, "employees_scheduled": 0},
                "coverage": [],
                "fairness": {},
                "compliance": {"violations": [], "all_pass": None, "evaluated": False},
                "message": "No schedule for this period. Run /s3/solve first.",
            }

        # ---- open business days ----
        expected_dates = _open_schedule_dates(base, days)
        n_working = len(expected_dates)

        # ---- Employee hours ----
        emp_hours = {}
        for row in rows:
            eid = row["employee_id"]
            if eid not in emp_hours:
                emp_hours[eid] = {
                    "name": row["employee_name"],
                    "role": row.get("role", ""),
                    "hours": 0,
                    "shifts": 0,
                }
            emp_hours[eid]["hours"] += SLOT_HOURS.get(row["time_slot"], 7)
            emp_hours[eid]["shifts"] += 1

        # ---- Coverage (per-employee, real-time from schedule) ----
        coverage = []
        all_emps = load_employees()
        schedule_dates = {
            _schedule_date_text(row.get("schedule_date", ""))
            for row in rows
        }
        baseline_map, baseline_complete = _load_schedule_baseline(
            db,
            start_date,
            end_date,
            schedule_dates,
        )
        for e in all_emps:
            eid = e.id
            info = emp_hours.get(eid, {"shifts": 0})
            actual = info.get("shifts", 0)
            if actual < 0:
                actual = 0
            baseline_shifts = baseline_map.get(eid, 0) if baseline_complete else None
            coverage_values = _coverage_values(actual, baseline_shifts)
            coverage.append({
                "employee": e.name,
                "role": e.role,
                "filled": actual,
                **coverage_values,
            })

        DUAL_ROLE_IDS = {'E005'}  # Ahmad baker+deputy  # Raj (barista+cashier), Ahmad (baker+deputy)
        # ---- 3. Fairness (hours gap within same role, excludes dual-role) ----
        fairness = {}
        all_emps_lookup = {e.id: e for e in all_emps}
        for role in ROLES:
            role_emps = {eid: info for eid, info in emp_hours.items()
                        if all_emps_lookup.get(eid) and all_emps_lookup[eid].role == role and eid not in DUAL_ROLE_IDS}
            if len(role_emps) >= 2:
                hrs = [info["hours"] for info in role_emps.values()]
                fairness[role] = {
                    "employees": {info["name"]: info["hours"] for _, info in role_emps.items()},
                    "gap_hours": max(hrs) - min(hrs),
                    "fair": (max(hrs) - min(hrs)) <= 8,
                }
            elif len(role_emps) == 1:
                info = list(role_emps.values())[0]
                fairness[role] = {
                    "employees": {info["name"]: info["hours"]},
                    "gap_hours": 0,
                    "fair": True,
                    "note": "single employee in role",
                }

        # ---- 4. Compliance ----
        violations = _schedule_compliance_violations(rows, all_emps, days)

        total_hours = sum(info["hours"] for info in emp_hours.values())
        n_emps = len(emp_hours)

        result = {
            "status": "ok",
            "period": {
                "start": start_date,
                "end": end_date,
                "days": days,
                "working_days": n_working,
            },
            "coverage": coverage,
            "fairness": fairness,
            "compliance": {
                "violations": violations,
                "all_pass": len(violations) == 0,
                "evaluated": True,
            },
            "summary": {
                "total_shifts": len(rows),
                "total_hours": total_hours,
                "avg_hours_per_emp": round(total_hours / n_emps, 1) if n_emps else 0,
                "employees_scheduled": n_emps,
            },
        }
        _save_kpi_snapshot(start_date, result, "auto", "kpi_query")
        return result

    except Exception as e:
        logger.error("KPI error: %s", e)
        return {"status": "error", "message": str(e)}

# ======================================================================
# S3 Production Scheduler endpoints (from s3_scheduling/scheduler.py)
# ======================================================================
def _load_day1_stock(breads, start_date):
    day1_stock = {product_name: 0 for product_name in breads}
    db = None
    cursor = None
    try:
        selected_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        now = operation_now()
        balance_time = (
            now
            if selected_date == now.date()
            else datetime.combine(selected_date, datetime.min.time())
        )
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            """
            SELECT bi.product_name,
                   COALESCE(SUM(
                       CASE
                           WHEN it.transaction_type = 'inflow'
                               THEN ABS(it.quantity)
                           WHEN it.transaction_type = 'outflow'
                               THEN -ABS(it.quantity)
                           ELSE 0
                       END
                   ), 0) AS units
            FROM batch_inventory bi
            JOIN products p ON p.product_name = bi.product_name
            LEFT JOIN inventory_transactions it
                   ON it.batch_id = bi.batch_id
                  AND it.transaction_time < %s
            WHERE p.category = 'bakery'
              AND DATE(bi.production_time) = DATE(%s) - INTERVAL 1 DAY
            GROUP BY bi.product_name
            HAVING units > 0
            """,
            (
                balance_time.strftime("%Y-%m-%d %H:%M:%S"),
                start_date,
            ),
        )
        for product_name, units in cursor.fetchall():
            if product_name in day1_stock:
                day1_stock[product_name] = int(units or 0)
    except Exception:
        logger.exception("Failed to load Day-1 batch inventory")
    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close()
    return day1_stock


@router.get("/plan/7day", dependencies=[Depends(require_manager)])
async def get_7day_production_plan(date: str = None):
    """
    Forecasting Dashboard Panel 2: 7-day production plan grid.
    Uses real S2 quantile models for demand prediction.
    Query: GET /s3/plan/7day?date=2026-06-30
    """
    from s3_scheduling.scheduler import Scheduler, generate_7day_s2_forecast
    from datetime import datetime as _dt

    if date is None:
        date = _dt.now().strftime("%Y-%m-%d")
    start_date = _dt.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")

    s = Scheduler()
    day1_stock = _load_day1_stock(s.breads, start_date)
    forecast = generate_7day_s2_forecast(start_date)
    result = s.generate_7day_plan(start_date, day1_stock, forecast)
    day1_stock_total = sum(day1_stock.values())
    dashboard_7day = dict(result["dashboard_7day"])
    dashboard_7day["day1_stock_total"] = day1_stock_total

    return {
        "status": "ok",
        "generated_at": _dt.now().isoformat(),
        "dashboard_7day": dashboard_7day,
        "weekly_summary": {
            "total_bake": result["weekly_summary"]["total_bake"],
            "total_profit": result["weekly_summary"]["total_profit"],
            "total_revenue": result["weekly_summary"]["total_revenue"],
            "day1_stock_total": day1_stock_total,
            "daily_profits": result["weekly_summary"]["daily_profits"],
            "scenarios": result["weekly_summary"]["scenarios"],
            "top_products": result["weekly_summary"]["top_products"],
        },
    }


@router.get("/materials", dependencies=[Depends(require_manager)])
async def get_materials_procurement(date: str = None):
    """
    Forecasting Dashboard Panel 3: Raw material procurement list.
    Query: GET /s3/materials?date=2026-06-30
    """
    from s3_scheduling.scheduler import Scheduler, generate_7day_s2_forecast
    from datetime import datetime as _dt2

    if date is None:
        date = _dt2.now().strftime("%Y-%m-%d")
    start_date = _dt2.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")

    s = Scheduler()
    day1_stock = _load_day1_stock(s.breads, start_date)
    forecast = generate_7day_s2_forecast(start_date)
    result = s.generate_7day_plan(start_date, day1_stock, forecast)

    return {
        "status": "ok",
        "generated_at": _dt2.now().isoformat(),
        "dashboard_materials": result["dashboard_materials"],
    }

# ======================================================================
# KPI Attendance endpoints
# ======================================================================

from kpi.attendance import AttendanceSystem

_attendance_system = None

def _get_attendance():
    global _attendance_system
    if _attendance_system is None:
        _attendance_system = AttendanceSystem()
    return _attendance_system


def _attendance_summary(employees):
    attended_statuses = {
        "on_time",
        "late",
        "early_leave",
        "late_and_early_leave",
        "present",
    }
    return {
        "total": len(employees),
        "present": sum(row["status"] in attended_statuses for row in employees),
        "absent": sum(row["status"] == "absent" for row in employees),
        "late": sum(
            row["status"] in {"late", "late_and_early_leave"}
            for row in employees
        ),
        "early_leave": sum(
            row["status"] in {"early_leave", "late_and_early_leave"}
            for row in employees
        ),
        "scheduled": sum(row["status"] == "scheduled" for row in employees),
    }


def _attach_attendance_metrics(employees, metrics):
    metric_fields = (
        "scheduled_days",
        "attended_days",
        "attendance_rate",
        "attendance_display",
        "punctuality_rate",
        "shift_completion_rate",
        "late_count",
        "early_leave_count",
        "absent_days",
    )
    result = []
    for employee in employees:
        row = dict(employee)
        employee_metrics = metrics.get(employee["id"], {})
        for field in metric_fields:
            if field in employee_metrics:
                row[field] = employee_metrics[field]
        result.append(row)
    return result


@router.get("/attendance", dependencies=[Depends(require_manager)])
async def get_attendance_dashboard(date: str = ""):
    """Return schedule-based attendance and month-to-date employee metrics.
    If date is provided and not today, returns historical attendance for that date."""
    att = _get_attendance()
    selected_time = operation_now()
    today_str = selected_time.strftime("%Y-%m-%d")
    target_date = date if date else today_str

    schedule = []
    try:
        sched_resp = await get_schedule(date=target_date, days=1)
        schedule = sched_resp.get("schedule", [])
    except Exception:
        pass

    if date and date != today_str:
        employees = att.get_date_attendance(date, schedule)
        selected = datetime.strptime(date, "%Y-%m-%d")
        period_end = selected + timedelta(days=1, seconds=-1)
    else:
        employees = att.get_today_attendance(schedule)
        selected = datetime.strptime(today_str, "%Y-%m-%d")
        period_end = selected_time

    metrics = att.get_monthly_metrics(
        selected.year,
        selected.month,
        period_end=period_end,
    )
    employees = _attach_attendance_metrics(employees, metrics)
    summary = _attendance_summary(employees)
    return {
        "status": "ok",
        "today_attendance": {
            "date": target_date,
            "employees": employees,
            **summary,
        },
        "monthly_summary": metrics,
    }


@router.post("/attendance/punch", dependencies=[Depends(get_current_user)])
async def punch_attendance(emp_id: str = "", pin: str = ""):
    """Record a punch-in or punch-out for an employee."""
    global _kpi_cache
    att = _get_attendance()
    success, msg, record = att.punch(emp_id, pin)
    if success:
        _kpi_cache = None
    return {"status": "ok" if success else "error", "message": msg, "record": record}


@router.get("/attendance/self")
async def get_staff_attendance_status(
    emp_id: str = "",
    user=Depends(get_current_user),
):
    """Return the selected employee's attendance snapshot for the business date."""
    employee_id = str(emp_id or "").strip()
    selected_time = operation_now()
    today_str = selected_time.strftime("%Y-%m-%d")
    if not employee_id:
        return {
            "status": "error",
            "message": "Employee ID is required",
            "date": today_str,
        }

    db = get_db()
    try:
        employee_result = (
            q(db, "employees")
            .select("id,name,role,available")
            .eq("id", employee_id)
            .execute()
        )
        if not employee_result.data:
            return {
                "status": "error",
                "message": f"Employee {employee_id} not found",
                "date": today_str,
            }
        employee = employee_result.data[0]
    finally:
        db.close()

    try:
        schedule_response = await get_schedule(date=today_str, days=1)
        schedule = [
            row
            for row in schedule_response.get("schedule", [])
            if row.get("employee_id") == employee_id
        ]
    except Exception:
        schedule = []

    attendance_rows = _get_attendance().get_today_attendance(schedule)
    attendance = attendance_rows[0] if attendance_rows else None
    if attendance and attendance.get("punch_out"):
        next_action = "complete"
    elif attendance and attendance.get("punch_in"):
        next_action = "punch_out"
    elif attendance:
        next_action = "punch_in"
    else:
        next_action = "none"

    return {
        "status": "ok",
        "account": user.get("sub", ""),
        "date": today_str,
        "employee": {
            "id": employee.get("id"),
            "name": employee.get("name"),
            "role": employee.get("role"),
        },
        "attendance": attendance,
        "next_action": next_action,
    }


@router.post("/attendance/correct")
async def correct_attendance(payload: dict, user=Depends(require_manager)):
    """Correct a scheduled attendance record with manager audit metadata."""
    global _kpi_cache
    att = _get_attendance()
    success, msg, record = att.correct_punch(
        payload.get("employee_id", ""),
        payload.get("date", ""),
        payload.get("punch_in", ""),
        payload.get("punch_out", ""),
        payload.get("reason", ""),
        user.get("sub", "manager"),
    )
    if success:
        _kpi_cache = None
    return {"status": "ok" if success else "error", "message": msg, "record": record}


# ======================================================================

@router.get("/attendance/history", dependencies=[Depends(require_manager)])
async def get_attendance_history(date: str = ""):
    """Get attendance for a specific historical date."""
    if not date:
        date = operation_now().strftime("%Y-%m-%d")
    att = _get_attendance()
    schedule = []
    try:
        sched_resp = await get_schedule(date=date, days=1)
        schedule = sched_resp.get("schedule", [])
    except Exception:
        pass
    employees = att.get_date_attendance(date, schedule)
    selected = datetime.strptime(date, "%Y-%m-%d")
    period_end = selected + timedelta(days=1, seconds=-1)
    metrics = att.get_monthly_metrics(
        selected.year,
        selected.month,
        period_end=period_end,
    )
    employees = _attach_attendance_metrics(employees, metrics)
    summary = _attendance_summary(employees)
    return {
        "status": "ok",
        "date": date,
        "employees": employees,
        "monthly_summary": metrics,
        **summary,
    }

# KPI Ranking endpoint (Z-Score + BSC + Cross-Role)
# ======================================================================

import numpy as _np_kpi
_kpi_cache = None

@router.get("/kpi/ranking", dependencies=[Depends(require_manager)])
async def get_kpi_ranking(month: str = ""):
    """Shift+KPI Dashboard Panel 3: Cross-role KPI ranking with Z-Score + BSC."""
    global _kpi_cache
    cache_key = month if month else "current"
    if _kpi_cache and _kpi_cache.get("cache_key") == cache_key:
        return {"status": "ok", "cached": True, "data": _kpi_cache["data"]}

    from kpi.calculator import KPICalculator
    from kpi.collector import KPIDataCollector
    
    year = None
    mon = None
    if month:
        parts = month.split("-")
        if len(parts) == 2:
            year = int(parts[0])
            mon = int(parts[1])

    _np_kpi.random.seed(42)
    collector = KPIDataCollector()
    employees_data = [e for e in collector.collect_monthly(year=year, month=mon) if e.get('role') != 'manager']

    calc = KPICalculator()
    ranked = calc.full_pipeline(employees_data)
    report = calc.generate_report(ranked, month=month if month else None)
    dashboard_kpi = calc.dashboard_format(report)

    _kpi_cache = {"cache_key": cache_key, "data": dashboard_kpi}
    return {"status": "ok", "cached": False, "data": dashboard_kpi}
