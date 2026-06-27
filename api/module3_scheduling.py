"""
S3 Shift Scheduling -- Demand-driven CP-SAT solver (9 employees, 4 roles (1 dual-role)).

Connects to S2 forecast to determine required staff per shift.
- Baker: 4 bakers, 2 per slot, 6:00-19:00
- Cashier: 3 cashiers, coverage-driven  
- Barista: 2 baristas, 1 per slot
- Manager: 1 manager, optional per slot

Model: 10 employees, each with exactly ONE role. 2 shifts/day (7h each).
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import json
from ortools.sat.python import cp_model
import logging
from datetime import datetime, timedelta
import sys, os, logging
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import COFFEE_DEMAND_RATIO
from db.mysql_client import get_db, q
import asyncio, concurrent.futures

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
    unavailable_dates: List[str] = []
    is_deputy: bool = False  # deputy manager (baker who covers manager off-shift)
    secondary_roles: List[str] = []  # e.g. barista who can also cashier


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
# Default employees -- 8 people, 4 roles x 2
# ======================================================================

DEFAULT_EMPLOYEES = [
    Employee(id="E001", name="Ali",     role="baker",    min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E002", name="Mei",     role="cashier",  min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E003", name="Raj",     role="barista",  min_hours_per_week=14, max_hours_per_week=56, secondary_roles=["cashier"]),
    Employee(id="E004", name="Siti",    role="baker",    min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E005", name="Ahmad",   role="baker",    min_hours_per_week=14, max_hours_per_week=56, is_deputy=True),
    Employee(id="E006", name="Priya",   role="baker",    min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E007", name="Kumar",   role="baker",    min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E008", name="David",   role="baker",    min_hours_per_week=14, max_hours_per_week=56),
    Employee(id="E009", name="Chen",    role="barista",  min_hours_per_week=14, max_hours_per_week=56, secondary_roles=["cashier"]),
    Employee(id="E010", name="Fatima",  role="manager",  min_hours_per_week=14, max_hours_per_week=42),
]

TIME_SLOTS = ["06:00-13:00", "12:00-19:00"]
ROLES = ["baker", "cashier", "barista", "manager"]
SLOT_HOURS = {"06:00-13:00": 7, "12:00-19:00": 7}
def _baseline_path(week_start):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", f"schedule_baseline_{week_start}.json")



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

def _fetch_demand_forecast(start_date: str, days: int = 7) -> Dict[str, dict]:
    """Fetch S2 forecast, aggregate by date, and compute data-driven demand levels.

    Within-week relative ranking via S2 forecast: top 1/3 = high, bottom 1/3 = low, middle = normal.

    Returns: {date: {"total_units": int, "coffee_units": int, "demand_level": str}}
    """
    try:
        from api.module2_forecast import _do_forecast
        forecast_data = _do_forecast(None, days, start_date=start_date)
        forecasts = forecast_data.get("forecasts", [])

        daily = {}
        for f in forecasts:
            d = f.get("forecast_date", "")
            if d < start_date:
                continue
            if d not in daily:
                daily[d] = {"total_units": 0, "baker_units": 0, "coffee_units": 0}

            demand = int(f.get("predicted_demand", 0))
            freshness = f.get("freshness_status", "Fresh")

            # Only count Fresh demand for production planning
            if freshness in ("Fresh", "Total"):
                daily[d]["baker_units"] += demand

            daily[d]["total_units"] += demand if freshness in ("Fresh", "Total") else 0

            # Estimate coffee demand as proportional to total bakery demand
            # ~60% of bakery customers also buy coffee
            if freshness in ("Fresh", "Total"):
                daily[d]["coffee_units"] += int(demand * COFFEE_DEMAND_RATIO)

        # --- Fixed demand level classification (stable across restarts) ---
        # Thresholds: >=250 high, >=150 normal, >0 low, 0 = closed
        for d, item in daily.items():
            total = item["total_units"]
            if total == 0:
                daily[d]["demand_level"] = "low"  # closed day
            elif total >= 250:
                daily[d]["demand_level"] = "high"
            elif total >= 130:
                daily[d]["demand_level"] = "normal"
            else:
                daily[d]["demand_level"] = "low"

        return daily

    except Exception as e:
        print(f"S2 forecast fetch failed: {e}")
        return {}



def solve_shift_schedule(
    employees: List[Employee],
    start_date: str,
    num_days: int = 7,
    demand_forecast: Optional[Dict[str, dict]] = None,
    shop_closed_weekdays: Optional[set] = None,
) -> List[ShiftResult]:
    """Assign employees to shifts based on demand forecast.

    Per-role requirements per shift come from S2 forecast:
    - High day: 2 bakers/slot (4 total), 2 cashiers, 2 baristas, 1 manager
    - Normal day: 2 bakers/slot, 2 cashiers, 1 barista
    - Low day: 2 bakers/slot, 1 cashier, 1 barista
    """
    if shop_closed_weekdays is None:
        shop_closed_weekdays = {0}  # Monday

    if demand_forecast is None:
        demand_forecast = {}

    base = datetime.strptime(start_date, "%Y-%m-%d")

    # Heuristic fallback: if forecast is empty, use day-of-week pattern
    if not demand_forecast:
        for d in range(num_days):
            dt = base + timedelta(days=d)
            dow = dt.weekday()
            date_str = dt.strftime("%Y-%m-%d")
            if dow == 0:  # Monday closed
                demand_forecast[date_str] = {"total_units": 0, "demand_level": "low"}
            elif dow in (5, 6):  # Sat/Sun = high
                demand_forecast[date_str] = {"total_units": 300, "demand_level": "high"}
            elif dow in (3, 4):  # Thu/Fri = normal
                demand_forecast[date_str] = {"total_units": 220, "demand_level": "normal"}
            else:  # Tue/Wed = low
                demand_forecast[date_str] = {"total_units": 100, "demand_level": "low"}
    emp_list = [e for e in employees if e.available]
    if not emp_list:
        return []

    num_employees = len(emp_list)
    num_slots = len(TIME_SLOTS)
    num_roles = len(ROLES)

    # Group employees by role
    role_to_emps = {role: [] for role in ROLES}
    emp_idx_map = {}
    for idx, e in enumerate(emp_list):
        role_to_emps[e.role].append(idx)
        emp_idx_map[idx] = e

    # Build daily demand requirements
    daily_demand = {}
    for d in range(num_days):
        dt = base + timedelta(days=d)
        date_str = dt.strftime("%Y-%m-%d")
        
        if dt.weekday() in shop_closed_weekdays:
            daily_demand[d] = {"baker": 0, "cashier": 0, "barista": 0, "manager": 0}
            continue
        
        fc = demand_forecast.get(date_str, {})
        level = fc.get("demand_level", "normal")
        if level == "high":
            req = {"baker": 5, "cashier": 2, "barista": 1, "manager": 1, "_level": "high"}
        elif level == "low":
            req = {"baker": 2, "cashier": 1, "barista": 1, "manager": 0, "_level": "low"}
        else:  # normal
            req = {"baker": 3, "cashier": 1, "barista": 1, "manager": 1, "_level": "normal"}  # dual-role allowed
        
        # Clamp to available employees per role (skip manager: 1 person can do both slots)
        for role in ROLES:
            if role == "manager":
                continue  # 1 manager can cover both slots on high days
            # Count primary + secondary role holders
            available = len(role_to_emps[role])
            for e_idx in range(num_employees):
                sec = getattr(emp_idx_map[e_idx], "secondary_roles", []) or []
                if role in sec:
                    available += 1
            req[role] = min(req[role], available)
        
        daily_demand[d] = req
        daily_demand[d]["_level"] = level

    model = cp_model.CpModel()

    # Decision variables
    shift = {}
    for e_idx in range(num_employees):
        for d in range(num_days):
            for s in range(num_slots):
                for r in range(num_roles):
                    shift[(e_idx, d, s, r)] = model.NewBoolVar(
                        f"shift_e{e_idx}_d{d}_s{s}_r{r}"
                    )

    # --- Constraint 1: Employee works primary or secondary roles ---
    for e_idx in range(num_employees):
        emp = emp_idx_map[e_idx]
        allowed = {emp.role}
        sec = getattr(emp, "secondary_roles", []) or []
        allowed.update(sec)
        for d in range(num_days):
            for s in range(num_slots):
                # Block disallowed roles
                for r_idx, role_name in enumerate(ROLES):
                    if role_name not in allowed:
                        model.Add(shift[(e_idx, d, s, r_idx)] == 0)
                # At most 1 role per slot (cannot be barista AND cashier simultaneously)
                slot_roles = [shift[(e_idx, d, s, r)] for r in range(num_roles)]
                model.Add(sum(slot_roles) <= 1)

    # --- Count high-demand days for hour relaxation ---
    high_day_count = sum(1 for d in range(num_days) if daily_demand[d].get("_level") == "high")

    # --- Constraint 2: Frontline coverage per slot ---
    # Raj (dual-role barista+cashier): his barista shift counts as +1 cashier
    # Find Raj index for dual-role cashier bonus
    raj_indices = [e_idx for e_idx in range(num_employees)
                   if "cashier" in (getattr(emp_idx_map[e_idx], "secondary_roles", []) or [])]
    barista_r_idx = ROLES.index("barista")
    cashier_r_idx = ROLES.index("cashier")
    for d in range(num_days):
        req = daily_demand[d]
        is_high = req.get("_level") == "high"
        for s in range(num_slots):
            for role_name in ["cashier", "barista"]:
                if req.get(role_name, 0) == 0:
                    continue
                slot_shifts = [shift[(e_idx, d, s, ROLES.index(role_name))]
                               for e_idx in range(num_employees)]
                if role_name == "cashier":
                    # Raj's barista shift = +1 cashier (disabled on high days)
                    is_high_day = req.get("_level") == "high"
                    raj_barista = sum(shift[(ridx, d, s, barista_r_idx)] for ridx in raj_indices)
                    model.Add(sum(slot_shifts) + raj_barista >= req["cashier"])
                    # At least 1 person assigned cashier role per slot (main counter)
                    dedicated_cashier = [shift[(e_idx, d, s, cashier_r_idx)]
                                        for e_idx in range(num_employees)]
                    model.Add(sum(dedicated_cashier) >= 1)

                elif role_name == "barista":
                    model.Add(sum(slot_shifts) == 1)  # always 1 barista per slot

    # --- High days: NO dual-role (everyone sticks to primary role) ---
    for d in range(num_days):
        if daily_demand[d].get("_level") == "high":
            for e_idx in range(num_employees):
                emp = emp_idx_map[e_idx]
                sec = getattr(emp, "secondary_roles", []) or []
                for s in range(num_slots):
                    for sec_role in sec:
                        r_idx = ROLES.index(sec_role)
                        model.Add(shift[(e_idx, d, s, r_idx)] == 0)

    # --- Manager + Deputy coverage ---
    mgr_r_idx = ROLES.index("manager")
    baker_r_idx = ROLES.index("baker")
    # Find deputy baker
    deputy_indices = [e_idx for e_idx in range(num_employees)
                      if getattr(emp_idx_map[e_idx], "is_deputy", False)]
    for d in range(num_days):
        req = daily_demand[d]
        if req.get("manager", 0) == 0:
            continue
        is_high_day = req.get("_level") == "high"
        # Manager present per daily_demand (1 slot normal, 2 slots high)
        day_mgr = [shift[(e_idx, d, s, mgr_r_idx)]
                   for s in range(num_slots)
                   for e_idx in range(num_employees)]
        mgr_needed = min(req.get("manager", 1), num_slots)  # 1 or 2 slots
        model.Add(sum(day_mgr) >= mgr_needed)
        # Deputy baker must also work >= 1 slot/day
        for didx in deputy_indices:
            day_dep = [shift[(didx, d, s, baker_r_idx)]
                       for s in range(num_slots)]
            model.Add(sum(day_dep) >= 1)
            # Each slot: manager OR deputy present
            for s in range(num_slots):
                slot_mgr = [shift[(e_idx, d, s, mgr_r_idx)]
                            for e_idx in range(num_employees)]
                model.Add(sum(slot_mgr) + shift[(didx, d, s, baker_r_idx)] >= 1)
            # Non-high days: manager and deputy in DIFFERENT slots
            # High days: they CAN overlap (deputy is a baker, all-hands)
            if not is_high_day:
                for s in range(num_slots):
                    slot_mgr = [shift[(e_idx, d, s, mgr_r_idx)]
                                for e_idx in range(num_employees)]
                    model.Add(sum(slot_mgr) + shift[(didx, d, s, baker_r_idx)] <= 1)

    # --- Constraint 3: Baker -- demand-driven per slot ---
    # Low/normal: 2 bakers/slot. High: all 4 bakers/slot.
    MORNING_SLOT = 0  # 06:00-13:00
    AFTERNOON_SLOT = 1  # 12:00-19:00
    for d in range(num_days):
        req = daily_demand[d]
        bakers_per_slot = req.get("baker", 0)
        if bakers_per_slot == 0:
            continue
        r_idx = ROLES.index("baker")
        baker_count = len(role_to_emps.get("baker", []))
        actual = min(bakers_per_slot, baker_count)
        morning_shifts = [shift[(e_idx, d, MORNING_SLOT, r_idx)]
                          for e_idx in range(num_employees)]
        model.Add(sum(morning_shifts) == actual)
        afternoon_shifts = [shift[(e_idx, d, AFTERNOON_SLOT, r_idx)]
                            for e_idx in range(num_employees)]
        model.Add(sum(afternoon_shifts) == actual)
        # Daily total: bakers_per_slot * 2
        day_total = [shift[(e_idx, d, s, r_idx)]
                     for s in range(num_slots)
                     for e_idx in range(num_employees)]
        model.Add(sum(day_total) == actual * 2)
    # --- Constraint 4: At most 2 shifts per employee per day ---
    for e_idx in range(num_employees):
        for d in range(num_days):
            daily_shifts = [shift[(e_idx, d, s, r)]
                            for s in range(num_slots)
                            for r in range(num_roles)]
            model.Add(sum(daily_shifts) <= 2)

    # --- Constraint 4b: On non-high days, same employee cannot work both slots of the same role ---
    for d in range(num_days):
        if daily_demand[d].get("baker", 0) < 2:
            for r_idx in range(num_roles):
                for e_idx in range(num_employees):
                    model.Add(shift[(e_idx, d, 0, r_idx)] + shift[(e_idx, d, 1, r_idx)] <= 1)

    # --- Constraint 5: Weekly hours bounds (relaxed when colleagues sick) ---
    # Count unavailable employees per role
    sick_count_by_role = {}
    for e_idx in range(num_employees):
        emp = emp_idx_map[e_idx]
        if emp.unavailable_dates:
            sick_count_by_role[emp.role] = sick_count_by_role.get(emp.role, 0) + 1
    for e_idx in range(num_employees):
        emp = emp_idx_map[e_idx]
        weekly_hours = sum(
            shift[(e_idx, d, s, r)] * SLOT_HOURS[TIME_SLOTS[s]]
            for d in range(num_days) for s in range(num_slots) for r in range(num_roles)
        )
        # REMOVED: no lower limit
        # model.Add(weekly_hours >= int(emp.min_hours_per_week))
        # Relax max when colleagues are sick or during high-demand weeks
        max_h = int(emp.max_hours_per_week)
        sick_in_role = sick_count_by_role.get(emp.role, 0)
        if sick_in_role > 0:
            total_in_role = len(role_to_emps.get(emp.role, []))
            available = max(1, total_in_role - sick_in_role)
            max_h = int(max_h * total_in_role / available)
        # High-demand days: relax max for all roles
        if high_day_count > 1:
            if emp.role == "cashier":
                max_h = min(max_h + high_day_count * 7, 56)
            elif emp.role in ("barista", "baker", "manager"):
                max_h = min(max_h + high_day_count * 7, 56)
        # REMOVED: no upper limit
        # model.Add(weekly_hours <= max_h)

    # --- Constraint 6: Unavailable dates ---
    for e_idx in range(num_employees):
        emp = emp_idx_map[e_idx]
        for d in range(num_days):
            dt = base + timedelta(days=d)
            if dt.strftime("%Y-%m-%d") in emp.unavailable_dates:
                for s in range(num_slots):
                    for r in range(num_roles):
                        model.Add(shift[(e_idx, d, s, r)] == 0)

    # --- Objective: balance hours ---
    hour_vars = []
    for e_idx in range(num_employees):
        h = sum(
            shift[(e_idx, d, s, r)] * SLOT_HOURS[TIME_SLOTS[s]]
            for d in range(num_days) for s in range(num_slots) for r in range(num_roles)
        )
        hour_vars.append(h)

    avg_hours = model.NewIntVar(0, 56, "avg_hours")
    # Floor division: avg_hours = floor(sum / num_employees)
    model.Add(avg_hours * num_employees <= sum(hour_vars))
    model.Add(sum(hour_vars) < (avg_hours + 1) * num_employees)

    max_dev = model.NewIntVar(0, 56, "max_dev")
    for h in hour_vars:
        model.Add(h - avg_hours <= max_dev)
        model.Add(avg_hours - h <= max_dev)
    model.Minimize(max_dev)

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30  # increased for larger model
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger = logging.getLogger("s3.solver")
        logger.error("Solver status: %s (OPTIMAL=%s FEASIBLE=%s INFEASIBLE=%s)",
                     solver.StatusName(status), cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.INFEASIBLE)
        return []  # INFEASIBLE: no valid schedule
    results = []
    for d in range(num_days):
        dt = base + timedelta(days=d)
        if dt.weekday() in shop_closed_weekdays:
            continue
        date_str = dt.strftime("%Y-%m-%d")
        fc = demand_forecast.get(date_str, {})
        level = fc.get("demand_level", "normal")
        
        baker_target = None
        for s in range(num_slots):
            for r_idx, role_name in enumerate(ROLES):
                for e_idx in range(num_employees):
                    if solver.Value(shift[(e_idx, d, s, r_idx)]) == 1:
                        emp = emp_idx_map[e_idx]
                        prod_target = None
                        if role_name == "baker" and s == 0:
                            # Morning baker gets production target
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
@router.get("/schedule")
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


# ======================================================================
# POST /s3/solve -- Demand-driven generation
# ======================================================================


def _save_baseline(results, week_start_str=None):
    """Save per-employee shift counts as baseline after solve."""
    baseline = {}
    for r in results:
        eid = r.employee_id
        if eid not in baseline:
            baseline[eid] = {"name": r.employee_name, "role": r.role, "shifts": 0}
        baseline[eid]["shifts"] += 1
    try:
        bp = _baseline_path(week_start_str or "unknown")
        os.makedirs(os.path.dirname(os.path.abspath(bp)), exist_ok=True)
        with open(os.path.abspath(bp), 'w') as f:
            json.dump({"week_start": week_start_str or "", "employees": baseline}, f, indent=2)
    except Exception:
        pass

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
        return d < datetime.now().date()
    except Exception:
        return False

def _persist_schedule(results, base, num_days):
    db = get_db()
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


def _rebuild_from_employees(start_date, num_days, employees):
    base = datetime.strptime(start_date, "%Y-%m-%d")
    demand_forecast = _fetch_demand_forecast(start_date, num_days)
    results = solve_shift_schedule(
        employees, start_date, num_days,
        demand_forecast=demand_forecast,
        shop_closed_weekdays={0},
    )
    requested_end = base + timedelta(days=num_days)
    results = [r for r in results if r.date >= start_date and r.date < requested_end.strftime("%Y-%m-%d")]
    try:
        _persist_schedule(results, base, num_days)
        _save_baseline(results, start_date)
    except Exception:
        logger.error("Failed to persist schedule: %s", sys.exc_info())
        raise
    pass  # KPI saved by _compute_kpi
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

def _solve_impl(payload: dict) -> dict:
    start_date = payload.get("start_date", datetime.now().strftime("%Y-%m-%d"))
    if _is_past_date(start_date):
        return {"status": "error", "message": "Cannot modify past schedules. Select today or a future date."}
    num_days = min(payload.get("days", 7), 14)
    unavailable_map = payload.get("unavailable", {})

    employees = load_employees()
    for e in employees:
        if e.id in unavailable_map:
            e.unavailable_dates = unavailable_map[e.id]

    results = _rebuild_from_employees(start_date, num_days, employees)
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


@router.post("/solve")
async def solve_schedule(payload: dict):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_s3_executor, _solve_impl, payload)




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
@router.post("/swap")
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

    # Skill-based swap check: each must be able to do the other's assigned role
    from_skills = {from_emp.role} | set(getattr(from_emp, "secondary_roles", []) or [])
    to_skills = {to_emp.role} | set(getattr(to_emp, "secondary_roles", []) or [])
    from_role = from_shift.get("role", "")
    to_role = to_shift.get("role", "")
    if to_role not in from_skills:
        return {"status": "rejected", "reason": f"{from_emp.name} cannot take {to_emp.name}'s {to_role} shift (skills: {sorted(from_skills)})"}
    if from_role not in to_skills:
        return {"status": "rejected", "reason": f"{to_emp.name} cannot take {from_emp.name}'s {from_role} shift (skills: {sorted(to_skills)})"}

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

    if to_date in to_emp.unavailable_dates:
        return {"status": "rejected", "reason": f"{to_emp.name} is unavailable on {to_date}"}

    try:
        db = get_db()
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
    start_date = payload.get("start_date", datetime.now().strftime("%Y-%m-%d"))
    if _is_past_date(start_date):
        return {"status": "error", "message": "Cannot modify past schedules. Select today or a future date."}
    num_days = min(payload.get("days", 7), 14)
    _clear_all_sick_dates()
    employees = load_employees()
    results = _rebuild_from_employees(start_date, num_days, employees)
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
    pass  # KPI saved by _compute_kpi
    return _build_schedule_response(schedule_list)


@router.post("/resync")
async def resync_schedule(payload: dict):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_s3_executor, _resync_impl, payload)


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
                "replaced_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    start_date = payload.get("start_date", date or datetime.now().strftime("%Y-%m-%d"))
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
    pass  # KPI saved by _compute_kpi
    return _build_schedule_response(schedule_list)


@router.post("/sick")
async def mark_sick(payload: dict):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_s3_executor, _sick_impl, payload)


# ======================================================================
# POST /s3/unsick -- Remove sick leave + resync
# ======================================================================
def _unsick_impl(payload: dict) -> dict:
    employee_id = payload.get("employee_id", "")
    date = payload.get("date", "")
    start_date = payload.get("start_date", date or datetime.now().strftime("%Y-%m-%d"))
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


@router.post("/unsick")
async def unmark_sick(payload: dict):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_s3_executor, _unsick_impl, payload)

# ======================================================================
# GET /s3/kpi -- Scheduling KPIs
# ======================================================================

@router.get("/kpi")
async def get_kpi(
    start_date: str = Query(None),
    days: int = Query(7, ge=1, le=14),
):
    """Compute coverage, fairness, and compliance KPIs for the schedule."""
    try:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        base = datetime.strptime(start_date, "%Y-%m-%d")
        # Allow KPI queries for any date range (simulated timeline)
        end_date = (base + timedelta(days=days - 1)).strftime("%Y-%m-%d")

        db = get_db()
        r = q(db, "shift_schedule").select("*")\
            .gte("schedule_date", start_date)\
            .lte("schedule_date", end_date)\
            .order("schedule_date,time_slot").execute()
        rows = r.data if r.data else []

        if not rows:
            n_working = sum(1 for i in range(days) if (base + timedelta(days=i)).weekday() != 0)
            return {
                "status": "ok",
                "period": {"start": start_date, "end": end_date, "days": days, "working_days": n_working},
                "summary": {"total_shifts": 0, "total_hours": 0, "avg_hours_per_emp": 0, "employees_scheduled": 0},
                "coverage": [],
                "fairness": {},
                "compliance": {"violations": [], "all_pass": True},
                "message": "No schedule for this period. Run /s3/solve first.",
            }

        # ---- working days (exclude Monday) ----
        expected_dates = set()
        for i in range(days):
            d = base + timedelta(days=i)
            if d.weekday() != 0:
                expected_dates.add(d.strftime("%Y-%m-%d"))
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
        # Check if anyone has unavailable_dates in this period
        start_date_obj = base.date()
        end_date_obj = (base + timedelta(days=days - 1)).date()
        # Build per-employee coverage: Expected = baseline, Actual = current schedule
        # Load baseline (original schedule) for expected shift counts
        baseline_map = {}
        baseline_path = _baseline_path(start_date)
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path) as f:
                    bl = json.load(f)
                for eid, edata in bl.get("employees", {}).items():
                    baseline_map[eid] = edata.get("shifts", 0)
            except Exception:
                pass
        for e in all_emps:
            eid = e.id
            info = emp_hours.get(eid, {"shifts": 0})
            actual = info.get("shifts", 0)
            if actual < 0:
                actual = 0
            # Has unavailable dates this week? expected = baseline, rate shows gap
            # Otherwise: expected = actual, rate always 100%
            has_unavail = any(
                start_date_obj <= datetime.strptime(d, "%Y-%m-%d").date() <= end_date_obj
                for d in (e.unavailable_dates or [])
            ) if e.unavailable_dates else False
            baseline_shifts = baseline_map.get(eid, 0)
            if has_unavail:
                expected = baseline_shifts
                rate = round(actual / expected * 100, 1) if expected > 0 else 0.0
            else:
                expected = actual
                rate = 100.0
            extra = max(0, actual - baseline_shifts)
            coverage.append({
                "employee": e.name,
                "role": e.role,
                "filled": actual,
                "expected": expected,
                "rate_pct": rate,
                "extra_shifts": extra,
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
        employees = load_employees()
        emp_lookup = {e.id: e for e in employees}
        violations = []

        # Count high-demand days for hour relaxation (same logic as solver)
        high_dates = set()
        for row2 in rows:
            if row2.get("demand_level") == "high":
                d = row2["schedule_date"]
                high_dates.add(d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d))
        kpi_high_days = len(high_dates)

        for eid, info in emp_hours.items():
            emp = emp_lookup.get(eid)
            if not emp:
                continue


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
# GET /s3/prep_checklist -- Morning preparation checklist
# ======================================================================
@router.get("/prep_checklist")
async def get_prep_checklist():
    """Return Day-1 batches that still need to be moved from Fresh Area to Discount Area."""
    db = get_db()
    r = q(db, "batch_inventory").select("*").eq("freshness_status", "Day-1").gt("quantity", 0).execute()
    items = []
    for row in (r.data or []):
        if row.get("sales_area", "Fresh Area") != "Day-1 Area":
            items.append({
                "batch_id": row.get("batch_id"),
                "product_name": row.get("product_name"),
                "quantity": row.get("quantity", 0),
                "production_time": row.get("production_time", ""),
                "action": "Move to Discount Area (cashier display cabinet)",
            })
    return {
        "status": "ok",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "items": items,
        "acknowledged": False,
    }


# ======================================================================
# POST /s3/prep_acknowledge -- Acknowledge preparation checklist complete
# ======================================================================
@router.post("/prep_acknowledge")
async def acknowledge_prep():
    """Mark all Day-1 batches as moved to Discount Area."""
    db = get_db()
    day1_batches = q(db, "batch_inventory").select("*").eq("freshness_status", "Day-1").gt("quantity", 0).execute()
    updated = 0
    for row in (day1_batches.data or []):
        if row.get("sales_area", "Fresh Area") != "Day-1 Area":
            q(db, "batch_inventory").update({"sales_area": "Day-1 Area"}).eq("batch_id", row["batch_id"]).execute()
            updated += 1
    return {
        "status": "ok",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "batches_updated": updated,
        "message": f"Prep checklist acknowledged. {updated} batches moved to Discount Area.",
    }



# ======================================================================
# S3 Production Scheduler endpoints (from s3_scheduling/scheduler.py)
# ======================================================================
@router.get("/plan/7day")
async def get_7day_production_plan(date: str = None):
    """
    Forecasting Dashboard Panel 2: 7-day production plan grid.
    Uses real S2 quantile models for demand prediction.
    Query: GET /s3/plan/7day?date=2026-06-30
    """
    import os as _os3, sys as _sys3, json as _json3
    _base3 = _os3.path.dirname(_os3.path.dirname(_os3.path.abspath(__file__)))
    if _base3 not in _sys3.path:
        _sys3.path.insert(0, _base3)
    from s3_scheduling.scheduler import Scheduler, generate_7day_s2_forecast
    import numpy as np
    from datetime import datetime as _dt, timedelta as _td

    if date is None:
        date = _dt.now().strftime("%Y-%m-%d")
    start_dt = _dt.strptime(date, "%Y-%m-%d")
    if start_dt.weekday() != 0:
        start_dt -= _td(days=start_dt.weekday())
    start_date = start_dt.strftime("%Y-%m-%d")

    s = Scheduler()
    day1_stock = {p: np.random.randint(0, 5) for p in s.breads}
    forecast = generate_7day_s2_forecast(start_date)
    result = s.generate_7day_plan(start_date, day1_stock, forecast)

    return {
        "status": "ok",
        "generated_at": _dt.now().isoformat(),
        "dashboard_7day": result["dashboard_7day"],
        "weekly_summary": {
            "total_bake": result["weekly_summary"]["total_bake"],
            "total_profit": result["weekly_summary"]["total_profit"],
            "total_revenue": result["weekly_summary"]["total_revenue"],
            "daily_profits": result["weekly_summary"]["daily_profits"],
            "scenarios": result["weekly_summary"]["scenarios"],
            "top_products": result["weekly_summary"]["top_products"],
        },
    }


@router.get("/materials")
async def get_materials_procurement(date: str = None):
    """
    Forecasting Dashboard Panel 3: Raw material procurement list.
    Query: GET /s3/materials?date=2026-06-30
    """
    import os as _os4, sys as _sys4
    _base4 = _os4.path.dirname(_os4.path.dirname(_os4.path.abspath(__file__)))
    if _base4 not in _sys4.path:
        _sys4.path.insert(0, _base4)
    from s3_scheduling.scheduler import Scheduler, generate_7day_s2_forecast
    import numpy as np
    from datetime import datetime as _dt2, timedelta as _td2

    if date is None:
        date = _dt2.now().strftime("%Y-%m-%d")
    start_dt = _dt2.strptime(date, "%Y-%m-%d")
    if start_dt.weekday() != 0:
        start_dt -= _td2(days=start_dt.weekday())
    start_date = start_dt.strftime("%Y-%m-%d")

    s = Scheduler()
    day1_stock = {p: np.random.randint(0, 5) for p in s.breads}
    forecast = generate_7day_s2_forecast(start_date)
    result = s.generate_7day_plan(start_date, day1_stock, forecast)

    return {
        "status": "ok",
        "generated_at": _dt2.now().isoformat(),
        "dashboard_materials": result["dashboard_materials"],
    }


@router.get("/eval")
async def get_paper_evaluation():
    """
    Paper evaluation: S3 on test set with real lag features.
    Query: GET /s3/eval
    Note: Takes ~30 seconds, caches result for subsequent calls.
    """
    import os as _os5, sys as _sys5, json as _json5
    _base5 = _os5.path.dirname(_os5.path.dirname(_os5.path.abspath(__file__)))
    if _base5 not in _sys5.path:
        _sys5.path.insert(0, _base5)

    _eval_cache = _os5.path.join(_base5, "s3_scheduling", "outputs", "paper_eval.json")

    # Return cached result if available
    if _os5.path.exists(_eval_cache):
        with open(_eval_cache, "r") as f:
            cached = _json5.load(f)
        return {"status": "ok", "cached": True, **cached}

    # Run fresh evaluation
    from s3_scheduling.scheduler import run_paper_evaluation
    result = run_paper_evaluation(save_path=_eval_cache)
    if result is None:
        return {"status": "error", "message": "Evaluation failed. Check S2 models."}
    return {"status": "ok", "cached": False, **result}
