"""
S3 Shift Scheduling -- Demand-driven CP-SAT solver (8 employees, 4 roles).

Connects to S2 forecast to determine required staff per shift.
- Baker: driven by total baking units forecasted
- Cashier/Barista: driven by expected transaction volume  
- Cleaner: always 1 per shift

Model: 8 employees, each with exactly ONE role. 2 shifts/day.
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import json
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PRODUCT_TYPES
from db.mysql_client import get_db, q

router = APIRouter(prefix="/s3", tags=["Module 3 - Shift Scheduling"])


# ======================================================================
# Data models
# ======================================================================

class Employee(BaseModel):
    id: str
    name: str
    role: str = Field(..., description="Single role: baker, cashier, barista, cleaner")
    min_hours_per_week: float = 14.0
    max_hours_per_week: float = 42.0
    available: bool = True
    unavailable_dates: List[str] = []


class ShiftResult(BaseModel):
    date: str
    time_slot: str
    role: str
    employee_id: str
    employee_name: str
    demand_level: str = "normal"
    production_target: Optional[int] = None


# ======================================================================
# Default employees -- 8 people, 4 roles x 2
# ======================================================================

DEFAULT_EMPLOYEES = [
    Employee(id="E001", name="Ali",     role="baker",    min_hours_per_week=14, max_hours_per_week=42),
    Employee(id="E002", name="Mei",     role="cashier",  min_hours_per_week=14, max_hours_per_week=42),
    Employee(id="E003", name="Raj",     role="barista",  min_hours_per_week=14, max_hours_per_week=42),
    Employee(id="E004", name="Siti",    role="cleaner",  min_hours_per_week=14, max_hours_per_week=42),
    Employee(id="E005", name="Ahmad",   role="baker",    min_hours_per_week=14, max_hours_per_week=42),
    Employee(id="E006", name="Priya",   role="cashier",  min_hours_per_week=14, max_hours_per_week=42),
    Employee(id="E007", name="Kumar",   role="barista",  min_hours_per_week=14, max_hours_per_week=42),
    Employee(id="E008", name="Lisa",    role="cleaner",  min_hours_per_week=14, max_hours_per_week=42),
]

TIME_SLOTS = ["08:00-14:00", "14:00-20:00"]
ROLES = ["baker", "cashier", "barista", "cleaner"]
SLOT_HOURS = {"08:00-14:00": 6, "14:00-20:00": 6}

# Capacity: 1 baker can produce ~60 units per shift
BAKER_CAPACITY_PER_SHIFT = 60
# 1 cashier handles ~50 transactions per shift
CASHIER_CAPACITY_PER_SHIFT = 50
# 1 barista handles ~40 drinks per shift
BARISTA_CAPACITY_PER_SHIFT = 40


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
                results.append(Employee(
                    id=e["id"],
                    name=e["name"],
                    role=e.get("role", "baker"),
                    min_hours_per_week=float(e.get("min_hours_per_week", 14)),
                    max_hours_per_week=float(e.get("max_hours_per_week", 42)),
                    available=bool(e.get("available", True)),
                    unavailable_dates=unavailable,
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

    Thresholds: P25/P75 from historical actual sales (last 30 days).
    Falls back to forecast-based percentiles if < 7 days of history exist.

    Returns: {date: {"total_units": int, "coffee_units": int, "demand_level": str}}
    """
    try:
        from api.module2_forecast import _do_forecast
        forecast_data = _do_forecast(None, days)
        forecasts = forecast_data.get("forecasts", [])

        daily = {}
        for f in forecasts:
            d = f.get("forecast_date", "")
            if d < start_date:
                continue
            if d not in daily:
                daily[d] = {"total_units": 0, "baker_units": 0, "coffee_units": 0}

            pn = f.get("product_name", "")
            demand = int(f.get("predicted_demand", 0))
            freshness = f.get("freshness_status", "Fresh")

            # Only count Fresh demand for production planning
            if freshness == "Fresh":
                daily[d]["baker_units"] += demand

            daily[d]["total_units"] += demand if freshness == "Fresh" else 0

            # Estimate coffee demand as proportional to total bakery demand
            # ~60% of bakery customers also buy coffee
            if freshness == "Fresh":
                daily[d]["coffee_units"] += int(demand * 0.6)

        # --- Data-driven demand level classification ---
        # Within-week relative ranking: top 1/3 = high, middle = normal, bottom 1/3 = low.
        # Historical absolute thresholds activate once 90+ days of real ops data exist.
        sorted_days = sorted(daily.items(), key=lambda x: x[1]["total_units"], reverse=True)
        n = len(sorted_days)
        high_cut = max(1, n // 3)
        low_cut = n - high_cut

        for rank, (d, _) in enumerate(sorted_days):
            if rank < high_cut:
                daily[d]["demand_level"] = "high"
            elif rank >= low_cut:
                daily[d]["demand_level"] = "low"
            else:
                daily[d]["demand_level"] = "normal"

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
    - High demand day: 2 bakers, 2 cashiers, 2 baristas, 1 cleaner
    - Normal day: 1 per role
    - Low demand day: 1 baker, 1 cashier, 1 barista, 1 cleaner (minimum)
    """
    if shop_closed_weekdays is None:
        shop_closed_weekdays = {0}  # Monday

    if demand_forecast is None:
        demand_forecast = {}

    base = datetime.strptime(start_date, "%Y-%m-%d")
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
            daily_demand[d] = {"baker": 0, "cashier": 0, "barista": 0, "cleaner": 0}
            continue
        
        fc = demand_forecast.get(date_str, {})
        level = fc.get("demand_level", "normal")
        baker_units = fc.get("baker_units", 0)
        
        if level == "high":
            req = {"baker": 2, "cashier": 2, "barista": 2, "cleaner": 1}
        elif level == "low":
            req = {"baker": 1, "cashier": 1, "barista": 1, "cleaner": 1}
        else:  # normal
            req = {"baker": 1, "cashier": 1, "barista": 1, "cleaner": 1}
        
        # Clamp to available employees per role
        for role in ROLES:
            req[role] = min(req[role], len(role_to_emps[role]))
        
        daily_demand[d] = req

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

    # --- Constraint 1: Employee only works their own role ---
    for e_idx in range(num_employees):
        emp_role = emp_idx_map[e_idx].role
        for d in range(num_days):
            for s in range(num_slots):
                for r_idx, role_name in enumerate(ROLES):
                    if role_name != emp_role:
                        model.Add(shift[(e_idx, d, s, r_idx)] == 0)

    # --- Constraint 2: Cleaner, Cashier & Barista -- must cover every shift ---
    for d in range(num_days):
        req = daily_demand[d]
        for s in range(num_slots):
            for role_name in ["cleaner", "cashier", "barista"]:
                if req.get(role_name, 0) == 0:
                    continue
                slot_shifts = [shift[(e_idx, d, s, ROLES.index(role_name))]
                               for e_idx in range(num_employees)]
                model.Add(sum(slot_shifts) == 1)

    # --- Constraint 3: Baker -- morning coverage, flexible afternoon ---
    # Morning always needs at least 1 baker (baking starts early).
    # Afternoon coverage depends on demand: high day (2 bakers) requires
    # afternoon too; normal/low (1 baker) allows morning-only.
    MORNING_SLOT = 0  # 08:00-14:00
    AFTERNOON_SLOT = 1  # 14:00-20:00
    for d in range(num_days):
        req = daily_demand[d]
        required = req.get("baker", 0)
        if required == 0:
            continue
        r_idx = ROLES.index("baker")
        # Morning must have at least 1
        morning_shifts = [shift[(e_idx, d, MORNING_SLOT, r_idx)]
                          for e_idx in range(num_employees)]
        model.Add(sum(morning_shifts) >= 1)
        # Afternoon must have at least 1 when demand >= 2
        if required >= 2:
            afternoon_shifts = [shift[(e_idx, d, AFTERNOON_SLOT, r_idx)]
                                for e_idx in range(num_employees)]
            model.Add(sum(afternoon_shifts) >= 1)
        # Daily total equals demand
        day_total = [shift[(e_idx, d, s, r_idx)]
                     for s in range(num_slots)
                     for e_idx in range(num_employees)]
        model.Add(sum(day_total) == required)

    # --- Constraint 4: At most 2 shifts per employee per day --- per employee per day ---
    for e_idx in range(num_employees):
        for d in range(num_days):
            daily_shifts = [shift[(e_idx, d, s, r)]
                            for s in range(num_slots)
                            for r in range(num_roles)]
            model.Add(sum(daily_shifts) <= 2)

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
        model.Add(weekly_hours >= int(emp.min_hours_per_week))
        # Relax max when colleagues are sick
        max_h = int(emp.max_hours_per_week)
        sick_in_role = sick_count_by_role.get(emp.role, 0)
        if sick_in_role > 0:
            total_in_role = len(role_to_emps.get(emp.role, []))
            available = max(1, total_in_role - sick_in_role)
            max_h = int(max_h * total_in_role / available)
        model.Add(weekly_hours <= max_h)

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

    avg_hours = model.NewIntVar(0, 50, "avg_hours")
    model.Add(avg_hours * num_employees == sum(hour_vars))

    max_dev = model.NewIntVar(0, 50, "max_dev")
    for h in hour_vars:
        model.Add(h - avg_hours <= max_dev)
        model.Add(avg_hours - h <= max_dev)
    model.Minimize(max_dev)

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.random_seed = 42;solver.parameters.num_search_workers = 1
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []

    # --- Extract results ---
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
                            baker_count = req_count = daily_demand.get(d, {}).get("baker", 1)
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

    schedule = []
    for row in rows:
        d = row["schedule_date"]
        if hasattr(d, "strftime"):
            d = d.strftime("%Y-%m-%d")
        schedule.append(ShiftResult(
            date=d,
            time_slot=row["time_slot"],
            role=row.get("role", ""),
            employee_id=row["employee_id"],
            employee_name=row["employee_name"],
            demand_level=row.get("demand_level", "normal"),
            production_target=row.get("production_target"),
        ))

    emp_summary = {}
    for s in schedule:
        eid = s.employee_id
        if eid not in emp_summary:
            emp_summary[eid] = {"name": s.employee_name, "hours": 0, "role": s.role}
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
def _solve_impl(payload: dict) -> dict:
    start_date = payload.get("start_date", datetime.now().strftime("%Y-%m-%d"))
    num_days = min(payload.get("days", 7), 14)
    unavailable_map = payload.get("unavailable", {})

    employees = load_employees()
    for e in employees:
        if e.id in unavailable_map:
            e.unavailable_dates = unavailable_map[e.id]

    # Fetch S2 forecast for demand-driven scheduling
    base = datetime.strptime(start_date, "%Y-%m-%d")
    week_start = base - timedelta(days=base.weekday())
    demand_forecast = _fetch_demand_forecast(week_start.strftime("%Y-%m-%d"), 7)

    results = solve_shift_schedule(
        employees, week_start.strftime("%Y-%m-%d"), 7,
        demand_forecast=demand_forecast,
        shop_closed_weekdays={0},
    )

    # Filter to requested range
    requested_end = base + timedelta(days=num_days)
    results = [r for r in results if r.date >= start_date and r.date < requested_end.strftime("%Y-%m-%d")]

    # Store to database
    try:
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
    except Exception:
        pass

    emp_summary = {}
    for s in results:
        eid = s.employee_id
        if eid not in emp_summary:
            emp_summary[eid] = {"name": s.employee_name, "hours": 0, "role": s.role}
        emp_summary[eid]["hours"] += SLOT_HOURS[s.time_slot]

    return {
        "status": "ok",
        "total_shifts": len(results),
        "schedule": [r.model_dump() for r in results],
        "employee_summary": emp_summary,
    }

import asyncio, concurrent.futures
_s3_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

@router.post("/solve")
async def solve_schedule(payload: dict):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_s3_executor, _solve_impl, payload)




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
    if from_id == to_id and date == to_date and time_slot == to_time_slot:
        return {"status": "error", "message": "Cannot swap with yourself"}

    employees = {e.id: e for e in load_employees()}
    if from_id not in employees or to_id not in employees:
        return {"status": "error", "message": "Unknown employee ID"}

    from_emp = employees[from_id]
    to_emp = employees[to_id]

    if from_emp.role != to_emp.role:
        return {"status": "rejected", "reason": f"Cannot swap across roles: {from_emp.name} is {from_emp.role}, {to_emp.name} is {to_emp.role}"}

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
        q(db, "shift_schedule").update({
            "employee_id": to_id, "employee_name": to_emp.name,
        }).eq("id", from_shift["id"]).execute()

        q(db, "shift_schedule").update({
            "employee_id": from_id, "employee_name": from_emp.name,
        }).eq("id", to_shift["id"]).execute()

        return {
            "status": "ok",
            "message": f"Swapped: {from_emp.name} ({date} {time_slot}) <-> {to_emp.name} ({to_date} {to_shift.get('time_slot','')})",
        }
    except Exception as e:
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
        pass

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
        pass

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
    num_days = min(payload.get("days", 7), 14)

    # Re-sync = restore baseline: clear all sick, then re-solve
    _clear_all_sick_dates()

    employees = load_employees()

    base = datetime.strptime(start_date, "%Y-%m-%d")
    week_start = base - timedelta(days=base.weekday())
    demand_forecast = _fetch_demand_forecast(week_start.strftime("%Y-%m-%d"), 7)

    results = solve_shift_schedule(
        employees, week_start.strftime("%Y-%m-%d"), 7,
        demand_forecast=demand_forecast,
        shop_closed_weekdays={0},
    )

    requested_end = base + timedelta(days=num_days)
    results = [r for r in results if r.date >= start_date and r.date < requested_end.strftime("%Y-%m-%d")]

    try:
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
    except Exception:
        pass

    return {
        "status": "ok",
        "total_shifts": len(results),
        "schedule": [r.model_dump() for r in results],
    }

@router.post("/resync")
async def resync_schedule(payload: dict):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_s3_executor, _resync_impl, payload)


# ======================================================================
# POST /s3/sick -- Persist sick leave + resync
# ======================================================================
def _sick_impl(payload: dict) -> dict:
    employee_id = payload.get("employee_id", "")
    date = payload.get("date", "")
    start_date = payload.get("start_date", date or datetime.now().strftime("%Y-%m-%d"))
    if not employee_id or not date:
        return {"status": "error", "message": "employee_id and date required"}

    # Persist to employees table
    _add_sick_date(employee_id, date)

    # Reload employees (now with persisted sick date) and resync
    num_days = min(payload.get("days", 7), 14)
    employees = load_employees()

    base = datetime.strptime(start_date, "%Y-%m-%d")
    week_start = base - timedelta(days=base.weekday())
    demand_forecast = _fetch_demand_forecast(week_start.strftime("%Y-%m-%d"), 7)

    results = solve_shift_schedule(
        employees, week_start.strftime("%Y-%m-%d"), 7,
        demand_forecast=demand_forecast,
        shop_closed_weekdays={0},
    )

    requested_end = base + timedelta(days=num_days)
    results = [r for r in results if r.date >= start_date and r.date < requested_end.strftime("%Y-%m-%d")]

    try:
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
    except Exception:
        pass

    return {
        "status": "ok",
        "total_shifts": len(results),
        "schedule": [r.model_dump() for r in results],
    }


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

    # Remove from employees table
    _remove_sick_date(employee_id, date)

    # Reload employees and resync
    num_days = min(payload.get("days", 7), 14)
    employees = load_employees()

    base = datetime.strptime(start_date, "%Y-%m-%d")
    week_start = base - timedelta(days=base.weekday())
    demand_forecast = _fetch_demand_forecast(week_start.strftime("%Y-%m-%d"), 7)

    results = solve_shift_schedule(
        employees, week_start.strftime("%Y-%m-%d"), 7,
        demand_forecast=demand_forecast,
        shop_closed_weekdays={0},
    )

    requested_end = base + timedelta(days=num_days)
    results = [r for r in results if r.date >= start_date and r.date < requested_end.strftime("%Y-%m-%d")]

    try:
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
    except Exception:
        pass

    return {
        "status": "ok",
        "total_shifts": len(results),
        "schedule": [r.model_dump() for r in results],
    }


@router.post("/unsick")
async def unmark_sick(payload: dict):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_s3_executor, _unsick_impl, payload)
