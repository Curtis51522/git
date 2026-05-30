# S3 - Gap Analysis, Code Review and Changes

## Requirements vs Current Implementation

| # | Requirement | Current | Status |
|---|------------|---------|:------:|
| 1 | CP-SAT constraint optimization | Implemented, 8 emp x 2 shifts x 4 roles | OK |
| 2 | S2 forecast-driven demand | _fetch_demand_forecast() | OK |
| 3 | Monday closed | shop_closed_weekdays={0} | OK |
| 4 | Employee unavailable dates (sick) | sick/resync/swap full loop | OK |
| 5 | Hour fairness (minimize std dev) | Minimize(max_dev) | OK |
| 6 | DB persistence | shift_schedule table | OK |
| 7 | Multi-skill employees (A = cashier + coffee) | SINGLE role per employee, hardcoded | MISSING |
| 8 | Employee preferences (early/late, weekend) | No field exists | MISSING |
| 9 | Emergency dynamic reschedule (<5s heuristic) | Full CP-SAT re-solve, not heuristic | MISSING |
| 10 | Morning Preparation Checklist | Not implemented | MISSING |
| 11 | Hourly granularity | Only 2 shifts (08-14, 14-20) | MISSING |
| 12 | Peak-hour Heatmap | Not implemented | MISSING |
| 13 | S1 hourly transaction volume driven | Uses S2 demand_level as proxy | PARTIAL |
| 14 | KPI Dashboard (5 metrics) | Not implemented | MISSING |
| 15 | Fairness Index (preference satisfaction) | Hour balance only, no preference tracking | PARTIAL |
| 16 | Push tasks to S4/S5 | Not implemented | MISSING |
| 17 | Swap (same-role, cross-date) | Implemented | OK |

---

## Priority Filter

### Thesis P0 (defense-critical)

| Item | Why fatal |
|------|-----------|
| Multi-skill employees | CP-SAT's core differentiator. Single-role = "How is this different from manual scheduling?" |
| Dynamic reschedule (<5s heuristic) | Requirements explicitly say heuristic, not full re-solve. Different concept entirely |

### Acceptable Gaps (defensible in defense)

| Item | Defense |
|------|---------|
| Morning Checklist | "S3 reads S1 freshness results to generate - architecture in place, implementation trivial" |
| Hourly granularity | "2-shift model is appropriate for a small bakery. CP-SAT supports hourly; just parameter change" |
| Peak-hour Heatmap | "Visualization layer, not algorithmic contribution" |
| KPI Dashboard | "S5 Agent covers operational KPIs; S3 focus is optimization" |
| S1 hourly data | "S2 forecast is forward-looking, more valuable for scheduling than historical transactions" |
| S4/S5 push | "API endpoints available; frontend integration is deployment concern" |
| Employee preferences | "Hour fairness objective achieves similar outcome with fewer parameters" |

---

## Code Review: Engineering Quality

### Pipeline Order: N/A

S3 is a CP-SAT solver module, not a training pipeline. The operational flow:
Load employees -> Fetch S2 forecast -> Build CP-SAT model -> Solve -> Extract -> Persist -> Return
This is correct.

### What's Good

- Pydantic schema validation
- CP-SAT constraint modeling is clean, per-role, well-commented
- Hour fairness via Minimize(max_dev)
- Sick leave persistence + resync loop
- Cross-date swap support
- Monday-closed handling
- Dynamic S2 forecast integration
- ThreadPoolExecutor for async compatibility

### Issues Found

| # | Issue | Severity | Fix |
|---|-------|:--------:|------|
| 1 | DRY violation: _solve_impl, _resync_impl, _sick_impl, _unsick_impl duplicate ~80 lines of identical DB write + solve logic | HIGH | Extract shared _rebuild_and_persist() function |
| 2 | except Exception: pass on ALL DB operations - silent failures | HIGH | Log errors; return degraded status |
| 3 | Zero logging in 811 lines - all output via print() | MEDIUM | Add logging module |
| 4 | Swap uses 2 separate UPDATEs without DB transaction - second failure leaves DB inconsistent | MEDIUM | Wrap in transaction |
| 5 | Semicolon join: random_seed=42;solver.parameters.num_search_workers=1 | LOW | Split to two lines |
| 6 | Comment lies: claims "P25/P75 from historical actual sales" but code does within-week ranking | LOW | Fix comment or implement |
| 7 | Dead code: BAKER_CAPACITY_PER_SHIFT and friends defined but unused | LOW | Remove or use |
| 8 | _add_sick_date / _remove_sick_date duplicate JSON parse logic | LOW | Merge to _update_sick_dates(id, date, action) |
| 9 | No input validation on payload dicts in POST endpoints | LOW | Add Pydantic request models |

---

## Changes

### Fix 1: Extract Shared Rebuild Logic (HIGH)

Four functions (_solve_impl, _resync_impl, _sick_impl, _unsick_impl) share this pattern:
load employees -> fetch forecast -> solve -> filter -> write DB -> return

Extract to one function:

`
def _rebuild_and_persist(start_date, num_days, employees, demand_forecast):
    base = datetime.strptime(start_date, ""%Y-%m-%d"")
    week_start = base - timedelta(days=base.weekday())
    
    results = solve_shift_schedule(
        employees, week_start.strftime(""%Y-%m-%d""), 7,
        demand_forecast=demand_forecast,
        shop_closed_weekdays={0},
    )
    
    requested_end = base + timedelta(days=num_days)
    results = [r for r in results
               if r.date >= start_date
               and r.date < requested_end.strftime(""%Y-%m-%d"")]
    
    _persist_schedule(results, base, num_days)
    return results

def _persist_schedule(results, base, num_days):
    try:
        db = get_db()
        for i in range(num_days):
            d = (base + timedelta(days=i)).strftime(""%Y-%m-%d"")
            q(db, ""shift_schedule"").delete().eq(""schedule_date"", d).execute()
        for r in results:
            q(db, ""shift_schedule"").insert({
                ""schedule_date"": r.date,
                ""time_slot"": r.time_slot,
                ""employee_id"": r.employee_id,
                ""employee_name"": r.employee_name,
                ""role"": r.role,
                ""staff_count"": 1,
                ""demand_level"": r.demand_level,
                ""production_target"": r.production_target,
            }).execute()
    except Exception as e:
        logger.error(""Failed to persist schedule: %s"", e)
        raise

def _rebuild_from_employees(start_date, num_days, employees):
    base = datetime.strptime(start_date, ""%Y-%m-%d"")
    week_start = base - timedelta(days=base.weekday())
    demand_forecast = _fetch_demand_forecast(week_start.strftime(""%Y-%m-%d""), 7)
    return _rebuild_and_persist(start_date, num_days, employees, demand_forecast)
`

Then each _impl function becomes:

`
def _solve_impl(payload):
    start_date, num_days = _parse_solve_payload(payload)
    employees = load_employees_with_unavailable(payload)
    results = _rebuild_from_employees(start_date, num_days, employees)
    return _build_solve_response(results)

def _sick_impl(payload):
    start_date, num_days = _parse_solve_payload(payload)
    _add_sick_date(payload[""employee_id""], payload[""date""])
    employees = load_employees()  # reloads with new sick date
    results = _rebuild_from_employees(start_date, num_days, employees)
    return _build_solve_response(results)
`

### Fix 2: Add Logging (HIGH)

`
import logging
logger = logging.getLogger(""s3.scheduling"")

# In solve_shift_schedule():
logger.info(""Solving schedule: %d employees, %d days, start=%s"",
            num_employees, num_days, start_date)
# After solve:
logger.info(""Solver status: %s, time: %.2fs, shifts: %d"",
            status, solver.WallTime(), len(results))

# In DB operations:
logger.error(""Failed to persist schedule for %s: %s"", date, e)
# Replace all except Exception: pass with:
logger.error(""DB operation failed: %s"", e, exc_info=True)
`

### Fix 3: Swap Transaction (MEDIUM)

`
@router.post(""/swap"")
async def swap_employees(payload: dict):
    ...
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(""START TRANSACTION"")
        try:
            q(db, ""shift_schedule"").update({
                ""employee_id"": to_id, ""employee_name"": to_emp.name,
            }).eq(""id"", from_shift[""id""]).execute()
            q(db, ""shift_schedule"").update({
                ""employee_id"": from_id, ""employee_name"": from_emp.name,
            }).eq(""id"", to_shift[""id""]).execute()
            cur.execute(""COMMIT"")
        except Exception:
            cur.execute(""ROLLBACK"")
            raise
    except Exception as e:
        logger.error(""Swap transaction failed: %s"", e)
        return {""status"": ""error"", ""message"": str(e)}
`

### Fix 4: Dead Code Cleanup (LOW)

`
# Remove unused constants:
# BAKER_CAPACITY_PER_SHIFT = 60  -- DELETE
# CASHIER_CAPACITY_PER_SHIFT = 50  -- DELETE
# BARISTA_CAPACITY_PER_SHIFT = 40  -- DELETE
`

Or use them in demand calculation:

`
# In daily_demand building:
baker_needed = max(1, baker_units // BAKER_CAPACITY_PER_SHIFT)
`

### Fix 5: Fix Comment (LOW)

`
# Before:
# Thresholds: P25/P75 from historical actual sales (last 30 days).
# Falls back to forecast-based percentiles if < 7 days of history exist.

# After:
# Thresholds: Within-week relative ranking via S2 forecast.
# Top 1/3 = high demand, bottom 1/3 = low demand, middle = normal.
# Historical absolute thresholds pending real transaction data.
`

### Fix 6: Semicolon (LOW)

`
# Before:
solver.parameters.random_seed = 42;solver.parameters.num_search_workers = 1

# After:
solver.parameters.random_seed = 42
solver.parameters.num_search_workers = 1
`

---

## Module Comparison

| Dimension | S1 (pre-fix) | S2 | S3 |
|-----------|:---:|:---:|:---:|
| DRY | - | OK | BROKEN (4x duplication) |
| Logging | NO | NO | NO |
| Error handling | Basic | Basic | SILENT (except: pass) |
| Thread safety | - | Cache race | No txn on swap |
| Schema validation | OK | OK | OK |
| Pipeline completeness | BROKEN | OK | N/A (solver) |
| Code organization | 4/10 | 8/10 | 6/10 |

---

## Implementation Order

1. Extract _rebuild_and_persist() + _rebuild_from_employees() (DRY fix)
2. Add logging throughout
3. Wrap swap in DB transaction
4. Fix comment + remove dead code + fix semicolon
5. (Thesis) Multi-skill employee support
6. (Thesis) Dynamic reschedule heuristic
