"""
KPI Integrated Demo - Robust Z + BSC Aggregation
=================================================
Pipeline: Attendance + Shift + Collector + Calculator -> Dashboard JSON

3 panels:
  Panel 1 - Today Attendance (punch card real-time)
  Panel 2 - Weekly Shift (7-day roster grid)
  Panel 3 - Monthly KPI Ranking (Robust Z x BSC x cross-role)

Usage:
  python kpi/demo.py
  python kpi/demo.py --save kpi/outputs/dashboard.json
"""

import sys, os, json, argparse, numpy as np
from datetime import datetime, timedelta

_PAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PAR_DIR not in sys.path:
    sys.path.insert(0, _PAR_DIR)

from kpi.calculator import KPICalculator
from kpi.attendance import AttendanceSystem
from kpi.shift import ShiftScheduler
from kpi.collector import KPIDataCollector
from kpi.config import ROLES, BSC_WEIGHTS

np.random.seed(42)

EMPLOYEES = [
    {"id": "E001", "name": "Zhang Wei", "role": "baker",   "pin": "1001"},
    {"id": "E002", "name": "Li Na",     "role": "cashier", "pin": "1002"},
    {"id": "E003", "name": "Wang Lei",  "role": "barista", "pin": "1003", "role2": "cashier"},
    {"id": "E004", "name": "Liu Yang",  "role": "baker",   "pin": "1004"},
    {"id": "E005", "name": "Chen Hao",  "role": "baker",   "pin": "1005"},
    {"id": "E006", "name": "Zhao Min",  "role": "baker",   "pin": "1006"},
    {"id": "E007", "name": "Huang Jian","role": "baker",   "pin": "1007"},
    {"id": "E008", "name": "Wu Tao",    "role": "baker",   "pin": "1008"},
    {"id": "E009", "name": "Lin Yue",   "role": "barista", "pin": "1009", "role2": "cashier"},
    {"id": "E010", "name": "Sun Jie",   "role": "manager", "pin": "1010"},
]


def run_full_integrated_demo(save_path=None):
    print("=" * 70)
    print("  KPI INTEGRATED DEMO - Robust Z + BSC")
    print("=" * 70)

    # Module 1: Attendance
    print("\n[1/5] Attendance System")
    att = AttendanceSystem()
    for emp in EMPLOYEES:
        att.register_employee(emp["id"], emp["name"], emp["role"], emp["pin"])

    today = datetime.now().strftime("%Y-%m-%d")
    for emp in EMPLOYEES:
        late_minutes = np.random.choice([-10, -5, 0, 3, 8, 18, 45], p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05])
        punch_in = (datetime.now().replace(hour=8, minute=0, second=0) + timedelta(minutes=int(late_minutes)))
        record = {
            "id": emp["id"], "name": emp["name"], "role": emp["role"],
            "date": today,
            "punch_in": punch_in.strftime("%H:%M"),
            "punch_out": None,
            "status": "on_time" if late_minutes <= 0 else ("late" if late_minutes <= 15 else "absent"),
        }
        att.records.append(record)

    today_att = att.get_today_attendance()
    present_count = sum(1 for e in today_att if e["status"] != "absent")
    print(f"  Today: {present_count}/{len(EMPLOYEES)} present")

    # Module 2: Shift Scheduler
    print("\n[2/5] Shift Scheduler")
    scheduler = ShiftScheduler()
    for emp in EMPLOYEES:
        scheduler.add_employee(emp["id"], emp["name"], emp["role"])
    week_grid = scheduler.get_weekly_grid()
    coverage = week_grid["coverage"]
    gap_count = len(coverage["issues"])
    print(f"  Coverage: {'COMPLETE' if coverage['complete'] else str(gap_count) + ' gaps'}")

    # Module 3: KPI Data Collector
    print("\n[3/5] KPI Data Collector")
    collector = KPIDataCollector()
    employees_data = collector.collect_monthly()
    print(f"  Collected KPIs for {len(employees_data)} employee entries")

    # Module 4: KPI Calculator
    print("\n[4/5] KPI Calculator - Robust Z -> BSC -> Cross-Role Ranking")
    calc = KPICalculator()
    ranked = calc.full_pipeline(employees_data)
    report = calc.generate_report(ranked)
    dashboard_kpi = calc.dashboard_format(report)
    if report["top_performer"]:
        print(f"  Top: {report['top_performer']['name']} ({report['top_performer']['role']}) score={report['top_performer']['score']:.4f}")

    # Module 5: Trends
    print("\n[5/5] Trend Generator")
    trends = collector.generate_trend_data(months=6)

    # Assemble dashboard
    dashboard = {
        "generated_at": datetime.now().isoformat(),
        "panel_1_attendance": att.dashboard_format()["today_attendance"],
        "panel_2_shift": {
            "week_start": week_grid["week_start"],
            "week_end": week_grid["week_end"],
            "employees": [
                {"name": e["name"], "role": e["role"], "schedule": e["shifts"]}
                for e in week_grid["grid"]
            ],
            "coverage_ok": coverage["complete"],
        },
        "panel_3_kpi": dashboard_kpi["kpi_summary"],
        "role_breakdown": dashboard_kpi["role_breakdown"],
        "trends": [
            {"month": t["month"], "avg_score": t["avg_score"],
             "top": t["top_performer"]["name"] if t["top_performer"] else "-"}
            for t in trends
        ],
    }

    # Summary
    print("\n" + "=" * 70)
    print("  DASHBOARD SUMMARY")
    print("=" * 70)
    print(f"\n  Panel 1 - Today Attendance ({today}):")
    print(f"    Present: {dashboard['panel_1_attendance']['present']}  "
          f"Late: {dashboard['panel_1_attendance']['late']}  "
          f"Absent: {dashboard['panel_1_attendance']['absent']}")

    print(f"\n  Panel 3 - Monthly KPI Ranking:")
    print(f"    {'Rank':<6} {'Name':<14} {'Role':<10} {'Score':<10} {'%ile'}")
    print(f"    {'-'*6} {'-'*14} {'-'*10} {'-'*10} {'-'*6}")
    for e in dashboard["panel_3_kpi"]["ranking"]:
        ev = f' [{e.get("evaluated_as","")}]' if e.get("evaluated_as") else ''
        print(f"    {e['rank']:<6} {e['name']:<14} {e['role']+ev:<10} {e['score']:>+8.4f}  {e['percentile']:>5.1f}%")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Saved: {save_path}")

    print("\n" + "=" * 70)
    print("  KPI Pipeline Complete")
    print("=" * 70)
    return dashboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    run_full_integrated_demo(args.save)
