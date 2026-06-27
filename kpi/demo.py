'''KPI Integrated Demo — All 6 Modules Connected
=================================================
Full pipeline: Attendance + Shift + AHP + Collector + Calculator → Dashboard JSON

Matches dashboard-designs.md Shift+KPI Dashboard:
  Panel 1 — Today Attendance (punch card real-time)
  Panel 2 — Weekly Shift (7-day roster grid)
  Panel 3 — Monthly KPI Ranking (Z-Score x BSC x cross-role)

Usage:
  python kpi/demo.py
  python kpi/demo.py --save kpi/outputs/dashboard.json
'''

import sys, os, json, argparse
import numpy as np
from datetime import datetime, timedelta

_PAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PAR_DIR not in sys.path:
    sys.path.insert(0, _PAR_DIR)

from kpi.calculator import KPICalculator
from kpi.attendance import AttendanceSystem
from kpi.shift import ShiftScheduler
from kpi.ahp import AHPCalculator
from kpi.collector import KPIDataCollector
from kpi.config import ROLES, BSC_WEIGHTS

np.random.seed(42)

# ============================================================
# EMPLOYEE ROSTER
# ============================================================
EMPLOYEES = [
    {"id": "B001", "name": "Zhang Wei", "role": "baker",   "pin": "1001"},
    {"id": "B002", "name": "Li Ming",   "role": "baker",   "pin": "1002"},
    {"id": "B003", "name": "Wang Fang", "role": "baker",   "pin": "1003"},
    {"id": "R001", "name": "Chen Yu",   "role": "barista", "pin": "2001"},
    {"id": "R002", "name": "Liu Na",    "role": "barista", "pin": "2002"},
    {"id": "R003", "name": "Huang Li",  "role": "barista", "pin": "2003"},
    {"id": "C001", "name": "Zhou Jie",  "role": "cashier", "pin": "3001"},
    {"id": "C002", "name": "Wu Min",    "role": "cashier", "pin": "3002"},
    {"id": "M001", "name": "Sun Tao",   "role": "manager", "pin": "4001"},
]


def run_full_integrated_demo(save_path=None):
    '''Run all 6 KPI modules end-to-end and return dashboard-ready dict.'''
    print("=" * 70)
    print("  KPI INTEGRATED DEMO — 6 Modules Pipeline")
    print("=" * 70)

    # ── Module 1: Attendance System ──
    print("\n[1/6] Attendance System — register + punch simulation")
    att = AttendanceSystem()
    for emp in EMPLOYEES:
        att.register_employee(emp["id"], emp["name"], emp["role"], emp["pin"])

    today = datetime.now().strftime("%Y-%m-%d")
    for emp in EMPLOYEES:
        late_minutes = np.random.choice([-10, -5, 0, 3, 8, 18, 45], p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05])
        punch_in = (datetime.now().replace(hour=8, minute=0, second=0) + timedelta(minutes=int(late_minutes)))
        record = {
            "id": emp["id"],
            "name": emp["name"],
            "role": emp["role"],
            "date": today,
            "punch_in": punch_in.strftime("%H:%M"),
            "punch_out": None,
            "status": "on_time" if late_minutes <= 0 else ("late" if late_minutes <= 15 else "absent"),
        }
        att.records.append(record)

    today_att = att.get_today_attendance()
    present_count = sum(1 for e in today_att if e["status"] != "absent")
    print(f"  Today: {present_count}/{len(EMPLOYEES)} present")

    # ── Module 2: Shift Scheduler ──
    print("\n[2/6] Shift Scheduler — 7-day auto roster")
    scheduler = ShiftScheduler()
    for emp in EMPLOYEES:
        scheduler.add_employee(emp["id"], emp["name"], emp["role"])
    week_grid = scheduler.get_weekly_grid()
    coverage = week_grid["coverage"]
    gap_count = len(coverage["issues"])
    print(f"  Coverage: {'COMPLETE' if coverage['complete'] else str(gap_count) + ' gaps'}")
    if coverage["issues"]:
        for issue in coverage["issues"][:3]:
            print(f"    GAP: {issue['date']} {issue['shift']} {issue['role']} ({issue['assigned']}/{issue['required']})")

    # ── Module 3: AHP Weight Calibration ──
    print("\n[3/6] AHP — BSC dimension weight calibration")
    ahp = AHPCalculator()
    bsc_result = ahp.bsc_dimension_weights()
    print(f"  BSC weights: {bsc_result['dimension_weights']}")
    print(f"  Consistency: CR={bsc_result['CR']:.4f} ({'PASS' if bsc_result['consistent'] else 'FAIL — needs recalibration'})")

    if not bsc_result["consistent"]:
        print("  Recalibrating...")
        recalibrated = [
            [1, 1, 1/2, 2],
            [1, 1, 1/2, 2],
            [2, 2, 1, 3],
            [1/2, 1/2, 1/3, 1],
        ]
        bsc_result = ahp.bsc_dimension_weights(custom_comparisons=recalibrated)
        print(f"  Recalibrated CR={bsc_result['CR']:.4f}")

    # ── Module 4: KPI Data Collector ──
    print("\n[4/6] KPI Data Collector — gathering raw values")
    collector = KPIDataCollector()
    employees_data = collector.collect_monthly()
    print(f"  Collected KPIs for {len(employees_data)} employees")

    # ── Module 5: KPI Calculator (3-step pipeline) ──
    print("\n[5/6] KPI Calculator — Z-Score -> BSC -> Cross-Role Ranking")
    calc = KPICalculator()
    calc.bsc_weights = bsc_result["dimension_weights"]
    ranked = calc.full_pipeline(employees_data)
    report = calc.generate_report(ranked)
    dashboard_kpi = calc.dashboard_format(report)
    print(f"  Top performer: {report['top_performer']['name']} ({report['top_performer']['role']}) — {report['top_performer']['score']:.4f}")

    # ── Module 6: Monthly Trend ──
    print("\n[6/6] Trend Generator — 6-month KPI history")
    trends = collector.generate_trend_data(months=6)

    # ═══════════════════════════════════════════════════════
    # ASSEMBLE DASHBOARD OUTPUT (matches [[dashboard-designs]])
    # ═══════════════════════════════════════════════════════
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
            {
                "month": t["month"],
                "avg_score": t["avg_score"],
                "top": t["top_performer"]["name"] if t["top_performer"] else "-",
            }
            for t in trends
        ],
    }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  DASHBOARD SUMMARY")
    print("=" * 70)
    print(f"\n  Panel 1 — Today Attendance ({today}):")
    print(f"    Present: {dashboard['panel_1_attendance']['present']}  "
          f"Late: {dashboard['panel_1_attendance']['late']}  "
          f"Absent: {dashboard['panel_1_attendance']['absent']}")

    print(f"\n  Panel 2 — Weekly Shift:")
    print(f"    {dashboard['panel_2_shift']['week_start']} ~ {dashboard['panel_2_shift']['week_end']}")
    print(f"    Coverage: {'OK' if dashboard['panel_2_shift']['coverage_ok'] else 'GAPS'}")

    print(f"\n  Panel 3 — Monthly KPI Ranking:")
    print(f"    {'Rank':<6} {'Name':<14} {'Role':<10} {'Score':<10} {'%ile'}")
    print(f"    {'-'*6} {'-'*14} {'-'*10} {'-'*10} {'-'*6}")
    for e in dashboard["panel_3_kpi"]["ranking"]:
        print(f"    {e['rank']:<6} {e['name']:<14} {e['role']:<10} {e['score']:>+8.4f}  {e['percentile']:>5.1f}%")

    print(f"\n  Role Averages:")
    for role, data in dashboard["role_breakdown"].items():
        print(f"    {data['name']:<10} avg={data['avg_score']:+.4f}  top={data['top']}")

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Saved: {save_path}")

    print("\n" + "=" * 70)
    print("  KPI Pipeline Complete — Ready for Dashboard")
    print("=" * 70)

    return dashboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KPI Full Integrated Demo")
    parser.add_argument("--save", type=str, default=None, help="Save JSON output")
    args = parser.parse_args()
    run_full_integrated_demo(args.save)
