"""
KPI Data Collector - 7-Metric System (MySQL only, no fabricated data)
======================================================================
Reads real data: attendance_records, orders, material_wastage_log, employees.

Dual-role: employees with role2 run through both role pipelines,
           calculator picks the higher score.

Metrics computed:
  1. revenue_contribution - Attributed revenue (distributed by attendance_rate)
  2. revenue_growth       - Revenue contribution MoM change
  3. work_hours           - Total punch hours this month
  4. hours_vs_avg         - Hours vs 9-person avg
  5. attendance_rate      - Attended / completed scheduled shifts
  6. waste_rate           - Material wastage (baker/barista only)
  7. punctuality          - On-time arrival rate (cross-role)
  8. shift_completion     - Full-shift completion rate (cross-role)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from kpi.attendance import calculate_period_metrics, resolve_month_period_end

logger = logging.getLogger('kpi.collector')


class KPIDataCollector:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else _PROJECT_ROOT / "data"

    def _get_db(self):
        from db.mysql_client import get_db
        return get_db()

    # ==================================================================
    # Main entry: collect all KPIs for a month
    # ==================================================================

    def collect_monthly(self, year=None, month=None):
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month
        month_str = f"{year}-{month:02d}"

        employees = self._load_employees()
        if not employees:
            return []

        # Phase 1: attendance and work hours (per person, before role expansion)
        attendance_data = self._collect_attendance(employees, month_str)

        # Phase 2: expand dual roles BEFORE revenue attribution
        all_entries = []
        for emp in employees:
            eid = emp["id"]
            entry = {"id": eid, "name": emp["name"], "role": emp["role"],
                     "role2": emp.get("role2"), "is_dual_role": False, "primary_role": emp["role"],
                     "kpis": dict(attendance_data.get(eid, {}))}
            all_entries.append(entry)
            # Add secondary role entry if dual-role
            role2 = emp.get("role2")
            if role2 and role2 != emp["role"]:
                sec_entry = {"id": eid, "name": emp["name"], "role": role2,
                             "is_dual_role": True, "primary_role": emp["role"],
                             "kpis": dict(attendance_data.get(eid, {}))}
                all_entries.append(sec_entry)

        # Phase 3: revenue data (per role-entry, distributed by attendance rate)
        revenue_data = self._collect_revenue(all_entries, month_str, attendance_data)
        for entry in all_entries:
            eid = entry["id"]
            role = entry["role"]
            key = (eid, role)
            if key in revenue_data:
                entry["kpis"].update(revenue_data[key])

        # Phase 4: inventory/waste (per role)
        waste_data = self._collect_waste(all_entries, month_str)
        for entry in all_entries:
            eid = entry["id"]
            role = entry["role"]
            key = (eid, role)
            if key in waste_data:
                entry["kpis"].update(waste_data[key])

        return all_entries

    # ==================================================================
    # Employee loading
    # ==================================================================

    def _load_employees(self):
        try:
            db = self._get_db()
            cur = db.cursor(dictionary=True)
            cur.execute("SELECT id, name, role, role2 FROM employees WHERE available=1")
            rows = cur.fetchall()
            cur.close()
            if rows:
                return [{"id": r["id"], "name": r["name"], "role": r["role"], "role2": r.get("role2")} for r in rows]
        except Exception as e:
            logger.warning('Employee load failed: %s', e)
        return []


    # Dual-role expansion handled inline in collect_monthly


    # ==================================================================
    # Schedule-based attendance metrics
    # ==================================================================

    def _collect_attendance(self, employees, month_str):
        kpis = {}
        try:
            db = self._get_db()
            cur = db.cursor(dictionary=True)
            cur.execute(
                "SELECT emp_id, status, date, punch_in, punch_out FROM attendance_records WHERE date LIKE %s",
                (month_str + "%",)
            )
            records = cur.fetchall()
            cur.close()

            cur2 = db.cursor(dictionary=True)
            cur2.execute(
                """
                SELECT schedule_date, time_slot, employee_id, employee_name, role
                FROM shift_schedule
                WHERE schedule_date LIKE %s
                ORDER BY schedule_date, time_slot, employee_id
                """,
                (month_str + "%",),
            )
            schedules = cur2.fetchall()
            cur2.close()

            year, month = (int(part) for part in month_str.split("-"))
            period_end = resolve_month_period_end(year, month, records)
            kpis = calculate_period_metrics(
                employees,
                schedules,
                records,
                period_end,
            )
        except Exception as e:
            logger.warning('Attendance collection failed: %s', e)
        return kpis

    # ==================================================================
    # Metric 1: Revenue Contribution  +  Metric 2: Revenue Growth
    # ==================================================================

    def _collect_revenue(self, employees, month_str, attendance_data):
        """Attribute revenue by attendance_rate proportion within role pools."""
        kpis = {}
        try:
            db = self._get_db()
            cur = db.cursor(dictionary=True)

            # Current month revenue by category
            cur.execute("""
                SELECT 
                    CASE WHEN p.category = 'bakery' THEN 'bakery'
                         WHEN p.category = 'beverages' THEN 'beverage'
                         ELSE 'other' END as cat,
                    SUM(oi.quantity * oi.unit_price) as rev
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                JOIN products p ON oi.product_name = p.product_name
                WHERE o.order_date LIKE %s
                GROUP BY cat
            """, (month_str + "%",))
            rows = cur.fetchall()
            cur.close()

            rev_map = {}
            for r in rows:
                rev_map[r["cat"]] = float(r["rev"] or 0)
            bakery_rev = rev_map.get("bakery", 0)
            bev_rev = rev_map.get("beverage", 0)
            total_rev = bakery_rev + bev_rev

            # Also get previous month for growth calculation
            prev_month = self._prev_month_str(month_str)
            cur = db.cursor()
            cur.execute("""
                SELECT SUM(oi.quantity * oi.unit_price) as rev
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                WHERE o.order_date LIKE %s
            """, (prev_month + "%",))
            row = cur.fetchone()
            prev_total = float(row[0] or 0) if row else 0
            cur.close()

            # Attribution by attendance_rate proportion within role pool
            role_pools = {
                "baker": bakery_rev,
                "barista": bev_rev,
                "cashier": total_rev,  # cashier touches all transactions
            }

            for role, pool_rev in role_pools.items():
                role_emps = [e for e in employees if e["role"] == role]
                if not role_emps or pool_rev <= 0:
                    continue
                # Sum work hours for this role (more granular than attendance_rate)
                total_hrs = sum(
                    attendance_data.get(e["id"], {}).get("work_hours", 0)
                    for e in role_emps
                ) or 1
                for emp in role_emps:
                    eid = emp["id"]
                    hrs = attendance_data.get(eid, {}).get("work_hours", 0)
                    share = hrs / total_hrs if total_hrs > 0 else 1.0 / len(role_emps)
                    attributed = pool_rev * share

                    # Previous month attribution (same proportion)
                    prev_month_att = prev_total * (hrs / total_hrs) if prev_total > 0 and total_hrs > 0 else 0

                    # Revenue growth
                    if prev_month_att > 0:
                        growth = round((attributed - prev_month_att) / prev_month_att * 100, 1)
                    else:
                        growth = 0.0

                    key = (eid, role)
                    if key not in kpis:
                        kpis[key] = {}
                    kpis[key]["revenue_contribution"] = round(attributed, 2)
                    kpis[key]["revenue_growth"] = growth

            # Hours vs average (Metric 4)
            all_hours = [attendance_data.get(e["id"], {}).get("work_hours", 0) for e in employees]
            avg_hours = sum(all_hours) / max(len(all_hours), 1)
            for emp in employees:
                eid = emp["id"]
                wh = attendance_data.get(eid, {}).get("work_hours", 0)
                if avg_hours > 0:
                    hours_vs = round((wh - avg_hours) / avg_hours * 100, 1)
                else:
                    hours_vs = 0.0
                key = (eid, emp["role"])
                if key not in kpis:
                    kpis[key] = {}
                kpis[key]["hours_vs_avg"] = hours_vs

        except Exception as e:
            logger.warning('Revenue collection failed: %s', e)
        return kpis

    # ==================================================================
    # Metric 6: Waste Rate (baker/barista only)
    # ==================================================================

    def _collect_waste(self, employees, month_str):
        kpis = {}
        try:
            db = self._get_db()
            cur = db.cursor(dictionary=True)
            cur.execute(
                "SELECT AVG(wastage_rate) as avg_waste FROM material_wastage_log WHERE check_date LIKE %s",
                (month_str + "%",)
            )
            row = cur.fetchone()
            avg_waste = round(float(row["avg_waste"] or 0) * 100, 1) if row and row["avg_waste"] else 0.0
            cur.close()

            for emp in employees:
                role = emp["role"]
                if role in ("baker", "barista"):
                    kpis[(emp["id"], role)] = {"waste_rate": avg_waste}
        except Exception as e:
            logger.warning('Waste collection failed: %s', e)
        return kpis

    # ==================================================================
    # Helpers
    # ==================================================================

    def _prev_month_str(self, month_str):
        y, m = int(month_str[:4]), int(month_str[5:7])
        if m == 1:
            return f"{y-1}-12"
        return f"{y}-{m-1:02d}"

    # ==================================================================
    # Trend data for dashboard (last N months)
    # ==================================================================

    def generate_trend_data(self, months=6):
        trends = []
        now = datetime.now()
        for i in range(months - 1, -1, -1):
            dt = now.replace(day=1) - timedelta(days=1)
            dt = dt.replace(day=1)
            if i > 0:
                dt = (dt - timedelta(days=i * 30)).replace(day=1)
            data = self.collect_monthly(year=dt.year, month=dt.month)
            if data:
                scores = [e["kpis"].get("revenue_contribution", 0) for e in data if e.get("role") != "manager"]
                avg_score = sum(scores) / max(len(scores), 1)
                top = max(data, key=lambda e: e.get("total_score", 0.0)) if data else None
                trends.append({
                    "month": dt.strftime("%Y-%m"),
                    "avg_score": round(avg_score, 1),
                    "top_performer": {"name": top["name"], "role": top["role"]} if top else None,
                })
        return trends
