"""
KPI Data Collector — POS + Sales + Inventory Integration
===========================================================
Gathers raw KPI values from system data sources for automated monthly reporting.

Data sources:
  - Attendance records       → attendance_rate, punctuality_rate
  - POS sales data          → upselling_rate, checkout_speed, sales_growth
  - Inventory               → waste_rate, inventory_accuracy
  - Manager ratings         → product_quality, customer_satisfaction, cross_skills

Integrates with: [[commercial-requirements]] — "POS auto-collection"
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict


class KPIDataCollector:
    """Collect raw KPI values from system data."""

    def __init__(self, data_dir=None):
        if data_dir is None:
            import os as _os
            _base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            data_dir = _os.path.join(_base, "data")
        self.data_dir = data_dir

    def collect_monthly(self, year=None, month=None):
        """
        Collect all KPIs for a given month from available data sources.

        Returns:
            list of employee dicts ready for KPICalculator.full_pipeline()
        """
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month

        month_str = f"{year}-{month:02d}"

        employees = self._load_employees()
        attendance = self._collect_attendance(employees, month_str)
        sales = self._collect_sales_data(month_str)
        inventory = self._collect_inventory(month_str)
        ratings = self._collect_manager_ratings(employees, month_str)

        result = []
        for emp in employees:
            emp_id = emp["id"]
            entry = {
                "id": emp_id,
                "name": emp["name"],
                "role": emp["role"],
                "kpis": {},
            }

            # Merge all data sources
            entry["kpis"].update(attendance.get(emp_id, {}))
            entry["kpis"].update(sales.get(emp_id, {}))
            entry["kpis"].update(inventory.get(emp_id, {}))
            entry["kpis"].update(ratings.get(emp_id, {}))

            result.append(entry)

        return result

    def _load_employees(self):
        """Load employee roster. Falls back to sample data."""
        emp_path = os.path.join(self.data_dir, "employees.json")
        if os.path.exists(emp_path):
            with open(emp_path) as f:
                return json.load(f)
        # Fallback sample
        return [
            {"id": "B001", "name": "Zhang Wei", "role": "baker"},
            {"id": "B002", "name": "Li Ming", "role": "baker"},
            {"id": "B003", "name": "Wang Fang", "role": "baker"},
            {"id": "R001", "name": "Chen Yu", "role": "barista"},
            {"id": "R002", "name": "Liu Na", "role": "barista"},
            {"id": "R003", "name": "Huang Li", "role": "barista"},
            {"id": "C001", "name": "Zhou Jie", "role": "cashier"},
            {"id": "C002", "name": "Wu Min", "role": "cashier"},
            {"id": "M001", "name": "Sun Tao", "role": "manager"},
        ]

    def _collect_attendance(self, employees, month_str):
        """
        Collect attendance KPIs.
        Tries to load from attendance records, falls back to simulation.
        """
        # Try real data
        att_path = os.path.join(self.data_dir, "attendance.json")
        if os.path.exists(att_path):
            with open(att_path) as f:
                records = json.load(f)
            return self._compute_attendance_kpis(employees, records, month_str)

        # Simulate realistic data
        kpis = {}
        for emp in employees:
            kpis[emp["id"]] = {
                "attendance_rate": round(np.random.uniform(88, 100), 1),
                "punctuality": round(np.random.uniform(85, 100), 1),
            }
        return kpis

    def _compute_attendance_kpis(self, employees, records, month_str):
        """Compute attendance_rate and punctuality from punch records."""
        month_records = [r for r in records if r.get("date", "").startswith(month_str)]
        kpis = {}
        for emp in employees:
            emp_records = [r for r in month_records if r.get("id") == emp["id"]]
            days_present = len(emp_records)
            on_time = sum(1 for r in emp_records if r.get("status") == "on_time")
            # Assume 26 working days
            working_days = 26
            kpis[emp["id"]] = {
                "attendance_rate": round(min(100, days_present / working_days * 100), 1),
                "punctuality": round(on_time / max(days_present, 1) * 100, 1),
            }
        return kpis

    def _collect_sales_data(self, month_str):
        """
        Collect sales-derived KPIs: upselling_rate, checkout_speed, sales_growth.
        Simulated for now — in production, reads from POS orders table.
        """
        kpis = {}
        # Cashier KPIs
        cashiers = ["C001", "C002"]
        for cid in cashiers:
            kpis[cid] = {
                "checkout_speed": round(np.random.uniform(3.5, 7.5), 1),
                "accuracy_rate": round(np.random.uniform(96, 100), 1),
                "upselling_rate": round(np.random.uniform(12, 32), 1),
            }
        # Manager KPIs
        kpis["M001"] = {
            "sales_growth": round(np.random.uniform(-3, 12), 1),
            "team_profit_margin": round(np.random.uniform(58, 72), 1),
        }
        return kpis

    def _collect_inventory(self, month_str):
        """
        Collect inventory KPIs: waste_rate, inventory_accuracy, daily_output.
        Simulated for now.
        """
        kpis = {}
        # Baker: waste_rate, daily_output
        bakers = ["B001", "B002", "B003"]
        for bid in bakers:
            kpis[bid] = {
                "daily_output": int(np.random.normal(185, 20)),
                "waste_rate_baker": round(np.random.uniform(2.5, 7.5), 1),
            }

        # Barista: waste_rate, drinks_per_hour
        baristas = ["R001", "R002", "R003"]
        for rid in baristas:
            kpis[rid] = {
                "drinks_per_hour": int(np.random.normal(26, 4)),
                "waste_rate_barista": round(np.random.uniform(1.5, 4.5), 1),
            }

        # Manager
        kpis["M001"] = {
            "inventory_accuracy": round(np.random.uniform(91, 99), 1),
        }

        # Normalize waste_rate keys (baker and barista have same KPI name)
        for emp_id, vals in kpis.items():
            if "waste_rate_baker" in vals:
                vals["waste_rate"] = vals.pop("waste_rate_baker")
            if "waste_rate_barista" in vals:
                vals["waste_rate"] = vals.pop("waste_rate_barista")

        return kpis

    def _collect_manager_ratings(self, employees, month_str):
        """
        Collect subjective ratings: product_quality, customer_satisfaction,
        cross_skills, latte_art_skill, staff_retention.
        In production: manager fills a monthly form.
        """
        kpis = {}
        for emp in employees:
            role = emp["role"]
            ratings = {
                "customer_satisfaction": round(np.random.uniform(3.5, 5.0), 1),
            }
            if role == "baker":
                ratings["product_quality"] = round(np.random.uniform(3.5, 5.0), 1)
                ratings["cross_skills"] = np.random.randint(1, 5)
            elif role == "barista":
                ratings["latte_art_skill"] = np.random.randint(1, 4)
            elif role == "manager":
                ratings["staff_retention"] = round(np.random.uniform(78, 100), 1)
            kpis[emp["id"]] = ratings
        return kpis

    def generate_trend_data(self, months=6):
        """
        Generate multi-month trend data for KPI visualization.

        Returns:
            list of monthly reports (for line charts in dashboard)
        """
        from kpi.calculator import KPICalculator
        calc = KPICalculator()

        now = datetime.now()
        trends = []

        for i in range(months):
            target = now - timedelta(days=30 * (months - 1 - i))
            employees = self.collect_monthly(target.year, target.month)
            ranked = calc.full_pipeline(employees)
            report = calc.generate_report(ranked, month=target.strftime("%Y-%m"))
            trends.append({
                "month": target.strftime("%Y-%m"),
                "avg_score": round(np.mean([e["total_score"] for e in ranked]), 4),
                "top_performer": report["top_performer"],
                "total_employees": len(ranked),
            })

        return trends
