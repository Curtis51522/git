"""
KPI Configuration - 7-Metric System with BSC Dimensions
=========================================================
7 objective metrics, no manager ratings needed.

Metrics:
  1. revenue_hr     - Personal attributed revenue per work hour
  2. revenue_growth - Revenue/hr change vs previous month
  3. work_hours     - Total work hours this month
  4. hours_vs_avg   - Work hours vs 9-person average
  5. attendance_rate - Actual / expected working days
  6. waste_rate     - Material wastage rate (baker/barista only)
  7. punctuality    - On-time punch rate (cross-role shared)

BSC Framework:
  Financial (25%)   - revenue_hr, revenue_growth
  Internal (35%)    - work_hours, hours_vs_avg, attendance_rate, punctuality, waste_rate
  Customer (20%)    - (reserved for future kpi_ratings)
  Learning (20%)    - (reserved for future kpi_ratings)
"""

BSC_WEIGHTS = {
    "Financial":       0.25,
    "Customer":        0.20,
    "Internal Process": 0.35,
    "Learning & Growth": 0.20,
}

# Per-role KPI catalog: each KPI has weight, direction, bsc_dimension, cross_role flag
SHARED_KPIS = {
    "revenue_contribution": {
        "name_cn": "Revenue Contribution", "unit": "CNY", "direction": "higher_better",
        "bsc_dimension": "Financial", "weight": 0.60,
    },
    "revenue_growth": {
        "name_cn": "Revenue Growth MoM", "unit": "%", "direction": "higher_better",
        "bsc_dimension": "Financial", "weight": 0.40,
    },
    "work_hours": {
        "name_cn": "Work Hours", "unit": "h", "direction": "higher_better",
        "bsc_dimension": "Internal Process", "weight": 0.25,
    },
    "hours_vs_avg": {
        "name_cn": "Hours vs Avg", "unit": "%", "direction": "higher_better",
        "bsc_dimension": "Internal Process", "weight": 0.15,
    },
    "attendance_rate": {
        "name_cn": "Attendance Rate", "unit": "%", "direction": "higher_better",
        "bsc_dimension": "Internal Process", "weight": 0.25,
    },
    "punctuality": {
        "name_cn": "Punctuality", "unit": "%", "direction": "higher_better",
        "bsc_dimension": "Internal Process", "weight": 0.20, "cross_role": True,
    },
    "waste_rate": {
        "name_cn": "Waste Rate", "unit": "%", "direction": "lower_better",
        "bsc_dimension": "Internal Process", "weight": 0.15,
    },
}

ROLES = {}

def _build_role(role_name, extra_kpis=None):
    kpis = dict(SHARED_KPIS)
    if extra_kpis:
        kpis.update(extra_kpis)
    return {"name_cn": role_name, "kpis": kpis}

ROLES["baker"]    = _build_role("Baker")
ROLES["barista"]  = _build_role("Barista")
ROLES["cashier"]  = _build_role("Cashier", {"waste_rate": None})  # exclude waste_rate
ROLES["manager"]  = _build_role("Manager")

# Remove None values
for role in ROLES:
    ROLES[role]["kpis"] = {k: v for k, v in ROLES[role]["kpis"].items() if v is not None}
