"""
KPI Configuration — Roles, Metrics, BSC Weights
=================================================
Based on: [[kpi-research]] | [[employee-KPI-normalization-research]]

BSC Framework:
  Financial (25%)    — profit, waste, upselling
  Customer (25%)     — satisfaction, complaints
  Internal (30%)     — output, speed, accuracy, punctuality
  Learning (20%)     — cross-skills, training

Each role has 5-7 KPIs (Mambetakunova 2026 recommendation).
"""

# ============================================================
# BSC DIMENSION WEIGHTS (from AHP — can be recalibrated)
# ============================================================
BSC_WEIGHTS = {
    "Financial":       0.25,
    "Customer":        0.25,
    "Internal Process": 0.30,
    "Learning & Growth": 0.20,
}

# ============================================================
# ROLE DEFINITIONS
# ============================================================
ROLES = {
    "baker": {
        "name_cn": "面包师",
        "kpis": {
            "daily_output": {
                "name_cn": "日出品量",
                "unit": "units/day",
                "direction": "higher_better",   # higher = better
                "bsc_dimension": "Internal Process",
                "weight": 0.30,
            },
            "waste_rate": {
                "name_cn": "浪费率",
                "unit": "%",
                "direction": "lower_better",
                "bsc_dimension": "Financial",
                "weight": 0.20,
            },
            "product_quality": {
                "name_cn": "产品合格率",
                "unit": "score 1-5",
                "direction": "higher_better",
                "bsc_dimension": "Customer",
                "weight": 0.20,
            },
            "punctuality": {
                "name_cn": "准时率",
                "unit": "%",
                "direction": "higher_better",
                "bsc_dimension": "Internal Process",
                "weight": 0.15,
            },
            "cross_skills": {
                "name_cn": "交叉技能数",
                "unit": "count",
                "direction": "higher_better",
                "bsc_dimension": "Learning & Growth",
                "weight": 0.15,
            },
        },
    },
    "barista": {
        "name_cn": "咖啡师",
        "kpis": {
            "drinks_per_hour": {
                "name_cn": "每小时出品",
                "unit": "drinks/h",
                "direction": "higher_better",
                "bsc_dimension": "Internal Process",
                "weight": 0.30,
            },
            "customer_satisfaction": {
                "name_cn": "客户满意度",
                "unit": "score 1-5",
                "direction": "higher_better",
                "bsc_dimension": "Customer",
                "weight": 0.25,
            },
            "waste_rate": {
                "name_cn": "原料浪费率",
                "unit": "%",
                "direction": "lower_better",
                "bsc_dimension": "Financial",
                "weight": 0.20,
            },
            "punctuality": {
                "name_cn": "准时率",
                "unit": "%",
                "direction": "higher_better",
                "bsc_dimension": "Internal Process",
                "weight": 0.15,
            },
            "latte_art_skill": {
                "name_cn": "拉花等级",
                "unit": "level 1-3",
                "direction": "higher_better",
                "bsc_dimension": "Learning & Growth",
                "weight": 0.10,
            },
        },
    },
    "cashier": {
        "name_cn": "收银员",
        "kpis": {
            "checkout_speed": {
                "name_cn": "结账速度",
                "unit": "items/min",
                "direction": "higher_better",
                "bsc_dimension": "Internal Process",
                "weight": 0.25,
            },
            "accuracy_rate": {
                "name_cn": "准确率",
                "unit": "%",
                "direction": "higher_better",
                "bsc_dimension": "Internal Process",
                "weight": 0.25,
            },
            "upselling_rate": {
                "name_cn": "追加销售率",
                "unit": "%",
                "direction": "higher_better",
                "bsc_dimension": "Financial",
                "weight": 0.20,
            },
            "customer_satisfaction": {
                "name_cn": "客户满意度",
                "unit": "score 1-5",
                "direction": "higher_better",
                "bsc_dimension": "Customer",
                "weight": 0.20,
            },
            "punctuality": {
                "name_cn": "准时率",
                "unit": "%",
                "direction": "higher_better",
                "bsc_dimension": "Internal Process",
                "weight": 0.10,
            },
        },
    },
    "manager": {
        "name_cn": "店长",
        "kpis": {
            "team_profit_margin": {
                "name_cn": "团队毛利率",
                "unit": "%",
                "direction": "higher_better",
                "bsc_dimension": "Financial",
                "weight": 0.25,
            },
            "sales_growth": {
                "name_cn": "销售增长率",
                "unit": "% MoM",
                "direction": "higher_better",
                "bsc_dimension": "Financial",
                "weight": 0.20,
            },
            "inventory_accuracy": {
                "name_cn": "盘点准确率",
                "unit": "%",
                "direction": "higher_better",
                "bsc_dimension": "Internal Process",
                "weight": 0.20,
            },
            "staff_retention": {
                "name_cn": "员工留存率",
                "unit": "%",
                "direction": "higher_better",
                "bsc_dimension": "Learning & Growth",
                "weight": 0.15,
            },
            "customer_satisfaction": {
                "name_cn": "客户满意度",
                "unit": "score 1-5",
                "direction": "higher_better",
                "bsc_dimension": "Customer",
                "weight": 0.20,
            },
        },
    },
}
