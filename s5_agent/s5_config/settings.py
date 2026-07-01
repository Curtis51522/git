# s5_agent config
# Central configuration for all S5 agents ¡ª single source of truth
# =============================================================================

PORT = 8001
HOST = "127.0.0.1"

# Main bakery system API base (S1-S4 on :8002)
API_BASE = "http://127.0.0.1:8002"

# =============================================================================
# THRESHOLDS ¡ª all agent alert/decision thresholds live here.
# Tune these in one place; every agent reads from this dict.
# =============================================================================

THRESHOLDS = {
    # --- PromoAgent & PricingAgent ---
    # discount_rate = total_discount / total_revenue
    # Day-1 fixed 20% discount on aged products usually produces ~4% effective rate.
    # Only flag when effective discount exceeds this.
    "promo_high_discount_rate": 0.15,   # >15% effective discount = excessive

    # --- ProfitAgent ---
    "profit_low_margin_pct": 20,        # <20% gross margin = alert

    # --- StaffingAgent ---
    "staffing_min_heads": 3,            # <3 staff on shift = understaffed

    # --- WastageAgent ---
    "wastage_abnormal_rate": 0.15,        # >15% wastage on a material = alert

    # --- ProductStockAgent ---
    "product_day1_ratio": 0.30,         # >30% of products are Day-1 = alert

    # --- ProductionAgent ---
    "production": {
        "ovens": 2,
        "bakers": 5,
        "hours_per_shift": 8,
        "units_per_baker_per_oven_per_hour": 60,
    },

    # --- InventoryAgent ---
    "inventory_fresh_low": 10,          # fresh units below this + total > threshold = high waste risk
    "inventory_total_high": 60,         # total units above this + fresh low = high waste risk
    "inventory_default_price": 5.90,    # fallback unit price when product not found
}

# =============================================================================
# External service endpoints (S1-S4 on :8002)
# =============================================================================

S1_INVENTORY_URL  = f"{API_BASE}/s1/inventory"
S2_FORECAST_URL   = f"{API_BASE}/s2/forecast"
S3_SCHEDULE_URL   = f"{API_BASE}/s3/schedule"
S3_KPI_URL        = f"{API_BASE}/s3/kpi"
S3_ATTENDANCE_URL = f"{API_BASE}/s3/attendance"
S4_REVENUE_URL    = f"{API_BASE}/s4/revenue/daily"
S4_MATERIALS_URL  = f"{API_BASE}/s4/inventory/materials"
S4_DASHBOARD_URL  = f"{API_BASE}/s4/inventory/dashboard"
S4_WASTAGE_URL    = f"{API_BASE}/s4/inventory/wastage/summary"
S1_BATCH_URL      = f"{API_BASE}/s1/batch_inventory"
S4_COMBO_URL      = f"{API_BASE}/s4/combo"
