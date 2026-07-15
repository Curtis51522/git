# s5_agent config
# Central configuration for all S5 agents - single source of truth
# =============================================================================

PORT = 8001
HOST = "127.0.0.1"

# Main bakery system API base (S1-S4 on :8002)
API_BASE = "http://127.0.0.1:8002"

# =============================================================================
# THRESHOLDS - all agent alert/decision thresholds live here.
# Tune these in one place; every agent reads from this dict.
# =============================================================================

THRESHOLDS = {
    # --- PromoAgent ---
    # discount_rate = total_discount / total_revenue
    # Day-1 fixed 20% discount on aged products usually produces ~4% effective rate.
    # Only flag when effective discount exceeds this.
    "promo_high_discount_rate": 0.15,   # >15% effective discount = excessive
    "promotion_loss_concentration_pct": 20,
    "promotion_target_margin_floor_pct": 30,
    "promotion_target_sell_through_floor_pct": 50,

    # --- ProfitAgent ---
    "profit_low_margin_pct": 20,        # <20% gross margin = alert
    "profit_expired_cost_alert_pct": 5,  # >=5% of revenue = material closing loss

    # --- WastageAgent ---
    "wastage_abnormal_rate": 0.15,        # >15% wastage on a material = alert

    # --- InventoryAgent ---
    "inventory_fresh_low": 10,          # fresh units below this + total > threshold = high waste risk
    "inventory_total_high": 60,         # total units above this + fresh low = high waste risk
    "inventory_default_price": 5.90,    # fallback unit price when product not found
    "inventory_high_sell_through_pct": 80,
    "inventory_low_sell_through_pct": 40,
    "inventory_flow_min_baked_units": 3,
}

# =============================================================================
# External service endpoints (S1-S4 on :8002)
# =============================================================================

S1_INVENTORY_URL  = f"{API_BASE}/s1/inventory"
S1_INFLOW_HISTORY_URL = f"{API_BASE}/s1/inflow/history"
S2_FORECAST_URL   = f"{API_BASE}/s2/forecast"
S4_REVENUE_URL    = f"{API_BASE}/s4/revenue/daily"
S4_MATERIALS_URL  = f"{API_BASE}/s4/inventory/materials"
S4_DASHBOARD_URL  = f"{API_BASE}/s4/inventory/dashboard"
S4_WASTAGE_URL    = f"{API_BASE}/s4/inventory/wastage/summary"
S1_BATCH_URL      = f"{API_BASE}/s1/batch_inventory"
S4_COMBO_URL      = f"{API_BASE}/s4/combo"
