import os


# s5_agent config
# Central configuration for all S5 agents - single source of truth
# =============================================================================

PORT = int(os.getenv("S5_PORT", "8001"))
HOST = os.getenv("S5_HOST", "127.0.0.1")

# Main bakery system API base (S1-S4 on :8002)
API_BASE = os.getenv("BAKERY_API_BASE", "http://127.0.0.1:8002").rstrip("/")


def api_url(path: str) -> str:
    return f"{API_BASE}/{path.lstrip('/')}"

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

S1_INVENTORY_URL = api_url("s1/inventory")
S1_INFLOW_HISTORY_URL = api_url("s1/inflow/history")
S4_DASHBOARD_URL = api_url("s4/inventory/dashboard")
