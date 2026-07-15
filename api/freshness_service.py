"""
Freshness & Discount Engine
- Ages inventory daily: Fresh -> Day-1 -> Expired
- Applies automatic discounts based on age
- Assigns tray_color for visual recognition
"""
from datetime import datetime
from db.mysql_client import get_db, q

# Discount rates by freshness level
DISCOUNT_MAP = {
    "Fresh":    0.0,   # full price
    "Day-1":    0.20,  # 20% off
    "Expired":  1.0,   # unsellable / destroyed
}

# Tray colors for visual system
FRESHNESS_COLORS = {
    "Fresh":    "green",
    "Day-1":    "yellow",
    "Expired":  "black",
}

def get_freshness(production_time_str: str, reference_date: str = None) -> str:
    """Determine freshness status based on days since production.
    
    Same calendar day:         Fresh (full price)
    Next calendar day:         Day-1 (20% off, switches at midnight)
    2+ calendar days later:    Expired (destroyed, recorded as waste)
    """
    if not production_time_str:
        return "Fresh"
    
    try:
        prod_time = datetime.strptime(production_time_str[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            prod_time = datetime.strptime(production_time_str[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return "Fresh"
    
    if reference_date:
        ref = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        ref = datetime.now()
    
    days_diff = (ref.date() - prod_time.date()).days
    
    if days_diff == 0:
        return "Fresh"
    elif days_diff == 1:
        return "Day-1"
    else:
        return "Expired"

def update_all_freshness(reference_date: str = None):
    """Run freshness update on all inventory batches.
    When a batch expires (2+ calendar days old), record it as waste and delete from inventory.
    """
    db = get_db()
    batches = (
        q(db, "batch_inventory")
        .select("*")
        .gt("quantity_remaining", 0)
        .execute()
    )
    product_rows = q(db, "products").select("product_name,material_cost").execute()
    product_costs = {
        row["product_name"]: float(row.get("material_cost") or 0)
        for row in (product_rows.data or [])
    }
    
    updated = 0
    expired_cleared = 0
    for batch in batches.data:
        prod_time = batch.get("production_time", "")
        if not prod_time:
            continue
        
        new_freshness = get_freshness(prod_time, reference_date)
        old_freshness = batch.get("freshness_status", "Fresh")
        
        if new_freshness == old_freshness:
            continue
        
        if new_freshness == "Expired":
            # Record waste transaction before deleting
            q(db, "inventory_transactions").insert({
                "transaction_type": "outflow",
                "batch_id": batch.get("batch_id", ""),
                "product_name": batch.get("product_name", ""),
                "quantity": batch.get("quantity_remaining", 0),
                "unit_price": product_costs.get(batch.get("product_name", ""), 0),
                "discount_applied": 1.0,
                "freshness_status": "Expired",
            }).execute()
            # Delete from batch_inventory
            q(db, "batch_inventory").delete().eq("batch_id", batch["batch_id"]).execute()
            expired_cleared += 1
        else:
            tray_color = FRESHNESS_COLORS.get(new_freshness, "green")
            q(db, "batch_inventory").update({
                "freshness_status": new_freshness,
                "tray_color": tray_color,
            }).eq("batch_id", batch["batch_id"]).execute()
            updated += 1
    
    return {
        "updated": updated,
        "expired_cleared": expired_cleared,
        "reference_date": reference_date or str(datetime.now().date()),
    }

def get_discount_rate(freshness: str) -> float:
    """Get discount rate for a freshness level."""
    return DISCOUNT_MAP.get(freshness, 0)

def get_tray_color(freshness: str) -> str:
    """Get tray color for a freshness level."""
    return FRESHNESS_COLORS.get(freshness, "green")


def get_sellable_batches():
    """Get all sellable batches (not expired), ordered by freshness (oldest first = FIFO)."""
    db = get_db()
    return (
        q(db, "batch_inventory")
        .select("*")
        .gt("quantity_remaining", 0)
        .neq("freshness_status", "Expired")
        .order("production_time", desc=False)
        .execute()
    )
