"""
Freshness & Discount Engine
- Ages inventory daily: Fresh -> Day-1 -> Day-2 -> Near-Expired
- Applies automatic discounts based on age
- Assigns tray_color for visual recognition
"""
from datetime import datetime, timedelta
from db.mysql_client import get_db, q

# Discount rates by freshness level
DISCOUNT_MAP = {
    "Fresh":         0.0,   # full price
    "Day-1":         0.10,  # 10% off (9?)
    "Day-2":         0.20,  # 20% off (8?)
    "Near-Expired":  0.30,  # 30% off (7?)
    "Expired":       1.0,   # unsellable
}

# Tray colors for visual system
FRESHNESS_COLORS = {
    "Fresh":         "green",
    "Day-1":         "yellow",
    "Day-2":         "orange",
    "Near-Expired":  "red",
    "Expired":       "black",
}

def get_freshness(production_time_str: str, reference_date: str = None) -> str:
    """Determine freshness status based on days since production.
    
    Day 0-1 (0-24h):    Fresh
    Day 1-2 (24-48h):   Day-1 (10% off)
    Day 2-3 (48-72h):   Day-2 (20% off)
    Day 3-4 (72-96h):   Near-Expired (30% off)
    Day 4+ (>96h):       Expired
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
    
    hours_old = (ref - prod_time).total_seconds() / 3600
    
    if hours_old < 24:
        return "Fresh"
    elif hours_old < 48:
        return "Day-1"
    elif hours_old < 72:
        return "Day-2"
    elif hours_old < 96:
        return "Near-Expired"
    else:
        return "Expired"

def update_all_freshness(reference_date: str = None):
    """Run daily freshness update on all inventory batches."""
    db = get_db()
    batches = q(db, "batch_inventory").select("*").gt("quantity", 0).execute()
    
    updated = 0
    for batch in batches.data:
        prod_time = batch.get("production_time", "")
        if not prod_time:
            continue
        
        new_freshness = get_freshness(prod_time, reference_date)
        old_freshness = batch.get("freshness_status", "Fresh")
        
        if new_freshness != old_freshness:
            discount = DISCOUNT_MAP.get(new_freshness, 0)
            tray_color = FRESHNESS_COLORS.get(new_freshness, "green")
            
            q(db, "batch_inventory").update({
                "freshness_status": new_freshness,
                "tray_color": tray_color,
            }).eq("batch_id", batch["batch_id"]).execute()
            
            updated += 1
    
    return {"updated": updated, "reference_date": reference_date or str(datetime.now().date())}

def get_discount_rate(freshness: str) -> float:
    """Get discount rate for a freshness level."""
    return DISCOUNT_MAP.get(freshness, 0)

def get_tray_color(freshness: str) -> str:
    """Get tray color for a freshness level."""
    return FRESHNESS_COLORS.get(freshness, "green")


def cleanup_expired_batches(retention_days: int = 30):
    """Delete Expired batch_inventory records older than retention_days.
    
    Keeps expired records for the retention period to support waste analysis
    and audit, then removes them to prevent unbounded table growth.
    Returns count of deleted records.
    """
    db = get_db()
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")
    cursor = db.cursor()
    # Count before delete
    cursor.execute(
        "SELECT COUNT(*) FROM batch_inventory WHERE freshness_status = %s AND production_time < %s",
        ("Expired", cutoff)
    )
    count = cursor.fetchone()[0]
    if count > 0:
        cursor.execute(
            "DELETE FROM batch_inventory WHERE freshness_status = %s AND production_time < %s",
            ("Expired", cutoff)
        )
        db.commit()
    return {"deleted": count, "retention_days": retention_days, "cutoff": cutoff}

def get_sellable_batches():
    """Get all sellable batches (not expired), ordered by freshness (oldest first = FIFO)."""
    db = get_db()
    return q(db, "batch_inventory").select("*")         .gt("quantity", 0)         .neq("freshness_status", "Expired")         .order("production_time", desc=False)         .execute()
