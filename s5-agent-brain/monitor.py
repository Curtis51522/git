# Alert Monitor - background task that checks S1/S2/S3 for anomalies
# Runs periodically and generates alerts via alert_store.
import asyncio, logging, time, httpx
from typing import Dict, Any, List
from alert_store import add_alert, clear_expired

logger = logging.getLogger("s5.monitor")

# Data source URLs (main system on :8002)
S1_BATCH_URL = "http://localhost:8002/s1/batch_inventory"
S2_FORECAST_URL = "http://localhost:8002/s2/forecast"
S3_SCHEDULE_URL = "http://localhost:8002/s3/schedule"

CHECK_INTERVAL_SEC = 300  # 5 minutes


async def _fetch(url: str, timeout: int = 10) -> dict:
    """Fetch JSON from a URL."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


async def check_inventory() -> int:
    """Check S1 inventory for stockout, low stock, and waste risks.
    Returns alert count."""
    count = 0
    try:
        data = await _fetch(S1_BATCH_URL)
        items = data.get("inventory", [])

        # Aggregate by product + freshness
        from collections import defaultdict
        prod_fresh = defaultdict(lambda: {"total": 0, "Fresh": 0, "Day-1": 0})
        for item in items:
            pname = item.get("product_name", "")
            qty = item.get("quantity", 0)
            freshness = item.get("freshness_status", "Fresh")
            prod_fresh[pname]["total"] += qty
            prod_fresh[pname][freshness] = prod_fresh[pname].get(freshness, 0) + qty

        for pname, stats in prod_fresh.items():
            total = stats["total"]
            day1 = stats.get("Day-1", 0)
            fresh = stats.get("Fresh", 0)

            # Stockout
            if total == 0:
                add_alert("inventory", "critical", f"Stockout: {pname}",
                          f"{pname} has zero inventory across all batches.")
                count += 1
            elif total <= 10:
                add_alert("inventory", "warning", f"Low stock: {pname}",
                          f"{pname} has only {total} units remaining.")
                count += 1

            # Waste risk: Day-1 ratio > 40%
            if total > 0 and day1 > 0:
                day1_ratio = day1 / total
                if day1_ratio >= 0.6:
                    add_alert("inventory", "warning", f"High waste risk: {pname}",
                              f"{day1}/{total} ({day1_ratio:.0%}) of {pname} is Day-1 stock. "
                              f"Prioritize sale or discount to avoid spoilage.")
                    count += 1
                elif day1_ratio >= 0.4:
                    add_alert("inventory", "info", f"Waste watch: {pname}",
                              f"{day1}/{total} ({day1_ratio:.0%}) of {pname} is Day-1. Monitor closely.")
                    count += 1
    except Exception as e:
        logger.warning("Inventory check failed: %s", e)
    return count


async def check_forecast() -> int:
    """Check S2 forecast for anomalies vs inventory. Returns alert count."""
    count = 0
    try:
        today = time.strftime("%Y-%m-%d")
        url = f"{S2_FORECAST_URL}?days=1&product=all&date={today}"
        data = await _fetch(url)

        # Get batch inventory for comparison
        inv_data = await _fetch(S1_BATCH_URL)
        items = inv_data.get("inventory", [])
        from collections import defaultdict
        inv_map = defaultdict(int)
        for item in items:
            inv_map[item.get("product_name", "")] += item.get("quantity", 0)

        forecasts = data.get("forecasts", [])
        for fc in forecasts:
            pname = fc.get("product_name", "")
            pred = fc.get("predicted_demand", 0)
            lower = fc.get("lower_bound", 0)
            upper = fc.get("upper_bound", 0)
            inv_qty = inv_map.get(pname, 0)

            if pred > 0 and inv_qty < lower:
                add_alert("forecast", "warning", f"Undersupply risk: {pname}",
                          f"Forecast {lower}-{upper} but only {inv_qty} in stock. {lower - inv_qty} unit gap.")
                count += 1
            if inv_qty > upper * 2 and upper > 0:
                add_alert("forecast", "warning", f"Overstock risk: {pname}",
                          f"Inventory {inv_qty} vs forecast upper bound {upper}. {inv_qty - upper} surplus.")
                count += 1
    except Exception as e:
        logger.warning("Forecast check failed: %s", e)
    return count


async def check_schedule() -> int:
    """Check S3 schedule for staffing gaps on today date. Returns alert count."""
    count = 0
    try:
        data = await _fetch(S3_SCHEDULE_URL)
        schedule_list = data.get("schedule", [])
        today = time.strftime("%Y-%m-%d")
        today_entries = [s for s in schedule_list if s.get("date", "") == today]
        roles_today = set(s.get("role", "") for s in today_entries)
        if "baker" not in roles_today:
            add_alert("schedule", "critical", "No bakers scheduled today",
                      "No baker assigned for {}. Production impossible.".format(today))
            count += 1
        if "cashier" not in roles_today:
            add_alert("schedule", "critical", "No cashiers scheduled today",
                      "No cashier assigned for {}. Store cannot open.".format(today))
            count += 1
    except Exception as e:
        logger.warning("Schedule check failed: %s", e)
    return count


async def run_full_check() -> Dict[str, int]:
    """Run all checks and return alert counts per source."""
    results = {}
    results["inventory"] = await check_inventory()
    results["forecast"] = await check_forecast()
    results["schedule"] = await check_schedule()
    clear_expired()
    total = sum(results.values())
    if total > 0:
        logger.info("Monitor check complete: %d new alerts", total)
    return results


async def start_monitor(interval_sec: int = CHECK_INTERVAL_SEC):
    """Start the background monitoring loop."""
    logger.info("Alert monitor started (interval=%ds)", interval_sec)
    while True:
        await asyncio.sleep(interval_sec)
        try:
            await run_full_check()
        except Exception as e:
            logger.error("Monitor loop error: %s", e)