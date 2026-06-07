# Memory Store - Persistent agent memory for S5
# Stores daily snapshots and query logs in MySQL.
# Enforces 1000-row cap on query_log. Provides baseline comparison.
import json, logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger("s5.memory")

QUERY_LOG_CAP = 1000

_db = None

def _get_db():
    global _db
    if _db is None:
        import sys, os
        _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from db.mysql_client import get_db
        _db = get_db()
    return _db


def save_snapshot(date_str: str, data: Dict[str, Any]) -> bool:
    """Save daily inventory/forecast/waste/profit snapshot. Upserts on date."""
    try:
        db = _get_db()
        cur = db.cursor()
        json_str = json.dumps(data, default=str)
        cur.execute(
            "INSERT INTO s5_daily_snapshot (snapshot_date, data) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE data = VALUES(data)",
            (date_str, json_str))
        db.commit()
        cur.close()
        logger.info("Snapshot saved for %s", date_str)
        return True
    except Exception as e:
        logger.warning("save_snapshot failed: %s", e)
        return False


def save_query(query: str, intent: str, product: str,
               agent_results: Dict, decision: str = "",
               summary: str = "", target_date: str = "") -> bool:
    """Log a user query. Enforces 1000-row cap (deletes oldest if exceeded)."""
    try:
        db = _get_db()
        cur = db.cursor()
        json_str = json.dumps(agent_results, default=str)
        cur.execute(
            "INSERT INTO s5_query_log (query_text, intent, product, agent_results, "
            "decision, llm_summary, target_date) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (query, intent, product, json_str, decision, summary, target_date))
        # Enforce cap
        cur.execute("SELECT COUNT(*) FROM s5_query_log")
        count = cur.fetchone()[0]
        if count > QUERY_LOG_CAP:
            cur.execute(
                "DELETE FROM s5_query_log ORDER BY created_at ASC LIMIT %s",
                (count - QUERY_LOG_CAP,))
        db.commit()
        cur.close()
        return True
    except Exception as e:
        logger.warning("save_query failed: %s", e)
        return False


def get_baseline(product: str, weekday: int, days_back: int = 28) -> Optional[Dict]:
    """Get average metrics for same product + weekday over past N days."""
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT data FROM s5_daily_snapshot "
            "WHERE snapshot_date >= DATE_SUB(CURRENT_DATE, INTERVAL %s DAY) "
            "AND WEEKDAY(snapshot_date) = %s "
            "ORDER BY snapshot_date DESC",
            (days_back, weekday))
        rows = cur.fetchall()
        cur.close()
        if not rows:
            return None
        snapshots = [json.loads(r[0]) for r in rows]
        avg = {"inventory": {}, "forecast": {}, "waste": {}, "profit": {}}
        for key in avg:
            if snapshots[0].get(key):
                for pname in snapshots[0][key]:
                    vals = [s.get(key, {}).get(pname, 0) for s in snapshots if pname in s.get(key, {})]
                    avg[key][pname] = round(sum(vals) / max(len(vals), 1), 1)
        avg["sample_count"] = len(rows)
        return avg
    except Exception as e:
        logger.warning("get_baseline failed: %s", e)
        return None


def compare_baseline(current_data: Dict, product: str = "all") -> List[Dict]:
    """Compare current snapshot to 4-week same-weekday baseline. Returns anomalies."""
    try:
        today = datetime.now()
        baseline = get_baseline(product, today.weekday())
        if not baseline:
            return []
        anomalies = []
        curr_inv = current_data.get("inventory", {})
        base_inv = baseline.get("inventory", {})
        for pname, curr_qty in curr_inv.items():
            base_qty = base_inv.get(pname, curr_qty)
            if base_qty > 0:
                ratio = curr_qty / base_qty
                if ratio < 0.5:
                    anomalies.append({
                        "product": pname,
                        "metric": "inventory",
                        "current": int(curr_qty),
                        "baseline": round(base_qty, 1),
                        "deviation": f"{ratio:.0%}",
                        "severity": "warning"
                    })
                elif ratio > 2.0:
                    anomalies.append({
                        "product": pname,
                        "metric": "inventory",
                        "current": int(curr_qty),
                        "baseline": round(base_qty, 1),
                        "deviation": f"{ratio:.0%}",
                        "severity": "warning"
                    })
        return anomalies
    except Exception as e:
        logger.warning("compare_baseline failed: %s", e)
        return []


def get_context(product: str, intent: str, days: int = 14) -> str:
    """Return compact historical context string for LLM synthesis."""
    try:
        today = datetime.now()
        weekday = today.weekday()
        baseline = get_baseline(product, weekday, days_back=28)
        if not baseline or baseline.get("sample_count", 0) < 2:
            return ""
        parts = []
        base_inv = baseline.get("inventory", {})
        base_waste = baseline.get("waste", {})
        if product == "all":
            total_base = sum(base_inv.values())
            total_waste = sum(base_waste.values())
            parts.append(f"4-week {today.strftime('%A')} avg stock: {total_base:.0f} units, avg waste: {total_waste:.0f}")
        elif product in base_inv:
            parts.append(f"4-week {today.strftime('%A')} avg stock: {base_inv[product]:.0f}, avg waste: {base_waste.get(product, 0):.0f}")
        return " | ".join(parts) if parts else ""
    except Exception as e:
        logger.warning("get_context failed: %s", e)
        return ""
