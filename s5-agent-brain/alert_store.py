# Alert Store - persistent alert storage (JSON file)
# Tracks alert history with acknowledgment state.
import json, os, logging, time
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("s5.alerts")

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts.json")

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _load() -> dict:
    if not os.path.exists(STORE_PATH):
        return {"alerts": [], "next_id": 1}
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"alerts": [], "next_id": 1}


def _save(store: dict) -> None:
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, default=str)


def add_alert(source: str, severity: str, title: str, detail: str, ttl_sec: int = 86400) -> Optional[int]:
    """Add an alert. Returns alert_id. Deduplicates by title within TTL window."""
    store = _load()
    now = time.time()
    for a in store["alerts"]:
        if a.get("title") == title and not a.get("acknowledged"):
            # Update existing alert with fresh data
            a["detail"] = detail
            a["severity"] = severity
            a["created_at"] = datetime.now().isoformat()
            a["created_ts"] = now
            _save(store)
            return a["alert_id"]
    alert_id = store["next_id"]
    store["next_id"] += 1
    store["alerts"].append({
        "alert_id": alert_id,
        "source": source,
        "severity": severity,
        "title": title,
        "detail": detail,
        "acknowledged": False,
        "created_at": datetime.now().isoformat(),
        "created_ts": now,
        "ttl_sec": ttl_sec,
    })
    _save(store)
    logger.info("Alert #%d [%s] %s: %s", alert_id, severity, source, title)
    return alert_id


def get_alerts(limit: int = 100, unacked_only: bool = False) -> List[Dict[str, Any]]:
    """Get alerts, newest first. Auto-expires old ones."""
    store = _load()
    now = time.time()
    active = []
    changed = False
    for a in store["alerts"]:
        if now - a.get("created_ts", 0) > a.get("ttl_sec", 86400):
            changed = True
            continue
        if unacked_only and a.get("acknowledged"):
            continue
        active.append(a)
    if changed:
        store["alerts"] = active
        _save(store)
    active.sort(key=lambda x: -x.get("created_ts", 0))
    return active[:limit]


def acknowledge(alert_id: int = None, ack_all: bool = False) -> int:
    """Acknowledge one or all alerts. Returns count acknowledged."""
    store = _load()
    count = 0
    for a in store["alerts"]:
        if ack_all or a.get("alert_id") == alert_id:
            if not a.get("acknowledged"):
                a["acknowledged"] = True
                a["acked_at"] = datetime.now().isoformat()
                count += 1
            if not ack_all:
                break
    if count > 0:
        _save(store)
    return count


def get_unacked_count() -> int:
    """Count unacknowledged, non-expired alerts."""
    store = _load()
    now = time.time()
    count = 0
    for a in store["alerts"]:
        if now - a.get("created_ts", 0) < a.get("ttl_sec", 86400) and not a.get("acknowledged"):
            count += 1
    return count


def clear_expired() -> int:
    """Remove expired alerts. Returns count removed."""
    store = _load()
    now = time.time()
    old_len = len(store["alerts"])
    store["alerts"] = [a for a in store["alerts"] if now - a.get("created_ts", 0) < a.get("ttl_sec", 86400)]
    removed = old_len - len(store["alerts"])
    if removed > 0:
        _save(store)
    return removed