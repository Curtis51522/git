"""Alert Store -- persistent storage for B1 Proactive Anomaly Monitor alerts.

Stores alerts in MySQL (same database as the rest of the system).
Alerts are append-only: never deleted, only marked acknowledged.
"""

import json
import logging
from typing import List, Dict, Optional

from db.mysql_client import get_db

logger = logging.getLogger("s5.alerts")

# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------
def init_alerts_table():
    """Create the alerts table if it does not exist.  Called on server startup."""
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id    INT AUTO_INCREMENT PRIMARY KEY,
            created_at  DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
            severity    VARCHAR(10) NOT NULL DEFAULT 'info',
            rule        VARCHAR(10) NOT NULL DEFAULT 'general',
            message     TEXT        NOT NULL,
            params_json JSON        NOT NULL DEFAULT ('{}'),
            root_cause  TEXT        NULL,
            acknowledged TINYINT    NOT NULL DEFAULT 0,
            acked_at    DATETIME    NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    db.commit()
    logger.info("alerts table ready")


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------
def create_alert(severity: str, rule: str, message: str,
                 params: Optional[Dict] = None,
                 root_cause: Optional[str] = None) -> int:
    """Insert a new alert and return its alert_id."""
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """INSERT INTO alerts (severity, rule, message, params_json, root_cause)
           VALUES (%s, %s, %s, %s, %s)""",
        (severity, rule, message,
         json.dumps(params or {}, default=str), root_cause),
    )
    db.commit()
    alert_id = cur.lastrowid
    logger.debug("Alert #%d created [%s/%s]: %s", alert_id, severity, rule, message[:80])
    return alert_id


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def list_alerts(since: Optional[str] = None,
                severity: Optional[str] = None,
                acknowledged: Optional[bool] = None,
                limit: int = 50) -> List[Dict]:
    """Fetch alerts, newest first, with optional filters."""
    db = get_db()
    cur = db.cursor(dictionary=True)

    clauses = []
    params = []

    if since:
        clauses.append("created_at >= %s")
        params.append(since)
    if severity:
        clauses.append("severity = %s")
        params.append(severity)
    if acknowledged is not None:
        clauses.append("acknowledged = %s")
        params.append(1 if acknowledged else 0)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM alerts {where} ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    cur.execute(sql, tuple(params))
    rows = cur.fetchall()

    for row in rows:
        if isinstance(row.get("params_json"), str):
            row["params_json"] = json.loads(row["params_json"])
        for dt_col in ("created_at", "acked_at"):
            if row.get(dt_col) and hasattr(row[dt_col], "isoformat"):
                row[dt_col] = row[dt_col].isoformat()
        row["acknowledged"] = bool(row.get("acknowledged", 0))

    return rows


def get_unacked_count() -> int:
    """Return number of unacknowledged alerts (for frontend badge)."""
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0 AND severity != 'info'")
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------
def acknowledge_alert(alert_id: int) -> bool:
    """Mark a single alert as acknowledged."""
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "UPDATE alerts SET acknowledged = 1, acked_at = NOW() WHERE alert_id = %s",
        (alert_id,),
    )
    db.commit()
    return cur.rowcount > 0


def acknowledge_all() -> int:
    """Acknowledge all unacknowledged alerts.  Returns count affected."""
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "UPDATE alerts SET acknowledged = 1, acked_at = NOW() WHERE acknowledged = 0"
    )
    db.commit()
    return cur.rowcount


def enrich_alert(alert_id: int, root_cause: str):
    """Attach root-cause analysis result to an existing alert."""
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "UPDATE alerts SET root_cause = %s WHERE alert_id = %s",
        (root_cause, alert_id),
    )
    db.commit()
