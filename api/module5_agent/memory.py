"""
Memory Store ? Episodic + Reflective Memory for S5 Agent Query.

Park et al. (2023) inspired: Working Memory (current query) +
Episodic Memory (query/response pairs) + Reflective Memory (LLM-synthesized insights).

Storage: MySQL (same bakery_ai database). No vector DB needed at this scale.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from db.mysql_client import get_db

logger = logging.getLogger("s5.memory")

# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------
def init_memory_tables():
    """Create memory tables if they do not exist. Called on first use."""
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS s5_memory_episodic (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            session_id      VARCHAR(64)  NOT NULL,
            query           TEXT         NOT NULL,
            intent          VARCHAR(32)  NOT NULL DEFAULT "general",
            product         VARCHAR(64)  DEFAULT "",
            target_date     VARCHAR(16)  DEFAULT "",
            response        TEXT         NOT NULL,
            data_snapshot   JSON         NOT NULL DEFAULT ("{}"),
            importance      FLOAT        NOT NULL DEFAULT 0.5,
            created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session (session_id, created_at),
            INDEX idx_product (product, created_at),
            INDEX idx_importance (importance DESC, created_at DESC)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS s5_memory_reflections (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            session_id      VARCHAR(64)  NOT NULL,
            period_start    DATETIME     NULL,
            period_end      DATETIME     NULL,
            topic           VARCHAR(128) DEFAULT "",
            insight         TEXT         NOT NULL,
            evidence_ids    JSON         NULL,
            created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session_period (session_id, period_start)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    db.commit()
    logger.info("Memory tables ready")


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------
def _compute_importance(intent: str, data_snapshot: dict) -> float:
    """Score episode importance for later retrieval weighting."""
    base = {
        "stock_query": 0.6,
        "cross_source_audit": 0.7,
        "schedule_audit": 0.5,
        "waste_analysis": 0.5,
        "promo_eval": 0.4,
    }.get(intent, 0.3)

    # Boost for critical situations
    forecast = data_snapshot.get("forecast", 0)
    inventory = data_snapshot.get("inventory", 0)
    if forecast > 0 and inventory / forecast < 0.2:
        base = min(1.0, base + 0.3)  # stockout risk
    elif forecast > 0 and inventory / forecast > 2.0:
        base = min(1.0, base + 0.2)  # overstock

    return round(base, 2)


# ---------------------------------------------------------------------------
# Snapshot truncation
# ---------------------------------------------------------------------------
def _truncate_snapshot(snapshot: dict, max_bytes: int = 5000) -> dict:
    """Truncate data_snapshot to max_bytes by dropping large fields."""
    text = json.dumps(snapshot, default=str)
    if len(text) <= max_bytes:
        return snapshot

    # Keep only essential keys
    essential = {}
    for key in ("forecast", "inventory", "capacity", "product", "intent", "status"):
        if key in snapshot:
            essential[key] = snapshot[key]
    text2 = json.dumps(essential, default=str)
    if len(text2) <= max_bytes:
        return essential

    # Last resort: just keep forecast + inventory
    minimal = {k: snapshot[k] for k in ("forecast", "inventory") if k in snapshot}
    return minimal if minimal else {"_truncated": True}


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------
class MemoryStore:
    """Multi-turn conversation memory for S5 Agent Query."""

    # ---- Episodic Memory ------------------------------------------------
    def store_episode(
        self,
        session_id: str,
        query: str,
        intent: str,
        product: str = "",
        target_date: str = "",
        response: str = "",
        data_snapshot: dict | None = None,
    ) -> int:
        """Store a query-response pair. Returns the episode ID."""
        session_id = session_id or "default"
        data_snapshot = data_snapshot or {}
        data_snapshot = _truncate_snapshot(data_snapshot)
        importance = _compute_importance(intent, data_snapshot)

        db = get_db()
        cur = db.cursor()
        cur.execute(
            """INSERT INTO s5_memory_episodic
               (session_id, query, intent, product, target_date, response, data_snapshot, importance)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                session_id, query, intent, product, target_date, response,
                json.dumps(data_snapshot, default=str), importance,
            ),
        )
        db.commit()
        return cur.lastrowid

    # ---- Retrieval ------------------------------------------------------
    def retrieve_episodes(
        self,
        session_id: str,
        product: str = "",
        target_date: str = "",
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve relevant past episodes, ordered by composite score."""
        session_id = session_id or "default"
        db = get_db()
        cur = db.cursor(dictionary=True)

        clauses = ["session_id = %s"]
        params = [session_id]

        if product:
            clauses.append("product = %s")
            params.append(product)
        if target_date:
            clauses.append("target_date = %s")
            params.append(target_date)

        where = "WHERE " + " AND ".join(clauses)
        sql = f"SELECT * FROM s5_memory_episodic {where} ORDER BY importance DESC, created_at DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

        for row in rows:
            if isinstance(row.get("data_snapshot"), str):
                try:
                    row["data_snapshot"] = json.loads(row["data_snapshot"])
                except json.JSONDecodeError:
                    row["data_snapshot"] = {}
            if row.get("created_at") and hasattr(row["created_at"], "isoformat"):
                row["created_at"] = row["created_at"].isoformat()

        return rows

    # ---- Context Builder ------------------------------------------------
    def get_recent_context(self, session_id: str, n: int = 3, product: str = "") -> str:
        """Build a short context string from recent episodes for prompt prepending."""
        episodes = self.retrieve_episodes(session_id, product=product, limit=n)
        if not episodes:
            return ""

        lines = ["[Previous conversation context]"]
        for ep in episodes:
            q_short = ep["query"][:120]
            r_short = ep["response"][:120]
            lines.append("Q: " + q_short)
            lines.append("A: " + r_short)
        lines.append("[End of context]")
        return chr(10).join(lines) + chr(10)

    # ---- Reflective Memory ----------------------------------------------
    def generate_reflection(self, session_id: str) -> Optional[str]:
        """Synthesize insights from recent episodes via DeepSeek.
        
        Returns the insight text, or None if not enough data / LLM unavailable.
        """
        episodes = self.retrieve_episodes(session_id, limit=20)
        if len(episodes) < 5:
            logger.debug("Not enough episodes for reflection (%d)", len(episodes))
            return None

        try:
            from api.module5_agent.llm_client import compose_reflection
        except ImportError:
            logger.warning("compose_reflection not available")
            return None

        snapshot = [
            {
                "query": ep["query"],
                "intent": ep["intent"],
                "product": ep.get("product", ""),
                "response": ep["response"][:200],
                "date": ep.get("created_at", ""),
            }
            for ep in episodes
        ]

        insight = compose_reflection(session_id, snapshot)
        if not insight:
            return None

        # Store reflection
        db = get_db()
        cur = db.cursor()
        cur.execute(
            """INSERT INTO s5_memory_reflections
               (session_id, period_start, period_end, topic, insight, evidence_ids)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                session_id,
                episodes[-1].get("created_at") if episodes else None,
                episodes[0].get("created_at") if episodes else None,
                "weekly_summary",
                insight,
                json.dumps([ep.get("id") for ep in episodes]),
            ),
        )
        db.commit()
        logger.info("Reflection stored for session %s", session_id)
        return insight

    def get_reflections(self, session_id: str, limit: int = 3) -> list[dict]:
        """Retrieve recent reflections for a session."""
        session_id = session_id or "default"
        db = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            """SELECT * FROM s5_memory_reflections
               WHERE session_id = %s
               ORDER BY created_at DESC LIMIT %s""",
            (session_id, limit),
        )
        rows = cur.fetchall()
        for row in rows:
            if row.get("created_at") and hasattr(row["created_at"], "isoformat"):
                row["created_at"] = row["created_at"].isoformat()
        return rows


    # ---- Auto Reflection ----------------------------------------------
    def auto_reflect_all(self) -> list[dict]:
        """Run reflection on all active sessions with sufficient episodes.
        
        Called once daily by the B1 monitor. Returns list of generated reflections.
        """
        db = get_db()
        cur = db.cursor()
        # Find sessions with >= 5 episodes in the last 48 hours
        cur.execute("""
            SELECT session_id, COUNT(*) as cnt
            FROM s5_memory_episodic
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 2 DAY)
            GROUP BY session_id
            HAVING cnt >= 5
            ORDER BY cnt DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        
        results = []
        for (session_id, cnt) in rows:
            try:
                insight = self.generate_reflection(session_id)
                if insight:
                    results.append({
                        "session_id": session_id,
                        "episode_count": cnt,
                        "insight": insight,
                    })
                    logger.info("Auto-reflection for %s: %s", session_id, insight[:80])
            except Exception as e:
                logger.warning("Auto-reflection failed for %s: %s", session_id, e)
        
        return results


# Singleton
_memory_store: Optional[MemoryStore] = None


def get_memory() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        init_memory_tables()
        _memory_store = MemoryStore()
    return _memory_store
