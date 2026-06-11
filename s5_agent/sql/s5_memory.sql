-- S5 v3: Agentic Memory tables
-- Run: mysql -u root bakery_ai < s5_memory.sql

CREATE TABLE IF NOT EXISTS s5_daily_snapshot (
    id INT AUTO_INCREMENT PRIMARY KEY,
    snapshot_date DATE UNIQUE NOT NULL,
    data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_date (snapshot_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS s5_query_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_text TEXT NOT NULL,
    intent VARCHAR(64) NOT NULL,
    product VARCHAR(128) NOT NULL,
    agent_results JSON NOT NULL,
    decision TEXT,
    llm_summary TEXT,
    target_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_intent (intent),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Cleanup event: delete snapshots older than 30 days (runs daily at 3am)
DROP EVENT IF EXISTS s5_cleanup_snapshots;
CREATE EVENT s5_cleanup_snapshots
ON SCHEDULE EVERY 1 DAY STARTS CURRENT_DATE + INTERVAL 1 DAY
DO DELETE FROM s5_daily_snapshot WHERE snapshot_date < DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY);
