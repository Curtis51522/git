CREATE TABLE IF NOT EXISTS deployment_ready (
    marker_key VARCHAR(64) NOT NULL PRIMARY KEY,
    schema_version VARCHAR(64) NOT NULL,
    initialized_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO deployment_ready (marker_key, schema_version)
SELECT candidate.marker_key, candidate.schema_version
FROM (
    SELECT 'release-snapshot' AS marker_key, '1' AS schema_version
) AS candidate
WHERE (
    SELECT COUNT(DISTINCT table_name)
    FROM information_schema.tables
    WHERE table_schema = DATABASE()
      AND table_type = 'BASE TABLE'
      AND table_name IN (
          'orders',
          'order_items',
          'products',
          'raw_materials',
          'users'
      )
) = 5
ON DUPLICATE KEY UPDATE
    schema_version = candidate.schema_version,
    initialized_at = CURRENT_TIMESTAMP;
