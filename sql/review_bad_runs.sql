WITH latest_failure_ids AS (
    SELECT service_id, MAX(failure_id) AS latest_failure_id
    FROM service_failures
    WHERE service_id IS NOT NULL
    GROUP BY service_id
),
latest_failures AS (
    SELECT sf.service_id, sf.failure_stage, sf.error_message
    FROM service_failures sf
    JOIN latest_failure_ids lf
      ON lf.service_id = sf.service_id
     AND lf.latest_failure_id = sf.failure_id
)
SELECT
    s.service_id AS run_id,
    s.status,
    lf.failure_stage,
    lf.error_message AS failure_message
FROM services s
JOIN latest_failures lf ON lf.service_id = s.service_id
WHERE lower(COALESCE(s.status, '')) = 'failed'
ORDER BY s.created_at DESC, s.service_id;
