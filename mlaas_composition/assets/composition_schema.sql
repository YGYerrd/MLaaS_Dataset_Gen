CREATE TABLE IF NOT EXISTS composition_runs (
  run_id TEXT PRIMARY KEY,
  input_fingerprint TEXT NOT NULL UNIQUE,
  manifest_path TEXT NOT NULL,
  source_db TEXT NOT NULL,
  executor_version TEXT NOT NULL,
  settings_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  summary_json TEXT
);
CREATE TABLE IF NOT EXISTS source_services (
  run_id TEXT NOT NULL REFERENCES composition_runs(run_id),
  service_id TEXT NOT NULL,
  family_id TEXT,
  sample_count INTEGER,
  snapshot_json TEXT NOT NULL,
  PRIMARY KEY (run_id, service_id)
);
CREATE TABLE IF NOT EXISTS compositions (
  run_id TEXT NOT NULL REFERENCES composition_runs(run_id),
  combination_id TEXT NOT NULL,
  family_id TEXT NOT NULL,
  composition_size INTEGER NOT NULL CHECK (composition_size >= 2),
  status TEXT NOT NULL CHECK (status IN ('pending','running','completed','failed')),
  attempt_count INTEGER NOT NULL DEFAULT 0,
  worker TEXT,
  started_at TEXT,
  finished_at TEXT,
  wall_s REAL,
  aggregation_provenance_json TEXT,
  failure_stage TEXT,
  failure_service_id TEXT,
  error_message TEXT,
  PRIMARY KEY (run_id, combination_id)
);
CREATE TABLE IF NOT EXISTS composition_members (
  run_id TEXT NOT NULL,
  combination_id TEXT NOT NULL,
  service_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  sample_count INTEGER,
  aggregation_weight REAL,
  status TEXT NOT NULL DEFAULT 'pending',
  training_wall_s REAL,
  state_hash TEXT,
  evidence_json TEXT,
  PRIMARY KEY (run_id, combination_id, service_id),
  FOREIGN KEY (run_id, combination_id) REFERENCES compositions(run_id, combination_id),
  FOREIGN KEY (run_id, service_id) REFERENCES source_services(run_id, service_id)
);
CREATE TABLE IF NOT EXISTS composition_metrics (
  run_id TEXT NOT NULL,
  combination_id TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  domain TEXT NOT NULL,
  unit TEXT,
  direction TEXT NOT NULL,
  value_num REAL,
  value_int INTEGER,
  value_bool INTEGER,
  value_text TEXT,
  value_json TEXT,
  PRIMARY KEY (run_id, combination_id, metric_name),
  FOREIGN KEY (run_id, combination_id) REFERENCES compositions(run_id, combination_id),
  CHECK ((value_num IS NOT NULL) + (value_int IS NOT NULL) +
         (value_bool IS NOT NULL) + (value_text IS NOT NULL) + (value_json IS NOT NULL) = 1)
);
CREATE INDEX IF NOT EXISTS idx_composition_status ON compositions(run_id, status);
CREATE INDEX IF NOT EXISTS idx_composition_member_service ON composition_members(run_id, service_id);
