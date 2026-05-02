PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS services (
  service_id              TEXT PRIMARY KEY,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
  status                  TEXT NOT NULL DEFAULT 'completed',
  case_name               TEXT,
  task_family             TEXT,
  task_type               TEXT,
  modality                TEXT,
  input_schema            TEXT,
  output_schema           TEXT,
  dataset                 TEXT,
  dataset_name            TEXT,
  dataset_config          TEXT,
  train_split             TEXT,
  benchmark_split         TEXT,
  model_type              TEXT,
  model_id                TEXT,
  hf_task                 TEXT,
  training_regime         TEXT,
  dataset_variant         TEXT,
  split_variant           TEXT,
  knob_variant            TEXT,
  service_config_json     TEXT,
  registry_metadata_json  TEXT,
  functional_attributes_json TEXT,
  metadata_json           TEXT
);

CREATE INDEX IF NOT EXISTS idx_services_task ON services(task_family, modality);
CREATE INDEX IF NOT EXISTS idx_services_dataset ON services(dataset_name, dataset_config);
CREATE INDEX IF NOT EXISTS idx_services_model ON services(model_id, model_type);
CREATE INDEX IF NOT EXISTS idx_services_training_regime ON services(training_regime);

CREATE TABLE IF NOT EXISTS service_metrics (
  metric_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  service_id     TEXT NOT NULL,
  metric_name    TEXT NOT NULL,
  domain         TEXT NOT NULL,
  unit           TEXT,
  direction      TEXT NOT NULL DEFAULT 'neutral',
  value_num      REAL,
  value_int      INTEGER,
  value_bool     INTEGER,
  value_text     TEXT,
  value_json     TEXT,
  created_at     TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (service_id) REFERENCES services(service_id) ON DELETE CASCADE,
  CHECK (domain IN ('quality','qos','performance','latency','runtime','resource','cost','reliability','explainability','metadata')),
  CHECK (direction IN ('higher_better','lower_better','neutral')),
  CHECK (value_bool IS NULL OR value_bool IN (0,1)),
  CHECK (
    (value_num  IS NOT NULL) +
    (value_int  IS NOT NULL) +
    (value_bool IS NOT NULL) +
    (value_text IS NOT NULL) +
    (value_json IS NOT NULL)
    = 1
  )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_service_metric ON service_metrics(service_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_service_metrics_domain ON service_metrics(domain);

CREATE TABLE IF NOT EXISTS service_artifacts (
  artifact_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  service_id      TEXT NOT NULL,
  artifact_type   TEXT NOT NULL,
  artifact_uri    TEXT NOT NULL,
  metadata_json   TEXT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (service_id) REFERENCES services(service_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_service_artifacts_service ON service_artifacts(service_id);
CREATE INDEX IF NOT EXISTS idx_service_artifacts_type ON service_artifacts(artifact_type);

CREATE TABLE IF NOT EXISTS service_split_provenance (
  service_id              TEXT NOT NULL,
  split_name              TEXT NOT NULL,
  samples_count           INTEGER,
  data_distribution_json  TEXT,
  split_config_json       TEXT,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (service_id, split_name),
  FOREIGN KEY (service_id) REFERENCES services(service_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_service_split_provenance_service
ON service_split_provenance(service_id);

CREATE TABLE IF NOT EXISTS service_failures (
  failure_id           INTEGER PRIMARY KEY AUTOINCREMENT,
  service_id           TEXT,
  row_index            INTEGER,
  case_name            TEXT,
  manifest_group_id    TEXT,
  failure_stage        TEXT NOT NULL,
  error_message        TEXT,
  resolved_config_json TEXT,
  traceback_text       TEXT,
  created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_service_failures_service_id ON service_failures(service_id);
CREATE INDEX IF NOT EXISTS idx_service_failures_group ON service_failures(manifest_group_id);
CREATE INDEX IF NOT EXISTS idx_service_failures_stage ON service_failures(failure_stage);

CREATE VIEW IF NOT EXISTS v_service_metrics AS
SELECT
  sm.metric_id,
  sm.service_id,
  s.task_family,
  s.modality,
  s.dataset_name,
  s.model_id,
  s.training_regime,
  sm.metric_name,
  sm.domain,
  sm.unit,
  sm.direction,
  sm.value_num,
  sm.value_int,
  sm.value_bool,
  sm.value_text,
  sm.value_json,
  sm.created_at
FROM service_metrics sm
JOIN services s ON s.service_id = sm.service_id;
