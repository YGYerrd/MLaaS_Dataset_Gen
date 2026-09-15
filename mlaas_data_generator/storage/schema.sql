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
  artifact_sha256 TEXT,
  size_bytes      INTEGER,
  artifact_format TEXT,
  artifact_role   TEXT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (service_id) REFERENCES services(service_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_service_artifacts_service ON service_artifacts(service_id);
CREATE INDEX IF NOT EXISTS idx_service_artifacts_type ON service_artifacts(artifact_type);

CREATE TABLE IF NOT EXISTS model_lineages (
  lineage_id                  TEXT PRIMARY KEY,
  fl_run_id                   TEXT,
  base_model_id               TEXT,
  base_model_hash             TEXT,
  initial_checkpoint_uri      TEXT,
  initial_checkpoint_sha256   TEXT,
  provenance_json             TEXT NOT NULL,
  created_at                  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_model_lineages_fl_run ON model_lineages(fl_run_id);

-- Immutable base checkpoint storage for a lineage.  The model-state digest is
-- deliberately distinct from both the parameter-schema digest and the digest
-- of the serialized checkpoint artefact.  This lets checkpoint containers and
-- compression change without changing the identity of the represented state.
CREATE TABLE IF NOT EXISTS lineage_checkpoint_storage (
  lineage_id                  TEXT PRIMARY KEY,
  storage_policy_version      TEXT NOT NULL,
  state_hash_spec_version     TEXT NOT NULL,
  parameter_schema_hash       TEXT,
  base_state_sha256           TEXT,
  base_artifact_ref           TEXT NOT NULL,
  base_artifact_sha256        TEXT NOT NULL,
  base_compression_codec      TEXT,
  base_compression_settings_json TEXT,
  base_logical_bytes          INTEGER,
  base_physical_bytes         INTEGER,
  metadata_json               TEXT,
  created_at                  TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (lineage_id) REFERENCES model_lineages(lineage_id) ON DELETE CASCADE,
  CHECK (base_logical_bytes IS NULL OR base_logical_bytes >= 0),
  CHECK (base_physical_bytes IS NULL OR base_physical_bytes >= 0)
);
CREATE INDEX IF NOT EXISTS idx_lineage_checkpoint_storage_state
ON lineage_checkpoint_storage(base_state_sha256);

CREATE TABLE IF NOT EXISTS service_generation_provenance (
  service_id                  TEXT PRIMARY KEY,
  provenance_version          INTEGER NOT NULL,
  architecture_id             TEXT,
  model_revision              TEXT,
  dataset_revision            TEXT,
  parameter_schema_hash       TEXT,
  label_map_hash              TEXT,
  preprocessing_contract_hash TEXT,
  dataset_contract_hash       TEXT,
  logical_dataset_contract_hash TEXT,
  partition_manifest_hash     TEXT,
  base_model_id               TEXT,
  base_model_hash             TEXT,
  final_model_hash            TEXT,
  lineage_id                  TEXT,
  generation_mode             TEXT,
  family_id                   TEXT,
  family_contract_version     TEXT,
  initial_state_hash          TEXT,
  family_document_ref         TEXT,
  family_document_hash        TEXT,
  manifest_reference          TEXT,
  composition_compatible      INTEGER,
  parent_checkpoint_id        TEXT,
  cohort_id                   TEXT,
  fl_run_id                   TEXT,
  client_id                   TEXT,
  round_id                    INTEGER,
  compatibility_contract_hash TEXT,
  parent_global_round         INTEGER,
  parent_state_hash           TEXT,
  training_universe_hash      TEXT,
  benchmark_cohort_hash       TEXT,
  scheduled_to_participate    INTEGER,
  participated                INTEGER,
  provenance_json             TEXT NOT NULL,
  created_at                  TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (service_id) REFERENCES services(service_id) ON DELETE CASCADE,
  FOREIGN KEY (lineage_id) REFERENCES model_lineages(lineage_id)
);
CREATE INDEX IF NOT EXISTS idx_service_generation_provenance_architecture ON service_generation_provenance(architecture_id);
CREATE INDEX IF NOT EXISTS idx_service_generation_provenance_lineage ON service_generation_provenance(lineage_id);
CREATE INDEX IF NOT EXISTS idx_service_generation_provenance_fl_run ON service_generation_provenance(fl_run_id);

-- Shared-parent generation has no rounds, participation, or global aggregation.
CREATE TABLE IF NOT EXISTS parent_checkpoints (
  parent_checkpoint_id TEXT PRIMARY KEY,
  family_id TEXT NOT NULL,
  selection_hash TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('preparing','training','completed','failed','interrupted')),
  parent_state_hash TEXT,
  document_json TEXT NOT NULL,
  error_message TEXT,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS service_cohorts (
  cohort_id TEXT PRIMARY KEY,
  parent_checkpoint_id TEXT NOT NULL REFERENCES parent_checkpoints(parent_checkpoint_id),
  manifest_hash TEXT NOT NULL,
  scheduled_clients INTEGER NOT NULL CHECK(scheduled_clients > 0),
  status TEXT NOT NULL CHECK(status IN ('prepared','running','completed','partial','failed','interrupted')),
  document_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS cohort_clients (
  service_id TEXT PRIMARY KEY,
  cohort_id TEXT NOT NULL REFERENCES service_cohorts(cohort_id),
  client_id TEXT NOT NULL,
  recipe_hash TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed','interrupted')),
  sample_count INTEGER,
  partition_hash TEXT,
  final_state_hash TEXT,
  document_json TEXT NOT NULL,
  error_message TEXT,
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(cohort_id, client_id)
);

-- Per-service persistence decision and measured storage outcome.  Patch
-- semantics and compression are separate fields: decompression recovers the
-- patch stream, while patch semantics explains how that stream reconstructs
-- the target tensor state from the lineage base.
CREATE TABLE IF NOT EXISTS service_checkpoint_storage (
  service_id                         TEXT PRIMARY KEY,
  lineage_id                         TEXT,
  storage_policy_version             TEXT NOT NULL,
  storage_mode                       TEXT NOT NULL,
  state_hash_spec_version            TEXT NOT NULL,
  parameter_schema_hash              TEXT,
  base_state_sha256                  TEXT,
  target_state_sha256                TEXT,
  patch_semantics                    TEXT,
  patch_semantics_version            TEXT,
  patch_compression_codec            TEXT,
  patch_compression_settings_json    TEXT,
  patch_artifact_ref                 TEXT,
  patch_artifact_sha256              TEXT,
  full_artifact_ref                  TEXT,
  full_artifact_sha256               TEXT,
  reconstruction_artifact_ref        TEXT NOT NULL,
  reconstruction_artifact_sha256     TEXT NOT NULL,
  target_logical_bytes               INTEGER,
  full_checkpoint_physical_bytes     INTEGER,
  patch_logical_bytes                INTEGER,
  patch_physical_bytes               INTEGER,
  reconstruction_physical_bytes      INTEGER,
  selected_physical_bytes            INTEGER,
  savings_bytes                      INTEGER,
  savings_ratio                      REAL,
  round_trip_status                  TEXT NOT NULL DEFAULT 'not_run',
  round_trip_verified_at             TEXT,
  round_trip_error                   TEXT,
  metadata_json                      TEXT,
  parent_global_round                INTEGER,
  explicit_base_state_ref            TEXT,
  created_at                         TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (service_id) REFERENCES services(service_id) ON DELETE CASCADE,
  FOREIGN KEY (lineage_id) REFERENCES model_lineages(lineage_id),
  CHECK (storage_mode IN ('shared_base_exact_patch', 'full_object')),
  CHECK (target_logical_bytes IS NULL OR target_logical_bytes >= 0),
  CHECK (full_checkpoint_physical_bytes IS NULL OR full_checkpoint_physical_bytes >= 0),
  CHECK (patch_logical_bytes IS NULL OR patch_logical_bytes >= 0),
  CHECK (patch_physical_bytes IS NULL OR patch_physical_bytes >= 0),
  CHECK (reconstruction_physical_bytes IS NULL OR reconstruction_physical_bytes >= 0),
  CHECK (selected_physical_bytes IS NULL OR selected_physical_bytes >= 0),
  CHECK (round_trip_status IN ('not_run', 'passed', 'failed'))
);
CREATE INDEX IF NOT EXISTS idx_service_checkpoint_storage_lineage
ON service_checkpoint_storage(lineage_id);
CREATE INDEX IF NOT EXISTS idx_service_checkpoint_storage_mode
ON service_checkpoint_storage(storage_mode);
CREATE INDEX IF NOT EXISTS idx_service_checkpoint_storage_round_trip
ON service_checkpoint_storage(round_trip_status);

-- Append-only execution evidence belongs to the dataset, not to a manifest.
-- No cascading FK: replacing a service on retry must retain earlier attempts.
CREATE TABLE IF NOT EXISTS service_execution_provenance (
  execution_id TEXT PRIMARY KEY,
  service_id TEXT NOT NULL,
  run_id TEXT,
  started_at TEXT NOT NULL,
  document_sha256 TEXT NOT NULL,
  document_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_service_execution_provenance_service
ON service_execution_provenance(service_id);
CREATE TRIGGER IF NOT EXISTS service_execution_provenance_no_replace
BEFORE INSERT ON service_execution_provenance
WHEN EXISTS (SELECT 1 FROM service_execution_provenance WHERE execution_id = NEW.execution_id)
BEGIN
  SELECT RAISE(ABORT, 'execution provenance is immutable');
END;
CREATE TRIGGER IF NOT EXISTS service_execution_provenance_no_update
BEFORE UPDATE ON service_execution_provenance BEGIN
  SELECT RAISE(ABORT, 'execution provenance is immutable');
END;
CREATE TRIGGER IF NOT EXISTS service_execution_provenance_no_delete
BEFORE DELETE ON service_execution_provenance BEGIN
  SELECT RAISE(ABORT, 'execution provenance is immutable');
END;

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
  fl_run_id            TEXT,
  client_id            TEXT,
  round_id             INTEGER,
  created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_service_failures_service_id ON service_failures(service_id);
CREATE INDEX IF NOT EXISTS idx_service_failures_group ON service_failures(manifest_group_id);
CREATE INDEX IF NOT EXISTS idx_service_failures_stage ON service_failures(failure_stage);

-- Active multi-round FedAvg coordination state. Global checkpoints are
-- training infrastructure and deliberately have no foreign key to services.
CREATE TABLE IF NOT EXISTS fedavg_runs (
  fl_run_id                    TEXT PRIMARY KEY,
  lineage_id                   TEXT NOT NULL,
  status                       TEXT NOT NULL,
  client_count                 INTEGER NOT NULL,
  round_count                  INTEGER NOT NULL,
  participation_rate           REAL NOT NULL,
  schedule_seed                INTEGER NOT NULL,
  schedule_version             TEXT NOT NULL,
  heterogeneity_intensity      REAL,
  partition_generator_version  TEXT NOT NULL DEFAULT 'sequential_depletion_v1',
  compatibility_contract_hash TEXT,
  compatibility_contract_json TEXT,
  g0_state_hash                TEXT,
  partition_plan_hash          TEXT,
  partition_plan_ref           TEXT,
  training_universe_hash       TEXT,
  benchmark_cohort_hash        TEXT,
  benchmark_cohort_ref         TEXT,
  status_reason                TEXT,
  started_at                   TEXT NOT NULL,
  completed_at                 TEXT,
  metadata_json                TEXT,
  CHECK (client_count >= 2),
  CHECK (round_count >= 2),
  CHECK (participation_rate > 0 AND participation_rate <= 1),
  CHECK (status IN ('running','complete','incomplete','failed','interrupted'))
);

CREATE TABLE IF NOT EXISTS fedavg_global_states (
  fl_run_id              TEXT NOT NULL,
  global_round           INTEGER NOT NULL,
  state_hash             TEXT NOT NULL,
  cas_ref                 TEXT NOT NULL,
  parameter_schema_hash  TEXT,
  parent_global_round    INTEGER,
  parent_state_hash      TEXT,
  storage_format         TEXT NOT NULL,
  compression_json       TEXT,
  logical_bytes          INTEGER,
  physical_bytes         INTEGER,
  verified_at            TEXT NOT NULL,
  metadata_json          TEXT,
  PRIMARY KEY (fl_run_id, global_round),
  FOREIGN KEY (fl_run_id) REFERENCES fedavg_runs(fl_run_id) ON DELETE CASCADE,
  CHECK (global_round >= 0)
);
CREATE INDEX IF NOT EXISTS idx_fedavg_global_state_hash
ON fedavg_global_states(fl_run_id, state_hash);

CREATE TABLE IF NOT EXISTS fedavg_client_partitions (
  fl_run_id                         TEXT NOT NULL,
  client_id                         TEXT NOT NULL,
  requested_partition_strategy     TEXT NOT NULL,
  requested_partition_params_json  TEXT NOT NULL,
  requested_quota                  INTEGER NOT NULL,
  effective_quota                  INTEGER NOT NULL,
  quota_shortfall                  INTEGER NOT NULL,
  target_bucket_counts_json        TEXT NOT NULL,
  target_bucket_proportions_json   TEXT NOT NULL,
  realised_bucket_counts_json      TEXT NOT NULL,
  realised_bucket_proportions_json TEXT NOT NULL,
  majority_bucket_share            REAL NOT NULL,
  normalised_bucket_entropy        REAL NOT NULL,
  represented_bucket_count         INTEGER NOT NULL,
  l1_allocation_deviation          INTEGER NOT NULL,
  total_variation_distance         REAL NOT NULL,
  target_deviation_count           INTEGER NOT NULL,
  resolution_reason                TEXT,
  partition_hash                   TEXT NOT NULL,
  metadata_json                    TEXT,
  created_at                       TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (fl_run_id, client_id),
  FOREIGN KEY (fl_run_id) REFERENCES fedavg_runs(fl_run_id) ON DELETE CASCADE,
  CHECK (requested_quota >= 1),
  CHECK (effective_quota >= 1),
  CHECK (quota_shortfall >= 0),
  CHECK (target_deviation_count * 2 = l1_allocation_deviation)
);

CREATE TABLE IF NOT EXISTS fedavg_rounds (
  fl_run_id                  TEXT NOT NULL,
  round_id                   INTEGER NOT NULL,
  status                     TEXT NOT NULL,
  scheduled_count            INTEGER NOT NULL,
  successful_count           INTEGER NOT NULL,
  failed_count               INTEGER NOT NULL,
  aggregation_method         TEXT,
  aggregation_version        TEXT,
  contributor_metadata_json  TEXT,
  excluded_metadata_json     TEXT,
  parent_state_hash          TEXT,
  output_state_hash          TEXT,
  started_at                 TEXT NOT NULL,
  completed_at               TEXT,
  status_reason              TEXT,
  metadata_json              TEXT,
  PRIMARY KEY (fl_run_id, round_id),
  FOREIGN KEY (fl_run_id) REFERENCES fedavg_runs(fl_run_id) ON DELETE CASCADE,
  CHECK (round_id >= 1),
  CHECK (status IN ('running','complete','incomplete','failed','interrupted'))
);

-- Task-aware evaluation metrics for the aggregated global model produced by
-- each round. These are separate from client service metrics because G1/G2/...
-- are coordination states, not independently advertised services.
CREATE TABLE IF NOT EXISTS fedavg_round_metrics (
  fl_run_id      TEXT NOT NULL,
  round_id       INTEGER NOT NULL,
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
  PRIMARY KEY (fl_run_id, round_id, metric_name),
  FOREIGN KEY (fl_run_id, round_id)
    REFERENCES fedavg_rounds(fl_run_id, round_id) ON DELETE CASCADE,
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
CREATE INDEX IF NOT EXISTS idx_fedavg_round_metrics_domain
ON fedavg_round_metrics(domain);

CREATE TABLE IF NOT EXISTS fedavg_client_rounds (
  fl_run_id                  TEXT NOT NULL,
  client_id                  TEXT NOT NULL,
  round_id                   INTEGER NOT NULL,
  service_id                 TEXT NOT NULL UNIQUE,
  scheduled                  INTEGER NOT NULL,
  participated               INTEGER NOT NULL,
  succeeded                  INTEGER NOT NULL,
  status                     TEXT NOT NULL,
  status_reason              TEXT,
  sample_count               INTEGER,
  aggregation_weight         REAL,
  training_seed              INTEGER,
  partition_seed             INTEGER,
  partition_hash             TEXT,
  parent_global_round        INTEGER,
  parent_state_hash          TEXT,
  final_state_hash           TEXT,
  compatibility_contract_hash TEXT,
  metadata_json              TEXT,
  created_at                 TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (fl_run_id, client_id, round_id),
  FOREIGN KEY (fl_run_id) REFERENCES fedavg_runs(fl_run_id) ON DELETE CASCADE,
  CHECK (scheduled IN (0,1)),
  CHECK (participated IN (0,1)),
  CHECK (succeeded IN (0,1)),
  CHECK (round_id >= 1)
);
CREATE INDEX IF NOT EXISTS idx_fedavg_client_round_service
ON fedavg_client_rounds(service_id);

-- Strict composition-validation provenance.  These tables never replace the
-- original service record or its catalogue metrics.
CREATE TABLE IF NOT EXISTS service_regenerations (
  regeneration_id           TEXT PRIMARY KEY,
  service_id                TEXT NOT NULL,
  cohort_id                 TEXT NOT NULL,
  cohort_plan_hash          TEXT NOT NULL,
  seed                      INTEGER NOT NULL,
  resolved_config_json      TEXT NOT NULL,
  config_sha256             TEXT NOT NULL,
  code_sha256               TEXT,
  dependency_versions_json  TEXT,
  provenance_json           TEXT,
  original_accuracy         REAL,
  regenerated_accuracy      REAL,
  accuracy_difference       REAL,
  artifact_uri              TEXT,
  created_at                TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (service_id) REFERENCES services(service_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_service_regenerations_service ON service_regenerations(service_id);
CREATE INDEX IF NOT EXISTS idx_service_regenerations_cohort ON service_regenerations(cohort_id);

CREATE TABLE IF NOT EXISTS composition_validation_runs (
  validation_run_id         TEXT PRIMARY KEY,
  request_id                TEXT NOT NULL,
  cohort_id                 TEXT,
  cohort_manifest_json      TEXT,
  inputs_json               TEXT,
  provenance_json           TEXT,
  summary_json              TEXT,
  created_at                TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS composition_validation_candidates (
  validation_run_id             TEXT NOT NULL,
  candidate_id                  TEXT NOT NULL,
  service_ids_json              TEXT NOT NULL,
  validation_status             TEXT NOT NULL,
  status_reason                 TEXT,
  selection_source              TEXT,
  search_method                 TEXT,
  recorded_surrogate_utility    REAL,
  recorded_surrogate_quality    REAL,
  regenerated_surrogate_utility REAL,
  measured_utility              REAL,
  measured_quality_score        REAL,
  utility_prediction_error      REAL,
  quality_prediction_error      REAL,
  ensemble_accuracy             REAL,
  ensemble_f1_macro             REAL,
  ensemble_rmse                 REAL,
  ensemble_mae                  REAL,
  actual_primary_metric_name    TEXT,
  actual_secondary_metric_name  TEXT,
  parallel_latency_s            REAL,
  serial_latency_s              REAL,
  prediction_wall_s             REAL,
  aggregation_method            TEXT,
  aggregation_version           TEXT,
  fl_run_id                     TEXT,
  round_id                      INTEGER,
  parent_global_round           INTEGER,
  parent_state_hash             TEXT,
  compatibility_contract_hash  TEXT,
  parameter_schema_hash         TEXT,
  aggregated_state_hash         TEXT,
  selected_total_samples        INTEGER,
  cohort_id                     TEXT,
  parent_checkpoint_id          TEXT,
  local_example_exposures        INTEGER,
  unique_training_examples      INTEGER,
  contributor_provenance_json   TEXT,
  aggregation_provenance_json   TEXT,
  aggregation_wall_s            REAL,
  inference_latency_s           REAL,
  actual_accuracy               REAL,
  actual_f1_macro               REAL,
  actual_rmse                   REAL,
  actual_mae                    REAL,
  qos_violations_json           TEXT,
  selected_flag                 INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (validation_run_id, candidate_id),
  FOREIGN KEY (validation_run_id) REFERENCES composition_validation_runs(validation_run_id) ON DELETE CASCADE,
  CHECK (validation_status IN (
    'completed','unsupported','regeneration_failed','cohort_isolation_failed',
    'checkpoint_failed','aggregation_failed','inference_failed'
  )),
  CHECK (selected_flag IN (0,1))
);
CREATE INDEX IF NOT EXISTS idx_composition_validation_candidates_status ON composition_validation_candidates(validation_status);

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
