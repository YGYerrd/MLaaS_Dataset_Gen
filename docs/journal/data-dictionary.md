# Working data dictionary

> Historical September 2026 draft: its 11,907-service Lambda cohort is not the
> retained November dataset. The old shards, audit exports and one-off audit
> CLIs were removed. The current reconciled classification subset contains
> 6,975 services in 70 pools; see [retained datasets](../datasets.md).

This dictionary describes the current code schema and its important interpretation rules. The selected dataset is the 11,907-record Lambda inclusion roster, not the historical stores inspected for draft 0.1. The exact SQL schema is [schema.sql](../../mlaas_data_generator/storage/schema.sql); the [audit summary](evidence/lambda-valid-completed-summary.json) establishes cohort coverage for the named checks. `NULL`, absent keys and unsupported evaluation states must remain distinguishable.

The derived inclusion roster is keyed by `service_id`; `source_database` identifies the original shard. `valid_completed_record` means the documented record/artifact audit passed, and `exclusion_reason` describes a failed check in the full checks CSV. `primary_metric_name` and `primary_metric_value` retain task-specific quality; `metric_score` is the stored normalised score. `realised_training_samples`, `retention_policy` and `audit_examples` describe recorded training and reservation properties. `explainability_usable` requires complete status, a true sufficiency flag and a finite score; it is not independent verification of the underlying trials. Runtime and Git columns reproduce recorded provenance, including the dirty-working-tree flag. This derived CSV is an index into the full records, not an export of all metrics or artifacts.

| Table / artifact | Fields | Meaning and interpretation |
| --- | --- | --- |
| `services` | `service_id` | Record identifier and join key; not a provider identifier |
| `services` | `family_id` | Current family linkage where present; older schemas may only link through provenance |
| `services` | `status`, `created_at`, `updated_at` | Stored outcome and record timestamps; timestamps alone do not establish complete attempt history |
| `services` | `task_family`, `task_type`, `hf_task`, `modality` | Recorded task taxonomy and input modality; historical fill-mask and token records can have broad `classification` task types |
| `services` | `model_id`, `model_type` | Source model reference and recorded model/backend descriptor; inspect semantics before naming an architecture or provider |
| `services` | `dataset_name`, `dataset_config`, `train_split`, `benchmark_split` | Dataset and split references; exact revision/content identity resides in provenance where available |
| `services` | `training_regime`, `dataset_variant`, `split_variant`, `knob_variant` | Recorded generation regime and variant descriptors |
| `services` | `input_schema`, `output_schema` | Declared input/output structure; not sufficient on its own to prove executable interoperability |
| `services` | `service_config_json`, `registry_metadata_json`, `functional_attributes_json`, `metadata_json` | Structured configuration/metadata; fields depend on record generation version |
| `service_metrics` | `service_id`, `metric_name` | Unique observation key in the current schema; the named metric determines interpretation |
| `service_metrics` | `domain`, `unit`, `direction` | Domain, recorded units and whether higher/lower is better or direction is neutral |
| `service_metrics` | `value_num`, `value_int`, `value_bool`, `value_text`, `value_json` | Exactly one typed value is required by the current SQL constraint; a JSON value can contain a structured observation |
| Metric names | `metric_score`, task-specific quality names | Normalised versus raw task-specific quality; do not compare across tasks without an explicit estimand |
| Metric names | `evaluation_protocol_version` | Metric protocol identity; absent on many historical observations |
| Metric names | `trust_status`, `explainability_status` | Complete, insufficient, unsupported or failed evaluation as applicable; always inspect before interpreting score fields |
| Metric names | `trust_evidence_sufficient`, `explainability_evidence_sufficient` | Evidence-gate outcomes under the recorded protocol; not general safety or interpretability certification |
| `service_generation_provenance` | `model_revision`, `dataset_revision`, `architecture_id` | Versioned source and architecture identities where recorded |
| `service_generation_provenance` | `family_id`, `family_contract_version`, `generation_mode` | Family and generation interpretation; old contracts are not implicitly upgraded |
| `service_generation_provenance` | `parameter_schema_hash`, `label_map_hash`, `preprocessing_contract_hash`, `dataset_contract_hash` | Compatibility/identity evidence; agreement in one hash does not replace all other checks |
| `service_generation_provenance` | `initial_state_hash`, `base_model_hash`, `final_model_hash` | Model-state identities; different from serialization-file checksums |
| `service_generation_provenance` | `family_document_ref`, `family_document_hash`, `client_recipe_ref`, `client_recipe_hash` | Artifact locations and integrity metadata for the recorded family and training recipe |
| `service_generation_provenance` | `composition_compatible` | Historical publication marker; the current consumer verifies actual evidence independently |
| `service_generation_provenance` | `parent_checkpoint_id`, `cohort_id`, `fl_run_id`, `round_id` | Generation-mode-specific relationships; independent training is not automatically federated training |
| `service_split_provenance` | `split_name`, `samples_count`, `data_distribution_json`, `split_config_json` | Split count/distribution metadata; reconstruct exact membership from the applicable recipe/reservation artifacts |
| `service_execution_provenance` | `execution_id`, `service_id`, `run_id`, `started_at`, `document_sha256`, `document_json` | Recorded execution context and immutable document; run/environment coverage varies |
| `service_artifacts` | `artifact_type`, `artifact_uri`, `artifact_sha256`, `size_bytes`, `artifact_format`, `artifact_role` | External artifact identity, reference and metadata where recorded; a path is not an embedded payload or proof the payload is accessible |
| `service_checkpoint_storage` | `storage_mode`, state hashes, patch/full artifact references | Storage interpretation, including representation-specific dependencies; recipe-only retention is described by the relevant configuration and provenance |
| Family document | Source, preprocessing, task, initial state and evaluation reservation | Shared context for supported client recipes; preserve the complete versioned document |
| Client recipe | Ordered selected source rows, quantities, effective settings, seeds, observations and fingerprints | Recipe for replay; depends on raw data, shared state and runtime, and is not a compressed copy of all final weights |
| Request document | Schema/profile, functional requirements, QoS constraints, weights, family/pool identities | Synthetic selection conditions and frozen candidate space |
| Search result | Method, budget, seed, selected IDs, predicted quality/utility and objective version | Outputs of surrogate selection, not measured ground-truth labels |
| Validation result | Status, measured quality/latency, errors, reconstruction and aggregation timing, provenance | Actual validation where successful; other utility dimensions can retain catalogue evidence |

Before deposit, supplement this overview with a machine-readable schema for every distributed JSON/CSV artifact, enumerated units and metric formulas, allowed values, null semantics, foreign-key relationships and examples. The dictionary should describe distributed files rather than every internal optional feature of the software.
