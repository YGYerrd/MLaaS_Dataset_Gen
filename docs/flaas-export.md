# FLaaS flat export contract v1

`flaas_flat_v1` exports one row per client in each **completed composition execution**.
The unique key is `(run_id, combination_id, client_id)`. `client_id` is the existing
service identifier. Repeated executions of a combination are retained. Failed,
pending and running compositions are excluded, even if some members completed.

```powershell
python -m mlaas_composition export-flaas `
  --db outputs/compositions.db `
  --source-db outputs/pooled/services.db `
  --output outputs/flaas_flat.csv
```

Use `--overwrite` to replace a previous export, or `--path-map PATH` to override
the artifact relocation map next to the service database. Both databases are
opened read-only in explicit read transactions. No training, simulation, tensor
loading, downloads, or database updates are performed. Recipes and family
documents are verified against their recorded SHA-256 hashes and identities.
No timestamps or machine-specific source paths are introduced into the CSV.

The command writes one UTF-8 CSV with a BOM and CRLF record separators. Numeric
values use decimal points and retain Python's round-trip precision. Identifiers
remain text. Missing values are empty cells, never substituted with zero. Headers
are emitted even when no completed executions exist. The console reports row,
execution and distinct-client counts and missing counts for every affected
required attribute. Validation completes before atomic publication; an error
leaves any previous output intact and removes the temporary file.

## Required attributes

These 38 columns cover the 36 distinct attributes in the numbered PDF table
plus `training_deadline` and `deadline_violation` from its monitoring discussion.
The duplicated `global_accuracy_per_round` appears once. The ambiguous heading
`last_selected_round/participation history` becomes `last_selected_round`, with
`participation_history` retained separately. Types below describe nonblank cells.

`snapshot` means the run's `source_services.snapshot_json`; `evidence` means
`composition_members.evidence_json`; `recipe` means the verified immutable client
recipe. Optional absent values use `not_recorded` unless a more specific reason
is listed. Missing recipe/family files use `artifact_missing` for dependent fields.

| Column | Type / unit | Source or calculation |
| --- | --- | --- |
| client_id | text | Membership `service_id`; matches snapshot, recipe and evidence. |
| dataset | text | Snapshot `dataset_name`. |
| task_type | text | Snapshot `task_type`; not inferred from model names. |
| model_id | text | Snapshot canonical model ID, not a local cache path. |
| data_volume | integer / samples | Member `sample_count`, checked against snapshot and recipe realised count. Counts the local partition once, not once per epoch. |
| class_distribution | JSON text / class counts | Recipe `observed_metrics.train_distribution_json.value`; keys sorted lexicographically, zero-count classes preserved; counts sum to data_volume. |
| split_strategy | text | First recorded recipe metric among `split_strategy_effective`, `split_strategy`, `data_distribution`; retains iid/dirichlet/shard. |
| cpu_cores | integer / physical cores | Original service `metadata_json.hardware_snapshot.cpu.physical_cores`. |
| ram_gb | number / decimal GB | Original service `metadata_json.hardware_snapshot.memory.total_mb * 1048576 / 1000000000`. Despite its name, the source total_mb is MiB. |
| client_availability | boolean, blank in v1 | `not_recorded`; success does not prove pre-selection availability. |
| local_epochs | integer / epochs | Evidence `training.training_epochs`, the configured replay budget. |
| batch_size | integer / examples | Evidence `training.logical_batch_size`; microbatch size is separate. |
| learning_rate | number | Recipe `training.effective_settings.learning_rate`. |
| local_training_time_s | number / seconds | Recipe `observed_metrics.train_time_s.value`, a historical training-loop observation used as the pre-selection estimate, without hardware scaling. |
| communication_time_s | number / seconds, blank | `not_recorded`. |
| communication_round | integer, blank | `no_round_history`. |
| participation_history | text, blank | `no_round_history`; unknown is not an empty history. |
| training_completion_status | text | Member status, required to be completed within completed compositions. |
| combination_id | text | Composition identifier. Not globally unique across runs. |
| global_model_accuracy | number / fraction [0,1] | Measured composition metric `accuracy`, required for completed exports; never inferred from member accuracies. |
| round_deadline_s | number / seconds, blank | `not_recorded`; executor timeouts are not FL round deadlines. |
| upload_time_s | number / seconds, blank | `not_recorded`. |
| download_time_s | number / seconds, blank | `not_recorded`. |
| model_size_mb | number / decimal MB | Sum of `numel * dtype_bytes` across verified family `contracts.parameter_schema.entries`, divided by 1000000; includes buffers. Empty schema is not_recorded; unsupported dtype or inconsistent shape fails validation. |
| upload_throughput_mbps | number / megabits per second, blank | `not_recorded`. |
| round_completion_time_s | number / seconds, blank | `not_recorded`; sequential executor duration is not a distributed round measurement. |
| selected_client_ids | JSON text / ID array | Composition members in ordinal order, checked against aggregation provenance `selected_service_order`. |
| squared_sample_loss_sum | number / squared loss | Evidence field of the same name; checked against final-local-model loss statistics. `loss_telemetry_absent` for historical executions without measurements. |
| last_selected_round | integer, blank | `no_round_history`. |
| global_accuracy_per_round | number / fraction, blank | `no_round_history`; independent experiments do not form a learning curve. |
| observed_training_duration_s | number / seconds | Evidence `training.train_time_s`, measured in the composition replay. |
| local_loss_current_model | number / loss | Evidence field of the same name; mean loss of the shared initial model on local training rows. `loss_telemetry_absent` if unmeasured. |
| candidate_pool_size | integer / clients, blank | `not_recorded`; family size does not establish the algorithm's candidate-pool configuration. |
| local_loss_l2_squared | number / squared loss | Same final-local-model sum of squared per-example losses as squared_sample_loss_sum. It is not the square of a mean. `loss_telemetry_absent` if unmeasured. |
| emd_score | number, blank | `definition_deferred`; no distance or reference population was selected. |
| cumulative_model_strength | number, blank | `no_round_history`; cannot calculate a recurrence from unrelated executions. |
| training_deadline | number / allowed seconds, blank | `not_recorded`; reserved monitoring field. |
| deadline_violation | boolean, blank | `not_recorded`; unknown deadline does not mean false. |

CPU and RAM are capacities of the **original generation host**, not allocations
to dedicated FL clients or measurements of the later replay host. Hardware
fields use `source_service_absent` if no matching source service is present;
an existing service with conflicting identities is an error. Recipe enrichment
can still be verified directly against the recorded snapshot when a service is
absent. Model size is a derived uncompressed full-state payload size, not an
observed transfer: headers, compression, protocol choices and network overhead
are excluded. Dtypes determine sizes; parameter counts are never used as MB.

The historical time and observed replay time may come from different hardware.
Loss measurements are fixed-model passes over local training rows, separate
from optimizer-trajectory loss and from the final aggregate's audit results.
See [composition execution and losses](recipe-composition.md#execute-a-composition-manifest).

## Supporting columns

Absent optional supporting values are blank; `missing_attributes` covers the
required attributes, not these supporting columns.

| Columns | Type / unit | Definition |
| --- | --- | --- |
| schema_version | text | Constant `flaas_flat_v1`. |
| run_id, family_id | text | Composition execution run and compatible family. |
| composition_size, member_ordinal | integer | Membership size and zero-based order. |
| aggregation_weight | number / fraction | Recorded member weight; each execution sums to one within 1e-9 tolerance. |
| composition_status | text | Constant `completed` for this export. |
| started_at, finished_at | timestamp text | Composition's original timestamp strings, unchanged. |
| client_recipe_hash, family_document_hash, initial_state_hash | text | Snapshot provenance identities; recipe/family hashes identify original file bytes. |
| materialized_state_hash | text | Actual reconstructed client state from membership, checked against evidence. |
| completed_epochs | number / epochs | Evidence `training.completed_training_epochs`. |
| micro_batch_size | integer / examples | Evidence `training.micro_batch_size_effective`. |
| gradient_accumulation_steps | integer | Evidence `training.gradient_accumulation_steps_effective`. |
| hardware_scope | text | `original_generation_host` when CPU or RAM exists. |
| cpu_cores_basis | text | `physical` when CPU data exists. |
| local_training_time_basis | text | `historical_catalogue_train_loop` when historical timing exists. |
| observed_training_duration_basis | text | `composition_train_loop` when observed timing exists. |
| model_size_basis | text | `uncompressed_full_state_schema` when derivable. |
| reconstruction_duration_s | number / seconds | Member `training_wall_s`, including preparation, replay and state capture. |
| execution_wall_s, aggregation_wall_s | number / seconds | Composition metrics `total_wall_s`, `aggregation_wall_s`. These are not additive with nested stage durations. |
| audit_examples | integer / examples | Composition metric `audit_examples`. |
| client_loss_statistics_version | text | Validated `client_loss_statistics_v1`; blank if no loss telemetry. |
| loss_population | text | `selected_local_training_rows`. |
| local_loss_current_model_scope | text | `initial_model_on_local_training_rows`. |
| squared_sample_loss_scope | text | `final_local_model_on_local_training_rows`. |
| loss_definition | text | Stored per-example loss definition, checked across initial and final passes. |
| missing_attributes | text | Semicolon-separated missing required fields in contract order. |
| missing_attribute_reasons | text | Matching semicolon-separated `field=reason` pairs. |

## Analysis and validation boundaries

Composition-level values repeat on member rows. Before computing mean global
accuracy or comparing composition sizes, reduce to one row per
`(run_id, combination_id)`; averaging all member rows overweights larger groups.
Do not use observed replay durations, final-local losses or global accuracy as
features supposedly known before selection. This completed-only dataset cannot
estimate failure rates, deadline violations, or multi-round accuracy progression.

Source identity conflicts, changed artifact bytes, malformed JSON (including
duplicate keys/non-finite constants), duplicate row keys, invalid membership,
inconsistent counts, missing measured accuracy, or inconsistent loss telemetry
abort publication. Historical missing telemetry and absent optional artifacts
produce blanks with reasons. Numeric validations preserve legitimate zeros.

Run the focused regression checks with:

```powershell
python -m pytest tests/composition/test_flaas_export.py -q
```
