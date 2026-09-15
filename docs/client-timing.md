# Client execution timings

New executions record `timing_scope=service_start_through_primary_commit_v2`.
Historical records retain their original timing definitions and are not backfilled.

`runtime_total_s`, `runtime_s`, and `service_runtime_s` now cover service execution
through the primary database commit, including setup, training, evaluation,
perturbation, checkpointing, provenance, and the final in-memory state needed for
FedAvg aggregation. For FedAvg, measurement starts before the executor restores
the parent model. `client_runtime_s` includes that same client boundary and is
displayed beside `train` in concise terminal output.

The final transaction that records these timing measurements and completion
metadata, terminal rendering, manifest-progress bookkeeping, and round aggregation
are outside the total. A duration cannot include the future write of its own final
value. `db_write_s` includes opening the writer and the primary commit, excluding
that last telemetry transaction. `completed_at` is updated after the primary
commit; it no longer describes the point before checkpointing.

## Stage metrics

| Metric | Boundary |
| --- | --- |
| `training_runtime_s` / `finetune_s` | Training wrapper, including optimizer setup and any OOM retry |
| `train_time_s` | HF training loop; monotonic elapsed time |
| `evaluation_runtime_s` | Evaluation wrapper, including metric calculation |
| `setup_s` | Service setup through resource tracker startup, before workload |
| `initial_state_s` | Schema/identity setup, parent validation and local preparation |
| `model_reset_s` | Verified parent restoration, including an executor-provided restore |
| `local_preparation_s` | Training-state reset and verification that weights are unchanged |
| `pretrain_snapshot_s` | Initial weight snapshot for the update signature |
| `signature_s` | Post-training snapshot and signature generation |
| `workload_runtime_s` | Initial snapshot, training, signature and evaluation |
| `resource_tracker_start_s` / `resource_tracker_stop_s` | Monitoring lifecycle overhead |
| `hardware_snapshot_s` | Hardware metadata collection using the latest available GPU sample |
| `checkpoint_s` | Checkpoint work; historical rounds retain only in-memory state and provenance |
| `post_checkpoint_s` | Remaining provenance work and final state capture after checkpoint stage |
| `final_state_capture_s` | Capture, or reuse, of the final aggregation state; part of `post_checkpoint_s` |
| `non_training_runtime_s` | Full total minus the training wrapper |

These metrics are nested, not additive: for example `initial_state_s` overlaps
`setup_s`, and `signature_s` is part of `workload_runtime_s`. The executor's
`model_reset_s` precedes service-local `setup_s` when a prepared model is supplied.
Round duration still includes aggregation and shared run/round work.

## Performance and diagnostic behavior

### Checkpoint substeps

New executions persist `checkpoint_profile` with format
`mlaas-checkpoint-profile-v1` in `service_generation_provenance.provenance_json`.
If checkpoint work fails, completed and failed spans are retained in the failed
service's metadata. Each observed step records `calls`, `failed_calls`,
`inclusive_s` and `exclusive_s`. These are monotonic host elapsed times, including
blocking transfers made by existing code, without adding GPU synchronization.
Profiling holds no tensors or payloads and is scoped to each service.

Inclusive values are also queryable as `checkpoint_<step>_s` service metrics:

| Step | Boundary |
| --- | --- |
| `weights` / `patch_persist` | Exact patch persistence, including nested substeps |
| `base_load_verify` | Read, decompress, deserialize and verify the lineage base |
| `state_capture`, `state_hash`, `content_hash` | Model capture, canonical state digest, CAS payload digests |
| `state_serialize`, `patch_serialize`, `state_compare` | Canonical serialization and exact reconstructed-byte comparison |
| `patch_selection`, `patch_build` | Representation selection and changed-tensor construction |
| `candidate_xor_compression`, `candidate_replace_compression`, `candidate_per_tensor_hybrid_compression` | Candidate encoding only; absent when skipped |
| `compression`, `decompression` | All codec calls across selection, storage measurement and CAS verification |
| `full_baseline` | Same-codec full-state CAS size measurement, after state serialization |
| `patch_write`, `cas_read_verify`, `patch_apply_verify`, `model_restore_verify` | Publication and integrity/reconstruction checks |
| `publish` | Immutable payload/manifest creation or existing-content comparison |
| `support_artifacts`, `provenance_write`, `reconstruction_manifest_write`, `storage_metadata`, `artifact_metadata` | Runner support assets, JSON provenance and checkpoint metadata |
| `native_save`, `native_restore_verify` | Full-object fallback, when used |

These inclusive timings **must not be added together**. For example, candidate
compression belongs to both `patch_selection` and `compression`. Only exclusive
times sum to `checkpoint_profile.profiled_s`; uninstrumented checkpoint work and
collector overhead account for the difference from `checkpoint_s`. Missing steps
mean that path was not observed, rather than a measured zero. Historical rows are
not backfilled. Timings stay outside immutable checkpoint manifests and hashes.

The replay command and an observed BERT breakdown are documented in
[generation efficiency](cleanup.md).

### Existing runtime behavior

- Immutable tensor states cache their digest. Initial hashing streams the existing
  canonical serialization framing without allocating another full-model buffer.
  Checkpoint hashes and integrity verification remain compatible.
- The executor supplies its verified parent state to the service, avoiding a
  second restore. Standalone services restore themselves. Local preparation still
  checks the actual weights against the expected parent hash.
- Non-final client states are reused for aggregation instead of captured twice.
- Resource sampling runs asynchronously. GPU counter queries are shared for up to
  one second and bounded by subprocess timeouts. Startup and shutdown do not wait
  for OS queries. Shutdown uses completed samples; a very short client may have no
  GPU sample, represented as unavailable rather than zero. Hardware metadata
  records the age of its reused GPU sample as `gpu_sample_age_s`.
- HF durations and timeout checks use a monotonic clock. Precision fallbacks also
  record `precision_fallback_error`, preserving the original exception for debugging.
- HF training and evaluation retain PyTorch's GPU allocation cache between
  batches. Existing service/parent cleanup still releases cached GPU memory;
  reserved VRAM may consequently remain higher during a run.
- Disk-backed HF image batches use one CPU reader with two batches of lookahead
  during training and evaluation. Reads overlap model execution, preserving
  batch order, labels, padding, and the selected training recipe. GPU encoding
  stays on the calling thread. Readers are joined before returning on completion,
  timeout, or failure so the image cache can be closed safely. In-memory inputs
  retain synchronous batching. Set `MLAAS_PREFETCH_BATCHES=0` to disable lookahead,
  or choose a depth from 1 to 8 (default 2). Extra host memory is bounded by that
  many prepared batches, plus the current batch and temporary read buffers.

Already-running Python processes continue using their loaded implementation.
These changes apply when a new process starts.
