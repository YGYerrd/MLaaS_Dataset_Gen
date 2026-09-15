# Generator checkpoint storage specification

New manifests default to reduced INT8 Safetensors final checkpoints, with separate source/reload hashes and sampled performance validation. FP16 remains selectable. See [the settings, identity contract and measured results](checkpoint_formats.md). The exact-storage specification below remains available through `checkpoint_weight_format="native"` and for legacy manifests.

## Current local artifact store

The 9 September 2026 consolidation captured all 402 completed services in
`outputs/service_artifacts_int8/`: 392 use quality-validated INT8 weights, while
10 retain exact weights because their INT8 conversions exceeded the quality
tolerance. The same store preserves exact lineage bases, model configuration,
tokenizers/processors, reconstruction records, and provenance. Its name does
not mean every file is quantized.

Both live catalogues and 510 resolved manifest bindings now reference this store.
`outputs/service_artifacts/` is a compatibility junction to it, preserving paths
inside immutable, hashed family documents. **Keep this junction**; it is no longer
a separate copy of the weights. Generation was not resumed.

Verification covered all 402 completed-service resume checks, actual composition
and audit inference for all nine families (including exact/INT8 mixtures), and
preservation of service records, metrics, split records, and original model
identities. Pre-existing missing historical artifacts and perturbation exports
were recorded separately; the migration introduced no new missing artifacts.
Backups, per-service quality evidence, verification results, and the cleanup
record are under `outputs/artifact_store_migration_20260909/`.
The retired physical store was removed after verification: 2,722 redundant files,
108.63 GB in total. The consolidated store occupies 43.18 GB. See the local
[completion audit](../outputs/artifact_store_migration_20260909/REPORT.md).

## One-off migration of trained legacy checkpoints

For another store, replace the example source and destination below with two
separate directories. The compatibility junction and its target above cannot
serve as separate source/destination stores. Convert exact checkpoints without
retraining:

```powershell
.venv/Scripts/python.exe -m mlaas_data_generator.migrate_checkpoints --store-root outputs/legacy_service_artifacts --output-store outputs/migrated_service_artifacts --weight-format int8 --all --dry-run
.venv/Scripts/python.exe -m mlaas_data_generator.migrate_checkpoints --store-root outputs/legacy_service_artifacts --output-store outputs/migrated_service_artifacts --weight-format int8 --all > outputs/checkpoint_migration_int8.jsonl
```

Use repeated `--service-id SERVICE_ID` arguments instead of `--all` to select a batch.
The command now defaults to INT8 Safetensors. Use `--weight-format fp16` for the
previous representation and a separate FP16 destination. `--compression zstd
--compression-level 1` or `--compression zlib` selects the same optional outer
compression used by new services. The Python `migrate_checkpoint()` API retains
its FP16 default for compatibility; pass `representation="int8"` explicitly.

To switch an already-running FP16 migration, stop that process with Ctrl+C and
start the INT8 command above. Use the original exact store as the source, not
the partially converted FP16 store. Existing FP16 artifacts are retained, and
conflicting representations cannot overwrite each other's service manifests.

INT8 uses `int8_block256_safetensors_v1` within `mlaas-reduced-checkpoint-v1`.
Each block of up to 256 floating-point values has one positive FP32 scale:
`scale = max(abs(block)) / 127`, `q = clip(round(value / scale), -127, 127)`.
Rounding uses nearest-even, zero blocks use scale 1, and the loader reconstructs
`q * scale` in the original execution dtype. Blocks follow flattened C-order.
The layout names auxiliary scale tensors, which the decoder verifies alongside
the weight tensors. FP16/FP32/FP64 weights are eligible; integers, BF16, tensors
with nonfinite values, and tensors with unrepresentable scales are retained exactly.
This uses about 1.016 bytes per eligible weight, compared with 2 for FP16 and
4 for FP32, plus manifest and Safetensors headers. Tiny tensors may grow because
of scale metadata. Shared parents are still exact. INT8 generally introduces more
weight error than FP16; prediction quality must be evaluated separately.

The command verifies and reconstructs saved exact base/patch weights, then calls the
production `persist_reduced_checkpoint()` encoder and the normal reconstruction
decoder. The output is exactly `mlaas-reduced-checkpoint-v1`, including the original
source identity, separate reload hash, original execution dtypes and weight-error
measurements. Exact immutable parent tensors are copied into the destination CAS
and deduplicated, including explicit FedAvg round parents. Zero-patch references
are supported too. The destination must be separate from and outside the source
directory tree. The existing storage overflow policy applies to new objects.

This is a tensor-storage migration: it does not train, evaluate predictions, delete
source artifacts, or update catalogue rows, metrics, provenance or support assets.
`quality_validation.status` is `not_run`; payload integrity and round-trip checks
do not establish prediction-quality equivalence. Native full-object checkpoints
without an exact service manifest are outside this command's scope. Keep existing
configuration/tokenizer/processor support artifacts for model construction.

Reload a converted checkpoint with the existing API:

```python
from mlaas_data_generator.checkpoint_store import restore_service_state

reload_hash = restore_service_state("outputs/service_artifacts_int8", service_id, model)
```

The JSONL report on stdout records each source manifest hash, destination, source
and reload identities, validation status, and any failure. Progress goes to stderr.
Failures do not stop the remaining batch and produce exit code 1. Reruns verify
and skip matching destination checkpoints; conflicts or corruption fail without
overwriting them. Already-reduced sources are verified and skipped. A dry run
verifies exact source reconstruction for new conversions and checks existing
outputs without writing files.
Retain the original artifacts until any desired quality evaluation and catalogue
cutover have been handled separately; migration initially requires extra disk space.

Migration reuses the parent's original compressed bytes instead of recompressing
them. Within a batch, verified destination parent references are reused while
their files' size and timestamps remain unchanged; only small references are
cached in memory. Each source parent is loaded once per service, and each source
patch is applied and checked against its original target hash. Production
persistence still decodes and verifies the saved reduced payload before publishing;
the command avoids a redundant second reload afterward. On restart the default
`--verify-existing hash` compares source identity, representation and compression
metadata, then verifies saved payload sizes and SHA-256 hashes without loading
the original weights or decoding the reduced weights. Shared parent payloads
are hashed once per run unless their files change. Reports mark these checks as
`existing_verification="hash"`, with `round_trip_verified=false` because a new
decode was not performed. Newly converted checkpoints always receive the full
round-trip check. Use `--verify-existing full` to repeat source reconstruction
and output decoding for an explicit audit; it is much slower. Fast resume trusts
the source identity recorded in the manifests and does not audit old source payloads.
Progress separates pending new conversions from resume checks. Its estimate uses
only new conversion timings, so fast skips do not distort it; it excludes remaining
resume-check time and mixed model sizes can shift it.
Use the default `--compression none` to avoid outer compression work.

## External drive overflow

This checkout supports a git-ignored `storage.local.json` at the repository root:

```json
{"overflow_dir": "D:\\MLaaS", "reserve_bytes": 21474836480}
```

`MLAAS_OVERFLOW_DIR` overrides the destination. Without either setting, storage
behaviour is unchanged. Create the destination folder before use and keep its
drive letter stable. New tensor objects use the external drive when writing their
encoded payload and manifest would leave less than 20 GiB on the primary volume.
The payload and manifest stay together and saved references contain their actual
absolute paths. Existing objects are not moved. Native checkpoint and support
artifact staging also switches below the reserve, though their final size is not
known before serialization. Small lineage/service references, databases, logs,
and dataset caches remain in their configured locations.

An unavailable or insufficiently spacious overflow folder raises an error when
overflow is needed. This is a pre-write space check, not a guarantee against other
processes consuming space concurrently or the drive disconnecting during a write.
Restart an already-running generator to load this code before relying on overflow.

Shared-parent generation stores each task-trained parent once as the immutable
lineage base. Every successful client has a direct lossless patch from that parent,
with all patches retained and no round or aggregation state. Parent checkpoints
are infrastructure records, not services. See [the generation workflow](shared_parent_generation.md).

The optional exact path persists independent neural-network checkpoints using
`shared_base_exact_patch_v1`. A lineage owns one immutable canonical base
state, and each supported independent service owns a direct lossless patch
from that base. Historical multi-round FedAvg retained client weights only in its final
round. Earlier client-rounds keep hashes, metrics, partition records, parent
global-state references, and other historical provenance, but publish no
client weight patch or reconstruction manifest. Final-round clients use
`fedavg_round_base_exact_patch_v1` and are patched directly from the full
immutable global state `G(R-1)`. Patches never form chains. Models
whose state cannot be captured and restored exactly use the explicit
`full_object` fallback.

## Canonical tensor state

The only tensor-state hash specification is `mlaas-tensor-state-v1`. Its
SHA-256 input is the following binary stream:

```text
ASCII bytes "MLAAS-TENSOR-STATE-V1\0"
uint64_be entry_count
for each entry in native model-state order:
    uint64_be name_utf8_nfc_length
    name_utf8_nfc_bytes
    uint64_be canonical_dtype_ascii_length
    canonical_dtype_ascii_bytes
    uint64_be rank
    uint64_be dimension[0] ... uint64_be dimension[rank - 1]
    uint64_be tensor_byte_length
    contiguous_c_order_little_endian_tensor_bytes
```

Supported canonical dtypes are `bool`, signed and unsigned integers of 8, 16,
32, or 64 bits, `float16`, `float32`, `float64`, `complex64`, `complex128`, and
`bfloat16`. Scalars have rank zero. Empty tensors have a zero-length payload.
Names are normalized with Unicode NFC. Duplicate names, negative dimensions,
object dtypes, sparse tensors, meta tensors, and unimplemented quantized
representations are rejected.

Framework name, parameter-versus-buffer classification, trainability,
architecture metadata, aliases, parameter-schema hashes, storage paths, and
compression settings are not part of this hash. Re-serializing the same state
with another supported compression codec therefore does not change its state
identity.

## Parameter schema is separate

`parameter_schema_hash` identifies the ordered layout and model semantics
captured by generator provenance, including parameter/buffer kind and
trainability. `base_state_sha256` and `target_state_sha256` identify tensor
values under the binary specification above. Neither hash substitutes for the
other, and schema JSON never contains tensor values.

## Patch semantics and compression

The semantic format is `mlaas-tensor-patch-v1`. It binds an immutable base
state hash to a target state hash and represents changed entries with exact
byte-XOR or exact replacement operations. Unchanged entries are inherited from
the base. Applying a patch requires the same ordered entry descriptors and is
successful only when the reconstructed canonical state hash equals the target
hash.

Patch semantics/version and compression are recorded independently. The
current storage envelope supports `none`, `zlib` and `zstd`; its codec and level live
under storage encoding metadata and in dedicated database columns. XOR is a
reconstruction operation, not a compression codec.

`checkpoint_patch_operation_policy` chooses the lossless representation:

| Value | Selection work |
| --- | --- |
| `adaptive` | Compare XOR, replacement and hybrid encodings using the existing size policy |
| `fixed_xor` (new manifest default) | XOR every changed tensor; skip candidate compression comparisons |
| `fixed_replace` | Store every changed tensor as exact replacement; skip candidate comparisons |

All policies inherit unchanged tensors and retain the same compression controls,
optional full-state storage measurement, canonical hashes, strict model restoration and
persisted-patch read-back checks. Fixed policies can consume more storage than
adaptive selection; measured negative savings remain visible. The fixed decision
is recorded as `operation_selection.selection_version=3`, with strategy
`fixed_lossless_operation_v1`, the requested policy, and
`candidate_comparison_performed=false`. This appears in the immutable service
manifest, artifact metadata, reconstruction/generation provenance and database
checkpoint metadata. Tensor and patch semantic formats remain version 1.
Use new service IDs when changing an existing service's storage policy; immutable
manifests reject conflicting decisions even if the tensor values are identical.

Legacy configurations preserve adaptive decisions. Exact reconstruction compares the
canonical entry descriptors and raw bytes directly, avoiding two temporary
full-model serializations while still distinguishing signed zero and NaN payloads.
See [the measured checkpoint profile](cleanup.md) and
[the timing definitions](client-timing.md#checkpoint-substeps).

## Storage measurement

When `checkpoint_measure_full=true`, the exact path encodes an in-memory full target-state
candidate with the same content-addressed envelope and compression settings as
the persisted patch. It records actual encoded payload and manifest byte
lengths for both candidates, rather than estimating from parameter counts.
This is an apples-to-apples canonical tensor-state comparison, not an estimate
of a framework-native `.keras`, `.pt`, or `save_pretrained` container.

```text
incremental_savings_bytes = full_state_candidate_bytes - patch_bytes
incremental_savings_ratio = incremental_savings_bytes / full_state_candidate_bytes
```

The shared base size and reconstruction-manifest size are recorded separately,
allowing lineage totals to be calculated as full global bases plus their direct
client-round service patches.
Negative savings are retained honestly; they are not rewritten as zero.

## Verification and fallback

Before a patch is published, the generator captures the target, reconstructs
it from the persisted base and patch, verifies every content object, strictly
loads it into the model, and recaptures the target state hash. Wrong bases,
schema mismatches, corrupt payloads, missing/unexpected state keys, and failed
round trips fail closed.

The canonical lineage base `G0` is stored once in the content-addressed store.
FedAvg also stores every aggregated `G1...GR` as a full immutable tensor state
and verifies it after CAS read-back before it may parent another round. Global
states are training infrastructure and never rows in `services`. Model
configuration, architecture, tokenizer, processor, and image-processor assets
are separate content-addressed reconstruction support objects for retained
final-round clients. Together with the client patch and its explicit
`G(R-1)` base, these restore the complete trained model state. A native full
object is retained instead when exact tensor reconstruction is unsupported.
Optimizer, scheduler, RNG, and dataloader cursor state are outside this model
reconstruction policy.
