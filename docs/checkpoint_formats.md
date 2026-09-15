# Checkpoint formats and settings

New manifests save final neural-model weights as INT8 Safetensors. Training,
catalogue evaluation, update signatures, original model hashes and execution dtypes
remain unchanged. Only the stored representation is reduced. Existing manifests and
checkpoint artifacts retain their saved behavior.

## Settings

| Setting | New manifest default | Meaning |
| --- | --- | --- |
| `checkpoint_weight_format` | `int8` | Blockwise INT8; `fp16` selects the previous reduced representation and `native` selects exact source weights |
| `checkpoint_reduced_compression` | `none` | Normal Safetensors file; optional `zstd` or `zlib` outer compression |
| `checkpoint_reduced_compression_level` | `1` | Outer compression level |
| `checkpoint_validation_samples` | `256` | Maximum catalogue examples used to compare original/reloaded quality |
| `checkpoint_quality_tolerance` | `0.01` | Permitted primary-metric degradation |
| `checkpoint_patch_operation_policy` | `fixed_xor` | Exact-path/fallback operation; skips adaptive candidate comparisons |
| `checkpoint_patch_compression` | `zlib` | Exact-path codec; `zstd` and `none` also supported |
| `checkpoint_measure_full` | `false` | Skip full-source comparison encoding; unmeasured size/savings fields are NULL |

FP16/FP32/FP64 floating tensors are quantized to signed INT8 in blocks of 256 values,
with one FP32 scale per block. This uses about 1.016 bytes per floating weight before
container overhead, compared with 2 for FP16 and 4 for FP32. Integer and BF16 tensors
keep their dtype. Nonfinite tensors and tensors whose scales cannot be represented
safely retain their original precision. Unsupported tensor layouts use the native
fallback. Models identical to their base retain a zero-payload base reference.

The runner evaluates the original and decoded model on the same deterministic
catalogue subset before publishing. The tolerance is absolute for higher-is-better
metrics (one percentage point for accuracy), and relative for MAE/MSE/RMSE/loss/
perplexity (1%). Failed quality checks fall back to exact storage. Original catalogue
metrics remain unchanged, and the original model is always restored after checking.
This is sampled performance evidence, not a guarantee on every input.
Both passes complete the same bounded subset without the normal evaluation time
limit truncating them differently. Their cost is included in checkpoint runtime and
reported separately as `reduced_quality` in the checkpoint profile.

## Identity, integrity and downstream loading

Normal manifest startup skips a service immediately when its database status is
`completed`. It does not bind its family preparation, inspect checkpoint files,
hash payloads or reconstruct weights. Checkpoint verification remains separate from
normal resume. `--rerun-existing` explicitly bypasses this skip. Manifest progress
prints at most once every ten seconds plus a final summary, without redrawing a bar
after other output.

`mlaas-reduced-checkpoint-v1` records separate source and reload hashes, ordered tensor
names, shapes, source/stored dtypes, codec version, payload SHA-256, weight error and
quality validation. `final_model_hash` stays the original source identity. The
checkpoint's `target_state_sha256` is the hash of the decoded representation expanded
back into the original execution dtypes. Storage choices alone do not change service
IDs or sampling/training seeds.

The database's existing `full_object` mode is qualified by
`storage_policy_version="reduced_precision_v1"`. Resume, diagnostics and composition
understand the separate hashes. A passed round-trip check verifies the stored
representation; it does not claim bit equality with the evaluated source.

The existing `reconstruct_service_state()` / `restore_service_state()` APIs support
both legacy exact patches and reduced representations. A normal `.safetensors` payload
can also be opened by Safetensors tools. An outer-compressed payload must first be
decompressed. Model configuration/tokenizer/processor support artifacts remain shared.
The project loader restores original runtime dtypes: storage reduction does not imply
reduced inference memory or faster model execution.
For INT8, use the project loader so block scales are applied and auxiliary scale
tensors are removed before loading weights into the model. A generic Safetensors
reader exposes the quantized integers and scales without performing dequantization.

With a compatible model already constructed, downstream loading uses the same API:

```python
from mlaas_data_generator.checkpoint_store import restore_service_state

reload_hash = restore_service_state("outputs/service_artifacts_int8", service_id, model)
```

`reload_hash` identifies the reconstructed weights. The original evaluated identity
remains in the service's `final_model_hash` and the representation's source hash.

New HF exports freeze storage settings in both expanded and compact manifests. New
cohort exports carry `checkpoint_storage_options` separately from the frozen training
recipe. Existing manifests keep their original settings. Regenerating a manifest does
not convert or overwrite completed services. All 1,000 current-run recipe bindings
were checked against the updated resolver and matched.

Representations are implemented through `reduced_checkpoint.REPRESENTATIONS` with
separate encode/decode methods and versioned layout metadata. The default `int8`
representation now stores symmetric blocks of 256 values with FP32 scale tensors,
using `int8_block256_safetensors_v1`. It reuses the integrity and quality checks;
FP16 remains available explicitly. The [one-off legacy migration](checkpoint_storage.md#one-off-migration-of-trained-legacy-checkpoints)
defaults to INT8 and supports explicit FP16 selection. Historical benchmark scripts and measurements are in the [cleanup archive](cleanup.md).

## Storage measurement

Set `checkpoint_measure_full=true` when measured savings against a full native source
are needed. The exact path measures a canonical full-state candidate; the reduced path
measures a native Safetensors payload using the selected outer codec and common
reconstruction-reference overhead. With comparison disabled, actual stored bytes are
still measured, while comparison/savings fields remain unavailable.
