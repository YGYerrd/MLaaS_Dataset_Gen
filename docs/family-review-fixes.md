# Fixes found during the family review

The isolated September 2026 review found a runtime-data traversal bug while
checking SegFormer on ADE20K-mini. The worker reached task-contract binding before
training, then recursively converted the private prepared image/mask arrays into
Python lists. Repeated stack snapshots show the path from
`resolve_task_contract` through `plain(config)` into `ndarray.tolist()`.

Task-contract resolution now takes shallow copies of the input mappings, excludes
private runtime entries, and normalizes only the fields selected into the task
contract. Sentence-pair dataset arguments are copied before semantic resolution,
so resolving a contract does not mutate the caller. Existing semantic fields and
contract hashes remain unchanged for ordinary inputs. Restart workers to activate
the fix; it does not require changing model or dataset identities.

Structured-task evaluation also retains the per-class prediction and target counts
that its evaluator already computes. These use the existing
`metric_stat_class_<label>_pred_total` and `_target_total` names in service metrics.
This makes collapse review possible from persisted results, without another
inference pass. No extra model evaluation is introduced.

The relevant test selection passed 95 tests, including regression cases that
reject traversal of runtime arrays, preserve sentence-pair inputs, retain task
semantics, and verify prediction-count persistence from a single evaluation.

There is a separate numerical issue in the SegFormer pilot: the corrected
three-class head produced non-finite logits during its initial FP16 evaluation.
The review schedules a three-service FP32 follow-up using the same data, seeds and
training recipes. The label-head fix, runtime-data traversal fix and precision
choice are separate pieces of evidence; none alone certifies useful model quality.

Review artifacts and actual outcomes live in `outputs/family_review_20260913/`.
The original catalogues and their historical failures are preserved.

The full UPerNet/Apple-materials attempt revealed a separate preparation capacity
limit. Its segmentation path retains every processed FP32 image and int64 mask
in lists. The worker reached roughly 37 GB committed memory while still below
one fifth of the 33,118-image training split on a 32 GB host. It was stopped
before exhausting host memory and paging space, with no unfinished service rows
inserted. The review holds this exact family out; the convolution-layout timing
patch does not establish bounded-memory preparation or segmentation quality.
This review does not change the segmentation storage contract.

DPT/FoodSeg103 reached full preparation (4,983 training and 2,135 evaluation
images), but its worker also committed about 37 GB. Repeated snapshots remained
in model transfer to the GPU while system commit headroom fell below 2 GB.
That attempt was stopped and held as inconclusive too. The log confirms that the
36-layer DPT layout guard activated, which is evidence of activation but not of
end-to-end training or quality under this full preparation.

Review-specific validation additionally tests that cached family documents cannot
validate another service's incompatible or missing provenance, initial-model
comparisons share evaluation/initialization/precision, and saved checkpoints
still exist and contain data. The focused clearance selection passes 10 tests.

## DPT/UPerNet host-RAM preparation fix (2026-09-14)

Segmentation preparation now spills both processed FP32 images and int64 masks
to temporary files when their combined retained or projected split size exceeds
64 MiB. This applies to training and evaluation splits. The retained pool is
bounded by that budget plus individual processing buffers and row metadata;
training additionally retains only its current and configured prefetched batches.
Small splits keep their existing in-memory representation.

Disk-backed structured arrays preserve each sample's shape, dtype, class IDs,
ignore values and decode-error placeholders. Client selections share the files
without loading their arrays. Class inference, pixel distribution summaries and
prepared-data hashing scan one mask/image at a time; hashes remain identical to
the former per-sample lists. Batch encoding and training recipes are unchanged.
Classification retains its existing dense-array padding and hash convention.

The cache uses `MLAAS_IMAGE_CACHE_DIR` when set, otherwise the operating system's
temporary directory. It requires disk space for the processed pixels and masks
(approximately `N * (12 * image_height * image_width + 8 * mask_height * mask_width)`
bytes). Existing preparation cleanup closes both kinds of stores, including on
failures; views keep their backing file alive until released. Disk-write failures
abort preparation even with the decode-error skip policy.

Synthetic tests of 80 total 384-square and 512-square samples represented 225 MiB
and 400 MiB of expanded data while peaking at approximately 8 MiB and 14 MiB of
traced allocations, including class inference and prepared-data hashing. These
measure Python/NumPy allocations with a deterministic local processor, not total
process RSS or the original datasets. Regression coverage also checks selection,
decode failures, cache cleanup, fingerprints, distribution counts and CPU loss
backpropagation through the normal segmentation batch encoder.
The combined focused and related regression selection passed 105 tests.

The historical report and family clearance decisions remain unchanged. Full
DPT/FoodSeg103 and UPerNet/Apple-materials pilots have not been rerun by this fix;
end-to-end model quality still requires fresh evidence.

## SegFormer/ADE20K-mini production contract (2026-09-14)

The review's final manifest already requested FP32 for its 100 SegFormer-B0 /
`qubvel-hf/ade20k-mini` clients. Their referenced family document has three labels
(`background=0`, `person=1`, `car=2`), ignore index 255, and label reduction disabled.
Its file hash was verified. The remaining gaps were in fresh production generation
and initial preparation: the precision policy did not cover this pair, and the
preparation configuration omitted precision settings.

Production now applies `segformer_ade20k_mini_fp32_stability_v1` to SegFormer on this
dataset. Generated rows, resolved older rows, the service runner, and initial
family preparation explicitly select FP32 with mixed precision disabled. The HF
runtime also checks dataset identity and the loaded model's family, so reapplying
runtime knobs or reconstructing a local SegFormer checkpoint cannot enable AMP
for this pair. This policy does not force FP32 for SegFormer on full ADE20K or for
other architectures on ADE20K-mini.

Trainable mini-dataset rows explicitly carry `num_labels=3` and `num_classes=3`.
Preprocessing and model construction require the audited mapping and unreduced
masks. A separate preparation cache namespace prevents implicit discovery of old
150-label preparations; explicitly referenced families with incompatible counts,
label names/IDs, or reduction semantics are rejected before dataset/model loading.
Historical artifacts remain readable and are not rewritten. Existing corrected
families remain compatible: all 100 reviewed rows were resolved and checked against
their unchanged frozen family, with matching preparation keys and FP32 settings.

Validation passed 159 focused and related tests. This includes a real Transformers
SegFormer processor and tiny randomly initialized SegFormer model, starting from a
local 150-label base, resizing to the three-label head, initial structured
evaluation, a successful optimizer update, local checkpoint serialization and
exact tensor restoration, and evaluation after reload. A simulated CUDA-device
configuration verifies that the policy disables autocast before GPU operations.
These checks establish contract propagation and runtime behavior; they do not
rerun the full reviewed GPU pilots or establish new segmentation quality results.

## MobileBERT and ViT/SVHN production follow-through (2026-09-14)

Fresh MobileBERT classification/pair manifests now enable normalization, clipping
and FP32 explicitly, preserve them in the frozen recipe, and restore the model
adaptation through both training and inference checkpoint loaders. See
[MobileBERT production defaults](mobilebert-stability.md#production-defaults-verified-on-2026-09-14).
The existing 400 reviewed MobileBERT rows and their four frozen family documents
were checked and left unchanged.

ViT/SVHN's ordinary supported recipes previously borrowed same-model/task
calibration from other datasets, with targets as low as 46–77 optimizer updates.
The review's three healthy follow-ups each used 730 examples for six epochs,
552 attempted updates and 550 successful updates. The new optional, frozen
`vit_svhn_reviewed_opportunity_v1` policy uses 4,380 sample presentations and 552
planned updates as a conservative engineering reference for supported clients.
Epochs scale with the selected quantity, physical batch and accumulation; quantity
draws and their mean are preserved. This is not evidence that 730 distinct samples,
six epochs, or 550 successful updates form universal quality thresholds.

Supported partitions require all ten source classes, including under shard and
Dirichlet sampling. Source labels determine coverage; evaluation labels are not
consulted. Missing source classes or too few samples for coverage fail explicitly.
Challenging and stress recipes retain their weaker coverage/budget possibilities.
The policy does not apply to other datasets or clients resuming trained parents.
Older frozen policies retain their previous counting and budget behavior.

For this policy, planned update counting includes the final partial accumulation
window, matching HFCore. Actual training records both attempted and successful
updates. `training_opportunity_met` and `training_opportunity_status` distinguish
meeting the reference from a resource-limited run, a shortfall, or missing
measurements. Time limits remain in force; the recipe's supported band alone does
not claim that its runtime budget or model quality was achieved. Quantity and
joint-recipe changes remain extrapolations from the reviewed run, not additional
family clearance.

Tests cover generated-roster quantity reassignment, varying sample quantities and
epochs, ten-class partitions, reduced realized quantities, skipped/early updates,
and a real tiny ViT training loop with a partial accumulation window. Reference
values are bound to the preserved review results by their SHA-256. The review's
100 ViT/SVHN rows remain at their original 730-sample/six-epoch recipes. Full GPU
pilots and catalogue generation were not rerun by this change.
