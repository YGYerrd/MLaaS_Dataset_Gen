# DPT batch-size slowdown: controlled diagnostic

Date: 13 September 2026.

Implementation update: the tested workaround is now enabled automatically by
HFCore for DPT models on Windows ROCm. It is implemented in
`mlaas_data_generator/models/adapters/rocm_dpt_layout.py`, with regression tests
covering platform/model scope, output and gradient preservation, repeated setup,
deepcopy, serialization, and HFCore integration. All 47 targeted tests passed.
The production setup path was benchmarked on the same prepared images: median
batch-size-two evaluation **38.35 ms**, training update **161.83 ms**, and peak
training allocation **4.017 GiB**. Raw results are in
`outputs/checks/dpt_batch_diagnostic/default-fp16-runtime.json`.
Restart a running manifest process to activate the change. The diagnostic
script keeps its original baseline by explicitly disabling the runtime fix;
use `--runtime-layout` to test the production configuration.

## Finding

The reported slowdown is reproducible and is a software/kernel-layout pathology,
not exhaustion of the RX 9070 XT's 16 GB VRAM. With identical prepared images,
FP16 precision, model weights and resolution, physical batch size two selects
extremely slow NHWC convolution kernels. Making convolution inputs contiguous
NCHW removes the slowdown while retaining physical batch size two.

The initial investigation below used only `scripts/diagnose_dpt_batch.py` and
its output files. The subsequent production implementation is described above.

## Controls and method

- GPU: AMD Radeon RX 9070 XT, gfx1201, 15.922 GiB reported capacity.
- Runtime: Windows; PyTorch `2.9.1+rocmsdk20260116`; HIP
  `7.2.26024-f6f897bd3d`; Transformers `5.7.0`.
- Model: `Intel/dpt-large-ade`, local revision
  `5ca59e2439c8340340eafccba2377816903f444a`, original 150-class head.
- Dataset: cached FoodSeg103 validation rows 0 and 1. These are diagnostic
  examples, not a reconstruction of the exact randomized service partition.
- Preparation: the repository's `_process_split`, DPT's saved image processor,
  480 x 480 RGB, no augmentation. Prepared arrays are saved once and reused.
- FP32 parameter storage with FP16 autocast; native ROCm BatchNorm workaround;
  gradient checkpointing enabled, matching the HFCore configuration.
- Input tensor SHA-256:
  `eb692d0ddaa9ac797fea99a9f018454885d39a4552ecedee5d31d5297f58f13b`.
- Label tensor SHA-256:
  `939ae942de6a26d39ac44a2d1fa8e8ee88890e13bd9a4ed98fce874ef9a8492f`.
- Evaluation compares the same two-image workload: two forwards of size one
  against one forward of size two. Conditions run in order 1, 2, 2, 1, with
  warm-up excluded and three measured repetitions per condition.
- Training uses SGD, learning rate 1e-5, weight decay 0.05, and a matched
  effective batch of two images. Physical size one uses accumulation two;
  physical size two uses accumulation one. This accumulation adjustment is
  deliberate, so the two cases process the same images per optimizer update.
  It is not a claim of identical BatchNorm/dropout behavior across batch sizes.
  Initial weights and RNG are reset between training conditions. One warm-up
  update and three measured updates are performed per condition.
- The decisive workaround comparison holds physical batch size two AND
  accumulation one fixed; only convolution input layout changes.
- Prepared images and masks remain on the GPU during measured compute. Transfers
  are timed separately. CUDA synchronization brackets wall-clock measurements.
- Separate processes test default MIOpen versus `MIOPEN_FIND_MODE=FAST`.
  MIOpen databases and kernel caches are isolated under the diagnostic directory.
- The user confirmed no other GPU-heavy job was running.

## Warm results

All evaluation times below cover the same two images. Training times cover one
optimizer update on the same two images. GiB values are PyTorch peak allocated
device memory, including the model.

| Configuration | Evaluation, ms | Training update, ms | Eval peak, GiB | Train peak, GiB |
|---|---:|---:|---:|---:|
| Default, physical batch 1 | 41.39 | 210.59 | 2.470 | 4.011 |
| Default, physical batch 2 | 4,672.53 | 8,970.39 | 2.658 | 4.022 |
| Contiguous convolution inputs, batch 1 | 41.44 | 207.07 | 2.470 | 4.017 |
| Contiguous convolution inputs, batch 2 | 38.77 | 163.91 | 2.658 | 4.017 |
| MIOpen FAST alone, batch 2 | 4,673.57 | Not tested | 2.658 | — |

For physical batch size two, changing layout gives about **120.5x faster
evaluation** and **54.7x faster training updates**. The two-image microbatch
workaround also avoids the slow path, but is slower than retaining batch size
two with the layout correction.

The baseline's roughly 8.97 seconds per physical batch of two reproduces the
user log's roughly 17.9 seconds per two training batches. At 256 physical
batches, this predicts approximately 38.3 minutes, matching service 641.
This is a compute-path explanation, not a data preparation estimate.

## Where the time goes

Layer-level GPU events localize the delay to convolution operations. Examples:

| Layer, physical batch 2 | Default, ms | Contiguous inputs, ms |
|---|---:|---:|
| `head.head.0` | 1,139.55 | 1.19 |
| `auxiliary_head.head.0` | 1,132.73 | 1.16 |
| `neck.convs.0` | 282.89 | 0.38 |

The baseline FindDb entry for a 256-channel, 240 x 240, 3 x 3 convolution with
batch size two identifies `NHWC-NHWC-NHWC-FP16` and lists only
`ConvDirectNaiveConvFwd`, measured at about 1,133 ms. The batch-size-one NCHW
entry includes GEMM and Winograd candidates measured around 1.25 ms and 0.63 ms.
Changing layout at the same batch size makes fast kernels available.

The installed DPT readout code permutes projected tokens back into feature maps
without a final contiguous conversion (`modeling_dpt.py`, near line 550).
This is a plausible origin of the layout entering the convolution chain.
The measured MIOpen layout and kernel difference is direct evidence; the exact
upstream layout-propagation/dispatch decision was not traced through native
PyTorch internals.

The tested workaround attaches pre-hooks to Conv2d and ConvTranspose2d modules
that replace their input with `input.contiguous()`. It preserves values, shape,
weights, precision, batch size, loss and BatchNorm configuration. It changes
memory layout, and can introduce a device-side copy when needed. The measured
speedup includes that copy's cost.

## Alternative explanations

- **VRAM capacity / paging:** not the cause in this reproduction. Evaluation
  peaks at 2.66 GiB allocated and at most 3.15 GiB reserved, with more than
  12 GiB reported free. Training peaks at 4.02 GiB allocated, with more than
  10 GiB free. Sampled Windows process counters also showed modest dedicated
  usage (approximately 3.75–4.58 GiB) and shared usage (approximately
  77–107 MiB), not multi-gigabyte shared-memory spill. Those Windows samples
  are spot checks, not a continuous residency trace.
- **Input transfers:** images took 0.90 ms and labels 0.61 ms to transfer once.
  The 4.67-second slowdown persists with both already resident on the GPU.
- **Synchronization / Python overhead:** convolution dispatch itself takes
  about 0.05 ms for the worst layers; their device execution takes hundreds
  of milliseconds. Synchronization exposes the cost but does not explain it.
- **Optimizer:** the SGD optimizer phase is approximately 21.5 ms in both
  slow and corrected batch-size-two runs. Baseline forward/backward phases
  are about 4.68/4.26 seconds; corrected phases are about 43/98 ms.
- **Allocator cache buildup:** the slowdown persists after `empty_cache()`
  and in a fresh process; order 1,2,2,1 returns to fast batch-size-one results.
- **MIOpen search setting:** FAST alone did not remove the NHWC slowdown.
- **Cold compilation:** real but separate. Baseline first evaluation calls
  took 16.6/22.0 seconds for the two physical batch sizes; first training
  updates took 18.7/31.1 seconds. Warm calls still exhibit the large gap.

## Numerical check and limits

A separate same-model, same-weights, physical-batch-two comparison changed only
the layout hooks. Both outputs were finite. Pixel argmax agreement was
**99.902%**, mean absolute logit difference **0.001665**, maximum absolute
difference **0.015627**, and relative L2 difference **0.000832**. Evaluation
loss was **10.398562** versus **10.399912**. FP16 outputs are close, not bitwise
identical, because the selected convolution implementation changes.

Batch-size-two training losses after the first measured update were 7.107378
versus 7.107783 and declined over the three measured updates in both runs.
This short diagnostic establishes the performance cause and basic numerical
behavior; it does not establish full-training quality equivalence.

This Windows PyTorch build exposes only the CPU profiler activity. GPU events
are available, but some aggregate phase event readings disagree with synchronized
wall time, particularly during backward execution. The report therefore uses
synchronized wall time for end-to-end comparisons; layer events are supported
by the independently recorded MIOpen solver timings. No hardware kernel trace
was captured. MIOpen also emitted an invalid elapsed-time warning during initial
selection. The benchmark processes wrote their complete JSON measurements but
returned shell exit code 1 without a terminating Python traceback; the final
verification also printed `DIAGNOSTIC_COMPLETE`. This runtime exit anomaly is
recorded separately and was not treated as a successful process-status check.

## Action supported by the evidence

Use a narrowly scoped DPT/Windows-ROCm contiguous-convolution-input workaround,
then validate a representative service end to end before broad rollout. It is
preferable here to reducing physical batch size, because it retains the original
batch/BatchNorm behavior. Microbatching plus accumulation remains a measured
fallback, with its BatchNorm caveat. A larger GPU is not indicated by these
measurements. No artificial VRAM pressure test was needed to explain a slowdown
that reproduces with more than 10 GiB free.

## Reproduce and inspect

From the repository root, with the existing offline model and dataset cache:

```powershell
.venv/Scripts/python.exe scripts/diagnose_dpt_batch.py --output outputs/checks/dpt_batch_diagnostic --layers
.venv/Scripts/python.exe scripts/diagnose_dpt_batch.py --output outputs/checks/dpt_batch_diagnostic --contiguous-conv --layers
.venv/Scripts/python.exe scripts/diagnose_dpt_batch.py --output outputs/checks/dpt_batch_diagnostic --find FAST --eval-only --repeats 2
.venv/Scripts/python.exe scripts/diagnose_dpt_batch.py --output outputs/checks/dpt_batch_diagnostic --verify-layout --eval-order 2 --eval-only --repeats 1
```

Raw measurements, prepared inputs, hashes, layer timings, and MIOpen databases:
`outputs/checks/dpt_batch_diagnostic/`. The compact comparison is `summary.json`.
Console logs are the adjacent `dpt_batch_diagnostic*.log` files.

PyTorch distinguishes allocated tensor memory from reserved allocator memory,
and notes that its memory tools cannot see all external GPU allocations:
[CUDA memory semantics](https://docs.pytorch.org/docs/main/notes/cuda.html),
[memory visibility limits](https://docs.pytorch.org/docs/main/torch_cuda_memory).
AMD documents FAST as FindDb/immediate fallback selection, rather than a
guarantee that every convolution will use a fast solver:
[MIOpen environment variables](https://rocmdocs.amd.com/projects/MIOpen/en/latest/reference/env_variables.html).
