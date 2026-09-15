# CLIP contrastive batch validation

Validated on 2026-09-13 using the cached `openai/clip-vit-large-patch14` checkpoint,
AMD Radeon RX 9070 XT (15.922 GiB exposed), PyTorch 2.9.1+rocmsdk20260116 and
HIP 7.2.26024-f6f897bd3d. This is a memory/correctness smoke test with synthetic
pairs, not a dataset quality result.

The production HFCore path received the old physical-batch request of 1. Its
16 GiB policy selected 8, with gradient accumulation 2. Inputs were 17 fixed
distinct synthetic pairs, 224x224 float32 prepared images and 77 tokens, FP16
autocast, FP32 model parameters, gradient checkpointing, AdamW at 1e-6, and zero
weight decay. Eight epochs exercised partial batches of 8, 7 and 2.

| Measurement | Result |
|---|---:|
| Successful optimizer updates | 10 |
| Initial AMP skipped updates | 6 |
| Training examples processed | 136 (17 x 8) |
| Peak allocated GPU memory | 8.146 GiB |
| Peak reserved GPU memory | 8.402 GiB |
| Maximum text-projection weight change | 9.93e-6 |
| Mean recorded training loss | 2.2849 |
| Training wall time, warm kernel cache | 8.40 s |
| Evaluation loss, encoding batch 1 | 2.54832 |
| Evaluation loss, encoding batch 4 | 2.54937 |
| Evaluation candidates for both | 17 |

The first shorter smoke run had only four attempted updates, all skipped by AMP,
and therefore did not establish optimizer-state memory usage. The longer run above
includes actual AdamW state allocation and verified weight changes. Production
now rejects a retrieval run with zero successful updates.

The CPU regression tests compare pooled loss with dense symmetric CE, including
more than one CPU score block, and enforce candidate-set invariance across
encoding batches and synthetic OOM recovery. They also check nonzero weight
updates, no singleton training forwards, sample preservation, recipe defaults,
partial-batch handling and explicit failure when the two-pair floor cannot fit.
The final regression selection passed 120 tests (retrieval, microbatch recovery,
precision, evaluation protocols, recipe calibration and manifest pipeline).

Reproduce from the workspace with the cached model path appropriate to the machine:

```powershell
.venv/Scripts/python.exe scripts/diagnose_contrastive_batch.py --model-path D:/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41 --output outputs/checks/clip_contrastive_batch
```

Artifacts: `outputs/checks/clip_contrastive_batch/result.json`, `prepared.npz`, and
`outputs/checks/clip_contrastive_batch.log`. The completed GPU process returned
exit code 1 after writing the results and printing `DIAGNOSTIC_COMPLETE`, without
a Python traceback, matching the previously observed ROCm teardown behavior.
The successful-update counter, changed weights and completed result file provide
the validation evidence; the process exit itself was not clean.
