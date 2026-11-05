# Task extension smoke evidence — 11 September 2026

All three profiles completed both independent-family and trained-parent lifecycles:
four frozen clients per mode, exact native checkpoints, catalogue diagnostics,
strict v5 request generation and search, and validation of the selected subset.
These are small mechanical verification runs, not benchmark accuracy claims.
The backend-wide declarations remain experimental for the new measured protocols.

| Profile / runtime | Mode | Parent | Catalogue-selected individual | Selected aggregate | Audit supervision |
| --- | --- | ---: | ---: | ---: | --- |
| TinyBERT / WikiANN, ROCm GPU | Independent | 0.084734 | 0.317647 | 0.291560 | 68 tokens |
| TinyBERT / WikiANN, ROCm GPU | Shared parent | 0.379783 | 0.591335 | 0.400666 | 68 tokens |
| SegFormer-B0 / ADE20K-mini, CPU, 128 pixels | Independent | 0.010433 | 0.007023 | 0.006820 | 8,192 pixels |
| SegFormer-B0 / ADE20K-mini, CPU, 128 pixels | Shared parent | 0.005280 | 0.044041 | 0.005441 | 8,192 pixels |
| FLAN-T5-small / XSum, ROCm GPU | Independent | 0.099173 | 0.115840 | 0.124107 | 128 target tokens |
| FLAN-T5-small / XSum, ROCm GPU | Shared parent | 0.101104 | 0.093107 | 0.163396 | 128 target tokens |

The primary metrics are supervised-token weighted F1, corpus macro IoU and decoded
ROUGE-1 F1 respectively. Every audit partition has eight source examples. Parent
and individual baselines use the same audit membership and task protocol as the
aggregate. The individual is selected using catalogue quality with a stable ID
tie break. Aggregate quality improvement is not a pass requirement.

The successful runtime is PyTorch 2.9.1 with Transformers 5.7.0 on Windows; GPU
runs use the AMD Radeon RX 9070 XT. Segmentation uses a separately frozen 128-pixel
CPU processor protocol. Summarisation limits sources to 32 tokens, targets to 16
tokens and generation to 12 new tokens with deterministic decoding. The runs use
seed 42, 48 declared source training examples and 16 source evaluation examples.
The cached Arrow file hashes and selected row positions identify the actual smoke
source; these results do not promote the unrestricted production datasets.

Local evidence bundles, including full revisions and task/protocol identities:

- Token classification
- Semantic segmentation
- Seq2seq summarisation

Each directory contains the immutable recipe manifests, database, reservations,
diagnostics and validation reports. `verification.json` and its `shared_parent/`
counterpart record deterministic stored-state aggregation, contributor weights,
preserved tensors, baseline scores and rejected parent/schema/contract mismatches.
Resume verifies roster identity and unchanged immutable artifact bytes and
modification times. Prediction hashes provide a streaming replay identity without
retaining full logits. Runtime versions, hardware, memory and verification code
hashes are recorded in the evidence JSON. These output directories are ignored by
Git; copy the bundles separately when sharing the results.

The first segmentation GPU trial failed with `miopenStatusUnknownError`. The
512-pixel CPU trial was stopped before completion. Their separate evidence folders
retain failed/interrupted status; neither certifies that runtime/protocol. Earlier
development trials also exposed local checkpoint loading, target-hash comparison,
recipe partition and CSV contract-decoding errors. Those were fixed before the
final fixtures completed, without changing the final frozen client rosters.

The active regression suite passes **299 tests**, including 37 restored archived
token, generation-preprocessor and image-adapter checks. Unit/integration coverage
includes corrupt checkpoint payloads, semantic/protocol changes, grouped audit
overlap, changed eligible membership, absent/ignored labels, uneven batches,
masked targets and decoded-output failures. See [task capabilities](task-capabilities.md)
for reproduction commands and the supported/excluded protocol boundaries.
