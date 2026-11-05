# Reconstruction task expansion — 15 September 2026

**Passed: 18 fresh-process comparisons across six task profiles.** Each profile
has two small IID clients with different sample counts and one separately trained unretained twin.
Generation and materialisation use the normal raw-source family executor.

| Profile | Model / dataset | Comparison |
| --- | --- | --- |
| Pair classification | TinyBERT / GLUE MRPC | Class predictions, accuracy, F1 and loss |
| Pair regression | TinyBERT / GLUE STS-B | Score RMSE, Pearson/Spearman and loss |
| Token classification | TinyBERT / WikiANN English | Supervised-token predictions, weighted F1, accuracy and loss |
| Summarisation | FLAN-T5-small / XSum | Decoded output agreement, ROUGE-1/2 and loss |
| Segmentation | SegFormer-B0 / ADE20K-mini | Valid-pixel predictions, IoU/Dice and loss |
| Masked LM | TinyBERT / AG News | Masked-token predictions, masked accuracy, perplexity proxy and loss |

All 18 comparisons have 100% prediction agreement, zero metric deltas and zero
training-loss deltas. Every replay verified the ordered selected rows and effective
settings. Original database metrics remained unchanged. Client checkpoint reads
were blocked during replay; reference weights were loaded only afterward.

The store contains 18 recipes, six common initial checkpoints and 12 retained
client checkpoints. All six unretained twins reconstructed successfully.

## Scope and common gate

The run used CPU FP32, native reference checkpoints, 8 or 12 training rows per
client, one epoch and a 16-row official evaluation slice (eight catalogue rows).
Sources stay full raw pools: for example XSum exposes 204,045 training rows while
the grouped clients prepare only their 20-row union. Model/dataset revisions,
preprocessing, source fingerprints and task protocols are recorded in the artifacts.
Segmentation uses 128-pixel processing and the existing three-label ADE20K-mini
contract. Generation uses 32 source tokens, 16 target tokens and 12 new tokens.

The common gate is fresh-process checkpoint-free replay, the same selected rows
and effective recipe, task-appropriate fidelity within tolerance, and rejection of
corrupt recipe/source evidence. Corruption checks are shared regression tests;
there is no per-task certification checklist or minimum number of comparisons.
These measurements establish reconstruction fidelity for the recorded scope.

The initial run exposed canonical token/masked-LM task names being resolved as
unknown, and list indices being passed to disk-backed segmentation data during
replay. Both were fixed. The resumed run reused nine intact passing pair and
summarisation comparisons and executed the remaining nine comparisons. The
summary identifies reused reports explicitly.

Validation also passed 215 generator/task/checkpoint regression tests, followed
by 71 focused tests after the final fixes, including the added masking exclusions.
Those counts overlap. Original corruption tests and the new recipe/source and
effective-setting drift tests all passed.

## Run and inspect

```powershell
python scripts/validate_client_reconstruction.py --suite tasks --device cpu --output-dir outputs/new_task_replay
python scripts/validate_client_reconstruction.py --suite tasks --task-profile masked_lm --device cpu --output-dir outputs/new_masked_replay
python -m mlaas_data_generator.cli.main task-capabilities --reconstruction-report <report-path>
```

The recorded run used cached models under `D:\huggingface\hub`, a writable dataset
cache at `.test-tmp/stage-f-hf-datasets`, offline HF mode, and
`OMP_NUM_THREADS=MKL_NUM_THREADS=4`. The profile harness resolves cached revision
pins and records the exact configuration.

- Summary and immutable comparison reports
- Exact configurations
- [Reconstruction usage and capability semantics](client-reconstruction.md)

Retention defaults and downstream composition/FedAvg consumers are unchanged by
this expansion. Translation, captioning, detection, retrieval, VQA, GPU/mixed
precision and recovery trajectories are outside this measured cohort.
