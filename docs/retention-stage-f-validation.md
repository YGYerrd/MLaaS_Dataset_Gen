# Stage E/F validation — 15 September 2026

**Passed: 30 comparisons across six families.** These comprise 24 distinct
training configurations and six additional checkpoint-free twins. Every replay
ran in a fresh process using recorded raw rows, the family initial state, and
the effective recipe. Client checkpoints were read only after replay, for the
comparison. The six twins had no client checkpoint or service weight reference.

## Measured fidelity

| Model / dataset | Comparisons | Training rows per client | Reference rows | Original and reconstructed quality range |
| --- | ---: | ---: | ---: | --- |
| MLP / digits | 5 | 120–240 | 180 | Accuracy 0.388889–0.827778 |
| Logistic regression / wine | 5 | 48–96 | 18 | Accuracy 0.722222 |
| MLP / diabetes | 5 | 120–240 | 45 | RMSE 0.717493–0.762231 |
| TinyBERT / AG News | 5 | 16–32 | 128 | Accuracy 0.273438–0.515625 |
| DeiT tiny / beans | 5 | 16–32 | 64 | Accuracy 0.328125–0.578125 |
| TinyStories-1M / TinyStories | 5 | 16–32 | 32 | Perplexity 7.322225–7.811980 |

Across all 30 comparisons:

- Prediction agreement: **100%**, including generated text.
- Maximum absolute quality/evaluation-loss delta: **0**.
- Maximum absolute training-loss delta: **0**.
- Parameter relative L2 error: **0**; update cosine similarity: **1**.
- Every original model changed from its initial state.
- Original database metrics remained unchanged.

Bitwise equality occurred here but is not required by the gate. The measured
quality ranges describe the small validation clients; this experiment assesses
reconstruction fidelity, not their suitability for deployment.

## Scope and storage

All cases used CPU FP32 and native validation checkpoints. The cohort varies
Varied sample counts, IID, explicit class-label/score-bin Dirichlet and shard sampling;
seeds, quantities, batch sizes, epochs, dropout, gradient accumulation, warmup
and weight decay. Generation uses IID with varied sample counts. The recipe captures
the adapter's effective generation settings, including its actual token limit.

The database contains **30 authoritative recipes, six common initial checkpoints,
and 24 retained validation checkpoints**. SQLite integrity and foreign-key checks
pass. The materialisation CLI also exported a checkpoint-free digits service;
reloading its `.keras` export preserved all parameter values (Keras changes the
variable path prefixes when loading a standalone model).

**209 regression tests passed.** These cover true recipe-only persistence,
manifest execution/resume, absent-checkpoint reconstruction, effective settings,
integrity failures, fidelity failures, and existing generator/checkpoint paths.

The gate exposed two issues that were corrected: official text-supervised
evaluation splits were rejected for lacking numeric classification labels, and
family lookup treated the loader's generic task type as different from the
equivalent resolved HF task. Passing comparisons for unaffected tasks were
reused only after checking report content hashes, recipe identity, reference,
device and thresholds; failed generation cases were rerun.

GPU/mixed-precision runs, segmentation, detection, token classification, other
large models, interrupted training, skipped updates, and memory-recovery
trajectories are outside this measured cohort. Following this validation,
supported independent services default to `recipe_only` without validation
checkpoints; `weight_retention=checkpoint` remains an explicit option. The
functional gate and its tolerances are unchanged; bitwise equality is diagnostic.
Composition and FedAvg consumers remain unchanged.

### Default-policy regression validation

After switching the default, **248 targeted regression tests passed**, including
manifest execution and resume with no retention flags, explicit checkpoint
retention, and optional validation references. A full comparison with a deliberately
perturbed reconstructed parameter passed the functional gate with `bitwise_equal=false`;
original metrics were unchanged. Classification, generation and regression gate
tests also accept small quality, output and loss differences within the existing
tolerances. This verifies the policy change; the 30-service CPU cohort above was
not rerun for it.

## Reproduce and inspect

```powershell
python scripts/validate_client_reconstruction.py --suite representative --device cpu --output-dir outputs/new_stage_f_run
```

This uses the cached HF model revisions and records the exact configurations.
The completed run used a writable copy of the three raw dataset caches under
`.test-tmp/stage-f-hf-datasets`, with `HF_HUB_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`, and `OMP_NUM_THREADS=MKL_NUM_THREADS=4`.

Local evidence:

- Summary and per-service report references
- Database/storage verification
- Exact training configurations
- [Usage and gate thresholds](client-reconstruction.md)
