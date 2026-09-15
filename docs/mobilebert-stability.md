# MobileBERT instability investigation and verified fix

The MobileBERT/MRPC service `svc_7350ae945284__debug_no_weights` failed after one SGD update with a non-finite training loss. Its original learning rate was 0.01, batch size 8, gradient accumulation 2, and execution precision FP32.

## Findings

A fresh CPU FP32 load of the pinned `google/mobilebert-uncased` revision `1f90a6c24c7879273a291d34a849033eba2dbc0f` already produced logits around -6 million to -11 million on two fixed sentence pairs, before any training. Cross-entropy was approximately 2.59 million. The final two encoder outputs reached approximately 1,007 and 74.4 million in absolute magnitude. Eager attention and SDPA reproduced the same behavior. Every loaded checkpoint tensor matched the cached source tensor, and the checkpoint file hash matched its content-addressed cache filename.

The checkpoint config uses `normalization_type=no_norm` and disables classifier activation. Upstream MobileBERT implements NoNorm as a learned affine transform, and its disabled-activation pooler passes the first-token features directly to the classification head. This offers no normalization between the observed extreme encoder features and the new task head. [Upstream implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/mobilebert/modeling_mobilebert.py).

This is a measured numerical conditioning problem, not evidence of corrupted cached weights or a demonstrated attention-backend defect. The original failure records did not retain the first invalid gradient/parameter, so the precise first tensor to become non-finite is unknown.

There is also a separate data limitation: the MRPC RMSprop recipe trains on 242 examples of class 0 and zero examples of class 1. Its continued poor accuracy cannot be repaired by stabilizing arithmetic alone.

## Applied changes

- Opt-in `mobilebert_head_normalization=layer_norm_v1` applies parameter-free feature normalization immediately before the MobileBERT linear classifier. Original parameters and checkpoint tensor keys are retained.
- Opt-in `max_grad_norm=1.0` clips accumulated gradients after AMP unscaling, and rejects non-finite gradient norms before an optimizer update. It records gradient norms and clipping counts, including grouped execution.
- The normalization policy is stored in the model config and service configuration and included in the family preparation identity. Reloading through the generator restores the normalization. Untagged existing checkpoints keep their original behavior.
- Fixed retry IDs use `__mbnorm_clip_v1`, keeping their changed model semantics distinct from the old results. Recipes, learning rates, seeds and pinned revisions were retained.

Normalization is an explicit model adaptation, not a transparent numerical equivalence claim. Load these checkpoints through the generator with their saved configuration; generic Transformers code does not implement this custom normalization solely from its config marker.

## Final validation

Six MobileBERT services (three MRPC, three STSB) were run with full checkpoint persistence and verification. All six completed, and all six checkpoint round-trip statuses are `passed`. The previously failed SGD service completed 110 successful optimizer updates, clipping 96; its training loss was 0.6504.

| Dataset / optimizer | Previous result | Fixed result | Fixed training loss |
|---|---|---|---:|
| MRPC / SGD | Non-finite loss failure | Accuracy 0.6863 | 0.6504 |
| MRPC / AdamW | Accuracy 0.3162; training loss 2.81 million | Accuracy 0.6863 | 0.6733 |
| MRPC / RMSprop | Accuracy 0.3162; single-class training data | Accuracy 0.3137 | 0.2549 |
| STSB / Adam | Pearson -0.1166 | Pearson 0.6874 | 0.2061 |
| STSB / AdamW | Pearson -0.0921 | Pearson 0.2687 | 0.2907 |
| STSB / RMSprop | Pearson -0.0521 | Pearson 0.5058 | 0.1309 |

The quality comparison is indicative, not a controlled same-benchmark comparison: the old debug run evaluated 408 MRPC / 1,050 STSB examples, while the full family pipeline evaluated 204 / 988. MRPC accuracy of 0.6863 is the fixed benchmark's majority-class baseline, so these short recipes are numerically stable but not yet strong classifiers. No benchmark-driven recipe search or label balancing was performed.

124 distinct targeted tests passed across the stability, AMP, microbatch, runner, checkpoint, and no-save-weights suites. Tests include rejection before mutation, grouped-client clipping propagation, state/RNG preservation, distinct family identity, and restoration from a saved model config. The final database also passed SQLite quick-check and foreign-key checks.

## Artifacts

- Final retry manifest: `outputs/checks/mobilebert_stability_20260913/final/retry.csv`
- Verified results: `outputs/checks/mobilebert_stability_20260913/final/services.db`
- Results CSV: `outputs/checks/mobilebert_stability_20260913/final/results.csv`
- Per-service before/after metrics: `outputs/checks/mobilebert_stability_20260913/comparison.json`
- Full log: `outputs/checks/mobilebert_stability_20260913/final/run.log`

The merged catalogue and its original failed records were left unchanged. The verified results live in the isolated database above.

## Production defaults verified on 2026-09-14

Fresh registry-generated MobileBERT sequence-classification and sentence-pair
training recipes now explicitly select `layer_norm_v1`, `max_grad_norm=1.0`, and
FP32. These settings survive compact manifest save/load and are frozen in the
training policy; recipe validation rejects changes to them. Clients of an already
adapted parent inherit the same runtime settings. Historical calibration evidence
without an adaptation binding is not credited as support for the changed model.
New recipes remain varied and can still have weak quality outcomes.

Family preparation retains the normalization marker and FP32 configuration. The
model-config marker changes architecture identity even though tensor keys are
unchanged. Both fine-tuning and inference adapters now restore normalization from
saved model configs; an explicit conflicting normalization setting fails on a
tagged checkpoint. Untagged historical checkpoints and frozen recipes retain
their original semantics.

The existing review roster's 400 MobileBERT rows already requested normalization,
clipping and FP32. All four referenced family documents passed file-hash and
normalization-marker checks, and still match their resolved preparation keys.
Local tests exercised a real tiny MobileBERT optimizer update, tensor restoration,
and identical adapted logits after both fine-tuning and inference reloads. The
full reviewed GPU pilots were not rerun.
