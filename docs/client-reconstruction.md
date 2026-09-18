# Durable families and client reconstruction recipes

Independent generation persists a `raw_source_family_v4` family (with v3 readers) and a
`client_reconstruction_recipe_v1` artifact for each trained service. Supported
independent services default to `recipe_only`, without client checkpoints.
[Recipe-backed composition](recipe-composition.md) verifies their family/recipe
evidence, searches without replay, then materialises and aggregates selected
services for audit evaluation. Existing records with the old
`composition_compatible=0` marker are verified by the new consumer without a
database backfill. FedAvg generation and checkpoint decoding remain available.

## Family contract

The existing family preparation document now identifies the raw source, model
and revisions, task/label/preprocessing contracts, shared fitted state, reserved
evaluation contract, and one common initial checkpoint. Raw Arrow fingerprints
or raw numeric-content hashes identify the source before preprocessing.

Family lookup excludes client IDs, quantities, distributions and sample seeds.
Neither its identity nor its saved configuration contains the client union or a
hash of the prepared training pool. Reopening a family prepares the current
clients' rows. Dynamic padding widths are execution details; fingerprints ignore
unused padding. Masked-language-model masks use stable source-row seeds.

Tabular scalers are fitted on the first family's selected union and frozen as
shared state. Later clients and reconstruction reuse those parameters. Therefore
creating a new family in a separate store with a different initial scaler fit can
produce a different family identity. HF label vocabularies and target
normalisation are recorded and verified against cheap raw-label fits on the
pinned source. No pool-wide tokenisation is needed for these fits.

## Authoritative client recipe

`service_generation_provenance.client_recipe_ref` and `client_recipe_hash` point
to an immutable, content-addressed JSON artifact. The same artifact is registered
as `client_reconstruction_recipe` in `service_artifacts`. It contains:

- Family reference and common initial-state hash.
- Ordered accepted `split:source_position` IDs, requested/realised quantities,
  and resolved distribution parameters and replacement evidence.
- Effective training settings, actual adapter arguments, fitted optimizer
  configuration where available, budget adjustments and memory-recovery evidence.
- Sampling, training, initialization and evaluation seeds; observed completion,
  optimizer updates and stopping reason.
- Raw-source, selected prepared-content and original-model fingerprints.
- Original observed metrics and runtime/version provenance.

Recipes are saved after training and evaluation. Completion verification checks
recipe integrity as well as the family and any retained checkpoint. The old
prepared-pool family versions require regeneration; no migration is necessary.

## Weight retention (Stage E)

Manifest rows and Python configurations accept `weight_retention=recipe_only`
or `weight_retention=checkpoint`. Supported trained independent families (`mlp`,
`cnn`, `logreg`, `hf_finetune`, `hf_train`, `transformers_finetune`) default to
`recipe_only`; other models, inference, shared-parent and FedAvg modes keep their
checkpoint policy. Explicit policies take precedence. Recipe-only services keep normal
IDs, metrics, provenance and authoritative recipes. This is separate from
diagnostic `no_save_weights`; explicitly combining the two is rejected.
Diagnostic runs continue to bypass durable preparation. Python callers execute
recipe-only clients through the manifest family executor; a bare `ServiceRunner`
cannot run an unprepared recipe-only service.

`retain_validation_checkpoint=false` is the default: recipe-only services store
no client checkpoint or patch. Set it to `true` to retain a validation reference,
or set `weight_retention=checkpoint` to retain client weights. The common family initial
checkpoint remains. After recipe persistence and verification, grouped execution
resets model weights and training state to the common initial state.
Recipe-only execution results do not carry trained tensor copies.

```powershell
python -m mlaas_data_generator.cli.main run-manifest --file clients.csv
python -m mlaas_data_generator.cli.main run-manifest --file clients.csv --weight-retention checkpoint
python -m mlaas_data_generator.cli.main run-manifest --file clients.csv --retain-validation-checkpoint
```

Reconstruction recreates a model from the recorded family, data and training
recipe for downstream benchmark use. Identical numerical weights or identical
AMP skip decisions are not required; benchmark fidelity is evaluated separately.

The storage-saving path requires completed epochs and allows skipped optimizer
updates up to and including 20% of attempted updates. The denominator is the
observed `optimizer_step_count`, not the number of batches or planned updates.
For example, 102 skips out of 512 attempts (19.92%) are accepted; 103 out of 512
(20.12%) are rejected. A nonzero skip count requires an observed attempt count,
and training must perform successful updates. No AMP scale history is stored.
OOM recovery and loss-precision retries remain unsupported. An unsupported
trajectory fails the service explicitly. Checkpoint retention can preserve those runs and their
stopping/recovery evidence. Inference-only, unsupported model families,
shared-parent and FedAvg modes cannot use `recipe_only` yet.

[GPU diagnostic comparisons](skip-update-reconstruction-validation.md) reproduced
native checkpoint hashes exactly at 10% and 13.33% skipped updates. The 20% limit
is a policy allowance, not an empirical guarantee for every run up to 20%; the
diagnostic did not measure runs near that boundary. Functional fidelity checks
and completion/recovery requirements are unchanged.

## Materialisation (Stage F)

`materialize_service` uses the recipe, family, recorded raw rows and common
initial state. It verifies prepared-content hashes, resets optimizer/RNG state,
replays effective settings and verifies the adapter's observed settings, ordered
selected rows, attempted update count and completed epochs. Both the original
and reconstructed run must satisfy the 20% skip limit independently; their
successful and skipped update counts need not match. Older recipes are checked
against the current eligibility policy using their recorded observations.
The original wall-clock cutoff is omitted
when replaying completed epochs. It does not read a
client checkpoint. Keep the context open when accessing its prepared reference
set; image-store resources close when the context exits.

```python
from mlaas_data_generator.client_reconstruction import materialize_service

with materialize_service("services.db", "client_id", device="cpu") as replay:
    model = replay["model"]
    training_observations = replay["training"]
```

Export reconstructed weights on demand and write separate observations:

```powershell
python -m mlaas_data_generator.client_reconstruction --db services.db --service-id client_id --output-dir outputs/materialized --materialize-only
```

## Validation

Generate 24 services across six offline tabular contracts, plus one truly
unretained client per family (30 comparisons):

```powershell
python scripts/validate_client_reconstruction.py --output-dir outputs/recipe_check
```

Use a new output directory. The cohort varies sample counts and IID/class-label or
score-bin Dirichlet/shard sampling, sample counts, batch sizes, epochs and seeds.
`--suite representative` replaces three tabular contracts with cached TinyBERT
text classification, DeiT image classification and TinyStories generation.
HF cases also vary gradient accumulation, warmup and weight decay.

Each comparison runs in a fresh process. An unretained client is compared
against its separately trained retained twin. `--resume-generation` verifies
completed services before reusing them. Use `--db <path> --service-id <id>` to
validate existing recipes instead.

For a resumed run, `--reuse-comparisons` can also reuse content-verified passing
reports with matching recipe, reference, device and thresholds. Use it only
when the comparison implementation for those tasks has not changed. The summary
lists every reused comparison; failed or missing cases still execute.

A single comparison:

```powershell
python -m mlaas_data_generator.client_reconstruction --db services.db --service-id client_id --output-dir outputs/comparison --device cpu
```

Immutable reports contain reference/reconstructed quality metrics, training
loss, prediction hashes and agreement, parameter relative L2 error and update
cosine similarity. Reference row IDs and prepared fingerprints identify the
comparison set. Original database metrics remain unchanged. Native validation
checkpoints isolate replay fidelity from quantisation effects.

The common gate is small:

- Reconstruct in a fresh process, with client-checkpoint reads blocked during replay.
- Verify the same ordered selected rows and effective training settings.
- Compare task-appropriate outputs and metrics within tolerance.
- Reject corrupt recipe/source evidence (covered by shared regression tests).

There is no minimum cohort size. Every requested comparison must pass. Default
tolerances are 0.02 for quality, 98% prediction agreement,
`0.0001 + 0.01 * abs(reference)` for training/evaluation loss, and
`0.0001 + 0.01 * reference_RMS` for regression prediction RMSE.
Correlation metrics use the quality tolerance, not the loss tolerance.

Generation compares exact generated-output agreement and task quality metrics.
Parameter similarity is diagnostic; bitwise equality is not required.
`--tolerances` accepts explicit JSON overrides. Failed gates return a nonzero
exit status and preserve failures in `summary.json`.

The gate establishes functional fidelity for the tested contracts and hardware.
Bitwise equality remains a reported observation, including on GPU and mixed
precision; it never determines whether reconstruction passes. Composition
records original and replayed identities separately; FedAvg generation is unchanged.

### Additional task profiles

The `tasks` suite contains pair classification (TinyBERT/MRPC), pair regression
(TinyBERT/STS-B), token classification (TinyBERT/WikiANN), summarisation
(FLAN-T5-small/XSum), segmentation (SegFormer-B0/ADE20K-mini at 128 pixels), and
masked LM (TinyBERT/AG News). It reuses the normal raw loader and cached pinned
revisions. Each profile trains two small IID clients with different sample counts plus one unretained
twin, then compares all three in fresh processes. Evaluation uses a declared
16-row official split slice. These are reconstruction checks, not quality benchmarks.

```powershell
python scripts/validate_client_reconstruction.py --suite tasks --device cpu --output-dir outputs/task_replay
python scripts/validate_client_reconstruction.py --suite tasks --task-profile masked_lm --device cpu --output-dir outputs/masked_replay
```

`--task-profile` can be repeated. Token and masked-LM preparation exclude sequences
without supervised positions and advance through the remaining client candidates.
Masked-LM families freeze the masking probability, source-row seed, special-token
exclusion and 80/10/10 replacement rule. Evaluation measures masked-token accuracy
and `exp(clip(token_mean_cross_entropy, -50, 50))`, named `perplexity_proxy`.
Comparisons ignore unsupervised positions. Token classification compares supervised
tokens; segmentation compares valid pixels; summarisation compares decoded text.

### Inspect reconstruction coverage

The existing capability report now includes `recipe_persistence`, `materialization`
and `reconstruction_validation`. Publication can establish recipe/materialisation
eligibility without a client checkpoint. Measured validation stays pending until
comparison evidence is supplied; backend availability does not imply fidelity.

```powershell
python -m mlaas_data_generator.cli.main task-capabilities --task fill_mask --backend hf_finetune
python -m mlaas_data_generator.cli.main task-capabilities --reconstruction-report <content-addressed-report.json>
```

The second command verifies report integrity and displays the exact family,
recipe, model/dataset configuration, task contract, device and runtime measured.
Reports are the evidence store; there is no separate certification registry.
Old comparisons can remain as observations, but current validation reuse requires
the v3 comparison evidence. Masked-LM support here does not enable composition.

## Measured result

The [Stage E/F validation report](retention-stage-f-validation.md) records
30 passing CPU comparisons across six families: 100% prediction agreement,
zero quality/training-loss deltas, and six successfully reconstructed services
with no stored client checkpoint. The report states the tested scope and limits.

The [task expansion report](reconstruction-task-validation.md) records 18 further
passing comparisons covering pair classification/regression, token classification,
summarisation, segmentation and masked LM, with six checkpoint-free twins.
