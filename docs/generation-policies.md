# Catalogue-informed recipe generation

Client data selection now follows [Stage A + B](tokenization-selection.md): IID
uses uniform raw rows, skewed profiles require an explicit axis,
and `sample_size` is honored without a second capacity scaling in the allocator.
This supersedes the historical sampling descriptions below; training-budget
policies remain unchanged.

New structured tasks also freeze a [task contract](task-capabilities.md).
Fill-mask, segmentation and summarisation recipes use IID with independently varied
sample counts. Shared-parent recipes take their supported current profiles from
the parent's frozen contract.
Historical recipes retain their existing readers and identities.
Historical calibration templates are not applied to new task protocols: they do
not bind the new semantic and evaluation hashes. These recipes remain explicitly
uncalibrated until comparable evidence exists.

New clients use `catalogue_training_policies_v2` and `conditional_balancing_v5`.
Training policies contain one flat `limits` object; resource tiers are no longer generated.
The ordinary `hf-manifest` command enables calibration automatically for
independent text/image/token classifiers and confirmed sentence-score regressors. No database queries, model training or benchmark
filtering happen during manifest generation.

## What changes

IID, Dirichlet and shard remain available, as do Adam, AdamW, SGD
and RMSprop for HF text. Quantities retain their independently shuffled fixed
+/-75% spread before capacity adjustment. An average of 1,000 gives initial
draws between 250 and 1,750. When a true training capacity is known and is below
the maximum draw, the whole quantity range is scaled proportionally before
recipes are frozen. For example, WNLI's 635 training rows scale that range to
approximately 91–635. `unconstrained_sample_size` retains the original draw.
The requested average can therefore exceed the final realized average.
Synthetic generation uses the actual 2,000-row source and training split
(1,600 rows at the default 20% test split). Loader caps are not treated as
dataset cardinalities. Scaling, rather than clipping high draws to the pool
size, preserves quantity differences.

Within each calibrated family training policy, generation allocates clients as follows:

| Recipe band | Target allocation | Behavior |
| --- | ---: | --- |
| supported | 70% | Use an observed joint recipe with broadly viable historical outcomes |
| challenging | 20% | Explore the existing broad optimizer-specific ranges |
| stress | 10% | Use the shortest exploration epoch budget and highest allowed rate for the selected optimizer |

Counts use deterministic largest-remainder rounding. Small rosters cannot
represent every band or optimizer. Manifest truncation recalculates the recipe
allocation for the final roster. These percentages describe intended training
conditions, **not guaranteed healthy/weak/collapsed outcome proportions**.
They are initial experimental settings, not percentages inferred from history.

Supported recipes preserve the observed optimizer, LR, batch size, accumulation,
weight decay and warmup together. They never combine independently estimated
best ranges. Starting from the observed epoch count, smaller quantities receive
the fewest epochs needed for the recorded median update budget, capped at 200
for new v5 policies (the cap is frozen in the policy).
The frozen LR and effective batch are not changed to achieve that budget.
A request that cannot supply it fails explicitly. Wall-time limits still apply;
actual convergence and composition performance are not guaranteed.

Classification sampling uses only the training distribution. For supported
Dirichlet clients, the sampled class proportions are blended equally with the
source proportions; supported shards retain at least two available classes.
Required selected classes receive at least one example when capacity permits.
Challenging Dirichlet clients retain a 10% source mixture; challenging and stress
shards can remain single-class. Stress Dirichlet uses the original unblended
sampling. IID sampling retains the source distribution in expectation.
This keeps non-IID variation while reserving the most extreme binary profiles
for the difficult portion of the catalogue. Binary supported shards necessarily
retain both labels; multiclass shards can still specialize in a subset.
No evaluation labels enter these sampling rules. Token clients apply these
controls to their declared sentence-level entity-presence buckets when available;
they do not sample individual tokens or turn masked padding into class labels.

`calibration_band` is visible in the compact client review CSV. The selected
template ID, evidence hash, fallback scope, joint settings and coverage controls
are frozen in the companion file and training-policy hash. A band is a recipe
annotation; it must not be interpreted as measured model quality.

## Historical evidence and fallback

The checked-in `mlaas_data_generator/clients/recipe_evidence.json` was mined from
`outputs/services.db`: 19,406 total records. Eligible independent HF records
include 2,420 text classifiers, 1,700 image classifiers, 2,045 token classifiers
and 1,795 sentence-pair models. 6,466 observations have usable metrics/contracts;
252 records have nonfinite or missing loss, and 1,242 binary sentence-pair records
have unverified regression contracts (their legacy correlation names contain
classification metrics). Another 1,672 records have insufficient training-class
coverage for attributing outcomes to their training recipe. Those skewed records
remain in the descriptive family outcome counts.

The evidence contains 472 text, 545 image, 308 token and 101 continuous-score
joint recipe groups. These are observations, not all supported recipes. Task
pools are strictly separate: a token or image model cannot borrow text-classifier
evidence just because optimizer settings happen to match.

Other tasks, generic CNN/MLP clients, pretrained task-parent clients and multiround
FedAvg retain their existing policies. In particular, old generic regression
metrics are not imported as if their historical MSE/RMSE issue were resolved.

Each observation is compared to its own benchmark class distribution:

- `healthy` classification: accuracy and task-specific F1 each exceed the
  majority-class baseline by at least 0.05. Text and tokens use support-weighted
  F1; images use their explicitly stored macro F1.
- `healthy` sentence-score regression: both Pearson and Spearman are at least
  0.3. These are correlation criteria, not guarantees of low absolute error.
- `collapsed_signature`: the two metrics match a constant-class baseline.
  Multiclass signatures are candidates, not proof of constant predictions.
- `weak`: other finite classification outcomes.

Token histograms must contain every supervised label and sum to the recorded
supervised-token count; truncated top-k summaries cannot calibrate recipes.
Image/token coverage thresholds scale with the number of classes, so a balanced
100-class dataset is not rejected for having less than 5% of each label.
Zero correlation is classified as weak, never as proof of regression collapse.
Sentence-score generation requires an explicit regression dataset contract.
WNLI/MRPC resolve to two-class sentence-pair classification before recipe
selection, preprocessing and model construction. Their heads use cross-entropy
with integer labels, and their metrics are accuracy/F1. ClassLabel IDs retain
their dataset-defined meanings. These families can use the explicitly recorded
text-classification evidence fallback; historical WNLI/MRPC records with
unverified regression contracts remain excluded from calibration. STS-B and
SICK score prediction retain regression heads, losses and correlation metrics.

These thresholds are transparent triage rules, not universal model-quality
standards. Execution failures, inference-only services, time-censored training,
nonfinite records and incompatible tasks are excluded from recipe estimation.
Imported duplicate observations are counted once. There are no attempt failure
rates: this catalogue mostly provides completed-service evidence.

A supported joint recipe needs at least three usable observations, at least
50% healthy outcomes and at most 25% constant-class signatures. Evidence is
selected in this order, subject to matching input length and resource limits:

1. Same model, dataset/subset and task.
2. Same model and task, borrowing evidence from other datasets.
3. Same task, borrowing evidence from other models.
4. Explicit uncalibrated broad defaults if none qualifies.

Fallback scope is recorded. In particular, DeBERTa/CoLA has no qualifying local
recipe in this snapshot and borrows task-level evidence. That is a prospective
starting point, not evidence that the borrowed recipe has been validated for
CoLA. Templates are sampled without ranking for highest accuracy.

The source observation digest and complete evidence document hash are frozen.
Rebuilding evidence is an explicit offline operation:

```powershell
.venv/Scripts/python.exe -m mlaas_data_generator.clients.calibration --db outputs/services.db --output mlaas_data_generator/clients/recipe_evidence.json
```

Review the resulting JSON diff, then generate a **new** manifest using the
normal command. New service/policy hashes identify the changed settings.
Historical benchmarks have informed calibration and must not be presented as
untouched final evaluation. Freeze policies before a main experiment and assess
them on reserved data; do not resample finished providers until a desired score
distribution appears. This change itself does not reserve or run a new evaluation.

## Compatibility and remaining defaults

Existing versionless, `conditional_balancing_v1`, `conditional_balancing_v2`, saved `conditional_balancing_v3` and `conditional_balancing_v4`
manifests keep their frozen interpretation and partition behavior. Explicit variant generation and multiround FedAvg also use task/model defaults and explicit time budgets. Shared-parent
cohorts use the same library but do not borrow independent-initialization evidence.

Task/model limits bound batch sizes and effective batches. Wall time uses explicit budgets. The
five-step minimum remains for uncalibrated/exploratory recipes; supported recipes
also enforce their frozen historical update budget. Runtime validation uses the
actual available sample count. Zero successful optimizer updates still fail,
including for existing v2 manifests. `SUCCESS` continues to mean execution
completed; weak services are retained without a quality acceptance gate.

When v5 partitioning discovers less usable support than the frozen maximum
quantity (including shard support), it applies the same proportional quantity
rule. Training derives an effective budget from the actual row count while
retaining the frozen recipe and hash. It deterministically raises epochs to
preserve planned optimizer updates; if the entire subset is smaller than one
accumulation cycle, it reduces accumulation to permit an update. This exception
and both budgets are recorded in `training_budget_adjustment`. The frozen
200-epoch cap and original wall-time limit still apply; insufficient capacity
fails explicitly. Old manifests keep their sample/budget interpretation;
generate a new manifest to use the capacity policy.

MobileNet, EfficientNet, MobileViT and MobileBERT manifests explicitly request FP32 with mixed
precision disabled. The runtime enforces this even for old FP16 recipes or
reused models identified by their model configuration. This temporary stability
policy does not alter the checkpoint storage format. Re-enable mixed precision
only after its forward/backward path has been validated for these families.

HF evaluation that produces nonfinite logits or loss under mixed precision
restarts once in FP32 within the original evaluation time budget. Partial
metrics are discarded; `eval_fp32_retry_count` and `precision_fallback_reason`
record the fallback. Nonfinite FP32 results still fail validation.

On Windows ROCm, DPT models use contiguous NCHW inputs for standard convolution
layers. This avoids the measured slow NHWC MIOpen path at physical batch size
two while preserving the requested batch size, precision and BatchNorm behavior.
The runtime logs `using contiguous DPT convolution inputs` when it installs the
workaround. It applies to newly initialized runtimes, including existing manifest
recipes. For diagnostic comparisons, set `MLAAS_ROCM_DPT_CONTIGUOUS_CONV=0` before
starting the process. See the controlled diagnostic.

The same layout workaround also applies to UPerNet models whose configured
backbone is ConvNeXt. A controlled `openmmlab/upernet-convnext-tiny` GPU test
confirmed slow NHWC kernels in the feature-pyramid convolutions, with ample
free VRAM. Setup logs `using contiguous UPerNet-ConvNeXt convolution inputs`.
Other UPerNet backbones are excluded. Set
`MLAAS_ROCM_UPERNET_CONTIGUOUS_CONV=0` before process startup only when running
diagnostic controls. See the UPerNet measurements.

Text-classification dataset aliases resolve to classification before preparing
the parent model and freezing label contracts. Preparation and execution use
the same task resolver. New text preparations use a new cache namespace;
historical preparations with an `unknown` label task must be regenerated with
fresh service IDs. Sentence-pair completion metrics follow the dataset contract:
MRPC/WNLI classification requires accuracy and F1; STS-B regression requires
Pearson and Spearman. A sentence-pair input alone does not imply regression.

Small randomly initialized generic classifiers keep their 5/10/20 epoch budgets
and 2x quantity multiplier. Generic regression uses 40/80 exploration epochs and its existing regularization ranges. Broad backend/task ranges remain in
`clients/policies.py`; supported templates are layered on them, not hardcoded
as another family-specific model registry.

## Measuring recipe and prediction diversity

New HF manifests write per-family recipe counts and the largest recipe share to
their `.preflight.json`. Quantity and seed differences do not count as different
joint training recipes. The report also shows data-profile, optimizer and band
counts. It is descriptive and does not resample providers.

After a family has trained and saved checkpoints, compare its actual outputs:

```powershell
.venv/Scripts/python.exe -m mlaas_data_generator.cli.main family-diagnostics --db outputs/test_services.db --family-id FULL_FAMILY_ID --output-dir outputs/family_diversity
```

Use the full `family_id` from `service_generation_provenance`. The command reloads
and verifies the frozen family preparation, restores members sequentially and
checks their source model hashes. It predicts only on `catalogue_metric_ids`;
training/audit overlap or changed membership fails before prediction. The audit
subset remains reserved. Historical evaluation exposure remains explicitly
unknown; this diagnostic does not certify an untouched external test set.

Outputs are `summary.json`, numeric `predictions.npz` (load with
`allow_pickle=False`) and CSVs for services, prediction profiles, pairs, per-class
performance and excluded checkpoints. Classification/token reports include
constant outputs, identical prediction vectors, disagreement, shared errors and
per-class recall. Regression reports include prediction MAE/RMSE between clients,
residual correlation and prediction spread relative to target spread. Quality
is reported alongside diversity; disagreement alone is not evidence of useful
composition. No models are trained, filtered by quality or aggregated.

The existing `cohort-diagnostics` command remains for trained-parent cohorts.
These diagnostics cover generation-eligible token families; they do not extend
any composition evaluator that still restricts itself to example-level
classification and scalar regression.

## Retrieval contrastive batches on a 16 GiB baseline

Retrieval recipes use a conservative physical-batch target of 16 pairs for models
up to 250M parameters, 8 for models up to 700M, and 4 above that. Higher image
resolutions reduce the starting target by powers of two using pixel area relative
to 224x224, with an absolute floor of two pairs. This is a starting heuristic,
not a guarantee that every optimizer, precision or architecture fits.

HFCore applies this policy to existing manifests too: CLIP-L/14 at 224px requesting
batch 1 now starts at 8, providing seven negatives per positive. Gradient
accumulation remains separate: batch 8 with accumulation 2 uses a nominal optimizer
batch of 16, but each positive still has seven in-forward negatives. Accumulation
does not create more contrastive candidates. Requested, logical and effective
microbatch sizes and observed minimum/maximum candidate counts are recorded.

Partial batches are rebalanced without dropping or repeating pairs. A 17-pair
partition at capacity 8 becomes 8, 7, 2. Forward/backward OOM recovery reduces the
physical batch, replaying the uncommitted accumulation window, but never below
two. If two pairs do not fit, training fails explicitly. An odd-sized batch cannot
be partitioned into pairs at capacity two; that case also fails explicitly instead
of discarding an example or introducing a duplicate as a false negative.
Optimizer-step OOM remains non-retryable because parameters may already have been
partially updated. A retrieval run with zero successful optimizer updates fails.

Evaluation encodes images/text in memory-safe batches (including singletons),
then computes symmetric contrastive cross-entropy against the complete evaluation
candidate pool using CPU score blocks and the model's learned logit scale. The
pool must contain at least two aligned pairs. The recorded loss protocol is
`pooled_symmetric_contrastive_v1`; candidate count is explicit. Encoding-batch
changes can introduce ordinary floating-point differences, but no longer change
the loss candidate set. Recall already uses the complete pool. Inference-only
evaluation continues to omit loss.

Previously completed singleton-training services are not retroactively repaired;
rerun affected services to obtain trained weights and meaningful loss. See
the CLIP validation report.
