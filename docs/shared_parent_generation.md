# Shared-parent service generation

This optional workflow separates task-parent training from service variation.
The main generation path is now `hf-manifest -> run-manifest`; all commands and
existing parent artifacts documented here remain supported. To use already
trained parents, start at `cohort-manifest` with the existing `parents.json` file.

```text
historical evidence -> automatic top-N recipes per family -> separate full-task parent checkpoints
    -> frozen independent client recipes -> service catalogue -> composition
```

No parameter aggregation occurs during generation. Downstream composition can
aggregate selected compatible client checkpoints using the existing sample-count
weighting and objective v2. Old FedAvg catalogues and request schemas v1/v2 remain
readable; active FedAvg generation commands are retired.

## Automatically select historical recipes

```powershell
$Run = "outputs/shared_parent"
function Invoke-Generator { python -m mlaas_data_generator.cli.main @args }
Invoke-Generator select-parent-configs `
    --history-db outputs/services.db `
    --output "$Run/parent_selection.json" `
    --limit 5
```

Selection opens the history database read-only in one consistent transaction.
The execution-ready JSON selects up to `--limit` distinct recipes for every eligible
family (default five). Each recipe becomes a separate parent lineage. Families
with fewer eligible recipes select all available recipes without duplication;
`requested_parent_count`, `selected_parent_count`, and `eligible_recipe_count`
record the requested and realised counts. The adjacent CSV contains the complete
candidate evidence and is informational only. Unsupported/incomplete observations have explicit
exclusion reasons. Recipes with identical training/data conditions are ranked
within their evaluation protocol by median task quality, observation count,
dispersion, and recipe hash. Accuracy, token-classification F1, and sentence-similarity
Pearson rank higher values first; regression RMSE ranks lower values first. There
is no minimum quality threshold. Different evaluation protocols are interleaved
in deterministic protocol-hash order; scores are compared only within a protocol.
If a recipe occurs under multiple protocols, its first occurrence is selected and
later occurrences are skipped before applying the limit. All protocol evidence
remains in the CSV. `parent_rank` records selection order, not a score comparison
across incompatible protocols.

The `parent_selection_v2` document contains each family's `selected_parents`,
including recipe/configuration, metric, median quality, observation count,
dispersion, evaluation protocol, historical service IDs, and missing provenance.
It can be passed directly to `train-parents` without copying or editing any fields.
Each `parent_id` hashes the family and recipe hash; execution additionally hashes
the selected configuration and evidence into `parent_checkpoint_id`. Multiple
parents in one family remain separate checkpoint and cohort records. Regenerate
older selection files with `select-parent-configs` to use the new schema. Missing
Hub revisions are resolved and pinned before training; unresolvable revisions
fail that parent. Historical exposure to future audit examples remains unknown
unless independently established; no unseen-data claim follows from selection.

Supported initial families are HF sequence, token, and image classification,
sentence similarity, and the existing generic CNN, MLP, and logistic neural
classification/regression paths. Inference-only and non-restorable estimators
are excluded.

## Train parents

```powershell
Invoke-Generator train-parents --file "$Run/parent_selection.json" --dry-run
Invoke-Generator train-parents `
    --file "$Run/parent_selection.json" `
    --db outputs/cohort_services.db `
    --output "$Run/parents.json"
```

The optional dry run validates all selections and prints `family_count`, total
`parent_count`, and each parent's effective full-pool configuration without training,
downloading data, or creating the database. Training executes all selected parents.
Parent training removes historical sample caps and short time limits, retaining
the selected epoch schedule. This is a new baseline experiment using historical
hyperparameters, not an exact replay of the historical observation.

Before training, preparation freezes model/dataset revisions, preprocessing and
label contracts, training IDs, and evaluation membership. Up to 512 audit examples
are reserved, capped at half the evaluation source. Smaller evaluation splits use
all available rows instead of failing a preferred benchmark size: WNLI's 71 rows
provide 36 catalogue examples and 35 audit examples. Only two evaluation examples
are required to keep both partitions non-empty and disjoint. Ordinary classification
is label-stratified; regression and token examples use deterministic sampling.
The remaining evaluation pool is capped at **1,024 dataset examples** by default,
using all remaining examples when fewer are available. Set `benchmark_sample_size`
in config to change the cap and `evaluation_seed` (default `42`) to change the
deterministic subset. Every service in a family uses that same frozen subset;
the audit holdout and task-specific metric calculations are unchanged. The cap
never depends on task, dataset, model, or training quantity.
New independent families use preparation version `raw_source_family_v4`. Older
frozen preparations remain readable for reconstruction, but executing the new
policy requires a new manifest and service IDs, preserving historical metrics.
Prepared feature/label hashes also detect changed data when clients or compositions
reload the frozen source, even when example IDs and output shapes are unchanged.

Parents are immutable infrastructure records, never services. A weaker measured
parent is reported and remains usable. Incomplete training, non-finite evaluation,
or checkpoint-integrity failures block that parent. Repeating the command verifies
and reuses completed parents. Interrupted parents restart their training schedule;
optimizer and dataloader positions are not resumed.

Large HF image-classification splits use temporary disk storage once their
estimated float32 pixel storage exceeds 64 MiB. Images are processed individually;
only the requested training or evaluation batch is loaded into RAM. Evaluation
and client selections share the same store without copying full pixel arrays.
Smaller splits retain the in-memory path. Split-wide padding, augmentations,
example order and prepared-data hashes have the same semantics on both paths.
Detection, segmentation and multimodal preprocessing retain their existing paths.

The temporary files use the system temporary directory by default. To put them
on a larger drive, create a directory and set `MLAAS_IMAGE_CACHE_DIR` in the same
terminal before launching the generator, for example:

```powershell
New-Item -ItemType Directory -Force D:/mlaas-image-cache
$env:MLAAS_IMAGE_CACHE_DIR = "D:/mlaas-image-cache"
```

Allow space for the expanded training and evaluation pools (about 55.7 GiB for
SVHN at 224 by 224 RGB float32). This is temporary storage, not a reusable cache
between runs. Parent completion, failure and interruption release image files,
model references and backend caches. Completed checkpoint artifacts remain.
Disk write failures fail the parent rather than silently dropping examples.

The two-example structural minimum is checked before preprocessing and again
before model construction, then enforced by the final reservation check. Evaluation
splits below the former 128-example preference retain their available membership.
Composition validation has its own audit minimum; for small frozen reservations,
set `--min-audit-samples` no higher than the available audit count.
Concise logging shows parent position and preparation, hashing, training and
evaluation stages. Long image preparation/hash scans and training/evaluation
loops report progress approximately every ten seconds, at sample/batch boundaries.
A single slow backend operation can still delay the next update.

Restart a running process to use these changes. Repeating the original command
with the same selection and database reuses completed parents; incomplete parents
restart. The storage implementation does not change parent checkpoint identities.

The output records successful parents and failures separately. `reservation_ref`
points to the exact audit source file accepted by `mlaas-compose validate`.

### Continue directly into client training

Add `--train-clients` to train or reuse all selected parents, save the complete
client cohorts of successful parents, and then execute them in the same command:

```powershell
Invoke-Generator train-parents `
    --file "$Run/parent_selection.json" `
    --db outputs/cohort_services.db `
    --output "$Run/parents.json" `
    --client-manifest "$Run/clients.csv" `
    --clients-per-parent 100 `
    --seed 42 `
    --train-clients
```

The manifest is written before any client training begins and remains available
for review. This option starts clients automatically without a review pause. To
review **before** training clients, use the same command without `--train-clients`:
`--client-manifest` alone saves the recipes and stops. Then execute the saved file
with `run-manifest --file "$Run/clients.csv" --db outputs/cohort_services.db`.
Without either option, `train-parents` retains its parent-only behavior.

The chained command accepts the same client quantity, seed, and heterogeneity
options as `cohort-manifest`, plus `--log-mode concise|debug` for client execution.
With `--train-clients` and no `--client-manifest`, the manifest defaults to
`<parents output stem>.clients.csv`; the parent output itself defaults to
`<selection stem>.parents.json`. An explicit client path can use CSV or XLSX.
The corresponding `.coverage.json` records recipe coverage.

Add `--dry-run` to print the parent plans, client count, manifest path, and whether
automatic execution was requested. It trains nothing and writes no outputs;
actual client recipes require completed parents and are created during execution.

Repeating an identical chained command reuses completed parents, preserves an
identical saved manifest, and skips verified completed clients. If existing
recipes differ from the requested recipes, execution stops without overwriting
it. Use a new manifest path for changed settings, or `run-manifest` to resume the
saved roster directly. Keep the parent checkpoints for client reconstruction.

Parent failures do not block clients of successfully completed parents. Once all
parents have been attempted, the chain creates the manifest from completed parents
and executes it when `--train-clients` is set. If no parents complete, no client
manifest is created or executed. Parent failures still produce a nonzero exit code
after the client stage. Repeating the command retries failed parents and adds their
complete client cohorts to the saved manifest when they succeed, preserving existing
recipes and skipping already completed clients. Changed or removed existing cohorts
are still rejected.
Client failures return a nonzero exit code and leave the manifest, results, and
checkpoints available for retry. Results are written beside the manifest as
`<manifest stem>_results.csv`.

## Freeze and execute clients

```powershell
python -m mlaas_data_generator.cli.main cohort-manifest --parents outputs/shared_parent/parents.json --file outputs/shared_parent/clients.csv
python -m mlaas_data_generator.cli.main run-manifest --file outputs/shared_parent/clients.csv --dry-run
python -m mlaas_data_generator.cli.main run-manifest --file outputs/shared_parent/clients.csv --db outputs/cohort_services.db
```

Defaults are 100 clients per parent, seed 42, heterogeneity intensity 0.5, and
average sample quantity 600. Override these with `--clients-per-parent`, `--seed`,
`--heterogeneity-intensity`, and `--average-samples` (`--base-client-quantity` is an
alias). Quantity spread is fixed at 0.75: an average of 1000 gives bounds 250 and
1750. Rounded midpoints of equal-width strata cover this interval and are shuffled
across client IDs independently of training limits and distribution intensity.
The maximum quantity is derived from the average; `--max-client-quantity` has been
removed. New cohorts use `client_recipes_v4` identities; saved manifests retain
their frozen quantities when resumed.

Every CSV/XLSX row contains a hashed authoritative `client_recipe`. The manifest
also binds complete cohort membership. Regenerate the manifest to change recipes;
editing individual training fields invalidates their hashes.

Data profiles (IID, Dirichlet, shard) are balanced independently of sample counts. Each family has one explicit training space without resource tiers.
New training recipes use `conditional_balancing_v4` and the shared
[coarse task/backend policies](generation-policies.md). Compatibility families
share these policies; the selected parent's recipe and quality do not add values
to the training grids.

Optimizer, batch size, epochs, weight decay and supported warmup are balanced
within each family, with separate seeded draws. Learning rate is balanced
within optimizer, and accumulation within feasible batch/resource limits. The
HF text RMSprop range is `1e-5` to `2e-5`; weak SGD recipes remain available.
Model size can reduce resource limits, without changing the coarse rate policy.

Quantity keeps its independent fixed spread. Gradient recipes require at least
five planned optimizer updates. The feasibility check adjusts only infeasible
batch/epoch/accumulation combinations within the policy, preserving the quantity
and other draws. This can alter marginal balance
for tiny quantities. Actual local counts are checked before training, and new
clients with no recorded successful optimizer updates fail as invalid training.
Poor accuracy alone does not exclude a completed service.

Each new manifest has a companion **`<name>.coverage.json`** containing its frozen
policy, per-tier knob counts, optimizer/rate and batch/accumulation counts, and
batch/epoch coverage. Zero counts expose coverage gaps without requiring
calibration for each family. The policy and hash are embedded in each recipe.
New policies create new cohort identities. Saved `conditional_balancing_v1`
policies retain their original grids, parent anchors and replay behavior.

Each client
samples without replacement from the full training pool. Examples can recur across
clients; scarcity within a requested bucket reduces quantity rather than forcing
unrequested buckets into a shard. Requested and realised counts, reductions, and
cross-client reuse are recorded.

New recipes use `independent_source_pool_v2`, and the partition version is part
of the cohort identity. Shard clients retain either the floor or ceiling of
`K ** (1 - intensity)` buckets. Frozen support draws are evenly spaced across
the shard roster and assigned with a separate seed. For five binary shards at
intensity `0.5`, this gives three single-label specialists and two clients with
both labels. Intensity `0` retains all buckets; `1` retains one. Selected buckets,
expected coverage, and realised coverage are recorded separately. Existing v1
manifests still execute their original algorithm and reproduce their partition
hashes. Generate a new manifest to adopt the changed sampling policy; see the
[sharding review](cleanup.md).

Clients execute sequentially within a cohort. Each restores the same parent,
creates fresh training state, and uses the same complete catalogue benchmark.
The stored parameter layout, preprocessing, and output contracts remain fixed.
Training resource limits are deliberate local conditions; zero optimizer steps
or failed evaluation/checkpoint checks are failures. Quality alone never excludes
a technically valid client, and clients with equal resulting weights remain
separate observations.

Rerun the complete manifest to resume. Completed client checkpoints are verified
and skipped; failed or interrupted clients retry from the parent. Cohort status
and counts distinguish complete, partial, failed, and interrupted work. There is
no automatic replacement of failed or weak clients. The usual `--workers` setting
does not introduce concurrent execution inside a shared-parent cohort.

Results are written beside the manifest as `<name>_results.csv`. If any clients
fail, `<name>_failed.csv` retains their complete cohorts so a retry preserves the
frozen roster and skips completed services.
If a published client checkpoint is damaged, verification excludes that client
and continues the cohort. Its frozen weights are not replaced by a new training
run; restoring the original artifact makes it eligible again on the next resume.

## Post-generation behavioural diversity

```powershell
python -m mlaas_data_generator.cli.main cohort-diagnostics --db outputs/cohort_services.db --cohort-id COHORT_ID --output-dir outputs/shared_parent/diagnostics
```

Replace `COHORT_ID` with the `cohort_id` value in the client manifest.
This optional post-generation command reconstructs completed client checkpoints
one at a time and predicts on the **complete frozen catalogue-metric subset**.
`--batch-size` controls inference memory use. Every pair uses the same examples;
the reserved composition audit is not used for diagnostic predictions. The command
does not train, aggregate, rank, filter, or modify service catalogue records.

For ordinary classification it reports:

- **Pairwise disagreement:** fraction of examples assigned different predicted classes.
- **Error overlap:** fraction of all examples both clients predict incorrectly.
- **Error Jaccard:** shared errors divided by examples where either client is wrong.
  It is null when both clients are error-free.
- **Per-class accuracy:** fraction correct among examples whose true label is that
  class (class recall), with minimum, maximum, mean, median, population standard
  deviation, and maximum-minus-minimum spread across clients. Classes absent from
  the benchmark have zero support and null accuracy/spread.

Token classification uses the same metrics over supervised token positions,
excluding the frozen ignore/padding label. It reports both example and token
counts. Multilabel outputs are explicitly unsupported. Regression and sentence
similarity instead report pairwise prediction MAE/RMSE and residual Pearson
correlation; constant residuals produce null correlation. They do not invent class
thresholds or define a categorical error overlap.

Outputs are `summary.json`, `pairs.csv`, `services.csv`, `per_class.csv`,
`per_class_spread.csv`, and `excluded.csv`. The JSON includes definitions, summary
statistics, parent/cohort identities, checkpoint and prediction hashes, exact
evaluation IDs, and a report hash. Pair tables contain each unordered pair once
(4,950 pairs for 100 clients). Inference retains hard decisions rather than full
probability tensors, and comparisons avoid a clients-by-clients-by-examples array.

Unfinished clients and checkpoint/inference failures are listed as exclusions;
statistics describe only the successfully diagnosed clients. Fewer than two
usable clients produces `insufficient_clients` with no pairwise statistics.
Rerun the command after generation resumes to refresh the report. These are
descriptive catalogue diagnostics; they do not establish generalisation to the
audit set or resolve unknown historical evaluation exposure.

## Composition and evidence

```powershell
python -m mlaas_composition generate-requests --db outputs/cohort_services.db --output outputs/shared_parent/requests.json --compatibility-mode strict --seed 42
python -m mlaas_composition evaluate --db outputs/cohort_services.db --requests-file outputs/shared_parent/requests.json --output-dir outputs/shared_parent/composition --search-budget 100
```

New strict requests use schema v3: `cohort_id`, `parent_state_hash`, and
`compatibility_contract_hash`, plus immutable eligible-service membership. Objective
v2 is unchanged. Legacy-only strict catalogues continue to emit schema v2; explicitly
selecting the v2 request profile retains its legacy contract. Large candidate spaces
are searched within the configured budget; exact references report `too_large`.

Selected classification/regression compositions can be validated with the parent's
`reservation_ref` as `--cohort-file`. Frozen membership overrides runtime sampling
preferences. Validation checks the complete parent training universe and the exact
catalogue/audit partition before aggregation. It reports local example exposures
and unique example coverage separately because clients can overlap. Audit
reservation proves isolation from the new generation pipeline, while unknown
historical evaluation exposure retains `mechanical_validation_only` reporting.

Additive tables `parent_checkpoints`, `service_cohorts`, and `cohort_clients` track
generation; service provenance explicitly stores generation mode, family, parent,
and cohort IDs. Existing FedAvg coordination tables remain historical records.

## Verification before research runs

Run the retained core checks from the repository root:

```powershell
python -m pytest -q
```

The extended shared-parent tests and real-backend smoke campaigns are in the
[cleanup archive](cleanup.md). Restore them into a separate checkout when broader
research validation is needed; the core suite does not cover every backend or
shared-parent resume scenario.
