# Recipe-backed composition: end-to-end validation

The `recipe_composition_v6` request profile searches verified `raw_source_family_v3`
and `raw_source_family_v4` services. It uses request schema `6.0.0` and
`collaborative_objective.v3`. Existing request/objective versions retain their
scoring definitions. The commands below describe manual validation of this
workflow and batch execution of generated composition manifests.

## Prepare a family

Generate trained independent services with `weight_retention=recipe_only` using
the normal manifest executor. Leave `retain_validation_checkpoint=false` to
exercise genuinely checkpoint-free reconstruction. Keep the database, raw data,
pinned model/dataset sources, family artifacts, shared initial checkpoint and
client recipe artifacts available at their recorded locations.

Use at least two completed services from one family. Six or more, with varied
training counts, distributions and training settings, gives search meaningful
subset choices. Members must share the same family/initial state, architecture,
parameter schema, preprocessing, label semantics and evaluation protocol.
Different architectures belong to different families. Keep the family fixed
after generating requests: membership changes invalidate frozen requests.

Classification and regression use the existing adapters. Token classification,
segmentation and summarisation retain their declared experimental capability
status. Masked LM and other tasks without a composition evaluation protocol are
excluded even when recipe reconstruction is supported.

The audit reservation must already exist in the family; by default validation
requires at least 64 audit examples. Validation uses its exact saved rows and
does not create a new split. A bounded catalogue evaluation may leave additional
source rows unused; those remain reserved from training too.

## Generate a random composition manifest

Use a prepared pool directory containing `pools.csv`, `services.csv`, and
`services.db` to select compositions of **2, 3, 5, 7, and 10 services**:

```powershell
mlaas-compose generate-manifest `
  --prepared-dir outputs/pooled `
  --compositions 1000 `
  --seed 42 `
  --output outputs/pooled/composition_manifest.csv
```

From a source checkout, replace `mlaas-compose` with
`.\.venv\Scripts\python.exe -m mlaas_composition`.
`--compositions` is a required positive approximate total; the seed defaults to
42. Every eligible family/size pair receives the same quota: the requested total
divided by the pair count, rounded to the nearest integer (halves round up), with
a minimum of one. The current 70 pools give 350 pairs, so requesting 1,000 yields
1,050 compositions, and requesting fewer than 350 still yields at least 350.
For a different prepared dataset, pairs lacking enough eligible services are
skipped. Quotas are capped at the number of unique combinations and sampling
attempts are bounded; any shortfall is reported without redistribution.

Only verified pools and verified, completed services are eligible. Each
composition contains distinct services from one family. Services may appear in
multiple compositions, but duplicate combinations are excluded. Sampling uses a
small, fixed preference for shorter recorded training times. Sizes are determined
from actual membership, including 3 and 7 even if the preparation report lists
older supported sizes.

The CSV contains `combination_id`, `family_id`, `composition_size`, and
`service_ids` (a sorted JSON array). Combination IDs hash the family and sorted
membership, so the same selection retains its ID across seeds and runs.
`composition_manifest.metadata.json` records the generator version, seed,
requested/actual counts, sizes, eligible pair count, quota, normalized input
fingerprint, skipped pairs, and shortfalls. Identical normalized inputs, count,
seed, and generator version produce byte-identical CSV and metadata files,
independent of source row order or directory location. The fingerprint covers
membership and training times, rather than every byte of the database.

Both outputs are protected against replacement unless `--overwrite` is supplied;
prepared input files cannot be replaced by this command. The database is opened
read-only. Generation performs no training, scoring, artifact replay, or
aggregation. This selection manifest does not certify aggregation success and
is not an input to the existing `evaluate` or `validate` commands.

## Execute a composition manifest

Run the generated CSV directly with one complete composition per GPU worker:

```powershell
python -m mlaas_composition run-manifest `
  --file outputs/pooled/composition_manifest.csv `
  --source-db outputs/pooled/services.db `
  --db outputs/compositions.db `
  --dry-run

# Remove --dry-run after reviewing the metadata validation results.
```

`--dry-run` verifies recipes, family contracts, audit reservations and checkpoint
references without reading tensor payloads, training, or creating results files.
Tensor integrity and raw-data reconstruction are verified during execution.
The companion manifest metadata JSON is optional; the CSV and source catalogue
provide execution inputs.

Automatic execution uses every visible GPU, with at most one process per GPU.
`--workers 2` caps this at two GPUs; `--device cpu` runs sequentially on the CPU.
Inherited GPU visibility masks are respected. Automatic mode fails explicitly
when no GPU is available. Members train sequentially in service-ID order, using
their saved recipes and the existing trainer. Every composition retrains all its
members, including services with retained checkpoints and services already used
in another composition. Training follows the recorded completed budget rather
than the original wall-clock cutoff, as in normal recipe reconstruction.

The existing sample-weighted parameter aggregator produces one global model.
Only that model is evaluated on the family's exact reserved audit set (minimum
64 examples), using the existing task metrics and quality normalisation. Member
quality values refer to original catalogue measurements. New member/global
weights are discarded; the shared initial checkpoint remains a required input.
`training_wall_s` includes member preparation, replay and state capture, excluding
the separate client-loss measurement passes described below.

Each newly executed member also evaluates its exact selected **training rows**
once before local training and once after it. These passes use evaluation mode,
without optimizer updates, and preserve model buffers, training modes and RNG
state. HF forward passes use native weight precision without autocast, with
float32 losses and float64 accumulation. Evaluation can reduce its microbatch
size after an OOM without changing the training recipe. No catalogue or audit
rows enter these measurements, and no client weights are saved.

The three attributes are stored at the top level of
`composition_members.evidence_json`:

- `local_loss_current_model`: mean per-example loss of the shared **initial**
  model on this client's training rows, before its local update.
- `squared_sample_loss_sum`: sum of squared per-example losses of the **final
  local** model on those same rows.
- `local_loss_l2_squared`: the same sum, explicitly defined as the squared L2
  norm of that per-example loss vector, not the square of its mean.

These are fixed-model evaluation losses, not losses accumulated along the
training trajectory or the loss of the final aggregated model. Classification
uses cross-entropy (multilabel: mean binary cross-entropy over labels);
regression uses per-example mean squared error. Token/sequence tasks and
segmentation first average over each example's supervised tokens/pixels,
excluding ignored targets, and then square that example's mean. Segmentation
uses the existing labels-at-logits-resolution convention; seq2seq uses teacher
forcing. Keras uses the model's unregularized compile loss without batch
reduction. Empty supervision, non-finite losses and unsupported models fail
explicitly rather than recording a substitute value.

`client_loss_statistics` embeds version `client_loss_statistics_v1`, the ordered
training-row hash, initial/final model hashes, loss definition, sample count,
sum, mean, squared sum, precision, batching and timing for both passes. The
separate `client_loss_measurement_wall_s` appears in member evidence and as a
composition metric summed across members. `total_wall_s` includes this overhead;
`training.train_time_s` remains the trainer's original measurement.

For example, extract the attributes without retraining:

```sql
SELECT run_id, combination_id, service_id,
       json_extract(evidence_json, '$.local_loss_current_model') AS local_loss_current_model,
       json_extract(evidence_json, '$.squared_sample_loss_sum') AS squared_sample_loss_sum,
       json_extract(evidence_json, '$.local_loss_l2_squared') AS local_loss_l2_squared
FROM composition_members
WHERE status = 'completed';
```

Results use a separate SQLite database. Its five tables are
`composition_runs`, `source_services`, `compositions`, `composition_members`,
and `composition_metrics`. Service snapshots contain identifiers, selected
catalogue metric references and artifact hashes/locations. Recipes, model
artifacts and datasets remain in the source store. The coordinator alone writes
results, and a lock prevents concurrent executors using the same destination.

Repeating the command resumes a matching manifest/source-evidence/executor
fingerprint. Completed compositions are skipped; failed or interrupted ones
restart from their first member. Changed inputs create a new run in the same
database. Use a new destination to deliberately repeat identical inputs.
A failed member prevents aggregation of that composition; other compositions
continue, and the command exits nonzero if any failed. Completion and metrics
are committed together. Worker crashes are recorded and replacement processes
handle subsequent jobs.

Client-loss telemetry is separately versioned and does not change the existing
training/resume fingerprint. Completed compositions from older executors remain
skipped and retain their original results; their absent loss attributes stay
NULL. The executor reports `completed_without_client_losses` and prints a notice
when skipping them. To measure their final-local-model losses, explicitly replay
the desired combinations into a separate results database. Existing running
worker processes need a restart to load the new measurement code; use the same
manifest, source and destination to resume remaining compositions.

`run-manifest` automatically reads `path_map.json` beside `--source-db`, when
present. Use `--path-map path/to/map.json` to select a different map. The
`composition_path_map_v1` format contains a `prefixes` list of `recorded` and
`resolved` directory paths; relative destinations are resolved against the map's
directory. The longest matching directory prefix wins. This lets the imported
catalogue at `outputs/pooled` retain its Linux references while reading local
recipes, family documents, reservations, support artifacts and checkpoints.
The same mappings apply during preflight and CPU/GPU replay. Original document
contents, catalogue records and integrity checks remain unchanged. The map's
HF cache fields are preparation metadata; configure Hugging Face caches through
the usual runtime environment separately. A verified preparation roster does
not establish that local replay will succeed; use dry-run before a long batch.

Historical Hugging Face architecture hashes include the original snapshot cache
path. During member replay and global evaluation, the source path in the service's
identity provenance is used after verifying its full hash against the frozen family.
Older records without that path fall back to the verified recipe's model source.
This handles image recipes that recorded only a repository ID. The pinned model
still loads from the local cache. The complete original architecture hash must still match; model
settings, tensor schemas and data checks remain enforced. Family mismatch errors
name the differing fields, such as `architecture_id` or `evaluation_data_hash`.

## Run from the repository root in PowerShell

Set the actual database path and a new experiment directory. These commands use
the repository's existing virtual environment.

```powershell
$pythonExe = '.\.venv\Scripts\python.exe'
$db = 'outputs\your_family\services.db'
$run = 'outputs\recipe_composition_01'

& $pythonExe -m mlaas_composition list-families --db $db --output "$run\family_services.csv"
```

Copy the desired `family_id` from that output. `eligible_services` must be at
least two; `exclusion_reasons` explains records that cannot be used. Existing
raw-source records with the old `composition_compatible=0` marker are checked
against their actual family and recipe evidence, so they need no database
backfill. New publications advertise eligibility, which composition still
verifies independently.

```powershell
$familyId = 'family_REPLACE_WITH_THE_REPORTED_HASH'

& $pythonExe -m mlaas_composition generate-requests `
  --db $db --family-id $familyId `
  --profile recipe_composition_v6 --compatibility-mode strict `
  --requests 1 --seed 42 --output "$run\requests.json"

& $pythonExe -m mlaas_composition evaluate `
  --db $db --requests-file "$run\requests.json" `
  --output-dir "$run\search" --search-budget 100 `
  --reference-max-candidates 10000 --seed 42
```

`evaluate` runs MAB, GA, greedy and random search using only catalogue evidence.
It does not replay training or evaluate audit models. `search_results.csv` holds
each method's selected IDs, predicted quality, predicted utility, per-example
latency estimate, budget, seed and objective version. `references.csv` contains
an exhaustive *surrogate* reference only when its candidate space is below the
specified limit; it is not a measured global optimum.

Reconstruct and evaluate all four frozen selections:

```powershell
& $pythonExe -m mlaas_composition validate-search `
  --db $db --requests-file "$run\requests.json" `
  --evaluation-dir "$run\search" --output-dir "$run\validation" `
  --device cpu

$LASTEXITCODE
```

Use `--device cuda` for GPU reconstruction/evaluation, preferably matching the
generation hardware and precision when comparing timings. Replay entails
training again and can take as long as the selected clients' original training.
Members are materialised sequentially; canonical tensor states for the selected
subset are retained in host memory for the existing aggregator. Identical
selections within a request reuse one execution. Different subsets may replay
the same member again. No persistent reconstructed-client cache is created.

To validate just one selected method instead:

```powershell
& $pythonExe -m mlaas_composition validate `
  --db $db --requests-file "$run\requests.json" `
  --evaluation-dir "$run\search" --request-id req_000000 `
  --selection-source search --search-method greedy `
  --output-dir "$run\greedy_validation" --device cpu
```

No `--cohort-file` is necessary for these raw-source families: it is resolved
from their verified reservation. Historical checkpoint compositions still need
an explicit cohort file. `--audit-samples` does not resize frozen reservations;
`--min-audit-samples` is an evidence floor, not a sampling instruction.

For a larger comparison, use a fresh experiment directory, increase `--requests`
and repeat search and `validate-search`. The latter accepts `--request-id` to
restrict execution. It never changes membership or runs search again.

## Check the results

An exit code of zero means all requested algorithm selections completed.
Nonzero indicates at least one failed/unsupported selection; inspect its saved
reason and stage. Partial batch rows are saved as each selection finishes.

```powershell
Import-Csv "$run\validation\validation_comparison.csv" |
  Select-Object request_id, search_method, validation_status, `
    predicted_quality_score, measured_quality_score, quality_prediction_error, `
    predicted_utility_score, measured_utility, utility_prediction_error, `
    reconstruction_wall_s, aggregation_wall_s, inference_latency_s_per_example, execution_reused

Get-Content "$run\validation\validation_summary.json"
```

The batch produces:

- `validation_comparison.csv`: one row per request/method, including failures.
- `algorithm_summary.csv`: completed/failed counts and quality/utility signed
  mean errors and mean absolute errors by method.
- `validation_summary.json`: overall completion counts and comparison scope.
- `request_000000/<method>/`: per-selection metrics, summary and
  `aggregation_provenance.json`. Directory indices follow request-file order.

Confirm every requested row has `validation_status=completed`. In each
aggregation provenance record, check the family/recipe identities, contributor
sample counts and normalised weights, original and materialised state hashes,
and audit reservation hash. For recipe-only contributors the materialisation
method must be `recipe_replay`, with `client_checkpoint_reads=0` and successful
selected-row/effective-settings verification. `pipeline_audit_reservation_verified`
confirms the evaluated reservation matched the family and selected training rows.

The aggregator averages compatible floating/complex parameters using realised
training-example counts, retaining integer/boolean tensors from the shared
initial state. Overlapping client data contributes once per client exposure;
unique example counts and total exposures are reported separately.

## Interpret prediction errors

Predicted quality is the sample-weighted mean of the original catalogue
`metric_score` values. This is an uncalibrated heuristic, not a learned predictor
or a guarantee that averaging parameters preserves individual-model quality.
Measured quality uses the same named primary metric and generator normalisation:
accuracy for classification, transformed RMSE for ordinary regression,
transformed Pearson for sentence similarity, and the declared structured-task
metric where supported.

Both predicted and measured utility use objective v3. Measured utility replaces
quality and latency with audit measurements; compatibility, reliability, trust,
explainability and resource dimensions still use catalogue evidence. It is not
a fresh measurement of every aggregate property. Timing estimates are converted
from recorded task totals or HF batch means to seconds per example, but hardware,
batching and execution overhead can still affect their accuracy.

Errors are **measured minus predicted**. A completed workflow can have a large
prediction error or an aggregate worse than its members. Those are experimental
results, not reconstruction failures. Recipe replay permits numerically different
weights; this workflow records replay checks but does not certify fidelity to
the original model. Use the separate reconstruction comparison tooling when a
retained reference or matching retained twin is available.

The existing conservative `mechanical_validation_only` label remains when
historical/pretraining evaluation exposure is unknown. It does not mean that
the aggregate was simulated: successful rows contain real model evaluation on
the family-reserved audit set. Audit measurements are not fed back into search.

Search snapshots hash requests, predictions and artifact evidence. If family
membership or selected evidence changes, generate a new experiment rather than
editing old output CSVs. Validation appends its existing result tables in SQLite;
original service metrics and recipes are not overwritten.
