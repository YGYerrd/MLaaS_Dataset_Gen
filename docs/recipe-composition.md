# Recipe-backed composition: end-to-end validation

The `recipe_composition_v6` request profile searches verified `raw_source_family_v3`
and `raw_source_family_v4` services. It uses request schema `6.0.0` and
`collaborative_objective.v3`. Existing request/objective versions retain their
scoring definitions. No tests or model executions were run while implementing
this workflow; the commands below are the manual validation procedure.

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
