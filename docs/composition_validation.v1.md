# Selected composition validation

Structured task requests use schema v5 and bind task semantics plus evaluation
protocol hashes. See [task capabilities](task-capabilities.md) for named token,
segmentation and decoded summarisation metrics, capability decisions and evidence.
Request readers v1–v4 remain available with their existing semantics.

`mlaas-compose validate` executes exactly one composition that was already
selected by the evaluator or by a named search method. It never reruns search,
changes composition membership, or evaluates the unselected candidate space.
Classification and regression are supported only when every member is a
successful, reconstructable client update from one shared parent. New shared-parent cohorts and historical FedAvg snapshots are supported.

The cohort input is an immutable JSON source contract:

```json
{
  "training_ids": ["train:0", "train:1"],
  "metric_source_ids": ["test:0", "test:1"],
  "metric_source_labels": [0, 1],
  "contract": {
    "dataset_name": "example",
    "dataset_config": null,
    "benchmark_split": "test",
    "heldout_reserved_before_catalogue_metrics": true,
    "heldout_provenance_ref": "outputs/cohort_reservation.json"
  }
}
```

The validator deterministically splits `metric_source_ids` into disjoint
catalogue-metric and audit partitions. Classification uses label-stratified
sampling; regression uses deterministic random sampling. Training IDs must
already be disjoint from that source. Audit IDs and the label-contract hash are
checked before the materialized composition model is executed.

For an unbiased empirical study, the cohort contract must prove that the audit
source was reserved before catalogue metrics were calculated. Without the two
held-out provenance fields shown above, execution is allowed for mechanical
verification but the summary reports `mechanical_validation_only`.

The validator reconstructs each selected client state and the common global
parent. It verifies the run, round, parent, architecture, parameter schema,
label, preprocessing, and dataset contracts, then calls the same
sample-weighted FedAvg aggregator used during generation. Weights are
re-normalized over the selected subset using persisted training-example counts.
Integer and boolean tensors remain inherited from the parent.

The aggregated state is restored into one compatible model and evaluated once.
Classification reports accuracy and macro F1 from that model's probabilities.
Regression reports RMSE and MAE from continuous outputs. Measured utility uses
actual quality and single-model inference latency; one-time aggregation cost is
reported separately.

```bash
python -m mlaas_composition validate \
  --db outputs/services.db \
  --requests-file outputs/requests.json \
  --evaluation-dir outputs/composition_run \
  --request-id req_000001 \
  --selection-source evaluator \
  --cohort-file outputs/cohort_source.json \
  --output-dir outputs/validation_req_000001
```

To validate a composition selected by an existing search result:

```bash
python -m mlaas_composition validate \
  --db outputs/services.db \
  --requests-file outputs/requests.json \
  --evaluation-dir outputs/composition_run \
  --request-id req_000001 \
  --selection-source search \
  --search-method greedy \
  --cohort-file outputs/cohort_source.json \
  --output-dir outputs/validation_req_000001_greedy
```

`validation_candidates.csv` records predicted and measured quality/utility,
task metrics, inference and aggregation timings, state hashes, sample totals,
and contributor weights. `validation_summary.json` presents the comparison
directly. `aggregation_provenance.json` contains the reproducible aggregation
record. Legacy `ensemble_*` database fields remain nullable for compatibility.

Prediction artifacts, checkpoint regeneration, stability campaigns, and
parallel member execution are not accepted by this validation operation. No
exhaustive optimum or cross-algorithm claim is made.

## Shared-parent reservations and schema v3

For a shared-parent cohort, pass the parent document's `reservation_ref` to
`--cohort-file`. Its immutable `reserved_evaluation_v1` document records explicit
training, catalogue-metric, and audit IDs; validation does not resample them.
Strict request schema v3 binds the cohort, parent state, compatibility contract,
and eligible-service membership. No FedAvg run/round fields are required.

Validation checks parent and client checkpoint provenance and the complete
parent training universe. Sample-weighted downstream aggregation retains its
existing semantics, and additionally reports unique training examples and local
example exposures. Unknown historical evaluation exposure keeps the result at
`mechanical_validation_only`, even though the new pipeline's audit reservation
is verified.
