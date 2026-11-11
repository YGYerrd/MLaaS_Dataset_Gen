# Small-scale local round experiments

Use `generate-manifest --rounds N` to generate a round experiment with composition
sizes **2, 3 and 5**. Each composition is a separate federated trajectory. Its
clients keep their recorded local datasets and training budgets, but train from
the preceding round's global model. Round 1 starts from the verified family
initial model. Only classification families with measured accuracy are supported.

## Generate, execute and export

```powershell
python -m mlaas_composition generate-manifest `
  --prepared-dir outputs/pooled `
  --compositions 30 `
  --rounds 5 `
  --timing-db outputs/compositions.db `
  --deadline-profile moderate `
  --availability-probability 1.0 `
  --seed 42 `
  --output outputs/pooled/round_manifest.csv

python -m mlaas_composition run-manifest `
  --file outputs/pooled/round_manifest.csv `
  --source-db outputs/pooled/services.db `
  --db outputs/round_compositions.db `
  --dry-run

python -m mlaas_composition run-manifest `
  --file outputs/pooled/round_manifest.csv `
  --source-db outputs/pooled/services.db `
  --db outputs/round_compositions.db

python -m mlaas_composition export-rounds `
  --db outputs/round_compositions.db `
  --output outputs/composition_rounds.csv
```

Change `--rounds 5` to the required count. Execution uses one complete trajectory
per GPU worker; `--device cpu` is also supported. Use a separate results database
for round studies. `export-rounds` is the round-history export; `export-flaas`
retains its independent-composition contract and rejects a database containing
round studies.

`--compositions` retains the existing approximate, balanced family/size quota.
It produces at least one composition per eligible family/size pair, so requesting
30 can produce more than 30. For a small pilot, add `--family-id FAMILY_ID` and
`--compositions 3`: this produces one composition of each size if the selected
family has at least five services and timing evidence for all three sizes.
Family IDs are in `outputs/pooled/pools.csv`; `--family-id` can be repeated.

The CSV keeps its four membership columns and adds `rounds` and a JSON
`round_config`. Configuration contains the frozen baseline, deadline profile,
deadline in seconds, and one availability map per round. The companion metadata
records the settings, fingerprints and any skipped family/size pairs.
Completion times and accuracies are populated only by actual execution.
Commands without `--rounds` retain the existing one-shot behavior and size set.

## Timing baseline and deadlines

No network setup or bandwidth simulation is required. Clients execute locally
in sorted service-ID order. Their state is passed to the aggregator in-process;
there is no separately measured upload/download stage. Timings describe this
local sequential execution, including preparation, state capture, aggregation
and global evaluation. They do not estimate distributed parallel round duration.

The baseline for each **family and composition size** is the median of
`total_wall_s - client_loss_measurement_wall_s` from completed independent
executions in `--timing-db`. Missing optional loss timing is treated as no
recorded loss overhead. The new round executor omits the optional per-client
loss passes. Failed/incomplete executions and previous round studies are excluded.
Pairs without matching timing evidence are skipped and listed in metadata.
The baseline retains its observation count, range and source-observation hash.

| Deadline profile | Multiplier of baseline |
| --- | ---: |
| tight | 0.75 |
| moderate | 1.00 |
| relaxed | 1.50 |

These are experiment settings, not evidence that a particular deadline is
optimal. Hardware and workload differences can change actual durations. The
baseline is frozen before execution and never recalculated from later results.
Generate three manifests with the same count, seed and availability probability
and different profiles to compare identical memberships and availability
schedules under the three deadlines.

The deadline is the round's time budget and update-acceptance cutoff. New client
training is not started after the cutoff, and updates arriving late are excluded.
Training already running is **not preempted**. Aggregation and audit evaluation
finish so a measured global accuracy is available. Consequently, actual round
duration can exceed the deadline; `deadline_violation` reports that overrun. This
is not a hard real-time execution limit. A completed client may still belong to
a round that overruns during aggregation or evaluation.

Sample-weighted FedAvg uses only on-time available clients. One contributor is
allowed. If none contribute, the previous global state is retained and evaluated,
with `round_status=no_update`; the accuracy is still measured on the fixed audit
partition. No training data enters that partition.

## Round attributes

The export has one row for **every composition client in every completed round**,
including unavailable, late and deadline-skipped clients. Its key is
`(run_id, combination_id, communication_round, client_id)`.

| Attribute | Meaning |
| --- | --- |
| `communication_round` | One-based round index, reset for each composition trajectory. |
| `client_availability` | Boolean from that client's predeclared availability schedule; independent of training success or deadline eligibility. |
| `round_deadline_s` | Frozen time budget for the round, in seconds. |
| `round_completion_time_s` | Measured wall time from round start through training, aggregation and audit evaluation, in seconds. Excludes initial trajectory preflight and publication of the completed round result. |
| `global_accuracy_per_round` | Measured global model accuracy after this round, as a fraction in [0,1]. |

Availability defaults to all clients available. `--availability-probability 0.8`
generates independent seeded Bernoulli availability per client and round. The
schedule is an experiment input, not observed hardware uptime. No minimum number
of available clients is forced. Exported `client_status` and `update_accepted`
distinguish availability from whether training ran and its update contributed.

Supporting columns include deadline profile/violation, round status, aggregation
weight, client training wall time, parent/global state hashes, baseline seconds
and sample count, and explicit timing/communication/availability descriptions.
Round-level measurements repeat across client rows; reduce to one row per
`(run_id, combination_id, communication_round)` before averaging round accuracy
or duration.

Results are stored in `composition_rounds`, linked to the existing composition
tables. `result_json` holds per-client availability, timings and update evidence.
`composition_members` reflects each client's latest completed round.
Completed rounds are committed as they finish, including when a later round
fails. The export includes those earlier rounds and reports their parent
composition's status. Repeating an unchanged completed manifest skips completed
trajectories. Failed or interrupted trajectories restart from round 1 and replace
their partial history, since intermediate global checkpoints are not persisted.

## Verification

```powershell
python -m pytest tests/composition/test_rounds.py tests/generator/test_round_initial_state.py -q
```
