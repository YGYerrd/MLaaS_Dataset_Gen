# Terminal output

Both independent services and FedAvg default to `concise` output. Enable full diagnostics with:

```powershell
mlaas-generate run-manifest --file outputs/fed_manifest.csv --log-mode debug
```

Both CLI entry points accept `--log-mode concise|debug`. You can also set `log_mode`
in a manifest row/defaults or service configuration, or set `$env:MLAAS_LOG_MODE`.
The CLI option overrides the manifest; the manifest overrides the environment/configuration
default. Worker processes receive the resolved setting.

## Once per FedAvg run

The summary prints `fl_run_id`, model, dataset, task, training regime, resource tier,
dataset/split variants, clients, rounds, participation rate, actual train pool and
benchmark sizes, heterogeneity intensity, partition generator version, device,
precision and explainability state. Effective device/precision are used when available.
Unavailable values are labelled `n/a`; they are not guessed.

## Once per client completion

After checkpoint verification, concise mode prints two lines per participating client.
The first shows participant position, client ID, strategy/parameter, sample count,
class/bucket counts, majority share and normalised entropy. The second shows optimizer,
effective batch size, learning rate, epochs, primary/secondary metrics and training time.
Counts follow sorted class/bucket labels (numeric labels sort numerically). Metrics use
three decimal places, majority uses one percentage decimal, and time uses one decimal.
Regression and other tasks retain their own metric names; unavailable values show `n/a`.

Each round starts with its round number and participant count. Unscheduled clients are
listed separately and do not consume participant positions. Each completed round ends
with aggregation confirmation, global primary/secondary metrics and wall-clock duration,
including client execution, aggregation, evaluation and persistence. Manifest counts
still update after each client, with progress output at run boundaries in concise mode
so it does not interrupt or overwrite the two-line blocks. Debug mode retains lifecycle
messages and full metric output; SQLite retains full-precision results in both modes.

## Abnormal conditions

Print requested/effective strategy differences, quota shortfalls, partition fallbacks,
nonzero target allocation deviations, single-class/bucket partitions, training timeouts,
training failures and aggregation failures when they occur. Zero/false anomaly fields
stay quiet in concise mode. Skipped clients remain visible.

## Debug only

Concise mode also reports parent preparation/training/evaluation stages, image
preparation and hashing progress, and periodic training/evaluation batch counts.
These updates are throttled to about ten seconds between completed samples or
batches, with initial/final training and evaluation updates. Fine-grained batch
diagnostics remain debug-only.

Service IDs, full case names, raw provenance JSON, partition hashes, run partition seeds,
bucket specifications, L1 allocation deviation, source fields, checkpoint compression,
per-stage timings, per-batch training/forward-pass logs and perturbation progress are
reserved for debug output. Known diagnostic fields are filtered from result metrics too.
Full provenance and metrics are still stored in SQLite/artifacts in both modes.
Third-party libraries may emit their own warnings or progress messages.

## Example

Illustrative FedAvg excerpt (the shared run summary above the rounds is unchanged):

```text
========== SERVICE RUN SUMMARY ==========
  fl_run_id: run_01
  model: model_a
  dataset: dataset_a
  task: classification
  training regime: finetune_transfer
  resource tier: light
  dataset variant: 0
  split variant: 1
  clients: 4
  rounds: 3
  participation: 0.75
  train pool: 3200
  benchmark size: 200
  heterogeneity intensity: 0.5
  partition generator version: joint_target_allocator_v2
  device: cuda:0
  mixed precision: True
  precision: bf16
  explainability: False

========== ROUND 1/3 ==========
Participants: 3/4
Not participating: client_001

[1/3] client_000 | quantity_skew ×2.69 | n=1501 | classes=753/748 | majority=50.2% | H=1.000
AdamW | bs=8 | lr=2e-05 | epochs=1 | acc=0.721 | F1=0.716 | train=10.1s

[2/3] client_002 | dirichlet α=0.124 | n=557 | classes=412/145 | majority=74.0% | H=0.827
SGD | bs=8 | lr=5e-05 | epochs=1 | acc=0.638 | F1=0.604 | train=2.1s

[3/3] client_003 | shard k=2 | n=557 | classes=480/77 | majority=86.2% | H=0.579
Adam | bs=16 | lr=3e-05 | epochs=2 | acc=0.552 | F1=0.487 | train=3.6s

Aggregation complete
Global model: acc=0.763 | F1=0.757
Round duration: 27.4s
=====================
```

An abnormal partition adds notices inside its completion block, for example:

```text
  Abnormal: quota_shortfall=12
  Abnormal: target_deviation_count=3
  Abnormal: single-class/bucket partition
```

After a final-round client checkpoint is saved, verified and committed, both modes print:

```text
Round 3 Client client_000 weights have been saved.
```

Earlier rounds and unsuccessful saves do not print this confirmation.
