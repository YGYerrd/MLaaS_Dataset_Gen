# Setup and usage guide

For token classification, semantic segmentation and seq2seq summarisation, see
[task capabilities and v5 composition](task-capabilities.md). That guide includes
the capability report, scoped runtime evidence and the opt-in smoke commands.

This project generates controlled families of compatible Machine Learning as a Service records. The main workflow generates heterogeneous recipes and trains each service independently from one frozen initial checkpoint per family. Generation performs no model aggregation.



Generation uses [catalogue-informed recipe policies](generation-policies.md). Independent text/image/token classifiers and confirmed sentence-score regressors sample 70% historically supported joint recipes, 20% challenging recipes and 10% stress recipes where evidence is available, with explicit family/model/task fallbacks. These are recipe allocations, not guaranteed outcome percentages. Other task types and trained-parent clients retain broad defaults. Optimizers, learning rates, data profiles and sample quantities still vary; weak completed models remain in the benchmark. Manifest preflight reports summarize recipe diversity; `family-diagnostics` measures actual prediction diversity after training.



The active family workflow is:



```text

hf-manifest -> run-manifest -> composition within a frozen family

```



`hf-manifest` defaults to 100 training recipes per selected model/dataset/task contract. `run-manifest` pins model and dataset revisions, initializes any required task head once, and saves the complete starting state before training. Every client samples from the full official training split and uses the same frozen catalogue benchmark; separate evaluation examples are reserved for composition audit. Client subsets may overlap.



```powershell

python -m mlaas_data_generator.cli.main hf-manifest --task-keys text_classification --models-per-task 1 --datasets-per-model 1 --clients-per-family 100 --seed 42 --output outputs/clients.csv

python -m mlaas_data_generator.cli.main run-manifest --file outputs/clients.csv --db outputs/services.db

python -m mlaas_composition generate-requests --db outputs/services.db --output outputs/family_requests.json --compatibility-mode strict --profile controlled_collaborative_v4

```



Manifest generation is lightweight. Set `--seed 42` once to control recipe sampling, initialization, evaluation, and the derived per-client sampling/training seeds. Omitting `--seed` generates one fresh global seed, recorded in the shared family definitions for replay. The optional `--initialization-seed` and `--evaluation-seed` flags override that seed only for their respective components. Changing the global seed also changes the prepared family; reuse it when generating clients for the same family. Explicit legacy variant counts override the default 100; combining them with an explicit `--clients-per-family` is rejected. `--total-services` caps the final rows, including partial families. Inference-only rows remain available as controls.



Independent-family generation checks dataset contracts before expanding clients. These checks apply across model architectures and task types: training must use a full official split and evaluation must use a separate source. If evaluation has fewer examples than the preferred benchmark size, preparation uses the available split. Only two evaluation examples are required to keep catalogue and audit partitions non-empty and disjoint; WNLI's 71 validation examples yield 36 catalogue and 35 audit examples. Declared unlabeled evaluation splits are rejected. Incompatible candidates are excluded before the dataset/model limits are applied; `<manifest stem>.preflight.json` records the excluded contracts and reasons. This is an offline registry check, so unknown counts remain unverified until loading. Preparation checks actual evaluation size before preprocessing and again before model construction, and cannot silently create a holdout from the official training pool. Inference controls and existing shared-parent split contracts retain their own behavior. To retain a training-only dataset such as Cats vs Dogs, a separate versioned holdout protocol is required; regenerating currently selects a compatible dataset. Changed prepared recipes require a new manifest and service IDs.



`hf-manifest` now writes a compact review table by default, usually 14 columns: service/family references, enabled status, subset strategy and size, sampling/training seeds, optimizer, learning rate, batch size, epochs, weight decay, gradient accumulation, and warmup. Task-specific settings appear as extra columns when they vary. Shared model, dataset, preprocessing, initialization/evaluation settings, training policy, and snapshotted defaults are stored once per scope in `<manifest stem>.families.json`. Repeated configuration JSON, derived fields, selection metadata, and empty historical FedAvg columns are omitted from the review table. CSV and XLSX use the same companion format; XLSX also contains a reference-only `family_summary` sheet with model/dataset names and seeds.



Keep the review file and its `.families.json` companion together with matching stems. Run the same `run-manifest --file ...` command: it expands the review file into the existing execution schema. You can reorder or remove clients, toggle `enabled`, and edit the displayed recipe values within the recorded policy before preparation. Keep `service_id` and `family_ref` unchanged; regenerate to change shared definitions or create new clients. Review references identify local definitions, not composition families. Missing or inconsistent definitions produce an error instead of falling back to current defaults.



Use `hf-manifest --manifest-format expanded ...` to export the previous full-row format. Existing CSV/XLSX manifests, defaults rows/sheets, parent cohorts, and their resume paths remain supported without conversion. Failed-row retry exports remain self-contained expanded CSVs and retain the frozen defaults and preparation references.



Both `hf-manifest` and `cohort-manifest` use the shared client library in `mlaas_data_generator/clients/`. It owns profile balancing, client quantities, sampling/training seeds, optimizer-specific learning rates, resource limits, and partitioning. New clients use IID, Dirichlet, and shard profiles; `--heterogeneity-intensity` controls distribution skew (default `0.5`, range `0–1`). Both generators use explicit task/model training limits without resource tiers. All profiles receive stratified sample quantities independently of training limits and heterogeneity intensity; Dirichlet and shards change bucket distributions. Sampling never duplicates a row within a client, allows overlap between clients, and records capacity shortfalls.



New shard recipes use `independent_source_pool_v2`: fractional bucket coverage is distributed across the shard roster instead of rounding every client to the same count. With two labels, intensity `0.5`, and five shard clients, three select one label and two select both. Intensity `0` selects all buckets; `1` selects one. This is strict bucket-support sampling, with source proportions within the selected buckets. Small quotas can represent fewer buckets than selected. Saved v1 manifests retain their original sampling behavior; generate a new manifest to use v2. See the [sharding review](cleanup.md) for the design and limits.



Use `--average-samples 1000` (`--avg-sample-size` is also accepted by `hf-manifest`) to centre client counts on 1000. The fixed `sample_quantity_spread = 0.75` gives bounds `round(1000 * 0.25) = 250` and `round(1000 * 1.75) = 1750`. Each cohort takes rounded midpoints of equal-width strata and shuffles them across client IDs using a separate seeded stream. Training limits and distribution intensity do not change these quantities. Without an override, HF manifest profiles supply the average (test: 128, balanced: 768, benchmark: 1600); parent cohorts default to 600. Source scarcity can still reduce realised training counts and is recorded by partitioning.



Shared logic lives in `clients/recipes.py`, `quantities.py`, `training.py`, `resources.py`, and `partitioning.py`. Generator modules supply contracts and lineage, and the runner supplies the prepared source pool. Saved recipes retain their versioned interpretation. New explicit variants use the current distribution profiles and the same independent quantity spread. Historical profile names and allocation semantics remain readable for frozen manifests; they are never generated as new profiles. Existing saved manifests keep their frozen quotas; newly generated parent cohorts receive new recipe hashes and resume identities.



The client library defines recipes, not another training engine. Independent services, parent training, and parent-based clients use the existing `ServiceRunner._train_model` and model backends. Optimizer/runtime reset and CUDA OOM recovery remain shared runner operations; adapter argument negotiation lives in `models/train_eval.py`. Backend fixes therefore apply to both generation workflows. Their preparation, lineage, and resume orchestration stay separate where their contracts differ. An executor's verified checkpoint restore is reused by the runner, while optimizer and gradient state are prepared for every client. Adapter signature checks happen before fitting; errors raised during fitting cannot trigger an accidental second training attempt.



The manifest is disposable execution input. It selects services and supplies their
configuration, but execution does not create or consult a manifest-specific
`.resolved/` binding directory. Completed services persist their resolved requested
and effective configuration, family identity, provenance and artifact references in
SQLite. Reordering rows, renaming the manifest, deleting unrelated rows, or changing
the manifest size therefore has no effect on resume. A completed `(family_id,
service_id)` is skipped; unfinished or absent services run. Family preparation uses
the available evaluation split, with at least two examples for separate catalogue
and audit partitions, and never substitutes a sliced training pool.

New services default to `benchmark_evaluation_v1`: trust uses up to 32 independent
inputs with 4 trials each; explanation uses up to 16 with 4 random controls per
masking amount and replacement. Evidence floors are 16 and 8 respectively.
To adopt these metric settings for unfinished rows of an existing manifest,
resume with `--metric-evaluation-profile benchmark`; a new manifest is not
required for this metric-only override. Completed services stay skipped. See the
[evaluation protocol and resume instructions](metric-evaluation-protocol.md).



Successfully saved, structurally compatible services can compose across manifests sharing the exact initial checkpoint, task interface, dataset, and evaluation contract. Accuracy and diversity do not gate compatibility. Family-bound requests use schema v4 and freeze the eligible pool; regenerate requests after eligible membership changes. Existing catalogues and request schemas remain readable.



The optional [shared-parent workflow](shared_parent_generation.md) remains supported:



```text

select-parent-configs -> train-parents -> cohort-manifest -> run-manifest

```



Existing trained parents can be passed directly to `cohort-manifest --parents <parents.json> --file <clients.csv>`; no selection or parent retraining is required. Parent identities, checkpoint artifacts, cohort diagnostics, and resume behavior are preserved. Its default catalogue remains `outputs/cohort_services.db`.



Supported independent services default to [recipe-only retention](client-reconstruction.md).
When client checkpoints are retained, new manifests default to [INT8 Safetensors](checkpoint_formats.md), with unchanged training/evaluation precision and source model identities. Reload performance is checked on a catalogue sample, with exact-storage fallback if quality drops. FP16 remains selectable; existing manifests retain their saved checkpoint settings.

HF adapters keep floating model weights in FP32; mixed precision controls forward computation through autocast. This prevents FP16-loaded checkpoints from supplying FP16 gradients to GradScaler. A nonfinite mixed-precision training loss receives one FP32 forward retry of the same batch, preserving the scaler and accumulated gradients. `train_fp32_loss_retry_count` records these retries; a loss that remains nonfinite still fails before that batch's optimizer update. FP32 weights can require more memory than a model previously loaded entirely in FP16. Restart an existing worker process to pick up changes to this training path.


For the source map and quick start, see the [project README](../README.md).

## CNN hidden layers

New independent CNN families automatically draw their dense hidden layers uniformly from `[32]`, `[64]`, `[128]`, and `[256]` (25% probability each). The internal pool is `CNN_HIDDEN_LAYER_POOL` in `mlaas_data_generator/config.py`; no architecture argument is required.

```powershell
python -m mlaas_data_generator.cli.main hf-manifest --task-keys keras_image_classification --clients-per-family 100 --seed 42 --output outputs/cnn_clients.csv
python -m mlaas_data_generator.cli.main run-manifest --file outputs/cnn_clients.csv --db outputs/services.db
```

The manifest seed makes the draw reproducible. Changing the seed makes a fresh draw, which may select the same size. Client count and task ordering do not affect the architecture draw. The convolution filters remain 32 and 64; other model types keep their existing architecture.

The draw happens once per CNN model/dataset contract before expanding training clients. Generated manifests store the result as `hidden_layers`, including in shared `.families.json` definitions, so all clients in a family have matching shapes for aggregation. Different sizes produce different service and family identities and must be aggregated separately. Existing manifests retain their recorded architecture; execution and resume never redraw it. The historical `generation_mode="fedavg"` builder path retains its existing defaults.

## Python And Platform Prerequisites



Python 3.12 is the recommended baseline for current Windows ROCm PyTorch environments. Python 3.11 remains fine for Linux and CPU-only setups.



On Ubuntu or Debian:



```bash

sudo apt update

sudo apt install -y git rsync unzip sqlite3 python3 python3-venv python3-dev build-essential

```



Check Python:



```bash

python3 --version

```



## Create And Activate A Virtual Environment



From the repository root:



```bash

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

```



Your shell prompt should now show `(.venv)`.



On Windows PowerShell, the activation command is:



```powershell

.\.venv\Scripts\Activate.ps1

```



For the Windows ROCm environment in this repository, prefer the helper script so the ROCm SDK target-family override is set before imports:



```powershell

.\scripts\Activate-ROCm-Venv.ps1

```



### Windows ROCm (AMD Radeon) Setup



For native Windows ROCm PyTorch on supported AMD GPUs, use Python 3.12 and AMD's ROCm 7.2 wheel set instead of generic `pip install torch`.



This repository includes a bootstrap script that creates a Python 3.12 virtual environment, installs the AMD ROCm SDK and PyTorch wheels, then installs the remaining project dependencies:



```powershell

powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_rocm_venv.ps1

```



If you only need the Hugging Face and PyTorch workflows, and want to avoid installing the repo's optional TensorFlow/Keras path on Windows, use:



```powershell

powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_rocm_venv.ps1 -SkipTensorFlow

```



Native Windows ROCm currently applies to the PyTorch path in this project. TensorFlow ROCm support is still Linux-oriented in AMD's documentation, so generic Keras/TensorFlow model paths on Windows should be treated as CPU-only unless you move those workflows to Linux or WSL.



HF models on Windows ROCm use native PyTorch GPU BatchNorm kernels to avoid the

MIOpen HIPRTC `type_traits` compilation failure seen with LeViT and MobileNet.

BatchNorm parameters and running statistics still train normally, and checkpoint

keys remain compatible. This can affect throughput; other layers retain their

normal backend. Restart the Python process to pick up this workaround. To test a

fixed ROCm installation with MIOpen BatchNorm, set

`$env:MLAAS_ROCM_NATIVE_BATCHNORM = "0"` before starting Python.



## Install Dependencies



For CPU-only use, install the requirements directly:



```bash

python -m pip install -r requirements.txt

```



For an NVIDIA, CUDA, or ROCm Linux machine, install the PyTorch wheel recommended by the official PyTorch selector first:



```bash

# Choose the exact command for your OS, Python version, and GPU from:

# https://pytorch.org/get-started/locally/

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt

```



The `requirements.txt` file uses normal package constraints for `torch`, `torchvision`, and `torchaudio`, so a compatible GPU build installed first should remain installed. On Windows ROCm, use `.\scripts\setup_windows_rocm_venv.ps1` so the AMD ROCm wheels are installed before the shared requirements file.



Verify the key packages:



```bash

python - <<'PY'

import pandas

import torch

import transformers

import datasets



print("pandas", pandas.__version__)

print("torch", torch.__version__)

print("cuda_available", torch.cuda.is_available())

print("transformers", transformers.__version__)

print("datasets", datasets.__version__)

PY

```



## Optional Environment Variables



Set these before running large jobs if you want caches and outputs on a fast disk with enough space:



```bash

export MLAAS_OUTDIR=/mnt/fast/mlaas-outputs

export HF_HOME=/mnt/fast/huggingface

export HF_DATASETS_CACHE=$HF_HOME/datasets

```



If you need private Hugging Face models or datasets, export a token:



```bash

export HF_TOKEN=<your-token>

```



For the service loop, you can also put the token in a repo-local [`.hf_token`](../.hf_token) file. The manifest runner loads that file automatically before executing services. Environment variables still take precedence, so `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` will override the file when set.



## CLI



Run commands from the repository root:



```bash

python -m mlaas_data_generator.cli.main <command> [options]

```



Commands:



| Command | Purpose |

| --- | --- |

| `select-parent-configs` | Automatically select up to `--limit` historical recipes per family (default 5), ready for training. |

| `train-parents` | Train immutable task parents from selected recipes. |

| `cohort-manifest` | Freeze independent heterogeneous client recipes. |

| `cohort-diagnostics` | Report pairwise behavioural diversity and per-class accuracy spread on the frozen catalogue benchmark. |

| `hf-manifest` | Main path: generate heterogeneous independent service recipes, defaulting to 100 per family. |

| `run-manifest` | Validate or execute reviewed service rows. |



Check the installed CLI:



```bash

python -m mlaas_data_generator.cli.main --help

python -m mlaas_data_generator.cli.main hf-manifest --help

python -m mlaas_data_generator.cli.main run-manifest --help

```



## Compose A Completed Catalogue



Composition is a downstream workflow and is owned by `mlaas_composition`, not

the service generator. It accepts either the generator's normalized SQLite

catalogue or the published SQLite export with a `flattened_services` table.

First generate a frozen request file. This is the only step that samples

controlled synthetic requests:



```bash

python -m mlaas_composition generate-requests \

  --db outputs/services.db \

  --output outputs/requests.controlled_collaborative_v2.json \

  --requests 1000 \

  --compatibility-mode functional \

  --seed 20260818

```



Use `--compatibility-mode strict` to generate requests pinned to one

parameter-compatible shared-parent cohort (schema v3), or a legacy FedAvg snapshot (schema v2). Strict requests freeze

the post-constraint eligible service IDs and fail evaluation if catalogue

membership changes. Generation trains clients independently; composition selects subsets of the resulting services.



Then evaluate that immutable file. The evaluation command never generates

requests:



```bash

python -m mlaas_composition evaluate \

  --db outputs/services.db \

  --requests-file outputs/requests.controlled_collaborative_v2.json \

  --output-dir outputs/composition_run_20260818 \

  --search-budget 100 \

  --seed 20260818

```



The evaluation writes catalogue, request, candidate-composition, exact-reference,

and search-result CSV files plus `run_manifest.json`. The manifest records the

seed, command, code revision, dependency versions, input hashes, protocol-asset

hashes, and result-file hashes. The checked-in request profiles, principles,

JSON Schema, and objective definitions live in `mlaas_composition/assets/`; the

current semantics are documented in `docs/composition_objective.v2.md`, while

v1 remains frozen for reproducing older request files.



To execute one already-selected composition on an immutable held-out cohort,

use `validate`. It reconstructs only the selected members' stored checkpoints,

applies the generator's sample-weighted FedAvg parameter aggregation, restores

the resulting state into one model, and records predicted-versus-measured

quality and utility. It does not train models or average member predictions:



```bash

python -m mlaas_composition validate \

  --db outputs/services.db \

  --requests-file outputs/requests.controlled_collaborative_v2.json \

  --evaluation-dir outputs/composition_run_20260818 \

  --request-id req_000001 \

  --selection-source evaluator \

  --cohort-file outputs/cohort_source.json \

  --output-dir outputs/validation_req_000001

```



Use `--selection-source search --search-method <method>` to validate the saved

selection from one existing search method without rerunning or changing search.

See `docs/composition_validation.v1.md` for the cohort contract and output fields.



The perturbation metric contract is documented in

`docs/perturbation_metric_contract.v1.md`: `trust_score` means

perturbation-response stability, while the faithfulness/alignment

`explainability_score` is omitted when a semantic target cannot be formed.



Collaborative requests follow the paper's four-part request structure: functional

requirements, composition-level QoS targets, preference weights, and constraints.

Their objective uses the available proxies for data compatibility, model-update

alignment, compute time, latency, accuracy, reliability, explainability, and cost.

V2 data compatibility prefers exact logical dataset, preprocessing, and

label/output provenance, falls back to legacy schema fields, and reports neutral

missing evidence explicitly. Split strategy and partition quantity are intended

heterogeneity and do not reduce compatibility. `resource_cost_score` is consumed

directly as a higher-is-better efficiency score.

Composition lengths remain deliberately short (2--4 services); sequential workflows

are not part of this implementation.



## Build A Manifest



The manifest builder reads:



- `mlaas_data_generator/registry/models.py`

- `mlaas_data_generator/registry/datasets.py`



Start small on a new machine:



```bash

mkdir -p outputs



python -m mlaas_data_generator.cli.main hf-manifest \

  --manifest-profile test \


  --task-keys text_classification,image_classification,tabular_regression \

  --models-per-task 4 \

  --datasets-per-model 1 \

  --training-regimes finetune_transfer,inference_only \

  --dataset-variants-per-pair 1 \

  --split-variants-per-pair 1 \

  --knob-variants-per-pair 2 \

  --total-services 8 \

  --output outputs/service_manifest.xlsx

```



This writes an Excel workbook with a compact `services` sheet, a reference-only `family_summary` sheet, and a companion `service_manifest.families.json` file. Use `--manifest-format expanded` for the legacy `services`/`defaults` workbook layout.

By default, `hf-manifest` uses one fresh global seed on each run and records it in the manifest. Pass `--seed <int>` to reproduce recipe sampling, initialization, evaluation, and derived per-client seeds with the same generation options.



Useful manifest profiles:



| Profile | Use case |

| --- | --- |

| `test` | Small smoke runs for a new machine. |

| `balanced` | Moderate sample sizes and runtime. |

| `benchmark` | Larger runs for stronger hardware. |



Common task keys include:



| Task key | Typical workload |

| --- | --- |

| `text_classification` | Text sequence classification. |

| `token_classification` | Named entity or token label tasks. |

| `sentence_similarity` | Pair scoring and similarity. |

| `fill_mask` | Masked language modelling. |

| `text_generation` | Causal language modelling. |

| `text2text_generation` | Summarisation and sequence-to-sequence generation. |

| `image_classification` | Image classification. |

| `object_detection` | Object detection. |

| `image_segmentation` | Segmentation. |

| `image_captioning` | Image-to-text generation. |

| `text_image_retrieval` | Image/text retrieval. |

| `visual_question_answering` | VQA. |

| `tabular_regression` | Generic tabular regression service rows. |



### Shared-parent generation



Use the [shared-parent workflow](shared_parent_generation.md) to select historical recipes, train full-task parents, and generate 100 or more compatible client services. Active `--generation-mode fedavg` and `refresh-fedavg-manifest` commands are retired. Legacy FedAvg manifests are rejected with migration guidance; historical catalogues remain readable.



## Review The Manifest



Open `outputs/service_manifest.xlsx` before executing it.



Important columns:



| Column | Purpose |

| --- | --- |

| `enabled` | Set to `false` to skip a row. Missing values default to enabled. |

| `service_id` | Primary service identifier. Missing values are generated deterministically. |

| `case_name` | Human-readable model/dataset/regime label. |

| `dataset`, `dataset_name`, `dataset_config` | Dataset source and provider identifiers. |

| `model_type`, `hf_model_id`, `hf_task` | Runner and model identifiers. |

| `task_type`, `task`, `task_tag`, `modality` | Functional compatibility attributes. |

| `train_split`, `test_split`, `benchmark_split` | Training and benchmark split names. |

| `training_regime` | `finetune_transfer`, `inference_only`, or `generic`. |

| `training_epochs`, `batch_size`, `learning_rate`, `optimizer` | Training and runtime knobs. |

| `sample_size` | Training examples requested for this client; uses the available source if smaller. |
| `benchmark_sample_size` | Global maximum evaluation dataset examples, independent of training quantity. Seq2seq generation, image captioning, and VQA are capped at 25% of this value at runtime, including existing manifests (for example, **2,048** becomes **512**). Smaller explicit service caps are respected. New independent families select a bounded raw evaluation pool before preprocessing, with room for a separate audit holdout of up to 512 rows. |
| `evaluation_seed` | Fixed evaluation sampling seed (default **42**), independent of client/training seeds. Families freeze both this seed and the cap. |
| `max_length`, `timeout_s`, `max_train_time_s`, `max_eval_time_s`, `device` | Sequence length and runtime controls. |

| `input_schema`, `output_schema` | Compatibility metadata for later composition work. |



For first runs on a new computer, set the desired `sample_size`, use `--manifest-profile test`, and set `enabled=false` for rows you do not want to run yet.



`--resource-tier` has been removed. Compatible models and datasets are sampled uniformly without replacement, without size, quality-score or dataset-cost ranking. `--max-models-per-family` remains a constraint on architecture variants; it does not multiply `--models-per-task`. The same explicit `--seed` reproduces the same selection; omit it for a fresh random seed. When `--total-services` truncates an independent manifest, families are shuffled within tasks and selected in randomized task-balanced rounds, preserving whole client families except possibly the final partial family. Tasks with fewer available families leave the rotation when exhausted. Equal probability is conditional on compatibility and configured caps; this is not uniform sampling over all model/dataset pairs globally. Training ranges depend on task, backend, model size, and frozen calibration evidence. Set time budgets explicitly with `--timeout-s` (default 1800), `--max-train-time-s` (120), and `--max-eval-time-s` (120), all in seconds. `--manifest-profile` sets the default average sample count only for independent generation; it does not change model eligibility, training ranges, or time budgets. New recipes use `conditional_balancing_v5` with flat `training_policy.limits`; existing tiered manifests retain their frozen validation rules. Known data capacity scales the requested quantity range before freezing. Smaller realized partitions receive a deterministic, recorded training-budget adjustment within the original time limit; see [generation policies](generation-policies.md).

Every non-dry manifest execution registers its requested service IDs in
`manifest_service_runs` before preflight and dispatch. Ordinary service attempts
finish as `completed`, `failed`, or `interrupted`; intentionally untrained FedAvg
observations retain their existing explicit statuses. Startup recovery closes
abandoned local attempts, including roster members never dispatched. Live,
remote, and ownerless historical attempts are not assumed dead. A hard-killed
process cannot persist its own final record; recovery runs on the next startup.

Failure config JSON is bounded to 128 KiB, omits private runtime fields, and
summarizes arrays without their values. Errors and tracebacks are truncated.
Terminal status is committed before optional diagnostics. If SQLite cannot save
it, a durable `<database>.terminal-status` journal records the status for replay;
a failure of both stores is surfaced. Existing oversized failure records are not
rewritten or compacted by these changes.

### Production recipe-only services

Supported independent services default to `weight_retention=recipe_only`: they
retain their family, reconstruction recipe and observations, with no client
weights. Use `--weight-retention checkpoint` (or the same manifest setting) to
retain checkpoints. Add `--retain-validation-checkpoint` when a recipe-only run
needs retained reference weights for comparison. Other modes keep their
checkpoint policy. The fidelity gate checks functional results; it does not
require bitwise-equal reconstructed weights.
See [client reconstruction](client-reconstruction.md) for supported replay
trajectories, materialisation commands and the Stage F fidelity gate.

### Run without trust and explainability scoring

Add `--no-trust-explainability` to skip both scoring stages:

```powershell
python -m mlaas_data_generator.cli.main run-manifest --file outputs/clients.csv --db outputs/services.db --no-trust-explainability
```

The flag is also supported by `python -m mlaas_data_generator.cli.run_manifest`
and the Python API (`run_manifest(..., no_trust_explainability=True)`). It overrides
manifest settings and `--metric-evaluation-profile`. Training, quality evaluation,
weight retention, update signatures and service IDs keep their normal settings.
Stored results mark metric evaluation as disabled and leave both scores null.
Completed services still skip by default; use `--rerun-existing` to rerun them.
Failed-row retry manifests retain the disabled scoring settings.

### Debug training without weight files

Use `--no-save-weights` to train and evaluate independent clients while skipping
initial/final checkpoint writes, compression, reconstruction files and checkpoint
verification:

```powershell
.venv/Scripts/python.exe -m mlaas_data_generator.cli.main run-manifest --file outputs/clients.csv --db outputs/debug_services.db --no-save-weights
```

The same flag is supported by `python -m mlaas_data_generator.cli.run_manifest`.
For individual manifest rows or Python configurations, set `no_save_weights=true`.
The older `save_weights` field does not control production weight retention;
use `weight_retention` for that policy.

Debug service IDs receive an idempotent `__debug_no_weights` suffix, preserving
normal providers and allowing the same manifest to run normally later. Add
`--rerun-existing` to repeat already completed debug clients. Training/evaluation
metrics, timing, split metadata and completed/failed/interrupted statuses remain
in SQLite. Debug observations explicitly have `composition_compatible=0` and
cannot be restored or used as saved composition providers.

This mode bypasses durable independent-family preparation and checkpoint
verification; grouped HF execution can still reuse models in memory. It is for
generator/training debugging, not a substitute for validating the saved-provider
pipeline or its frozen family benchmark contract. Dataset/model downloads and
caches, logs and diagnostic artifacts may still be written; the mode skips
generated weight checkpoints rather than all filesystem activity. FedAvg and
shared-parent execution reject this mode because those workflows require durable
checkpoints. It does not remove any existing weight files.

Add `--smoke-test` to omit perturbation metrics, update signatures, and the two
weight snapshots used only for those signatures. Training, evaluation, partition
sizes, epochs, and decoding limits are unchanged. This option is also accepted as
`smoke_test=true` in manifest rows. It appends `__smoke_test` to service IDs (after
`__debug_no_weights` when both flags are used), so partial smoke-test observations
do not replace full observations. The omitted metrics are explicitly marked as
disabled in the stored results.

```powershell
.venv/Scripts/python.exe -m mlaas_data_generator.cli.main run-manifest --file outputs/clients.csv --db outputs/smoke_services.db --no-save-weights --smoke-test
```

Manifest executions reuse raw HF dataset splits and generation tokenizer
templates in bounded caches within each process (64 splits and four tokenizers).
The split key includes repository, configuration, revision, and split. Recipes
still apply their own sampling, row limits, preprocessing, and tokenizer settings;
cached objects are copied before use. The caches expire with the run, do not store
model weights, and do not eliminate first-time downloads or tokenization for a
different model.

Independent runs select raw client candidates before preparation, including runs
that retain checkpoints. Grouped execution tokenizes the union of the clients'
selections. Fill-mask Dirichlet and shard profiles default to
`raw_text_length_bucket`, using fixed character-length ranges from the configured
raw text column. Existing explicit `masked_token_id` requests are rejected before
preparation; change those rows to `raw_text_length_bucket`. Other explicit
prepared-feature axes retain their existing preparation path. Dynamic padding
may use a smaller width. See
[selection before tokenization](tokenization-selection.md) for supported paths,
replacement after filtering, and compatibility details.

T5-family models now use FP32 for training and evaluation, including manifests
that previously requested FP16. The policy and effective precision are recorded.
For other models, a nonfinite AMP training loss triggers one FP32 retry and keeps
FP32 enabled for the remaining training and evaluation. A nonfinite loss after
that retry still fails the service. Evaluation rejects nonfinite losses/logits
instead of recording them as successful results with null losses.

Seq2seq quality metrics now use source-only generated predictions after training.
Reference labels are used separately to compute teacher-forced loss. Stored
evaluation metadata distinguishes these protocols; historical teacher-forced
ROUGE must not be compared directly with the new generated-text ROUGE. Genuine
generation can take longer than the old teacher-forced evaluation. These changes
take effect in newly started processes; they do not update an active run or its
previously stored metrics.

### Isolated Windows ROCm vision workaround

Manifest execution automatically applies `MIOPEN_FIND_MODE=FAST` when Windows
ROCm is running on GPU architecture `gfx1201` (including RX 9070 XT) and the
registered model family is `detr`, `conditional-detr`, or `deformable-detr`.
The lookup is `MIOPEN_FAST_FAMILIES` in `cli/miopen_worker.py`. DETR-DC5 is the
observed case on ROCm 7.2; the other ResNet-50 DETR variants are precautionary
family matches. The rule applies regardless of dataset and uses the selected
GPU's architecture. CPU, NVIDIA, Linux, other architectures, and unlisted
families retain their existing execution path. No command-line flag is needed.

```powershell
Remove-Item Env:MIOPEN_FIND_MODE -ErrorAction SilentlyContinue
.venv/Scripts/python.exe -m mlaas_data_generator.cli.main run-manifest --file outputs/all_families_5_clients.csv --db outputs/all_families_test.db --no-save-weights
```

Each selected model group runs in a fresh subprocess with FAST present before
torch/MIOpen loads. The parent environment is never changed, and the subprocess
exits before that worker continues to the next group. Models are still reused
within each group. With `--no-grouped-hf`, isolation is per service instead.
Remove any manually exported global `MIOPEN_FIND_MODE` once, as shown above,
to avoid applying that global override to ordinary services. The automatic
policy does not remove existing user environment settings.
GPU affinity is inherited from the scheduling worker.
The selected services store `miopen_find_mode=FAST` and
`miopen_process_isolated=true` in their service configuration.

This workaround changes convolution algorithm selection, not sample counts,
resolution, epochs, precision, or service IDs. It adds process startup costs and
does not promise a speedup; compare subsequent batch timings. It cannot change
an already running process. Completed services remain skipped unless explicitly
rerun. Interrupted groups retain already committed successful clients.



For GPU runs, leave `device` blank or set it to `auto` unless you need to force a device. PyTorch exposes ROCm devices through the `torch.cuda` API, so the runner will still resolve a supported AMD ROCm GPU as `cuda`.



On multi-GPU Linux machines, this project uses GPUs by pinning worker processes to individual GPUs. With grouped HF execution enabled, different Hugging Face model groups can be scheduled onto different GPUs while still reusing prepared models and datasets inside each group.



For a 2-GPU NVIDIA Linux VM, use:



```bash

python -m mlaas_data_generator.cli.main run-manifest \

  --file outputs/service_manifest.xlsx \

  --sheet services \

  --db outputs/services.db \

  --workers 2

```



`--workers 2` starts two GPU-pinned worker processes. On grouped HF runs, different HF groups can run on different GPUs. On row-local runs, independent rows can run on different GPUs.



When GPU-parallel worker processes are used, each GPU writes to its own SQLite database file to avoid concurrent writes into the same SQLite database. For example, `--db outputs/services.db --workers 2` will produce files such as `outputs/services.gpu0.db` and `outputs/services.gpu1.db`.



Terminal output defaults to concise mode for ordinary services and shared-parent cohorts. Use `--log-mode debug` for full diagnostics. See [terminal output settings and examples](terminal-output.md).



During `run-manifest`, the CLI now shows a live manifest progress footer with completed rows out of total enabled rows plus current worker/GPU assignments. In multi-worker runs, progress is counted from completed manifest rows rather than submitted tasks, so grouped HF workers and parallel GPU processes still report accurate manifest-level completion.



Use `--no-grouped-hf` only when you want the most aggressive row-level parallelism and are willing to trade away grouped model/dataset reuse. Keeping grouped HF enabled is usually the better default when many rows share the same HF model.



Legacy or expanded CSV manifests can include a row with `service_id=defaults`. Legacy or expanded XLSX manifests can include a `defaults` sheet. Compact review manifests instead load their recorded defaults from the `.families.json` companion.



## Validate A Manifest



Dry-run validation does not train models. It normalizes column names, applies defaults, validates enabled rows, resolves missing `service_id` values, and writes `outputs/service_manifest_results.csv`.



```bash

python -m mlaas_data_generator.cli.main run-manifest \

  --file outputs/service_manifest.xlsx \

  --sheet services \

  --dry-run

```



If validation fails, check:



- missing required columns such as `dataset`, `model_type`, or `task_type`

- invalid `training_regime`

- missing `hf_model_id` or `hf_task` for Hugging Face rows

- stale sheet names if you changed `--sheet`



## Run The Program



After the dry run succeeds, execute the enabled service rows:



```bash

python -m mlaas_data_generator.cli.main run-manifest \

  --file outputs/service_manifest.xlsx \

  --sheet services \

  --db outputs/services.db

```



By default, `run-manifest` checks each resolved `service_id` against the target

SQLite database (including its `gpuN` shard files) and skips rows that are

already present. Skipped rows are reported with `status=skipped` and are not

added to the failed-row retry manifest. Use `--rerun-existing` only when you

intentionally want to replace an existing service record, such as when retrying

a service previously stored with `status=failed`.



For the 2-GPU Linux VM shown above, a good default is:



```bash

python -m mlaas_data_generator.cli.main run-manifest \

  --file outputs/service_manifest.xlsx \

  --sheet services \

  --db outputs/services.db \

  --workers 2

```



If you explicitly want to maximize row-level spreading instead of grouped reuse, use:



```bash

python -m mlaas_data_generator.cli.main run-manifest \

  --file outputs/service_manifest.xlsx \

  --sheet services \

  --db outputs/services.db \

  --workers 2 \

  --no-grouped-hf

```



To confirm both GPUs are active while the run is in progress:



```bash

watch -n 1 nvidia-smi

```



Or capture a compact view:



```bash

nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv -l 1

```



The run writes:



| Path | Contents |

| --- | --- |

| `outputs/services.db` | SQLite database containing service records and metrics. |

| `outputs/services.gpu0.db`, `outputs/services.gpu1.db`, ... | Per-GPU SQLite databases created automatically during GPU-parallel runs. |

| `outputs/service_manifest_results.csv` | Per-row success/failure summary. |

| `outputs/service_manifest_failed.csv` | Retry manifest containing only failed row-level runs from the latest manifest execution. |

| `outputs/service_failures.log` | Detailed validation or runtime failures. |

| `outputs/service_artifacts_int8/` | Consolidated checkpoint store, including validated INT8 weights, required exact weights, reconstruction support, and provenance. |
| `outputs/service_artifacts/` | Compatibility junction to the consolidated store for frozen historical references; keep this link. |



Successful rows are written to the SQLite database configured by `CONFIG["db_path"]`, `MLAAS_DB_PATH`, `MLAAS_SQL_DB_PATH`, or the `--db` override.



## Database Tables



The active schema keeps service observations separate from immutable parents and cohort coordination:



| Table | Contents |

| --- | --- |

| `services` | One row per manifest service instance. |

| `service_metrics` | Typed quality, QoS, latency, runtime, resource, cost, reliability, explainability, and metadata metrics. |

| `service_artifacts` | Optional model, report, or output artifact references. |

| `model_lineages` | Initial-model lineage and optional FL run identity. |

| `lineage_checkpoint_storage` | One immutable canonical base-state object and its storage measurements per lineage. |

| `service_generation_provenance` | Pinned model/dataset, schema, preprocessing, labels, state lineage, and runtime provenance. |

| `service_checkpoint_storage` | Per-service exact-patch or full-object decision, independent compression metadata, verification result, and measured storage savings. |

| `service_split_provenance` | Optional split and distribution provenance. |

| `service_failures` | Validation and execution failure details. |

| `parent_checkpoints` | Reviewed selection, frozen parent document, state identity, and training status. |

| `service_cohorts` | Frozen client membership, sample reuse, and cohort status/counts. |

| `cohort_clients` | Per-client recipe, partition, checkpoint, status, and failure evidence. |

| `fedavg_runs` (legacy) | Multi-round run configuration, compatibility contract, shared-data hashes, and final status. |

| `fedavg_global_states` (legacy) | Full immutable `G0...GR` CAS references; these are never services. |

| `fedavg_rounds` (legacy) | Participation, quorum, aggregation contributors, input/output state hashes, and timings. |

| `fedavg_client_rounds` (legacy) | One status/provenance row for each client-round service observation. |



The generator-side checkpoint format and canonical state hash are specified in

[`docs/checkpoint_storage.md`](checkpoint_storage.md).



## Query Results



Use SQLite directly:



```bash

sqlite3 outputs/services.db ".tables"

sqlite3 outputs/services.db "select service_id, status, task_type, training_regime from services limit 10;"

```



Or load results in Python:



```bash

python - <<'PY'

import sqlite3

import pandas as pd



conn = sqlite3.connect("outputs/services.db")

df = pd.read_sql_query("select * from services limit 10", conn)

print(df)

PY

```



## Scaling Up On A More Powerful Machine



After the smoke run works:



1. Increase `--total-services`.

2. Increase `--average-samples`, `--clients-per-family`, and the explicit time budgets as needed.

3. Add more `--task-keys`.

4. Increase `--models-per-task` or `--datasets-per-model`.

5. Increase `sample_size` in the manifest or use `--avg-sample-size` when generating it.



Example larger manifest:



```bash

python -m mlaas_data_generator.cli.main hf-manifest \

  --manifest-profile balanced \


  --task-keys text_classification,token_classification,sentence_similarity,image_classification,object_detection \

  --models-per-task 8 \

  --datasets-per-model 2 \

  --training-regimes finetune_transfer,inference_only \

  --dataset-variants-per-pair 1 \

  --split-variants-per-pair 1 \

  --knob-variants-per-pair 2 \

  --total-services 40 \

  --output outputs/service_manifest_balanced.xlsx

```



Validate it:



```bash

python -m mlaas_data_generator.cli.main run-manifest \

  --file outputs/service_manifest_balanced.xlsx \

  --sheet services \

  --dry-run

```



Run it:



```bash

python -m mlaas_data_generator.cli.main run-manifest \

  --file outputs/service_manifest_balanced.xlsx \

  --sheet services \

  --db outputs/services_balanced.db

```



## Tests



Install requirements first, then run:



```bash

python -m pytest

```



Focused checks:



```bash

python -m pytest \

  tests/generator/test_service_manifest_pipeline.py \

  tests/generator/test_service_storage.py \

  tests/generator/test_service_runner.py

```



## Troubleshooting

If family preparation reports CVE-2025-32434 or says `torch.load` requires
PyTorch 2.6, check the interpreter used by the failing run:

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

Install PyTorch 2.6 or newer with matching torchvision and torchaudio builds
for your platform, then restart any running workers or notebook kernels before
retrying. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
For this repository's Windows ROCm environment, activate it with
`.\scripts\Activate-ROCm-Venv.ps1`; use the setup script above if it needs to be
rebuilt. The loader prefers safetensors, but checkpoints that fall back to
PyTorch `.bin` weights require the patched PyTorch version. Do not bypass the
Transformers security check; see [CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434).



If `torch.cuda.is_available()` is `False`, check the installed PyTorch build first. On NVIDIA, verify the CUDA install you selected. On AMD Windows ROCm, verify that you used `.\scripts\setup_windows_rocm_venv.ps1`, that Python is 3.12, and that the installed wheel version matches AMD's current Windows ROCm support matrix.



If a Hugging Face dataset or model fails to download, check internet access, disk space, `HF_HOME`, `HF_DATASETS_CACHE`, and whether the model or dataset requires `HF_TOKEN`.



If Excel output fails, confirm `openpyxl` is installed in the active virtual environment:



```bash

python -m pip show openpyxl

```



If a run fails partway through, inspect:



```bash

tail -n 80 outputs/service_failures.log

python - <<'PY'

import sqlite3

import pandas as pd



conn = sqlite3.connect("outputs/services.db")

print(pd.read_sql_query("select * from service_failures order by failure_id desc limit 10", conn))

PY

```



To rerun only failed rows after fixing the cause, use:



```bash

python -m mlaas_data_generator.cli.main run-manifest \

  --file outputs/service_manifest_failed.csv \

  --db outputs/services.db \

  --rerun-existing

```



## Extending



- Add HF models in `mlaas_data_generator/registry/models.py`.

- Add HF datasets in `mlaas_data_generator/registry/datasets.py`.

- Keep new execution behavior row-local: one manifest row produces one independent service record.

- Add future composition logic in a separate layer that reads the service table; do not couple composition to service generation.

