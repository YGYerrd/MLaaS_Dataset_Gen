# MLaaS Service Dataset Generator

Generate comparable MLaaS service records from reviewed manifest rows.

The active workflow is:

```text
registry -> hf-manifest -> review manifest -> run-manifest --dry-run -> run-manifest -> SQLite service records
```

Each manifest row describes one independent service instance. Executing a row trains or loads one model, evaluates it on its benchmark split, records functional attributes and service metrics, then stores one service record in SQLite.

## What To Copy To Another Computer

Copy the project source, not the virtual environment.

Keep:

- `mlaas_data_generator/`
- `requirements.txt`
- `README.md`
- any custom registry, manifest, SQL, or experiment files you need

Usually do not copy:

- `.venv/` or `venv/`
- `__pycache__/`
- `.pytest_cache/`
- old Hugging Face caches

Copy only if you want previous results:

- `outputs/`
- `weights/`
- existing `.db`, `.csv`, and `.xlsx` output files

## Get The Project Onto Linux

### Option 1: Clone From Git

On the Linux computer:

```bash
git clone <your-repository-url> MLaaS-Dataset-Generator
cd MLaaS-Dataset-Generator
```

If the repository is private, set up SSH keys or authenticate with HTTPS first.

### Option 2: Transfer A Zip Or Tarball

From the old machine, create an archive of the project folder. Exclude `.venv`, caches, and large old outputs unless you need them.

On Linux, unpack it:

```bash
tar -xzf MLaaS-Dataset-Generator.tar.gz
cd MLaaS-Dataset-Generator
```

If you transferred a `.zip` file:

```bash
unzip MLaaS-Dataset-Generator.zip
cd MLaaS-Dataset-Generator
```

### Option 3: Copy Over SSH

From the old machine, copy the folder to the Linux computer:

```bash
rsync -av --exclude ".venv" --exclude "__pycache__" --exclude ".pytest_cache" MLaaS-Dataset-Generator/ user@linux-host:~/MLaaS-Dataset-Generator/
```

Then SSH into the Linux computer:

```bash
ssh user@linux-host
cd ~/MLaaS-Dataset-Generator
```

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

If your system has multiple Python versions, use the one you want explicitly:

```bash
python3.11 --version
python3.11 -m venv .venv
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

For the service loop, you can also put the token in a repo-local [`.hf_token`](./.hf_token) file. The manifest runner loads that file automatically before executing services. Environment variables still take precedence, so `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` will override the file when set.

## CLI

Run commands from the repository root:

```bash
python -m mlaas_data_generator.cli.main <command> [options]
```

Commands:

| Command | Purpose |
| --- | --- |
| `hf-manifest` | Build reviewed service rows from the model and dataset registries. |
| `run-manifest` | Validate or execute reviewed service rows. |

Check the installed CLI:

```bash
python -m mlaas_data_generator.cli.main --help
python -m mlaas_data_generator.cli.main hf-manifest --help
python -m mlaas_data_generator.cli.main run-manifest --help
```

## Build A Manifest

The manifest builder reads:

- `mlaas_data_generator/registry/models.py`
- `mlaas_data_generator/registry/datasets.py`

Start small on a new machine:

```bash
mkdir -p outputs

python -m mlaas_data_generator.cli.main hf-manifest \
  --manifest-profile test \
  --resource-tier light \
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

This writes an Excel workbook with a `services` sheet and a `defaults` sheet.

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
| `resource_tier` | Workload budget: `light`, `medium`, `heavy`, or `stress_test`. |
| `training_epochs`, `batch_size`, `learning_rate`, `optimizer` | Training and runtime knobs. |
| `max_samples`, `max_length`, `timeout_s`, `max_train_time_s`, `max_eval_time_s`, `device` | Workload and runtime controls. |
| `input_schema`, `output_schema` | Compatibility metadata for later composition work. |

For first runs on a new computer, reduce risk by keeping `max_samples` low, using `--manifest-profile test`, and setting `enabled=false` for rows you do not want to run yet.

`--resource-tier` controls model, dataset, and knob selection. If omitted, it follows the profile: `test -> light`, `balanced -> medium`, and `benchmark -> heavy`. Use `stress_test` only when you intentionally want the largest allowed services.

For GPU runs, leave `device` blank or set it to `auto` unless you need to force a device. PyTorch exposes ROCm devices through the `torch.cuda` API, so the runner will still resolve a supported AMD ROCm GPU as `cuda`.

On multi-GPU Linux machines, this project uses GPUs most effectively by running multiple manifest rows in parallel, with one worker process pinned to one GPU. Because each manifest row is an independent service, this is more reliable than trying to split one row across multiple GPUs.

For a 2-GPU NVIDIA Linux VM, use:

```bash
python -m mlaas_data_generator.cli.main run-manifest \
  --file outputs/service_manifest.xlsx \
  --sheet services \
  --db outputs/services.db \
  --workers 2 \
  --no-grouped-hf
```

`--workers 2` starts two row-level worker processes. Each worker is pinned to a single visible GPU with `CUDA_VISIBLE_DEVICES`, so two eligible rows can run at the same time across GPU 0 and GPU 1.

Use `--no-grouped-hf` when you want maximum dual-GPU utilization. The grouped HF mode intentionally reuses prepared models and datasets inside one process, which is good for cache reuse but limits row-level GPU parallelism.

CSV manifests can include a row with `service_id=defaults`. XLSX manifests can include a `defaults` sheet.

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

For the 2-GPU Linux VM shown above, prefer:

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
| `outputs/service_manifest_results.csv` | Per-row success/failure summary. |
| `outputs/service_failures.log` | Detailed validation or runtime failures. |

Successful rows are written to the SQLite database configured by `CONFIG["db_path"]`, `MLAAS_DB_PATH`, `MLAAS_SQL_DB_PATH`, or the `--db` override.

## Database Tables

The active schema is service-only:

| Table | Contents |
| --- | --- |
| `services` | One row per manifest service instance. |
| `service_metrics` | Typed quality, QoS, latency, runtime, resource, cost, reliability, explainability, and metadata metrics. |
| `service_artifacts` | Optional model, report, or output artifact references. |
| `service_split_provenance` | Optional split and distribution provenance. |
| `service_failures` | Validation and execution failure details. |

There are no active federated workflow or model-averaging tables.

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
2. Move from `--manifest-profile test` / `--resource-tier light` to `balanced` / `medium` or `benchmark` / `heavy`.
3. Add more `--task-keys`.
4. Increase `--models-per-task` or `--datasets-per-model`.
5. Increase `max_samples` in the manifest or use `--avg-sample-size`.

Example larger manifest:

```bash
python -m mlaas_data_generator.cli.main hf-manifest \
  --manifest-profile balanced \
  --resource-tier medium \
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
python -m pytest mlaas_data_generator/test
```

Focused checks:

```bash
python -m pytest \
  mlaas_data_generator/test/test_service_manifest_pipeline.py \
  mlaas_data_generator/test/test_service_storage.py \
  mlaas_data_generator/test/test_service_runner.py
```

## Troubleshooting

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

## Extending

- Add HF models in `mlaas_data_generator/registry/models.py`.
- Add HF datasets in `mlaas_data_generator/registry/datasets.py`.
- Keep new execution behavior row-local: one manifest row produces one independent service record.
- Add future composition logic in a separate layer that reads the service table; do not couple composition to service generation.
