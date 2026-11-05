# MLaaS

Generate machine-learning service catalogues, then search and evaluate compatible
service compositions. Each service trains independently from a frozen family
checkpoint.

The retained catalogues are `outputs/legacy/services.db` (19,406 services) and
`outputs/pooled/services.db` (6,975 services in 70 pools). Their signatures and
pooled replay dependencies live alongside them. See [dataset layout and limits](docs/datasets.md).

## Quick start

Use Python 3.12 and a virtual environment. Install a suitable PyTorch build before
installing the project dependencies. For this Windows ROCm setup:

```powershell
.\scripts\Activate-ROCm-Venv.ps1
```

For a fresh environment, follow the [setup guide](docs/usage.md#python-and-platform-prerequisites),
or use `scripts/setup_windows_rocm_venv.ps1` on Windows ROCm.

From the repository root:

```powershell
python -m mlaas_data_generator.cli.main hf-manifest --task-keys text_classification --models-per-task 1 --datasets-per-model 1 --clients-per-family 10 --seed 42 --output outputs/clients.csv
python -m mlaas_data_generator.cli.main run-manifest --file outputs/clients.csv --dry-run
python -m mlaas_data_generator.cli.main run-manifest --file outputs/clients.csv --db outputs/services.db
python -m mlaas_composition generate-requests --db outputs/services.db --output outputs/requests.json --compatibility-mode strict --profile controlled_collaborative_v4
python -m mlaas_composition evaluate --db outputs/services.db --requests-file outputs/requests.json --output-dir outputs/composition_run --search-budget 100 --seed 42
```

Review the manifest before execution. A compact manifest needs its
`clients.families.json` companion only while it is being expanded for execution.
Completed services persist their resolved configuration, provenance and artifact
references in SQLite, so the original manifest is not required for interpretation,
reproduction or resume.

To execute a CSV produced by `mlaas-compose generate-manifest`, use
`mlaas-compose run-manifest --file outputs/pooled/composition_manifest.csv --source-db outputs/pooled/services.db --db outputs/compositions.db`.
This retrains and evaluates one composition per GPU and stores lightweight service
references and global results separately. See the
[batch executor guide](docs/recipe-composition.md#execute-a-composition-manifest)
for dry-run, resume, CPU execution and source artifact requirements.

## Key files

| File or folder | Purpose |
| --- | --- |
| [cli/main.py](mlaas_data_generator/cli/main.py) | Generator commands |
| [cli/manifest/](mlaas_data_generator/cli/manifest/) | Manifest creation, review and preparation |
| [cli/run_manifest.py](mlaas_data_generator/cli/run_manifest.py) | Execute and resume manifests |
| [services/runner.py](mlaas_data_generator/services/runner.py) | Train, evaluate and save each service |
| [clients/](mlaas_data_generator/clients/) | Client recipes, quantities and partitioning |
| [registry/](mlaas_data_generator/registry/) | Supported models and datasets |
| [config.py](mlaas_data_generator/config.py) | Runtime defaults |
| [storage/schema.sql](mlaas_data_generator/storage/schema.sql) | Catalogue database schema |
| [mlaas_composition/](mlaas_composition/) | Request generation, search and validation |
| [tests/](tests/) | Small core regression suite |
| [docs/](docs/README.md) | Usage, design contracts and research material |
| [scripts/](scripts/) | Environment helpers, pool preparation and reconstruction validation |
| [sql/](sql/) | Reporting queries |
| `outputs/` | Retained legacy and pooled datasets; see [dataset layout](docs/datasets.md) (ignored by Git) |

## Checks

```powershell
python -m pytest -q
python -m mlaas_data_generator.cli.main --help
python -m mlaas_composition --help
```

The core suite covers manifest execution, service records, checkpoint integrity,
composition requests, search and validation. Obsolete diagnostic scripts and their dedicated tests were removed;
see the [cleanup record](docs/cleanup.md).

See the [full usage guide](docs/usage.md) for installation, manifest options, GPU
workers, retries and troubleshooting. The optional
[shared-parent workflow](docs/shared_parent_generation.md) remains supported.
