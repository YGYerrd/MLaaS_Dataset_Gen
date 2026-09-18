# Core regression tests

Run from the repository root with the project environment active:

```powershell
python -m pytest -q
```

The active suite includes the core checks below, task-contract/runtime tests and
restored token, generation-preprocessor and image-adapter fixtures:

| Area | Test modules |
| --- | --- |
| Manifest creation, validation and resume | `generator/test_service_manifest_pipeline.py` |
| Historical recipe calibration, diversity and replay | `generator/test_recipe_calibration.py` |
| Distribution profiles, independent sample counts and frozen roster replay | `generator/test_distribution_profiles.py` |
| Prediction diversity and frozen family diagnostics | `generator/test_behavioural_diagnostics.py` |
| Service execution and metrics | `generator/test_service_runner.py` |
| Versioned evaluation, task outputs and evidence | `generator/test_production_evaluation.py`, `generator/test_evaluation_outputs.py`, `generator/test_evaluation_integration.py` |
| SQLite records and artifacts | `generator/test_service_storage.py` |
| Exact checkpoint integrity and reconstruction | `generator/test_checkpoint_store.py` |
| Checkpoint quality verification | `generator/test_checkpoint_verification.py` |
| Request protocol and CLI evaluation | `composition/test_protocol.py` |
| Composition search | `composition/test_search.py` |
| Selected-composition validation | `composition/test_validation.py` |
| Update-signature integration | `integration/test_update_signature_compositions.py` |

`generator/conftest.py` isolates manifest result, retry and failure exports from
local research runs. Tests use temporary files and synthetic/mocked inputs.

The larger backend, policy, legacy and checkpoint-format suites, helper fixtures
and manual smoke campaigns are in the [cleanup archive](../docs/cleanup.md).
Restore them when making changes that need that broader coverage.

The opt-in real backend harness is `python -m tests.smoke.run_task_extensions`.
It is separate from pytest and needs cached model/data files. See
[task capabilities](../docs/task-capabilities.md) for commands, scope and evidence.
