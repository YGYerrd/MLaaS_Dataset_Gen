# Documentation

Start with the [project README](../README.md) for the main workflow and key files.

| Guide | Contents |
| --- | --- |
| [Setup and usage](usage.md) | Installation, manifest review, GPU workers, retries, queries and troubleshooting |
| [Generation policies](generation-policies.md) | Client recipes, training policies and reproducibility |
| [Shared-parent generation](shared_parent_generation.md) | Optional parent selection, training and cohort workflow |
| [Checkpoint formats](checkpoint_formats.md) | INT8, FP16 and exact storage settings |
| [Checkpoint storage](checkpoint_storage.md) | Integrity, reconstruction, migration and storage contracts |
| [Client timings](client-timing.md) | Runtime measurements and profiling fields |
| [Terminal output](terminal-output.md) | Concise and debug logging |
| [Composition objective v2](composition_objective.v2.md) | Current composition scoring |
| [Composition objective v1](composition_objective.v1.md) | Frozen legacy scoring contract |
| [Composition validation](composition_validation.v1.md) | Held-out validation and evidence |
| [Task extension plan](task-extension-plan.md) | Capability contracts and staged token, segmentation and generation support |
| [Task capabilities](task-capabilities.md) | Capability decisions, structured task protocols, v5 requests and real smoke verification |
| [Perturbation metrics](perturbation_metric_contract.v1.md) | Stability and explainability definitions |
| [Core tests](../tests/README.md) | Retained coverage and test commands |
| [Cleanup and recovery](cleanup.md) | Archived files and recovery instructions |

Research papers remain in `reference/`; slides and presentation sources remain in
[presentation/](presentation/README.md). Dated investigations and benchmark scripts
are stored in the local cleanup archive.
