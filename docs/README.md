# Documentation

Start with the [project README](../README.md) for the main workflow and key files.

| Guide | Contents |
| --- | --- |
| [Setup and usage](usage.md) | Installation, manifest review, GPU workers, retries, queries and troubleshooting |
| [Generation policies](generation-policies.md) | Client recipes, training policies and reproducibility |
| [Client reconstruction](client-reconstruction.md) | Raw-source durable families, authoritative recipes and checkpoint comparison |
| [Recipe-backed composition](recipe-composition.md) | Schema v6 search, recipe replay, parameter aggregation and end-to-end audit validation |
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
| [Evaluation protocol](metric-evaluation-protocol.md) | Versioned trust, independent faithfulness, evidence gates and reconstruction pilots |
| [Benchmark preset pilot](benchmark-evaluation-pilot.md) | Default 32/16 input budgets, independent evidence floors and the same 12 reconstructed profiles |
| [Evaluation pilot results](metric-evaluation-pilot.md) | Measured coverage, runtime limits and full-budget synthetic controls |
| [Legacy perturbation metrics](perturbation_metric_contract.v1.md) | Historical stability and explainability definitions |
| [Core tests](../tests/README.md) | Retained coverage and test commands |
| [Cleanup and recovery](cleanup.md) | Archived files and recovery instructions |

Research papers remain in `reference/`; slides and presentation sources remain in
[presentation/](presentation/README.md). Dated investigations and benchmark scripts
are stored in the local cleanup archive.
