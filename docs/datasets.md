# Retained datasets

All local dataset outputs live under `outputs/`, which is ignored by Git. A normal
clone does not include these databases or artifacts. Run commands from the repository root.

| Dataset | Database | Services | Available signature files |
| --- | --- | ---: | ---: |
| Legacy | `outputs/legacy/services.db` | 19,406 completed | 14,783 |
| Reconciled classification pools | `outputs/pooled/services.db` | 6,975 completed in 70 families | 6,975 |

Each directory contains its own `update_signatures/` and `retention.json`.
The retention record lists artifact SHA-256 hashes, database table counts and the
checks performed during the 5 November 2026 consolidation. Signature lookup paths
in `service_artifacts` and `service_metrics` now point to these directories.

The legacy database references 18,781 distinct signature paths. **3,998 were
already missing before cleanup** and were not found in the available old output
trees or recovery archive. Their references remain in the database, and
`outputs/legacy/retention.json` lists them explicitly. Some legacy services never
recorded signatures. The legacy catalogue is retained for historical analysis;
old experiment checkpoints are not retained as a replay environment.

## Pooled dataset contents

`outputs/pooled/` contains:

- `services.db`: the reconciled, completed classification services, their metrics,
  generation/execution/split provenance, and source reconciliation records.
- `update_signatures/`: all 6,975 referenced update signatures.
- `service_artifacts/`: the selected immutable client recipes, family documents,
  reservations, reconstruction support, and shared initial-state payloads. These
  are required dependencies of the dataset, rather than disposable test outputs.
- `services.csv` and `pools.csv`: the verified service and family rosters consumed
  by the composition-manifest generator.
- `reconciliation.csv`, `summary.json` and `REPORT.txt`: the preparation decisions
  and their validation boundaries.
- `dependencies.json`, `dependency_files.csv`, `path_map.json` and
  `replay_manifest.json`: pinned dependency identities and relocated local paths.
- `composition_manifest.csv` and its metadata JSON: the current 1,050 planned
  compositions created on 4 November. These are selections, not execution results.

The preparation audit verified records, artifacts and dependencies. It did not
retrain models, evaluate compositions, or certify cross-platform numerical replay.
The original Linux/CUDA runtime differs from the local Windows/ROCm runtime.

## Location and replay limits

Hash-bound recipe and family JSON files retain their original bytes and recorded
Linux paths. Relocated file locations are available in the database's
`preparation_artifact_locations` table and `prepared_service_locations` view, as
well as `path_map.json`. The resolved locations in those records point to
`outputs/pooled/`. Do not edit signed documents to change their paths.

The current replay/validation readers do not automatically apply this path map;
they still require a path-resolver integration or the original recorded filesystem
layout. This was a limitation of the prepared dataset before consolidation and
remains a prerequisite for running its recipes here. Manifest selection works
directly from the relocated directory.

Pinned model and raw-data dependencies remain in `D:\huggingface\hub` and
`D:\huggingface\datasets` as recorded in the dependency inventory. Cleanup did not
modify those external caches or external storage at `D:\MLaaS`.

The old Lambda shards, other experiment databases, one-off validation exports,
logs, temporary tooling and recovery archives were removed. The historical journal
draft describes a broader 11,907-service cohort; the retained classification subset
is explicitly the 6,975-service dataset above.
