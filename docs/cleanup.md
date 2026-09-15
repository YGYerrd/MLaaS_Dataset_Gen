# Project cleanup and recovery

The 9 September 2026 cleanup keeps the generation and composition workflows, a
small core test suite, environment helpers, SQL reports and current documentation.

107 files were removed from the working tree after a verified backup:

| Area | Files archived | Reason |
| --- | ---: | --- |
| Extended tests and fixtures | 68 | Keep eight core test modules instead of the full suite |
| Package-local composition tests | 5 | Duplicates or stale copies of the main tests |
| Generator legacy modules and utilities | 22 | Unused orchestration, old CLI helpers, compatibility shims and one-off audits |
| Diagnostic and benchmark scripts | 5 | Investigations outside the normal workflow |
| Dated investigation documents | 7 | Historical evidence; current checkpoint settings extracted into a dedicated guide |

The retained 114 checks cover manifests, execution, SQLite records, exact
checkpoints, requests, search, composition validation and update signatures. The
original suite collected 868 checks. The core suite does not provide the original
backend, reduced-checkpoint, policy and legacy regression coverage.

Runtime diagnostics, shared-parent generation, checkpoint migration, request
schema versions and FedAvg modules imported by the active workflow remain.
The separate unused `mlaas_data_generator/federated/` implementation is archived.
Archived Python import paths are no longer supported in the cleaned working tree.
No run databases, manifests, checkpoint artifacts, credentials or environments
were removed. Existing Git staging was left intact; review both staged and
unstaged changes before committing.

## Recovery archive

Local snapshot: [cleanup-20260909-190115.zip](../.project-archive/cleanup-20260909-190115.zip).
It contains 257 pre-cleanup source/document files, including their uncommitted
working-tree contents. `CLEANUP_MANIFEST.json` records original paths, SHA-256
hashes, the removed-file list and the retained test modules. Every archived file
was verified against its original bytes before removal.

The archive is ignored by Git. Copy it separately if moving or sharing the project;
it will not be included in a normal commit or clone. It excludes Git state,
credentials, run outputs and Python environments.

To recover a separate copy, run from the repository root:

```powershell
Expand-Archive -LiteralPath .project-archive/cleanup-20260909-190115.zip -DestinationPath .project-archive/recovered-20260909
```

Inspect that copy and restore individual files as needed. For the former full test
suite or investigation commands, use the recovered source tree with the existing
environment. Do not extract over ongoing work: the archive contains older copies
of active files as well as removed files.

Historical sharding, partitioning, RMSprop, regression correction, recipe evidence
and checkpoint benchmark reports retain their original names under `docs/` inside
the archive. Benchmark scripts and smoke campaigns also retain their original
paths and reproduction commands there.
