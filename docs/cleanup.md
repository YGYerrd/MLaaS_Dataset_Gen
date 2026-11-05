# Repository cleanup: 5 November 2026

The repository keeps the working generator/composition packages, current regression
tests, environment helpers, SQL queries and authored documentation. Dataset files
are consolidated under `outputs/legacy/` and `outputs/pooled/`; see
[retained datasets](datasets.md) for counts, contents and limitations.

Removed material includes obsolete benchmark/pilot scripts, family-review and
diagnostic scripts, their two dedicated test modules, dated investigation reports,
one-off CSV/JSON/log exports, superseded experiment databases and Lambda source
shards, the compressed Lambda extract, generated PDF/slide copies, temporary
tooling, test run directories and local recovery archives.

The active pool-preparation checks formerly imported from the journal audit CLI
now live in `mlaas_data_generator/artifact_audit.py`. The pool-preparation script
requires explicit source-root and roster arguments. Reconstruction validation and
the current optional backend smoke harness remain available.

Before removing superseded outputs, both retained databases passed SQLite
`integrity_check`. Every retained file was hashed before and after relocation.
Database table counts were preserved; only signature lookup paths and the pooled
artifact-location lookup table were updated. Signed recipe/family artifacts were
not rewritten. Both relocated databases passed `quick_check`, and foreign-key
checks were compared with their pre-cleanup state.

Each dataset's `retention.json` records hashes and coverage. The legacy catalogue
had 3,998 missing referenced signature files before cleanup; that gap is recorded,
not silently treated as complete coverage. All 6,975 pooled signatures were retained.

The previous `.project-archive` and `.outputs_cleanup_recovery_*` folders are no
longer recovery sources. Tracked source history remains in Git; deleted ignored
run outputs are not recoverable through Git. Existing working-tree changes were
preserved, and no commit or staging operation was performed.

Validation: the retained suite ran 949 tests (948 initially passed; one stale
composition-eligibility assertion was corrected). All 73 tests in the affected
weight-retention and pool-preparation modules then passed. The relocated pooled
inputs regenerated the same 1,050-composition CSV and metadata. Generator CLI,
pool-preparation and reconstruction-validation help commands also succeeded.
