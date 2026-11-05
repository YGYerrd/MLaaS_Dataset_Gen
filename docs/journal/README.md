# Dataset journal submission

> Historical September 2026 draft: its 11,907-service Lambda cohort is not the
> retained November dataset. The old shards, audit exports and one-off audit
> CLIs were removed. The current reconciled classification subset contains
> 6,975 services in 70 pools; see [retained datasets](../datasets.md).

The new article centres on the service dataset, provenance, reconstruction and reproducible reuse. It is separate from the accepted ICSOC paper's camera-ready revision.

**Current state:** author-review draft 0.2, updated on 21 September 2026 using the selected Lambda extract at `outputs_ml_lambda/outputs`. All **11,907 unique completed services** passed the defined record/artifact inclusion audit. The cohort spans **11 model IDs, 21 dataset names and 123 families**. No external submission has been made; author information, public deposition and executable validation remain outstanding.

| File | Purpose |
| --- | --- |
| [manuscript.md](manuscript.md) | Full first narrative draft in a provisional Scientific Data Data Descriptor structure, with explicit unresolved evidence and author inputs |
| [submission-plan.md](submission-plan.md) | Venue recommendation, verified local inventory, contribution beyond ICSOC, remaining work and draft cover letter |
| [reuse-protocols.md](reuse-protocols.md) | Defined tasks, labels, baselines, splits, metrics and claim boundaries; proposed experiments, not results |
| [data-dictionary.md](data-dictionary.md) | Important implemented records and interpretation rules; coverage is release-dependent |
| audit_dataset.py | Reproducible read-only inventory of named SQLite catalogues |
| audit_lambda_release.py | Read-only Lambda inclusion audit with path relocation, recipe/family checks and shared-artifact checksums |
| [Lambda audit summary](evidence/lambda-valid-completed-summary.json) | Current cohort totals, validity rules, limitations and exact roster checksum |
| Valid completed roster | One row per included service, with source shard, identities, metrics and evidence statuses |
| Per-record checks | Inclusion decision for every completed candidate |
| Excluded noncompleted rows | Failed source records, preserving shard overlap |
| Family summary | Valid service count and saved audit size for each family |
| [Initial local inventory](evidence/catalogue-audit-2026-09-21.json) | Historical first-draft evidence only; does not contribute records to the selected Lambda cohort |

The concrete task counts are 4,175 image classifiers, 2,800 text classifiers, 4,500 fill-mask services and 432 token classifiers. Trust evaluation is unsupported for all included records; 1,283 have complete, sufficient explanation evidence with finite scores. These restrictions do not invalidate their recorded training and task measurements.

"Valid completed" means the checks documented in the audit summary passed. It does not mean models were replayed, all original raw data were retrieved or compositions were executed. The conference's 25,900-service and 10,000-request counts remain prior-work context and are not the journal cohort totals.
