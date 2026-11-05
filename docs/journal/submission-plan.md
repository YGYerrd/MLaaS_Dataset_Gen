# Journal submission plan

> Historical September 2026 draft: its 11,907-service Lambda cohort is not the
> retained November dataset. The old shards, audit exports and one-off audit
> CLIs were removed. The current reconciled classification subset contains
> 6,975 services in 70 pools; see [retained datasets](../datasets.md).

**Status: manuscript draft 0.2 updated to the audited Lambda cohort of 11,907 valid completed service records; no journal submission, repository publication or external message has been sent.**

The user's request is for a new journal article centred on the dataset and its uses. The conference acceptance message and reviews are reference material. Their camera-ready deadline, registration requirement and artifact-track invitation are not instructions to perform those separate tasks.

## Article position

Working title: **A dataset of artificial intelligence services with training provenance and reconstruction recipes**.

The central contribution should be an interpretable and reusable collection of measured service variants, with enough provenance to understand or reproduce each applicable observation. Composition is one supported reuse setting. The journal paper should be assessed on the data's description, technical quality and availability, rather than on whether MAB outperforms another search algorithm.

The user selected the Lambda records for this update. The actual extracted directory is `C:\Users\Josh Boland\Documents\MLaaS\outputs_ml_lambda`, containing its source databases under `outputs`. The supplied spelling `outputs\_ml\_lambda` does not exist locally; the matching extracted Lambda directory was used and this interpretation was reported during the work. The journal cohort is the unique completed records passing the audit in that directory. No records from the earlier three-store inventory are added. Public release/version naming remains to be completed.

## Journal recommendation

**Provisional first choice: Scientific Data, Data Descriptor.** Its remit and review criteria fit a data-centred description with substantive technical validation. Complementary descriptors can expand prior publications through additional documentation, data or validation. Because the ICSOC article itself already describes the dataset, the new material must be demonstrated concretely. Acceptance or fit is not guaranteed. [Referee guidance](https://www.nature.com/sdata/policies/for-referees), [complementary descriptor guidance](https://www.nature.com/documents/Sci_Data_complementary_data_descriptor_checklist.pdf).

Use its dataset-oriented structure. Keep the proposed downstream benchmark programme in this working folder; the journal's current Usage Notes guidance excludes worked case studies, and a Data Descriptor is not the place for an extensive general application-results section. If application experiments become the main contribution, revisit the journal and article type. [Submission guidelines](https://www.nature.com/sdata/publish/submission-guidelines).

Current advertised APC on 21 September 2026: **£2,150 / US$2,690 / €2,390**, with applicable taxes additional; the price is determined at acceptance. Check institutional funding or coverage before committing. No payment or funding commitment has been made. [Official fees](https://www.nature.com/sdata/open-access).

**Alternative to consider: Data, Data Descriptor.** It provides a dataset-specific format with summary, data description, methods and optional usage notes, plus dataset access and license information. This is a format-based alternative; no acceptance likelihood, ranking comparison or fee quotation has been established here. [Author instructions](https://www.mdpi.com/journal/data/instructions).

Data in Brief was considered but is not recommended without clarifying how its requirements apply to a dataset already described in a conference paper. Its author guide could not be fully retrieved during this review; do not select it on the basis of old or partial guidance.

## What distinguishes the new paper

| Conference emphasis or reviewer concern | Journal contribution to establish | Evidence available now | Work still required |
| --- | --- | --- | --- |
| Approximately 1,500 source models expanded to 25,900 service records | Exact observational units and source-to-service lineage | Lambda roster: 11,907 valid completed services, 11 model IDs, 21 dataset names, 123 families; all independent recipe-only records | Reconcile content overlap with the conference snapshot and publish the selected manifest |
| Providers, libraries and architectures mixed | Consistent taxonomy with separate source, backend, architecture and endpoint fields | Source IDs and current contracts | Audit every release label and explicitly record unknowns |
| Model weights/biases said to be included | Exact artifact and retention inventory | Shared base payload/support checksums verified; all included client recipes omit trained client checkpoints | Verify original raw-source availability, executable reconstruction and redistribution conditions |
| Similarity interpreted as collaboration | Defined aggregation operator and measured outcomes | Current family checks and audit-validation implementation | Execute representative compatible selections; retain poor outcomes and failures |
| MAB reference treated too strongly | Frozen surrogate references labelled by search scope | Bounded-arm implementation and objective v3 | Publish unique-candidate/call budgets and small exhaustive controls |
| Broader uses only discussed | Reusable inputs, targets and defensible evaluation partitions | Proposed reuse protocols and rich metadata | Freeze splits; run the chosen data-quality/reuse demonstrations |
| Reproduction path unclear | Versioned data, code, dependencies and expected outputs | Current generator/reconstruction/composition guides | Fresh-environment replay of deposited examples and accessible release links |
| Static QoS and broad workflow claims | Explicit operational meaning and capability boundaries | Metric protocols and status flags | Unit audit; restrict availability/temporal/DAG claims to observed evidence |

The draft is newly written and does not reproduce conference figures or algorithm-result plots. Its final related-work disclosure must cite the accepted conference article and identify reused data and any shared results precisely. Do not invent a percentage threshold for a journal extension. [Editorial policy on overlap](https://www.nature.com/sdata/policies/editorial-and-publishing-policies).

## Selected Lambda cohort and audit results

The [inclusion audit summary](evidence/lambda-valid-completed-summary.json) and per-service roster supersede the first draft's provisional data-source discussion. Source databases and artifacts were read without modification. Each database had its own read transaction; no source size/modification change was detected. The extracted files were audited directly; equivalence to the compressed archive was not separately established.

| Database within `outputs_ml_lambda/outputs` | Completed candidates | Valid completed records | Failed source rows |
| --- | ---: | ---: | ---: |
| `validation.db` | 0 | 0 | 700 |
| `validation.gpu0.db` | 6,174 | 6,174 | 7,711 |
| `validation.gpu1.db` | 5,733 | 5,733 | 5,614 |
| Total source rows by outcome | 11,907 | 11,907 | 14,025 |

There are **11,907 distinct completed service IDs**, with no duplicate completed IDs or conflicting completed recipe hashes. Every candidate passes the defined checks; no minimum accuracy threshold removes weak models. The stores contain 25,932 rows across 24,532 distinct service IDs. Exactly 12,625 IDs have no completed record, and 700 IDs have a completed record and a failed record in another store. The valid completion is included once while failed evidence remains in the exclusion CSV. Source rows are not an independent-attempt denominator, so no failure-rate claim is made.

The valid cohort comprises 4,175 image classifiers, 2,800 text classifiers, 4,500 fill-mask services and 432 token classifiers. The database's broad `classification` label combines the first two, giving three `task_family` labels but four concrete `hf_task` profiles. There are 11 model IDs, 21 dataset names and 123 families; 112 families contain 100 completed valid members, while 11 have fewer. Counts of source names are not provider counts or proof of independent observations.

The manifest coverage check reconciles all observed and included IDs to the 37,300-row enabled source manifest. Exactly 12,768 planned IDs have no row in any of the three extracted service tables. They are not counted as services or assigned an inferred execution outcome.

All included records use generation mode `independent`, `raw_source_family_v4`, `recipe_only` retention and `benchmark_evaluation_v1`. The audit checks finite named task metrics and losses against the authoritative recipe, recipe/family/reservation identities, selected-row uniqueness and evaluation separation, supported optimiser history, and shared-base/support file checksums. It verified **12,434 unique files, 123 family documents, 123 base manifests and 54 reconstruction-support artifacts**. Model and tokenizer configuration bundles can contain multiple files. Source Linux paths are resolved in memory to the extraction location; hashed source documents remain unchanged.

**Validity scope:** these are valid completed records under `lambda_completed_record_audit_v1`, not a claim of successful model replay or composition. Raw sources were not retrieved, tensor states were not decoded, individual metric trials were not independently re-evaluated, and no whole-database `integrity_check` was run. A stored-payload checksum does not establish functional fidelity. The selected roster's SHA-256 is `e274611059a9cd7d744c143b5dd7ef6ecae6607d8d0cb0dce2bebbb795e50b11`.

Trust status is unsupported for all 11,907 records. Explanation status is complete for 1,283, insufficient for 345 and unsupported for 10,279. The 1,283 complete rows also have a positive sufficiency flag and finite score. These are the eligible recorded labels under that filter, not a fresh validation of all trials. The 4,500 fill-mask records lack the current composition protocol; other records still need instance-level composition checks. The initial 19,406-record/merged/production audit remains historical evidence and does not supply this cohort's totals.

Recorded runtime provenance gives Python 3.11.16 and PyTorch 2.5.1+cu121 for every included recipe, with a dirty Git working tree. Record timestamps span 18 September 2026 09:30:45 to 20 September 2026 01:18:00 as stored. This identifies recorded software observations, not a complete portable environment or independently established hardware profile.

The saved two-service reconstruction report at `outputs/recipe_validation_stage2_verified/summary.json` records exact original/replayed matches for two clients from one family. It is a narrow existing pilot, not a fresh or release-wide validation. The 12-profile metric pilot is similarly development evidence. The current `docs/recipe-composition.md` explains the manual validation procedure and explicitly states that no executions were run while implementing that workflow; implementation documentation alone is not a benchmark result.

## Ordered completion work

1. **Cohort selection and record audit completed.** Preserve the Lambda roster/checksum, source-shard mapping and excluded rows. Complete source-content deduplication, prior-publication overlap reconciliation and the formal public release manifest.
2. **Complete the data description.** Task/source/recipe counts and selected evidence coverage are now documented. Add a complete metric-unit/missingness inventory, formal source citations and distributable artifact list. Keep raw task metrics alongside normalised scores.
3. **Complete executable validation.** Family/recipe/support/base-file checks are complete for this roster. Verify original data dependencies and measure representative reconstruction fidelity. Execute composition examples only for verified eligible families; use the selected 123-family roster to set the sample and resource estimate.
4. **Choose the reuse demonstration.** The recommended first choices are reconstruction and one well-defined attribute-prediction task. Add composition once the eligible cohort and audit evidence are available. Publish split files, baseline configurations, predictions, failures and exact commands. Avoid presenting proposed experiments as results.
5. **Prepare the deposit.** Select an appropriate repository and version, confirm data/code licenses and upstream redistribution conditions, package the exact files, and provide anonymous reviewer access. Scientific Data requires downloadable review data initially and a qualifying public repository for later rounds. [Data policies](https://www.nature.com/sdata/policies/data-policies).
6. **Finalise the manuscript package.** Replace all author inputs, release markers and draft notes. Add final data and source citations, render the selected journal's submission format, and verify figures/tables against the deposited data. Agree author order, contributions, funding, interests and corresponding-author details. Confirm any required AI-use disclosure. [Journal AI policy](https://www.nature.com/sdata/policies/ai).
7. **Submit through the chosen journal account.** Upload the final manuscript, cover letter, related conference paper and required data links; complete truthful author declarations. The current draft cannot supply those declarations or a public deposit identifier. Submission intent is already expressed; further questions should obtain missing facts rather than repeatedly request generic permission to proceed.

## Cover-letter draft

**Draft only. Do not submit until the bracketed facts are resolved and the manuscript is complete.**

Dear Editors,

Please consider our manuscript, “A dataset of artificial intelligence services with training provenance and reconstruction recipes,” as a Data Descriptor in Scientific Data.

The manuscript describes a Lambda-derived cohort of 11,907 completed machine learning service records from 11 model identifiers, 21 dataset names and 123 generation families [FINAL PUBLIC RELEASE NAME/VERSION]. It documents source and training provenance, measurement semantics, reconstruction dependencies and supported-use boundaries. Record validation checks metrics, recipe/family/reservation identities, training/evaluation separation and shared artifact checksums. [ADD COMPLETED EXECUTABLE VALIDATION BEFORE SUBMISSION.] The data are available to reviewers at [DATA URL/IDENTIFIER], with code and reproduction instructions at [CODE URL/VERSION].

We disclose the related paper by Kanneganti, Mistry, Fattah and Krishna, “A Collaborative Artificial Intelligence as a Service (AIaaS) Composition Dataset,” accepted at ICSOC 2026. That paper reports a collaborative composition dataset and a surrogate-based selection benchmark. The present submission shares [EXACT DATA/CONTENT OVERLAP] and adds [EXACT NEW DATA, DOCUMENTATION AND VALIDATION]. We provide the related manuscript for editorial assessment.

[INSERT CONFIRMED AUTHOR APPROVAL, EXCLUSIVE-SUBMISSION, FUNDING AND INTEREST DECLARATIONS REQUIRED BY THE JOURNAL. These facts have not been established in this conversation.]

Yours sincerely,

[CORRESPONDING AUTHOR, AFFILIATION AND CONTACT]

## Sources and reproducibility

Policy pages were checked on 21 September 2026; live requirements and fees should be rechecked at submission. The paper and pasted message were inspected as supplied; the acceptance status was not independently verified. Git HEAD at inspection was `5c4cc2b659386b70e0dc8680e2cd9d64e771b7e4`, with existing local changes. Source-file hashes and the audit's limited identity digests are recorded in the JSON. They are not release-wide content checksums.

Reproduce the Lambda inclusion audit from the repository root, using new output names to retain the current observations:

```powershell
& '.venv/Scripts/python.exe' -u docs/journal/audit_lambda_release.py --root outputs_ml_lambda/outputs --output-dir outputs_ml_lambda/journal_audit_repeat --summary-copy docs/journal/evidence/lambda-valid-completed-repeat.json
```

The audit reads the three named databases and hashes stored dependencies; it does not train models, alter statuses, rewrite Linux paths or merge the original databases. A changing source can produce new counts; preserve the audit timestamp and source information with any quoted number.
