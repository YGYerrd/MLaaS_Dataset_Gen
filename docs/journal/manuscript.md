# A dataset of artificial intelligence services with training provenance and reconstruction recipes

> Historical September 2026 draft: its 11,907-service Lambda cohort is not the
> retained November dataset. The old shards, audit exports and one-off audit
> CLIs were removed. The current reconciled classification subset contains
> 6,975 services in 70 pools; see [retained datasets](../datasets.md).

**Author-review draft 0.2 | 21 September 2026 | Provisional format: Scientific Data, Data Descriptor**

**Editorial note - remove before submission.** This is a new dataset-centred manuscript, separate from the ICSOC camera-ready revision. The selected source is the Lambda extract at `outputs_ml_lambda/outputs`. The journal cohort contains the unique completed service records that pass the documented record and artifact audit; failed records are retained in an exclusion inventory. The exact inclusion roster and its checksum accompany the draft. Public deposition and executable validation remain outstanding. Bracketed author inputs identify unresolved information. No new training, model reconstruction or composition experiment was performed for this update. See [submission plan](submission-plan.md).

**Authors, affiliations and corresponding author:** [AUTHOR INPUT: agree the journal author list, order, affiliations, contributions and corresponding author. Do not automatically reuse the conference author list.]

## Abstract

Research on artificial intelligence service selection requires data linking model capabilities to their training and evaluation conditions. We describe 11,907 completed service records from 11 source model identifiers, 21 dataset names and 123 generation families. The collection covers image classification, text classification, fill-mask prediction and token classification. Each included record links measured task attributes to generation provenance and a reconstruction recipe containing selected training-row identities, effective settings and a shared initial state. An inclusion audit verifies recorded metrics, recipe and family identities, saved evaluation boundaries and the checksums of shared checkpoint and reconstruction-support files. Individual trained checkpoints are not retained in this cohort. The records support catalogue analysis, attribute-prediction studies and reconstruction research, with collaborative selection requiring additional task-specific eligibility and execution checks. Explicit evidence statuses delimit metric reuse: trust evaluation is unsupported throughout the cohort, while 1,283 records have complete, sufficient explanation evidence and finite scores. Stored-artifact validation is distinguished from model replay and measured composition performance.

## Background & Summary

Artificial intelligence services are commonly described by the task they perform and a small set of performance attributes. Research using these descriptions must also account for how a model was obtained, which examples it encountered, how its outputs were evaluated and whether the recorded execution conditions match the intended use. These distinctions become particularly important when several trained models originate from a common source but differ in their training data or optimisation settings.

We use artificial intelligence as a service (AIaaS) as the broader service context and machine learning as a service (MLaaS) for the learned-model services represented here. A record is an evaluated model configuration or training outcome. It is not necessarily a separately hosted commercial endpoint. The dataset therefore represents controlled service-generation and evaluation conditions; measurements from local execution do not establish the behaviour of a commercial provider's deployment.

Existing service-composition resources provide useful context. HSC includes AI service descriptions, quality-of-service attributes and generated composition cases with requirements and workflows [1]. The earlier collaborative AIaaS composition paper associated with this work reports 25,900 service records and 10,000 generated requests and studies composition selection using a surrogate objective [2]. Those counts describe that paper's reported dataset. They are not assumed to describe the current journal release.

This descriptor develops the data documentation needed to interpret and reuse the service records: the observational unit, source-to-service provenance, recorded training variation, measurement semantics, reconstruction dependencies and task-specific eligibility. It follows the motivation of dataset datasheets, which make collection and intended-use assumptions explicit [3]. Persistent identifiers, documented formats and traceable provenance are also relevant to the FAIR principles [4]; implementing these features does not by itself certify that an undeposited collection satisfies every FAIR principle.

A running example is a family of text classifiers that share a model architecture, label vocabulary and initial state. Each member is trained on a recorded selection of examples using a recorded configuration. A researcher can filter members by measured quality and latency, investigate how training variation relates to those measurements, or select compatible members for parameter aggregation. Catalogue scores describe the members before aggregation. The aggregate's quality must be measured separately. This example illustrates the relationship among the records without presuming that combining models improves performance.

**Relationship to prior work.** The selected journal cohort is the audited Lambda collection, rather than a union of the three local databases inspected for the first draft. Its contribution is the fuller provenance, reconstruction and reuse description, together with record and artifact validation. The conference article remains the source for its original composition experiment. No 10,000-request collection or conference algorithm-result set was established as part of this Lambda audit. [RELEASE REQUIRED: reconcile source/content overlap with the conference dataset and identify any reused results precisely. Different service identifiers alone do not prove independent observations.]

## Methods

### Observational units and terminology

The primary observational unit is a service record identified by `service_id`. Every included Lambda record has generation mode `independent`, family contract `raw_source_family_v4` and recipe-only checkpoint retention. Other generation modes supported by the software are outside this cohort. Execution attempts and failures are separate observations. A completed execution is not a guarantee of useful predictive quality, adequate evidence for every metric or composition eligibility.

A source model is identified by its model reference and, where recorded, its revision. A model architecture describes its computational structure. A software library supplies an implementation, and a repository hosts source artifacts. These categories are distinct: Hugging Face is a repository ecosystem, Transformers and scikit-learn are libraries, and BERT is an architecture family. Provider identity is a separate concept and is only applicable where the service's hosting provenance supports it.

For supported independent generation, a family connects clients to a common initial checkpoint and compatible task, preprocessing, label and parameter contracts. Client-specific choices such as selected examples, quantities and optimiser settings produce distinct service observations within that family. Family size does not count independent source models, and neither family size nor the number of service records counts commercial providers.

### Sources and service generation

The generator uses configured model and dataset sources to prepare training and evaluation data. Source references, revisions and data contracts are retained where supported. The family document records the shared preparation and initial-state dependencies. Client recipes record the ordered source-row identities actually accepted for training, allowing the intended selection to be distinguished from the realised sample count.

Training variation is generated through explicit recipes. The current generation policies support independent variation in quantities and, for suitable task contracts, IID, Dirichlet and shard sampling. A non-IID profile requires a defined sampling axis; a class-distribution profile must not be applied to an unlabelled text task as though its examples were classification labels. Structured tasks retain their own sampling and supervision units. Policy and recipe versions preserve the interpretation of historical records.

Some current policies select joint training settings using historical catalogue evidence. Supported, challenging and stress bands describe intended recipe conditions, not measured outcome classes. The evidence source and policy version must accompany these records. Because historical evaluations informed those choices, they cannot also be presented as previously untouched final tests of the same selection policy. A journal evaluation must freeze the policy before assessing it on reserved evidence.

The durable client recipe records effective settings and observations, including realised quantities, seeds, training completion and relevant stopping information. Requested settings alone are insufficient when execution changes the effective batch configuration or available sample count. Records with failed or unsupported execution remain distinguishable from completed observations. A failure count in a database is not automatically a complete denominator for all attempted runs.

The extracted generation manifest contains 37,300 enabled service IDs. Only completed observations passing the inclusion audit enter the journal cohort; planned rows are not service measurements. Included record timestamps run from 18 September 2026 09:30:45 to 20 September 2026 01:18:00, as stored in the source databases. The inclusion roster records the source shard, service and family IDs, model/dataset identities and revisions, recipe hash, task metric and retention policy. [RELEASE REQUIRED: package the exact source manifest and environment evidence, verify timestamp timezone and hardware details, and reconcile content-level duplication and prior-publication overlap.]

### Training, catalogue evaluation and audit data

The current family workflow reserves evaluation data separately from client training. Catalogue evaluation supplies the attributes used during selection. A distinct saved audit reservation supports subsequent measurement of selected models or aggregates. Where source-group information is available, the reservation can preserve document or other group boundaries rather than treating related rows as independent examples.

Overlapping training data among clients is permitted and is part of the recorded training configuration. Total client exposures can therefore exceed the number of distinct source examples. Both quantities matter for interpreting aggregation weights and the statistical dependence among services. Reserved evaluation rows must be checked against the union of relevant training rows, including any shared-parent preparation that applies to the chosen generation mode.

These reservations address exposure within the recorded workflow. They do not establish that a pretrained model has never encountered the source benchmark. Unknown pretraining or historical evaluation exposure remains a limitation. Repeated requests from one family also remain statistically dependent even when their preference vectors differ.

### Performance and evidence semantics

Metrics are stored with a name, domain, unit, optimisation direction and typed value. Task quality is interpreted using its named metric and reduction: this cohort includes image/text classification accuracy, masked-token accuracy and supervised-token weighted F1. Segmentation and summarisation observations are not included merely because the generator supports them or uses some of their source datasets for another task. A generic `metric_score` is a normalised task-specific value; equal values across different task metrics do not establish equivalent performance.

Timing fields require their original measurement denominator. A task total, batch mean and per-example latency are different quantities. Hardware, precision, batching and runtime provenance are needed before converting or comparing them. Resource-cost scores are modelled indicators unless the record explicitly supplies monetary measurements. Neither a cost proxy nor predictive stability establishes billed endpoint cost or service availability.

The repository's trust evaluation measures behaviour under specified perturbations. Its explainability evaluation uses a specified faithfulness protocol and controls. These operational scores do not measure general trustworthiness, safety or human interpretability. Their protocol, independent input counts, exclusions and uncertainty evidence are part of the observation.

Under the current benchmark evidence policy, sufficiency requires minimum independent successful-input counts of 16 for trust and 8 for explanation, input and trial coverage of at least 80%, and a maximum 95% interval width of 0.15. These are protocol settings. All 11,907 included records identify `benchmark_evaluation_v1` and mark trust evaluation unsupported. Explanation status is complete for 1,283 records, insufficient for 345 and unsupported for 10,279. The 1,283 complete records also have a positive stored sufficiency flag and a finite explanation score; the trial evidence was not independently re-evaluated in this audit. A numeric field must not override an unsupported or insufficient status.

### Checkpoint retention and reconstruction

Model artifacts are referenced by the catalogue; full tensor parameters are not ordinary database columns. Storage modes must distinguish retained checkpoints, any supported checkpoint patches, shared initial states and recipe-only records. A hash or update signature does not establish that the complete trained state is available.

For supported independent families, recipe-only retention keeps the common family initial checkpoint and client reconstruction recipe while omitting the individual trained checkpoint. Reconstruction verifies dependencies and selected prepared content, resets model and training state, and replays the effective configuration. It requires the relevant raw sources, preprocessing artifacts and compatible runtime as well as the recipe itself.

Replay is distinct from checkpoint decoding. Successful replay checks do not guarantee identical floating-point parameters across environments. Functional fidelity can be assessed against a retained reference or an appropriately matched retained twin using prediction agreement, metric differences and runtime observations. Comparisons without such a reference cannot establish recovery of the original model's behaviour. Any exact state equality is reported only for the configurations in which it was measured.

### Requests and collaborative selection

The request generator produces synthetic requirements containing functional conditions, preference weights and constraints. Their distribution is controlled by a recorded profile and seed. The requests model experimental selection scenarios; they are not observed user preferences or production demand. A regenerated request collection is a new version and must not be described as the unchanged 10,000-request conference collection without identity checks.

For recipe-based collaborative composition, the current schema version 6 binds requests to verified family and recipe evidence and objective version 3. Candidate membership is frozen for search and later validation. Compatibility requires the relevant shared architecture, parameter schema, initial state, preprocessing, label semantics and evaluation protocol. Reconstruction support alone does not imply composition support; masked-language-model records currently lack the composition evaluation protocol required by this workflow.

Search operates on catalogue evidence using MAB, genetic, greedy and random strategies. The current MAB implementation samples a bounded arm set and applies UCB1 within that set. It does not enumerate every subset of the whole catalogue. Repeated evaluations and unique candidates are separate search-budget quantities, particularly for a deterministic surrogate. Exhaustive enumeration is used only below an explicit candidate-space limit, yielding an exact reference for the frozen surrogate and candidate pool, not a measured global optimum of deployed performance.

For supported selections, compatible floating or complex parameters are averaged with weights proportional to realised client training-example counts. Integer and Boolean tensors are retained from the shared initial state. This is a specific parameter-aggregation experiment related to weighted model averaging [5]. It does not establish output ensembling, cross-architecture composition or a multi-stage service workflow.

The aggregate is then measured on the saved audit reservation. Predicted quality is the training-count-weighted mean of member catalogue scores, an uncalibrated heuristic. Objective version 3's measured utility replaces predicted quality and latency with audit measurements while retaining catalogue evidence for other dimensions. It is consequently not a fresh measurement of every aggregate property. Prediction error is defined as measured minus predicted.

## Data Records

The implemented storage model uses SQLite for structured records and referenced artifacts for larger objects. JSON describes versioned contracts, requests and provenance. CSV exports can provide tabular views, but must preserve identifiers and retain links to the full records. Table 1 summarises the principal relationships; [the working data dictionary](data-dictionary.md) records their interpretation.

**Table 1. Principal implemented record groups. Presence and coverage must be reported for the selected release.**

| Record group | Principal identifier | Contents and relationship |
| --- | --- | --- |
| `services` | `service_id` | Execution status, task, modality, model and dataset references, training regime and configuration |
| `service_metrics` | `service_id`, `metric_name` | Typed observations with domain, unit and direction; joins to services |
| `service_generation_provenance` | `service_id` | Source revisions, family identity, contract hashes and recipe references |
| `service_split_provenance` | `service_id`, `split_name` | Split counts and recorded distribution/configuration information |
| `service_execution_provenance` | `execution_id` | Recorded run identity and immutable execution document |
| `service_artifacts` | `artifact_id` | Service-linked artifact types, references and integrity metadata where recorded |
| `service_checkpoint_storage` | `service_id` | Explicit checkpoint storage mode and associated state/artifact references |
| Family and recipe documents | Content-addressed reference | Shared preparation and initial state; ordered client row selections and effective training recipe |
| Request files | Versioned request identifier | Synthetic requirements, constraints, objective version and frozen candidate membership |
| Search and validation outputs | Run, request and method | Selected members, surrogate scores, measured observations, status and failure reasons |

The local inclusion roster is valid_completed_services.csv. It contains one row per included service, with a source database link key and recipe identity. The companion completed-record audit records every completed candidate's validation decision, and excluded source rows preserve noncompleted records without treating duplicate shard entries as independent failures. Family counts are supplied in family_summary.csv. The [audit summary](evidence/lambda-valid-completed-summary.json) records the rules, timestamp and roster checksum. [RELEASE REQUIRED: assign a public repository/version, exact distributable file list, licenses and formal data citation. Local paths are not a public data deposit.]

## Data Overview

The inclusion roster contains 11,907 distinct completed service IDs across 123 generation families, 11 source model IDs and 21 dataset names. These source identifiers do not count architectures or hosted providers. Image and text classification share the broad `classification` family label; the four concrete task profiles are separated in Table 2 using `hf_task`. No quality-performance threshold was used to remove weak models. Failed and unobserved planned services are outside the cohort. The earlier 19,406-record, merged and production inventories are retained as historical authoring evidence and contribute no rows to this total.

**Table 2. Completed service records passing the Lambda inclusion audit.**

| Concrete task profile (`hf_task`) | Completed valid records |
| --- | ---: |
| Image classification (`image_classification`) | 4,175 |
| Text classification (`sequence_classification`) | 2,800 |
| Fill-mask prediction (`fill_mask`) | 4,500 |
| Token classification (`token_classification`) | 432 |
| Total | 11,907 |

## Technical Validation

### Record inventory and provenance coverage

The audit opened `validation.db`, `validation.gpu0.db` and `validation.gpu1.db` from the extracted Lambda directory with read-only connections and separate read transactions. The main database contains 700 failed rows and no completed rows. GPU shard 0 contains 6,174 completed and 7,711 failed rows; GPU shard 1 contains 5,733 completed and 5,614 failed rows. Failed IDs overlap between stores, so the 14,025 failed source rows must not be interpreted as independent attempts. Completed identities are reconciled separately before producing the inclusion roster.

All 11,907 completed candidates pass `lambda_completed_record_audit_v1`. The checks require finite named primary metrics, normalised scores and training/evaluation losses, agreement between database metrics and the authoritative recipe, recipe and family hashes and identities, unique selected training rows disjoint from catalogue/audit rows, supported completed-epoch optimiser history, and recipe-only retention. Weak but finite models remain included. Unsupported trust or explanation evaluation does not invalidate the service record; it restricts reuse of those attributes.

The audit also verifies the recorded family reservation, reconstruction-support files and stored shared-checkpoint payload checksums. Linux artifact paths are mapped in memory to the extracted Windows directory without modifying source databases or hashed artifact bytes. This establishes recorded-data and stored-file integrity for the stated checks. It does not load or retrain models, decode checkpoint tensor states, retrieve original training sources, independently verify every metric trial or establish composition eligibility. [VALIDATION REQUIRED: verify raw-source availability/content and executable reconstruction on representative release records; complete license and publication-overlap checks.]

### Existing reconstruction observations

An inspected two-service verification report contains clients trained on 160 and 240 examples from one family. The stored report records equal original and reconstructed states for both clients, and zero differences in reported loss, accuracy and F1. The respective original and reconstructed accuracies are 0.7888888717 and 0.8888888955. Both records indicate that training changed the initial state. These observations support reconstruction for those two recorded cases only. They were read from saved evidence and were not rerun during manuscript preparation.

[VALIDATION REQUIRED: retain the verified pilot's complete environment and evidence in the deposit, then report representative reconstruction coverage across the release's task, architecture, precision and retention strata. Include failed and unsupported cases, prediction agreement, metric deltas, runtime and storage measurements. Do not extrapolate two exact matches into a dataset-wide fidelity guarantee.]

### Measurement and composition validation

The repository contains a documented 12-profile benchmark evaluation pilot and separate directional synthetic controls. Those records are useful protocol-development evidence, but their task coverage and original evidence rules differ from a final release-wide validation. Later policy reassessment must remain distinguishable from the original evaluation. No dataset-wide claim about trust or explanation completeness follows from the existence of these pilots.

For the journal release, measurement validation should verify units, reductions, finite values and evidence gates by task. Composition validation should execute frozen selections and report measured primary quality, per-example latency, prediction errors and execution status. The strongest individual compatible service provides an interpretable comparator; random and simple weighted selection provide controls. Negative aggregation outcomes remain valid observations. Similarity among member attributes is not evidence of complementary prediction errors or improved aggregate performance.

[VALIDATION REQUIRED: add release-specific numerical evidence here. The new manuscript currently has no newly executed large-scale composition comparison. The companion reuse protocol specifies candidate tasks and baselines without presenting them as completed results.]

## Usage Notes

Users should first select the required capability and metric protocol, then filter completed records using evidence status and provenance coverage. A task label alone is insufficient for deciding whether a record can be reconstructed, aggregated or used as a target in a supervised prediction experiment. Legacy records should retain their original semantics rather than receiving inferred current-version annotations.

For catalogue analysis, join metrics to services by `service_id` and retain metric names, units and directions. When predicting an unknown attribute, exclude the target and any algebraic derivatives from input features. Normalisation and imputation must be fitted on training data only. Partition services by family or a stronger source-model grouping, depending on the claimed generalisation setting, to avoid distributing near-identical variants across training and test sets. Repeated requests involving a family should remain in the same evaluation partition.

For reconstruction, resolve the family and recipe references and verify the raw-source, preprocessing and shared-initial-state dependencies before replay. Record the new environment and observations separately from the original catalogue. Reconstruction may require repeating the original training workload; a smaller checkpoint archive does not imply cheaper compute.

For collaborative selection, follow the version-specific request and validation procedure in the deposited code. Preserve frozen candidate membership between search and audit evaluation. A composition may complete successfully while performing worse than an individual contributor. Report unsupported requests and failed executions alongside successful selections. The current recipe workflow does not supply the dependencies, topology or end-to-end measurements needed for a general service-orchestration benchmark.

## Data Availability

The selected local dataset is the Lambda inclusion roster and its referenced records and artifacts under `outputs_ml_lambda`. [AUTHOR INPUT: assign the public journal dataset version, repository, DOI or reviewer-download URL, distributable artifacts and access conditions. A public deposit has not been created. Cite the conference dataset separately if reused.]

## Code Availability

The local implementation contains the generator, versioned request schemas, objective definitions, search methods and validation procedures described above. The code observed during preparation was at Git commit `5c4cc2b659386b70e0dc8680e2cd9d64e771b7e4`, with pre-existing working-tree modifications. This commit alone therefore does not identify the complete authoring environment. The Lambda inclusion audit, its summary and per-record outputs accompany the draft; the initial general inventory is retained separately.

[RELEASE REQUIRED: deposit or tag the exact code and environment used for the journal release, including relevant local changes, dependency versions, example commands and expected outputs. Replace local references with durable public links.]

## Author Contributions

[AUTHOR INPUT: agreed roles for conceptualisation, methodology, software, generation, curation, validation, analysis and writing.]

## Competing Interests

[AUTHOR INPUT: confirm the appropriate declaration for every author.]

## Funding

[AUTHOR INPUT: funding sources and grant identifiers, or a confirmed no-specific-funding statement.]

## Acknowledgements

[AUTHOR INPUT: contributions not covered by authorship and any institutional support.]

## Use of artificial intelligence tools

Draft disclosure for author review: OpenAI Codex assisted with manuscript organisation and drafting, source review, a read-only catalogue inventory script, and proposed reuse protocols. It did not run new model-training or composition experiments for this draft. [AUTHOR INPUT: confirm and extend this disclosure for the complete study and document human verification of the submitted text, analysis and references.]

## References

1. Wang, X., Rong, D., Xu, H., He, X. & Wang, Z. HSC: An Artificial Intelligence Service Composition Dataset from Hugging Face. In *Service-Oriented Computing, ICSOC 2024*, LNCS 15405, 225-239. [Publisher record](https://doi.org/10.1007/978-981-96-0808-9_17). Publisher first-online date: 7 December 2024; confirm the final proceedings citation format when exporting the bibliography.
2. Kanneganti, D., Mistry, S., Fattah, S. & Krishna, A. *A Collaborative Artificial Intelligence as a Service (AIaaS) Composition Dataset*. Supplied manuscript, accepted as a full paper at ICSOC 2026 according to the supplied acceptance correspondence. [Local source](../reference/A_Collaborative_AIaaS_Composition_Dataset_1.pdf). Final publication details pending.
3. Gebru, T. et al. *Datasheets for Datasets*. [Author manuscript](https://arxiv.org/abs/1803.09010). Cite the chosen version consistently in the final bibliography.
4. Wilkinson, M. D. et al. The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data* **3**, 160018 (2016). [Publisher record](https://doi.org/10.1038/sdata.2016.18).
5. McMahan, H. B., Moore, E., Ramage, D., Hampson, S. & Agüera y Arcas, B. Communication-Efficient Learning of Deep Networks from Decentralized Data. *Proceedings of Machine Learning Research* **54**, 1273-1282 (2017). [Proceedings record](https://proceedings.mlr.press/v54/mcmahan17a.html).

**Bibliography work remaining:** add formal citations for every input dataset/model release actually retained in the final source manifest, the new data deposit and the archived code. Resolve the precise journal-release overlap with all related papers. Do not reuse unverified reference metadata from the conference PDF.
