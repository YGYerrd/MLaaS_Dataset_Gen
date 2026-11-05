# Dataset reuse protocols

> Historical September 2026 draft: its 11,907-service Lambda cohort is not the
> retained November dataset. The old shards, audit exports and one-off audit
> CLIs were removed. The current reconciled classification subset contains
> 6,975 services in 70 pools; see [retained datasets](../datasets.md).

**Proposed protocols, 21 September 2026. These experiments have not been run for the new journal manuscript.**

The selected source is now the Lambda cohort: 11,907 completed services passing the record/artifact audit, from 11 model IDs, 21 dataset names and 123 families. Use the inclusion roster as the input population. Its four task profiles are image classification (4,175), text classification (2,800), fill-mask (4,500) and token classification (432). No regression, segmentation or summarisation experiment can be claimed from this roster. All trust evaluations are unsupported; only the 1,283 complete, sufficient, finite-score explanation records are candidates for an explanation-target experiment.

The dataset's practical value should be demonstrated with defined inputs, targets and evaluation boundaries. The primary submission can remain a Data Descriptor: the main text explains reuse and validates data quality; substantial application results may require a separate research article or a different journal format. These protocols are a research work plan and are not automatically supplementary material for Scientific Data.

## Recommended priority

Start with provenance and reconstruction validation, followed by prediction of one well-defined measured attribute. Add actual composition evaluation for families with a supported audit protocol. These uses depend directly on the distinctive data recorded here. Service classification is a lower priority because task and model fields can make the target trivial. General workflow orchestration requires additional data not established in the current collection.

| Use | Unit and inputs | Target/evidence | Initial comparators | Evaluation | Current boundary |
| --- | --- | --- | --- | --- | --- |
| Service filtering and ranking | Synthetic request plus eligible service metadata | Feasibility and graded utility under a named objective; measured audit utility where available | Random eligible ranking; quality-only and latency-only ranking; fixed weighted score | Feasible precision at k, constraint-violation rate, NDCG at k where nonnegative relevance labels are defined; measured regret on audited pools | Synthetic relevance is not user satisfaction; MAB membership is not ground truth |
| QoS prediction | Completed service and features available before the target measurement | One named measured attribute, with unit and runtime context | Training-set task median; regularised linear model; random forest | MAE, median absolute error, within-task rank correlation; error by held-out family | Static measurements do not support forecasting time-varying endpoint QoS |
| Reconstruction fidelity and cost | Family, client recipe and raw-source dependencies | Replayed model versus retained reference or matched twin | Load retained checkpoint; replay recipe | Prediction agreement, primary-metric delta, state equality where applicable, replay wall time, bytes retained | No reference means no fidelity claim; replay success alone is insufficient |
| Collaborative selection and aggregation | Frozen compatible family pool, request and candidate size | Measured aggregate quality and latency on saved audit rows | Best individual; random subset; greedy; GA; bounded-arm MAB | Quality difference from best individual, measured feasibility, surrogate error, execution failures and total cost | Same compatible parameter family; not a DAG or cross-architecture pipeline |
| Service classification | Deliberately restricted service descriptions and admissible metadata | Reconciled task category | Majority class; simple linear classifier; tree model | Macro F1, balanced accuracy, confusion matrix; unseen-source performance | Remove task labels and aliases; decide whether model identifiers give away the label |
| Evidence coverage and failure analysis | Complete scheduled roster and attempts, with stage information | Evaluation support/sufficiency or execution outcome | Descriptive stratification first; calibrated classifier only with a valid denominator | Coverage by task and protocol; stage frequencies; calibration if modelled | Catalogue failures may not capture every attempt; unsupported metrics are not zero quality |

## Shared design

The audited inclusion roster fixes the current input service IDs and records source database, family contract, model/dataset revisions and recipe identities. Preserve its checksum when deriving splits, and include the original seed/settings artifacts in the public release. Every included service is an independent, recipe-only `raw_source_family_v4` observation. Different service identifiers are not proof of independent observations. Exact hashes can detect some duplicates, but provenance overlap may require a stronger grouping rule. Records with failed attempts in another shard remain represented once by their valid completion, with failure evidence preserved separately.

Separate three generalisation questions:

1. **New clients within known source models:** group by family so that all its service variants stay together. State explicitly that related source models can still appear on both sides.
2. **New source models:** hold out all families sharing the source model identity/revision. This is stronger than a family split.
3. **New datasets or environments:** hold out the relevant dataset or hardware/runtime group. Do not claim this setting unless enough independent groups exist.

For a sufficiently broad release, use a fixed 60/20/20 allocation of groups to training, validation and test, with seed 42, stratifying only where group structure permits it. Publish the actual group assignments. If there are too few groups for a credible held-out test, use grouped cross-validation and report that limitation. This allocation is proposed, not an existing released split. Calibration evidence used to generate recipes is development data and must be accounted for when selecting a final test population.

Freeze preprocessing, feature selection and hyperparameters using training and validation only. Reserve audit examples before model generation where the workflow requires them. Service-level train/test partitions for predictive benchmarks and example-level catalogue/audit reservations are different boundaries; both must be checked.

Report uncertainty by resampling the independent grouping unit, normally family or source model. Multiple requests or seeds within one family are repeated measurements, not additional independent families. A proposed reporting default is 2,000 bootstrap replicates with the resampling seed recorded. Confidence intervals are informative only when the group count is adequate; report the number of independent groups next to them.

## Protocol A - predict an observed attribute

Select a target with unambiguous semantics and sufficient measured coverage. Per-example inference latency is useful only after confirming the stored denominator, runtime context and conversion. Named task quality is another option, evaluated within comparable tasks. Exclude missing, failed or unsupported targets without relabelling them as zero. Record exclusion counts and coverage so the target-available subset's selection bias remains visible.

Choose the decision time first. A pre-training predictor may use declared model metadata and planned recipe settings, but cannot use realised training outcomes, elapsed training time, realised stopping behaviour or final model diagnostics. A post-training, pre-inference predictor can use additional realised training information if those fields would actually be available at that decision time. Run these as separate tasks if both are useful.

Remove the target, its normalised counterpart and derived proxies that algebraically contain it. Predicting accuracy while retaining `metric_score`, or latency while retaining its reciprocal cost-efficiency proxy, can leak the answer. Fit any imputation, category encoding and target transformation on training data alone. Report models within their task/environment strata before presenting any pooled result.

Train a task-median baseline, a regularised linear model and a small random forest under identical splits. Select their settings only on the validation groups. For positive latency, a log transform may stabilise training, but report final error in the original time unit as well. Report MAE, median absolute error and rank correlation, plus the number of records and independent groups. Do not interpret a model fit on static local timings as a forecast of future commercial endpoint latency.

Expected outputs: split roster, admissible feature list, target definition, preprocessing configuration, fitted baseline settings, per-service held-out predictions, grouped error report and exclusions.

## Protocol B - reconstruct service variants

Stratify eligible records by task contract, architecture/source model, data profile, precision and retention mode. Include ordinary and difficult recipes from the frozen roster. Select samples before looking at reconstruction performance. A minimum pilot should cover every claimed supported mode; the final number per stratum should follow available coverage and resource estimates rather than an invented universal sample-size rule.

For each sampled service, verify the common initial state, selected-row identities, preprocessing and effective settings. Obtain a retained validation checkpoint or explicitly matched retained twin. Measure predictions on the identical saved evaluation rows, using identical decoding and metric definitions. Keep checkpoint loading separate from recipe replay and track whether any client checkpoint was read during replay.

Report prediction agreement, signed and absolute primary-metric differences, optional tensor comparisons, wall time, memory where measured and dependency failures. Classifier disagreement is not directly applicable to regression or generated text; use task-specific prediction differences and quality measures. Define any acceptable fidelity tolerance before examining the test results. Do not require bitwise equality as a universal portability criterion or infer it from equal rounded metrics.

For storage accounting, count the shared base and family artifacts once per family, then all required client recipes, raw-data dependencies and retained references. Compare this with the actual checkpoint representation used by the baseline, including compression/deduplication. Report common dependencies and incremental storage separately. A compressed archive's size alone is not a controlled storage comparison. Report training replay cost alongside any storage saving.

Expected outputs: immutable sample roster, verification reports, per-example comparison where distributable, failures, dependency manifest, timing logs and a byte-accounting table.

## Protocol C - select and execute compatible compositions

First obtain a capability report for the actual Lambda records. Require enough verified members per family for the chosen subset size. Family membership and passing the record audit do not establish valid aggregation. The 4,500 fill-mask records lack the current composition evaluation protocol and cannot enter this experiment solely because their recipes are recorded. The remaining 7,407 image/text/token classification records still require instance-specific composition checks. Trust-constrained requests cannot be supported by this cohort's stored evidence; explanation-constrained requests must filter to adequate records before freezing their pools.

Use one request profile and objective version per reported experiment. Freeze the candidate pools, request seeds and selection inputs. Requests should include both achievable and demanding constraints; report how any infeasible requests were generated. Do not repeatedly resample requests until all methods succeed. Distinguish a designed stress distribution from observed application demand.

Evaluate random, greedy, genetic and MAB search under recorded objective-call and wall-time budgets. Record unique candidates as well as total calls. The current MAB samples at most 20 arms, further limited by budget and candidate-space size, and revisits them; equal call counts do not guarantee equal exploration. Its repeated deterministic evaluations should not be described as new performance observations. Test small spaces exhaustively to validate surrogate references; label large-space best-found references accordingly.

Freeze selections, then execute each unique selection on the saved audit reservation. Add the best individual member as a comparator. If parameter averaging and output ensembling are both investigated, implement and label them as different operators; the current aggregation results do not establish ensemble behaviour. Do not tune the surrogate or repeat search using the audit outcomes of the same test requests.

Record measured quality and latency, quality change from the best individual, failures, signed surrogate errors and mean absolute errors. Measure constraint violations from the applicable observed quantity. Report inherited utility dimensions separately from newly measured aggregate properties. Use objective differences rather than ratios when reference values can be zero or have unsuitable signs. A best-found surrogate reference is not an oracle for measured quality.

To examine complementarity, collect aligned member predictions on development rows and measure disagreement and joint error. Relate these to aggregate outcomes on distinct audit rows. Similar accuracy, runtime or parameter direction alone does not establish useful complementary errors. This analysis requires additional prediction artifacts and is not claimed as an existing dataset label.

Expected outputs: family eligibility report, request file, frozen search results, bounded/exhaustive reference labels, aggregation provenance, per-selection measurements, failures and grouped comparisons.

## Extensions that need new observations

General orchestration needs workflow topology, typed intermediate inputs/outputs, dependency constraints, operator semantics, scheduling/resource context and end-to-end traces. The current family/subset structure does not supply these simply by being called collaborative.

Temporal QoS prediction needs repeated time-stamped observations under identified environments, loads and locations. Deployment availability needs operational uptime/failure measurements and observation windows. User recommendation needs preference or relevance observations beyond synthetic constraints if claims concern human satisfaction. Treat each as a dataset extension, not an already validated use.
