# Task capability and composition extension plan

Status: implemented, 11 September 2026. Token classification, semantic segmentation
and seq2seq summarisation have completed the scoped independent and shared-parent
smoke gates. See [task capabilities](task-capabilities.md) for the implementation
and [smoke results](task-smoke-results.md) for evidence and runtime exclusions.
The workstreams below retain the original design and acceptance criteria.

The intended outcome is one capability resolver and one versioned task contract
used throughout generation and measured composition. Extend token classification
first, semantic segmentation second, and seq2seq summarisation third. A task is
enabled only for the task/backend/protocol combinations that have passed the
relevant checks; a registry entry alone is insufficient evidence.

## 1. Starting point and implementation boundaries

The current independent workflow freezes one initial checkpoint per family.
The optional shared-parent workflow starts clients from a trained task parent.
Both need the new contracts. Historical FedAvg records retain their existing
interpretation.

| Existing area | Relevant behavior and planned change |
| --- | --- |
| `hf_tasks.py`, `services/taxonomy.py`, `task_contract.py` | Task aliases, metric names and sentence-pair semantics exist separately. Resolve them into an explicit dataset-backed task contract. |
| `models/adapters/hf_task.py` | Task adapters already implement training, losses and metrics for tokens, segmentation and generation. Reuse them behind the shared contract. |
| `fedavg_eligibility.py`, `parent_selection.py` | Parent selection relies on a classification-oriented eligibility list and metric defaults. Replace active workflow decisions with capability and metric resolution; preserve historical entry points. |
| `cli/manifest/preparation.py`, `shared_parent.py`, `provenance.py` | Freeze model/data state and evaluation reservations. Add task semantics and evaluation protocol identities without weakening existing checks. |
| `clients/partitioning.py`, `clients/training.py`, `clients/policies.py` | Already separate sampling and training policy. Ask the task contract which units, profiles and supervision rules are valid. |
| `behavioural_diagnostics.py` | Supports classification, aligned tokens and scalar regression. Share prediction alignment and task metrics with composition. |
| `mlaas_composition/fedavg_validation.py`, `regeneration.py`, `validation.py`, `validation_cli.py` | Restrictions occur in member resolution, reconstruction, prediction shape, target loading, metrics and output reports. All need coordinated changes. |
| `mlaas_composition/protocol.py`, `pool.py` and protocol assets | Existing request schemas include v1-v4, including independent-family requests. Bind new task contracts and capabilities while retaining old readers. |

Paths above are relative to `mlaas_data_generator/` unless explicitly qualified.
The core parameter aggregator in `fedavg_aggregation.py` remains shared.

## 2. Workstream A: explicit capabilities

Introduce `mlaas_data_generator/task_capabilities.py` with a small typed registry
and resolver. Keep capability declarations separate from eligibility checks on a
particular model, dataset, request or checkpoint.

Resolve against canonical task, task variant, backend/model adapter, training
regime, generation mode and declared dataset semantics. Model/dataset/runtime
restrictions refine that answer. Unknown combinations receive an explicit reason
instead of inheriting a classification default.

Report these six capabilities independently:

| Capability | What must be specified |
| --- | --- |
| `training` | Trainable adapter, objective, valid supervision and successful-update accounting. |
| `checkpoint_restoration` | Capture/restore mechanism, required support artifacts and verified state identity. |
| `partitioning` | Source sampling unit, supported profiles and available skew axes. IID support does not imply shard or Dirichlet support. |
| `parameter_aggregation` | Operator identity and prerequisites: common parent, aligned tensor schema and compatible task semantics. |
| `diagnostics` | Available task metrics, prediction alignment and pairwise comparisons. Unsupported comparisons are explicit. |
| `measured_composition_validation` | Reconstruct, aggregate, predict and measure on a verified frozen audit protocol. |

Each declaration records `supported`, `experimental` or `unsupported`, its
implementation version, constraints and evidence references. Evidence distinguishes
code/unit coverage from a real backend smoke run. An instance decision separately
records `eligible`, `ineligible` or `pending`, with stage, stable reason code,
human-readable explanation and remediation. For example, a restorable backend
with a missing checkpoint is instance-ineligible; its backend capability remains
implemented.

Initial reason codes should cover `unknown_task_contract`,
`training_not_implemented`, `inference_only`, `checkpoint_missing`,
`checkpoint_integrity_failed`, `parent_mismatch`, `parameter_schema_mismatch`,
`target_semantics_mismatch`, `partition_axis_unavailable`,
`diagnostics_not_implemented`, `validation_not_implemented`,
`evaluation_protocol_mismatch` and `insufficient_audit_support`. Wrap existing
known-bad pair restrictions in structured reasons without silently removing them.

Integrate decisions at four stages:

1. Offline manifest preflight reports declared capabilities and pending checks.
2. Loaded preparation validates actual targets, partition support and model state.
3. Publication records completed training and verified checkpoint evidence.
4. Composition preflight rechecks selected members, artifacts and protocol.

Expose the matrix through an offline CLI report and include decisions in manifest
preflight and validation outputs. A new request profile for measured composition
requires the relevant capabilities before freezing its eligible pool. Existing
functional search remains distinguishable from executable parameter composition.
Validation of an already selected subset must fail with reasons, never silently
drop a member or rerun selection.

**Acceptance:** every registered task/backend combination has an explicit answer
for all six capabilities; missing context stays pending; existing supported paths
retain their decisions; an unsupported path explains both the missing capability
and the affected stage. Static support claims cannot override runtime integrity
failures.

## 3. Workstream B: versioned task contracts

Create a small `mlaas_data_generator/tasks/` package containing shared contract
types, resolution and task-specific protocol implementations. Keep
`task_contract.py` as a compatibility facade for existing callers. This is an
internal interface, with the current HF adapters providing backend execution.

Separate serializable, hashable declarations from runtime callables:

| Contract component | Required contents |
| --- | --- |
| Target semantics | Output kind, label/vocabulary meaning, ignore rules, alignment and task variant. Reject disagreement between config and loaded data. |
| Sampling | Source unit, grouping identity, permitted profiles and training-only bucket construction. |
| Training | Objective and reduction, valid supervision count, optimizer-update feasibility and budget units. |
| Prediction | Structured batches aligned to source IDs; task-specific token positions, masks or decoded text; prediction protocol hash. |
| Metrics | Named metrics, required finite values, direction, units/range, reduction and quality-score transform. |
| Audit | Reservation/grouping algorithm, target artifact identity, evaluation unit and minimum usable support. |

Use runtime operations for target validation, bucket construction, supervision
counting, prediction, metric accumulation/finalisation and diagnostics. The
orchestrator retains data loading, checkpoint verification, reservations,
aggregation and reporting. Task code must not bypass lineage or audit checks.

Replace positional `primary`/`secondary` interpretation at the shared boundary
with named metric results. Compatibility adapters preserve existing adapter and
catalogue semantics. Explicitly check the token accuracy/F1 mapping and
sentence-pair classification/regression cases before moving callers. Parent
selection must obtain the metric and direction from the contract; new tasks
must never fall back to accuracy. Unsupported historical metric protocols remain
excluded with reasons.

Introduce structured prediction/target batches rather than requiring everything
to fit `ServicePrediction.probabilities` or a scalar-label NumPy array. Retain
legacy wrappers. Load frozen targets by source ID and content hash, with numeric
or JSON artifacts as appropriate; large masks stay outside reservation JSON.
Support incremental metric accumulation so segmentation and generation do not
require retaining full probability/logit tensors.

Freeze two identities: task semantics and evaluation protocol. The latter includes
metric implementation/reduction, postprocessing, decoding and scoring settings.
Include these in new preparation/provenance contracts and bind them in requests
that use the new protocol. Keep client quantities, seeds, optimizer choices and
sample memberships as recorded heterogeneity, outside the shared semantic
compatibility identity.

Version changed preparation, reservation and request formats explicitly. Keep
request v1-v4 and saved recipes on their existing readers and hashes; introduce
the next request version for the new required fields. Update version checks in
pool construction and checkpoint resolution together. A semantics change creates
new artifacts and identities. Never rehash historical artifacts in place or infer
missing historical alignment/decoding settings.

Preserve sample-weighted aggregation using persisted training-example counts for
these extensions: sentences for tokens, images for segmentation, source-target
examples for summarisation. Record supervised tokens/pixels separately for loss,
budgets and evaluation. Changing the aggregation weight unit would require a
separate operator version. Keep integer/boolean tensors and existing exclusions
under the central aggregation policy.

**Acceptance:** existing classification/regression fixtures retain their named
metrics, scores and saved-artifact behavior; new contract hashes change when
semantics change; structured targets round-trip without losing IDs; unsupported
task outputs fail explicitly. Historical calibration evidence remains scoped to
its task/protocol and is not borrowed automatically for new tasks.

## 4. Workstream C: token classification end to end

Start with single-label token classification using the current supervised-position
policy. Keep token multilabel capability explicitly unsupported in this rollout.

1. Freeze tokenizer/revision, label map, tokenisation/truncation rules, special
   token handling, subword alignment and ignore index. Preserve sequence
   boundaries and alignment metadata through preprocessing and reconstruction.
2. Partition whole source sequences, grouped by document when that relationship
   exists. Reuse the declared sentence-level entity-presence axis for skewed
   sampling; never sample independent tokens or treat padding as a class.
3. Preserve optimizer-update feasibility rules and also require actual supervised
   positions. An all-ignored batch must not create a successful-update claim;
   a client with no usable supervision fails with a concrete reason.
4. Route member eligibility, restored-model inference, target loading, diagnostics
   and measured validation through the token contract. Cover both independent
   frozen families and trained-parent cohorts.
5. Use supervised-token weighted F1 as primary quality and token accuracy as a
   secondary metric, explicitly matching the established catalogue definition.
   Accumulate sufficient statistics over the full evaluation partition rather
   than averaging batch F1. For an NER-specific profile, add separately named
   entity-span F1 only with an explicit BIO/BILOU decoding contract; do not rename
   token F1 as entity F1.
6. Preserve example-level audit membership and report sequence and supervised-token
   counts. Prediction/target positions must align for every sequence. Any
   document-derived chunks must stay within the same reservation partition.

Tests must include padding, ignored special/subword positions, variable lengths,
truncation, all-ignored input, reordered labels, changed tokenizer/alignment,
document overlap, uneven batch sizes and incorrect token counts. Verify task
metrics against hand-calculated fixtures, including F1 aggregation across batches.

**Acceptance:** complete the common smoke gate below; restored and aggregate token
models pass through the public generation/search/validation path; catalogue and
validation use the same named metric definition. A classification/regression
regression suite passes after removing the old task-specific restrictions.

## 5. Workstream D: semantic segmentation, then generation

### Semantic segmentation

Choose semantic segmentation as the first vision extension. Its existing adapter
already provides batch metric statistics, which can exercise structured targets
and bounded-memory evaluation through the new boundary. Detection remains a
separately reported capability until its own implementation and smoke gate pass.

1. Freeze processor/revision, image normalisation, label map, background/void
   handling, mask dtype, ignore index and train/evaluation spatial transforms.
2. Use the existing logits-resolution evaluation behavior as an explicitly named
   initial profile: nearest-neighbour target resizing, fixed resolution rules,
   and declared class inclusion/absent-class conventions. A future original-image
   benchmark protocol gets its own identity and metrics.
3. Sample whole images and group related crops by original image. Enable IID and
   quantity skew first. Exclude shard/Dirichlet requests until a declared
   image-level bucket policy exists; do not silently relabel an IID fallback.
4. Count images for aggregation and valid pixels for supervision. Require usable
   pixels after transforms. Stream per-class confusion statistics for macro IoU
   and Dice, and record exact averaging definitions. Stream pixel disagreement
   and per-class diagnostics over the frozen catalogue partition.
5. Preserve the full parameter/head layout even when an individual client sees
   only a subset of classes. Extend checkpoint quality verification and output
   artifacts to use the same segmentation metric protocol.

Tests cover variable image sizes, non-contiguous dataset label IDs, absent classes,
all-void masks, resizing, crop grouping, mismatched processors and aggregation of
statistics across uneven batches. Freeze an explicit absent-class/empty-mask rule;
do not let a batch boundary change the reported score.

**Acceptance:** one pinned model/dataset/protocol passes the common smoke gate on
the intended runtime, including bounded memory use. Report that scoped combination
as supported, with other segmentation paths retaining their own evidence status.

### Seq2seq summarisation

Use seq2seq summarisation as the first generation profile so the extension tests
both teacher-forced training and actual decoding. Causal LM, translation and
captioning remain separately declared capabilities requiring their own profiles
and verification; they do not inherit support from this smoke run.

1. Freeze tokenizer/vocabulary, special tokens, source/target fields, prompt/task
   prefix, truncation, label masking and decoder-start behavior. Require the same
   parent and tensor schema, including tied parameter handling on restoration.
2. Sample whole source-target examples; group derived examples by source document.
   Start with IID and quantity skew. Require nonzero target supervision after
   masking and truncation, and record examples, supervised tokens and updates.
3. Separate teacher-forced token loss from decoded-output metrics. Freeze maximum
   generated length, deterministic decoding settings, stop conditions, text
   normalisation, reference handling and metric implementation. Audit references
   are used only for scoring, never supplied as decoding inputs.
4. Reuse the current ROUGE implementation through a named, versioned profile:
   ROUGE-1 primary quality and ROUGE-2 secondary, with an explicit 0-1 scale and
   reduction. Report target-token-weighted loss separately. Do not imply that
   similarly named metrics from other implementations are interchangeable.
5. Add catalogue diagnostics for generated lengths, empty outputs, repetition
   and identical outputs across clients. Pairwise comparisons operate on the
   same prompts under the same decoding protocol. They describe diversity,
   rather than treating exact-string disagreement as an error measure.
6. Measure generation latency with input/output token counts and the frozen
   decoding protocol. Record one-model inference separately from aggregation,
   load and preprocessing time. Keep the current objective formula; provide
   explicit metric normalisation for its quality input.

Tests cover masked padding, empty/all-masked targets, special tokens, truncation,
variable lengths, source-document grouping, tokenizer changes, deterministic
decoding replay, premature stopping and protocol mismatches. Empty generated text
is a measured poor outcome when the model actually emitted EOS; missing generation
results or backend failures are execution errors. Do not invent a quality gate.

**Acceptance:** one pinned seq2seq summarisation family passes the common smoke gate
with actual decoded predictions and separately reported training/evaluation loss.
Resource limits and evidence apply to that combination, not every generation model.

## 6. Common promotion gate and evidence

Use fast synthetic fixtures for unit/integration checks and a separate opt-in real
backend smoke suite. A mocked successful run cannot certify runtime support.
Add the reusable smoke harness under `tests/smoke/`, with artifacts written to a
new run-specific output directory.

For each new profile:

1. Select one small existing registry model/dataset pair, verify its loaded schema,
   and pin revisions before execution. Fix seed 42, one family and four client
   recipes with differing quantities/training conditions. Choose feasible budgets
   before the run; do not replace failed or weak clients to manufacture success.
2. Freeze the train/catalogue/audit source contract before training or scoring.
   Use a small declared evaluation reservation with enough valid task supervision;
   any smoke subset is a separately identified protocol. Preserve complete source
   IDs and hashes. Tiny smoke results establish mechanics, not research accuracy.
3. Exercise manifest generation, dry run, preparation, training, checkpoint
   publication, reload, diagnostics, strict request generation, search and selected
   composition validation through public entry points. All four planned clients
   must complete to pass this promotion fixture; the selected subset has at least
   two members. An incomplete fixture remains failed/incomplete without altering
   the frozen roster.
4. Validate only the subset selected using catalogue evidence. The smoke harness
   separately measures the parent and one individual selected on catalogue data
   as audit baselines; it must not choose winners or tune recipes on audit results.
   Use both independent-family and shared-parent lifecycle coverage. If one path
   lacks a real smoke, its capability retains the narrower evidence status.
5. Verify checkpoint round trips and restored-model predictions, deterministic
   aggregation of the same fixed stored states, contributor weights and preserved
   non-aggregatable state. Use exact storage for the first smoke; reduced storage
   remains scoped to formats that pass their existing task-specific verification.
   Fresh training need not be bitwise reproducible across hardware.
6. Exercise resume without rewriting completed artifacts and negative cases for
   tampered tensors, parent/schema/contract mismatches, changed input or target
   content, audit overlap and changed eligible membership. Unsupported selections
   return structured reasons, with no model execution or subset mutation.
7. Save code/runtime versions, hardware, model/data revisions, task/protocol hashes,
   recipe manifest, reservations, service/checkpoint identities, sample and
   supervision counts, aggregate provenance, metrics, timings and exclusions.

Passing requires correct execution, finite required metrics and verified contracts.
An aggregate may underperform its parent or an individual client and still pass
mechanical support. Report that result honestly. Audit reservation within the new
pipeline does not establish unknown historical/pretraining exposure as absent;
retain the existing evidence qualification.

## 7. Delivery sequence

| Change | Dependency | Reviewable completion criterion |
| --- | --- | --- |
| 1. Capability inventory and resolver | None | Offline matrix, structured reasons, characterization tests and preflight integration. No additional task enabled. |
| 2. Shared task contracts | Change 1 | Versioned semantics/protocols, named metrics and legacy compatibility for current tasks. |
| 3. Composition execution boundary | Change 2 | Structured targets/predictions, task-dispatched measurement and reports, new request profile, unchanged integrity enforcement. Existing task regressions pass. |
| 4. Token classification | Change 3 | Token fixtures, diagnostics reuse and recorded real smoke evidence. Enable only verified combinations. |
| 5. Semantic segmentation | Change 4 | Spatial target/metric protocol, bounded-memory diagnostics and recorded real smoke evidence. |
| 6. Seq2seq summarisation | Change 5 | Decoding protocol, text diagnostics and recorded real smoke evidence. |

Run focused meaningful tests during each change, then the retained core suite for
cross-cutting integration changes. Restore relevant archived adapter tests when
they cover behavior being modified; keep new regression coverage in the active
suite. Update usage, generation policies and composition-validation documentation
with the actual capability report and scoped smoke results before promotion.

The project is complete for this plan when these three task profiles can traverse
generation through measured composition under explicit contracts, support/exclusion
decisions agree across commands, and historical artifacts remain readable without
silent reinterpretation. No architecture-mixing or additional composition operator
is needed for this rollout.
