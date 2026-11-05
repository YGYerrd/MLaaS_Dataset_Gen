# Versioned trust and explanation evaluation

Service generation enables `benchmark_evaluation_v1` with the `benchmark`
profile by default: at most 32 trust inputs with 4 trials each, and at most 16
explanation inputs with 4 random controls per masking amount and replacement. The existing
`diagnostic`, `pilot` and `production` presets retain `production_evaluation_v1`
when selected explicitly through the manifest CLI.
Existing stored values without a version remain `legacy_unversioned` when read
by composition. The previous evaluator can be selected explicitly with
`evaluation_protocol_version=legacy_perturbation_v1`.

| Profile | Trust inputs × trials | Explanation inputs × candidate groups | Random controls per mask amount and replacement | Trust / explanation runtime budget |
| --- | --- | --- | --- | --- |
| benchmark (default) | 32 × 4 | 16 × 32 | 4 | 600 / 1,800 seconds |
| diagnostic | 8 × 4 | 4 × 8 | 4 | 30 / 30 seconds |
| pilot / production | 512 × 12 | 128 × 32 | 16 | 3,600 / 7,200 seconds |

These are maximum evaluation budgets, not required successful-input counts or
guarantees of sufficient evidence. Candidate
groups are limited by the available input. Separate configuration keys override
the new budgets: `trust_sample_count`, `trust_trials`,
`explainability_sample_count`, `explainability_candidate_units`, and
`metric_evaluation_random_trials`. Old `perturbation_*` sample limits do not
silently shrink a pilot. `metric_evaluation_budget_fractions` defaults to
`[0.1, 0.25, 0.5]`. Benchmark validation defaults to empirical replacement only;
`explanation_replacement_methods=["empirical", "mean"]` explicitly enables the
second mean/mask replacement. Diagnostic, pilot and production presets retain
both methods. The pilot reassessment
records the evidence for this runtime tradeoff. Occlusion ranking remains empirical.

For `benchmark_evaluation_v1`, each stage uses
`effective_target = min(configured_target, eligible_independent_inputs)`.
Select one row per independent source/document/patient component, without
replacement. Repeated source IDs cannot inflate evidence even if their supplied
group IDs differ. Eligibility is frozen from the held-out pool before inference;
subsequent unsupported outputs, invalid interventions and failures are recorded
as exclusions/failures and do not reduce that denominator. No row is duplicated
to reach a requested budget.

The trust evidence floor is **16 independent successful inputs**; explanation
requires **8**. These floors do not shrink with the available pool. For example,
18 eligible inputs give trust a target of 18 and explanation a target of 16;
a pool of 8 can meet the explanation floor but cannot meet the trust floor.
`trust_min_independent_inputs` and `explainability_min_independent_inputs`
override the separate floors. The historical
`metric_evaluation_min_independent_inputs` override applies to both unless a
metric-specific override is present.

Benchmark evidence also requires **80% input coverage and 80% trial coverage**.
Input coverage is successful independent inputs divided by the frozen effective
target; trial coverage is successful trials divided by effective target times
planned trials per input. `metric_evaluation_min_coverage` overrides both coverage
thresholds with a value in `(0, 1]`. Failed, excluded and unattempted inputs/trials
stay in these denominators. Thus 31/32 or 28/32 successful trust inputs can pass;
12/18 fails both the 16-input floor and coverage. No additional inputs are sampled
to replace failures beyond the frozen budget.

Python callers and manifest generation share the same profile resolver. For
manifest generation use `--metric-evaluation-profile benchmark` (the default), or set
`metric_evaluation_profile` on an individual service. The enable switch remains
`enable_perturbation_metrics` (with `explainability_enabled` as a compatibility
alias). Enabling this switch executes both independent stages. Explicitly
disabled rows and smoke tests remain disabled.

## Measurements and evidence

Trust compares complete task outputs: class probabilities and agreement,
regression changes against a frozen target scale, valid sequence labels,
segmentation masks and per-class overlap, matched detections, generated
sequences, or rankings over an explicitly shared candidate pool. Unsupported
contracts return missing scores with recorded reasons. Clean and perturbed
reference quality accompany stability; stability alone does not establish
correctness.

Image operators include noise, brightness, blur and JPEG compression, with
strength and actual-change evidence. Normalized tensors require their frozen
normalization contract. Numeric operators use feature scales and bounds while
preserving declared categorical, integer, one-hot and immutable features and
checking supplied linear constraints. Dataset-specific validity constraints
must be supplied; they cannot be inferred from feature scales. Text operating
perturbations require validated transformations with provenance. Arbitrary token
masking is used only as an explanation intervention, never assumed benign.

The explanation method is identified in each report. Occlusion rankings are
built from input-only candidate groups, independently of reference relevance
annotations. Fresh validation seeds evaluate both deletion and retention against
random and reversed controls across several masking amounts and the configured replacements.
The headline faithfulness score reports positive mean control advantage; it is
an experimental operational measure requiring task-specific validation.
Semantic alignment is a separate relevance-proxy diagnostic with provenance.
Random and reverse-occlusion methods are available as sanity controls via
`metric_evaluation_explanation_method`.

Reports contain means, bootstrap 95% intervals, lower-tail scores, class/subgroup
and operator/strength summaries, clean/perturbed quality, and requested,
attempted, successful, failed and excluded counts. Each stage also retains the
configured target, eligible independent count, effective target and evidence
floor. `trust_independent_inputs` and `explainability_independent_inputs` are
stored alongside scores in SQLite and the composition catalogue. Bootstrap units are inputs or
explicit document/patient groups, never individual perturbations. Frozen source
IDs, data/sample hashes, live model identity, seed, calibration and evaluation
definitions identify the measurement and protect resume from incompatible work.

`trust_status` and `explainability_status` are `complete`,
`insufficient_evidence`, `unsupported`, or `failed`. Benchmark evidence gates
require the independent-unit floor (16 trust / 8 explanation), both coverage
thresholds, verified model weights, no stage error, and a 95% interval no wider
than **0.15**. Explanation precision uses the signed control-gain interval before
clipping scores. Finishing every budgeted input is not an additional gate: a
deadline may leave sufficient evidence, with `time_budget_exhausted=true` still
recorded. Otherwise the measured mean remains diagnostic and evidence is false.
The older protocol retains its requested successful-input budget and stage
completion requirements (32 independent units for diagnostic / 128 for larger
profiles). Counts and precision requirements are frozen before measurement.
Missing scores are stored as JSON null, including clearing old values.

Each summary records `insufficiency_reasons` and `insufficiency_details` with
observed counts, coverage denominators, thresholds and CI width. Reasons
distinguish `too_few_independent_inputs`, `sample_coverage_below_minimum`,
`trial_coverage_below_minimum`, and `precision_requirement_not_met`; the precision
detail distinguishes an unavailable interval from an overly wide interval.
`exclusion_reasons` and `failure_reasons` retain counts of concrete intervention
or adapter problems. Top-level `trust_insufficiency_reasons` and
`explainability_insufficiency_reasons` are persisted alongside the scores.
Detailed service evidence is stored as a compressed, content-addressed
`.json.gz` artifact with its SHA-256 and size. SQLite keeps the aggregate metrics
and artifact reference, avoiding thousands of trial records embedded in each
artifact metadata row.

The service terminal summary includes a `Trust / Explainability` block with
`trust_score` and `explainability_score`, each stage's status, elapsed time,
independent `n`, input coverage, and any insufficiency/exclusion reasons.
Separate `trust_duration_s` and `explainability_duration_s` metrics are persisted.
Stage times include prior work when resuming saved progress; the terminal labels
this explicitly. Missing scores and unavailable times display as `n/a`.

## Recovery and composition

Each stage has its own clock. A failed or timed-out explanation stage cannot
remove trust measurements. `metric_evaluation_progress_path` enables atomic
progress snapshots, throttled to roughly two-second intervals and flushed at
stage boundaries; normal service execution assigns a path under the service
artifact root. Repeating the same model/sample/definition plan resumes saved
work. Runtime windows may be extended; changing the frozen evidence plan or
model requires a new progress file. Inference batches are used where the adapter
supports them. A single in-flight model call can finish after a stage deadline.

**An existing manifest does not need to be regenerated to evaluate unfinished
services with the benchmark preset.** Stop its current process normally and
resume the same command with `--metric-evaluation-profile benchmark`, retaining
the same file, database, worker and device options. For example:

```powershell
.venv\Scripts\python.exe -m mlaas_data_generator.cli.main run-manifest `
  --file outputs/current_manifest.csv --db outputs/services.db `
  --metric-evaluation-profile benchmark
```

The explicit flag replaces frozen metric budgets and progress paths while
preserving training settings, evaluation seeds and service IDs. Use it on each
resume of that original manifest. Restarting without it retains explicit or
frozen old settings. Completed service IDs are skipped, including GPU database
shards; their old scores are not backfilled. The reconstruction harness below
can measure completed recipe-enabled services separately. `--rerun-existing`
re-executes services and is not needed for this evaluation-only pilot.
New benchmark progress paths include the protocol and frozen plan hash, so
earlier evaluations cannot be mistaken for a resumed benchmark run. The final
coverage gate revision (`evidence_gate_version=coverage_precision_v1`), minimum
coverage and replacement list are included in that hash. A manually supplied
progress path from an incompatible plan must be changed; it is never silently
reused. Historical pilot scores and catalogue rows are not rewritten.

Composition preserves missing trust and explainability values instead of
assigning 0.5. Request either dimension through its positive preference weight,
QoS/hard minimum, or `evaluation_requirements`. Every selected member must have
a finite score, `benchmark_evaluation_v1` by default, `complete` status, and a true
`*_evidence_sufficient` flag for each requested dimension. Explicit
`evaluation_requirements.allow_legacy=true` supports historical experiments
with measured legacy scores; it never fills missing values or accepts a failed
current evaluation. Trust has its own preference weight and constraints.
Use `evaluation_requirements.protocol_version=production_evaluation_v1` to
request the previous protocol explicitly; versions are not mixed in one request.

## Historical reconstruction pilot

The September 2026 one-off pilot harness and its generated outputs were removed
during repository cleanup. The versioned evaluation contracts above remain in
production code and the current regression suite.
