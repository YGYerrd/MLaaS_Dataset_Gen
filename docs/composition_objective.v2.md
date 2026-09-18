# Collaborative composition objective v2

`collaborative_objective.v2` is the current surrogate objective implemented by
`mlaas_composition`. FedAvg produces service observations and provenance; it is
never the composition mechanism. Composition remains downstream subset
selection over completed services.

Requests keep `composition_type="collaborative"` and select either
`compatibility_mode="functional"` or `compatibility_mode="strict"`. Strict
requests pin one FedAvg compatibility contract and parent-state snapshot, then
freeze the final eligible service IDs with `eligible_pool_snapshot_hash`.
Evaluation aborts before search if that membership hash no longer matches.

## Data-parameter evidence

Data compatibility averages pairwise agreement over logical-dataset,
preprocessing/input, and label/output contracts. Each comparison uses exact
provenance hashes first, complete legacy fields second, and neutral `0.5` only
when neither source can decide. The emitted evidence JSON records exact and
legacy matches/mismatches plus indeterminate counts. Split strategy, partition
identity, and data quantity are deliberately excluded because they represent
intended service heterogeneity.

## Resource score

`resource_cost_score` is already normalized and higher-is-better. Therefore
`cost_efficiency` is its group mean, hard and composition-level requirements use
`min_resource_cost_score`, and no second inversion is applied.

The normalized weighted mean and squared one-sided QoS
penalty retain the v1 structure. V1 request files continue to select the frozen
`collaborative_objective.v1` implementation for reproducibility.

## Trust and explanation evidence

Trust has its own `trust` preference weight, `min_trust` QoS bound and
`min_trust_score` hard bound. Missing trust/explainability scores stay missing.
An unweighted missing dimension does not affect utility; requesting either
dimension requires a measured score for every selected service.

Eligibility also requires `benchmark_evaluation_v1` by default, a `complete` metric status
and a true evidence-sufficiency flag. The request can declare dimensions in
`evaluation_requirements` in addition to weights/bounds. Explicit
`allow_legacy=true` permits historical measured scores for experiment replay,
without accepting missing scores or failed current evaluations. Generated
synthetic requests include only dimensions supported throughout their compatible
pool. See the [evaluation protocol](metric-evaluation-protocol.md).
The previous protocol is available through an explicit
`evaluation_requirements.protocol_version=production_evaluation_v1`; requests
never mix protocol versions. Independent sample counts accompany both scores.
