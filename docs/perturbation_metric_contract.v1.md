# Perturbation metric contract v1

The perturbation stage emits two distinct concepts.

- `trust_score` is **perturbation-response stability**: the stability of a
  service's confidence and prediction under benign input perturbations. It is
  stored in the reliability domain and is not an interpretability metric.
- `explainability_score` is a faithfulness-and-alignment attribute. It is
  emitted only when the experiment can form an independent semantic target and
  compare the model's perturbation ranking against that target.

Semantic targets may come from configured relevant feature indices, token-label
spans, masked-token context, or labelled detection/segmentation regions.
Content-word and token-content heuristics are candidate units only; they are
not semantic targets and cannot by themselves enable `explainability_score`.

When no semantic target is available, the perturbation probe can still be
successful and record trust/stability diagnostics, but it writes
`explainability_supported_flag = false`,
`faithfulness_alignment_supported_flag = false`, and
`explainability_support_status = unsupported_no_semantic_target`. In that
case `explainability_score` is missing, never replaced by zero.

The ranking-only targeted-versus-random signal remains available as
`perturbation_ranking_selectivity_score`. It is a diagnostic and must not be
reported as explainability or faithfulness without semantic-target support.

For mixed samples, `explainability_support_status = partial_semantic_coverage`
and `explainability_semantic_supported_rate` must be reported with the score.
The validation suite covers known tabular feature, token, and image-region
targets and requires their alignment score to exceed a random ranking.
