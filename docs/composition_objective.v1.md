# Collaborative composition objective v1

`collaborative_objective.v1` is the frozen surrogate objective implemented by
`mlaas_composition.sampler.calculate_utility`. It applies only to collaborative
groups; it is not an oracle for deployed-service or ensemble quality.

For a selected group, let `d_i` be the eight normalized, higher-is-better
dimension scores and `w_i >= 0` their request weights. The unpenalized score is
`sum_i(w_i d_i) / sum_i(w_i)`. If all weights are zero, all eight weights are
set to one. The QoS penalty is the weighted mean of squared one-sided relative
violations: for a minimum target it is `max(0, target - actual) / |target|`,
and for a maximum target it is `max(0, actual - target) / |target|`. The final
utility is `clip(unpenalized_score - penalty, 0, 1)`.

The dimensions are data-distribution compatibility, update-signature alignment,
compute-time homogeneity, latency feasibility, metric-score homogeneity,
reliability homogeneity, mean faithfulness-oriented explainability score, and
one minus mean resource-cost score. Each dimension and its direction is listed
in `mlaas_composition/assets/objective.v1.json`.

## Worked example

Suppose only accuracy and latency have non-zero equal weights. A group has
`accuracy_compatibility = 0.90` and `latency_feasibility = 0.80`, giving an
unpenalized score of `(0.90 + 0.80) / 2 = 0.85`. Its mean accuracy is `0.82`
against `min_accuracy = 0.80`, so it has no accuracy violation. Its total
latency is `1.20` against `max_latency = 1.00`, yielding a latency violation of
`(1.20 - 1.00) / 1.00 = 0.20`; squared, this is `0.04`. If latency is the only
active QoS penalty, final utility is `0.85 - 0.04 = 0.81`.

The exact same calculation is covered by an automated regression test before a
release is produced.
