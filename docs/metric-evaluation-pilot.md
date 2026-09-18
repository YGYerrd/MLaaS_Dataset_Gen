# Evaluation protocol pilot — 16 September 2026

Implemented `production_evaluation_v1` and executed the larger requested budgets on 12 reconstructed model/dataset profiles. Original catalogue metrics remained unchanged. The production profile remains opt-in; none of these bounded real-model runs provides complete evidence for both metrics.

The requested plan was 512 trust inputs × 12 perturbations and 128 explanation inputs × up to 32 candidate groups, with 16 random controls per masking budget and replacement. Three masking budgets and two replacements produce 96 matched validation sets per explanation input, each with ranked/random/reversed deletion and retention comparisons.

Each real-model stage received a 60-second trust or 120-second explanation window. An in-flight inference can finish beyond that window. These are throughput and evidence pilots, with partial scores explicitly marked insufficient. CPU FP32 used four threads per process, two main workers, and one overlapping calibrated image rerun. Timings include local resource contention and are not isolated hardware benchmarks.

| Profile | Available held-out rows | Trust inputs with scores | Explanation inputs with scores | Trust status | Explanation status |
| --- | ---: | ---: | ---: | --- | --- |
| MLP / digits | 180 | 172 / 512 | 20 / 128 | insufficient_evidence | insufficient_evidence |
| Logistic regression / wine | 18 | 12 / 512 | 18 / 128 | insufficient_evidence | insufficient_evidence |
| MLP / diabetes | 45 | 40 / 512 | 26 / 128 | insufficient_evidence | insufficient_evidence |
| TinyBERT / AG News | 2022 | 0 / 512 | 5 / 128 | unsupported | insufficient_evidence |
| DeiT tiny / beans | 64 | 22 / 512 | 1 / 128 | insufficient_evidence | insufficient_evidence |
| TinyStories causal LM | 2044 | 0 / 512 | 0 / 128 | unsupported | unsupported |
| TinyBERT masked LM / AG News | 2044 | 0 / 512 | 3 / 128 | unsupported | insufficient_evidence |
| TinyBERT / MRPC | 400 | 0 / 512 | 9 / 128 | unsupported | insufficient_evidence |
| TinyBERT / STS-B | 1492 | 0 / 512 | 9 / 128 | unsupported | insufficient_evidence |
| SegFormer-B0 / ADE20K-mini | 322 | 153 / 512 | 7 / 128 | insufficient_evidence | insufficient_evidence |
| FLAN-T5-small / XSum | 2047 | 0 / 512 | 1 / 128 | insufficient_evidence | insufficient_evidence |
| TinyBERT / WikiANN | 2047 | 0 / 512 | 11 / 128 | unsupported | insufficient_evidence |

An input with a score can still have an unfinished trial budget; each JSON report also records completed input counts, attempted/successful/failed/excluded trials, confidence intervals, lower-tail scores, and per-operator/replacement/subgroup evidence.

The pilot found several deployment limits. Several frozen datasets have fewer than 512 eligible held-out rows, which cannot be repaired by repeating examples. Text trust needs validated meaning-preserving transformations. The causal-LM cohort lacks a prompt boundary for independent generation evaluation, so its inputs are excluded without exposing reference tokens to the model. Explanation evaluation needs substantially longer windows for the larger models; progress is retained for continuation. WikiANN also excluded 22 attempted explanation inputs when empirical replacements changed valid-token alignment; these exclusions identify a replacement-method limitation rather than successful evidence.

A separate full-budget synthetic pilot completed six controls and passed all eight directional checks. Known-feature faithfulness was 0.405, random ranking 0.018, reversed ranking 0, and the constant predictor 0. The constant predictor had trust 1 despite poor quality; the fragile predictor had trust 0.506. Model randomization changed the explanation ranking. These checks establish expected directional behavior, not universal scientific validation of the metric.

Validation: **777 tests passed** across the full repository suite. The replay compatibility fix only accommodates legacy sampling metadata; source bytes, selected rows, preprocessing, model, parameter schema, and frozen evaluation seed still must match.

- [Consolidated evidence and per-profile report references](../outputs/metric_evaluation_pilot_summary_20260916.json)
- [Full-budget synthetic control evidence](../outputs/metric_evaluation_sanity_20260916/report.json)
- [Protocol, configuration and resume instructions](metric-evaluation-protocol.md)
- [Real reconstruction pilot harness](../scripts/pilot_metric_evaluation.py)
- [Synthetic control harness](../scripts/pilot_evaluation_controls.py)
