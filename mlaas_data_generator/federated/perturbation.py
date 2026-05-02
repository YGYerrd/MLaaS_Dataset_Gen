from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PredictionProbe:
    prediction: Any
    confidence: float
    output_value: float | None = None
    scores: Any | None = None


def run_perturbation_stage(
    model,
    x_eval,
    y_eval=None,
    *,
    task_family: str | None = None,
    hf_task: str | None = None,
    config: dict | None = None,
    meta: dict | None = None,
    client_id: str | None = None,
    round_idx: int | None = None,
) -> dict:
    """Run a small post-evaluation perturbation probe and return aggregate metrics.

    The stage is deliberately best-effort. It never mutates model weights and
    returns a compact error metric instead of failing the client round.
    """
    config = config or {}
    if not _enabled(config):
        return {"perturbation_enabled_flag": False}

    start = time.time()
    sample_count = _sample_count(x_eval, y_eval)
    sample_limit = max(0, int(config.get("perturbation_sample_count", 1) or 0))
    if sample_count <= 0 or sample_limit <= 0:
        return {
            "perturbation_enabled_flag": True,
            "perturbation_supported_flag": False,
            "explainability_supported_flag": False,
            "perturbation_sample_count": 0,
            "perturbation_error": "no_eval_samples",
        }
    skip_reason = _heavy_multimodal_skip_reason(config, task_family=task_family, hf_task=hf_task)
    if skip_reason is not None:
        return {
            "perturbation_enabled_flag": True,
            "perturbation_supported_flag": False,
            "explainability_supported_flag": False,
            "perturbation_sample_count": 0,
            "perturbation_error": skip_reason,
            "perturbation_skip_reason": skip_reason,
            "perturbation_runtime_policy": "skip_heavy_multimodal",
            "perturbation_duration_s": float(time.time() - start),
            "perturbation_truncated_flag": False,
        }

    seed = _stable_seed(config.get("seed", 42), client_id, round_idx, task_family, hf_task)
    rng = np.random.default_rng(seed)
    selected = _select_indices(sample_count, min(sample_limit, sample_count), rng)

    target_units = max(1, int(config.get("perturbation_target_units", 1) or 1))
    candidate_limit = max(1, int(config.get("perturbation_candidate_units", 4) or 4))
    trust_trials = max(1, int(config.get("perturbation_trust_trials", 2) or 2))
    random_trials = max(1, int(config.get("explainability_random_trials", trust_trials) or trust_trials))
    strength = float(config.get("perturbation_random_strength", 0.02) or 0.02)
    budget_fractions = _budget_fractions(config.get("explainability_budget_fractions"))
    meaningful_drop_threshold = _config_float(config, "explainability_meaningful_drop_threshold", 0.2)
    selectivity_floor = _config_float(config, "explainability_selectivity_floor", 0.5)
    progress_interval = _progress_sample_interval(config)
    candidate_limit, trust_trials, random_trials, budget_fractions, max_duration_s, runtime_adjustments = (
        _resolve_runtime_limits(
            config,
            task_family=task_family,
            candidate_limit=candidate_limit,
            trust_trials=trust_trials,
            random_trials=random_trials,
            budget_fractions=budget_fractions,
        )
    )
    deadline = (start + float(max_duration_s)) if max_duration_s is not None else None
    truncated = False
    truncation_reason = None

    def _mark_truncated(reason):
        nonlocal truncated, truncation_reason
        truncated = True
        if truncation_reason is None:
            truncation_reason = str(reason)

    _progress_log(
        config,
        "stage starts | "
        f"client={client_id or 'global'} | round={round_idx if round_idx is not None else 'n/a'} "
        f"| task={task_family or 'unknown'} | hf_task={hf_task or 'n/a'} "
        f"| samples={len(selected)}/{sample_count} | candidate_units={candidate_limit} "
        f"| trust_trials={trust_trials} | random_trials={random_trials} | budgets={budget_fractions} "
        f"| time_budget_s={max_duration_s}",
    )
    if runtime_adjustments:
        _progress_log(config, f"runtime safety adjustments | {' | '.join(runtime_adjustments)}")

    per_sample = []
    errors = []
    for sample_number, idx in enumerate(selected, start=1):
        if _deadline_exhausted(deadline):
            _mark_truncated("time_budget_exhausted_before_sample")
            break
        sample_start = time.time()
        log_sample = _should_log_sample(sample_number, len(selected), progress_interval)
        try:
            if log_sample:
                _progress_log(
                    config,
                    f"sample {sample_number}/{len(selected)} starts | eval_index={idx}",
                )
            x_sample = _get_sample(x_eval, idx)
            y_sample = _get_sample(y_eval, idx) if y_eval is not None else None
            units = _meaningful_units(x_sample, y_sample, task_family=task_family, meta=meta, limit=candidate_limit)
            if not units:
                if log_sample:
                    _progress_log(
                        config,
                        f"sample {sample_number}/{len(selected)} skipped | reason=no_meaningful_units",
                    )
                continue

            baseline = _predict_probe(model, x_sample, y_sample, task_family=task_family, hf_task=hf_task)
            if baseline is None:
                if log_sample:
                    _progress_log(
                        config,
                        f"sample {sample_number}/{len(selected)} skipped | reason=baseline_probe_failed",
                    )
                continue
            baseline_quality, quality_metric = _task_quality(
                baseline,
                y_sample,
                baseline_probe=None,
                task_family=task_family,
                hf_task=hf_task,
            )

            if log_sample:
                _progress_log(
                    config,
                    f"sample {sample_number}/{len(selected)} candidate ranking starts | units={len(units)}",
                )
            ranking_start = time.time()
            scored_units = []
            for unit in units:
                if _deadline_exhausted(deadline):
                    _mark_truncated("time_budget_exhausted_during_ranking")
                    break
                perturbed = _apply_targeted_mask(
                    x_sample,
                    [unit],
                    task_family=task_family,
                    model=model,
                    meta=meta,
                )
                candidate = _predict_probe(model, perturbed, y_sample, task_family=task_family, hf_task=hf_task)
                if candidate is None:
                    continue
                scored_units.append(
                    (
                        _quality_degradation(
                            baseline_quality,
                            _task_quality(
                                candidate,
                                y_sample,
                                baseline_probe=baseline,
                                task_family=task_family,
                                hf_task=hf_task,
                            )[0],
                        ),
                        _confidence_drop(baseline, candidate),
                        _prediction_changed(baseline.prediction, candidate.prediction),
                        unit,
                    )
                )

            if not scored_units:
                if log_sample:
                    _progress_log(
                        config,
                        f"sample {sample_number}/{len(selected)} skipped | reason=no_scored_units "
                        f"| ranking_s={time.time() - ranking_start:.2f}",
                    )
                continue

            scored_units.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            ranked_units = [unit for _, _, _, unit in scored_units]
            expected_units, expected_source = _expected_units(
                x_sample,
                y_sample,
                task_family=task_family,
                hf_task=hf_task,
                meta=meta,
                model=model,
                config=config,
                candidate_units=units,
                limit=candidate_limit,
            )
            semantic_supported = bool(expected_units)
            if log_sample:
                _progress_log(
                    config,
                    f"sample {sample_number}/{len(selected)} candidate ranking ends | scored={len(scored_units)} "
                    f"| semantic_supported={semantic_supported} | semantic_source={expected_source or 'none'} "
                    f"| ranking_s={time.time() - ranking_start:.2f}",
                )
                _progress_log(
                    config,
                    f"sample {sample_number}/{len(selected)} budget scoring starts | budgets={len(budget_fractions)}",
                )
            budget_start = time.time()
            budget_scores = []
            self_faithfulness_scores = []
            semantic_behavior_scores = []
            semantic_sensitivity_scores = []
            semantic_selectivity_scores = []
            semantic_alignment_scores = []
            targeted_records = []
            for fraction in budget_fractions:
                if _deadline_exhausted(deadline):
                    _mark_truncated("time_budget_exhausted_during_budget_scoring")
                    break
                unit_count = _budget_unit_count(
                    len(ranked_units),
                    fraction,
                    fallback_units=target_units,
                )
                chosen_units = ranked_units[:unit_count]
                targeted_x = _apply_targeted_mask(
                    x_sample,
                    chosen_units,
                    task_family=task_family,
                    model=model,
                    meta=meta,
                )
                targeted = _predict_probe(model, targeted_x, y_sample, task_family=task_family, hf_task=hf_task)
                if targeted is None:
                    continue

                targeted_quality, _ = _task_quality(
                    targeted,
                    y_sample,
                    baseline_probe=baseline,
                    task_family=task_family,
                    hf_task=hf_task,
                )
                targeted_degradation = _quality_degradation(baseline_quality, targeted_quality)

                random_degradations = []
                for _ in range(random_trials):
                    if _deadline_exhausted(deadline):
                        _mark_truncated("time_budget_exhausted_during_random_trials")
                        break
                    random_units = _random_units(ranked_units, unit_count, rng)
                    random_target_x = _apply_targeted_mask(
                        x_sample,
                        random_units,
                        task_family=task_family,
                        model=model,
                        meta=meta,
                    )
                    random_target = _predict_probe(
                        model,
                        random_target_x,
                        y_sample,
                        task_family=task_family,
                        hf_task=hf_task,
                    )
                    if random_target is None:
                        continue
                    random_quality, _ = _task_quality(
                        random_target,
                        y_sample,
                        baseline_probe=baseline,
                        task_family=task_family,
                        hf_task=hf_task,
                    )
                    random_degradations.append(_quality_degradation(baseline_quality, random_quality))

                random_degradation = _mean_or_nan(random_degradations)
                if math.isnan(random_degradation):
                    random_degradation = 0.0

                self_faithfulness_score = _selectivity_score(targeted_degradation, random_degradation)
                semantic_degradation = np.nan
                semantic_random_degradation = np.nan
                semantic_sensitivity_score = np.nan
                semantic_selectivity_score = np.nan
                semantic_alignment_score = np.nan
                semantic_behavior_score = np.nan
                budget_score = self_faithfulness_score

                if semantic_supported:
                    semantic_unit_count = _budget_unit_count(
                        len(expected_units),
                        fraction,
                        fallback_units=target_units,
                    )
                    chosen_expected_units = expected_units[:semantic_unit_count]
                    semantic_x = _apply_targeted_mask(
                        x_sample,
                        chosen_expected_units,
                        task_family=task_family,
                        model=model,
                        meta=meta,
                    )
                    semantic_probe = _predict_probe(model, semantic_x, y_sample, task_family=task_family, hf_task=hf_task)
                    if semantic_probe is not None:
                        semantic_quality, _ = _task_quality(
                            semantic_probe,
                            y_sample,
                            baseline_probe=baseline,
                            task_family=task_family,
                            hf_task=hf_task,
                        )
                        semantic_degradation = _quality_degradation(baseline_quality, semantic_quality)

                        semantic_random_degradations = []
                        for _ in range(random_trials):
                            if _deadline_exhausted(deadline):
                                _mark_truncated("time_budget_exhausted_during_semantic_random_trials")
                                break
                            random_units = _random_units(units, semantic_unit_count, rng)
                            random_x = _apply_targeted_mask(
                                x_sample,
                                random_units,
                                task_family=task_family,
                                model=model,
                                meta=meta,
                            )
                            random_probe = _predict_probe(
                                model,
                                random_x,
                                y_sample,
                                task_family=task_family,
                                hf_task=hf_task,
                            )
                            if random_probe is None:
                                continue
                            random_quality, _ = _task_quality(
                                random_probe,
                                y_sample,
                                baseline_probe=baseline,
                                task_family=task_family,
                                hf_task=hf_task,
                            )
                            semantic_random_degradations.append(
                                _quality_degradation(baseline_quality, random_quality)
                            )
                        semantic_random_degradation = _mean_or_nan(semantic_random_degradations)
                        if math.isnan(semantic_random_degradation):
                            semantic_random_degradation = 0.0

                        semantic_sensitivity_score = _sensitivity_score(
                            semantic_degradation,
                            meaningful_drop_threshold,
                        )
                        semantic_selectivity_score = _selectivity_score(
                            semantic_degradation,
                            semantic_random_degradation,
                        )
                        semantic_alignment_score = _alignment_score(
                            ranked_units,
                            chosen_expected_units,
                        )
                        semantic_behavior_score = _semantic_behavior_score(
                            semantic_sensitivity_score,
                            semantic_selectivity_score,
                            selectivity_floor,
                        )
                        budget_score = _headline_explainability_score(
                            semantic_behavior_score,
                            semantic_alignment_score,
                            self_faithfulness_score,
                        )

                unit_fraction = float(len(chosen_units) / max(1, len(units)))
                budget_scores.append(budget_score)
                self_faithfulness_scores.append(self_faithfulness_score)
                semantic_behavior_scores.append(semantic_behavior_score)
                semantic_sensitivity_scores.append(semantic_sensitivity_score)
                semantic_selectivity_scores.append(semantic_selectivity_score)
                semantic_alignment_scores.append(semantic_alignment_score)
                targeted_records.append(
                    {
                        "budget_fraction": float(fraction),
                        "unit_count": int(len(chosen_units)),
                        "unit_fraction": unit_fraction,
                        "targeted": targeted,
                        "targeted_quality": targeted_quality,
                        "targeted_degradation": targeted_degradation,
                        "random_degradation": random_degradation,
                        "self_faithfulness_score": self_faithfulness_score,
                        "semantic_supported": _finite_or_none(semantic_behavior_score) is not None,
                        "semantic_target_source": expected_source if semantic_supported else "none",
                        "semantic_unit_count": int(_finite_or_none(semantic_unit_count) or 0)
                        if semantic_supported
                        else 0,
                        "semantic_degradation": semantic_degradation,
                        "semantic_random_degradation": semantic_random_degradation,
                        "semantic_behavior_score": semantic_behavior_score,
                        "semantic_sensitivity_score": semantic_sensitivity_score,
                        "semantic_selectivity_score": semantic_selectivity_score,
                        "semantic_alignment_score": semantic_alignment_score,
                        "score": budget_score,
                    }
                )

            if not targeted_records:
                if log_sample:
                    _progress_log(
                        config,
                        f"sample {sample_number}/{len(selected)} skipped | reason=no_targeted_records "
                        f"| budget_s={time.time() - budget_start:.2f}",
                    )
                continue
            final_targeted_record = targeted_records[-1]
            chosen_units = ranked_units[: int(final_targeted_record["unit_count"])]
            targeted = final_targeted_record["targeted"]
            if log_sample:
                _progress_log(
                    config,
                    f"sample {sample_number}/{len(selected)} budget scoring ends | records={len(targeted_records)} "
                    f"| score={float(_mean_or_nan(budget_scores)):.4f} | budget_s={time.time() - budget_start:.2f}",
                )
                _progress_log(
                    config,
                    f"sample {sample_number}/{len(selected)} trust perturbations starts | trials={trust_trials}",
                )
            trust_start = time.time()

            trust_changes = []
            trust_same = []
            trust_output_deltas = []
            for _ in range(trust_trials):
                if _deadline_exhausted(deadline):
                    _mark_truncated("time_budget_exhausted_during_trust_trials")
                    break
                random_x = _apply_benign_perturbation(
                    x_sample,
                    rng,
                    task_family=task_family,
                    model=model,
                    meta=meta,
                    strength=strength,
                )
                random_probe = _predict_probe(model, random_x, y_sample, task_family=task_family, hf_task=hf_task)
                if random_probe is None:
                    continue
                trust_changes.append(abs(_confidence_drop(baseline, random_probe)))
                trust_same.append(0.0 if _prediction_changed(baseline.prediction, random_probe.prediction) else 1.0)
                if baseline.output_value is not None and random_probe.output_value is not None:
                    denom = max(abs(float(baseline.output_value)), 1e-9)
                    trust_output_deltas.append(abs(float(random_probe.output_value) - float(baseline.output_value)) / denom)

            if not trust_same and not trust_changes and not trust_output_deltas:
                if log_sample:
                    _progress_log(
                        config,
                        f"sample {sample_number}/{len(selected)} skipped | reason=no_trust_records "
                        f"| trust_s={time.time() - trust_start:.2f}",
                    )
                continue

            targeted_drop = _confidence_drop(baseline, targeted)
            targeted_changed = _prediction_changed(baseline.prediction, targeted.prediction)
            unit_fraction = float(len(chosen_units) / max(1, len(units)))
            baseline_conf = _finite_or_none(baseline.confidence)
            relative_drop = (
                max(0.0, targeted_drop) / max(float(baseline_conf), 1e-9)
                if baseline_conf is not None
                else 0.0
            )
            output_delta = None
            if baseline.output_value is not None and targeted.output_value is not None:
                denom = max(abs(float(baseline.output_value)), 1e-9)
                output_delta = abs(float(targeted.output_value) - float(baseline.output_value)) / denom

            semantic_scored = bool(_finite_values(semantic_behavior_scores))
            semantic_source = expected_source if semantic_supported else "none"
            per_sample.append(
                {
                    "sample_index": int(idx),
                    "baseline_prediction": _jsonable_prediction(baseline.prediction),
                    "baseline_confidence": baseline_conf,
                    "targeted_confidence": _finite_or_none(targeted.confidence),
                    "targeted_confidence_drop": _finite_or_none(targeted_drop),
                    "targeted_prediction_changed": bool(targeted_changed),
                    "targeted_unit_fraction": unit_fraction,
                    "targeted_relative_drop": float(relative_drop),
                    "targeted_output_relative_delta": _finite_or_none(output_delta),
                    "baseline_quality": _finite_or_none(baseline_quality),
                    "targeted_quality": _finite_or_none(final_targeted_record["targeted_quality"]),
                    "targeted_quality_degradation": float(final_targeted_record["targeted_degradation"]),
                    "random_quality_degradation": float(final_targeted_record["random_degradation"]),
                    "explainability_self_faithfulness_score": float(_mean_or_nan(self_faithfulness_scores)),
                    "explainability_semantic_supported": semantic_scored,
                    "explainability_semantic_target_source": semantic_source,
                    "explainability_semantic_behavior_score": _mean_or_nan(semantic_behavior_scores),
                    "explainability_semantic_sensitivity_score": _mean_or_nan(semantic_sensitivity_scores),
                    "explainability_semantic_selectivity_score": _mean_or_nan(semantic_selectivity_scores),
                    "explainability_semantic_alignment_score": _mean_or_nan(semantic_alignment_scores),
                    "semantic_quality_degradation": _finite_or_none(final_targeted_record["semantic_degradation"]),
                    "semantic_random_quality_degradation": _finite_or_none(
                        final_targeted_record["semantic_random_degradation"]
                    ),
                    "explainability_budget_scores": [float(v) for v in budget_scores],
                    "explainability_self_faithfulness_budget_scores": [
                        float(v) for v in self_faithfulness_scores
                    ],
                    "explainability_semantic_behavior_budget_scores": [
                        float(v) for v in _finite_values(semantic_behavior_scores)
                    ],
                    "explainability_semantic_sensitivity_budget_scores": [
                        float(v) for v in _finite_values(semantic_sensitivity_scores)
                    ],
                    "explainability_semantic_selectivity_budget_scores": [
                        float(v) for v in _finite_values(semantic_selectivity_scores)
                    ],
                    "explainability_semantic_alignment_budget_scores": [
                        float(v) for v in _finite_values(semantic_alignment_scores)
                    ],
                    "explainability_score": float(_mean_or_nan(budget_scores)),
                    "explainability_quality_metric": quality_metric,
                    "trust_confidence_abs_delta_mean": _mean_or_nan(trust_changes),
                    "trust_prediction_same_rate": _mean_or_nan(trust_same),
                    "trust_output_relative_delta_mean": _mean_or_nan(trust_output_deltas),
                }
            )
            if log_sample:
                _progress_log(
                    config,
                    f"sample {sample_number}/{len(selected)} complete | "
                    f"targeted_drop={_finite_or_none(targeted_drop)} | "
                    f"semantic_drop={_finite_or_none(final_targeted_record['semantic_degradation'])} "
                    f"| trust_s={time.time() - trust_start:.2f} | sample_s={time.time() - sample_start:.2f}",
                )
        except Exception as exc:
            errors.append(type(exc).__name__)
            if log_sample:
                _progress_log(
                    config,
                    f"sample {sample_number}/{len(selected)} failed | error={type(exc).__name__} "
                    f"| sample_s={time.time() - sample_start:.2f}",
                )
            continue

    if not per_sample:
        reason = "unsupported_prediction_probe"
        if errors:
            reason = f"{reason}:{errors[0]}"
        _progress_log(
            config,
            f"stage ends unsupported | reason={reason} | duration_s={time.time() - start:.2f}",
        )
        return {
            "perturbation_enabled_flag": True,
            "perturbation_supported_flag": False,
            "explainability_supported_flag": False,
            "perturbation_sample_count": 0,
            "perturbation_error": reason,
            "perturbation_duration_s": float(time.time() - start),
            "perturbation_truncated_flag": bool(truncated),
            "perturbation_truncation_reason": truncation_reason,
            "perturbation_time_budget_s": max_duration_s,
        }

    targeted_drops = _values(per_sample, "targeted_confidence_drop")
    targeted_changes = [1.0 if item["targeted_prediction_changed"] else 0.0 for item in per_sample]
    relative_drops = _values(per_sample, "targeted_relative_drop")
    unit_fractions = _values(per_sample, "targeted_unit_fraction")
    output_deltas = _values(per_sample, "targeted_output_relative_delta")
    targeted_quality_degradations = _values(per_sample, "targeted_quality_degradation")
    random_quality_degradations = _values(per_sample, "random_quality_degradation")
    self_faithfulness_scores = _values(per_sample, "explainability_self_faithfulness_score")
    semantic_behavior_scores = _values(per_sample, "explainability_semantic_behavior_score")
    semantic_sensitivity_scores = _values(per_sample, "explainability_semantic_sensitivity_score")
    semantic_selectivity_scores = _values(per_sample, "explainability_semantic_selectivity_score")
    semantic_alignment_scores = _values(per_sample, "explainability_semantic_alignment_score")
    semantic_quality_degradations = _values(per_sample, "semantic_quality_degradation")
    semantic_random_quality_degradations = _values(per_sample, "semantic_random_quality_degradation")
    semantic_supported_rate = _mean_or_nan(
        [1.0 if item.get("explainability_semantic_supported") else 0.0 for item in per_sample]
    )
    trust_conf_delta = _values(per_sample, "trust_confidence_abs_delta_mean")
    trust_same = _values(per_sample, "trust_prediction_same_rate")
    trust_output_delta = _values(per_sample, "trust_output_relative_delta_mean")
    per_sample_explainability_scores = [_sample_explainability_score(item) for item in per_sample]
    per_sample_trust_scores = [_sample_trust_score(item) for item in per_sample]

    explainability_score = _mean_or_nan(per_sample_explainability_scores)
    if math.isnan(explainability_score):
        explainability_score = 0.0
    explainability_score = float(np.clip(explainability_score, 0.0, 1.0))
    quality_metrics = [
        str(item.get("explainability_quality_metric"))
        for item in per_sample
        if item.get("explainability_quality_metric")
    ]
    quality_metric = quality_metrics[0] if quality_metrics else _quality_metric_name(task_family, hf_task)

    confidence_stability = 1.0 - _mean_or_nan(trust_conf_delta)
    if math.isnan(confidence_stability):
        confidence_stability = 1.0 - _mean_or_nan(trust_output_delta)
    if math.isnan(confidence_stability):
        confidence_stability = 0.0
    confidence_stability = float(np.clip(confidence_stability, 0.0, 1.0))
    prediction_stability = _mean_or_nan(trust_same)
    if math.isnan(prediction_stability):
        prediction_stability = 0.0
    trust_score = float(np.clip((confidence_stability + prediction_stability) / 2.0, 0.0, 1.0))
    per_sample_trust_score_mean = _mean_or_nan(per_sample_trust_scores)
    if not math.isnan(per_sample_trust_score_mean):
        trust_score = per_sample_trust_score_mean

    result = {
        "perturbation_enabled_flag": True,
        "perturbation_supported_flag": True,
        "explainability_supported_flag": True,
        "explainability_task_family": str(task_family or "unknown"),
        "explainability_method": "semantic_sensitivity_faithfulness_v1",
        "explainability_quality_metric": quality_metric,
        "explainability_budget_fractions": budget_fractions,
        "explainability_meaningful_drop_threshold": meaningful_drop_threshold,
        "explainability_selectivity_floor": selectivity_floor,
        "perturbation_sample_count": int(len(per_sample)),
        "perturbation_baseline_confidence_mean": _mean_or_nan(_values(per_sample, "baseline_confidence")),
        "explainability_confidence_drop_mean": _mean_or_nan(targeted_drops),
        "explainability_confidence_drop_std": _std_or_nan(targeted_drops),
        "explainability_confidence_drop_p50": _percentile_or_nan(targeted_drops, 50),
        "explainability_confidence_drop_p10": _percentile_or_nan(targeted_drops, 10),
        "explainability_confidence_drop_p90": _percentile_or_nan(targeted_drops, 90),
        "explainability_prediction_change_rate": _mean_or_nan(targeted_changes),
        "explainability_unit_fraction_mean": _mean_or_nan(unit_fractions),
        "explainability_unit_fraction_p95": _percentile_or_nan(unit_fractions, 95),
        "explainability_targeted_degradation_mean": _mean_or_nan(targeted_quality_degradations),
        "explainability_random_degradation_mean": _mean_or_nan(random_quality_degradations),
        "explainability_self_faithfulness_score": _mean_or_nan(self_faithfulness_scores),
        "explainability_self_faithfulness_score_p10": _percentile_or_nan(self_faithfulness_scores, 10),
        "explainability_semantic_supported_flag": bool(semantic_supported_rate > 0.0)
        if not math.isnan(semantic_supported_rate)
        else False,
        "explainability_semantic_supported_rate": semantic_supported_rate,
        "explainability_semantic_target_source": _dominant_text(
            per_sample,
            "explainability_semantic_target_source",
        ),
        "explainability_semantic_degradation_mean": _mean_or_nan(semantic_quality_degradations),
        "explainability_semantic_random_degradation_mean": _mean_or_nan(semantic_random_quality_degradations),
        "explainability_semantic_behavior_score": _mean_or_nan(semantic_behavior_scores),
        "explainability_semantic_behavior_score_p10": _percentile_or_nan(semantic_behavior_scores, 10),
        "explainability_semantic_sensitivity_score": _mean_or_nan(semantic_sensitivity_scores),
        "explainability_semantic_sensitivity_score_p10": _percentile_or_nan(semantic_sensitivity_scores, 10),
        "explainability_semantic_selectivity_score": _mean_or_nan(semantic_selectivity_scores),
        "explainability_semantic_selectivity_score_p10": _percentile_or_nan(semantic_selectivity_scores, 10),
        "explainability_semantic_alignment_score": _mean_or_nan(semantic_alignment_scores),
        "explainability_semantic_alignment_score_p10": _percentile_or_nan(semantic_alignment_scores, 10),
        "explainability_score": explainability_score,
        "explainability_score_p10": _percentile_or_nan(per_sample_explainability_scores, 10),
        "trust_confidence_delta_mean": _mean_or_nan(trust_conf_delta),
        "trust_confidence_delta_std": _std_or_nan(trust_conf_delta),
        "trust_confidence_delta_p95": _percentile_or_nan(trust_conf_delta, 95),
        "trust_confidence_delta_max": _max_or_nan(trust_conf_delta),
        "trust_prediction_stability": float(np.clip(prediction_stability, 0.0, 1.0)),
        "trust_prediction_stability_min": _min_or_nan(trust_same),
        "trust_confidence_stability": confidence_stability,
        "trust_score": trust_score,
        "trust_score_p05": _percentile_or_nan(per_sample_trust_scores, 5),
        "trust_score_min": _min_or_nan(per_sample_trust_scores),
        "perturbation_duration_s": float(time.time() - start),
        "perturbation_truncated_flag": bool(truncated),
        "perturbation_truncation_reason": truncation_reason,
        "perturbation_time_budget_s": max_duration_s,
        "perturbation_samples": per_sample,
    }
    _progress_log(
        config,
        "stage ends | "
        f"client={client_id or 'global'} | round={round_idx if round_idx is not None else 'n/a'} "
        f"| samples={len(per_sample)} | explainability_score={result.get('explainability_score')} "
        f"| semantic_supported={result.get('explainability_semantic_supported_flag')} "
        f"| truncated={result.get('perturbation_truncated_flag')} "
        f"| duration_s={result.get('perturbation_duration_s'):.2f}",
    )
    return result


def _enabled(config):
    value = config.get("enable_perturbation_metrics", config.get("perturbation_enabled", True))
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _config_bool(config, key, default=False):
    value = (config or {}).get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _heavy_multimodal_skip_reason(config, *, task_family=None, hf_task=None):
    if _config_bool(config, "perturbation_allow_heavy_multimodal", False):
        return None
    family = str(task_family or "").strip().lower()
    hf = str(hf_task or "").strip().lower()
    if family in {"vqa", "retrieval"}:
        return "disabled_for_heavy_multimodal_task"
    if hf in {"visual_question_answering", "text_image_retrieval", "image_captioning"}:
        return "disabled_for_heavy_multimodal_task"
    return None


def _progress_enabled(config):
    value = (config or {}).get("perturbation_progress_logging", False)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _progress_log(config, message):
    if _progress_enabled(config):
        print(f"[Perturbation] {message}", flush=True)


def _progress_sample_interval(config):
    try:
        parsed = int((config or {}).get("perturbation_progress_sample_interval", 1) or 1)
    except Exception:
        parsed = 1
    return max(1, parsed)


def _should_log_sample(sample_number, total, interval):
    return sample_number == 1 or sample_number == total or sample_number % max(1, interval) == 0


def _config_float(config, key, default):
    try:
        return float(config.get(key, default))
    except Exception:
        return float(default)


def _budget_fractions(value):
    if value is None:
        return [0.05, 0.1, 0.2]
    if isinstance(value, str):
        pieces = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        pieces = list(value)
    else:
        pieces = [value]
    fractions = []
    for piece in pieces:
        try:
            parsed = float(piece)
        except Exception:
            continue
        if parsed > 1.0:
            parsed = parsed / 100.0
        if parsed > 0.0:
            fractions.append(float(min(1.0, parsed)))
    return fractions or [0.05, 0.1, 0.2]


def _resolve_runtime_limits(config, *, task_family, candidate_limit, trust_trials, random_trials, budget_fractions):
    budget_fractions = list(budget_fractions or [0.05, 0.1, 0.2])
    adjustments = []
    max_duration_s = _optional_positive_float(config.get("perturbation_max_duration_s"))
    family = str(task_family or "").strip().lower()

    if family in {"detection", "segmentation"}:
        candidate_limit, changed = _cap_int_value(
            candidate_limit,
            config.get("perturbation_detection_candidate_units_cap"),
        )
        if changed:
            adjustments.append(f"capped candidate_units={candidate_limit}")

        trust_trials, changed = _cap_int_value(
            trust_trials,
            config.get("perturbation_detection_trust_trials_cap"),
        )
        if changed:
            adjustments.append(f"capped trust_trials={trust_trials}")

        random_trials, changed = _cap_int_value(
            random_trials,
            config.get("perturbation_detection_random_trials_cap"),
        )
        if changed:
            adjustments.append(f"capped random_trials={random_trials}")

        budget_count_cap = _optional_positive_int(config.get("perturbation_detection_budget_count_cap"))
        if budget_count_cap is not None and len(budget_fractions) > budget_count_cap:
            budget_fractions = budget_fractions[:budget_count_cap]
            adjustments.append(f"capped budgets={len(budget_fractions)}")

        detection_max_duration_s = _optional_positive_float(config.get("perturbation_detection_max_duration_s"))
        if detection_max_duration_s is not None:
            new_limit = (
                detection_max_duration_s
                if max_duration_s is None
                else min(float(max_duration_s), float(detection_max_duration_s))
            )
            if max_duration_s != new_limit:
                adjustments.append(f"set time_budget_s={new_limit:g}")
            max_duration_s = float(new_limit)

    return candidate_limit, trust_trials, random_trials, budget_fractions, max_duration_s, adjustments


def _cap_int_value(value, cap):
    parsed_cap = _optional_positive_int(cap)
    if parsed_cap is None:
        return int(value), False
    value = max(1, int(value))
    capped = min(value, parsed_cap)
    return capped, capped != value


def _optional_positive_int(value):
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _optional_positive_float(value):
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _deadline_exhausted(deadline):
    return deadline is not None and time.time() >= float(deadline)


def _budget_unit_count(unit_count, fraction, *, fallback_units=1):
    unit_count = max(1, int(unit_count or 1))
    try:
        fraction = float(fraction)
    except Exception:
        fraction = 0.0
    requested = int(math.ceil(unit_count * max(0.0, fraction)))
    if requested <= 0:
        requested = int(fallback_units or 1)
    return max(1, min(unit_count, requested))


def _random_units(units, unit_count, rng):
    units = list(units or [])
    if not units:
        return []
    unit_count = max(1, min(len(units), int(unit_count or 1)))
    if unit_count >= len(units):
        return list(units)
    selected = rng.choice(len(units), size=unit_count, replace=False)
    return [units[int(i)] for i in selected]


def _quality_degradation(baseline_quality, perturbed_quality):
    base = _finite_or_none(baseline_quality)
    perturbed = _finite_or_none(perturbed_quality)
    if base is None or perturbed is None:
        return 0.0
    denom = max(abs(float(base)), 1e-9)
    return float(np.clip((float(base) - float(perturbed)) / denom, 0.0, 1.0))


def _faithfulness_score(targeted_degradation, random_degradation):
    return _selectivity_score(targeted_degradation, random_degradation)


def _selectivity_score(targeted_degradation, random_degradation):
    targeted = float(np.clip(_finite_or_none(targeted_degradation) or 0.0, 0.0, 1.0))
    random = float(np.clip(_finite_or_none(random_degradation) or 0.0, 0.0, 1.0))
    return float(np.clip((targeted - random) / max(1.0 - random, 1e-9), 0.0, 1.0))


def _sensitivity_score(targeted_degradation, threshold):
    targeted = float(np.clip(_finite_or_none(targeted_degradation) or 0.0, 0.0, 1.0))
    threshold = max(float(_finite_or_none(threshold) or 0.2), 1e-9)
    return float(np.clip(targeted / threshold, 0.0, 1.0))


def _semantic_behavior_score(sensitivity_score, selectivity_score, selectivity_floor):
    sensitivity = float(np.clip(_finite_or_none(sensitivity_score) or 0.0, 0.0, 1.0))
    selectivity = float(np.clip(_finite_or_none(selectivity_score) or 0.0, 0.0, 1.0))
    floor = float(np.clip(_finite_or_none(selectivity_floor) or 0.5, 0.0, 1.0))
    return float(np.clip(sensitivity * (floor + (1.0 - floor) * selectivity), 0.0, 1.0))


def _headline_explainability_score(semantic_behavior_score, semantic_alignment_score, self_faithfulness_score):
    semantic_behavior = _finite_or_none(semantic_behavior_score)
    semantic_alignment = _finite_or_none(semantic_alignment_score)
    self_faithfulness = float(np.clip(_finite_or_none(self_faithfulness_score) or 0.0, 0.0, 1.0))
    if semantic_behavior is None or semantic_alignment is None:
        return self_faithfulness
    return float(
        np.clip(
            (0.65 * float(semantic_behavior)) + (0.25 * float(semantic_alignment)) + (0.10 * self_faithfulness),
            0.0,
            1.0,
        )
    )


def _task_quality(probe, y_sample=None, *, baseline_probe=None, task_family=None, hf_task=None):
    metric = _quality_metric_name(task_family, hf_task)
    if probe is None:
        return None, metric

    family = str(task_family or "").strip().lower()
    if family == "regression":
        pred = _finite_or_none(probe.output_value if probe.output_value is not None else probe.prediction)
        target = _finite_or_none(_scalar_target(y_sample))
        if pred is not None and target is not None:
            return float(1.0 / (1.0 + abs(float(pred) - float(target)))), "inverse_absolute_error"
        if baseline_probe is not None:
            base = _finite_or_none(
                baseline_probe.output_value
                if baseline_probe.output_value is not None
                else baseline_probe.prediction
            )
            if pred is not None and base is not None:
                rel = abs(float(pred) - float(base)) / max(abs(float(base)), 1e-9)
                return float(1.0 / (1.0 + rel)), "prediction_stability"

    confidence = None
    if baseline_probe is not None:
        confidence = _confidence_for_prediction(probe, baseline_probe.prediction)
    if confidence is None:
        confidence = _finite_or_none(probe.confidence)
    if confidence is not None:
        return float(np.clip(confidence, 0.0, 1.0)), metric

    if baseline_probe is not None:
        same = not _prediction_changed(baseline_probe.prediction, probe.prediction)
        return (1.0 if same else 0.0), "prediction_stability"

    return 1.0, "prediction_stability"


def _quality_metric_name(task_family=None, hf_task=None):
    family = str(task_family or "").strip().lower()
    hf = str(hf_task or "").strip().lower()
    if family == "regression":
        return "inverse_absolute_error"
    if family == "retrieval":
        return "retrieval_confidence_or_rank_proxy"
    if family == "generation" or hf in {"causal_lm_generation", "seq2seq_generation", "image_captioning"}:
        return "sequence_likelihood_proxy"
    if family == "detection":
        return "detection_confidence_proxy"
    if family == "segmentation":
        return "segmentation_confidence_proxy"
    if family in {"token_classification", "fill_mask"}:
        return "token_confidence"
    if family == "clustering":
        return "assignment_stability"
    return "class_confidence"


def _scalar_target(value):
    if value is None:
        return None
    try:
        arr = np.asarray(value).reshape(-1)
    except Exception:
        return value
    if arr.size != 1:
        return None
    return arr[0]


def _stable_seed(*parts):
    text = ":".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _select_indices(count, limit, rng):
    if limit >= count:
        return list(range(count))
    return [int(i) for i in rng.choice(count, size=limit, replace=False)]


def _sample_count(x, y=None):
    source = y if y is not None else x
    if isinstance(source, dict):
        for value in source.values():
            try:
                return int(len(value))
            except Exception:
                continue
        return 0
    try:
        return int(len(source))
    except Exception:
        return 0


def _get_sample(x, idx):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _get_sample(v, idx) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x[int(idx)]
    if isinstance(x, (list, tuple)):
        return x[int(idx)]
    return x


def _batchify_x(sample):
    if isinstance(sample, dict):
        return {k: _batchify_value(v) for k, v in sample.items()}
    if isinstance(sample, str):
        return [sample]
    if isinstance(sample, (list, tuple)) and sample and isinstance(sample[0], str):
        return [list(sample)]
    return _batchify_value(sample)


def _batchify_y(sample):
    if sample is None:
        return None
    return _batchify_value(sample)


def _batchify_value(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1)
    return np.expand_dims(arr, axis=0)


def _predict_probe(model, x_sample, y_sample=None, *, task_family=None, hf_task=None):
    if hasattr(model, "core"):
        return _predict_hf_probe(model, x_sample, y_sample, task_family=task_family, hf_task=hf_task)
    return _predict_generic_probe(model, x_sample, task_family=task_family)


def _predict_generic_probe(model, x_sample, *, task_family=None):
    if not hasattr(model, "predict"):
        return None
    x_batch = _batchify_x(x_sample)
    try:
        try:
            raw = model.predict(x_batch, verbose=0)
        except TypeError:
            raw = model.predict(x_batch)
    except Exception:
        return None
    return _probe_from_array(raw, task_family=task_family)


def _predict_hf_probe(adapter, x_sample, y_sample=None, *, task_family=None, hf_task=None):
    core = getattr(adapter, "core", None)
    if core is None or getattr(core, "model", None) is None:
        return None
    torch = core.torch
    core.model.eval()
    xb = _batchify_x(x_sample)
    yb = _batchify_y(y_sample)
    with torch.no_grad():
        enc, labels_t, extra = core.task_spec.encode_batch(
            core.tokenizer,
            xb,
            yb,
            core.max_length,
            torch,
            core.device,
            ignore_index=core.label_pad_value,
            inference_only=True,
        )
        if bool(getattr(core.task_spec, "supports_generation", False)):
            ensure_left_padding = getattr(core, "_ensure_left_padding_for_decoder_only_generation", None)
            if callable(ensure_left_padding):
                ensure_left_padding()
            pred_t = core.task_spec.generate_predictions(
                core.model,
                enc,
                core.tokenizer,
                torch,
                core.generation_config,
            )
            confidence = np.nan
            if labels_t is not None:
                try:
                    teacher_inputs = core.task_spec.build_forward_inputs(enc, labels_t=labels_t, inference_only=False)
                    outputs = core.model(**teacher_inputs)
                    logits = _extract_hf_logits(core, outputs)
                    loss = core.task_spec.extract_loss(torch, outputs, logits, labels_t, extra)
                    if loss is not None:
                        confidence = float(np.exp(-float(loss.detach().cpu().item())))
                except Exception:
                    confidence = np.nan
            return PredictionProbe(
                prediction=_tensor_prediction(pred_t),
                confidence=confidence,
                output_value=None,
            )

        model_inputs = core.task_spec.build_forward_inputs(enc, labels_t=None, inference_only=True)
        outputs = core.model(**model_inputs)
        logits = _extract_hf_logits(core, outputs)
        pred_t = core.task_spec.preds_from_logits(torch, logits, extra)
        return _probe_from_hf_logits(torch, logits, pred_t, enc, task_family=task_family, hf_task=hf_task)


def _extract_hf_logits(core, outputs):
    extract_fn = getattr(core, "_extract_logits", None)
    if callable(extract_fn):
        return extract_fn(outputs)
    task_spec = getattr(core, "task_spec", None)
    extract_fn = getattr(task_spec, "extract_logits", None)
    if callable(extract_fn):
        return extract_fn(outputs)
    return outputs.logits


def _probe_from_array(raw, *, task_family=None):
    arr = np.asarray(raw)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        value = float(arr)
        return PredictionProbe(prediction=value, confidence=np.nan, output_value=value)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        sample = arr[0]
    else:
        sample = arr
    if sample.ndim == 0:
        value = float(sample)
        return PredictionProbe(prediction=value, confidence=np.nan, output_value=value)
    if sample.ndim == 1 and sample.size == 1 and (task_family or "") not in {"regression", "generation", "clustering"}:
        prob = float(sample.reshape(-1)[0])
        if 0.0 <= prob <= 1.0:
            pred = int(prob >= 0.5)
            conf = max(prob, 1.0 - prob)
            return PredictionProbe(
                prediction=pred,
                confidence=float(conf),
                output_value=None,
                scores=np.asarray([1.0 - prob, prob], dtype="float64"),
            )
    if sample.ndim == 1 and sample.size > 1 and (task_family or "") != "regression":
        probs = _as_probability_vector(sample.astype("float64"))
        pred = int(np.argmax(probs))
        return PredictionProbe(prediction=pred, confidence=float(np.max(probs)), output_value=None, scores=probs)
    if sample.ndim >= 2 and sample.shape[-1] > 1 and (task_family or "") != "regression":
        probs = _as_probability_vector(sample.reshape(-1, sample.shape[-1]).astype("float64"))
        pred = tuple(np.argmax(probs, axis=-1).astype(int).tolist()[:128])
        conf = float(np.mean(np.max(probs, axis=-1)))
        return PredictionProbe(prediction=pred, confidence=conf, output_value=None, scores=probs)
    value = float(np.asarray(sample).reshape(-1)[0])
    return PredictionProbe(prediction=value, confidence=np.nan, output_value=value)


def _probe_from_hf_logits(torch, logits, pred_t, enc, *, task_family=None, hf_task=None):
    logits_cpu = logits.detach().cpu()
    arr = logits_cpu.numpy()
    pred = _tensor_prediction(pred_t)
    if arr.ndim == 2:
        if str(task_family or "") == "retrieval" or str(hf_task or "") == "text_image_retrieval":
            row = arr[0].reshape(-1)
            if row.size == 0:
                return None
            pred_idx = int(np.argmax(row))
            if row.size > 1:
                ordered = np.sort(row)
                margin = float(ordered[-1] - ordered[-2])
            else:
                margin = float(row[pred_idx])
            confidence = float(1.0 / (1.0 + np.exp(-np.clip(margin, -50.0, 50.0))))
            return PredictionProbe(
                prediction=pred_idx,
                confidence=confidence,
                output_value=margin,
                scores=row,
            )
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return PredictionProbe(
            prediction=int(np.argmax(probs[0])),
            confidence=float(np.max(probs[0])),
            output_value=None,
            scores=probs[0],
        )
    if arr.ndim == 3:
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        mask = None
        try:
            if isinstance(enc, dict) and "attention_mask" in enc:
                mask = enc["attention_mask"].detach().cpu().numpy()[0].astype(bool)
        except Exception:
            mask = None
        if mask is not None and mask.shape[0] == probs.shape[0]:
            probs_used = probs[mask]
        else:
            probs_used = probs.reshape(-1, probs.shape[-1])
        if probs_used.size == 0:
            probs_used = probs.reshape(-1, probs.shape[-1])
        conf = float(np.mean(np.max(probs_used, axis=-1)))
        return PredictionProbe(prediction=pred, confidence=conf, output_value=None, scores=probs_used)
    if arr.ndim == 4:
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        max_probs = np.max(probs, axis=0)
        dominant = int(np.argmax(np.bincount(np.argmax(probs, axis=0).reshape(-1))))
        return PredictionProbe(prediction=dominant, confidence=float(np.mean(max_probs)), output_value=None, scores=probs)
    return PredictionProbe(prediction=pred, confidence=np.nan, output_value=None)


def _as_probability_vector(values):
    arr = np.asarray(values, dtype="float64")
    if arr.ndim == 1:
        if np.all(arr >= 0.0) and np.all(arr <= 1.0) and np.isclose(float(np.sum(arr)), 1.0, atol=1e-3):
            return arr
        shifted = arr - np.max(arr)
        exp = np.exp(np.clip(shifted, -50.0, 50.0))
        return exp / max(float(np.sum(exp)), 1e-12)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)


def _tensor_prediction(value):
    try:
        arr = value.detach().cpu().numpy()
    except Exception:
        arr = np.asarray(value)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    flat = np.asarray(arr).reshape(-1)
    if flat.size == 1:
        try:
            return int(flat[0])
        except Exception:
            return float(flat[0])
    return tuple(int(v) for v in flat[:128])


def _meaningful_units(x_sample, y_sample=None, *, task_family=None, meta=None, limit=8):
    modality = _modality(x_sample, task_family=task_family, meta=meta)
    if modality == "tokens":
        positions = _token_positions(x_sample)
        if str(task_family or "") == "token_classification":
            spans = _entity_spans(y_sample, positions)
            if spans:
                return _limited(spans, limit)
        return _limited([("token", int(pos)) for pos in positions], limit)
    if modality == "text":
        words = str(x_sample).split()
        return _limited([("word", i) for i in range(len(words))], limit)
    if modality == "image":
        return _limited(_image_patches(x_sample), limit)
    if modality == "numeric":
        arr = np.asarray(x_sample)
        if arr.ndim == 0:
            return []
        flat_size = int(arr.size)
        return _limited([("feature", i) for i in range(flat_size)], limit)
    return []


def _expected_units(
    x_sample,
    y_sample=None,
    *,
    task_family=None,
    hf_task=None,
    meta=None,
    model=None,
    config=None,
    candidate_units=None,
    limit=8,
):
    config = config or {}
    configured = _configured_feature_units(config.get("explainability_expected_feature_indices"))
    if not configured and isinstance(meta, dict):
        configured = _configured_feature_units(meta.get("explainability_expected_feature_indices"))
    if configured:
        return _limited(configured, limit), "configured_features"

    modality = _modality(x_sample, task_family=task_family, meta=meta)
    family = str(task_family or "").strip().lower()
    hf = str(hf_task or "").strip().lower()
    if modality == "tokens":
        positions = _token_positions(x_sample)
        if family == "token_classification":
            spans = _entity_spans(y_sample, positions)
            if spans:
                return _limited(spans, limit), "label_span"
        if family == "fill_mask" or hf == "fill_mask":
            mask_context = _mask_context_units(x_sample, positions, model=model, limit=limit)
            if mask_context:
                return mask_context, "mask_context"
        content_tokens = _token_content_units(x_sample, positions, model=model, limit=limit)
        if content_tokens:
            return content_tokens, "token_content_heuristic"
        return [], None

    if modality == "text":
        content_words = _content_word_units(x_sample, limit=limit)
        if content_words:
            return content_words, "content_word_heuristic"
        return [], None

    if modality == "image" and family in {"detection", "segmentation"}:
        image_units = _expected_image_units(x_sample, y_sample, candidate_units=candidate_units, limit=limit)
        if image_units:
            return image_units, "label_region"

    return [], None


def _configured_feature_units(value):
    if value is None:
        return []
    if isinstance(value, str):
        pieces = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        pieces = list(value)
    else:
        pieces = [value]
    units = []
    for piece in pieces:
        try:
            units.append(("feature", int(piece)))
        except Exception:
            continue
    return units


_TEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "many",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_TOKEN_STRIP_CHARS = " \t\r\n\"'`.,;:!?()[]{}<>/\\|+-=*_~@#$%^&"


def _content_word_units(text, *, limit):
    words = str(text).split()
    units = []
    for idx, word in enumerate(words):
        cleaned = _clean_unit_text(word)
        if _looks_like_content_word(cleaned):
            units.append(("word", idx))
    return _limited(units, limit)


def _token_content_units(x_sample, positions, *, model=None, limit):
    if not isinstance(x_sample, dict) or "input_ids" not in x_sample:
        return []
    tokenizer = getattr(getattr(model, "core", None), "tokenizer", None)
    if tokenizer is None:
        return []
    ids = np.asarray(x_sample.get("input_ids")).reshape(-1)
    units = []
    for pos in positions:
        if int(pos) < 0 or int(pos) >= ids.size:
            continue
        token_text = _token_text(tokenizer, int(ids[int(pos)]))
        cleaned = _clean_unit_text(token_text)
        if _looks_like_content_word(cleaned):
            units.append(("token", int(pos)))
    return _limited(units, limit)


def _mask_context_units(x_sample, positions, *, model=None, limit):
    if not isinstance(x_sample, dict) or "input_ids" not in x_sample:
        return []
    mask_token_id = _mask_token_id(model)
    if mask_token_id is None:
        return []
    ids = np.asarray(x_sample.get("input_ids")).reshape(-1)
    mask_positions = [int(pos) for pos in positions if int(pos) < ids.size and int(ids[int(pos)]) == mask_token_id]
    if not mask_positions:
        return []
    radius = max(1, int(math.ceil(max(1, limit) / max(1, len(mask_positions)))))
    context_positions = []
    valid = set(int(pos) for pos in positions)
    for mask_pos in mask_positions:
        for offset in range(1, radius + 1):
            for pos in (mask_pos - offset, mask_pos + offset):
                if pos in valid and pos not in mask_positions and pos not in context_positions:
                    context_positions.append(pos)
    return _limited([("token", int(pos)) for pos in context_positions], limit)


def _token_text(tokenizer, token_id):
    try:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        if isinstance(token, str):
            return token
    except Exception:
        pass
    try:
        return str(tokenizer.decode([int(token_id)]))
    except Exception:
        return ""


def _clean_unit_text(value):
    text = str(value or "").strip()
    while text.startswith("#"):
        text = text[1:]
    text = text.lstrip("\u0120\u2581")
    text = text.strip(_TOKEN_STRIP_CHARS).lower()
    return text


def _looks_like_content_word(text):
    if not text or text in _TEXT_STOPWORDS:
        return False
    if len(text) <= 1:
        return False
    return any(ch.isalpha() for ch in text)


def _expected_image_units(x_sample, y_sample, *, candidate_units=None, limit):
    patches = [unit for unit in (candidate_units or _image_patches(x_sample)) if unit and unit[0] == "patch"]
    if not patches:
        return []
    boxes = _label_boxes(y_sample, x_sample)
    if not boxes:
        return []
    selected = []
    for patch in patches:
        if any(_patch_box_iou(patch, box) > 0.0 for box in boxes):
            selected.append(patch)
    return _limited(selected, limit)


def _label_boxes(y_sample, x_sample):
    if isinstance(y_sample, dict):
        for key in ("boxes", "bbox", "bboxes"):
            if key in y_sample:
                return _normalise_boxes(y_sample.get(key), x_sample)
        if "mask" in y_sample:
            box = _mask_box(y_sample.get("mask"))
            return [box] if box is not None else []
    box = _mask_box(y_sample)
    return [box] if box is not None else []


def _normalise_boxes(value, x_sample):
    try:
        arr = np.asarray(value, dtype="float64").reshape(-1, 4)
    except Exception:
        return []
    image = _image_array_view(x_sample)
    if image is None:
        return []
    image = np.asarray(image)
    channel_first = image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3)
    height = float(image.shape[1] if channel_first else image.shape[0])
    width = float(image.shape[2] if channel_first else image.shape[1])
    boxes = []
    for x0, y0, x1, y1 in arr:
        if max(x0, y0, x1, y1) <= 1.0:
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        boxes.append((float(y0), float(y1), float(x0), float(x1)))
    return boxes


def _mask_box(value):
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim < 2 or arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = np.any(arr != 0, axis=0)
    else:
        arr = arr != 0
    positions = np.argwhere(arr)
    if positions.size == 0:
        return None
    y0, x0 = positions.min(axis=0)[:2]
    y1, x1 = positions.max(axis=0)[:2] + 1
    return (float(y0), float(y1), float(x0), float(x1))


def _patch_box_iou(patch, box):
    _, y0, y1, x0, x1, _ = patch
    by0, by1, bx0, bx1 = box
    inter_y = max(0.0, min(float(y1), float(by1)) - max(float(y0), float(by0)))
    inter_x = max(0.0, min(float(x1), float(bx1)) - max(float(x0), float(bx0)))
    inter = inter_y * inter_x
    if inter <= 0.0:
        return 0.0
    patch_area = max(0.0, float(y1 - y0)) * max(0.0, float(x1 - x0))
    box_area = max(0.0, float(by1 - by0)) * max(0.0, float(bx1 - bx0))
    return float(inter / max(patch_area + box_area - inter, 1e-9))


def _alignment_score(ranked_units, expected_units):
    expected_units = list(expected_units or [])
    if not expected_units:
        return np.nan
    top_units = list(ranked_units or [])[: len(expected_units)]
    if not top_units:
        return 0.0
    matches = 0
    for expected in expected_units:
        if any(_units_overlap(model_unit, expected) for model_unit in top_units):
            matches += 1
    return float(np.clip(matches / max(1, len(expected_units)), 0.0, 1.0))


def _units_overlap(a, b):
    if not a or not b:
        return False
    if a[0] == "patch" and b[0] == "patch":
        return _patch_box_iou(a, (float(b[1]), float(b[2]), float(b[3]), float(b[4]))) > 0.0
    return bool(_unit_positions(a) & _unit_positions(b))


def _unit_positions(unit):
    if not unit:
        return set()
    kind = unit[0]
    if kind in {"token", "word", "feature"} and len(unit) >= 2:
        return {(kind, int(unit[1]))}
    if kind == "span" and len(unit) >= 3:
        return {("token", int(pos)) for pos in range(int(unit[1]), int(unit[2]))}
    return {tuple(unit)}


def _limited(units, limit):
    units = list(units or [])
    if len(units) <= limit:
        return units
    positions = np.linspace(0, len(units) - 1, num=int(limit), dtype=int)
    return [units[int(i)] for i in positions]


def _entity_spans(y_sample, token_positions):
    if y_sample is None:
        return []
    try:
        labels = np.asarray(y_sample).reshape(-1)
    except Exception:
        return []
    valid_positions = [int(p) for p in token_positions if int(p) < labels.size]
    spans = []
    start = None
    prev = None
    for pos in valid_positions:
        label = labels[pos]
        try:
            label_int = int(label)
        except Exception:
            label_int = 0
        is_entity = label_int not in (0, -100)
        if is_entity and start is None:
            start = pos
        if (not is_entity or (prev is not None and pos != prev + 1)) and start is not None:
            end = prev + 1 if prev is not None else pos
            spans.append(("span", int(start), int(end)))
            start = pos if is_entity else None
        prev = pos
    if start is not None and prev is not None:
        spans.append(("span", int(start), int(prev + 1)))
    return spans


def _token_positions(x_sample):
    if not isinstance(x_sample, dict) or "input_ids" not in x_sample:
        return []
    input_ids = np.asarray(x_sample.get("input_ids")).reshape(-1)
    mask = x_sample.get("attention_mask")
    if mask is not None:
        try:
            keep = np.asarray(mask).reshape(-1).astype(bool)
        except Exception:
            keep = np.ones_like(input_ids, dtype=bool)
    else:
        keep = np.ones_like(input_ids, dtype=bool)
    positions = [i for i, keep_i in enumerate(keep) if keep_i]
    if len(positions) > 2:
        return positions[1:-1]
    return positions


def _image_patches(x_sample):
    arr = _image_array_view(x_sample)
    if arr is None or arr.ndim != 3:
        return []
    channel_first = arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3)
    h_axis = 1 if channel_first else 0
    w_axis = 2 if channel_first else 1
    height = int(arr.shape[h_axis])
    width = int(arr.shape[w_axis])
    grid = 3 if min(height, width) >= 12 else 2
    patches = []
    for gy in range(grid):
        for gx in range(grid):
            y0 = int(round(gy * height / grid))
            y1 = int(round((gy + 1) * height / grid))
            x0 = int(round(gx * width / grid))
            x1 = int(round((gx + 1) * width / grid))
            if y1 > y0 and x1 > x0:
                patches.append(("patch", y0, y1, x0, x1, bool(channel_first)))
    return patches


def _modality(x_sample, *, task_family=None, meta=None):
    if isinstance(x_sample, dict):
        if "input_ids" in x_sample:
            return "tokens"
        if "pixel_values" in x_sample:
            return "image"
    if isinstance(x_sample, str):
        return "text"
    if str(task_family or "") in {"detection", "segmentation"}:
        return "image"
    arr = None
    try:
        arr = np.asarray(x_sample)
    except Exception:
        arr = None
    if arr is not None and arr.ndim >= 3:
        return "image"
    if arr is not None and arr.ndim >= 1 and arr.dtype.kind in {"b", "i", "u", "f"}:
        return "numeric"
    return "unknown"


def _apply_targeted_mask(x_sample, units, *, task_family=None, model=None, meta=None):
    out = _copy_sample(x_sample)
    for unit in units:
        kind = unit[0] if unit else None
        if kind == "token":
            out = _mask_token(out, int(unit[1]), model=model)
        elif kind == "span":
            for pos in range(int(unit[1]), int(unit[2])):
                out = _mask_token(out, pos, model=model)
        elif kind == "word":
            words = str(out).split()
            idx = int(unit[1])
            if 0 <= idx < len(words):
                mask_token = _mask_token_text(model) or ""
                if mask_token:
                    words[idx] = mask_token
                else:
                    words.pop(idx)
                out = " ".join(words)
        elif kind == "patch":
            out = _mask_patch(out, unit)
        elif kind == "feature":
            out = _mask_feature(out, int(unit[1]))
    return out


def _apply_benign_perturbation(x_sample, rng, *, task_family=None, model=None, meta=None, strength=0.02):
    modality = _modality(x_sample, task_family=task_family, meta=meta)
    out = _copy_sample(x_sample)
    if modality == "image":
        return _image_noise(out, rng, strength=strength)
    if modality == "numeric":
        arr = np.asarray(out).astype("float32", copy=True)
        scale = float(np.nanstd(arr)) if arr.size else 0.0
        if not np.isfinite(scale) or scale == 0.0:
            scale = max(float(np.nanmax(np.abs(arr))) if arr.size else 1.0, 1.0)
        return arr + rng.normal(0.0, max(1e-8, strength * scale), size=arr.shape).astype(arr.dtype)
    if modality == "tokens":
        positions = _token_positions(out)
        if not positions:
            return out
        pos = int(rng.choice(positions))
        return _mask_token(out, pos, model=model)
    if modality == "text":
        return " ".join(str(out).split())
    return out


def _copy_sample(sample):
    if isinstance(sample, dict):
        return {k: _copy_sample(v) for k, v in sample.items()}
    if isinstance(sample, np.ndarray):
        return np.array(sample, copy=True)
    if isinstance(sample, list):
        return list(sample)
    if isinstance(sample, tuple):
        return tuple(sample)
    return sample


def _mask_token(sample, pos, *, model=None):
    if not isinstance(sample, dict) or "input_ids" not in sample:
        return sample
    out = _copy_sample(sample)
    ids = np.asarray(out["input_ids"]).copy()
    flat = ids.reshape(-1)
    if 0 <= int(pos) < flat.size:
        token_id = _mask_token_id(model)
        if token_id is None:
            token_id = _pad_token_id(model)
        flat[int(pos)] = int(token_id if token_id is not None else 0)
        out["input_ids"] = flat.reshape(ids.shape)
    return out


def _mask_token_id(model):
    tokenizer = getattr(getattr(model, "core", None), "tokenizer", None)
    value = getattr(tokenizer, "mask_token_id", None)
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _pad_token_id(model):
    tokenizer = getattr(getattr(model, "core", None), "tokenizer", None)
    value = getattr(tokenizer, "pad_token_id", None)
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _mask_token_text(model):
    tokenizer = getattr(getattr(model, "core", None), "tokenizer", None)
    value = getattr(tokenizer, "mask_token", None)
    return str(value) if value else None


def _image_array_view(sample):
    if isinstance(sample, dict):
        value = sample.get("pixel_values")
        if value is None:
            return None
        return np.asarray(value)
    return np.asarray(sample)


def _set_image_array(sample, arr):
    if isinstance(sample, dict):
        out = _copy_sample(sample)
        out["pixel_values"] = arr
        return out
    return arr


def _mask_patch(sample, unit):
    arr = np.asarray(_image_array_view(sample)).copy()
    if arr.ndim != 3:
        return sample
    _, y0, y1, x0, x1, channel_first = unit
    fill = float(np.nanmean(arr)) if arr.size else 0.0
    if channel_first:
        arr[:, int(y0):int(y1), int(x0):int(x1)] = fill
    else:
        arr[int(y0):int(y1), int(x0):int(x1), :] = fill
    return _set_image_array(sample, arr)


def _image_noise(sample, rng, *, strength):
    arr = np.asarray(_image_array_view(sample)).astype("float32", copy=True)
    if arr.size == 0:
        return sample
    orig_min = float(np.nanmin(arr))
    orig_max = float(np.nanmax(arr))
    scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0 if orig_max <= 1.5 else 255.0
    arr = arr + rng.normal(0.0, max(1e-8, strength * scale), size=arr.shape).astype("float32")
    if orig_min >= 0.0:
        upper = 1.0 if orig_max <= 1.5 else 255.0
        arr = np.clip(arr, 0.0, upper)
    return _set_image_array(sample, arr)


def _mask_feature(sample, idx):
    arr = np.asarray(sample).copy()
    flat = arr.reshape(-1)
    if 0 <= int(idx) < flat.size:
        flat[int(idx)] = 0
    return flat.reshape(arr.shape)


def _confidence_drop(baseline: PredictionProbe, candidate: PredictionProbe):
    b = _finite_or_none(baseline.confidence)
    c = _finite_or_none(candidate.confidence)
    if b is not None and c is not None:
        return float(b - c)
    if baseline.output_value is not None and candidate.output_value is not None:
        denom = max(abs(float(baseline.output_value)), 1e-9)
        return float(abs(float(candidate.output_value) - float(baseline.output_value)) / denom)
    return 0.0


def _confidence_for_prediction(probe: PredictionProbe, prediction):
    scores = getattr(probe, "scores", None)
    if scores is None:
        return None
    try:
        arr = np.asarray(scores, dtype="float64")
    except Exception:
        return None
    if arr.size == 0:
        return None
    try:
        pred_arr = np.asarray(prediction).reshape(-1)
    except Exception:
        pred_arr = np.asarray([prediction])
    if pred_arr.size == 1 and arr.ndim == 1:
        try:
            idx = int(pred_arr[0])
        except Exception:
            return None
        if 0 <= idx < arr.size:
            return float(arr[idx])
    if arr.ndim == 2 and pred_arr.size:
        values = []
        for pos, pred in enumerate(pred_arr[: arr.shape[0]]):
            try:
                idx = int(pred)
            except Exception:
                continue
            if 0 <= idx < arr.shape[1]:
                values.append(float(arr[pos, idx]))
        if values:
            return float(np.mean(values))
    return None


def _prediction_changed(a, b):
    try:
        arr_a = np.asarray(a)
        arr_b = np.asarray(b)
        if arr_a.shape != arr_b.shape:
            return True
        if arr_a.dtype.kind in {"f", "c"} or arr_b.dtype.kind in {"f", "c"}:
            return not bool(np.allclose(arr_a, arr_b, rtol=1e-4, atol=1e-6))
        return not bool(np.array_equal(arr_a, arr_b))
    except Exception:
        return a != b


def _finite_or_none(value):
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if np.isfinite(parsed) else None


def _finite_values(values):
    cleaned = [_finite_or_none(v) for v in (values or [])]
    return [float(v) for v in cleaned if v is not None]


def _mean_or_nan(values):
    cleaned = _finite_values(values)
    return float(np.mean(cleaned)) if cleaned else np.nan


def _std_or_nan(values):
    cleaned = _finite_values(values)
    return float(np.std(cleaned)) if cleaned else np.nan


def _percentile_or_nan(values, percentile):
    cleaned = _finite_values(values)
    return float(np.percentile(cleaned, percentile)) if cleaned else np.nan


def _min_or_nan(values):
    cleaned = _finite_values(values)
    return float(np.min(cleaned)) if cleaned else np.nan


def _max_or_nan(values):
    cleaned = _finite_values(values)
    return float(np.max(cleaned)) if cleaned else np.nan


def _sample_explainability_score(record):
    score = _finite_or_none(record.get("explainability_score"))
    if score is not None:
        return float(np.clip(score, 0.0, 1.0))
    signal = _finite_or_none(record.get("targeted_relative_drop"))
    if signal is None or signal == 0.0:
        signal = _finite_or_none(record.get("targeted_output_relative_delta"))
    if signal is None:
        signal = 0.0
    unit_fraction = _finite_or_none(record.get("targeted_unit_fraction"))
    compactness = 1.0 - min(1.0, unit_fraction if unit_fraction is not None else 1.0)
    return float(np.clip(max(0.0, signal) * max(0.0, compactness), 0.0, 1.0))


def _sample_trust_score(record):
    confidence_delta = _finite_or_none(record.get("trust_confidence_abs_delta_mean"))
    if confidence_delta is None:
        confidence_delta = _finite_or_none(record.get("trust_output_relative_delta_mean"))
    confidence_stability = 1.0 - (confidence_delta if confidence_delta is not None else 1.0)
    confidence_stability = float(np.clip(confidence_stability, 0.0, 1.0))

    prediction_stability = _finite_or_none(record.get("trust_prediction_same_rate"))
    prediction_stability = float(np.clip(prediction_stability if prediction_stability is not None else 0.0, 0.0, 1.0))
    return float(np.clip((confidence_stability + prediction_stability) / 2.0, 0.0, 1.0))


def _values(records, key):
    return [item.get(key) for item in records if item.get(key) is not None]


def _dominant_text(records, key):
    counts = {}
    for item in records or []:
        value = item.get(key)
        if not value or value == "none":
            continue
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    if not counts:
        return "none"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _jsonable_prediction(prediction):
    if isinstance(prediction, tuple):
        return list(prediction)
    if isinstance(prediction, np.ndarray):
        return prediction.tolist()
    if isinstance(prediction, np.integer):
        return int(prediction)
    if isinstance(prediction, np.floating):
        return float(prediction)
    return prediction
