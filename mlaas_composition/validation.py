"""Strict, held-out validation primitives for collaborative compositions.

This module deliberately keeps the audit cohort separate from model execution.
The runtime adapter supplies predictions only after it has proved that its
training and catalogue-metric data do not overlap the cohort plan.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .catalogue import number
from .pool import request_object
from .protocol import objective_version_for_request
from .sampler import calculate_constraint_penalty, calculate_utility, evaluate_provisional_composition


COHORT_PLAN_VERSION = "1.0.0"
MIN_AUDIT_SAMPLES = 64
DEFAULT_AUDIT_SAMPLES = 512


class CohortIsolationError(ValueError):
    """Raised whenever data partitions cannot be proved disjoint."""


@dataclass(frozen=True)
class CohortPlan:
    """Immutable three-way partition over stable source example identifiers."""

    cohort_id: str
    contract_hash: str
    seed: int
    training_ids: tuple[str, ...]
    catalogue_metric_ids: tuple[str, ...]
    audit_ids: tuple[str, ...]
    label_hash: str
    version: str = COHORT_PLAN_VERSION

    @property
    def audit_hash(self) -> str:
        return _hash_json(self.audit_ids)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"audit_hash": self.audit_hash}


@dataclass(frozen=True)
class ServicePrediction:
    service_id: str
    audit_ids: tuple[str, ...]
    probabilities: Any
    latency_s: float
    label_hash: str
    task_family: str = "classification"

    @property
    def outputs(self) -> np.ndarray:
        """Return classifier probabilities or continuous regression predictions."""
        return self.probabilities


def build_cohort_plan(
    *,
    training_ids: Sequence[str],
    metric_source_ids: Sequence[str],
    metric_source_labels: Sequence[Any],
    contract: Mapping[str, Any],
    seed: int = 42,
    audit_samples: int = DEFAULT_AUDIT_SAMPLES,
    minimum_audit_samples: int = MIN_AUDIT_SAMPLES,
    task_family: str = "classification",
) -> CohortPlan:
    """Partition a non-training source into catalogue-metric and audit rows.

    ``training_ids`` must already come from a source split that is disjoint from
    ``metric_source_ids``.  The deterministic label-stratified audit selection
    prevents the audit cohort from being chosen using service outcomes.
    """
    train = tuple(map(str, training_ids))
    source = tuple(map(str, metric_source_ids))
    labels = tuple(_json_scalar(value) for value in metric_source_labels)
    if len(source) != len(labels):
        raise CohortIsolationError("metric_source_ids and metric_source_labels must have equal length")
    if len(set(train)) != len(train) or len(set(source)) != len(source):
        raise CohortIsolationError("cohort source IDs must be unique and stable")
    if set(train).intersection(source):
        raise CohortIsolationError("training and metric/audit source IDs overlap")
    if len(source) < max(2, int(minimum_audit_samples) + 1):
        raise CohortIsolationError("insufficient metric source rows for a disjoint audit cohort")

    target = min(int(audit_samples), len(source) - 1)
    if target < int(minimum_audit_samples):
        raise CohortIsolationError("audit cohort is below minimum_audit_samples")
    if str(task_family).strip().lower() != "classification":
        rng = np.random.default_rng(seed)
        audit_positions = set(
            int(index) for index in rng.choice(len(source), target, replace=False)
        )
    else:
        audit_positions = _stratified_positions(labels, target, seed)
    audit = tuple(source[index] for index in audit_positions)
    metric = tuple(value for index, value in enumerate(source) if index not in audit_positions)
    if not metric:
        raise CohortIsolationError("catalogue-metric partition would be empty")
    assert_cohort_isolation(train, metric, audit)
    contract_hash = _hash_json(_normalise_json(contract))
    label_hash = cohort_label_hash(source, labels)
    cohort_id = f"cohort_{_hash_json({'contract': contract_hash, 'seed': int(seed), 'audit': audit})[:16]}"
    return CohortPlan(cohort_id, contract_hash, int(seed), train, metric, audit, label_hash)


def cohort_label_hash(ids: Sequence[str], labels: Sequence[Any]) -> str:
    """Hash the complete source-ID/label mapping used by an audit cohort."""
    pairs = sorted(
        zip(map(str, ids), (_json_scalar(value) for value in labels)),
        key=lambda item: item[0],
    )
    return _hash_json(pairs)


def assert_cohort_isolation(
    training_ids: Iterable[str], catalogue_metric_ids: Iterable[str], audit_ids: Iterable[str]
) -> None:
    partitions = [set(map(str, values)) for values in (training_ids, catalogue_metric_ids, audit_ids)]
    names = ("training", "catalogue_metric", "audit")
    for left, right in combinations(range(3), 2):
        if partitions[left].intersection(partitions[right]):
            raise CohortIsolationError(f"{names[left]} and {names[right]} cohorts overlap")
    if not all(partitions):
        raise CohortIsolationError("every strict cohort partition must be non-empty")


def supported_composition_reason(services: pd.DataFrame, *, cohort_contract_hash: str | None = None) -> str | None:
    """Return a human-readable incompatibility reason; never filter services."""
    if services.empty:
        return "composition has no services"
    task = services.get("task_family", pd.Series([], dtype=object)).astype(str).str.lower()
    modality = services.get("modality", pd.Series([], dtype=object)).astype(str).str.lower()
    model_type = services.get("model_type", pd.Series([], dtype=object)).astype(str).str.lower()
    hf_task = services.get("hf_task", pd.Series([], dtype=object)).astype(str).str.lower().str.replace("-", "_", regex=False)
    if task.nunique() == 1 and task.iloc[0] in {"token_classification", "segmentation", "generation"}:
        if not services.get("task_semantics_hash", pd.Series(index=services.index, dtype=object)).notna().all():
            return "unknown_task_contract: structured composition requires a frozen task contract"
        for field in ("task_semantics_hash", "evaluation_protocol_hash", "hf_task", "modality"):
            if field not in services or services[field].isna().any() or services[field].nunique() != 1:
                code = "evaluation_protocol_mismatch" if field == "evaluation_protocol_hash" else "target_semantics_mismatch"
                return f"{code}: services do not share {field}"
        from mlaas_data_generator.task_capabilities import resolve_capabilities
        for row in services.to_dict("records"):
            decision = resolve_capabilities(row, stage="composition")["measured_composition_validation"]
            if decision["decision"] == "ineligible":
                return f"{decision['reason_code']}: {decision['explanation']}"
        return None
    if task.nunique() != 1 or not task.isin(("classification", "regression")).all():
        return "services must share one supported task (classification or regression)"
    if not modality.isin(("text", "image", "tabular")).all() or modality.nunique() != 1:
        return "services must share one supported modality (text, image, or tabular)"
    supported_models = {
        "hf", "hf_finetune", "hf_train", "transformers", "transformers_finetune",
        "cnn", "mobilenetv2", "mlp", "logreg",
    }
    if not model_type.isin(supported_models).all():
        return "services do not use a runtime-validatable model family"
    hf_rows = model_type.isin(
        ("hf", "hf_finetune", "hf_train", "transformers", "transformers_finetune")
    )
    if hf_rows.any():
        if task.iloc[0] == "classification":
            expected = (
                "sequence_classification" if modality.iloc[0] == "text"
                else "image_classification"
            )
        else:
            expected = "sentence_similarity"
        if not hf_task[hf_rows].eq(expected).all():
            return f"Hugging Face services must share hf_task={expected}"
    for field in ("dataset_name", "dataset_config", "benchmark_split", "label_format"):
        if field in services and services[field].astype(str).str.lower().nunique() != 1:
            return f"services do not share {field}"
    if "label_hash" in services and services["label_hash"].astype(str).nunique() != 1:
        return "services do not share a label-map hash"
    if cohort_contract_hash and "cohort_contract_hash" in services:
        hashes = services["cohort_contract_hash"].dropna().astype(str).unique()
        if len(hashes) and (len(hashes) != 1 or hashes[0] != cohort_contract_hash):
            return "services do not share the requested cohort contract"
    return None


def average_probabilities(predictions: Sequence[ServicePrediction], plan: CohortPlan) -> tuple[np.ndarray, float]:
    """Validate all service outputs then return their probability mean and wall latency."""
    if not predictions:
        raise ValueError("at least one service prediction is required")
    expected_ids = tuple(plan.audit_ids)
    shapes = set()
    for prediction in predictions:
        if tuple(prediction.audit_ids) != expected_ids:
            raise CohortIsolationError(f"service '{prediction.service_id}' was not evaluated on the exact audit cohort")
        if prediction.label_hash != plan.label_hash:
            raise CohortIsolationError(f"service '{prediction.service_id}' label map does not match cohort plan")
        probabilities = np.asarray(prediction.probabilities, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(expected_ids):
            raise ValueError(f"service '{prediction.service_id}' returned invalid probability dimensions")
        if not np.isfinite(probabilities).all():
            raise ValueError(f"service '{prediction.service_id}' returned non-finite probabilities")
        shapes.add(probabilities.shape)
    if len(shapes) != 1:
        raise ValueError("service probability vectors have incompatible class dimensions")
    return np.mean([np.asarray(item.probabilities, dtype=float) for item in predictions], axis=0), max(float(item.latency_s) for item in predictions)


def average_continuous_predictions(
    predictions: Sequence[ServicePrediction], plan: CohortPlan
) -> tuple[np.ndarray, float]:
    """Validate and mean continuous outputs from regression composition members."""
    if not predictions:
        raise ValueError("at least one service prediction is required")
    expected_ids = tuple(plan.audit_ids)
    values: list[np.ndarray] = []
    for prediction in predictions:
        if tuple(prediction.audit_ids) != expected_ids:
            raise CohortIsolationError(
                f"service '{prediction.service_id}' was not evaluated on the exact audit cohort"
            )
        if prediction.label_hash != plan.label_hash:
            raise CohortIsolationError(
                f"service '{prediction.service_id}' label contract does not match cohort plan"
            )
        current = np.asarray(prediction.outputs, dtype=float)
        if current.ndim == 2 and current.shape[1] == 1:
            current = current[:, 0]
        if current.ndim != 1 or current.shape[0] != len(expected_ids):
            raise ValueError(
                f"service '{prediction.service_id}' returned invalid continuous prediction dimensions"
            )
        if not np.isfinite(current).all():
            raise ValueError(
                f"service '{prediction.service_id}' returned non-finite continuous predictions"
            )
        values.append(current)
    return np.mean(values, axis=0), max(float(item.latency_s) for item in predictions)


def measure_model_outputs(
    outputs: np.ndarray,
    plan: CohortPlan,
    labels: Sequence[Any],
    *,
    task_family: str,
    inference_latency_s: float,
) -> dict[str, float]:
    """Measure one materialized composition model on the immutable audit cohort."""
    from mlaas_data_generator.tasks.runtime import TaskEvaluation
    if isinstance(outputs, TaskEvaluation):
        if not np.isfinite(inference_latency_s) or inference_latency_s < 0:
            raise ValueError("aggregated-model inference latency must be finite and non-negative")
        return outputs.validated_metrics(source_ids=plan.audit_ids, target_labels=labels, task_family=task_family)
    task = str(task_family).strip().lower()
    expected = np.asarray(labels)
    if expected.shape[0] != len(plan.audit_ids):
        raise ValueError("audit labels do not align with the immutable cohort")
    if not np.isfinite(float(inference_latency_s)) or float(inference_latency_s) < 0.0:
        raise ValueError("aggregated-model inference latency must be finite and non-negative")

    values = np.asarray(outputs, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("aggregated model returned non-finite outputs")
    if task == "classification":
        if values.ndim != 2 or values.shape[0] != len(plan.audit_ids):
            raise ValueError("aggregated classifier returned invalid probability dimensions")
        predicted = np.argmax(values, axis=1)
        truth = expected.argmax(axis=-1) if expected.ndim > 1 and expected.shape[-1] > 1 else expected.astype(int).reshape(-1)
        accuracy = float(np.mean(predicted == truth))
        return {
            "accuracy": accuracy,
            "f1_macro": _macro_f1(truth, predicted),
            "quality_score": accuracy,
            "inference_latency_s": float(inference_latency_s),
        }
    if task == "regression":
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or values.shape[0] != len(plan.audit_ids):
            raise ValueError("aggregated regressor returned invalid prediction dimensions")
        truth = expected.astype(float).reshape(-1)
        residual = values - truth
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        mae = float(np.mean(np.abs(residual)))
        return {
            "rmse": rmse,
            "mae": mae,
            "quality_score": float(1.0 / (1.0 + max(0.0, rmse))),
            "inference_latency_s": float(inference_latency_s),
        }
    raise ValueError("runtime composition validation supports classification or regression")


def measured_recipe_quality(actual, outputs, labels, selected):
    """Use the catalogue's named metric and normalisation for a fair comparison."""
    from mlaas_data_generator.services.taxonomy import metric_score_value
    names = selected["primary_metric_name"].astype(str).str.lower().unique()
    if len(names) != 1:
        raise ValueError("selected services disagree on the primary quality metric")
    name = str(names[0])
    task = str(selected["task_family"].iloc[0])
    result = dict(actual)
    if task == "regression":
        truth = np.asarray(labels, dtype=float).reshape(-1)
        prediction = np.asarray(outputs, dtype=float).reshape(-1)
        result["mse"] = float(np.mean((prediction - truth) ** 2))
        if name in {"pearson", "spearman"}:
            from scipy.stats import pearsonr, spearmanr
            result["pearson"] = float(pearsonr(truth, prediction).statistic)
            result["spearman"] = float(spearmanr(truth, prediction).statistic)
    if name == "f1" and "f1" not in result and "f1_macro" in result:
        result["f1"] = result["f1_macro"]
    if name not in result or not np.isfinite(result[name]):
        raise ValueError(f"audit cannot measure the catalogue primary metric {name!r}")
    quality = metric_score_value(task, name, result[name])
    if not np.isfinite(quality) or not 0 <= quality <= 1:
        raise ValueError("audit quality normalisation is invalid")
    result.update(quality_score=float(quality), primary_metric_name=name,
                  secondary_metric_name=str(selected["auxiliary_metric_name"].iloc[0]))
    return result


def aggregate_and_measure(
    predictions: Sequence[ServicePrediction],
    plan: CohortPlan,
    labels: Sequence[Any],
    *,
    task_family: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the collaborative mean rule and calculate task-appropriate metrics."""
    task = str(task_family).strip().lower()
    expected = np.asarray(labels)
    if expected.shape[0] != len(plan.audit_ids):
        raise ValueError("audit labels do not align with the immutable cohort")
    if task == "classification":
        ensemble, parallel_latency = average_probabilities(predictions, plan)
        predicted = np.argmax(ensemble, axis=1)
        truth = expected.astype(int).reshape(-1)
        accuracy = float(np.mean(predicted == truth))
        return ensemble, {
            "accuracy": accuracy,
            "f1_macro": _macro_f1(truth, predicted),
            "quality_score": accuracy,
            "parallel_latency_s": parallel_latency,
            "serial_latency_s": float(sum(item.latency_s for item in predictions)),
        }
    if task == "regression":
        ensemble, parallel_latency = average_continuous_predictions(predictions, plan)
        truth = expected.astype(float).reshape(-1)
        residual = ensemble - truth
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        mae = float(np.mean(np.abs(residual)))
        return ensemble, {
            "rmse": rmse,
            "mae": mae,
            # Catalogue regression quality uses the same higher-is-better transform.
            "quality_score": float(1.0 / (1.0 + max(0.0, rmse))),
            "parallel_latency_s": parallel_latency,
            "serial_latency_s": float(sum(item.latency_s for item in predictions)),
        }
    raise ValueError("runtime composition validation supports classification or regression")


def run_parallel_predictions(
    service_ids: Sequence[str], provider: Callable[[str], ServicePrediction], *, max_parallelism: int
) -> tuple[list[ServicePrediction], float]:
    """Run every member; exceptions are propagated to preserve candidate identity."""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_parallelism), len(service_ids)))) as pool:
        futures = {pool.submit(provider, service_id): service_id for service_id in service_ids}
        outcomes: dict[str, ServicePrediction] = {}
        for future in as_completed(futures):
            outcomes[futures[future]] = future.result()
    return [outcomes[str(service_id)] for service_id in service_ids], time.perf_counter() - started


def measured_scores(
    selected: pd.DataFrame,
    request: Mapping[str, Any],
    *,
    ensemble_accuracy: float | None = None,
    actual_quality_score: float | None = None,
    parallel_latency_s: float,
) -> dict[str, float]:
    """Evaluate the frozen objective with observed quality and parallel latency."""
    qos = request_object(request, "qos_requirements")
    penalties = request_object(request, "constraint_penalties")
    weights = request_object(request, "preference_weights")
    objective_version = objective_version_for_request(request)
    scores = evaluate_provisional_composition(
        selected, qos_requirements=qos, objective_version=objective_version
    )
    quality = actual_quality_score if actual_quality_score is not None else ensemble_accuracy
    if quality is None or not np.isfinite(float(quality)):
        raise ValueError("measured composition quality must be finite")
    scores["mean_accuracy"] = float(quality)
    scores["accuracy_compatibility"] = float(quality)
    scores["total_latency"] = float(parallel_latency_s)
    scores["mean_latency"] = float(parallel_latency_s)
    scores["latency_feasibility"] = min(1.0, number(qos.get("max_latency"), 0.0) / max(parallel_latency_s, 1e-12)) if "max_latency" in qos else 1.0 / (1.0 + parallel_latency_s)
    penalty = calculate_constraint_penalty(
        scores, qos, penalties, objective_version=objective_version
    )
    utility = calculate_utility(
        scores, weights, qos, penalties, objective_version=objective_version
    )
    return {**scores, "constraint_penalty": penalty, "measured_utility": utility}


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    classes = np.unique(np.concatenate((labels.reshape(-1), predictions.reshape(-1))))
    if not len(classes):
        return 0.0
    scores: list[float] = []
    for value in classes:
        true_positive = float(np.sum((labels == value) & (predictions == value)))
        false_positive = float(np.sum((labels != value) & (predictions == value)))
        false_negative = float(np.sum((labels == value) & (predictions != value)))
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator <= 0.0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def bootstrap_interval(values: Sequence[float], *, seed: int = 42, draws: int = 2_000) -> tuple[float | None, float | None]:
    finite = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return None, None
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(finite, size=(int(draws), len(finite)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 2:
        return None
    left, right = (np.asarray(values, dtype=float) for values in zip(*pairs))
    if np.all(left == left[0]) or np.all(right == right[0]):
        return None
    return float(pd.Series(left).rank().corr(pd.Series(right).rank(), method="pearson"))


def evidence_summary(results: pd.DataFrame, *, seed: int = 42) -> dict[str, Any]:
    completed = results[results["validation_status"] == "completed"].copy()
    n = len(completed)
    correlation = spearman(completed.get("regenerated_surrogate_utility", []), completed.get("measured_utility", []))
    quality_correlation = spearman(completed.get("regenerated_surrogate_utility", []), completed.get("ensemble_accuracy", []))
    selected = completed[completed.get("selected_flag", False).astype(bool)] if "selected_flag" in completed else completed.iloc[0:0]
    best_utility = float(completed["measured_utility"].max()) if n else None
    best_quality = float(completed["ensemble_accuracy"].max()) if n else None
    selected_utility = float(selected.iloc[0]["measured_utility"]) if len(selected) else None
    selected_quality = float(selected.iloc[0]["ensemble_accuracy"]) if len(selected) else None
    utility_regret = None if selected_utility is None or best_utility is None else best_utility - selected_utility
    quality_regret = None if selected_quality is None or best_quality is None else best_quality - selected_quality
    ci = bootstrap_interval(completed["measured_utility"].tolist() if n else [], seed=seed)
    correlation_ci = _bootstrap_correlation(completed, "regenerated_surrogate_utility", "measured_utility", seed=seed)
    quality_correlation_ci = _bootstrap_correlation(completed, "regenerated_surrogate_utility", "ensemble_accuracy", seed=seed)
    utility_regret_ci = _bootstrap_regret(completed, "measured_utility", selected_utility, seed=seed)
    quality_regret_ci = _bootstrap_regret(completed, "ensemble_accuracy", selected_quality, seed=seed)
    # This label is deliberately secondary to the reported continuous evidence.
    if n < 5:
        tier = "insufficient_evidence"
    elif correlation is not None and correlation >= 0.70 and (utility_regret or 0.0) <= 0.02:
        tier = "strong"
    elif correlation is not None and correlation >= 0.40 and (utility_regret or 0.0) <= 0.05:
        tier = "moderate"
    else:
        tier = "weak"
    return {
        "candidate_count": int(len(results)), "completed_candidate_count": int(n),
        "unsupported_candidate_count": int((results["validation_status"] == "unsupported").sum()) if "validation_status" in results else 0,
        "correlation_spearman": correlation, "quality_correlation_spearman": quality_correlation,
        "correlation_spearman_ci95_low": correlation_ci[0], "correlation_spearman_ci95_high": correlation_ci[1],
        "quality_correlation_spearman_ci95_low": quality_correlation_ci[0], "quality_correlation_spearman_ci95_high": quality_correlation_ci[1],
        "selected_utility_regret": utility_regret, "selected_quality_regret": quality_regret,
        "selected_utility_regret_ci95_low": utility_regret_ci[0], "selected_utility_regret_ci95_high": utility_regret_ci[1],
        "selected_quality_regret_ci95_low": quality_regret_ci[0], "selected_quality_regret_ci95_high": quality_regret_ci[1],
        "measured_utility_ci95_low": ci[0], "measured_utility_ci95_high": ci[1], "evidence_tier": tier,
    }


def stability_summary(seed_predictions: Mapping[int, Mapping[str, ServicePrediction]], labels: Sequence[int]) -> dict[str, Any]:
    """Continuous seed-stability measures for a representative mini-campaign.

    Every seed must contain the same compatible services and audit IDs.  The
    result deliberately reports raw variation and pairwise rank agreement; it
    does not turn stability into a threshold decision.
    """
    service_rows: list[dict[str, Any]] = []
    for service_id in sorted({sid for predictions in seed_predictions.values() for sid in predictions}):
        entries = [items[service_id] for items in seed_predictions.values() if service_id in items]
        accuracies = [float((np.argmax(item.probabilities, axis=1) == np.asarray(labels)).mean()) for item in entries]
        agreements = []
        for left, right in combinations(entries, 2):
            agreements.append(float((np.argmax(left.probabilities, axis=1) == np.argmax(right.probabilities, axis=1)).mean()))
        service_rows.append({"service_id": service_id, "accuracy_mean": float(np.mean(accuracies)), "accuracy_std": float(np.std(accuracies)), "accuracy_range": float(np.ptp(accuracies)), "prediction_agreement_mean": float(np.mean(agreements)) if agreements else 1.0})
    candidate_by_seed: dict[int, dict[str, float]] = {}
    common_services = sorted(set.intersection(*(set(items) for items in seed_predictions.values()))) if seed_predictions else []
    expected_ids = tuple(next(iter(next(iter(seed_predictions.values())).values())).audit_ids) if common_services else ()
    expected_label_hash = next(iter(next(iter(seed_predictions.values())).values())).label_hash if common_services else ""
    for seed, items in seed_predictions.items():
        scores: dict[str, float] = {}
        for left, right in combinations(common_services, 2):
            pair = (items[left], items[right])
            if any(tuple(item.audit_ids) != expected_ids or item.label_hash != expected_label_hash for item in pair):
                raise CohortIsolationError("stability replicas do not share the exact audit cohort")
            probs = np.mean([item.probabilities for item in pair], axis=0)
            scores["|".join((left, right))] = float((np.argmax(probs, axis=1) == np.asarray(labels)).mean())
        candidate_by_seed[int(seed)] = scores
    ranking_rows: list[dict[str, Any]] = []
    for left_seed, right_seed in combinations(sorted(candidate_by_seed), 2):
        keys = sorted(set(candidate_by_seed[left_seed]).intersection(candidate_by_seed[right_seed]))
        left_scores = [candidate_by_seed[left_seed][key] for key in keys]
        right_scores = [candidate_by_seed[right_seed][key] for key in keys]
        ranking_rows.append({
            "seed_left": left_seed, "seed_right": right_seed, "candidate_count": len(keys),
            "spearman": spearman(left_scores, right_scores),
            "kendall_tau": _kendall_tau(left_scores, right_scores),
            "top_choice_overlap": bool(keys and max(keys, key=candidate_by_seed[left_seed].get) == max(keys, key=candidate_by_seed[right_seed].get)),
        })
    ensemble_values = [score for scores in candidate_by_seed.values() for score in scores.values()]
    ci = bootstrap_interval(ensemble_values)
    return {
        "service_stability": service_rows, "seeds": sorted(int(key) for key in seed_predictions),
        "ensemble_quality_mean": float(np.mean(ensemble_values)) if ensemble_values else None,
        "ensemble_quality_std": float(np.std(ensemble_values)) if ensemble_values else None,
        "ensemble_quality_ci95_low": ci[0], "ensemble_quality_ci95_high": ci[1],
        "candidate_scores_by_seed": candidate_by_seed, "candidate_ranking_stability": ranking_rows,
    }


def _stratified_positions(labels: Sequence[str], target: int, seed: int) -> set[int]:
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    total = len(labels)
    allocations = {label: min(len(indices), int(math.floor(target * len(indices) / total))) for label, indices in groups.items()}
    for label in sorted(groups):
        if allocations[label]:
            chosen.update(rng.choice(groups[label], allocations[label], replace=False).tolist())
    remaining = target - len(chosen)
    available = [index for index in range(total) if index not in chosen]
    if remaining:
        chosen.update(rng.choice(available, remaining, replace=False).tolist())
    return chosen


def _normalise_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise_json(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    return _json_scalar(value)


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(_normalise_json(value), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _kendall_tau(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    concordant = discordant = 0
    for i, j in combinations(range(len(left)), 2):
        sign = (left[i] - left[j]) * (right[i] - right[j])
        if sign > 0:
            concordant += 1
        elif sign < 0:
            discordant += 1
    denominator = concordant + discordant
    return None if not denominator else float((concordant - discordant) / denominator)


def _bootstrap_correlation(frame: pd.DataFrame, left: str, right: str, *, seed: int, draws: int = 2_000) -> tuple[float | None, float | None]:
    values = frame[[left, right]].dropna().to_numpy(dtype=float) if left in frame and right in frame else np.empty((0, 2))
    if len(values) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    samples = []
    for indices in rng.integers(0, len(values), size=(draws, len(values))):
        value = spearman(values[indices, 0], values[indices, 1])
        if value is not None and np.isfinite(value):
            samples.append(value)
    return (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))) if samples else (None, None)


def _bootstrap_regret(frame: pd.DataFrame, metric: str, selected_value: float | None, *, seed: int, draws: int = 2_000) -> tuple[float | None, float | None]:
    if selected_value is None or metric not in frame:
        return None, None
    values = frame[metric].dropna().to_numpy(dtype=float)
    if not len(values):
        return None, None
    rng = np.random.default_rng(seed)
    regrets = np.maximum(0.0, np.max(rng.choice(values, size=(draws, len(values)), replace=True), axis=1) - float(selected_value))
    return float(np.quantile(regrets, 0.025)), float(np.quantile(regrets, 0.975))
