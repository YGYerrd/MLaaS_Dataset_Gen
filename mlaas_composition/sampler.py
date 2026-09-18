"""Baseline collaborative candidate sampling from synthetic requirements."""

from __future__ import annotations

import json
from itertools import combinations, islice
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .catalogue import finite, number
from .pool import build_composition_instance, request_object
from .protocol import (
    LEGACY_OBJECTIVE_VERSION,
    OBJECTIVE_VERSION,
    RECIPE_OBJECTIVE_VERSION,
    objective_version_for_request,
)
from .signatures import compute_composition_mus


def generate_compositions(services: pd.DataFrame, requests: pd.DataFrame, *, candidates_per_request: int = 10, seed: int = 42) -> pd.DataFrame:
    """Sample peer-service groups for collaborative requirements only.

    Requirements must declare ``composition_type='collaborative'``, functional
    requirements, QoS targets, hard constraints, a ``composition_size``, and
    preference weights. Hard constraints are applied before candidate groups are
    formed; QoS targets contribute a penalty to the group-level objective.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _, request_row in requests.iterrows():
        request = request_row.to_dict()
        composition_type = str(request.get("composition_type") or "collaborative").lower()
        if composition_type != "collaborative":
            raise ValueError(f"Unsupported composition_type '{composition_type}'; only collaborative composition is supported")
        instance = build_composition_instance(services, request)
        objective_version = objective_version_for_request(request)
        preference_weights = request_object(request, "preference_weights")
        qos_requirements = _optional_request_object(request, "qos_requirements")
        constraint_penalties = _optional_request_object(request, "constraint_penalties")
        if instance.omega_size == 0:
            continue
        shuffled_services = instance.C_R.iloc[rng.permutation(len(instance.C_R))].reset_index(drop=True)
        for candidate_idx, combo in enumerate(islice(instance.iter_candidate_indices(), int(candidates_per_request))):
            selected = shuffled_services.iloc[list(combo)].reset_index(drop=True)
            scores = evaluate_provisional_composition(
                selected,
                qos_requirements=qos_requirements,
                objective_version=objective_version,
            )
            penalty = calculate_constraint_penalty(
                scores,
                qos_requirements,
                constraint_penalties,
                objective_version=objective_version,
            )
            score = calculate_utility(
                scores,
                preference_weights,
                qos_requirements,
                constraint_penalties,
                objective_version=objective_version,
            )
            rows.append({
                "request_id": str(request.get("request_id")),
                "candidate_id": f"{request.get('request_id')}_cand_{candidate_idx:03d}",
                "service_ids": json.dumps(selected["service_id"].astype(str).tolist()),
                "composition_type": "collaborative",
                "compatibility_mode": instance.compatibility_mode,
                "requested_pool_snapshot_hash": instance.requested_pool_snapshot_hash,
                "computed_pool_snapshot_hash": instance.computed_pool_snapshot_hash,
                "objective_version": objective_version,
                "composition_size": len(selected),
                "eligible_pool_size": len(instance.C_R),
                "candidate_space_size": instance.omega_size,
                "evaluator_name": "paper_informed_collaborative_heuristic",
                **scores,
                "constraint_penalty": penalty,
                "composability_score": score,
                "utility_score": score,
                **({"predicted_utility_score": score} if objective_version == RECIPE_OBJECTIVE_VERSION else {}),
                "selected_flag": False,
            })
    compositions = pd.DataFrame(rows)
    if compositions.empty:
        return compositions
    for _, group in compositions.groupby("request_id"):
        compositions.loc[group["utility_score"].idxmax(), "selected_flag"] = True
    for column in (
        "data_parameter_compatibility", "model_parameter_alignment", "compute_time_compatibility",
        "latency_feasibility", "accuracy_compatibility", "reliability_compatibility",
        "trust_score", "explainability_score", "cost_efficiency", "constraint_penalty", "dhs", "mus", "shs",
        "ses", "hsq", "srs", "composability_score", "utility_score",
    ):
        compositions[column] = compositions[column].astype(float).clip(0.0, 1.0)
    return compositions


def evaluate_provisional_composition(
    selected: pd.DataFrame,
    *,
    qos_requirements: Mapping[str, Any] | None = None,
    objective_version: str = OBJECTIVE_VERSION,
) -> dict[str, Any]:
    """Score a collaborative group with an explicit trust dimension.

    Update-signature cosine similarity is the available proxy for model-update
    alignment. This remains a ranking surrogate, not a measured ensemble result.
    """
    qos_requirements = qos_requirements or {}
    mean_latency = float(selected["latency"].astype(float).mean())
    total_latency = float(selected["latency"].astype(float).sum())
    mean_compute_time = float(selected["compute_time_s"].astype(float).mean())
    mean_accuracy = float(selected["metric_score"].astype(float).mean())
    if objective_version == RECIPE_OBJECTIVE_VERSION:
        counts = pd.to_numeric(selected["realised_sample_count"], errors="coerce").to_numpy(dtype=float)
        quality = pd.to_numeric(selected["metric_score"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(counts).all() or (counts <= 0).any() or not np.isfinite(quality).all():
            raise ValueError("recipe scoring requires verified training counts and finite quality")
        mean_accuracy = float(np.average(quality, weights=counts))
        total_latency = mean_latency
    mean_reliability = float(selected["reliability_score"].astype(float).mean())
    mean_trust = _complete_mean(selected["trust_score"])
    mean_explainability = _complete_mean(selected["explainability_score"])
    mean_resource_cost = float(selected["resource_cost_score"].astype(float).mean())
    if objective_version == LEGACY_OBJECTIVE_VERSION:
        data_compatibility = _dominant_fraction(selected["data_distribution"])
        data_evidence: dict[str, Any] | None = None
        indeterminate_fraction = None
        cost_efficiency = float(np.clip(1.0 - mean_resource_cost, 0.0, 1.0))
    elif objective_version in {OBJECTIVE_VERSION, RECIPE_OBJECTIVE_VERSION}:
        data_compatibility, data_evidence, indeterminate_fraction = _data_parameter_compatibility(selected)
        cost_efficiency = float(np.clip(mean_resource_cost, 0.0, 1.0))
    else:
        raise ValueError(f"Unsupported objective_version '{objective_version}'")
    latency_budget = number(qos_requirements.get("max_latency"), np.nan)
    latency_feasibility = (
        min(1.0, latency_budget / max(total_latency, 1e-12))
        if np.isfinite(latency_budget) and latency_budget >= 0.0
        else 1.0 / (1.0 + total_latency)
    )
    scores: dict[str, Any] = {
        # Paper-aligned collaborative dimensions (Table 4).
        "data_parameter_compatibility": data_compatibility,
        "model_parameter_alignment": compute_composition_mus(selected.to_dict(orient="records")),
        "compute_time_compatibility": _homogeneity(selected["compute_time_s"]),
        "latency_feasibility": float(latency_feasibility),
        "accuracy_compatibility": mean_accuracy if objective_version == RECIPE_OBJECTIVE_VERSION else _homogeneity(selected["metric_score"]),
        "reliability_compatibility": _homogeneity(selected["reliability_score"]),
        "trust_score": mean_trust,
        "explainability_score": mean_explainability,
        "cost_efficiency": cost_efficiency,
        # Aggregates are retained for QoS-target penalties and reportability.
        "mean_accuracy": mean_accuracy,
        "mean_reliability": mean_reliability,
        "mean_trust": mean_trust,
        "mean_explainability": mean_explainability,
        "mean_compute_time": mean_compute_time,
        "total_latency": total_latency,
        # Legacy aliases retained for existing downstream CSV consumers.
        "dhs": data_compatibility,
        "mus": compute_composition_mus(selected.to_dict(orient="records")),
        "shs": float(np.mean([_dominant_fraction(selected[key]) for key in ("task_family", "modality", "model_type")])),
        "ses": _service_efficiency_score(selected, objective_version=objective_version),
        "hsq": mean_accuracy,
        "srs": float(np.mean([mean_reliability, mean_trust, mean_explainability])),
        "mean_latency": mean_latency,
    }
    if objective_version == LEGACY_OBJECTIVE_VERSION:
        scores["mean_resource_cost"] = mean_resource_cost
    else:
        scores["mean_resource_cost_score"] = mean_resource_cost
        scores["data_parameter_evidence_json"] = json.dumps(
            data_evidence, sort_keys=True, separators=(",", ":")
        )
        scores["data_parameter_indeterminate_fraction"] = indeterminate_fraction
    if objective_version == RECIPE_OBJECTIVE_VERSION:
        scores.update(predicted_quality_score=mean_accuracy, predicted_latency_s_per_example=mean_latency,
                      prediction_method="sample_weighted_catalogue_quality_v1",
                      quality_metric_name=str(selected["primary_metric_name"].iloc[0]),
                      latency_unit="seconds_per_example")
    return scores


def _complete_mean(values: pd.Series) -> float:
    """A group has no measured mean if any member lacks a measurement."""
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if len(numeric) and numeric.map(finite).all() else float("nan")


def calculate_utility(
    scores: Mapping[str, float],
    weights: Mapping[str, Any],
    qos_requirements: Mapping[str, Any] | None = None,
    constraint_penalties: Mapping[str, Any] | None = None,
    *,
    objective_version: str = OBJECTIVE_VERSION,
) -> float:
    """Combine paper-aligned dimensions and subtract QoS-target penalties."""
    dimensions = {
        "data_parameters": scores.get("data_parameter_compatibility", scores["dhs"]),
        "model_parameters": scores.get("model_parameter_alignment", scores["mus"]),
        "compute_time": scores.get("compute_time_compatibility", scores["ses"]),
        "latency": scores.get("latency_feasibility", 1.0 / (1.0 + scores["mean_latency"])),
        "accuracy": scores.get("accuracy_compatibility", scores["hsq"]),
        "reliability": scores.get("reliability_compatibility", scores["srs"]),
        "trust": scores.get("trust_score", float("nan")),
        "explainability": scores.get("explainability_score", float("nan")),
        "resource_cost": scores.get(
            "cost_efficiency",
            (
                1.0 - scores["mean_resource_cost"]
                if objective_version == LEGACY_OBJECTIVE_VERSION
                else scores.get("mean_resource_cost_score", 0.5)
            ),
        ),
    }
    effective_weights = {name: max(0.0, number(weights.get(name), 0.0)) for name in dimensions}
    legacy_weights = {
        "quality": "accuracy", "diversity": "model_parameters",
    }
    for old_name, new_name in legacy_weights.items():
        effective_weights[new_name] += max(0.0, number(weights.get(old_name), 0.0))
    total = sum(effective_weights.values())
    if total == 0.0:
        effective_weights = {name: 1.0 for name in dimensions}
        total = float(len(effective_weights))
    active = {name: weight for name, weight in effective_weights.items() if weight > 0.0}
    missing = [name for name in active if not finite(dimensions[name])]
    if missing:
        raise ValueError(f"Requested objective dimensions lack measured scores: {missing}")
    # Never multiply missing, unrequested dimensions by zero (NaN * 0 is NaN).
    objective = sum(dimensions[name] * weight for name, weight in active.items()) / total
    return float(np.clip(
        objective
        - calculate_constraint_penalty(
            scores,
            qos_requirements,
            constraint_penalties,
            objective_version=objective_version,
        ),
        0.0,
        1.0,
    ))


def calculate_constraint_penalty(
    scores: Mapping[str, float],
    qos_requirements: Mapping[str, Any] | None,
    constraint_penalties: Mapping[str, Any] | None,
    *,
    objective_version: str = OBJECTIVE_VERSION,
) -> float:
    """Return the normalized squared-deviation penalty from the paper objective."""
    if not qos_requirements:
        return 0.0
    penalties = constraint_penalties or {}
    actuals: dict[str, float] = {
        "min_accuracy": scores["mean_accuracy"],
        "max_latency": scores["total_latency"],
        "max_compute_time": scores["mean_compute_time"],
        "min_reliability": scores["mean_reliability"],
        "min_trust": scores.get("mean_trust", float("nan")),
        "min_explainability": scores["mean_explainability"],
    }
    if objective_version == LEGACY_OBJECTIVE_VERSION:
        actuals["max_resource_cost"] = scores["mean_resource_cost"]
    elif objective_version in {OBJECTIVE_VERSION, RECIPE_OBJECTIVE_VERSION}:
        actuals["min_resource_cost_score"] = scores.get(
            "mean_resource_cost_score", scores.get("mean_resource_cost", 0.5)
        )
    else:
        raise ValueError(f"Unsupported objective_version '{objective_version}'")
    violations: list[tuple[float, float]] = []
    for name, actual in actuals.items():
        target = number(qos_requirements.get(name), np.nan)
        if not np.isfinite(target):
            continue
        if not finite(actual):
            raise ValueError(f"Requested QoS target '{name}' lacks a measured score")
        shortfall = target - actual if name.startswith("min_") else actual - target
        deviation = max(0.0, shortfall) / max(abs(target), 1e-12)
        violations.append((max(0.0, number(penalties.get(name), 1.0)), deviation**2))
    if not violations:
        return 0.0
    total_weight = sum(weight for weight, _ in violations)
    return float(sum(weight * value for weight, value in violations) / total_weight) if total_weight else 0.0


def _service_efficiency_score(
    selected: pd.DataFrame, *, objective_version: str = OBJECTIVE_VERSION
) -> float:
    latency = 1.0 / (1.0 + float(selected["latency"].astype(float).mean()))
    compute = 1.0 / (1.0 + float(selected["compute_time_s"].astype(float).mean()))
    mean_score = float(selected["resource_cost_score"].astype(float).mean())
    cost = 1.0 - mean_score if objective_version == LEGACY_OBJECTIVE_VERSION else mean_score
    return float(np.clip(np.mean([latency, compute, cost]), 0.0, 1.0))


def _dominant_fraction(values) -> float:
    items = [str(value) for value in values if value is not None]
    return 0.5 if not items else float(max(items.count(item) for item in set(items)) / len(items))


def _homogeneity(values) -> float:
    numbers = np.asarray(values, dtype=float)
    if not len(numbers):
        return 0.5
    centre = float(numbers.mean())
    scale = max(abs(centre), float(np.max(np.abs(numbers))), 1e-12)
    spread = float(np.mean(np.abs(numbers - centre)) / scale)
    return float(np.clip(1.0 - spread, 0.0, 1.0))


_DATA_EVIDENCE_DIMENSIONS = {
    "logical_dataset": (
        "logical_dataset_contract_hash",
        ("dataset", "dataset_name", "dataset_config", "train_split", "benchmark_split"),
    ),
    "preprocessing_input": (
        "preprocessing_contract_hash",
        ("modality", "hf_task", "input_schema"),
    ),
    "label_output": (
        "label_map_hash",
        ("task_family", "output_schema", "label_format"),
    ),
}


def _data_parameter_compatibility(
    selected: pd.DataFrame,
) -> tuple[float, dict[str, dict[str, int]], float]:
    audit = {
        dimension: {
            "exact_matches": 0,
            "exact_mismatches": 0,
            "legacy_matches": 0,
            "legacy_mismatches": 0,
            "indeterminate": 0,
        }
        for dimension in _DATA_EVIDENCE_DIMENSIONS
    }
    rows = selected.to_dict(orient="records")
    pairs = list(combinations(rows, 2)) if len(rows) > 1 else [(rows[0], rows[0])]
    values: list[float] = []
    for dimension, (hash_field, fallback_fields) in _DATA_EVIDENCE_DIMENSIONS.items():
        for left, right in pairs:
            left_hash = _evidence_value(left.get(hash_field))
            right_hash = _evidence_value(right.get(hash_field))
            if left_hash is not None and right_hash is not None:
                matches = left_hash == right_hash
                audit[dimension]["exact_matches" if matches else "exact_mismatches"] += 1
                values.append(1.0 if matches else 0.0)
                continue
            left_fallback = _fallback_tuple(left, fallback_fields)
            right_fallback = _fallback_tuple(right, fallback_fields)
            if left_fallback is not None and right_fallback is not None:
                matches = left_fallback == right_fallback
                audit[dimension]["legacy_matches" if matches else "legacy_mismatches"] += 1
                values.append(1.0 if matches else 0.0)
                continue
            audit[dimension]["indeterminate"] += 1
            values.append(0.5)
    indeterminate = sum(item["indeterminate"] for item in audit.values())
    return float(np.mean(values)), audit, float(indeterminate / len(values))


def _fallback_tuple(
    row: Mapping[str, Any], fields: tuple[str, ...]
) -> tuple[str, ...] | None:
    values = tuple(_evidence_value(row.get(field), normalize=True) for field in fields)
    return None if any(value is None for value in values) else values  # type: ignore[return-value]


def _evidence_value(value: Any, *, normalize: bool = False) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text.lower() in {"", "unknown", "none", "null", "nan", "n/a", "not available"}:
        return None
    return text.lower() if normalize else text


def _optional_request_object(request: Mapping[str, Any], field: str) -> dict[str, Any]:
    return request_object(request, field) if request.get(field) is not None else {}
