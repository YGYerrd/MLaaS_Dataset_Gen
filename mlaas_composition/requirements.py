"""Generate controlled synthetic requests for collaborative composition."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .catalogue import normalise_service_rows
from .protocol import (
    DEFAULT_PRINCIPLE_ID,
    DEFAULT_PROFILE_ID,
    REQUEST_SCHEMA_VERSION,
    load_request_principle,
    load_request_profile,
)


FUNCTIONAL_FIELDS = (
    "task_family",
    "task_type",
    "modality",
    "dataset",
    "dataset_name",
    "dataset_config",
    "benchmark_split",
    "input_schema",
    "output_schema",
    "label_format",
)


def generate_service_requests(
    services: pd.DataFrame,
    *,
    request_count: int = 25,
    seed: int = 42,
    profile_id: str = DEFAULT_PROFILE_ID,
    principle_id: str = DEFAULT_PRINCIPLE_ID,
    compatibility_mode: str = "functional",
) -> pd.DataFrame:
    """Generate deterministic, synthetic collaborative service requests.

    Each record follows the collaborative portion of the MSC paper: functional
    requirements, composition-level QoS targets, preference weights, constraints,
    and a short peer-group size. Nested objects are JSON encoded so the returned
    frame round-trips through CSV.
    """
    profile = load_request_profile(profile_id)
    principle = load_request_principle(principle_id)
    services = normalise_service_rows(services)
    compatibility_mode = str(compatibility_mode).lower()
    if compatibility_mode not in {"functional", "strict"}:
        raise ValueError("compatibility_mode must be 'functional' or 'strict'")
    if profile.get("request_schema_version") == "5.0.0" and compatibility_mode != "strict":
        raise ValueError("measured task requests require strict compatibility")
    rng = np.random.default_rng(seed)
    strict_groups = _strict_snapshot_groups(services, profile) if compatibility_mode == "strict" else []
    schema_version = profile.get("request_schema_version", REQUEST_SCHEMA_VERSION)
    if compatibility_mode == "strict" and strict_groups and schema_version != "5.0.0":
        schema_version = "4.0.0" if "family_id" in strict_groups[0][0] else "3.0.0" if "cohort_id" in strict_groups[0][0] else "2.0.0"
    if compatibility_mode == "strict" and not strict_groups:
        raise ValueError("No viable compatible family, shared-parent cohort, or legacy FedAvg snapshot is available")
    functional_pools = (
        _functional_task_pools(services)
        if compatibility_mode == "functional"
        else {}
    )
    if compatibility_mode == "functional" and not functional_pools:
        raise ValueError("No completed services are available for functional request generation")
    requests = []
    for idx in range(int(request_count)):
        if compatibility_mode == "strict":
            compatibility_requirements, task_pool = strict_groups[
                int(rng.integers(len(strict_groups)))
            ]
            task_pool = task_pool.reset_index(drop=True)
        else:
            tasks = sorted(functional_pools)
            task = str(rng.choice(tasks))
            task_pool = functional_pools[task]
            compatibility_requirements = {}

        proposed_size = _sample_composition_size(task_pool, rng, profile)
        anchor = task_pool.iloc[int(rng.integers(len(task_pool)))]
        functional = _functional_requirements(task_pool, anchor, proposed_size)
        compatible_pool = _filter_functional_requirements(task_pool, functional)
        composition_size = min(proposed_size, len(compatible_pool))
        reference = compatible_pool.sample(n=composition_size, random_state=int(rng.integers(2**32 - 1)))

        hard_constraints = _hard_constraints(reference, rng, principle)
        qos_requirements = _qos_requirements(reference, rng, principle)
        preference_weights = _preference_weights(rng, profile)
        constraint_penalties = _constraint_penalties(rng, profile)
        request = {
            "request_id": f"req_{idx:06d}",
            "request_schema_version": schema_version,
            "request_profile_id": profile["profile_id"],
            "principle_id": principle["principle_id"],
            "generation_seed": int(seed),
            "composition_type": "collaborative",
            "compatibility_mode": compatibility_mode,
            "compatibility_requirements": _json(compatibility_requirements),
            "eligible_pool_snapshot_hash": None,
            "functional_requirements": _json(functional),
            "qos_requirements": _json(qos_requirements),
            "hard_constraints": _json(hard_constraints),
            "composition_size": composition_size,
            "preference_weights": _json(preference_weights),
            "constraint_penalties": _json(constraint_penalties),
        }
        if schema_version == "5.0.0":
            request["required_capabilities"] = _json({"parameter_aggregation": True, "measured_composition_validation": True})
        if compatibility_mode == "strict":
            strict_eligible = _apply_v2_constraints(compatible_pool, hard_constraints)
            from .pool import eligible_pool_snapshot_hash

            request["eligible_pool_snapshot_hash"] = eligible_pool_snapshot_hash(
                strict_eligible["service_id"]
            )
        requests.append(request)
    return pd.DataFrame(requests)


def _functional_task_pools(services: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from .pool import build_composition_instance

    result: dict[str, pd.DataFrame] = {}
    for task in sorted(services["task_family"].dropna().astype(str).unique()):
        request = {
            "request_schema_version": REQUEST_SCHEMA_VERSION,
            "composition_type": "collaborative",
            "compatibility_mode": "functional",
            "compatibility_requirements": {},
            "functional_requirements": {"task_family": task},
            "hard_constraints": {},
            "composition_size": 1,
        }
        pool = build_composition_instance(services, request).C_R
        if not pool.empty:
            result[task] = pool
    return result


def _strict_snapshot_groups(
    services: pd.DataFrame, profile: dict
) -> list[tuple[dict[str, str | int], pd.DataFrame]]:
    if profile.get("request_schema_version") == "5.0.0":
        from .validation import supported_composition_reason
        groups = []
        for generation, keys, eligible_mask in (
            ("independent", ["family_id", "initial_state_hash", "compatibility_contract_hash"], services.composition_compatible.eq(1)),
            ("shared_parent", ["cohort_id", "parent_state_hash", "compatibility_contract_hash"], services.cohort_succeeded.eq(1))):
            keys += ["task_semantics_hash", "evaluation_protocol_hash"]
            candidates = services[services.service_status.eq("completed") & services.generation_mode.eq(generation) & eligible_mask]
            for values, frame in candidates.dropna(subset=keys).groupby(keys, sort=True):
                if len(frame) >= int(profile["composition_size"]["minimum"]) and supported_composition_reason(frame) is None:
                    groups.append((dict(zip(keys, values)), frame.sort_values("service_id").reset_index(drop=True)))
        return groups
    if profile.get("request_schema_version", "3.0.0") in {"3.0.0", "4.0.0"}:
        keys = ["family_id", "initial_state_hash", "compatibility_contract_hash"]
        common = ["architecture_id", "parameter_schema_hash", "logical_dataset_contract_hash", "preprocessing_contract_hash", "label_map_hash"]
        candidates = services[services["service_status"].eq("completed") & services["composition_compatible"].eq(1)
                              & services["family_contract_version"].eq("independent_family_v1")]
        groups = []
        for values, frame in candidates.dropna(subset=keys + common).groupby(keys, sort=True):
            if len(frame) >= int(profile["composition_size"]["minimum"]) and all(frame[k].nunique() == 1 for k in common):
                groups.append((dict(zip(keys, values)), frame.sort_values("service_id").reset_index(drop=True)))
        if groups or profile.get("request_schema_version") == "4.0.0":
            return groups
    if profile.get("request_schema_version", "3.0.0") == "3.0.0":
        common = ["architecture_id", "parameter_schema_hash", "logical_dataset_contract_hash",
                  "preprocessing_contract_hash", "label_map_hash"]
        keys = ["cohort_id", "parent_state_hash", "compatibility_contract_hash"]
        candidates = services[services["service_status"].eq("completed") & services["cohort_succeeded"].eq(1)]
        candidates = candidates.dropna(subset=common + keys + ["client_id"])
        groups = []
        for values, frame in candidates.groupby(keys, sort=True):
            if len(frame) < int(profile["composition_size"]["minimum"]):
                continue
            if any(frame[field].nunique() != 1 for field in common) or frame.client_id.duplicated().any():
                continue
            groups.append((dict(zip(keys, map(str, values))), frame.sort_values("client_id").reset_index(drop=True)))
        if groups:
            return groups
    required = (
        "architecture_id", "parameter_schema_hash", "logical_dataset_contract_hash",
        "preprocessing_contract_hash", "label_map_hash", "fl_run_id", "client_id",
        "round_id", "compatibility_contract_hash", "parent_global_round", "parent_state_hash",
    )
    mask = (
        services["service_status"].eq("completed")
        & services["fl_run_id"].notna()
        & services["participated"].map(lambda value: value == 1)
        & services["fedavg_succeeded"].map(lambda value: value == 1)
    )
    candidates = services[mask].dropna(subset=list(required)).copy()
    candidates = candidates[
        candidates.apply(
            lambda row: int(row["round_id"]) == int(row["parent_global_round"]) + 1,
            axis=1,
        )
    ]
    keys = [
        "compatibility_contract_hash", "fl_run_id", "parent_global_round", "parent_state_hash"
    ]
    minimum = int(profile["composition_size"]["minimum"])
    groups: list[tuple[dict[str, str | int], pd.DataFrame]] = []
    for values, frame in candidates.groupby(keys, sort=True, dropna=False):
        frame = frame.sort_values(
            ["fl_run_id", "client_id", "round_id", "service_id"], kind="stable"
        ).drop_duplicates(["fl_run_id", "client_id", "round_id"], keep="first")
        if len(frame) < minimum:
            continue
        if any(frame[field].nunique(dropna=False) != 1 for field in required[:5]):
            continue
        requirements = {
            "compatibility_contract_hash": str(values[0]),
            "fl_run_id": str(values[1]),
            "parent_global_round": int(values[2]),
            "parent_state_hash": str(values[3]),
        }
        groups.append((requirements, frame.reset_index(drop=True)))
    return groups


def _sample_composition_size(pool: pd.DataFrame, rng: np.random.Generator, profile: dict) -> int:
    if len(pool) <= 1:
        return 1
    bounds = profile["composition_size"]
    minimum = min(int(bounds["minimum"]), len(pool))
    maximum = min(int(bounds["maximum"]), len(pool))
    return int(rng.integers(minimum, maximum + 1))


def _functional_requirements(pool: pd.DataFrame, anchor: pd.Series, composition_size: int) -> dict[str, str]:
    requirements = {"task_family": str(anchor["task_family"])}
    eligible = pool.copy()
    for field in FUNCTIONAL_FIELDS[1:]:
        value = str(anchor[field]).strip().lower()
        if value in {"", "unknown", "none", "null", "nan"}:
            continue
        narrowed = eligible[eligible[field].astype(str).str.lower() == value]
        if len(narrowed) >= composition_size:
            requirements[field] = value
            eligible = narrowed
    return requirements


def _filter_functional_requirements(pool: pd.DataFrame, requirements: dict[str, str]) -> pd.DataFrame:
    eligible = pool
    for field, value in requirements.items():
        eligible = eligible[eligible[field].astype(str).str.lower() == str(value).lower()]
    return eligible


def _hard_constraints(reference: pd.DataFrame, rng: np.random.Generator, principle: dict) -> dict[str, float]:
    """Set feasible per-service bounds from a sampled reference composition."""
    low_min, low_max = principle["minimum_target_multiplier"]
    high_min, high_max = principle["maximum_target_multiplier"]
    return {
        "min_metric_score": float(np.clip(reference["metric_score"].min() * rng.uniform(low_min, low_max), 0.0, 1.0)),
        "max_latency": float(reference["latency"].max() * rng.uniform(high_min, high_max)),
        "min_resource_cost_score": float(np.clip(reference["resource_cost_score"].min() * rng.uniform(low_min, low_max), 0.0, 1.0)),
        "min_reliability_score": float(np.clip(reference["reliability_score"].min() * rng.uniform(low_min, low_max), 0.0, 1.0)),
        "min_explainability_score": float(np.clip(reference["explainability_score"].min() * rng.uniform(low_min, low_max), 0.0, 1.0)),
        "max_compute_time": float(reference["compute_time_s"].max() * rng.uniform(high_min, high_max)),
    }


def _qos_requirements(reference: pd.DataFrame, rng: np.random.Generator, principle: dict) -> dict[str, float]:
    """Create feasible composition-level targets used by the penalty objective."""
    low_min, low_max = principle["minimum_target_multiplier"]
    high_min, high_max = principle["maximum_target_multiplier"]
    return {
        "min_accuracy": float(np.clip(reference["metric_score"].mean() * rng.uniform(low_min, low_max), 0.0, 1.0)),
        "max_latency": float(reference["latency"].sum() * rng.uniform(high_min, high_max)),
        "max_compute_time": float(reference["compute_time_s"].mean() * rng.uniform(high_min, high_max)),
        "min_reliability": float(np.clip(reference["reliability_score"].mean() * rng.uniform(low_min, low_max), 0.0, 1.0)),
        "min_explainability": float(np.clip(reference["explainability_score"].mean() * rng.uniform(low_min, low_max), 0.0, 1.0)),
        "min_resource_cost_score": float(np.clip(reference["resource_cost_score"].mean() * rng.uniform(low_min, low_max), 0.0, 1.0)),
    }


def _preference_weights(rng: np.random.Generator, profile: dict) -> dict[str, float]:
    names = (
        "data_parameters", "model_parameters", "compute_time", "latency",
        "accuracy", "reliability", "explainability", "resource_cost",
    )
    values = rng.dirichlet(np.full(len(names), float(profile["preference_weight_dirichlet_alpha"])))
    return {name: float(value) for name, value in zip(names, values)}


def _constraint_penalties(rng: np.random.Generator, profile: dict) -> dict[str, float]:
    names = (
        "min_accuracy", "max_latency", "max_compute_time", "min_reliability",
        "min_explainability", "min_resource_cost_score",
    )
    values = rng.dirichlet(np.full(len(names), float(profile["constraint_penalty_dirichlet_alpha"])))
    return {name: float(value) for name, value in zip(names, values)}


def _json(value: dict[str, str] | dict[str, float]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _apply_v2_constraints(pool: pd.DataFrame, constraints: dict[str, float]) -> pd.DataFrame:
    eligible = pool
    lower = {
        "min_metric_score": "metric_score",
        "min_resource_cost_score": "resource_cost_score",
        "min_reliability_score": "reliability_score",
        "min_explainability_score": "explainability_score",
    }
    upper = {"max_latency": "latency", "max_compute_time": "compute_time_s"}
    for name, field in lower.items():
        eligible = eligible[eligible[field].astype(float) >= float(constraints[name])]
    for name, field in upper.items():
        eligible = eligible[eligible[field].astype(float) <= float(constraints[name])]
    return eligible.sort_values("service_id", kind="stable").reset_index(drop=True)
