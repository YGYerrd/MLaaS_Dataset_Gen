"""Construct collaborative composition instances and implicit search spaces."""

from __future__ import annotations

import itertools
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import pandas as pd

from .catalogue import finite, normalise_service_rows, number
from .protocol import (
    LEGACY_REQUEST_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    objective_version_for_schema,
    request_schema_version,
)
from .requirements import FUNCTIONAL_FIELDS


LOWER_BOUND_CONSTRAINTS = {
    "min_metric_score": "metric_score",
    "min_reliability_score": "reliability_score",
    "min_explainability_score": "explainability_score",
    "min_data_volume": "data_volume",
}
V2_LOWER_BOUND_CONSTRAINTS = {
    **LOWER_BOUND_CONSTRAINTS,
    "min_resource_cost_score": "resource_cost_score",
}
UPPER_BOUND_CONSTRAINTS = {
    "max_latency": "latency",
    "max_compute_time": "compute_time_s",
    "max_resource_cost_score": "resource_cost_score",
}
V2_UPPER_BOUND_CONSTRAINTS = {
    key: value for key, value in UPPER_BOUND_CONSTRAINTS.items()
    if key != "max_resource_cost_score"
}

STRICT_PROVENANCE_FIELDS = (
    "architecture_id",
    "parameter_schema_hash",
    "logical_dataset_contract_hash",
    "preprocessing_contract_hash",
    "label_map_hash",
)


class StrictCompatibilityUnsupported(ValueError):
    """Raised when a catalogue has no provenance capable of strict matching."""


class ProvenanceIntegrityError(ValueError):
    """Raised when matching strict provenance is incomplete or contradictory."""


class PoolSnapshotMismatch(ValueError):
    """Raised when an immutable strict request reconstructs different members."""

    def __init__(self, requested_hash: str, computed_hash: str) -> None:
        self.requested_hash = requested_hash
        self.computed_hash = computed_hash
        super().__init__(
            "Strict eligible pool membership changed: "
            f"requested {requested_hash}, computed {computed_hash}"
        )


@dataclass(frozen=True)
class CompositionInstance:
    """One collaborative instance: eligible pool ``C_R`` and implicit ``Omega_R``.

    ``eligible_services`` is the materialised catalogue subset ``C_R``.
    ``Omega_R`` is represented only by its cardinality and lazy iterators, so
    constructing an instance never allocates all possible service groups.
    """

    eligible_services: pd.DataFrame
    composition_size: int
    functional_requirements: Mapping[str, Any]
    hard_constraints: Mapping[str, Any]
    compatibility_mode: str = "functional"
    compatibility_requirements: Mapping[str, Any] | None = None
    requested_pool_snapshot_hash: str | None = None
    computed_pool_snapshot_hash: str | None = None
    objective_version: str | None = None

    @property
    def C_R(self) -> pd.DataFrame:
        """Return the eligible service pool ``C_R``."""
        return self.eligible_services

    @property
    def omega_size(self) -> int:
        """Return ``|Omega_R|`` without materialising its service groups."""
        if self.composition_size > len(self.eligible_services):
            return 0
        return math.comb(len(self.eligible_services), self.composition_size)

    @property
    def candidate_space_size(self) -> int:
        """Alias for ``omega_size`` used by existing output tables."""
        return self.omega_size

    def iter_candidate_indices(self) -> Iterator[tuple[int, ...]]:
        """Yield candidate indices lazily from ``Omega_R``."""
        return itertools.combinations(range(len(self.eligible_services)), self.composition_size)

    def iter_candidate_service_ids(self) -> Iterator[tuple[str, ...]]:
        """Yield candidate service-ID groups lazily from ``Omega_R``."""
        ids = self.eligible_services["service_id"].astype(str).tolist()
        for indices in self.iter_candidate_indices():
            yield tuple(ids[index] for index in indices)


def build_composition_instance(services: pd.DataFrame, request: Mapping[str, Any]) -> CompositionInstance:
    """Create one instance by applying a request's contract and hard bounds.

    No fallback is applied when filtering produces an empty pool. An empty
    ``C_R`` is an infeasible composition instance and remains visible to search
    algorithms and the benchmark harness.
    """
    if str(request.get("composition_type") or "collaborative").lower() != "collaborative":
        raise ValueError("Only collaborative composition requests are supported")

    functional_requirements = request_object(request, "functional_requirements")
    hard_constraints = request_object(request, "hard_constraints")
    composition_size = _composition_size(request)
    schema_version = request_schema_version(request)
    _validate_functional_requirements(functional_requirements)
    _validate_hard_constraints(hard_constraints, schema_version=schema_version)

    normalized = normalise_service_rows(services)
    if schema_version == "5.0.0":
        from .protocol import _validate_v2_request
        _validate_v2_request(request)
    strict_provenance_available = bool(
        normalized["composition_compatible"].eq(1).any()
        or
        (normalized["cohort_id"].notna() & normalized["cohort_succeeded"].eq(1)).any()
        or (
        normalized["fl_run_id"].notna().any()
        and normalized["compatibility_contract_hash"].notna().any()
        and normalized["participated"].notna().any()
        and normalized["fedavg_succeeded"].notna().any()
        )
    )
    eligible = normalized
    if schema_version not in {"4.0.0", "5.0.0"}:
        eligible = eligible[eligible["family_contract_version"].isna()]
    elif str(request.get("compatibility_mode")) != "strict":
        raise ValueError("family requests require strict compatibility")
    for field, required_value in functional_requirements.items():
        eligible = eligible[
            eligible[field].astype(str).str.lower() == str(required_value).strip().lower()
        ]

    if schema_version == LEGACY_REQUEST_SCHEMA_VERSION:
        eligible = _apply_hard_constraints(
            eligible,
            hard_constraints,
            lower=LOWER_BOUND_CONSTRAINTS,
            upper=UPPER_BOUND_CONSTRAINTS,
        ).reset_index(drop=True)
        computed_hash = eligible_pool_snapshot_hash(eligible["service_id"])
        return CompositionInstance(
            eligible_services=eligible,
            composition_size=composition_size,
            functional_requirements=functional_requirements,
            hard_constraints=hard_constraints,
            compatibility_mode="functional",
            compatibility_requirements={},
            computed_pool_snapshot_hash=computed_hash,
            objective_version=objective_version_for_schema(schema_version),
        )

    compatibility_mode = str(request.get("compatibility_mode") or "").lower()
    compatibility_requirements = request_object(request, "compatibility_requirements")
    eligible = _retain_completed_successful_observations(eligible)
    requested_hash = None

    if compatibility_mode == "functional":
        eligible = _deduplicate_latest_fedavg_observations(eligible)
    elif compatibility_mode == "strict":
        if not strict_provenance_available:
            raise StrictCompatibilityUnsupported(
                "Strict compatibility requires normalized generation provenance"
            )
        eligible = _filter_strict_snapshot(eligible, compatibility_requirements)
        if not eligible.empty:
            if schema_version == "5.0.0":
                from .validation import supported_composition_reason
                if reason := supported_composition_reason(eligible):
                    raise ProvenanceIntegrityError(reason)
            eligible = _deduplicate_strict_snapshot(eligible)
            _validate_strict_provenance(eligible)
        else:
            empty_hash = eligible_pool_snapshot_hash(())
            return CompositionInstance(
                eligible_services=eligible.reset_index(drop=True),
                composition_size=composition_size,
                functional_requirements=functional_requirements,
                hard_constraints=hard_constraints,
                compatibility_mode="strict",
                compatibility_requirements=compatibility_requirements,
                requested_pool_snapshot_hash=str(request.get("eligible_pool_snapshot_hash") or ""),
                computed_pool_snapshot_hash=empty_hash,
                objective_version=objective_version_for_schema(schema_version),
            )
        requested_hash = str(request.get("eligible_pool_snapshot_hash") or "").lower()
    else:
        raise ValueError("compatibility_mode must be 'functional' or 'strict'")

    eligible = _apply_hard_constraints(
        eligible,
        hard_constraints,
        lower=V2_LOWER_BOUND_CONSTRAINTS,
        upper=V2_UPPER_BOUND_CONSTRAINTS,
    ).sort_values("service_id", kind="stable").reset_index(drop=True)
    computed_hash = eligible_pool_snapshot_hash(eligible["service_id"])
    if compatibility_mode == "strict" and requested_hash != computed_hash:
        raise PoolSnapshotMismatch(requested_hash or "", computed_hash)

    return CompositionInstance(
        eligible_services=eligible,
        composition_size=composition_size,
        functional_requirements=functional_requirements,
        hard_constraints=hard_constraints,
        compatibility_mode=compatibility_mode,
        compatibility_requirements=compatibility_requirements,
        requested_pool_snapshot_hash=requested_hash,
        computed_pool_snapshot_hash=computed_hash,
        objective_version=objective_version_for_schema(schema_version),
    )


def eligible_pool_snapshot_hash(service_ids) -> str:
    """Hash sorted unique pool membership using the v2 frozen invariant."""
    canonical = "\n".join(sorted({str(value) for value in service_ids}))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _retain_completed_successful_observations(services: pd.DataFrame) -> pd.DataFrame:
    completed = services["service_status"].astype(str).str.lower().eq("completed")
    fedavg = services["fl_run_id"].notna()
    successful_fedavg = (
        services["participated"].map(lambda value: value == 1)
        & services["fedavg_succeeded"].map(lambda value: value == 1)
    )
    shared = services["cohort_id"].notna()
    return services[completed & (~fedavg | successful_fedavg) & (~shared | services["cohort_succeeded"].eq(1))].copy()


def _deduplicate_latest_fedavg_observations(services: pd.DataFrame) -> pd.DataFrame:
    fedavg = services[services["fl_run_id"].notna()].copy()
    independent = services[services["fl_run_id"].isna()].copy()
    fedavg = fedavg[
        fedavg["client_id"].notna()
        & fedavg["round_id"].notna()
    ]
    if not fedavg.empty:
        fedavg = fedavg.sort_values(
            ["fl_run_id", "client_id", "round_id", "service_id"],
            ascending=[True, True, False, True],
            kind="stable",
        ).drop_duplicates(["fl_run_id", "client_id"], keep="first")
    return pd.concat([independent, fedavg], ignore_index=True)


def _filter_strict_snapshot(
    services: pd.DataFrame, requirements: Mapping[str, Any]
) -> pd.DataFrame:
    for field in ("task_semantics_hash", "evaluation_protocol_hash"):
        if field in requirements:
            services = services[services[field].astype(str).eq(str(requirements[field]))]
    if "family_id" in requirements:
        mask = services["generation_mode"].eq("independent") & services["composition_compatible"].eq(1)
        mask &= services["family_contract_version"].isin(("independent_family_v1", "independent_family_v2"))
        for field in ("family_id", "initial_state_hash", "compatibility_contract_hash"):
            mask &= services[field].astype(str).eq(str(requirements[field]))
        return services[mask].copy()
    if "cohort_id" in requirements:
        mask = services["generation_mode"].eq("shared_parent")
        for field in ("cohort_id", "parent_state_hash", "compatibility_contract_hash"):
            mask &= services[field].astype(str).eq(str(requirements[field]))
        return services[mask].copy()
    parent_round = int(requirements["parent_global_round"])
    mask = (
        services["compatibility_contract_hash"].astype(str).eq(str(requirements["compatibility_contract_hash"]))
        & services["fl_run_id"].astype(str).eq(str(requirements["fl_run_id"]))
        & services["parent_global_round"].map(lambda value: value == parent_round)
        & services["parent_state_hash"].astype(str).eq(str(requirements["parent_state_hash"]))
    )
    return services[mask].copy()


def _deduplicate_strict_snapshot(services: pd.DataFrame) -> pd.DataFrame:
    if services["family_contract_version"].isin(("independent_family_v1", "independent_family_v2")).all():
        if services.service_id.duplicated().any():
            raise ProvenanceIntegrityError("duplicate service in frozen family")
        return services.sort_values("service_id", kind="stable")
    if services["generation_mode"].eq("shared_parent").all():
        if services.duplicated(["cohort_id", "client_id"]).any():
            raise ProvenanceIntegrityError("duplicate client identity in frozen cohort")
        return services.sort_values(["cohort_id", "client_id", "service_id"], kind="stable")
    return services.sort_values(
        ["fl_run_id", "client_id", "round_id", "service_id"],
        ascending=[True, True, True, True],
        kind="stable",
    ).drop_duplicates(["fl_run_id", "client_id", "round_id"], keep="first")


def _validate_strict_provenance(services: pd.DataFrame) -> None:
    if services["client_id"].isna().any():
        raise ProvenanceIntegrityError("Strict observations require a FedAvg client_id")
    legacy = services[services["generation_mode"].ne("shared_parent") & services["family_contract_version"].isna()]
    invalid_rounds = legacy[
        legacy.apply(
            lambda row: row["round_id"] is None
            or row["parent_global_round"] is None
            or int(row["round_id"]) != int(row["parent_global_round"]) + 1,
            axis=1,
        )
    ]
    if not invalid_rounds.empty:
        raise ProvenanceIntegrityError(
            "Strict observations must satisfy round_id = parent_global_round + 1"
        )
    for field in STRICT_PROVENANCE_FIELDS:
        values = services[field].dropna().astype(str)
        if len(values) != len(services):
            raise ProvenanceIntegrityError(
                f"Strict observations are missing required provenance field '{field}'"
            )
        if values.nunique() != 1:
            raise ProvenanceIntegrityError(
                f"Strict observations disagree on provenance field '{field}'"
            )


def _apply_hard_constraints(
    services: pd.DataFrame,
    constraints: Mapping[str, Any],
    *,
    lower: Mapping[str, str],
    upper: Mapping[str, str],
) -> pd.DataFrame:
    eligible = services
    for constraint, field in lower.items():
        if constraint in constraints:
            eligible = eligible[eligible[field].astype(float) >= number(constraints[constraint])]
    for constraint, field in upper.items():
        if constraint in constraints:
            eligible = eligible[eligible[field].astype(float) <= number(constraints[constraint])]
    return eligible


def request_object(request: Mapping[str, Any], field: str) -> dict[str, Any]:
    """Decode a JSON-object request field, accepting in-memory mappings too."""
    value = request.get(field)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field} must be a JSON object") from error
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError(f"{field} must be a JSON object")


def _composition_size(request: Mapping[str, Any]) -> int:
    try:
        value = int(request["composition_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("composition_size must be a positive integer") from error
    if value < 1:
        raise ValueError("composition_size must be a positive integer")
    return value


def _validate_functional_requirements(requirements: Mapping[str, Any]) -> None:
    if "task_family" not in requirements:
        raise ValueError("functional_requirements must include task_family")
    unsupported = set(requirements) - set(FUNCTIONAL_FIELDS)
    if unsupported:
        raise ValueError(f"Unsupported functional requirements: {sorted(unsupported)}")


def _validate_hard_constraints(constraints: Mapping[str, Any], *, schema_version: str) -> None:
    if schema_version != LEGACY_REQUEST_SCHEMA_VERSION:
        supported = set(V2_LOWER_BOUND_CONSTRAINTS) | set(V2_UPPER_BOUND_CONSTRAINTS)
    else:
        supported = set(LOWER_BOUND_CONSTRAINTS) | set(UPPER_BOUND_CONSTRAINTS)
    unsupported = set(constraints) - supported
    if unsupported:
        raise ValueError(f"Unsupported hard constraints: {sorted(unsupported)}")
    for name, value in constraints.items():
        if not finite(value):
            raise ValueError(f"Hard constraint '{name}' must be finite")
