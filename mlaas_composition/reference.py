"""Generate exact references by exhaustively evaluating tractable instances."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd

from .pool import CompositionInstance, build_composition_instance, request_object
from .protocol import OBJECTIVE_VERSION, objective_version_for_request
from .sampler import calculate_utility, evaluate_provisional_composition


DEFAULT_MAX_CANDIDATE_SPACE = 10_000


class ReferenceSearchSpaceTooLarge(ValueError):
    """Raised when an exhaustive reference would exceed its configured limit."""


@dataclass(frozen=True)
class ReferenceSolution:
    """An exact optimum over every candidate in a tractable ``Omega_R``."""

    service_ids: tuple[str, ...]
    utility_score: float
    scores: Mapping[str, Any]
    candidate_space_size: int
    evaluations: int
    evaluator_name: str


def find_exhaustive_reference(
    instance: CompositionInstance,
    preference_weights: Mapping[str, Any],
    *,
    max_candidate_space: int = DEFAULT_MAX_CANDIDATE_SPACE,
    evaluator: Callable[[pd.DataFrame], dict[str, Any]] | None = None,
    qos_requirements: Mapping[str, Any] | None = None,
    constraint_penalties: Mapping[str, Any] | None = None,
    evaluator_name: str = "paper_informed_collaborative_heuristic",
    objective_version: str | None = None,
) -> ReferenceSolution | None:
    """Return the verified optimum by evaluating every candidate in ``Omega_R``.

    The candidate iterator is lazy, but exhaustive evaluation is intentionally
    bounded. A zero-sized search space returns ``None``; an over-limit instance
    raises rather than being silently treated as an exact reference.
    """
    if max_candidate_space < 1:
        raise ValueError("max_candidate_space must be at least one")
    resolved_objective_version = objective_version or instance.objective_version or OBJECTIVE_VERSION
    resolved_evaluator = evaluator or (
        lambda selected: evaluate_provisional_composition(
            selected, objective_version=resolved_objective_version
        )
    )
    if instance.omega_size == 0:
        return None
    if instance.omega_size > max_candidate_space:
        raise ReferenceSearchSpaceTooLarge(
            f"Omega_R contains {instance.omega_size} candidates, exceeding the limit of {max_candidate_space}"
        )

    best: ReferenceSolution | None = None
    for indices in instance.iter_candidate_indices():
        selected = instance.C_R.iloc[list(indices)].reset_index(drop=True)
        service_ids = tuple(sorted(selected["service_id"].astype(str).tolist()))
        scores = resolved_evaluator(selected)
        utility = calculate_utility(
            scores,
            preference_weights,
            qos_requirements,
            constraint_penalties,
            objective_version=resolved_objective_version,
        )
        candidate = ReferenceSolution(
            service_ids=service_ids,
            utility_score=utility,
            scores=scores,
            candidate_space_size=instance.omega_size,
            evaluations=1,
            evaluator_name=evaluator_name,
        )
        if best is None or utility > best.utility_score or (
            utility == best.utility_score and service_ids < best.service_ids
        ):
            best = candidate

    assert best is not None
    return ReferenceSolution(
        service_ids=best.service_ids,
        utility_score=best.utility_score,
        scores=best.scores,
        candidate_space_size=best.candidate_space_size,
        evaluations=instance.omega_size,
        evaluator_name=best.evaluator_name,
    )


def generate_exhaustive_references(
    services: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    max_candidate_space: int = DEFAULT_MAX_CANDIDATE_SPACE,
) -> pd.DataFrame:
    """Create one exact-reference record per tractable synthetic request.

    Infeasible and over-limit requests remain in the output with a status so
    benchmark consumers never mistake missing references for exact optima.
    """
    records: list[dict[str, Any]] = []
    for _, request_row in requests.iterrows():
        request = request_row.to_dict()
        instance = build_composition_instance(services, request)
        objective_version = objective_version_for_request(request)
        qos_requirements = _optional_request_object(request, "qos_requirements")
        constraint_penalties = _optional_request_object(request, "constraint_penalties")
        record: dict[str, Any] = {
            "request_id": str(request.get("request_id")),
            "composition_type": "collaborative",
            "compatibility_mode": instance.compatibility_mode,
            "requested_pool_snapshot_hash": instance.requested_pool_snapshot_hash,
            "computed_pool_snapshot_hash": instance.computed_pool_snapshot_hash,
            "objective_version": objective_version,
            "eligible_pool_size": len(instance.C_R),
            "composition_size": instance.composition_size,
            "candidate_space_size": instance.omega_size,
            "reference_status": "",
            "evaluator_name": "paper_informed_collaborative_heuristic",
        }
        if instance.omega_size == 0:
            record["reference_status"] = "infeasible"
        elif instance.omega_size > max_candidate_space:
            record["reference_status"] = "too_large"
        else:
            solution = find_exhaustive_reference(
                instance,
                request_object(request, "preference_weights"),
                max_candidate_space=max_candidate_space,
                evaluator=lambda selected, qos=qos_requirements: evaluate_provisional_composition(
                    selected,
                    qos_requirements=qos,
                    objective_version=objective_version,
                ),
                qos_requirements=qos_requirements,
                constraint_penalties=constraint_penalties,
                objective_version=objective_version,
            )
            assert solution is not None
            record.update({
                "reference_status": "exact",
                "service_ids": json.dumps(solution.service_ids),
                "utility_score": solution.utility_score,
                "evaluations": solution.evaluations,
                **solution.scores,
            })
        records.append(record)
    return pd.DataFrame(records)


def _optional_request_object(request: Mapping[str, Any], field: str) -> dict[str, Any]:
    return request_object(request, field) if request.get(field) is not None else {}
