"""Fail-closed eligibility for requested production evaluation dimensions."""

from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd

from .catalogue import finite, number


PRODUCTION_EVALUATION_VERSION = "production_evaluation_v1"
DEFAULT_EVALUATION_VERSION = "benchmark_evaluation_v1"
SUPPORTED_EVALUATION_VERSIONS = frozenset({PRODUCTION_EVALUATION_VERSION, DEFAULT_EVALUATION_VERSION})
EVALUATION_DIMENSIONS = ("trust", "explainability")


def _object(request: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = request.get(name)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} must be a JSON object") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def requested_evaluation_dimensions(request: Mapping[str, Any]) -> tuple[str, ...]:
    requirements = _object(request, "evaluation_requirements")
    allowed = {*EVALUATION_DIMENSIONS, "protocol_version", "allow_legacy"}
    if unknown := set(requirements) - allowed:
        raise ValueError(f"Unsupported evaluation requirements: {sorted(unknown)}")
    for flag in (*EVALUATION_DIMENSIONS, "allow_legacy"):
        if flag in requirements and not isinstance(requirements[flag], bool):
            raise ValueError(f"evaluation_requirements.{flag} must be a boolean")
    if requirements.get("protocol_version", DEFAULT_EVALUATION_VERSION) not in SUPPORTED_EVALUATION_VERSIONS:
        raise ValueError("Unsupported production evaluation protocol version")
    weights = _object(request, "preference_weights")
    qos = _object(request, "qos_requirements")
    hard = _object(request, "hard_constraints")
    # An explicitly supplied all-zero/empty weight vector invokes the equal
    # weighting fallback in calculate_utility and therefore requests both.
    equal_weights = "preference_weights" in request and not any(number(v, 0.0) > 0 for v in weights.values())
    return tuple(dimension for dimension in EVALUATION_DIMENSIONS if (
        requirements.get(dimension, False)
        or number(weights.get(dimension), 0.0) > 0
        or equal_weights
        or f"min_{dimension}" in qos
        or f"min_{dimension}_score" in hard
    ))


def production_evidence_mask(services: pd.DataFrame, dimension: str, protocol_version=DEFAULT_EVALUATION_VERSION) -> pd.Series:
    return (
        services["evaluation_protocol_version"].eq(protocol_version)
        & services[f"{dimension}_status"].eq("complete")
        & services[f"{dimension}_evidence_sufficient"].eq(True)
        & services[f"{dimension}_score"].map(finite)
    )


def filter_evaluation_evidence(services: pd.DataFrame, request: Mapping[str, Any]) -> pd.DataFrame:
    """Require complete evidence for every requested trust/explanation metric.

    Historical experiment replay may explicitly allow unversioned scores. It
    still requires a measured score for every selected member, and never grants
    eligibility to a failed/incomplete current-protocol evaluation.
    """
    requirements = _object(request, "evaluation_requirements")
    eligible = services
    for dimension in requested_evaluation_dimensions(request):
        mask = production_evidence_mask(eligible, dimension, requirements.get("protocol_version", DEFAULT_EVALUATION_VERSION))
        if requirements.get("allow_legacy") is True:
            mask |= eligible["evaluation_protocol_version"].isin({"legacy_unversioned", "legacy_perturbation_v1"}) & eligible[f"{dimension}_score"].map(finite)
        eligible = eligible[mask]
    return eligible
