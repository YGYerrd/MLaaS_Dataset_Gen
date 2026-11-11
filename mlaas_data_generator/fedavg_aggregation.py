"""Canonical sample-weighted aggregation for the active FedAvg generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .checkpoint_store import TensorEntry, TensorState


FEDAVG_AGGREGATION_VERSION = "sample_weighted_parameter_delta_v1"
_FLOAT_DTYPES = {"float16", "float32", "float64", "bfloat16"}
_COMPLEX_DTYPES = {"complex64", "complex128"}


class FedAvgAggregationError(ValueError):
    """A round cannot be aggregated without weakening its state contract."""


@dataclass(frozen=True)
class FedAvgContributor:
    client_id: str
    service_id: str
    sample_count: int
    state: TensorState
    compatibility_contract_hash: str
    parent_state_hash: str


@dataclass(frozen=True)
class FedAvgAggregationResult:
    state: TensorState
    provenance: dict[str, Any]


def _entry_array(entry: TensorEntry) -> np.ndarray:
    if entry.dtype != "bfloat16":
        return entry.to_numpy()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch-backed models provide torch
        raise FedAvgAggregationError("bfloat16 aggregation requires PyTorch") from exc
    words = np.frombuffer(entry.data, dtype=np.dtype("<u2")).reshape(entry.shape).copy()
    return torch.from_numpy(words.view(np.int16)).view(torch.bfloat16).float().numpy()


def _entry_from_array(template: TensorEntry, value: np.ndarray) -> TensorEntry:
    if template.dtype == "bfloat16":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise FedAvgAggregationError("bfloat16 aggregation requires PyTorch") from exc
        tensor = torch.as_tensor(np.asarray(value, dtype=np.float32)).to(torch.bfloat16)
        words = tensor.view(torch.int16).cpu().numpy().astype("<i2", copy=False)
        data = words.tobytes(order="C")
    else:
        dtype = _entry_array(template).dtype
        data = np.asarray(value, dtype=dtype).tobytes(order="C")
    return TensorEntry(template.name, template.dtype, template.shape, data)


def aggregate_fedavg_states(
    parent: TensorState,
    contributors: Iterable[FedAvgContributor],
    *,
    compatibility_contract_hash: str,
    non_aggregatable_indices: Iterable[int] = (),
    minimum_contributors: int = 2,
) -> FedAvgAggregationResult:
    """Aggregate exact client states as weighted deltas from one parent state."""
    ordered = sorted(list(contributors), key=lambda item: item.client_id)
    if type(minimum_contributors) is not int or minimum_contributors not in (1, 2):
        raise FedAvgAggregationError("minimum_contributors must be 1 or 2")
    if len(ordered) < minimum_contributors:
        count = "two" if minimum_contributors == 2 else "one"
        raise FedAvgAggregationError(f"FedAvg requires at least {count} successful contributors")
    if len({item.client_id for item in ordered}) != len(ordered):
        raise FedAvgAggregationError("FedAvg contributors contain duplicate client IDs")
    if len({item.service_id for item in ordered}) != len(ordered):
        raise FedAvgAggregationError("FedAvg contributors contain duplicate service IDs")

    parent_descriptors = [entry.descriptor() for entry in parent.entries]
    for item in ordered:
        if int(item.sample_count) <= 0:
            raise FedAvgAggregationError("FedAvg contributor sample counts must be positive")
        if item.compatibility_contract_hash != compatibility_contract_hash:
            raise FedAvgAggregationError("FedAvg contributor compatibility contract mismatch")
        if item.parent_state_hash != parent.sha256:
            raise FedAvgAggregationError("FedAvg contributor parent-state hash mismatch")
        if [entry.descriptor() for entry in item.state.entries] != parent_descriptors:
            raise FedAvgAggregationError("FedAvg contributor tensor schema/order mismatch")

    total_samples = sum(int(item.sample_count) for item in ordered)
    excluded = {int(index) for index in non_aggregatable_indices}
    result_entries: list[TensorEntry] = []
    aggregated_indices: list[int] = []
    preserved_indices: list[int] = []
    for index, parent_entry in enumerate(parent.entries):
        if (
            index in excluded
            or parent_entry.dtype not in _FLOAT_DTYPES | _COMPLEX_DTYPES
        ):
            result_entries.append(parent_entry)
            preserved_indices.append(index)
            continue

        accumulator_dtype = np.complex128 if parent_entry.dtype in _COMPLEX_DTYPES else np.float64
        parent_array = np.asarray(_entry_array(parent_entry), dtype=accumulator_dtype)
        aggregate = np.array(parent_array, copy=True)
        for item in ordered:
            client_array = np.asarray(_entry_array(item.state.entries[index]), dtype=accumulator_dtype)
            weight = float(item.sample_count) / float(total_samples)
            aggregate += weight * (client_array - parent_array)
        if not np.all(np.isfinite(aggregate)):
            raise FedAvgAggregationError(f"FedAvg produced non-finite values for tensor {parent_entry.name!r}")
        result_entries.append(_entry_from_array(parent_entry, aggregate))
        aggregated_indices.append(index)

    state = TensorState(tuple(result_entries), framework=parent.framework)
    provenance = {
        "aggregation_method": "sample_weighted_fedavg",
        "aggregation_version": FEDAVG_AGGREGATION_VERSION,
        "weight_unit": "training_examples",
        "parent_state_hash": parent.sha256,
        "global_state_hash": state.sha256,
        "total_samples": total_samples,
        "contributors": [
            {
                "client_id": item.client_id,
                "service_id": item.service_id,
                "sample_count": int(item.sample_count),
                "normalized_weight": float(item.sample_count) / float(total_samples),
                "client_state_hash": item.state.sha256,
            }
            for item in ordered
        ],
        "aggregated_tensor_indices": aggregated_indices,
        "preserved_tensor_indices": preserved_indices,
        "accumulator_precision": {
            "floating": "float64",
            "complex": "complex128",
        },
    }
    return FedAvgAggregationResult(state=state, provenance=provenance)
