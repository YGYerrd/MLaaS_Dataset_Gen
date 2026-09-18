"""Deterministic disjoint client sampling from a reusable source pool."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from .image_store import DiskArrayStore

from mlaas_data_generator.fedavg_manifest import PARTITION_GENERATOR_VERSION
from ..clients.legacy_profiles import UNIFORM_PROFILE as LEGACY_UNIFORM_PROFILE
from ..clients.legacy_profiles import ALIASES as LEGACY_PROFILE_ALIASES

from .skew_axes import ResolvedSkewAxis, resolve_skew_axis
from .splitters import split_data


@dataclass(frozen=True)
class FedAvgPartitionSpec:
    client_id: str
    client_index: int
    strategy: str
    distribution_param: Any
    sample_size: int
    sample_seed: int
    skew_axis: str | None = None
    skew_axis_config: Mapping[str, Any] | None = None
    generator_version: str = PARTITION_GENERATOR_VERSION
    heterogeneity_intensity: float = 0.5
    requested_partition_params: Mapping[str, Any] | None = None
    requested_quota: int | None = None


@dataclass(frozen=True)
class FedAvgPartition:
    client_id: str
    indices: np.ndarray
    source_ids: tuple[Any, ...]
    x: Any
    y: Any
    metadata: dict[str, Any]


def _take_rows(value, indices: np.ndarray):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _take_rows(item, indices) for key, item in value.items()}
    if isinstance(value, DiskArrayStore):
        return value.take(indices)
    if isinstance(value, np.ndarray):
        return value[indices]
    values = indices.tolist()
    if isinstance(value, (list, tuple)):
        return [value[index] for index in values]
    return np.asarray(value, dtype=object)[indices]


def _strategy(value: Any) -> str:
    strategy = str(value or "iid").strip().lower().replace("-", "_")
    aliases = {
        **LEGACY_PROFILE_ALIASES,
        "niid": "dirichlet",
        "non_iid": "dirichlet",
        "non_iid_dirichlet": "dirichlet",
        "shards": "shard",
        "label_per_client": "shard",
    }
    return aliases.get(strategy, strategy)


def _stable_hash(*parts: Any) -> int:
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def _largest_remainder_counts(
    weights: Sequence[float], total: int, *, tie_keys: Sequence[Any]
) -> np.ndarray:
    total = int(total)
    values = np.asarray(weights, dtype=np.float64)
    if total < 0 or values.ndim != 1 or len(values) != len(tie_keys):
        raise ValueError("Invalid largest-remainder inputs")
    if len(values) == 0:
        if total:
            raise ValueError("Cannot allocate a positive total across no buckets")
        return np.zeros(0, dtype=int)
    values = np.maximum(values, 0.0)
    weight_sum = float(values.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        values = np.ones(len(values), dtype=np.float64)
        weight_sum = float(len(values))
    raw = values * (float(total) / weight_sum)
    counts = np.floor(raw).astype(int)
    remaining = total - int(counts.sum())
    order = sorted(
        range(len(values)),
        key=lambda index: (
            -float(raw[index] - counts[index]),
            _stable_hash("largest-remainder", tie_keys[index]),
        ),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def scale_requested_quotas(
    requested_quotas: Sequence[int],
    available_total: int,
    *,
    client_ids: Sequence[str] | None = None,
    quantity_clients: Sequence[bool] | None = None,
    heterogeneity_intensity: float = 0.5,
) -> np.ndarray:
    """Scale quotas with constrained largest-remainder integer rounding."""
    requested = np.asarray(requested_quotas, dtype=int)
    total = int(available_total)
    if requested.ndim != 1 or np.any(requested <= 0):
        raise ValueError("FedAvg requested quotas must all be positive")
    if total < len(requested):
        raise ValueError(
            "FedAvg training pool cannot provide at least one sample per client"
        )
    if int(requested.sum()) <= total:
        return requested.copy()

    ids = list(client_ids or [str(index) for index in range(len(requested))])
    if len(ids) != len(requested):
        raise ValueError("client_ids must align with requested quotas")
    raw = requested.astype(np.float64) * (float(total) / float(requested.sum()))
    effective = np.maximum(1, np.floor(raw).astype(int))

    while int(effective.sum()) > total:
        candidates = [index for index, value in enumerate(effective) if value > 1]
        if not candidates:
            raise ValueError("FedAvg quota scaling could not retain one sample per client")
        chosen = min(
            candidates,
            key=lambda index: (
                float(raw[index] - effective[index]),
                _stable_hash("quota-remove", ids[index]),
            ),
        )
        effective[chosen] -= 1
    while int(effective.sum()) < total:
        chosen = max(
            range(len(effective)),
            key=lambda index: (
                float(raw[index] - effective[index]),
                -_stable_hash("quota-add", ids[index]),
            ),
        )
        effective[chosen] += 1

    quantity = np.asarray(
        quantity_clients if quantity_clients is not None else [False] * len(requested),
        dtype=bool,
    )
    if len(quantity) != len(requested):
        raise ValueError("quantity_clients must align with requested quotas")
    quantity_indices = np.flatnonzero(quantity).tolist()
    bound = 8.0 ** float(heterogeneity_intensity)
    for _ in range(max(1, total * 2)):
        if len(quantity_indices) < 2:
            break
        low = min(quantity_indices, key=lambda index: (effective[index], ids[index]))
        high = max(quantity_indices, key=lambda index: (effective[index], ids[index]))
        if effective[high] / effective[low] <= bound + 1e-12:
            break
        old_gap = effective[high] / effective[low]
        moved = False
        if effective[high] > effective[low] + 1 and effective[low] < requested[low]:
            trial_gap = (effective[high] - 1) / (effective[low] + 1)
            if trial_gap < old_gap:
                effective[high] -= 1
                effective[low] += 1
                moved = True
        if moved:
            continue
        non_quantity = np.flatnonzero(~quantity).tolist()
        recipients = [
            index for index in non_quantity if effective[index] < requested[index]
        ]
        if effective[high] > 1 and recipients:
            recipient = max(
                recipients,
                key=lambda index: (
                    float(raw[index] - effective[index]),
                    -_stable_hash(ids[index]),
                ),
            )
            effective[high] -= 1
            effective[recipient] += 1
            continue
        donors = [
            index
            for index in non_quantity
            if effective[index] > 1 and effective[low] < requested[low]
        ]
        if donors:
            donor = min(
                donors,
                key=lambda index: (
                    float(raw[index] - effective[index]),
                    _stable_hash(ids[index]),
                ),
            )
            effective[donor] -= 1
            effective[low] += 1
            continue
        raise ValueError("Integer quota scaling cannot satisfy the historical quota ratio")
    if len(quantity_indices) >= 2:
        values = effective[quantity_indices]
        if int(values.max()) / int(values.min()) > bound + 1e-12:
            raise ValueError("Integer quota scaling violates the historical quota ratio")
    return effective


def _single_bucket_axis(
    y: Any, *, task_family: str | None, requested_axis: str | None, reason: str | None
) -> ResolvedSkewAxis:
    label = "__all__"
    if str(task_family or "").lower() == "classification":
        try:
            values = np.asarray(y, dtype=object).reshape(-1)
            unique = np.unique(values)
            if len(unique) == 1:
                label = str(unique[0])
        except Exception:
            pass
    count = len(y) if y is not None else 0
    return ResolvedSkewAxis(
        requested_axis=requested_axis,
        effective_axis=requested_axis or "single_global_bucket",
        axis_family="categorical",
        bucket_ids=np.zeros(count, dtype=np.int64),
        bucket_labels={"0": label},
        bucket_spec={
            "kind": "single_global_bucket",
            "ordered": False,
            "cardinality": 1,
        },
        source_fields=["y"] if y is not None else [],
        fallback_reason=reason,
    )


def _target_counts(
    spec: FedAvgPartitionSpec,
    *,
    quota: int,
    global_proportions: np.ndarray,
    bucket_labels: Sequence[str],
    run_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    strategy = _strategy(spec.strategy)
    params = dict(spec.requested_partition_params or {})
    h = float(spec.heterogeneity_intensity)
    rng = np.random.default_rng(int(run_seed) ^ int(spec.sample_seed))
    weights = global_proportions.copy()
    resolved: dict[str, Any] = {"bucket_count": len(weights)}

    if strategy == "dirichlet" and len(weights) > 1:
        alpha = float(params.get("alpha", 10.0 ** (1.0 - 2.0 * h)))
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("Dirichlet alpha must be positive")
        concentration = alpha * len(weights) * weights
        weights = rng.dirichlet(concentration)
        resolved.update(alpha=alpha, concentration=concentration.tolist())
    elif strategy == "shard" and len(weights) > 1:
        selected_count = int(
            params.get(
                "selected_bucket_count",
                max(1, round(len(weights) ** (1.0 - h))),
            )
        )
        selected_count = max(1, min(selected_count, len(weights)))
        selected = np.sort(rng.choice(len(weights), size=selected_count, replace=False))
        mask = np.zeros(len(weights), dtype=np.float64)
        mask[selected] = weights[selected]
        weights = mask / float(mask.sum())
        resolved.update(
            selected_bucket_count=selected_count,
            selected_bucket_labels=[bucket_labels[index] for index in selected],
        )
    elif strategy not in {"iid", LEGACY_UNIFORM_PROFILE, "dirichlet", "shard"}:
        raise ValueError(f"Unsupported FedAvg partition strategy: {strategy}")

    counts = _largest_remainder_counts(
        weights,
        quota,
        tie_keys=[(spec.client_id, label) for label in bucket_labels],
    )
    return counts, resolved


def _sample_target_counts(
    targets: np.ndarray,
    quotas: np.ndarray,
    capacities: np.ndarray,
    specs: Sequence[FedAvgPartitionSpec],
    bucket_labels: Sequence[str],
    global_proportions: np.ndarray,
    resolved_params: list[dict[str, Any]],
    *,
    run_seed: int,
) -> np.ndarray:
    """Keep feasible targets intact; redraw only clients that cannot fit.

    Reserve uniform targets first, then restricted shard targets,
    then Dirichlet targets. The order is stable under input permutations.
    No allocation is enlarged to consume unused source rows.
    """
    accepted = targets.copy()
    if np.all(accepted.sum(axis=0) <= capacities):
        return accepted
    remaining = capacities.copy()
    priority = {"iid": 0, LEGACY_UNIFORM_PROFILE: 0, "shard": 1, "dirichlet": 2}
    order = sorted(
        range(len(specs)),
        key=lambda i: (priority[_strategy(specs[i].strategy)], specs[i].client_id),
    )
    for client in order:
        spec = specs[client]
        strategy = _strategy(spec.strategy)
        target = accepted[client]
        redraws = 0
        while np.any(target > remaining) and strategy in {"shard", "dirichlet"} and redraws < 128:
            redraws += 1
            target, resolution = _target_counts(
                spec,
                quota=int(quotas[client]),
                global_proportions=global_proportions,
                bucket_labels=bucket_labels,
                run_seed=_stable_hash("capacity-redraw", run_seed, spec.client_id, redraws),
            )
            resolved_params[client] = resolution
        if np.any(target > remaining):
            raise ValueError(
                f"FedAvg source pool cannot satisfy client {spec.client_id!r} "
                f"({strategy}, quota={int(quotas[client])}) after {redraws} redraws; "
                "increase the source pool cap or reduce client sample sizes"
            )
        accepted[client] = target
        remaining -= target
        resolved_params[client]["capacity_redraw_count"] = redraws
    return accepted


def plan_fedavg_partitions(
    x,
    y,
    specs: Sequence[FedAvgPartitionSpec],
    *,
    run_seed: int,
    source_ids: Sequence[Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    task_family: str | None = None,
    hf_task: str | None = None,
) -> dict[str, FedAvgPartition]:
    """Sample bounded, disjoint client datasets and leave spare rows unused."""
    total = len(next(iter(x.values()))) if isinstance(x, dict) and x else len(x)
    ordered_specs = sorted(specs, key=lambda item: str(item.client_id))
    if not ordered_specs:
        raise ValueError("FedAvg partition specs are empty")
    if len({item.client_id for item in ordered_specs}) != len(ordered_specs):
        raise ValueError("FedAvg partition specs contain duplicate client IDs")
    if any(item.generator_version != PARTITION_GENERATOR_VERSION for item in ordered_specs):
        raise ValueError(f"FedAvg partitions require {PARTITION_GENERATOR_VERSION}")
    intensities = {float(item.heterogeneity_intensity) for item in ordered_specs}
    if len(intensities) != 1 or not all(0.0 <= value <= 1.0 for value in intensities):
        raise ValueError("FedAvg partition specs must share one intensity in [0, 1]")
    intensity = next(iter(intensities))
    requested = np.asarray(
        [int(item.requested_quota or item.sample_size) for item in ordered_specs],
        dtype=int,
    )
    if np.any(requested <= 0):
        raise ValueError("FedAvg client requested quotas must be positive")
    ids = list(range(total)) if source_ids is None else list(source_ids)
    if len(ids) != total:
        raise ValueError("FedAvg source IDs must align one-to-one with the training data")

    axis_requests = {
        json.dumps(
            [spec.skew_axis, dict(spec.skew_axis_config or {})],
            sort_keys=True,
            default=str,
        )
        for spec in ordered_specs
    }
    if len(axis_requests) != 1:
        raise ValueError("FedAvg clients must share one run-level skew axis")
    axis_spec = ordered_specs[0]
    axis = resolve_skew_axis(
        x,
        y,
        meta or {},
        split_name="train",
        task_family=task_family,
        hf_task=hf_task,
        requested_axis=axis_spec.skew_axis,
        axis_config=axis_spec.skew_axis_config,
    )
    if axis.bucket_ids is None:
        axis = _single_bucket_axis(
            y,
            task_family=task_family,
            requested_axis=axis.requested_axis,
            reason=axis.fallback_reason,
        )
    bucket_ids = np.asarray(axis.bucket_ids, dtype=np.int64).reshape(-1)
    if len(bucket_ids) != total:
        raise ValueError("Resolved FedAvg bucket IDs do not align with training data")
    internal_buckets = sorted(np.unique(bucket_ids).tolist())
    if not internal_buckets:
        raise ValueError("FedAvg training universe contains no partition buckets")
    bucket_labels = [
        str(axis.bucket_labels.get(str(value), str(value))) for value in internal_buckets
    ]
    capacities = np.asarray(
        [int(np.sum(bucket_ids == value)) for value in internal_buckets], dtype=int
    )
    global_proportions = capacities.astype(np.float64) / float(total)

    effective = scale_requested_quotas(
        requested,
        total,
        client_ids=[spec.client_id for spec in ordered_specs],
        quantity_clients=[
            _strategy(spec.strategy) == LEGACY_UNIFORM_PROFILE
            and not (spec.requested_partition_params or {}).get("quantity_sampler_version")
            for spec in ordered_specs
        ],
        heterogeneity_intensity=intensity,
    )
    targets = np.zeros((len(ordered_specs), len(internal_buckets)), dtype=int)
    resolved_params: list[dict[str, Any]] = []
    for client, spec in enumerate(ordered_specs):
        targets[client], resolution = _target_counts(
            spec,
            quota=int(effective[client]),
            global_proportions=global_proportions,
            bucket_labels=bucket_labels,
            run_seed=int(run_seed),
        )
        resolved_params.append(resolution)

    initial_targets = targets.copy()
    targets = _sample_target_counts(
        targets, effective, capacities, ordered_specs, bucket_labels,
        global_proportions, resolved_params, run_seed=int(run_seed),
    )
    realised = targets.copy()
    rng = np.random.default_rng(int(run_seed))
    shuffled_pools: dict[int, np.ndarray] = {}
    for bucket in internal_buckets:
        pool = np.flatnonzero(bucket_ids == bucket)
        shuffled_pools[int(bucket)] = np.asarray(rng.permutation(pool), dtype=int)
    offsets = {int(bucket): 0 for bucket in internal_buckets}
    chosen_by_client: list[np.ndarray] = []
    for client in range(len(ordered_specs)):
        chosen: list[int] = []
        for bucket_index, bucket in enumerate(internal_buckets):
            count = int(realised[client, bucket_index])
            start = offsets[int(bucket)]
            stop = start + count
            pool = shuffled_pools[int(bucket)]
            if stop > len(pool):
                raise ValueError("FedAvg sampled allocation exceeds a shuffled bucket pool")
            chosen.extend(pool[start:stop].tolist())
            offsets[int(bucket)] = stop
        chosen_by_client.append(np.asarray(chosen, dtype=int))

    result: dict[str, FedAvgPartition] = {}
    for client, spec in enumerate(ordered_specs):
        chosen = chosen_by_client[client]
        target = targets[client]
        actual = realised[client]
        quota = int(effective[client])
        target_map = {
            label: int(target[index]) for index, label in enumerate(bucket_labels)
        }
        actual_map = {
            label: int(actual[index]) for index, label in enumerate(bucket_labels)
        }
        target_proportions = {
            label: float(target[index]) / quota
            for index, label in enumerate(bucket_labels)
        }
        actual_proportions = {
            label: float(actual[index]) / quota
            for index, label in enumerate(bucket_labels)
        }
        l1 = int(np.abs(target - actual).sum())
        deviation_count = l1 / 2.0
        probabilities = actual.astype(np.float64) / float(quota)
        positive = probabilities[probabilities > 0.0]
        entropy = (
            0.0
            if len(internal_buckets) <= 1
            else float(
                -np.sum(positive * np.log(positive)) / np.log(len(internal_buckets))
            )
        )
        reasons = []
        if int(requested[client]) != quota:
            reasons.append("requested quotas proportionally scaled to available training pool")
        if axis.fallback_reason:
            reasons.append(str(axis.fallback_reason))
        if resolved_params[client].get("capacity_redraw_count", 0):
            reasons.append("client distribution redrawn to fit remaining source bucket capacity")
        resolution_reason = "; ".join(reasons) or None
        metadata = {
            "client_id": spec.client_id,
            "client_index": int(spec.client_index),
            "partition_generator_version": spec.generator_version,
            "heterogeneity_intensity": intensity,
            "requested_partition_strategy": _strategy(spec.strategy),
            "requested_partition_params": dict(spec.requested_partition_params or {}),
            "resolved_partition_params": resolved_params[client],
            "requested_quota": int(requested[client]),
            "effective_quota": quota,
            "quota_shortfall": int(requested[client] - quota),
            "initial_target_bucket_counts": {
                label: int(initial_targets[client, index])
                for index, label in enumerate(bucket_labels)
            },
            "capacity_redraw_count": resolved_params[client].get("capacity_redraw_count", 0),
            "source_pool_sample_count": total,
            "client_samples_used": int(effective.sum()),
            "unused_source_samples": total - int(effective.sum()),
            "target_bucket_counts": target_map,
            "target_bucket_proportions": target_proportions,
            "realised_bucket_counts": actual_map,
            "realised_bucket_proportions": actual_proportions,
            "majority_bucket_share": float(actual.max()) / quota,
            "normalised_bucket_entropy": entropy,
            "represented_bucket_count": int(np.count_nonzero(actual)),
            "l1_allocation_deviation": l1,
            "total_variation_distance": float(deviation_count / quota),
            "target_deviation_count": deviation_count,
            "resolution_reason": resolution_reason,
            "fallback_reason": resolution_reason,
            "sample_seed": int(spec.sample_seed),
            "sample_count": len(chosen),
            "indices": chosen.tolist(),
            "source_ids": [ids[index] for index in chosen],
            "requested_skew_axis": axis.requested_axis,
            "skew_axis": axis.effective_axis,
            "partition_bucket_unit": axis.effective_axis,
            "axis_family": axis.axis_family,
            "bucket_spec": axis.bucket_spec,
            "source_fields": axis.source_fields,
            "requested_strategy": _strategy(spec.strategy),
            "effective_strategy": _strategy(spec.strategy),
            "bucket_histogram": actual_map,
        }
        result[spec.client_id] = FedAvgPartition(
            client_id=spec.client_id,
            indices=chosen,
            source_ids=tuple(ids[index] for index in chosen),
            x=_take_rows(x, chosen),
            y=_take_rows(y, chosen),
            metadata=metadata,
        )
    all_chosen = np.concatenate(chosen_by_client)
    if len(np.unique(all_chosen)) != len(all_chosen):
        raise ValueError("FedAvg sampler assigned a source row twice")
    return result


def partition_client_data(
    x,
    y,
    *,
    num_clients: int,
    client_index: int,
    strategy: str,
    distribution_param: Any = None,
    sample_size: int | None = None,
    rng=None,
    meta: Mapping[str, Any] | None = None,
    task_family: str | None = None,
    hf_task: str | None = None,
    skew_axis: str | None = None,
    skew_axis_config: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Retained independent splitter primitive; not used by the FedAvg run sampler."""
    clients, resolution = split_data(
        x,
        y,
        num_clients,
        strategy=strategy,
        distribution_param=distribution_param,
        sample_size=sample_size,
        rng=rng,
        meta=meta,
        task_family=task_family,
        hf_task=hf_task,
        skew_axis=skew_axis,
        skew_axis_config=skew_axis_config,
    )
    client_key = f"client_{int(client_index) + 1}"
    if client_key not in clients:
        keys = sorted(clients)
        if not keys:
            raise ValueError("split_data produced no client partitions")
        client_key = keys[min(int(client_index), len(keys) - 1)]
    payload = clients[client_key]
    return payload["x"], payload["y"], resolution, {
        "client_key": client_key,
        "client_index": int(client_index),
    }


def partition_indices_for_client(
    total: int,
    *,
    num_clients: int,
    client_index: int,
    strategy: str = "iid",
    distribution_param: Any = None,
    seed: int,
) -> np.ndarray:
    """Retained independent splitter primitive; not used by the FedAvg run sampler."""
    rng = np.random.default_rng(int(seed))
    x = np.arange(total, dtype=int)
    y = np.zeros(total, dtype=int)
    clients, _ = split_data(
        x,
        y,
        num_clients,
        strategy=strategy,
        distribution_param=distribution_param,
        rng=rng,
    )
    client_key = f"client_{int(client_index) + 1}"
    if client_key not in clients:
        keys = sorted(clients)
        client_key = keys[min(int(client_index), len(keys) - 1)]
    return np.asarray(clients[client_key]["x"], dtype=int)
