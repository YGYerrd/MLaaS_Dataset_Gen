"""Independent client sampling: shared source capacity, unique rows per client."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..provenance import sha256_json
from ..data.fedavg_partition import _largest_remainder_counts, _take_rows
from ..data.skew_axes import resolve_skew_axis

LEGACY_PARTITION_VERSION = "independent_source_pool_v1"
FRACTIONAL_PARTITION_VERSION = "independent_source_pool_v2"
PARTITION_VERSION = "independent_source_pool_v4"
SUPPORTED_PARTITION_VERSIONS = frozenset({LEGACY_PARTITION_VERSION, FRACTIONAL_PARTITION_VERSION, "independent_source_pool_v3", PARTITION_VERSION})


def _regression_settings(config):
    def optional(key, default):
        value = config.get(key)
        return default if value is None or (isinstance(value, float) and np.isnan(value)) else value
    return optional("regression_support_policy", "general"), float(optional("min_target_bin_fraction", 0.6))


def validate_partition_config(config: Mapping[str, Any]) -> None:
    version = config.get("partition_generator_version", PARTITION_VERSION)
    if version not in SUPPORTED_PARTITION_VERSIONS:
        raise ValueError("unsupported independent client partition version")
    quantile = config.get("shard_support_quantile")
    if version != LEGACY_PARTITION_VERSION and config.get("split_strategy") == "shard" and quantile is not None:
        if not 0 <= float(quantile) < 1:
            raise ValueError("shard support quantile must be in [0,1)")
    if version == PARTITION_VERSION:
        policy, fraction = _regression_settings(config)
        if policy not in {"general", "specialist"}:
            raise ValueError("regression support policy must be general or specialist")
        if not 0.6 <= fraction <= 1:
            raise ValueError("minimum target bin fraction must be in [0.6,1]")


def _capacity_counts(requested, capacities, weights, support, *, require_coverage=False):
    """Redistribute overflow only inside selected support, without replacement."""
    available = np.where(np.isin(np.arange(len(capacities)), support), capacities, 0)
    target = min(int(requested.sum()), int(available.sum()))
    counts = np.minimum(requested, available)
    if require_coverage:
        if target < len(support):
            raise ValueError("client quota cannot meet required bucket coverage")
        counts[support] = np.maximum(counts[support], 1)
        # Coverage may move examples out of a disproportionately large bucket.
        while counts.sum() > target:
            counts[int(np.argmax(counts))] -= 1
    while counts.sum() < target:
        remaining = available - counts
        eligible = remaining > 0
        refill_weights = np.where(eligible, weights, 0.0)
        if refill_weights.sum() == 0:
            refill_weights = remaining.astype(float)
        extra = _largest_remainder_counts(refill_weights / refill_weights.sum(),
            target - int(counts.sum()), tie_keys=list(range(len(counts))))
        counts += np.minimum(extra, remaining)
    return counts


def client_partition(x: Any, y: Any, config: Mapping[str, Any], *,
                     source_ids: Sequence[str], meta: Mapping[str, Any], task_family: str) -> dict:
    """Return indices and evidence, without copying the complete training pool."""
    validate_partition_config(config)
    if config.get("task_contract"):
        from ..tasks.contracts import TaskContract, validate_partition
        contract = TaskContract.from_dict(config["task_contract"])
        validate_partition(contract, config)
    version = PARTITION_VERSION
    total = len(source_ids)
    if total < 1 or len(set(source_ids)) != total:
        raise ValueError("client source IDs must be unique and non-empty")
    quantity = config.get("sample_size")
    quota = int(config.get("requested_quota", 600) if quantity is None else quantity)
    if quota < 1:
        raise ValueError("client quantity must be positive")
    strategy = str(config.get("split_strategy", "iid"))
    if strategy not in {"iid", "dirichlet", "shard", "quantity_skew"}:
        raise ValueError(f"unsupported client data profile: {strategy}")
    intensity = float(config.get("heterogeneity_intensity", 0.5))
    if not 0 <= intensity <= 1:
        raise ValueError("heterogeneity intensity must be in [0,1]")
    rng = np.random.default_rng(int(config["sample_seed"]))
    if strategy in {"iid", "quantity_skew"}:
        # Quantity is resolved in the client recipe. Neither profile imposes
        # label quotas or needs any prepared-feature metadata.
        order = rng.permutation(total)
        indices = order[:min(quota, total)]
        doc = {
            "version": PARTITION_VERSION, "client_id": config.get("client_id"),
            "source_ids": [str(source_ids[i]) for i in indices],
            "sample_seed": int(config["sample_seed"]), "source_pool_sample_count": total,
            "requested_quota": quota, "effective_quota": len(indices),
            "requested_strategy": strategy, "effective_strategy": strategy, "strategy": strategy,
            "axis_family": None, "skew_axis": None, "bucket_spec": {},
            "target_bucket_counts": {}, "realised_bucket_counts": {}, "bucket_histogram": {},
            "normalised_bucket_entropy": None, "capacity_reduction_count": quota - len(indices),
            "cross_client_overlap_allowed": True, "allocation_policy": "uniform_without_replacement",
            "resolution_reason": "source_pool_exhausted" if quota > total else "requested_target_met",
            "quantity_policy": "requested_client_sample_size",
        }
        doc["partition_hash"] = sha256_json(doc)
        return {"indices": indices, "candidate_indices": order, "metadata": doc}
    if not config.get("skew_axis"):
        raise ValueError(f"{strategy} requires an explicit skew_axis")
    axis = resolve_skew_axis(x, y, meta, split_name="train", task_family=task_family,
                             hf_task=config.get("hf_task"), requested_axis=config.get("skew_axis"),
                             axis_config=config.get("skew_axis_config"))
    if axis.bucket_ids is None or axis.fallback_reason:
        raise ValueError(f"requested skew axis is unavailable: {config.get('skew_axis')}: {axis.fallback_reason}")
    bucket_ids = np.zeros(total, dtype=int) if axis.bucket_ids is None else np.asarray(axis.bucket_ids)
    if len(bucket_ids) != total:
        raise ValueError("skew axis does not align with source rows")
    buckets, capacities = np.unique(bucket_ids, return_counts=True)
    weights = capacities / capacities.sum()
    support_policy, min_fraction = _regression_settings(config)
    general_regression = task_family == "regression" and support_policy == "general"
    if general_regression and axis.effective_axis != "score_bin":
        raise ValueError("general-purpose regression requires target score_bin coverage")
    support = np.arange(len(buckets))
    shard_evidence = {}
    coverage = {}
    from .reviewed_opportunity import requirements
    required = requirements(config)
    if required and (axis.effective_axis != "class_label" or len(buckets) < required["minimum_classes"]):
        raise ValueError("supported classification source is missing required classes")
    # Keep declared coverage controls for explicitly skewed calibrated profiles.
    from .training import CALIBRATED_SAMPLER_VERSIONS
    if (config.get("recipe_sampler_version") in CALIBRATED_SAMPLER_VERSIONS
            and task_family in {"classification", "token_classification"}
            and axis.effective_axis in {"class_label", "entity_present_sentence"}):
        calibration = (config.get("training_policy") or {}).get("calibration", {})
        coverage = calibration.get("class_coverage", {}).get(config.get("calibration_band"), {})
        if coverage.get("require_source_classes") and len(buckets) < coverage["minimum_classes"]:
            raise ValueError("supported classification source is missing required classes")
    source_weights = weights.copy()
    if strategy == "dirichlet":
        alpha = float(config["distribution_param"] if config.get("distribution_param") is not None else (10 ** (1 - 2 * intensity)))
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Dirichlet concentration must be positive and finite")
        weights = rng.dirichlet(alpha * len(buckets) * weights)
    elif strategy == "shard":
        expected_count = len(buckets) ** (1 - intensity)
        # Generated recipes stratify this fractional draw across the roster.
        quantile = config.get("shard_support_quantile")
        if quantile is None:
            support_seed = int(sha256_json([int(config["sample_seed"]), "shard_support_v2"])[:8], 16)
            quantile = np.random.default_rng(support_seed).random()
        quantile = float(quantile)
        lower = int(np.floor(expected_count))
        count = lower + int(quantile < expected_count - lower)
        shard_evidence = {"shard_expected_bucket_count": float(expected_count),
                          "shard_support_quantile": quantile}
        if coverage:
            count = min(len(buckets), max(count, int(coverage["minimum_classes"])))
        if general_regression:
            count = min(len(buckets), max(count, 2, int(np.ceil(min_fraction * len(buckets)))))
            # End bins anchor the source target range; interior bins remain seeded.
            ends = np.unique([0, len(buckets) - 1])
            interior = np.setdiff1d(np.arange(len(buckets)), ends)
            extra = rng.choice(interior, count - len(ends), replace=False,
                               p=weights[interior] / weights[interior].sum()) if count > len(ends) else []
            support = np.sort(np.concatenate([ends, extra]).astype(int))
        else:
            support = np.sort(rng.choice(len(buckets), count, replace=False, p=weights))
        weights = weights * np.isin(np.arange(len(buckets)), support)
        weights /= weights.sum()
    if coverage and strategy == "dirichlet":
        mix = float(coverage["source_mix"])
        weights = mix * source_weights + (1.0 - mix) * weights
    effective_request = min(quota, total)
    requested = _largest_remainder_counts(weights, effective_request, tie_keys=buckets.tolist())
    realised = np.minimum(requested, capacities)
    clipped = realised.copy()
    require_classes = bool(coverage and coverage["minimum_classes"] > 1)
    realised = _capacity_counts(requested, capacities, weights, support, require_coverage=general_regression or require_classes)
    indices = np.concatenate([
        rng.choice(np.flatnonzero(bucket_ids == bucket), int(count), replace=False)
        for bucket, count in zip(buckets, realised) if count
    ]).astype(int)
    rng.shuffle(indices)
    if not len(indices):
        raise ValueError("client sample is empty")
    labels = [str(axis.bucket_labels.get(str(b), str(b))) for b in buckets]
    p = realised[realised > 0] / realised.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(len(buckets))) if len(buckets) > 1 else 0.0
    selected = [str(source_ids[i]) for i in indices]
    effective = strategy if axis.bucket_ids is not None else "iid"
    doc = {
        "version": version, "client_id": config.get("client_id") or f"client_{int(config['client_partition_index']):04d}",
        "source_ids": selected, "sample_seed": int(config["sample_seed"]),
        "source_pool_sample_count": total, "requested_quota": quota, "effective_quota": len(indices),
        "requested_strategy": strategy, "effective_strategy": effective,
        "strategy": effective, "axis_family": axis.axis_family, "skew_axis": axis.effective_axis,
        "bucket_spec": axis.bucket_spec, "source_fields": axis.source_fields,
        "target_bucket_counts": dict(zip(labels, map(int, requested))),
        "realised_bucket_counts": dict(zip(labels, map(int, realised))),
        "bucket_histogram": dict(zip(labels, map(int, realised))),
        "selected_bucket_count": len(support), "represented_bucket_count": int(np.count_nonzero(realised)),
        "normalised_bucket_entropy": entropy,
        "target_deviation_count": int(np.abs(requested - realised).sum()),
        "capacity_reduction_count": quota - len(indices),
        "fallback_reason": axis.fallback_reason,
        "resolution_reason": "capacity_reduction" if len(indices) < quota else "requested_target_met",
        "cross_client_overlap_allowed": True,
    }
    doc.update(heterogeneity_intensity=intensity, source_bucket_count=len(buckets),
               selected_buckets=[labels[i] for i in support], **shard_evidence)
    if coverage:
        doc.update(classification_sampling_policy="calibrated_class_coverage_v1",
                   calibration_band=config["calibration_band"], classification_coverage=coverage)
    doc.update(allocation_policy="redistribute_within_support",
               redistributed_sample_count=int(np.maximum(realised - clipped, 0).sum()),
               selected_support_capacity=int(capacities[support].sum()))
    if task_family == "regression":
        doc.update(regression_support_policy=support_policy,
                   min_target_bin_fraction=min_fraction if general_regression else None,
                   target_bin_coverage=float(np.count_nonzero(realised) / len(buckets)),
                   target_extremes_represented=bool(realised[0] > 0 and realised[-1] > 0),
                   controlled_specialist=support_policy == "specialist")
    doc["partition_hash"] = sha256_json(doc)
    # Continue through unused candidates deterministically if preparation rejects
    # examples. Prefer the declared support, then relax it only on exhaustion.
    remaining = rng.permutation(np.setdiff1d(np.arange(total), indices))
    in_support = np.isin(bucket_ids[remaining], buckets[support])
    order = np.concatenate([indices, remaining[in_support], remaining[~in_support]])
    doc["selected_support_capacity"] = int(capacities[support].sum())
    indices = order[:min(quota, total)]
    doc.update(version=PARTITION_VERSION, source_ids=[str(source_ids[i]) for i in indices],
               effective_quota=len(indices), capacity_reduction_count=quota - len(indices),
               support_relaxation_count=int(np.count_nonzero(~np.isin(bucket_ids[indices], buckets[support]))))
    update_bucket_evidence(doc, bucket_ids[indices], axis.bucket_labels)
    if strategy == "dirichlet":
        doc["dirichlet_alpha"] = alpha
    doc.pop("partition_hash", None)
    doc["partition_hash"] = sha256_json(doc)
    return {"indices": indices, "candidate_indices": order, "metadata": doc,
            "bucket_ids": bucket_ids, "bucket_labels": axis.bucket_labels}


def update_bucket_evidence(doc, bucket_ids, labels):
    """Describe accepted rows, including support relaxed after exhaustion."""
    from ..data.skew_axes import bucket_distribution
    histogram = bucket_distribution(bucket_ids, labels)
    counts = np.asarray(list(histogram.values()), dtype=float)
    p = counts / counts.sum() if counts.size else counts
    cardinality = int(doc.get("source_bucket_count", len(labels)))
    doc.update(realised_bucket_counts=histogram, bucket_histogram=histogram,
        represented_bucket_count=len(histogram),
        normalised_bucket_entropy=float(-(p * np.log(p)).sum() / np.log(cardinality)) if cardinality > 1 else 0.0,
        support_relaxation_count=sum(n for bucket, n in histogram.items() if bucket not in doc.get("selected_buckets", histogram)),
        target_deviation_count=sum(abs(n - histogram.get(bucket, 0)) for bucket, n in doc["target_bucket_counts"].items()))


def materialize_partition(x: Any, y: Any, partition: Mapping[str, Any]) -> dict:
    doc = partition["metadata"]
    return {"x_train": _take_rows(x, partition["indices"]), "y_train": _take_rows(y, partition["indices"]),
            "split_resolution": {"strategy": doc["effective_strategy"], "effective_strategy": doc["effective_strategy"],
                         "requested_strategy": doc["requested_strategy"], "bucket_distribution": doc["bucket_histogram"]},
            "partition_meta": doc}
