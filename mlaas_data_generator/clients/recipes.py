"""Generator-independent client heterogeneity; lineage belongs to the caller."""
from __future__ import annotations

from collections import Counter
import math
import numpy as np

from ..provenance import sha256_json
from . import training
from .partitioning import PARTITION_VERSION
from .quantities import SAMPLE_QUANTITY_SPREAD, sample_quantity_bounds, stratified_sample_quantities

PROFILES = ("iid", "dirichlet", "shard", "quantity_skew")


def recipe_summary(rows: list[dict]) -> dict:
    """Count joint settings separately from quantity/seed variation."""
    def clean(value):
        if hasattr(value, "item"):
            value = value.item()
        return None if isinstance(value, float) and not math.isfinite(value) else value
    groups = {}
    for row in rows:
        settings = {key: clean(row.get(key)) for key in (*training.KNOBS, "momentum")}
        key = sha256_json(settings)
        group = groups.setdefault(key, {"recipe_hash": key, "settings": settings, "count": 0})
        group["count"] += 1
    n = len(rows)
    return {"client_count": n, "unique_training_recipes": len(groups),
        "largest_recipe_share": max((g["count"] for g in groups.values()), default=0) / n if n else None,
        "unique_sample_quantities": len({clean(r.get("sample_size")) for r in rows}),
        "profile_counts": dict(Counter(str(clean(r.get("split_strategy"))) for r in rows)),
        "optimizer_counts": dict(Counter(str(clean(r.get("optimizer"))) for r in rows)),
        "calibration_band_counts": dict(Counter(clean(r.get("calibration_band")) or "uncalibrated" for r in rows)),
        "recipes": sorted(groups.values(), key=lambda g: (-g["count"], g["recipe_hash"]))}


def _seed(*parts) -> int:
    return int(sha256_json(parts)[:8], 16)


def assign_shard_support_quantiles(rows: list[dict], *, seed: str) -> None:
    """Freeze evenly spaced support draws across the final v2 shard roster."""
    shards = [row for row in rows if row.get("split_strategy") == "shard"
              and row.get("partition_generator_version") == PARTITION_VERSION]
    quantiles = [(i + 0.5) / len(shards) for i in range(len(shards))]
    np.random.default_rng(_seed(seed, "shard_support_v2")).shuffle(quantiles)
    for row, quantile in zip(shards, quantiles):
        row["shard_support_quantile"] = quantile


def sample_client_recipes(policy: dict, *, count: int, seed: str,
                          heterogeneity_intensity: float = 0.5,
                          average_samples: int = 600,
                          quantity_seed: str | None = None,
                          regression_support_policy: str = "general",
                          min_target_bin_fraction: float = 0.6) -> list[dict]:
    """Balance data profiles and training recipes and independently stratify local quantities.

    A seed is a recipe namespace, never a checkpoint or composition identity.
    """
    quantities = stratified_sample_quantities(average_samples, count=count,
        seed=seed if quantity_seed is None else quantity_seed)
    _, max_client_quantity = sample_quantity_bounds(average_samples)
    # Scale the realised integer draws, preserving their ordering and rounding.
    multiplier = policy.get("sample_quantity_multiplier", 1)
    quantities = [quantity * multiplier for quantity in quantities]
    max_client_quantity *= multiplier
    from .capacity import scale_quantity
    original_quantities = list(quantities)
    original_maximum = max_client_quantity
    quantities = [scale_quantity(q, original_maximum, policy.get("known_training_capacity"))[0] for q in quantities]
    max_client_quantity = scale_quantity(original_maximum, original_maximum, policy.get("known_training_capacity"))[1]
    if not 0 <= heterogeneity_intensity <= 1:
        raise ValueError("heterogeneity intensity must be in [0,1]")
    from .partitioning import validate_partition_config
    validate_partition_config({"regression_support_policy": regression_support_policy,
                               "min_target_bin_fraction": min_target_bin_fraction})
    rng = np.random.default_rng(_seed(seed, "profiles"))
    available_profiles = tuple(policy.get("partition_profiles", PROFILES))
    profiles = [available_profiles[i % len(available_profiles)] for i in range(count)]
    rng.shuffle(profiles)
    knobs = training.sample_training_knobs(policy, profiles, seed=seed)
    rows = []
    for index, profile in enumerate(profiles):
        client_id = f"client_{index:04d}"
        quantity = quantities[index]
        rows.append({**knobs[index], "recipe_sampler_version": policy["version"],
            "training_policy_hash": sha256_json(policy), "training_policy": policy,
            "client_id": client_id, "client_partition_index": index,
            "split_strategy": profile, "sample_size": quantity,
            "requested_quota": quantity, "max_client_quantity": max_client_quantity,
            "unconstrained_sample_size": original_quantities[index],
            "quantity_capacity_policy": policy.get("quantity_capacity_policy"),
            "partition_generator_version": PARTITION_VERSION,
            "heterogeneity_intensity": heterogeneity_intensity,
            "sample_seed": _seed(seed, client_id, "sample"),
            "training_seed": _seed(seed, client_id, "training")})
        rows[-1]["skew_axis"] = policy.get("skew_axis") if profile in {"dirichlet", "shard"} else None
        if policy.get("task") in {"regression", "tabular_regression", "sentence_similarity"}:
            rows[-1].update(regression_support_policy=regression_support_policy,
                            min_target_bin_fraction=min_target_bin_fraction)
        training.constrain_training_opportunity(rows[-1], seed=f"{seed}:{client_id}")
    assign_shard_support_quantiles(rows, seed=seed)
    return rows
