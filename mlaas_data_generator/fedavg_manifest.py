"""FedAvg manifest expansion and intensity-controlled client targets."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from .clients.quantities import QUANTITY_SAMPLER_VERSION, SAMPLE_QUANTITY_SPREAD, stratified_sample_quantities
from typing import Any, Mapping

from mlaas_data_generator.fedavg_eligibility import (
    fedavg_benchmark_sample_seed,
    fedavg_run_partition_seed,
)

PARTITION_GENERATOR_VERSION = "source_pool_sampler_v3"
FEDAVG_CLIENT_SAMPLE_CAP = 1500
LEGACY_PARTITION_GENERATOR_VERSION = "sequential_depletion_v1"
FEDAVG_CLIENT_PROFILES: tuple[str, ...] = (
    "iid",
    "dirichlet",
    "shard",
    "quantity_skew",
)

# Kept as a public compatibility alias.  V2 does not use the old severity
# grid: every run contains the four profiles and severity comes from h.
FEDAVG_SPLIT_STRATEGIES: tuple[tuple[str, float | None], ...] = tuple(
    (profile, None) for profile in FEDAVG_CLIENT_PROFILES
)

SAMPLE_SIZE_FRACTIONS: tuple[float, ...] = (0.5, 0.65, 0.8, 0.9, 1.0)


def _client_identity(value: str | int) -> str:
    return f"client_{int(value):03d}" if isinstance(value, int) else str(value)


def fedavg_client_seed(fl_run_id: str, client_id: str | int) -> int:
    identity = _client_identity(client_id)
    digest = hashlib.sha256(f"{fl_run_id}:client:{identity}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def fedavg_training_seed(fl_run_id: str, client_id: str | int, round_id: int) -> int:
    identity = _client_identity(client_id)
    digest = hashlib.sha256(
        f"{fl_run_id}:client:{identity}:round:{round_id}:training".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def fedavg_participation_schedule(
    fl_run_id: str,
    *,
    clients_per_run: int,
    rounds: int,
    participation_rate: float,
) -> tuple[dict[int, set[int]], int]:
    """Return a deterministic, balanced rotating client schedule."""
    clients_per_run = int(clients_per_run)
    rounds = int(rounds)
    if clients_per_run < 2:
        raise ValueError("fedavg_clients_per_run must be at least 2")
    if rounds < 2:
        raise ValueError("fedavg_rounds must be at least 2")
    if not 0.0 < float(participation_rate) <= 1.0:
        raise ValueError("fedavg_client_participation_rate must be in (0, 1]")

    digest = hashlib.sha256(f"{fl_run_id}:participation".encode("utf-8")).hexdigest()
    schedule_seed = int(digest[:8], 16)
    order = list(range(clients_per_run))
    random.Random(schedule_seed).shuffle(order)
    scheduled_count = min(
        clients_per_run,
        max(2, int(math.ceil(float(participation_rate) * clients_per_run))),
    )
    schedule: dict[int, set[int]] = {}
    for round_id in range(1, rounds + 1):
        offset = ((round_id - 1) * scheduled_count) % clients_per_run
        selected = {
            order[(offset + position) % clients_per_run]
            for position in range(scheduled_count)
        }
        schedule[round_id] = selected
    return schedule, schedule_seed


def fedavg_client_knobs(
    *,
    base_row: Mapping[str, Any],
    client_index: int,
    task_key: str,
    profile_batch_sizes: tuple[int, ...],
    profile_learning_rates: tuple[float, ...],
    profile_epochs: tuple[int, ...],
    profile_optimizers: tuple[str, ...],
) -> dict[str, Any]:
    batch_size = profile_batch_sizes[int(client_index) % len(profile_batch_sizes)]
    learning_rate = profile_learning_rates[int(client_index) % len(profile_learning_rates)]
    training_epochs = profile_epochs[int(client_index) % len(profile_epochs)]
    optimizer = profile_optimizers[int(client_index) % len(profile_optimizers)]
    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_epochs": training_epochs,
        "optimizer": optimizer,
        "knob_variant": int(client_index),
    }


def stratified_heterogeneity_intensities(run_count: int, *, seed: int) -> list[float]:
    """Return a seeded shuffle of midpoint strata covering ``[0, 1]``."""
    count = int(run_count)
    if count < 0:
        raise ValueError("fedavg_runs must be non-negative")
    values = [(rank + 0.5) / count for rank in range(count)] if count else []
    digest = hashlib.sha256(f"fedavg-heterogeneity:{int(seed)}:{count}".encode()).hexdigest()
    random.Random(int(digest[:16], 16)).shuffle(values)
    return values


def _normalise_intensity(value: float) -> float:
    intensity = float(value)
    if not math.isfinite(intensity) or not 0.0 <= intensity <= 1.0:
        raise ValueError("fedavg_heterogeneity_intensity must be in [0, 1]")
    return intensity


def _profile_roster(fl_run_id: str, clients_per_run: int) -> list[str]:
    roster = [
        FEDAVG_CLIENT_PROFILES[index % len(FEDAVG_CLIENT_PROFILES)]
        for index in range(int(clients_per_run))
    ]
    digest = hashlib.sha256(f"{fl_run_id}:partition-profiles".encode()).hexdigest()
    random.Random(int(digest[:16], 16)).shuffle(roster)
    return roster


def _requested_partition_params(
    profile: str,
    *,
    intensity: float,
    quantity_multiplier: float | None,
) -> dict[str, Any]:
    common = {
        "profile": profile,
        "heterogeneity_intensity": float(intensity),
        "target_basis": "global_bucket_proportions",
    }
    if profile == "iid":
        return common
    if profile == "dirichlet":
        common.update(
            alpha=float(10.0 ** (1.0 - 2.0 * intensity)),
            concentration="alpha * K * global_proportions",
        )
    elif profile == "shard":
        common.update(
            bucket_count_formula="max(1, round(K ** (1 - h)))",
            selected_bucket_weighting="global_prevalence",
        )
    elif profile == "quantity_skew":
        common.update(
            quota_multiplier=float(quantity_multiplier if quantity_multiplier is not None else 1.0),
            quota_multiplier_interval=[
                1 - SAMPLE_QUANTITY_SPREAD,
                1 + SAMPLE_QUANTITY_SPREAD,
            ],
            maximum_quantity_quota_ratio=(1 + SAMPLE_QUANTITY_SPREAD) / (1 - SAMPLE_QUANTITY_SPREAD),
        )
    return common


def expand_fedavg_run(
    base_row: Mapping[str, Any],
    *,
    run_index: int,
    clients_per_run: int,
    rounds: int,
    participation_rate: float,
    seed: int,
    heterogeneity_intensity: float = 0.5,
    original_fl_run_id: str | None = None,
    average_samples: int | None = None,
    service_id_fn,
    service_config_fn,
    client_knobs_fn,
) -> list[dict[str, Any]]:
    if int(clients_per_run) < 4:
        raise ValueError("fedavg_clients_per_run must be at least 4")
    intensity = _normalise_intensity(heterogeneity_intensity)
    base = dict(base_row)
    source_cap = base.get("fedavg_source_pool_cap")
    if source_cap is not None:
        source_cap = int(source_cap)
        if source_cap < 1:
            raise ValueError("fedavg_source_pool_cap must be positive or omitted")
    base_quota = int(average_samples if average_samples is not None else
                     (base.get("sample_size") or base.get("max_samples") or 512))
    quantities = stratified_sample_quantities(base_quota, count=int(clients_per_run),
                                             seed=f"{seed}:{run_index}")
    base["generation_mode"] = "fedavg"
    base["fedavg_run_index"] = int(run_index)
    base["fedavg_clients_per_run"] = int(clients_per_run)
    base["fedavg_rounds"] = int(rounds)
    base["fedavg_client_participation_rate"] = float(participation_rate)
    fl_run_id = service_id_fn(
        "fl_run",
        {
            "task": base.get("task"),
            "model": base.get("hf_model_id") or base.get("model_type"),
            "dataset": base.get("dataset_name"),
            "training_regime": base.get("training_regime"),
            "run_index": run_index,
            "seed": seed,
            "partition_generator_version": PARTITION_GENERATOR_VERSION,
            "heterogeneity_intensity": intensity,
            "fedavg_source_pool_cap": source_cap,
            "base_sample_size": base_quota,
            "quantity_sampler_version": QUANTITY_SAMPLER_VERSION,
        },
    )
    lineage_id = f"lineage__{fl_run_id}"
    run_partition_seed = fedavg_run_partition_seed(fl_run_id)
    benchmark_sample_seed = fedavg_benchmark_sample_seed(fl_run_id)
    schedule, schedule_seed = fedavg_participation_schedule(
        fl_run_id,
        clients_per_run=int(clients_per_run),
        rounds=int(rounds),
        participation_rate=float(participation_rate),
    )
    roster = _profile_roster(fl_run_id, int(clients_per_run))
    client_knobs: dict[int, dict[str, Any]] = {}
    for client_index in range(int(clients_per_run)):
        knobs = client_knobs_fn(
            base_row=base,
            client_index=client_index,
            task_key=str(base.get("task") or base.get("task_key") or ""),
        )
        profile = roster[client_index]
        requested_quota = quantities[client_index]
        multiplier = requested_quota / base_quota
        params = _requested_partition_params(
            profile,
            intensity=intensity,
            quantity_multiplier=multiplier,
        )
        params.update(quantity_sampler_version=QUANTITY_SAMPLER_VERSION,
                      average_samples=base_quota, sample_quantity_spread=SAMPLE_QUANTITY_SPREAD)
        knobs.update(
            split_strategy=profile,
            distribution_type=profile,
            distribution_param=params.get("alpha") if profile == "dirichlet" else None,
            requested_partition_params=json.dumps(
                params, sort_keys=True, separators=(",", ":")
            ),
            requested_quota=requested_quota,
            sample_size=requested_quota,
        )
        client_knobs[client_index] = knobs
    requested_total_samples = sum(
        max(0, int(knobs.get("sample_size") or 0)) for knobs in client_knobs.values()
    )
    benchmark_sample_size = max(1, base_quota // 5)
    rows: list[dict[str, Any]] = []
    for client_index in range(int(clients_per_run)):
        client_id = f"client_{client_index:03d}"
        for round_id in range(1, int(rounds) + 1):
            row = copy.deepcopy(base)
            row.update(client_knobs[client_index])
            row["fl_run_id"] = fl_run_id
            row["lineage_id"] = lineage_id
            row["client_id"] = client_id
            row["round_id"] = round_id
            row["client_partition_index"] = client_index
            row["run_partition_seed"] = run_partition_seed
            row["benchmark_sample_seed"] = benchmark_sample_seed
            row["benchmark_sample_size"] = benchmark_sample_size
            row["sample_seed"] = fedavg_client_seed(fl_run_id, client_id)
            row["training_seed"] = fedavg_training_seed(fl_run_id, client_id, round_id)
            row["run_train_pool_size"] = source_cap
            row["fedavg_source_pool_cap"] = source_cap
            row["requested_total_samples"] = requested_total_samples
            row["max_samples"] = source_cap
            row["heterogeneity_intensity"] = intensity
            row["partition_generator_version"] = PARTITION_GENERATOR_VERSION
            row["original_fl_run_id"] = original_fl_run_id
            row["participation_schedule_seed"] = schedule_seed
            row["participation_schedule_version"] = "balanced_rotating_v1"
            row["scheduled_to_participate"] = client_index in schedule[round_id]
            row["service_id"] = service_id_fn(
                "fedavg_client_round",
                {
                    "fl_run_id": fl_run_id,
                    "client_id": client_id,
                    "round_id": round_id,
                },
            )
            case_base = base.get("case_name") or base.get("service_id") or fl_run_id
            row["case_name"] = f"{case_base}__{client_id}__round_{round_id:03d}"
            row["notes"] = (
                "multi-round fedavg client observation; shared round-global parent; "
                f"run={fl_run_id}; round={round_id}"
            )
            row["service_config"] = service_config_fn(row)
            rows.append(row)
    return rows
