"""Active, genuine multi-round FedAvg coordinator for manifest execution."""

from __future__ import annotations

from mlaas_data_generator.terminal import debug_print, debug_enabled, terminal_scope, print_results, print_fedavg_summary, print_client_completion, compact_metrics

import hashlib
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mlaas_data_generator.checkpoint_store import (
    ContentAddressedTensorStore,
    TensorState,
    capture_model_tensor_state,
    deserialize_tensor_state,
    ensure_lineage_base,
    restore_model_tensor_state,
)
from mlaas_data_generator.data.fedavg_partition import (
    FedAvgPartitionSpec,
    plan_fedavg_partitions,
)
from mlaas_data_generator.fedavg_aggregation import (
    FEDAVG_AGGREGATION_VERSION,
    FedAvgContributor,
    aggregate_fedavg_states,
)
from mlaas_data_generator.fedavg_eligibility import (
    fedavg_benchmark_sample_seed,
    fedavg_compatibility_key,
    fedavg_run_partition_seed,
    is_fedavg_eligible_row,
)
from mlaas_data_generator.fedavg_manifest import (
    FEDAVG_CLIENT_PROFILES,
    PARTITION_GENERATOR_VERSION,
    FEDAVG_CLIENT_SAMPLE_CAP,
    fedavg_client_seed,
    fedavg_participation_schedule,
    fedavg_training_seed,
)
from mlaas_data_generator.provenance import (
    dataset_metadata,
    label_semantics,
    model_identity,
    parameter_schema,
    preprocessing_metadata,
    sha256_json,
    write_json_artifact,
)
from mlaas_data_generator.services import runner as service_runner
from mlaas_data_generator.services.runner import ServiceRunner, canonical_task_family
from mlaas_data_generator.services.taxonomy import canonical_metric_names, metric_domain
from mlaas_data_generator.storage.writer import make_writer


PARTICIPATION_SCHEDULE_VERSION = "balanced_rotating_v1"
AGGREGATION_METHOD = "sample_weighted_fedavg"
_CLIENT_FIXED_FIELDS = (
    "client_partition_index",
    "split_strategy",
    "distribution_type",
    "distribution_param",
    "sample_size",
    "sample_seed",
    "requested_partition_params",
    "requested_quota",
    "skew_axis",
    "skew_axis_config",
    "optimizer",
    "learning_rate",
    "batch_size",
    "training_epochs",
    "weight_decay",
    "warmup_ratio",
    "gradient_accumulation_steps",
)
_RUN_SHARED_FIELDS = (
    "lineage_id",
    "generation_mode",
    "fedavg_clients_per_run",
    "fedavg_rounds",
    "fedavg_client_participation_rate",
    "run_partition_seed",
    "run_train_pool_size",
    "fedavg_source_pool_cap",
    "requested_total_samples",
    "benchmark_sample_seed",
    "benchmark_sample_size",
    "evaluation_seed",
    "participation_schedule_seed",
    "participation_schedule_version",
    "heterogeneity_intensity",
    "partition_generator_version",
    "original_fl_run_id",
    "mixed_precision",
    "precision_type",
    "input_schema",
    "output_schema",
    "label_column",
    "column_mapping",
    "db_path",
)
_APPROVED_ROW_VARIATION = set(_CLIENT_FIXED_FIELDS) | {
    "client_id",
    "client_partition_index",
    "round_id",
    "service_id",
    "case_name",
    "notes",
    "service_config",
    "row_index",
    "_manifest_total",
    "knob_variant",
    "training_seed",
    "scheduled_to_participate",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(db_path: str, method: str, row: Mapping[str, Any]) -> None:
    writer = make_writer("sqlite", db_path=db_path)
    try:
        writer.start()
        getattr(writer, method)(row)
        writer.finish()
    except Exception:
        writer.abort()
        raise


def recover_abandoned_fedavg_runs(
    db_path: str,
    *,
    reason: str = "interrupted process detected during startup recovery",
    fl_run_id: str | None = None,
) -> dict[str, int]:
    """Convert durable ``running`` markers left by a stopped process."""
    database = Path(db_path)
    if not database.exists():
        return {"runs": 0, "rounds": 0}
    with sqlite3.connect(str(database)) as connection:
        has_fedavg_state = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fedavg_runs'"
        ).fetchone()
    if has_fedavg_state is None:
        return {"runs": 0, "rounds": 0}
    writer = make_writer("sqlite", db_path=str(db_path))
    try:
        writer.start()
        recovered = writer.interrupt_running_fedavg(
            reason=reason,
            fl_run_id=fl_run_id,
        )
        writer.finish()
        return recovered
    except Exception:
        writer.abort()
        raise


def _global_eval_metrics(
    config: Mapping[str, Any],
    model: Any,
    x_test: Any,
    y_test: Any,
    *,
    task_family: str,
    global_round: int,
    state_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one aggregated global state on the immutable run benchmark."""
    evaluator = ServiceRunner(config)
    started = time.perf_counter()
    loss, primary, secondary, eval_qos = evaluator._evaluate_model(
        model,
        x_test,
        y_test,
        inference_only=False,
        task_family=task_family,
    )
    elapsed_s = float(time.perf_counter() - started)
    hf_task = config.get("hf_task")
    primary_name, secondary_name = canonical_metric_names(
        task_family,
        config.get("metric_key"),
        hf_task=hf_task,
        task_tag=config.get("task_tag"),
    )
    metric_score = service_runner._service_metric_score_value(
        task_family=task_family,
        hf_task=hf_task,
        primary_name=primary_name,
        primary_value=primary,
        secondary_name=secondary_name,
        secondary_value=secondary,
    )
    service_runner._validate_service_metrics(
        task_family=task_family,
        hf_task=hf_task,
        primary_name=primary_name,
        primary_value=primary,
        secondary_name=secondary_name,
        secondary_value=secondary,
        metric_score=metric_score,
    )
    benchmark_samples = service_runner._sample_count(x_test, y_test)
    metrics: dict[str, Any] = {
        "loss": service_runner._metric(loss, "quality", direction="lower_better"),
        primary_name: service_runner._metric(
            primary,
            "quality",
            direction=service_runner._quality_direction(primary_name),
        ),
        "metric_score": service_runner._metric(
            metric_score, "quality", direction="higher_better"
        ),
        "primary_metric_name": service_runner._metric(primary_name, "metadata"),
        "evaluation_runtime_s": service_runner._metric(
            elapsed_s, "runtime", "s", "lower_better"
        ),
        "benchmark_set_size": service_runner._metric(
            benchmark_samples, "metadata", "samples", "neutral"
        ),
        "global_model_role": service_runner._metric(f"G{global_round}", "metadata"),
        "global_state_hash": service_runner._metric(state_hash, "metadata"),
    }
    if secondary_name:
        metrics["auxiliary_metric_name"] = service_runner._metric(
            secondary_name, "metadata"
        )
        metrics[secondary_name] = service_runner._metric(
            secondary,
            "quality",
            direction=service_runner._quality_direction(secondary_name),
        )
    for name, value in dict(eval_qos or {}).items():
        normalized = str(name).strip().lower()
        if value is not None and normalized not in metrics:
            metrics[normalized] = service_runner._metric(
                value, metric_domain(normalized),
                direction="lower_better" if normalized in {"mse", "rmse", "mae"} else "neutral",
            )
    summary = {
        "global_model": f"G{global_round}",
        "global_state_hash": state_hash,
        "benchmark_samples": benchmark_samples,
        "loss": loss,
        "primary_metric_name": primary_name,
        "primary_metric": primary,
        "auxiliary_metric_name": secondary_name,
        "auxiliary_metric": secondary if secondary_name else None,
        "metric_score": metric_score,
        "evaluation_runtime_s": elapsed_s,
        "eval_qos": dict(eval_qos or {}),
    }
    return metrics, summary


def _write_completed_global_round(
    db_path: str,
    *,
    global_state_row: Mapping[str, Any],
    round_row: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    """Commit global state, completed round, and evaluation metrics together."""
    writer = make_writer("sqlite", db_path=db_path)
    try:
        writer.start()
        writer.write_fedavg_global_state(global_state_row)
        writer.write_fedavg_round(round_row)
        writer.write_fedavg_round_metrics(
            str(round_row["fl_run_id"]), int(round_row["round_id"]), metrics
        )
        writer.finish()
    except Exception:
        writer.abort()
        raise


def _sort_entries(entries: Sequence[Any]) -> list[Any]:
    return sorted(
        entries,
        key=lambda entry: (
            int(entry.resolved.get("round_id") or 0),
            str(entry.resolved.get("client_id") or ""),
        ),
    )


def validate_fedavg_run_entries(entries: list[Any]) -> str | None:
    """Validate the complete client-by-round grid before any run work begins."""
    if not entries:
        return "FedAvg run has no client-round rows"
    rows = [entry.resolved for entry in entries]
    first = rows[0]
    fl_run_id = str(first.get("fl_run_id") or "")
    if not fl_run_id or len({str(row.get("fl_run_id") or "") for row in rows}) != 1:
        return "FedAvg rows must share one non-empty fl_run_id"
    if str(first.get("generation_mode") or "fedavg").lower() != "fedavg":
        return "FedAvg rows must declare generation_mode=fedavg"
    if any(not is_fedavg_eligible_row(row) for row in rows):
        return "FedAvg run contains an unsupported model, task, or training regime"

    clients = sorted({str(row.get("client_id") or "") for row in rows})
    declared_clients = _safe_int(first.get("fedavg_clients_per_run"))
    declared_rounds = _safe_int(first.get("fedavg_rounds"))
    if not clients or any(not client for client in clients):
        return "FedAvg rows require non-empty client IDs"
    if declared_clients is None or declared_clients < 4 or declared_clients != len(clients):
        return "FedAvg client grid does not match fedavg_clients_per_run"
    if declared_rounds is None or declared_rounds < 2:
        return "FedAvg requires fedavg_rounds >= 2"
    if str(first.get("partition_generator_version") or "") != PARTITION_GENERATOR_VERSION:
        return f"FedAvg requires partition_generator_version={PARTITION_GENERATOR_VERSION}"
    intensity = _safe_float(first.get("heterogeneity_intensity"))
    if intensity is None or not 0.0 <= intensity <= 1.0:
        return "FedAvg requires heterogeneity_intensity in [0, 1]"
    rounds = list(range(1, declared_rounds + 1))
    expected_grid = {(client, round_id) for client in clients for round_id in rounds}
    actual_grid = {
        (str(row.get("client_id") or ""), int(row.get("round_id") or 0)) for row in rows
    }
    if actual_grid != expected_grid or len(rows) != len(expected_grid):
        return "FedAvg manifest must contain exactly one row for every client-round pair"
    service_ids = [str(row.get("service_id") or "") for row in rows]
    if any(not value for value in service_ids) or len(service_ids) != len(set(service_ids)):
        return "FedAvg client-round service IDs must be non-empty and unique"

    compatibility = fedavg_compatibility_key(first)
    for row in rows[1:]:
        if fedavg_compatibility_key(row) != compatibility:
            return "FedAvg rows disagree on model/dataset compatibility fields"
        for field in _RUN_SHARED_FIELDS:
            if _json_value(row.get(field)) != _json_value(first.get(field)):
                return f"FedAvg run-wide field {field!r} is inconsistent"
        for field in set(first) | set(row):
            if str(field).startswith("_") or field in _APPROVED_ROW_VARIATION:
                continue
            if _json_value(row.get(field)) != _json_value(first.get(field)):
                return f"FedAvg field {field!r} is not approved client-local heterogeneity"
    for client in clients:
        client_rows = sorted(
            (row for row in rows if str(row.get("client_id")) == client),
            key=lambda row: int(row.get("round_id") or 0),
        )
        for row in client_rows[1:]:
            for field in _CLIENT_FIXED_FIELDS:
                if _json_value(row.get(field)) != _json_value(client_rows[0].get(field)):
                    return f"FedAvg client-local field {field!r} changes across rounds for {client}"
        if len({int(row.get("training_seed") or -1) for row in client_rows}) != declared_rounds:
            return f"FedAvg training seeds must be round-specific for {client}"

    first_by_client = {
        client: next(row for row in rows if str(row.get("client_id")) == client)
        for client in clients
    }
    profiles = {
        client: service_runner._canonical_split_strategy(
            row.get("split_strategy") or "iid"
        )
        for client, row in first_by_client.items()
    }
    from .clients.profiles import ROSTER_VERSION
    from .clients.legacy_profiles import PROFILES as LEGACY_PROFILES, UNIFORM_PROFILE as LEGACY_UNIFORM_PROFILE
    versions = set()
    for row in first_by_client.values():
        params = row.get("requested_partition_params")
        if not isinstance(params, Mapping):
            return "FedAvg requested partition parameters are invalid"
        version = params.get("profile_roster_version")
        if version is not None and version != ROSTER_VERSION:
            return "Unsupported FedAvg profile roster version"
        versions.add(version)
    if len(versions) != 1:
        return "FedAvg profile roster versions are inconsistent"
    # Unversioned saved runs retain their original balanced roster and hashes.
    roster = LEGACY_PROFILES if versions == {None} else FEDAVG_CLIENT_PROFILES
    if not set(roster).issubset(profiles.values()):
        return "FedAvg run must contain every declared distribution profile"
    expected_profile_counts = Counter(
        roster[index % len(roster)]
        for index in range(declared_clients)
    )
    if Counter(profiles.values()) != expected_profile_counts:
        return "FedAvg client profiles do not form the balanced repeated roster"
    requested_quotas: dict[str, int] = {}
    for client, row in first_by_client.items():
        quota = _safe_int(row.get("requested_quota"))
        params = row.get("requested_partition_params")
        if quota is None or quota < 1 or quota != _safe_int(row.get("sample_size")):
            return f"FedAvg requested quota is invalid for {client}"
        if (not isinstance(params, Mapping) or not params.get("quantity_sampler_version")) and quota > FEDAVG_CLIENT_SAMPLE_CAP:
            return f"FedAvg requested quota exceeds {FEDAVG_CLIENT_SAMPLE_CAP} for {client}"
        if not isinstance(params, Mapping):
            return f"FedAvg requested partition parameters are invalid for {client}"
        if service_runner._canonical_split_strategy(params.get("profile")) != profiles[client]:
            return f"FedAvg requested partition profile is inconsistent for {client}"
        requested_quotas[client] = quota
    from .clients.quantities import QUANTITY_SAMPLER_VERSION, SAMPLE_QUANTITY_SPREAD, stratified_sample_quantities
    quantity_params = [row["requested_partition_params"] for row in first_by_client.values()]
    if any(params.get("quantity_sampler_version") for params in quantity_params):
        average = _safe_int(quantity_params[0].get("average_samples"))
        if average is None or average < 1 or any(
            params.get("quantity_sampler_version") != QUANTITY_SAMPLER_VERSION
            or params.get("average_samples") != average
            or params.get("sample_quantity_spread") != SAMPLE_QUANTITY_SPREAD
            for params in quantity_params
        ):
            return "FedAvg sample quantity settings are inconsistent"
        expected = stratified_sample_quantities(average, count=declared_clients, seed="validation")
        if sorted(requested_quotas.values()) != sorted(expected):
            return "FedAvg requested quotas do not cover the sample quantity strata"
    else:
        non_quantity_quotas = {
            requested_quotas[client]
            for client in clients
            if profiles[client] != LEGACY_UNIFORM_PROFILE
        }
        if len(non_quantity_quotas) != 1:
            return "FedAvg non-quantity clients must share one requested base quota"
        quantity_quotas = [
            requested_quotas[client]
            for client in clients
            if profiles[client] == LEGACY_UNIFORM_PROFILE
        ]
        if len(quantity_quotas) >= 2 and max(quantity_quotas) / min(quantity_quotas) > 8.0**intensity + 1e-12:
            return "FedAvg requested historical quota ratio exceeds its intensity bound"
    if sum(requested_quotas.values()) != _safe_int(first.get("requested_total_samples")):
        return "FedAvg requested_total_samples does not equal requested client quotas"
    source_cap = first.get("fedavg_source_pool_cap")
    if source_cap is not None and (_safe_int(source_cap) is None or _safe_int(source_cap) < 1):
        return "FedAvg fedavg_source_pool_cap must be positive or omitted"

    participation_rate = _safe_float(first.get("fedavg_client_participation_rate"))
    if participation_rate is None or not 0.0 < participation_rate <= 1.0:
        return "FedAvg participation rate must be in (0, 1]"
    expected_schedule, expected_seed = fedavg_participation_schedule(
        fl_run_id,
        clients_per_run=declared_clients,
        rounds=declared_rounds,
        participation_rate=participation_rate,
    )
    if _safe_int(first.get("participation_schedule_seed")) != expected_seed:
        return "FedAvg participation schedule seed does not match the deterministic schedule"
    if str(first.get("participation_schedule_version") or "") != PARTICIPATION_SCHEDULE_VERSION:
        return "Unsupported FedAvg participation schedule version"
    if _safe_int(first.get("run_partition_seed")) != fedavg_run_partition_seed(fl_run_id):
        return "FedAvg run partition seed does not reproduce"
    if _safe_int(first.get("benchmark_sample_seed")) != fedavg_benchmark_sample_seed(fl_run_id):
        return "FedAvg benchmark seed does not reproduce"
    index_by_client = {
        str(row.get("client_id")): int(row.get("client_partition_index")) for row in rows
    }
    if len(set(index_by_client.values())) != declared_clients:
        return "FedAvg client partition indices must be unique"
    for row in rows:
        client_id = str(row.get("client_id"))
        round_id = int(row.get("round_id") or 0)
        if _safe_int(row.get("sample_seed")) != fedavg_client_seed(fl_run_id, client_id):
            return "FedAvg client partition seed does not reproduce"
        if _safe_int(row.get("training_seed")) != fedavg_training_seed(
            fl_run_id, client_id, round_id
        ):
            return "FedAvg client-round training seed does not reproduce"
    for round_id in rounds:
        scheduled = {
            index_by_client[str(row.get("client_id"))]
            for row in rows
            if int(row.get("round_id") or 0) == round_id
            and _as_bool(row.get("scheduled_to_participate"))
        }
        if scheduled != expected_schedule[round_id]:
            return f"FedAvg round {round_id} participation schedule does not reproduce"
        if len(scheduled) < 2:
            return f"FedAvg round {round_id} schedules fewer than two clients"
    return None


def _source_ids(meta: Mapping[str, Any], total: int) -> list[Any]:
    selected = meta.get("selected_train_rows")
    if isinstance(selected, np.ndarray):
        selected = selected.tolist()
    if isinstance(selected, (list, tuple)) and len(selected) == total:
        return list(selected)
    return list(range(total))


def _partition_specs(entries: Sequence[Any]) -> list[FedAvgPartitionSpec]:
    first_round = [entry for entry in entries if int(entry.resolved.get("round_id") or 0) == 1]
    specs: list[FedAvgPartitionSpec] = []
    for entry in sorted(first_round, key=lambda item: int(item.resolved["client_partition_index"])):
        row = entry.resolved
        specs.append(
            FedAvgPartitionSpec(
                client_id=str(row["client_id"]),
                client_index=int(row["client_partition_index"]),
                strategy=service_runner._canonical_split_strategy(row.get("split_strategy") or "iid"),
                distribution_param=row.get("distribution_param"),
                sample_size=int(row.get("sample_size") or 0),
                sample_seed=int(row.get("sample_seed") or 0),
                generator_version=str(row.get("partition_generator_version") or ""),
                heterogeneity_intensity=float(row.get("heterogeneity_intensity")),
                requested_partition_params=service_runner._parse_jsonish(
                    row.get("requested_partition_params")
                ),
                requested_quota=int(
                    row.get("requested_quota") or row.get("sample_size") or 0
                ),
                skew_axis=row.get("skew_axis"),
                skew_axis_config=service_runner._parse_jsonish(row.get("skew_axis_config")),
            )
        )
    return specs


def _client_round_row(
    row: Mapping[str, Any],
    *,
    status: str,
    reason: str | None,
    scheduled: bool,
    participated: bool,
    succeeded: bool,
    sample_count: int | None,
    partition_hash: str | None,
    parent_round: int | None,
    parent_hash: str | None,
    final_hash: str | None = None,
    weight: float | None = None,
    partition_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "fl_run_id": str(row["fl_run_id"]),
        "client_id": str(row["client_id"]),
        "round_id": int(row["round_id"]),
        "service_id": str(row["service_id"]),
        "scheduled": scheduled,
        "participated": participated,
        "succeeded": succeeded,
        "status": status,
        "status_reason": reason,
        "sample_count": sample_count,
        "aggregation_weight": weight,
        "training_seed": _safe_int(row.get("training_seed")),
        "partition_seed": _safe_int(row.get("sample_seed")),
        "partition_hash": partition_hash,
        "parent_global_round": parent_round,
        "parent_state_hash": parent_hash,
        "final_state_hash": final_hash,
        "compatibility_contract_hash": row.get("compatibility_contract_hash"),
        "metadata_json": {
            "participation_rate": row.get("fedavg_client_participation_rate"),
            "schedule_seed": row.get("participation_schedule_seed"),
            "schedule_version": row.get("participation_schedule_version"),
            "client_weights_retained": bool(
                succeeded
                and int(row.get("round_id") or 0)
                == int(row.get("fedavg_rounds") or -1)
            ),
            **(
                {
                    "partition_generator_version": partition_metadata.get(
                        "partition_generator_version"
                    ),
                    "heterogeneity_intensity": partition_metadata.get(
                        "heterogeneity_intensity"
                    ),
                    "target_deviation_count": partition_metadata.get(
                        "target_deviation_count"
                    ),
                    "partition_summary": {
                        key: partition_metadata.get(key)
                        for key in (
                            "requested_partition_strategy",
                            "requested_partition_params",
                            "requested_quota",
                            "effective_quota",
                            "quota_shortfall",
                            "target_bucket_counts",
                            "target_bucket_proportions",
                            "realised_bucket_counts",
                            "realised_bucket_proportions",
                            "majority_bucket_share",
                            "normalised_bucket_entropy",
                            "represented_bucket_count",
                            "l1_allocation_deviation",
                            "total_variation_distance",
                            "target_deviation_count",
                            "resolution_reason",
                            "partition_hash",
                            "partition_bucket_unit",
                        )
                    },
                }
                if partition_metadata is not None
                else {}
            ),
        },
    }


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not str(key).startswith("_")}


@terminal_scope(lambda entries, **kwargs: entries[0].resolved if entries else {})
def execute_fedavg_run(
    entries: list[Any],
    *,
    execute_entry: Callable[..., dict[str, Any]],
    build_dataset_args: Callable[[dict[str, Any]], dict[str, Any]],
    load_dataset=None,
    build_model=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute all rounds, persisting each verified global state before reuse."""
    load_dataset = load_dataset or service_runner.load_dataset
    validation_error = validate_fedavg_run_entries(entries)
    ordered_entries = _sort_entries(entries)
    first = ordered_entries[0].resolved if ordered_entries else {}
    fl_run_id = str(first.get("fl_run_id") or "")
    lineage_id = str(first.get("lineage_id") or "")
    db_path = str(first.get("db_path") or service_runner.CONFIG.get("db_path") or "mlaas.db")
    started_at = _utcnow()
    client_count = int(first.get("fedavg_clients_per_run") or 2)
    round_count = int(first.get("fedavg_rounds") or 2)
    participation_rate = float(first.get("fedavg_client_participation_rate") or 1.0)
    run_row = {
        "fl_run_id": fl_run_id or "invalid_fedavg_run",
        "lineage_id": lineage_id or "invalid_fedavg_lineage",
        "status": "running",
        "client_count": client_count,
        "round_count": round_count,
        "participation_rate": participation_rate,
        "schedule_seed": int(first.get("participation_schedule_seed") or 0),
        "schedule_version": str(first.get("participation_schedule_version") or PARTICIPATION_SCHEDULE_VERSION),
        "heterogeneity_intensity": _safe_float(first.get("heterogeneity_intensity")),
        "partition_generator_version": str(
            first.get("partition_generator_version") or PARTITION_GENERATOR_VERSION
        ),
        "compatibility_contract_hash": None,
        "compatibility_contract_json": None,
        "g0_state_hash": None,
        "partition_plan_hash": None,
        "partition_plan_ref": None,
        "training_universe_hash": None,
        "benchmark_cohort_hash": None,
        "benchmark_cohort_ref": None,
        "status_reason": None,
        "started_at": started_at,
        "completed_at": None,
        "metadata_json": {},
    }
    can_persist_run = bool(
        fl_run_id
        and lineage_id
        and client_count >= 4
        and round_count >= 2
        and 0.0 < participation_rate <= 1.0
    )
    if can_persist_run:
        _write(db_path, "write_fedavg_run", run_row)

    def fail_setup(message: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for entry in ordered_entries:
            result = execute_entry(entry, prepared=None, failure_message=message, status_override="failed")
            results.append(_public_result(result))
            if can_persist_run and all(
                key in entry.resolved for key in ("client_id", "round_id", "service_id")
            ):
                row = entry.resolved
                _write(
                    db_path,
                    "write_fedavg_client_round",
                    _client_round_row(
                        row,
                        status="failed",
                        reason=message,
                        scheduled=_as_bool(row.get("scheduled_to_participate")),
                        participated=False,
                        succeeded=False,
                        sample_count=None,
                        partition_hash=None,
                        parent_round=None,
                        parent_hash=None,
                    ),
                )
        run_row.update(status="failed", status_reason=message, completed_at=_utcnow())
        if can_persist_run:
            _write(db_path, "write_fedavg_run", run_row)
        return results, {
            "fl_run_id": fl_run_id,
            "lineage_id": lineage_id,
            "partition_generator_version": first.get("partition_generator_version"),
            "heterogeneity_intensity": first.get("heterogeneity_intensity"),
            "status": "failed",
            "error": message,
            "completed_rounds": 0,
        }

    if validation_error:
        return fail_setup(validation_error)

    try:
        dataset_args = build_dataset_args(first)
        dataset_args["max_samples"] = first.get("fedavg_source_pool_cap")
        dataset_args["benchmark_sample_size"] = first.get("benchmark_sample_size")
        dataset_args["benchmark_sample_seed"] = first.get("benchmark_sample_seed")
        train, test, loaded_meta = load_dataset(first.get("dataset", "hf"), **dataset_args)
        x_train, y_train = train
        x_test, y_test = test
        meta = dict(loaded_meta or {})
        x_test, y_test, benchmark_info = service_runner._cap_benchmark_split(
            x_test, y_test, config=first
        )
        prepared_dataset = ((x_train, y_train), (x_test, y_test), meta)
        task_family = canonical_task_family(
            first.get("task_type"), meta.get("hf_task") or first.get("hf_task")
        )
        model_build_config = dict(first)
        if str(first.get("training_regime") or "").lower() == "generic":
            # Keras weight decay is a local optimizer knob in FedAvg, not a
            # client-specific architectural regularizer.
            model_build_config["weight_decay"] = 0.0
        model = (
            build_model(model_build_config, meta, task_family)
            if build_model
            else ServiceRunner(model_build_config)._build_model(meta=meta, task_family=task_family)
        )
        schema_document = parameter_schema(model)
        schema_hash = sha256_json(schema_document)
        provenance_root = service_runner._provenance_artifact_root(first)
        compression = service_runner._checkpoint_compression(first)
        lineage_base = ensure_lineage_base(
            provenance_root,
            lineage_id,
            model,
            parameter_schema_hash=schema_hash,
            compression=compression,
        )
        g0 = capture_model_tensor_state(model)
        if g0.sha256 != lineage_base.tensor_state_sha256:
            raise ValueError("G0 model state does not match its immutable lineage base")
        store = ContentAddressedTensorStore(provenance_root)
        read_g0 = deserialize_tensor_state(store.read(lineage_base.base), framework=g0.framework)
        if read_g0.sha256 != g0.sha256:
            raise ValueError("G0 failed CAS read-back verification")

        total_train = service_runner._sample_count(x_train, y_train)
        source_ids = _source_ids(meta, total_train)
        training_universe = {
            "source_ids": source_ids,
            "sample_count": total_train,
            "source_pool_cap": first.get("fedavg_source_pool_cap"),
        }
        training_universe_hash = sha256_json(training_universe)
        benchmark_document = {
            "sampling": benchmark_info,
            "x": _json_value(x_test),
            "y": _json_value(y_test),
        }
        benchmark_hash = sha256_json(benchmark_document)
        benchmark_artifact = write_json_artifact(
            provenance_root, fl_run_id, "fedavg_benchmark_cohort", benchmark_document
        )

        partitions = plan_fedavg_partitions(
            x_train,
            y_train,
            _partition_specs(ordered_entries),
            run_seed=int(first.get("run_partition_seed") or 0),
            source_ids=source_ids,
            meta=meta,
            task_family=task_family,
            hf_task=meta.get("hf_task") or first.get("hf_task"),
        )
        partition_documents: dict[str, dict[str, Any]] = {}
        used_samples = sum(len(partition.indices) for partition in partitions.values())
        source_pool_summary = {
            "source_pool_sample_count": total_train,
            "requested_total_samples": int(first["requested_total_samples"]),
            "client_samples_used": used_samples,
            "unused_source_samples": total_train - used_samples,
        }
        print(f"[FedAvg source pool] available={total_train} used={used_samples} "
              f"unused={total_train - used_samples}", flush=True)
        for client_id, partition in partitions.items():
            document = dict(partition.metadata)
            document["partition_hash"] = sha256_json(document)
            partition_documents[client_id] = document
        partition_plan = {
            "version": "source_pool_partition_plan_v3",
            **source_pool_summary,
            "fl_run_id": fl_run_id,
            "partition_generator_version": first.get("partition_generator_version"),
            "heterogeneity_intensity": first.get("heterogeneity_intensity"),
            "run_partition_seed": first.get("run_partition_seed"),
            "training_universe_hash": training_universe_hash,
            "clients": partition_documents,
        }
        partition_plan_hash = sha256_json(partition_plan)
        partition_artifact = write_json_artifact(
            provenance_root, fl_run_id, "fedavg_partition_plan", partition_plan
        )

        for client_id, document in partition_documents.items():
            _write(
                db_path,
                "write_fedavg_client_partition",
                {
                    "fl_run_id": fl_run_id,
                    "client_id": client_id,
                    "requested_partition_strategy": document.get(
                        "requested_partition_strategy"
                    ),
                    "requested_partition_params_json": document.get(
                        "requested_partition_params"
                    ),
                    "requested_quota": document.get("requested_quota"),
                    "effective_quota": document.get("effective_quota"),
                    "quota_shortfall": document.get("quota_shortfall"),
                    "target_bucket_counts_json": document.get("target_bucket_counts"),
                    "target_bucket_proportions_json": document.get(
                        "target_bucket_proportions"
                    ),
                    "realised_bucket_counts_json": document.get(
                        "realised_bucket_counts"
                    ),
                    "realised_bucket_proportions_json": document.get(
                        "realised_bucket_proportions"
                    ),
                    "majority_bucket_share": document.get("majority_bucket_share"),
                    "normalised_bucket_entropy": document.get(
                        "normalised_bucket_entropy"
                    ),
                    "represented_bucket_count": document.get(
                        "represented_bucket_count"
                    ),
                    "l1_allocation_deviation": document.get(
                        "l1_allocation_deviation"
                    ),
                    "total_variation_distance": document.get(
                        "total_variation_distance"
                    ),
                    "target_deviation_count": document.get(
                        "target_deviation_count"
                    ),
                    "resolution_reason": document.get("resolution_reason"),
                    "partition_hash": document.get("partition_hash"),
                    "metadata_json": {
                        "skew_axis": document.get("skew_axis"),
                        "axis_family": document.get("axis_family"),
                        "partition_bucket_unit": document.get("partition_bucket_unit"),
                        "bucket_spec": document.get("bucket_spec"),
                    },
                },
            )

        identity = model_identity(model, first)
        labels = label_semantics(meta, task_family=task_family)
        preprocessing = preprocessing_metadata(meta)
        dataset_info = dataset_metadata(first, meta)
        compatibility_contract = {
            "version": "fedavg_compatibility_contract_v1",
            "architecture_id": identity["architecture_id"],
            "task_head_identity": {
                "task_family": task_family,
                "hf_task": first.get("hf_task"),
                "label_format": meta.get("label_format"),
                "num_labels": meta.get("num_labels", meta.get("num_classes")),
            },
            "model_revision": identity.get("model_revision"),
            "parameter_schema_hash": schema_hash,
            "label_output_semantics_hash": labels["label_map_hash"],
            "preprocessing_input_contract_hash": preprocessing["preprocessing_contract_hash"],
            "logical_dataset_contract_hash": dataset_info["logical_dataset_contract_hash"],
            "lineage_id": lineage_id,
            "g0_state_hash": g0.sha256,
        }
        compatibility_hash = sha256_json(compatibility_contract)
    except Exception as exc:  # noqa: BLE001
        return fail_setup(f"FedAvg setup failed: {exc}")

    run_row.update(
        compatibility_contract_hash=compatibility_hash,
        compatibility_contract_json=compatibility_contract,
        g0_state_hash=g0.sha256,
        partition_plan_hash=partition_plan_hash,
        partition_plan_ref=str(partition_artifact["artifact_uri"]),
        training_universe_hash=training_universe_hash,
        benchmark_cohort_hash=benchmark_hash,
        benchmark_cohort_ref=str(benchmark_artifact["artifact_uri"]),
        metadata_json={
            "parameter_schema_hash": schema_hash,
            **source_pool_summary,
            "original_fl_run_id": first.get("original_fl_run_id"),
        },
    )
    for entry in ordered_entries:
        entry.resolved["compatibility_contract_hash"] = compatibility_hash
        entry.resolved["training_universe_hash"] = training_universe_hash
        entry.resolved["benchmark_cohort_hash"] = benchmark_hash
        partition_summary = partition_documents[str(entry.resolved["client_id"])]
        entry.resolved["target_deviation_count"] = partition_summary.get(
            "target_deviation_count"
        )
    _write(db_path, "write_fedavg_run", run_row)
    _write(
        db_path,
        "write_fedavg_global_state",
        {
            "fl_run_id": fl_run_id,
            "global_round": 0,
            "state_hash": g0.sha256,
            "cas_ref": str(lineage_base.base.manifest_path),
            "parameter_schema_hash": schema_hash,
            "parent_global_round": None,
            "parent_state_hash": None,
            "storage_format": "full_immutable_tensor_state_v1",
            "compression_json": lineage_base.base.compression.as_dict(),
            "logical_bytes": lineage_base.base.logical_size_bytes,
            "physical_bytes": lineage_base.base.physical_size_bytes,
            "verified_at": _utcnow(),
            "metadata_json": {"role": "G0", "is_service": False},
        },
    )

    by_round = {
        round_id: [
            entry for entry in ordered_entries if int(entry.resolved.get("round_id")) == round_id
        ]
        for round_id in range(1, round_count + 1)
    }
    results: list[dict[str, Any]] = []
    parent_state: TensorState = g0
    parent_ref = lineage_base.base
    any_client_failures = False
    completed_rounds = 0
    prior_failure: str | None = None

    print_fedavg_summary(
        first, train_pool=total_train,
        benchmark_size=service_runner._sample_count(x_test, y_test),
        device=service_runner._resolve_execution_device(model),
        precision=getattr(getattr(model, "core", model), "effective_precision_type", None),
    )
    for round_id in range(1, round_count + 1):
        round_timer = time.perf_counter()
        print(f"\n========== ROUND {round_id}/{round_count} ==========", flush=True)
        round_started = _utcnow()
        round_entries = sorted(
            by_round[round_id], key=lambda entry: str(entry.resolved.get("client_id"))
        )
        scheduled_count = sum(
            _as_bool(entry.resolved.get("scheduled_to_participate")) for entry in round_entries
        )
        print(f"Participants: {scheduled_count}/{len(round_entries)}", flush=True)
        nonparticipants = [str(entry.resolved.get("client_id")) for entry in round_entries
                           if not _as_bool(entry.resolved.get("scheduled_to_participate"))]
        if nonparticipants:
            print("Not participating: " + ", ".join(nonparticipants), flush=True)
        participant_position = 0
        if prior_failure:
            print(f"[Round {round_id}/{round_count}] not executed | {prior_failure}", flush=True)
            for entry in round_entries:
                reason = "not executed because a prior FedAvg round failed"
                print(f"[Client skipped] round={round_id} client={entry.resolved.get('client_id')} | {reason}", flush=True)
                result = execute_entry(
                    entry,
                    prepared=None,
                    failure_message=reason,
                    status_override="not_executed_due_to_prior_round_failure",
                )
                results.append(_public_result(result))
                partition = partitions[str(entry.resolved["client_id"])]
                _write(
                    db_path,
                    "write_fedavg_client_round",
                    _client_round_row(
                        entry.resolved,
                        status="not_executed_due_to_prior_round_failure",
                        reason=reason,
                        scheduled=_as_bool(entry.resolved.get("scheduled_to_participate")),
                        participated=False,
                        succeeded=False,
                        sample_count=len(partition.indices),
                        partition_hash=partition_documents[partition.client_id]["partition_hash"],
                        parent_round=None,
                        parent_hash=None,
                        partition_metadata=partition_documents[partition.client_id],
                    ),
                )
            _write(
                db_path,
                "write_fedavg_round",
                {
                    "fl_run_id": fl_run_id,
                    "round_id": round_id,
                    "status": "failed",
                    "scheduled_count": scheduled_count,
                    "successful_count": 0,
                    "failed_count": 0,
                    "aggregation_method": None,
                    "aggregation_version": None,
                    "contributor_metadata_json": [],
                    "excluded_metadata_json": [],
                    "parent_state_hash": None,
                    "output_state_hash": None,
                    "started_at": round_started,
                    "completed_at": _utcnow(),
                    "status_reason": "not executed because a prior FedAvg round failed",
                    "metadata_json": {
                        "not_executed_count": len(round_entries),
                        "prior_failure": prior_failure,
                    },
                },
            )
            continue

        _write(
            db_path,
            "write_fedavg_round",
            {
                "fl_run_id": fl_run_id,
                "round_id": round_id,
                "status": "running",
                "scheduled_count": scheduled_count,
                "successful_count": 0,
                "failed_count": 0,
                "aggregation_method": AGGREGATION_METHOD,
                "aggregation_version": FEDAVG_AGGREGATION_VERSION,
                "contributor_metadata_json": [],
                "excluded_metadata_json": [],
                "parent_state_hash": parent_state.sha256,
                "output_state_hash": None,
                "started_at": round_started,
                "completed_at": None,
                "status_reason": None,
                "metadata_json": {},
            },
        )
        contributors: list[FedAvgContributor] = []
        client_records: dict[str, dict[str, Any]] = {}
        excluded: list[dict[str, Any]] = []
        for entry in round_entries:
            row = entry.resolved
            client_id = str(row["client_id"])
            debug_print(f"[Round {round_id}/{round_count}] Client {client_id} "
                  f"({len(client_records) + 1}/{len(round_entries)}) | "
                  f"{'starting' if _as_bool(row.get('scheduled_to_participate')) else 'not participating'}", flush=True)
            partition = partitions[client_id]
            partition_doc = partition_documents[client_id]
            partition_hash = str(partition_doc["partition_hash"])
            if not _as_bool(row.get("scheduled_to_participate")):
                reason = "client was not scheduled to participate in this round"
                result = execute_entry(
                    entry,
                    prepared=None,
                    failure_message=reason,
                    status_override="not_participating",
                )
                results.append(_public_result(result))
                client_records[client_id] = _client_round_row(
                    row,
                    status="not_participating",
                    reason=reason,
                    scheduled=False,
                    participated=False,
                    succeeded=False,
                    sample_count=len(partition.indices),
                    partition_hash=partition_hash,
                    parent_round=round_id - 1,
                    parent_hash=parent_state.sha256,
                    partition_metadata=partition_doc,
                )
                continue

            participant_position += 1
            if not debug_enabled():
                print(
                    f"\n[Round {round_id}/{round_count}] "
                    f"[{participant_position}/{scheduled_count}] {client_id} | training...",
                    flush=True,
                )
            client_started = time.perf_counter()
            restore_model_tensor_state(model, parent_state, verify=True)
            model_reset_s = time.perf_counter() - client_started
            prepared_row = {
                "_client_started_perf": client_started,
                "_model_reset_s": model_reset_s,
                "_fedavg_restored_parent": parent_state,
                "_prepared_dataset": prepared_dataset,
                "_prepared_model": model,
                "_prepared_benchmark_sample_info": benchmark_info,
                "_prepared_train_partition": {
                    "x_train": partition.x,
                    "y_train": partition.y,
                    "split_resolution": {
                        "strategy": partition_doc.get("effective_strategy"),
                        "requested_strategy": partition_doc.get("requested_strategy"),
                        "skew_axis": partition_doc.get("skew_axis"),
                        "axis_family": partition_doc.get("axis_family"),
                        "bucket_spec": partition_doc.get("bucket_spec"),
                        "bucket_distribution": partition_doc.get("bucket_histogram"),
                        "fallback_reason": partition_doc.get("fallback_reason"),
                    },
                    "partition_meta": partition_doc,
                    "client_partition_index": row.get("client_partition_index"),
                    "run_partition_seed": row.get("run_partition_seed"),
                    "fl_run_id": fl_run_id,
                },
                "_defer_terminal_results": True,
                "_fedavg_parent_state": parent_state,
                "_fedavg_parent_ref": parent_ref,
                "_fedavg_parent_round_id": round_id - 1,
                "_fedavg_lineage_base": lineage_base,
                # Earlier client-rounds remain provenance only. Their in-memory
                # states still contribute to aggregation; only final-round
                # client checkpoints are retained for reconstruction.
                "_retain_client_checkpoint": round_id == round_count,
                "compatibility_contract_hash": compatibility_hash,
                "parent_state_hash": parent_state.sha256,
                "training_universe_hash": training_universe_hash,
                "benchmark_cohort_hash": benchmark_hash,
            }
            result = execute_entry(entry, prepared=prepared_row)
            succeeded = str(result.get("status") or "").lower() == "success"
            final_state = result.pop("_final_tensor_state", None)
            verified = result.pop("_checkpoint_integrity_verified", None)
            if succeeded:
                if verified is False:
                    succeeded = False
                    result["status"] = "failed"
                    result["error_message"] = "client checkpoint integrity was not verified"
                if final_state is None:
                    final_state = capture_model_tensor_state(model)
                if not isinstance(final_state, TensorState):
                    succeeded = False
                    result["status"] = "failed"
                    result["error_message"] = "client did not produce a canonical tensor state"
            print_client_completion(
                {**row, **result.pop("_terminal_runtime_config", {})}, partition_doc,
                result.pop("_terminal_metrics", {}), status=result.get("status"),
                error=result.get("error_message") if not succeeded else None,
                position=participant_position, participants=scheduled_count,
            )
            if result.pop("_terminal_weights_saved", False) and succeeded:
                print(f"Round {round_id} Client {client_id} weights have been saved.", flush=True)
            results.append(_public_result(result))
            if succeeded:
                contributors.append(
                    FedAvgContributor(
                        client_id=client_id,
                        service_id=str(row["service_id"]),
                        sample_count=len(partition.indices),
                        state=final_state,
                        compatibility_contract_hash=compatibility_hash,
                        parent_state_hash=parent_state.sha256,
                    )
                )
                client_records[client_id] = _client_round_row(
                    row,
                    status="success",
                    reason=None,
                    scheduled=True,
                    participated=True,
                    succeeded=True,
                    sample_count=len(partition.indices),
                    partition_hash=partition_hash,
                    parent_round=round_id - 1,
                    parent_hash=parent_state.sha256,
                    final_hash=final_state.sha256,
                    partition_metadata=partition_doc,
                )
            else:
                any_client_failures = True
                reason = str(result.get("error_message") or "client-round execution failed")
                excluded.append({"client_id": client_id, "service_id": row["service_id"], "reason": reason})
                client_records[client_id] = _client_round_row(
                    row,
                    status="failed",
                    reason=reason,
                    scheduled=True,
                    participated=True,
                    succeeded=False,
                    sample_count=len(partition.indices),
                    partition_hash=partition_hash,
                    parent_round=round_id - 1,
                    parent_hash=parent_state.sha256,
                    partition_metadata=partition_doc,
                )

        # Persist client outcomes before aggregation so the database reflects
        # the last completed client even while global-state encoding is busy.
        for record in client_records.values():
            _write(db_path, "write_fedavg_client_round", record)

        if len(contributors) < 2:
            prior_failure = (
                f"FedAvg round {round_id} has {len(contributors)} successful contributors; two are required"
            )
            print(f"[Round {round_id}/{round_count}] FAILED | {prior_failure}", flush=True)
            _write_round_failure(
                db_path, fl_run_id, round_id, round_started, scheduled_count,
                contributors, excluded, parent_state.sha256, prior_failure,
            )
            continue

        aggregation_started = time.perf_counter()
        debug_print(
            f"[FedAvg] aggregation starts | fl_run_id={fl_run_id} "
            f"| round={round_id} | contributors={len(contributors)} "
            f"| compression={compression.as_dict()}",
            flush=True,
        )
        _write(
            db_path,
            "write_fedavg_round",
            {
                "fl_run_id": fl_run_id,
                "round_id": round_id,
                "status": "running",
                "scheduled_count": scheduled_count,
                "successful_count": len(contributors),
                "failed_count": scheduled_count - len(contributors),
                "aggregation_method": AGGREGATION_METHOD,
                "aggregation_version": FEDAVG_AGGREGATION_VERSION,
                "contributor_metadata_json": [
                    {
                        "client_id": contributor.client_id,
                        "service_id": contributor.service_id,
                        "sample_count": contributor.sample_count,
                    }
                    for contributor in contributors
                ],
                "excluded_metadata_json": excluded,
                "parent_state_hash": parent_state.sha256,
                "output_state_hash": None,
                "started_at": round_started,
                "completed_at": None,
                "status_reason": None,
                "metadata_json": {"phase": "aggregating"},
            },
        )
        try:
            aggregation = aggregate_fedavg_states(
                parent_state, contributors, compatibility_contract_hash=compatibility_hash
            )
            global_ref = store.put_state(aggregation.state, compression=compression)
            read_back = deserialize_tensor_state(
                store.read(global_ref), framework=aggregation.state.framework
            )
            if read_back.sha256 != aggregation.state.sha256:
                raise ValueError("aggregated global state failed CAS read-back verification")
            restore_model_tensor_state(model, read_back, verify=True)
        except Exception as exc:  # noqa: BLE001
            prior_failure = f"FedAvg round {round_id} aggregation failed: {exc}"
            print(f"[Round {round_id}/{round_count}] FAILED | {prior_failure}", flush=True)
            _write_round_failure(
                db_path, fl_run_id, round_id, round_started, scheduled_count,
                contributors, excluded, parent_state.sha256, prior_failure,
            )
            continue
        aggregation_elapsed_s = time.perf_counter() - aggregation_started
        debug_print(
            f"[FedAvg] aggregation ends | fl_run_id={fl_run_id} "
            f"| round={round_id} | elapsed_s={aggregation_elapsed_s:.3f} "
            f"| output_state_hash={read_back.sha256}",
            flush=True,
        )

        _write(
            db_path,
            "write_fedavg_round",
            {
                "fl_run_id": fl_run_id,
                "round_id": round_id,
                "status": "running",
                "scheduled_count": scheduled_count,
                "successful_count": len(contributors),
                "failed_count": scheduled_count - len(contributors),
                "aggregation_method": AGGREGATION_METHOD,
                "aggregation_version": FEDAVG_AGGREGATION_VERSION,
                "contributor_metadata_json": aggregation.provenance["contributors"],
                "excluded_metadata_json": excluded,
                "parent_state_hash": parent_state.sha256,
                "output_state_hash": read_back.sha256,
                "started_at": round_started,
                "completed_at": None,
                "status_reason": None,
                "metadata_json": {
                    "phase": "global_evaluation",
                    "aggregation_duration_s": aggregation_elapsed_s,
                },
            },
        )
        try:
            global_metrics, global_eval = _global_eval_metrics(
                first,
                model,
                x_test,
                y_test,
                task_family=task_family,
                global_round=round_id,
                state_hash=read_back.sha256,
            )
        except Exception as exc:  # noqa: BLE001
            prior_failure = f"FedAvg round {round_id} global evaluation failed: {exc}"
            print(f"[Round {round_id}/{round_count}] FAILED | {prior_failure}", flush=True)
            _write_round_failure(
                db_path, fl_run_id, round_id, round_started, scheduled_count,
                contributors, excluded, parent_state.sha256, prior_failure,
            )
            print(
                f"[FedAvg global eval] FAILED | fl_run_id={fl_run_id} "
                f"| global_model=G{round_id} | round={round_id} | error={exc}",
                flush=True,
            )
            continue
        secondary_summary = ""
        if global_eval.get("auxiliary_metric_name"):
            secondary_summary = (
                f" | {global_eval['auxiliary_metric_name']}="
                f"{global_eval['auxiliary_metric']}"
            )
        debug_print(
            f"[FedAvg global eval] fl_run_id={fl_run_id} "
            f"| global_model=G{round_id} | round={round_id} "
            f"| benchmark_samples={global_eval['benchmark_samples']} "
            f"| loss={global_eval['loss']} "
            f"| {global_eval['primary_metric_name']}={global_eval['primary_metric']}"
            f"{secondary_summary} | metric_score={global_eval['metric_score']} "
            f"| elapsed_s={global_eval['evaluation_runtime_s']:.3f}",
            flush=True,
        )

        if debug_enabled():
            print_results(f"[Round {round_id}/{round_count} results] contributors={len(contributors)} "
                          f"failed={scheduled_count - len(contributors)} "
                          f"not_participating={len(round_entries) - scheduled_count}", global_metrics)
        aggregation_provenance = dict(aggregation.provenance)
        aggregation_provenance.update(
            {
                "fl_run_id": fl_run_id,
                "round_id": round_id,
                "compatibility_contract_hash": compatibility_hash,
                "included_tensor_categories": ["floating", "complex"],
                "excluded_tensor_categories": ["integer", "boolean", "explicit_non_aggregatable"],
            }
        )
        aggregation_artifact = write_json_artifact(
            provenance_root, fl_run_id, f"fedavg_round_{round_id:03d}_aggregation",
            aggregation_provenance,
        )
        weights = {
            item["client_id"]: float(item["normalized_weight"])
            for item in aggregation.provenance["contributors"]
        }
        for client_id, record in client_records.items():
            if client_id in weights:
                record["aggregation_weight"] = weights[client_id]
            _write(db_path, "write_fedavg_client_round", record)
        global_state_row = {
            "fl_run_id": fl_run_id,
            "global_round": round_id,
            "state_hash": read_back.sha256,
            "cas_ref": str(global_ref.manifest_path),
            "parameter_schema_hash": schema_hash,
            "parent_global_round": round_id - 1,
            "parent_state_hash": parent_state.sha256,
            "storage_format": "full_immutable_tensor_state_v1",
            "compression_json": global_ref.compression.as_dict(),
            "logical_bytes": global_ref.logical_size_bytes,
            "physical_bytes": global_ref.physical_size_bytes,
            "verified_at": _utcnow(),
            "metadata_json": {
                "role": f"G{round_id}",
                "is_service": False,
                "aggregation_provenance_ref": str(aggregation_artifact["artifact_uri"]),
                "global_eval": global_eval,
            },
        }
        completed_round_row = {
                "fl_run_id": fl_run_id,
                "round_id": round_id,
                "status": "incomplete" if excluded else "complete",
                "scheduled_count": scheduled_count,
                "successful_count": len(contributors),
                "failed_count": scheduled_count - len(contributors),
                "aggregation_method": AGGREGATION_METHOD,
                "aggregation_version": FEDAVG_AGGREGATION_VERSION,
                "contributor_metadata_json": aggregation.provenance["contributors"],
                "excluded_metadata_json": excluded,
                "parent_state_hash": parent_state.sha256,
                "output_state_hash": read_back.sha256,
                "started_at": round_started,
                "completed_at": _utcnow(),
                "status_reason": None,
                "metadata_json": {
                    "aggregation_provenance_ref": str(aggregation_artifact["artifact_uri"]),
                    "aggregation_duration_s": aggregation_elapsed_s,
                    "global_eval": global_eval,
                    "included_tensor_categories": aggregation_provenance["included_tensor_categories"],
                    "excluded_tensor_categories": aggregation_provenance["excluded_tensor_categories"],
                },
        }
        try:
            _write_completed_global_round(
                db_path,
                global_state_row=global_state_row,
                round_row=completed_round_row,
                metrics=global_metrics,
            )
        except Exception as exc:  # noqa: BLE001
            prior_failure = f"FedAvg round {round_id} persistence failed: {exc}"
            print(f"[Round {round_id}/{round_count}] FAILED | {prior_failure}", flush=True)
            _write_round_failure(
                db_path, fl_run_id, round_id, round_started, scheduled_count,
                contributors, excluded, parent_state.sha256, prior_failure,
            )
            continue
        parent_state = read_back
        parent_ref = global_ref
        completed_rounds += 1
        print("\nAggregation complete", flush=True)
        print(f"Global model: {compact_metrics(global_metrics)}", flush=True)
        print(f"Round duration: {time.perf_counter() - round_timer:.1f}s", flush=True)
        print("=====================", flush=True)

    status = "failed" if prior_failure else ("incomplete" if any_client_failures else "complete")
    run_row.update(
        status=status,
        status_reason=prior_failure,
        completed_at=_utcnow(),
        metadata_json={
            "parameter_schema_hash": schema_hash,
            **source_pool_summary,
            "completed_rounds": completed_rounds,
            "final_global_state_hash": parent_state.sha256 if completed_rounds == round_count else None,
            "original_fl_run_id": first.get("original_fl_run_id"),
            "partition_generator_version": first.get("partition_generator_version"),
            "heterogeneity_intensity": first.get("heterogeneity_intensity"),
        },
    )
    _write(db_path, "write_fedavg_run", run_row)
    print(f"[FedAvg complete] run={fl_run_id} status={status} "
          f"rounds={completed_rounds}/{round_count} client_results={len(results)}", flush=True)
    return results, {
        "fl_run_id": fl_run_id,
        "lineage_id": lineage_id,
        **source_pool_summary,
        "partition_generator_version": first.get("partition_generator_version"),
        "heterogeneity_intensity": first.get("heterogeneity_intensity"),
        "status": status,
        "client_count": client_count,
        "round_count": round_count,
        "completed_rounds": completed_rounds,
        "final_global_state_hash": parent_state.sha256 if completed_rounds == round_count else None,
        "error": prior_failure,
    }


def _write_round_failure(
    db_path: str,
    fl_run_id: str,
    round_id: int,
    started_at: str,
    scheduled_count: int,
    contributors: Sequence[FedAvgContributor],
    excluded: Sequence[Mapping[str, Any]],
    parent_hash: str,
    reason: str,
) -> None:
    _write(
        db_path,
        "write_fedavg_round",
        {
            "fl_run_id": fl_run_id,
            "round_id": round_id,
            "status": "failed",
            "scheduled_count": scheduled_count,
            "successful_count": len(contributors),
            "failed_count": scheduled_count - len(contributors),
            "aggregation_method": AGGREGATION_METHOD,
            "aggregation_version": FEDAVG_AGGREGATION_VERSION,
            "contributor_metadata_json": [
                {
                    "client_id": item.client_id,
                    "service_id": item.service_id,
                    "sample_count": item.sample_count,
                    "client_state_hash": item.state.sha256,
                }
                for item in contributors
            ],
            "excluded_metadata_json": list(excluded),
            "parent_state_hash": parent_hash,
            "output_state_hash": None,
            "started_at": started_at,
            "completed_at": _utcnow(),
            "status_reason": reason,
            "metadata_json": {"quorum": 2},
        },
    )
