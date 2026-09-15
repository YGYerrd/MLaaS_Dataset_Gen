"""CLI orchestration for selected compatible checkpoint compositions."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .catalogue import load_services_from_db
from .fedavg_validation import (
    FedAvgAggregationExecutionError,
    FedAvgCheckpointError,
    FedAvgInferenceError,
    FedAvgProvenanceError,
    execute_selected_fedavg_composition,
)
from .pool import request_object, build_composition_instance
from .protocol import objective_version_for_request, request_schema_version, REQUEST_SCHEMA_VERSION
from .sampler import evaluate_provisional_composition
from .validation import (
    CohortIsolationError,
    CohortPlan,
    build_cohort_plan,
    measure_model_outputs,
    measured_scores,
    supported_composition_reason,
)


_RETIRED_RUNTIME_MESSAGE = (
    "prediction artifacts, regeneration, stability campaigns, and parallel member "
    "execution are retired for validation; selected compositions must be materialized "
    "from exact compatible client checkpoints"
)


def validate_from_files(
    *,
    db_path: str | Path,
    requests: pd.DataFrame,
    evaluation_dir: str | Path,
    request_id: str,
    output_dir: str | Path,
    cohort_file: str | Path,
    audit_seed: int,
    audit_samples: int,
    min_audit_samples: int,
    selection_source: str = "evaluator",
    search_method: str | None = None,
    candidate_id: str | None = None,
    predictions_file: str | Path | None = None,
    max_parallelism: int | None = None,
    backfill_missing: bool = False,
    stability_seeds: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize and measure one already-selected compatible composition."""
    if (
        predictions_file is not None
        or max_parallelism is not None
        or bool(backfill_missing)
        or int(stability_seeds) != 0
    ):
        raise ValueError(_RETIRED_RUNTIME_MESSAGE)

    request_rows = requests[requests["request_id"].astype(str) == str(request_id)]
    if len(request_rows) != 1:
        raise ValueError(f"request_id '{request_id}' was not found exactly once")
    request = request_rows.iloc[0].to_dict()
    candidate = _read_selected_composition(
        evaluation_dir,
        request_id,
        selection_source=selection_source,
        search_method=search_method,
        candidate_id=candidate_id,
    )
    services = load_services_from_db(db_path)
    service_ids = candidate["service_ids"]
    if request_schema_version(request) in {REQUEST_SCHEMA_VERSION, "4.0.0", "5.0.0"}:
        instance = build_composition_instance(services, request)
        if len(service_ids) != instance.composition_size or not set(service_ids) <= set(instance.C_R.service_id):
            raise FedAvgProvenanceError("selected services do not match the frozen request's eligible cohort and composition size")
    selected = services[services["service_id"].astype(str).isin(service_ids)].copy()
    if len(selected) != len(service_ids):
        missing = sorted(set(service_ids) - set(selected["service_id"].astype(str)))
        raise ValueError(f"selected composition services were not found: {missing}")
    selected = selected.set_index("service_id").loc[service_ids].reset_index()
    task_values = selected["task_family"].astype(str).str.lower().unique()
    if len(task_values) != 1:
        raise ValueError("selected composition members do not share one task family")
    task_family = str(task_values[0])
    cohort = _read_cohort(
        cohort_file,
        seed=audit_seed,
        audit_samples=audit_samples,
        minimum=min_audit_samples,
        task_family=task_family,
    )
    heldout_verified, heldout_reason = _heldout_provenance(cohort_file)

    reason = supported_composition_reason(
        selected, cohort_contract_hash=cohort.contract_hash
    )
    record = _empty_record(candidate, request_id, service_ids, reason)
    if reason is None:
        try:
            execution = execute_selected_fedavg_composition(
                db_path=db_path, service_ids=service_ids, plan=cohort
            )
            if execution.task_family != task_family:
                raise FedAvgProvenanceError(
                    "aggregated execution task does not match selected catalogue services"
                )
            if execution.label_hash != cohort.label_hash:
                raise CohortIsolationError(
                    "aggregated model label contract does not match the audit cohort"
                )
            labels = _read_cohort_labels(cohort_file, cohort, task_family=task_family)
            actual = measure_model_outputs(
                execution.outputs,
                cohort,
                labels,
                task_family=task_family,
                inference_latency_s=execution.inference_latency_s,
            )
            qos = request_object(request, "qos_requirements")
            surrogate_scores = evaluate_provisional_composition(
                selected,
                qos_requirements=qos,
                objective_version=objective_version_for_request(request),
            )
            observed = measured_scores(
                selected,
                request,
                actual_quality_score=actual["quality_score"],
                parallel_latency_s=execution.inference_latency_s,
            )
            provenance = dict(execution.aggregation_provenance)
            recorded_utility = candidate.get("recorded_surrogate_utility")
            recorded_quality = candidate.get("recorded_surrogate_quality")
            record.update(
                {
                    "validation_status": "completed",
                    "status_reason": None,
                    "aggregation_method": provenance["aggregation_method"],
                    "aggregation_version": provenance["aggregation_version"],
                    "fl_run_id": provenance["fl_run_id"],
                    "cohort_id": provenance.get("cohort_id"),
                    "parent_checkpoint_id": provenance.get("parent_checkpoint_id"),
                    "local_example_exposures": provenance.get("local_example_exposures"),
                    "unique_training_examples": provenance.get("unique_training_examples"),
                    "round_id": provenance["round_id"],
                    "parent_global_round": provenance["parent_global_round"],
                    "parent_state_hash": provenance["parent_state_hash"],
                    "compatibility_contract_hash": provenance[
                        "compatibility_contract_hash"
                    ],
                    "parameter_schema_hash": provenance["parameter_schema_hash"],
                    "aggregated_state_hash": provenance["global_state_hash"],
                    "selected_total_samples": provenance["total_samples"],
                    "contributor_provenance_json": json.dumps(
                        provenance["contributors"], sort_keys=True
                    ),
                    "aggregation_provenance_json": json.dumps(
                        provenance, sort_keys=True
                    ),
                    "aggregation_wall_s": execution.aggregation_wall_s,
                    "inference_latency_s": execution.inference_latency_s,
                    "actual_accuracy": actual.get("accuracy"),
                    "actual_f1_macro": actual.get("f1_macro"),
                    "actual_rmse": actual.get("rmse"),
                    "actual_mae": actual.get("mae"),
                    "task_metrics_json": json.dumps(actual, sort_keys=True, allow_nan=False),
                    "measured_quality_score": actual["quality_score"],
                    "measured_utility": observed["measured_utility"],
                    "utility_prediction_error": (
                        None
                        if recorded_utility is None
                        else observed["measured_utility"] - float(recorded_utility)
                    ),
                    "quality_prediction_error": (
                        None
                        if recorded_quality is None
                        else actual["quality_score"] - float(recorded_quality)
                    ),
                    "actual_primary_metric_name": (
                        actual.get("primary_metric_name", "accuracy" if task_family == "classification" else "rmse")
                    ),
                    "actual_secondary_metric_name": (
                        actual.get("secondary_metric_name", "f1_macro" if task_family == "classification" else "mae")
                    ),
                    "qos_violations": json.dumps(
                        _qos_violations(observed, qos), sort_keys=True
                    ),
                    "data_parameter_evidence_json": surrogate_scores.get(
                        "data_parameter_evidence_json"
                    ),
                    "data_parameter_indeterminate_fraction": surrogate_scores.get(
                        "data_parameter_indeterminate_fraction"
                    ),
                }
            )
            if request_schema_version(request) == "5.0.0":
                from mlaas_data_generator.task_capabilities import resolve_capabilities
                checks = dict.fromkeys(("checkpoint_present", "checkpoint_integrity", "parent", "parameter_schema",
                                        "target_semantics", "evaluation_protocol", "audit_support"), True)
                record["task_capabilities_json"] = json.dumps({row["service_id"]: resolve_capabilities(row,
                    stage="composition", checks=checks) for row in selected.to_dict("records")}, sort_keys=True)
        except FedAvgProvenanceError as error:
            _record_failure(record, "unsupported", error)
        except CohortIsolationError as error:
            _record_failure(record, "cohort_isolation_failed", error)
        except FedAvgCheckpointError as error:
            _record_failure(record, "checkpoint_failed", error)
        except FedAvgAggregationExecutionError as error:
            _record_failure(record, "aggregation_failed", error)
        except FedAvgInferenceError as error:
            _record_failure(record, "inference_failed", error)
        except Exception as error:  # noqa: BLE001 - retain selected identity/evidence
            _record_failure(record, "inference_failed", error)

    frame = pd.DataFrame([record])
    summary = _selected_validation_summary(record)
    if record.get("task_metrics_json"):
        summary["task_metrics"] = json.loads(record["task_metrics_json"])
    if record.get("task_capabilities_json"):
        summary["task_capabilities"] = json.loads(record["task_capabilities_json"])
    summary.update(
        {
            "request_id": str(request_id),
            "cohort_id": cohort.cohort_id,
            "cohort_hash": cohort.audit_hash,
            "task_family": task_family,
            "validation_semantics": "selected_sample_weighted_fedavg_parameter_aggregation_v1",
            "heldout_provenance_verified": heldout_verified,
            "scientific_validity": (
                "empirical_validation"
                if heldout_verified
                else "mechanical_validation_only"
            ),
            "heldout_provenance_reason": heldout_reason,
        }
    )
    _write_outputs(db_path, output_dir, cohort, frame, summary)
    return frame, summary


def _empty_record(
    candidate: Mapping[str, Any],
    request_id: str,
    service_ids: list[str],
    reason: str | None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "request_id": str(request_id),
        "service_ids": json.dumps(service_ids),
        "selected_flag": bool(candidate["selected_flag"]),
        "selection_source": candidate["selection_source"],
        "search_method": candidate.get("search_method"),
        "recorded_surrogate_utility": candidate.get("recorded_surrogate_utility"),
        "recorded_surrogate_quality": candidate.get("recorded_surrogate_quality"),
        "regenerated_surrogate_utility": None,
        "validation_status": "unsupported",
        "status_reason": reason,
        "reason_code": reason.split(":", 1)[0] if reason and ":" in reason else "validation_not_implemented" if reason else None,
        "failure_stage": "composition_preflight" if reason else None,
        "measured_utility": None,
        "measured_quality_score": None,
        "utility_prediction_error": None,
        "quality_prediction_error": None,
        "actual_primary_metric_name": None,
        "actual_secondary_metric_name": None,
        "actual_accuracy": None,
        "actual_f1_macro": None,
        "actual_rmse": None,
        "actual_mae": None,
        "inference_latency_s": None,
        "aggregation_wall_s": None,
        "aggregation_method": None,
        "aggregation_version": None,
        "fl_run_id": None,
        "cohort_id": None,
        "parent_checkpoint_id": None,
        "local_example_exposures": None,
        "unique_training_examples": None,
        "round_id": None,
        "parent_global_round": None,
        "parent_state_hash": None,
        "compatibility_contract_hash": None,
        "parameter_schema_hash": None,
        "aggregated_state_hash": None,
        "selected_total_samples": None,
        "contributor_provenance_json": None,
        "aggregation_provenance_json": None,
        "qos_violations": None,
        "data_parameter_evidence_json": None,
        "data_parameter_indeterminate_fraction": None,
        # Deprecated prediction-ensemble columns remain null for DB compatibility.
        "ensemble_accuracy": None,
        "ensemble_f1_macro": None,
        "ensemble_rmse": None,
        "ensemble_mae": None,
        "parallel_latency_s": None,
        "serial_latency_s": None,
        "prediction_wall_s": None,
    }


def _record_failure(record: dict[str, Any], status: str, error: Exception) -> None:
    record["validation_status"] = status
    record["status_reason"] = f"{type(error).__name__}: {error}"
    cause = error
    while cause.__cause__ is not None:
        cause = cause.__cause__
    record["reason_code"] = getattr(cause, "code", getattr(error, "code", status))
    record["failure_stage"] = getattr(error, "stage", "composition")


def _read_selected_composition(
    evaluation_dir: str | Path,
    request_id: str,
    *,
    selection_source: str,
    search_method: str | None,
    candidate_id: str | None,
) -> dict[str, Any]:
    root = Path(evaluation_dir)
    source = str(selection_source or "evaluator").strip().lower()
    if source not in {"evaluator", "search"}:
        raise ValueError("selection_source must be 'evaluator' or 'search'")
    if candidate_id is not None and source == "search":
        raise ValueError("candidate_id cannot be combined with selection_source='search'")
    if source == "evaluator" and search_method:
        raise ValueError("search_method requires selection_source='search'")
    compositions = root / "compositions.csv"
    if candidate_id is not None:
        if not compositions.exists():
            raise ValueError(f"No compositions.csv exists in {root}")
        frame = pd.read_csv(compositions)
        matches = frame[
            frame["request_id"].astype(str).eq(str(request_id))
            & frame["candidate_id"].astype(str).eq(str(candidate_id))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"candidate_id '{candidate_id}' was not found exactly once for request '{request_id}'"
            )
        row = matches.iloc[0].to_dict()
        if not _as_bool(row.get("selected_flag")):
            raise ValueError(f"candidate_id '{candidate_id}' is not marked as selected")
        return _selected_row(row, selection_source="evaluator", search_method=None)
    if source == "evaluator":
        if not compositions.exists():
            raise ValueError(f"No compositions.csv exists in {root}")
        frame = pd.read_csv(compositions)
        matches = frame[
            frame["request_id"].astype(str).eq(str(request_id))
            & frame["selected_flag"].map(_as_bool)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one evaluator-selected composition for request '{request_id}', found {len(matches)}"
            )
        return _selected_row(
            matches.iloc[0].to_dict(),
            selection_source="evaluator",
            search_method=None,
        )
    if not search_method:
        raise ValueError("selection_source='search' requires search_method")
    results = root / "search_results.csv"
    if not results.exists():
        raise ValueError(f"No search_results.csv exists in {root}")
    frame = pd.read_csv(results)
    matches = frame[
        frame["request_id"].astype(str).eq(str(request_id))
        & frame["method"].astype(str).str.lower().eq(str(search_method).lower())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '{search_method}' search selection for request '{request_id}', found {len(matches)}"
        )
    return _selected_row(
        matches.iloc[0].to_dict(),
        selection_source="search",
        search_method=str(search_method).lower(),
    )


def _selected_row(
    row: Mapping[str, Any], *, selection_source: str, search_method: str | None
) -> dict[str, Any]:
    ids = list(map(str, json.loads(str(row["service_ids"]))))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("selected composition must contain unique service IDs")
    candidate_id = row.get("candidate_id")
    if candidate_id is None or (
        isinstance(candidate_id, float) and np.isnan(candidate_id)
    ):
        candidate_id = f"{row.get('request_id')}_{search_method or 'selected'}"
    return {
        "candidate_id": str(candidate_id),
        "service_ids": ids,
        "selected_flag": True,
        "selection_source": selection_source,
        "search_method": search_method,
        "recorded_surrogate_utility": _finite(row.get("utility_score")),
        "recorded_surrogate_quality": _finite(
            row.get("mean_accuracy", row.get("accuracy_compatibility"))
        ),
    }


def _read_cohort(
    path: str | Path,
    *,
    seed: int,
    audit_samples: int,
    minimum: int,
    task_family: str,
) -> CohortPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") in {"reserved_evaluation_v1", "reserved_evaluation_v2"}:
        from .validation import CohortPlan, assert_cohort_isolation, cohort_label_hash
        from mlaas_data_generator.provenance import sha256_json
        digest = sha256_json({k: v for k, v in payload.items() if k != "reservation_hash"})
        if digest != payload.get("reservation_hash"):
            raise CohortIsolationError("frozen cohort reservation hash mismatch")
        if payload["task_family"] != task_family:
            raise CohortIsolationError("frozen audit task does not match the composition")
        train, metric, audit = (tuple(payload[k]) for k in ("training_ids", "catalogue_metric_ids", "audit_ids"))
        assert_cohort_isolation(train, metric, audit)
        if len(audit) < minimum:
            raise CohortIsolationError("frozen audit cohort is below minimum_audit_samples")
        if set(metric) | set(audit) != set(payload["metric_source_ids"]):
            raise CohortIsolationError("frozen evaluation partitions do not cover their source")
        return CohortPlan("audit_" + digest[:24], sha256_json(payload["contract"]), int(payload["seed"]),
                          train, metric, audit, cohort_label_hash(payload["metric_source_ids"], payload["metric_source_labels"]))
    required = {"training_ids", "metric_source_ids", "metric_source_labels", "contract"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"cohort file is missing: {sorted(missing)}")
    plan = build_cohort_plan(
        training_ids=payload["training_ids"],
        metric_source_ids=payload["metric_source_ids"],
        metric_source_labels=payload["metric_source_labels"],
        contract=payload["contract"],
        seed=seed,
        audit_samples=audit_samples,
        minimum_audit_samples=minimum,
        task_family=task_family,
    )
    labels_by_id = dict(
        zip(map(str, payload["metric_source_ids"]), payload["metric_source_labels"])
    )
    audit_labels = [labels_by_id[item] for item in plan.audit_ids]
    supplied = payload.get("audit_labels")
    if supplied and list(supplied) != audit_labels:
        raise ValueError("cohort file audit_labels do not match the immutable cohort plan")
    return plan


def _read_cohort_labels(
    path: str | Path, plan: CohortPlan, *, task_family: str
) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = dict(
        zip(map(str, payload["metric_source_ids"]), payload["metric_source_labels"])
    )
    if payload.get("version") == "reserved_evaluation_v2":
        return [labels[item] for item in plan.audit_ids]
    dtype = float if str(task_family).lower() == "regression" else int
    return np.asarray([labels[item] for item in plan.audit_ids], dtype=dtype)


def _heldout_provenance(path: str | Path) -> tuple[bool, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") in {"reserved_evaluation_v1", "reserved_evaluation_v2"}:
        return False, "audit reserved before this generation pipeline; historical evaluation exposure is unknown"
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        return False, "cohort contract is missing"
    reserved = contract.get("heldout_reserved_before_catalogue_metrics") is True
    reference = contract.get("heldout_provenance_ref")
    if reserved and isinstance(reference, str) and reference.strip():
        return True, "cohort declares pre-catalogue reservation with provenance"
    return (
        False,
        "cohort isolation is mechanically verified, but pre-catalogue holdout reservation is not proven",
    )


def _qos_violations(
    scores: Mapping[str, Any], requirements: Mapping[str, Any]
) -> dict[str, float]:
    actuals = {
        "min_accuracy": scores.get("mean_accuracy"),
        "max_latency": scores.get("total_latency"),
        "max_compute_time": scores.get("mean_compute_time"),
        "min_reliability": scores.get("mean_reliability"),
        "min_explainability": scores.get("mean_explainability"),
        "max_resource_cost": scores.get("mean_resource_cost"),
        "min_resource_cost_score": scores.get("mean_resource_cost_score"),
    }
    violations: dict[str, float] = {}
    for key, target in requirements.items():
        actual = actuals.get(key)
        if actual is None:
            continue
        deviation = (
            float(actual) - float(target)
            if key.startswith("max_")
            else float(target) - float(actual)
        )
        violations[key] = float(max(0.0, deviation))
    return violations


def _write_outputs(
    db_path: str | Path,
    output_dir: str | Path,
    cohort: CohortPlan,
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "validation_candidates.csv", index=False)
    (root / "validation_summary.json").write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    row = frame.iloc[0].to_dict()
    aggregation = row.get("aggregation_provenance_json")
    (root / "aggregation_provenance.json").write_text(
        json.dumps(
            {
                "candidate_id": row.get("candidate_id"),
                "validation_status": row.get("validation_status"),
                "aggregation": json.loads(aggregation) if aggregation else None,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    run_id = str(uuid.uuid4())
    from mlaas_data_generator.storage.writer import make_writer

    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    try:
        writer.write_composition_validation_run(
            {
                "validation_run_id": run_id,
                "request_id": summary["request_id"],
                "cohort_id": cohort.cohort_id,
                "cohort_manifest_json": cohort.as_dict(),
                "inputs_json": {"candidate_count": len(frame)},
                "provenance_json": {
                    "strict_held_out": True,
                    "validation_semantics": summary["validation_semantics"],
                    "heldout_provenance_verified": summary[
                        "heldout_provenance_verified"
                    ],
                },
                "summary_json": dict(summary),
            }
        )
        for value in frame.to_dict(orient="records"):
            persisted = {
                "validation_run_id": run_id,
                "candidate_id": value["candidate_id"],
                "service_ids_json": value["service_ids"],
                "validation_status": value["validation_status"],
                "status_reason": value.get("status_reason"),
                "selection_source": value.get("selection_source"),
                "search_method": value.get("search_method"),
                "recorded_surrogate_utility": value.get(
                    "recorded_surrogate_utility"
                ),
                "recorded_surrogate_quality": value.get(
                    "recorded_surrogate_quality"
                ),
                "regenerated_surrogate_utility": None,
                "measured_utility": value.get("measured_utility"),
                "measured_quality_score": value.get("measured_quality_score"),
                "utility_prediction_error": value.get("utility_prediction_error"),
                "quality_prediction_error": value.get("quality_prediction_error"),
                "ensemble_accuracy": None,
                "ensemble_f1_macro": None,
                "ensemble_rmse": None,
                "ensemble_mae": None,
                "actual_primary_metric_name": value.get(
                    "actual_primary_metric_name"
                ),
                "actual_secondary_metric_name": value.get(
                    "actual_secondary_metric_name"
                ),
                "parallel_latency_s": None,
                "serial_latency_s": None,
                "prediction_wall_s": None,
                "qos_violations_json": value.get("qos_violations"),
                "selected_flag": bool(value.get("selected_flag", False)),
            }
            for field in _FEDAVG_PERSISTED_FIELDS:
                persisted[field] = value.get(field)
            writer.write_composition_validation_candidate(persisted)
        writer.finish()
    except Exception:
        writer.abort()
        raise


_FEDAVG_PERSISTED_FIELDS = (
    "cohort_id",
    "parent_checkpoint_id",
    "local_example_exposures",
    "unique_training_examples",
    "aggregation_method",
    "aggregation_version",
    "fl_run_id",
    "round_id",
    "parent_global_round",
    "parent_state_hash",
    "compatibility_contract_hash",
    "parameter_schema_hash",
    "aggregated_state_hash",
    "selected_total_samples",
    "contributor_provenance_json",
    "aggregation_provenance_json",
    "aggregation_wall_s",
    "inference_latency_s",
    "actual_accuracy",
    "actual_f1_macro",
    "actual_rmse",
    "actual_mae",
)


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _selected_validation_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_count": 1,
        "completed_candidate_count": int(
            record.get("validation_status") == "completed"
        ),
        "validation_status": record.get("validation_status"),
        "status_reason": record.get("status_reason"),
        "reason_code": record.get("reason_code"),
        "failure_stage": record.get("failure_stage"),
        "candidate_id": record.get("candidate_id"),
        "service_ids": json.loads(str(record.get("service_ids") or "[]")),
        "selection_source": record.get("selection_source"),
        "search_method": record.get("search_method"),
        "predicted_surrogate_utility": record.get("recorded_surrogate_utility"),
        "measured_utility": record.get("measured_utility"),
        "utility_prediction_error": record.get("utility_prediction_error"),
        "predicted_surrogate_quality": record.get("recorded_surrogate_quality"),
        "measured_quality_score": record.get("measured_quality_score"),
        "quality_prediction_error": record.get("quality_prediction_error"),
        "actual_primary_metric_name": record.get("actual_primary_metric_name"),
        "actual_secondary_metric_name": record.get("actual_secondary_metric_name"),
        "actual_accuracy": record.get("actual_accuracy"),
        "actual_f1_macro": record.get("actual_f1_macro"),
        "actual_rmse": record.get("actual_rmse"),
        "actual_mae": record.get("actual_mae"),
        "aggregation_method": record.get("aggregation_method"),
        "aggregation_version": record.get("aggregation_version"),
        "fl_run_id": record.get("fl_run_id"),
        "cohort_id": record.get("cohort_id"),
        "parent_checkpoint_id": record.get("parent_checkpoint_id"),
        "local_example_exposures": record.get("local_example_exposures"),
        "unique_training_examples": record.get("unique_training_examples"),
        "round_id": record.get("round_id"),
        "parent_global_round": record.get("parent_global_round"),
        "parent_state_hash": record.get("parent_state_hash"),
        "aggregated_state_hash": record.get("aggregated_state_hash"),
        "selected_total_samples": record.get("selected_total_samples"),
        "aggregation_wall_s": record.get("aggregation_wall_s"),
        "inference_latency_s": record.get("inference_latency_s"),
        "qos_violations": record.get("qos_violations"),
    }


__all__ = ["validate_from_files"]
