from __future__ import annotations

from ..clients.legacy import (
    _canonical_split_strategy, _stable_sample_seed, _sample_service_rows,
    _strategy_sample_indices, _counts_from_weights, _take_rows,
    _sample_count, _to_float, _safe_int,
)
from ..clients.partitioning import SUPPORTED_PARTITION_VERSIONS

from mlaas_data_generator.terminal import (
    debug_print, debug_enabled, terminal_scope,
    print_results, print_partition_notices, print_service_summary,
)

import hashlib
import json
import math
import os
import random
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from ..config import CONFIG
from ..data.accounting import finalize_accounting
from ..data.distributions import (
    get_data_distribution,
    get_mlm_masked_token_stats,
    get_retrieval_pair_stats,
    get_token_label_stats,
    get_vqa_answer_stats,
)
from ..data.skew_axes import axis_supports_strategy, bucket_distribution, resolve_skew_axis
from ..models.label_schema import infer_label_format, infer_num_labels
from ..models.train_eval import evaluate_model, evaluate_regression_model, train_local_model, resolve_fit_kwargs
from .update_signature import compute_and_store_update_signature
from ..storage.writer import make_writer
from ..checkpoint_profile import CheckpointProfile
from ..checkpoint_store import (
    TENSOR_PATCH_FORMAT,
    TENSOR_STATE_FORMAT,
    CheckpointStoreError,
    CompressionSpec,
    ContentObjectRef,
    LineageBaseResult,
    ServicePatchResult,
    TensorState,
    UnsupportedTensorStateError,
    capture_model_tensor_state,
    capture_model_tensor_state_or_fallback,
    ensure_lineage_base,
    persist_service_patch,
    persist_service_patch_from_base_state,
    restore_model_tensor_state,
)
from ..provenance import (
    PROVENANCE_VERSION, dataset_metadata, label_semantics, model_identity,
    parameter_schema, preprocessing_metadata, runtime_provenance,
    restore_lineage_initial_weights, restore_model_artifact, save_model_artifact,
    save_reconstruction_support_artifacts, sha256_file, sha256_json, write_json_artifact,
)
from .perturbation import run_perturbation_stage
from .system_metrics import ResourceTracker, capture_hardware_snapshot
from .taxonomy import (
    canonical_label_format, canonical_metric_names, canonical_task_family,
    metric_domain, metric_score_value,
)
from ..models.builders import prepare_keras_local_update

load_dataset = None
create_model = None


@dataclass
class ServiceExecutionResult:
    service_id: str
    status: str
    db_path: str
    metrics: dict[str, Any]
    error: str | None = None
    weights_saved: bool = False
    final_tensor_state: TensorState | None = None
    checkpoint_storage: dict[str, Any] | None = None


class ServiceExecutionError(RuntimeError):
    def __init__(self, message: str, *, failure_stage: str = "service_execution"):
        super().__init__(message)
        self.failure_stage = failure_stage


def _checkpoint_profile_metrics(profile: CheckpointProfile) -> dict[str, Any]:
    return {
        f"checkpoint_{name}_s": _metric(step["inclusive_s"], "runtime", "s", "lower_better")
        for name, step in profile.as_dict()["steps"].items()
    }


class _StageTimer:
    """Small helper for consistent stage timing in seconds."""

    def __init__(self):
        self._start = time.perf_counter()

    def elapsed_s(self) -> float:
        return float(time.perf_counter() - self._start)


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "na", "n/a", "nan", "null", "none", "not applicable", "not_applicable"}
    return False


def resolve_service_id(config: Mapping[str, Any]) -> str:
    explicit = config.get("service_id")
    if not _is_blank(explicit):
        return str(explicit).strip()

    parts = [
        config.get("task_type"),
        config.get("hf_task"),
        config.get("hf_model_id") or config.get("model_type"),
        config.get("dataset_name") or config.get("dataset"),
        config.get("dataset_config"),
        config.get("training_regime"),
        config.get("dataset_variant"),
        config.get("split_variant"),
        config.get("knob_variant"),
    ]
    raw = "|".join("" if value is None else str(value).strip().lower() for value in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    prefix = str(config.get("task") or config.get("task_type") or "service").strip().lower().replace(" ", "_")
    prefix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in prefix).strip("_") or "service"
    return f"{prefix}_{digest}"


from ..diagnostic_json import diagnostic_json, bounded_text
from ..execution_state import persist_terminal, execution_owner


class ServiceRunner:
    def __init__(self, config: Mapping[str, Any]):
        config = dict(config or {})
        self.config = dict(CONFIG)
        self.config.update(dict(config or {}))
        self.config.pop("max_samples", None)
        from ..execution_provenance import configuration_evidence
        self._requested_configuration = configuration_evidence(self.config)
        from ..task_contract import resolve_sentence_pair_contract
        resolve_sentence_pair_contract(self.config)
        from ..precision_policy import apply_precision_policy
        apply_precision_policy(self.config)
        from ..segmentation_policy import apply_segmentation_label_policy
        apply_segmentation_label_policy(self.config)
        # Saved configurations predating these options retain their exact policy.
        # New manifest generation freezes the current CONFIG explicitly.
        for key, legacy in (("checkpoint_patch_operation_policy", "adaptive"),
                            ("checkpoint_measure_full", True), ("checkpoint_weight_format", "native")):
            self.config[key] = config.get(key, legacy)
        storage_options = config.get("checkpoint_storage_options") or {}
        if not isinstance(storage_options, Mapping) or any(not key.startswith("checkpoint_") or key not in CONFIG for key in storage_options):
            raise ValueError("checkpoint_storage_options must contain supported checkpoint settings only")
        self.config.update(storage_options)
        from ..weight_precision import normalize_weight_format
        self.config["checkpoint_weight_format"] = normalize_weight_format(self.config["checkpoint_weight_format"])
        self.config["device"] = _normalize_requested_device(self.config.get("device"))
        self.service_id = resolve_service_id(self.config)
        from ..debug_mode import apply_debug_mode, debug_mode_error
        self.config["service_id"] = self.service_id
        apply_debug_mode(self.config)
        self.service_id = self.config["service_id"]
        if error := debug_mode_error(self.config):
            raise ValueError(error)

    @terminal_scope(lambda self: self.config)
    def run(self) -> ServiceExecutionResult:
        service_id = self.service_id
        db_path = str(self.config.get("db_path") or CONFIG["db_path"])
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = self.config.get("_client_started_perf", time.perf_counter())
        self._final_tensor_state = None
        self._resource_tracker = None
        self._checkpoint_profile = None
        writer = None
        self.config["_execution_started_at"] = started_at
        self.config["_execution_owner"] = execution_owner()

        try:
            self._persist_progress("executing", started_at=started_at)
            _resolve_hf_revisions(self.config)
            record, metrics, artifacts, split_provenance_rows, generation_provenance = self._execute_service(started_at=started_at)
            from ..execution_provenance import evidence_value, execution_document
            generation_evidence = evidence_value(generation_provenance)
            db_timer = _StageTimer()
            writer = make_writer("sqlite", db_path=db_path)
            writer.start()
            writer.write_service(record)
            writer.write_service_metrics(service_id, metrics)
            for row in split_provenance_rows:
                writer.write_service_split_provenance(service_id, **row)
            for artifact in artifacts:
                writer.write_service_artifact(service_id, **artifact)
            service_checkpoint_storage = None
            if generation_provenance:
                lineage = generation_provenance.pop("_lineage", None)
                lineage_checkpoint_storage = generation_provenance.pop(
                    "_lineage_checkpoint_storage", None
                )
                service_checkpoint_storage = generation_provenance.pop(
                    "_service_checkpoint_storage", None
                )
                if lineage:
                    writer.write_model_lineage(lineage)
                if lineage_checkpoint_storage:
                    writer.write_lineage_checkpoint_storage(lineage_checkpoint_storage)
                writer.write_service_generation_provenance(generation_provenance)
                if service_checkpoint_storage:
                    writer.write_service_checkpoint_storage(service_checkpoint_storage)
            writer.commit()
            # Include primary persistence and its commit. Only the final timing
            # bookkeeping transaction and terminal rendering are excluded.
            total_s = time.perf_counter() - t0
            completion_metrics = {
                key: _metric(total_s, "runtime", "s", "lower_better")
                for key in ("runtime_total_s", "runtime_s", "service_runtime_s")
            }
            client_start = self.config.get("_client_started_perf", t0)
            completion_metrics["client_runtime_s"] = _metric(time.perf_counter() - client_start, "runtime", "s", "lower_better")
            completion_metrics["db_write_s"] = _metric(db_timer.elapsed_s(), "runtime", "s", "lower_better")
            completion_metrics["non_training_runtime_s"] = _metric(
                total_s - float(metrics["training_runtime_s"]["value"]), "runtime", "s", "lower_better"
            )
            completion_metrics["timing_scope"] = _metric(
                "service_start_through_primary_commit_v2; excludes timing-bookkeeping commit and terminal output",
                "metadata",
            )
            old_total = float(metrics["runtime_total_s"]["value"])
            if "raw_resource_cost" in metrics:
                cost = float(metrics["raw_resource_cost"]["value"]) + total_s - old_total
                completion_metrics["raw_resource_cost"] = _metric(cost, "cost", None, "lower_better")
                completion_metrics["resource_cost_score"] = _metric(1.0 / (1.0 + cost), "cost", "score", "higher_better")
            metrics.update(completion_metrics)
            record["metadata_json"]["completed_at"] = datetime.now(timezone.utc).isoformat()
            writer.write_service_progress(record)
            writer.write_service_metrics(service_id, completion_metrics)
            writer.write_service_execution_provenance(
                service_id=service_id,
                run_id=self.config.get("_execution_run_id") or self.config.get("manifest_group_id"),
                started_at=started_at,
                document=execution_document(
                    requested=self._requested_configuration, effective=self.config,
                    metrics=metrics, record=record, splits=split_provenance_rows,
                    generation=generation_evidence, artifacts=artifacts,
                    runtime=runtime_provenance(),
                    training_adjustment=getattr(self, "_training_budget_adjustment", None),
                    training_config=getattr(self, "_effective_training_configuration", None),
                    fit_arguments=getattr(self, "_adapter_fit_arguments", None),
                ),
            )
            writer.finish()
            _log_service_timing(
                self.config,
                service_id,
                "DB write",
                db_timer.elapsed_s(),
                detail=f"db_path={db_path}",
            )
            if not self.config.get("_defer_terminal_results"):
                if not self.config.get("fl_run_id") and not debug_enabled():
                    print_service_summary({**self.config, "service_id": service_id}, metrics, service_checkpoint_storage)
                else:
                    label = "[Service results]"
                    if self.config.get("fl_run_id"):
                        label = f"[Client results] round={self.config.get('round_id')} client={self.config.get('client_id')}"
                    if debug_enabled():
                        label += f" service={service_id}"
                    print_results(label, metrics)
            weights_saved = bool(
                self.config.get("fl_run_id")
                and _safe_int(self.config.get("round_id")) is not None
                and _safe_int(self.config.get("round_id")) == _safe_int(self.config.get("fedavg_rounds"))
                and service_checkpoint_storage
                and service_checkpoint_storage.get("round_trip_status") == "passed"
            )
            if weights_saved and not self.config.get("_defer_terminal_results"):
                print(
                    f"Round {self.config['round_id']} Client {self.config.get('client_id')} "
                    "weights have been saved.",
                    flush=True,
                )
            return ServiceExecutionResult(service_id=service_id, status="success", db_path=db_path, metrics=metrics, weights_saved=weights_saved, final_tensor_state=self._final_tensor_state, checkpoint_storage=service_checkpoint_storage)
        except BaseException as exc:
            if writer is not None:
                writer.abort()
            elapsed = time.perf_counter() - t0
            status = "failed" if isinstance(exc, Exception) else "interrupted"
            failure_metrics = {key: _metric(elapsed, "runtime", "s", "lower_better")
                               for key in ("runtime_total_s", "runtime_s", "service_runtime_s", "client_runtime_s")}
            for key in ("training_memory_metadata", "evaluation_memory_metadata"):
                failure_metrics.update(_specify_metric_dict(self.config.get(key) or {}))
            failure_stage = getattr(exc, "failure_stage", "service_execution")
            failure_record = self._service_record(status=status, started_at=started_at,
                metadata={"started_at": started_at, "runtime_total_s": elapsed,
                          "error_type": type(exc).__name__, "error_message": bounded_text(str(exc)),
                          "failure_stage": failure_stage,
                          "training_budget_adjustment": getattr(self, "_training_budget_adjustment", None),
                          "checkpoint_profile": self._checkpoint_profile.as_dict() if self._checkpoint_profile else None})
            persist_terminal(self.config, status=status, reason=str(exc) or type(exc).__name__,
                             failure_stage=failure_stage, exc=exc, record=failure_record, metrics=failure_metrics)
            if status == "interrupted":
                raise
            return ServiceExecutionResult(service_id=service_id, status="failed", db_path=db_path,
                                          metrics=failure_metrics, error=bounded_text(str(exc)))
        finally:
            if self._resource_tracker is not None:
                self._resource_tracker.stop()

    def _execute_service(self, *, started_at: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        _assert_strict_cohort_access(self.config)
        service_start = time.perf_counter()
        stage_timings: dict[str, float] = {}
        training_regime = str(self.config.get("training_regime") or "finetune_transfer").strip().lower()
        prepared_dataset = self.config.get("_prepared_dataset")
        if prepared_dataset is None:
            dataset_args = dict(self.config.get("dataset_args") or {})
            dataset_args.setdefault("task", self.config.get("task") or self.config.get("task_type"))
            dataset_args.setdefault("inference_only", training_regime in {"inference_only", "inference"})
            dataset_args.setdefault("hf_dataset_revision", self.config.get("hf_dataset_revision"))
            dataset_args.setdefault("hf_model_revision", self.config.get("hf_model_revision"))
            dataset_args.pop("max_samples", None)
            dataset_timer = _StageTimer()
            self.config.pop("_pretokenization_split_info", None)
            from ..data.tokenization_selection import training_row_selection
            with training_row_selection([self.config]) as selection:
                train, test, meta = _load_dataset(self.config.get("dataset", "hf"), **dataset_args)
            selected_split = selection.materialize(self.service_id, train)
            if selected_split is not None:
                self.config["_pretokenization_split_info"] = selected_split
            stage_timings["dataset_load_s"] = dataset_timer.elapsed_s()
            _log_service_timing(
                self.config,
                self.service_id,
                "dataset load",
                stage_timings["dataset_load_s"],
                detail=f"dataset={self.config.get('dataset_name') or self.config.get('dataset')}",
            )
        else:
            dataset_timer = _StageTimer()
            train, test, meta = prepared_dataset
            stage_timings["dataset_load_s"] = float(self.config.get("_dataset_load_s") or dataset_timer.elapsed_s())
            _log_service_timing(
                self.config,
                self.service_id,
                "dataset load",
                stage_timings["dataset_load_s"],
                detail="source=prepared_dataset",
            )
        (x_train, y_train), (x_test, y_test) = train, test
        meta = dict(meta or {})
        stage_timings.update(_preprocessor_stage_timings(meta))
        meta.setdefault("hf_model_id", self.config.get("hf_model_id"))
        meta.setdefault("hf_task", self.config.get("hf_task"))
        meta.setdefault("task_tag", self.config.get("task_tag"))
        from ..task_contract import resolve_sentence_pair_contract
        resolve_sentence_pair_contract(self.config, meta)
        resolved_task_type = _resolved_task_type(self.config, meta)
        resolved_hf_task = meta.get("hf_task") or self.config.get("hf_task")
        resolved_task_tag = meta.get("task_tag") or self.config.get("task_tag")
        hf_metadata = _fetch_hf_metadata(self.config, meta)
        if hf_metadata:
            meta.setdefault("hf_metadata", hf_metadata)
            for key, value in hf_metadata.items():
                if value is not None:
                    meta.setdefault(key, value)

        split_timer = _StageTimer()
        prepared_partition = self.config.get("_prepared_train_partition")
        if isinstance(prepared_partition, Mapping):
            split_info = self._split_info_from_prepared_partition(prepared_partition, meta=meta)
        else:
            split_info = self._resolve_service_split(x_train, y_train, meta)
        stage_timings["split_selection_s"] = split_timer.elapsed_s()
        _log_service_timing(
            self.config,
            self.service_id,
            "split selection",
            stage_timings["split_selection_s"],
            detail=f"strategy={split_info['resolved'].get('effective_strategy')}",
        )
        x_train, y_train = split_info["x_train"], split_info["y_train"]
        prepared_benchmark_info = self.config.get("_prepared_benchmark_sample_info")
        if isinstance(prepared_benchmark_info, Mapping):
            benchmark_sample_info = dict(prepared_benchmark_info)
        else:
            x_test, y_test, benchmark_sample_info = _cap_benchmark_split(
                x_test,
                y_test,
                config=self.config,
            )
        split_provenance_rows = list(split_info["provenance_rows"])

        task_family = canonical_task_family(resolved_task_type, resolved_hf_task)
        if str(resolved_hf_task or "").strip().lower().replace("-", "_") == "sentence_similarity" and task_family == "regression":
            resolved_task_type = "regression"
        train_samples = _sample_count(x_train, y_train)
        benchmark_samples = _sample_count(x_test, y_test)
        _validate_service_compatibility(
            task_family=task_family,
            training_regime=training_regime,
            hf_model_id=self.config.get("hf_model_id") or meta.get("hf_model_id"),
            batch_size=_safe_int(self.config.get("batch_size")),
            train_samples=train_samples,
            benchmark_samples=benchmark_samples,
            task_contract=self.config.get("task_contract"),
        )
        primary_name, secondary_name = canonical_metric_names(
            task_family,
            self.config.get("metric_key"),
            hf_task=resolved_hf_task,
            task_tag=resolved_task_tag,
        )
        model_build_start = time.perf_counter()
        prepared_model = self.config.get("_prepared_model")
        if prepared_model is None:
            model = self._build_model(meta=meta, task_family=task_family)
        else:
            model = prepared_model
            _apply_runtime_knobs_to_model(model, self.config)
        model_build_s = time.perf_counter() - model_build_start
        if self.config.get("_model_reset_s") is not None:
            stage_timings["model_reset_s"] = float(self.config.get("_model_reset_s") or 0.0)
            _log_service_timing(
                self.config,
                self.service_id,
                "model reset",
                stage_timings["model_reset_s"],
                detail="source=prepared_model",
            )
        else:
            stage_timings["model_reset_s"] = 0.0
            _log_service_timing(
                self.config,
                self.service_id,
                "model reset",
                0.0,
                detail="not_applicable=row_local_model",
            )
        resolved_device = _resolve_execution_device(model)
        provenance_root = _provenance_artifact_root(self.config)
        lineage_id = str(self.config.get("lineage_id") or uuid.uuid4())
        fl_run_id = _safe_str(self.config.get("fl_run_id"))
        client_id = _safe_str(self.config.get("client_id"))
        round_id = _safe_int(self.config.get("round_id"))
        shared_parent = self.config.get("generation_mode") == "shared_parent"
        independent_family = self.config.get("_family_document") is not None
        if shared_parent and (fl_run_id is not None or round_id is not None or any(
            self.config.get(key) is None for key in (
                "_parent_state", "_parent_ref", "_parent_lineage_base", "_expected_compatibility",
                "cohort_id", "parent_checkpoint_id", "parent_state_hash", "client_id",
            )
        )):
            raise ServiceExecutionError(
                "shared-parent clients require a prepared cohort; use cohort-manifest and run-manifest",
                failure_stage="parent_contract",
            )
        if independent_family:
            client_id = self.service_id
        elif fl_run_id is None and not shared_parent:
            client_id, round_id = None, None
        initial_state_timer = _StageTimer()
        parameter_schema_document = parameter_schema(model)
        parameter_schema_hash = sha256_json(parameter_schema_document)
        identity = model_identity(model, self.config)
        labels = label_semantics(meta, task_family=task_family)
        preprocessing = preprocessing_metadata(meta)
        dataset_info = dataset_metadata(self.config, meta)
        expected_compatibility = self.config.get("_expected_compatibility")
        if expected_compatibility is not None:
            actual = {"architecture_id": identity["architecture_id"],
                      "parameter_schema_hash": parameter_schema_hash,
                      "label_map_hash": labels["label_map_hash"],
                      "preprocessing_contract_hash": preprocessing["preprocessing_contract_hash"],
                      "logical_dataset_contract_hash": dataset_info["logical_dataset_contract_hash"]}
            if self.config.get("task_contract"):
                from ..tasks.contracts import bind_task_contract
                task_contract = bind_task_contract(self.config, meta, model)
                actual.update(task_semantics_hash=task_contract.semantics_hash,
                              evaluation_protocol_hash=task_contract.protocol_hash)
            if actual != expected_compatibility:
                raise ServiceExecutionError("client compatibility differs from its parent", failure_stage="parent_contract")
        checkpoint_compression = _checkpoint_compression(self.config)
        lineage_base: LineageBaseResult | None = None
        native_base_round_trip_verified = False
        checkpoint_fallback: dict[str, Any] | None = None
        parent_state = self.config.get("_parent_state", self.config.get("_fedavg_parent_state"))
        parent_ref = self.config.get("_parent_ref", self.config.get("_fedavg_parent_ref"))
        parent_round = _safe_int(self.config.get("_parent_round_id", self.config.get("_fedavg_parent_round_id")))
        from ..debug_mode import weights_disabled
        no_save_weights = weights_disabled(self.config)
        if no_save_weights:
            base_artifact, base_hash = None, None
        elif parent_state is not None:
            if not isinstance(parent_state, TensorState) or not isinstance(
                parent_ref, ContentObjectRef
            ):
                raise ServiceExecutionError(
                    "client requires a canonical parent state and CAS reference",
                    failure_stage="fedavg_parent_restore",
                )
            if parent_ref.content_sha256 != parent_state.sha256:
                raise ServiceExecutionError(
                    "parent reference hash does not match the parent state",
                    failure_stage="fedavg_parent_restore",
                )
            declared_parent_hash = _safe_str(self.config.get("parent_state_hash"))
            if declared_parent_hash and declared_parent_hash != parent_state.sha256:
                raise ServiceExecutionError(
                    "manifest parent_state_hash does not match the persisted parent",
                    failure_stage="fedavg_parent_restore",
                )
            lineage_base = self.config.get("_parent_lineage_base", self.config.get("_fedavg_lineage_base"))
            if not isinstance(lineage_base, LineageBaseResult):
                raise ServiceExecutionError(
                    "client is missing its immutable parent lineage record",
                    failure_stage="fedavg_parent_restore",
                )
            restore_timer = _StageTimer()
            # Reuse the executor's verified restore of this exact parent object.
            # Standalone services still perform their own verified restore.
            if not (
                self.config.get("_restored_parent", self.config.get("_fedavg_restored_parent")) is parent_state
                and self.config.get("_prepared_model") is model
            ):
                restore_model_tensor_state(model, parent_state, verify=True)
                stage_timings["model_reset_s"] = restore_timer.elapsed_s()
            base_hash = parent_state.sha256
            base_artifact = {
                "artifact_type": "model_initial_checkpoint" if independent_family else "parent_checkpoint" if shared_parent else "fedavg_global_state",
                "artifact_uri": str(parent_ref.manifest_path),
                "metadata": {
                    "sha256": sha256_file(parent_ref.manifest_path),
                    "size_bytes": parent_ref.manifest_path.stat().st_size,
                    "format": TENSOR_STATE_FORMAT,
                    "role": "family_initial_state" if independent_family else "task_parent" if shared_parent else "round_global_parent",
                    "tensor_state_sha256": parent_state.sha256,
                    "global_round": parent_round,
                },
            }
            preparation_timer = _StageTimer()
            try:
                prepare_model_for_local_update(model, self.config)
            except Exception as exc:
                raise ServiceExecutionError(
                    f"local-update preparation failed: {exc}",
                    failure_stage="fedavg_local_preparation",
                ) from exc
            prepared_hash = capture_model_tensor_state(model).sha256
            if prepared_hash != parent_state.sha256:
                raise ServiceExecutionError(
                    "local-update preparation changed model parameters",
                    failure_stage="fedavg_local_preparation",
                )
            stage_timings["local_preparation_s"] = preparation_timer.elapsed_s()
        else:
            try:
                lineage_base = ensure_lineage_base(
                    provenance_root,
                    lineage_id,
                    model,
                    parameter_schema_hash=parameter_schema_hash,
                    compression=checkpoint_compression,
                )
                base_artifact = lineage_base.artifact()
                base_hash = lineage_base.tensor_state_sha256
            except UnsupportedTensorStateError as exc:
                checkpoint_fallback = exc.as_fallback_signal()
                base_artifact = save_model_artifact(
                    provenance_root,
                    self.service_id,
                    "initial",
                    model,
                    lineage_id=lineage_id,
                )
                if base_artifact is None:
                    raise ServiceExecutionError(
                        "model exposes neither an exact tensor state nor a native initial checkpoint",
                        failure_stage="checkpoint_persistence",
                    ) from exc
                native_base_round_trip_verified = restore_lineage_initial_weights(
                    provenance_root, lineage_id, model
                )
                if not native_base_round_trip_verified:
                    raise ServiceExecutionError(
                        "native initial checkpoint could not be restored into the constructed model",
                        failure_stage="checkpoint_round_trip",
                    ) from exc
                base_hash = (base_artifact or {}).get("metadata", {}).get("sha256")
        stage_timings["initial_state_s"] = initial_state_timer.elapsed_s()
        generated_artifacts = [item for item in (base_artifact,) if item]
        gpu_fallback_warning = _gpu_fallback_warning(self.config.get("device"), resolved_device)
        self._print_run_summary(
            task_family=task_family,
            split_info=split_info,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            resolved_device=resolved_device,
        )

        tracker = ResourceTracker()
        self._resource_tracker = tracker
        tracker_timer = _StageTimer()
        tracker.start()
        stage_timings["resource_tracker_start_s"] = tracker_timer.elapsed_s()
        train_metrics: dict[str, Any] = {}
        eval_qos: dict[str, Any] = {}
        loss = primary = secondary = math.nan
        train_runtime_s = 0.0
        update_signature_metrics: dict[str, Any] = {}
        update_signature_artifact: dict[str, Any] | None = None

        workload_start = time.perf_counter()
        stage_timings["setup_s"] = workload_start - service_start
        if training_regime in {"inference_only", "inference"}:
            stage_timings["finetune_s"] = 0.0
            stage_timings["signature_s"] = 0.0
            _log_service_timing(
                self.config,
                self.service_id,
                "finetune",
                0.0,
                detail="skipped=inference_only",
            )
            _log_service_timing(
                self.config,
                self.service_id,
                "signature",
                0.0,
                detail="skipped=inference_only",
            )
            eval_start = time.perf_counter()
            loss, primary, secondary, eval_qos = self._evaluate_model(model, x_test, y_test, inference_only=True, task_family=task_family)
            eval_runtime_s = time.perf_counter() - eval_start
        else:
            snapshot_timer = _StageTimer()
            signature_enabled = _config_bool(self.config.get("update_signature_enabled"), True)
            weights_before_training = _snapshot_model_weights(model) if signature_enabled else None
            stage_timings["pretrain_snapshot_s"] = snapshot_timer.elapsed_s()
            train_start = time.perf_counter()
            train_metrics = self._train_model(model, x_train, y_train, task_family=task_family)
            from ..clients.training import OPPORTUNITY_SAMPLER_VERSIONS
            if self.config.get("recipe_sampler_version") in OPPORTUNITY_SAMPLER_VERSIONS and self.config.get("optimizer") != "none":
                steps = train_metrics.get("successful_optimizer_step_count", train_metrics.get("optimizer_step_count"))
                if steps is not None and steps < 1:
                    raise ServiceExecutionError("client performed no successful optimizer steps", failure_stage="client_training")
            if shared_parent and train_metrics.get("optimizer_step_count", 1) < 1:
                raise ServiceExecutionError("client performed no optimizer steps", failure_stage="client_training")
            train_runtime_s = time.perf_counter() - train_start
            stage_timings["finetune_s"] = train_runtime_s
            _log_service_timing(
                self.config,
                self.service_id,
                "finetune",
                train_runtime_s,
                detail=f"train_samples={train_samples}",
            )
            signature_timer = _StageTimer()
            update_signature_metrics, update_signature_artifact = self._capture_update_signature(
                weights_before_training,
                _snapshot_model_weights(model) if signature_enabled else None,
            )
            stage_timings["signature_s"] = signature_timer.elapsed_s()
            _log_service_timing(
                self.config,
                self.service_id,
                "signature",
                stage_timings["signature_s"],
                detail=f"available={bool(update_signature_metrics.get('update_signature_available', {}).get('value'))}",
            )
            eval_start = time.perf_counter()
            loss, primary, secondary, eval_qos = self._evaluate_model(model, x_test, y_test, inference_only=False, task_family=task_family)
            eval_runtime_s = time.perf_counter() - eval_start
            train_metrics.setdefault("train_runtime_s", train_runtime_s)

        workload_runtime_s = time.perf_counter() - workload_start
        primary_score = _service_metric_score_value(
            task_family=task_family,
            hf_task=resolved_hf_task,
            primary_name=primary_name,
            primary_value=primary,
            secondary_name=secondary_name,
            secondary_value=secondary,
        )
        model_size = _count_model_params(model)
        _validate_service_metrics(
            loss_value=loss,
            task_family=task_family,
            hf_task=resolved_hf_task,
            primary_name=primary_name,
            primary_value=primary,
            secondary_name=secondary_name,
            secondary_value=secondary,
            metric_score=primary_score,
        )
        regression_controls = None
        if (task_family == "regression" and training_regime == "generic"
                and not isinstance(x_train, Mapping) and np.ndim(x_train) == 2):
            from ..models.regression_baselines import regression_baselines
            baseline_timer = _StageTimer()
            regression_controls = regression_baselines(x_train, y_train, x_test, y_test)
            stage_timings["regression_baselines_s"] = baseline_timer.elapsed_s()
        explainability, perturbation_artifact = _service_perturbation_metrics(
            model,
            x_test,
            y_test,
            config={**self.config, "service_id": self.service_id},
            meta=meta,
            task_family=task_family,
        )
        tracked_runtime_s = time.perf_counter() - workload_start
        tracker_timer = _StageTimer()
        usage = tracker.stop(tracked_runtime_s)
        self._resource_tracker = None
        stage_timings["resource_tracker_stop_s"] = tracker_timer.elapsed_s()
        runtime_total_s = time.perf_counter() - service_start
        perf_metrics = _performance_alias_metrics(
            eval_qos=eval_qos,
            train_metrics=train_metrics,
            training_regime=training_regime,
            train_runtime_s=train_runtime_s,
            eval_runtime_s=eval_runtime_s,
            runtime_total_s=runtime_total_s,
            benchmark_samples=benchmark_samples,
        )
        resource_metrics = _resource_metrics(
            runtime_total_s=runtime_total_s,
            workload_runtime_s=workload_runtime_s,
            train_metrics=train_metrics,
            eval_qos=eval_qos,
            model_size=model_size,
            usage=usage,
        )
        reliability = _reliability_metrics(status="completed", eval_qos=eval_qos)
        hardware_timer = _StageTimer()
        hardware_snapshot = capture_hardware_snapshot() if _config_bool(self.config.get("measure_system_metrics"), True) else None
        stage_timings["hardware_snapshot_s"] = hardware_timer.elapsed_s()
        accounting_meta = finalize_accounting(meta, batch_size=_safe_int(self.config.get("batch_size")))
        accounting = accounting_meta.get("accounting", {}) if isinstance(accounting_meta, Mapping) else {}
        train_distribution = _distribution_summary(x_train, y_train, meta=meta, task_family=task_family, hf_task=resolved_hf_task)
        benchmark_distribution = _distribution_summary(x_test, y_test, meta=meta, task_family=task_family, hf_task=resolved_hf_task)

        metrics: dict[str, Any] = {
            "loss": _metric(loss, "quality", direction="lower_better"),
            primary_name: _metric(primary, "quality", direction=_quality_direction(primary_name)),
            "metric_score": _metric(primary_score, "quality", direction="higher_better"),
            "primary_metric_name": _metric(primary_name, "metadata"),
            "auxiliary_metric_name": _metric(secondary_name, "metadata"),
            "model_build_s": _metric(model_build_s, "runtime", "s", "lower_better"),
            "workload_runtime_s": _metric(workload_runtime_s, "runtime", "s", "lower_better"),
            "evaluation_runtime_s": _metric(eval_runtime_s, "runtime", "s", "lower_better"),
            "eval_runtime_s": _metric(eval_runtime_s, "runtime", "s", "lower_better"),
            "training_runtime_s": _metric(train_runtime_s, "runtime", "s", "lower_better"),
            "train_runtime_s": _metric(train_runtime_s, "runtime", "s", "lower_better"),
            "runtime_total_s": _metric(runtime_total_s, "runtime", "s", "lower_better"),
            "runtime_s": _metric(runtime_total_s, "runtime", "s", "lower_better"),
            "service_runtime_s": _metric(runtime_total_s, "runtime", "s", "lower_better"),
            "compute_time_s": _metric(train_runtime_s + eval_runtime_s, "runtime", "s", "lower_better"),
            "model_params_count": _metric(model_size, "resource", "parameters", "lower_better"),
            "params_count": _metric(model_size, "resource", "parameters", "lower_better"),
            "model_size": _metric(model_size, "resource", "parameters", "lower_better"),
            "train_set_size": _metric(train_samples, "metadata", "samples", "neutral"),
            "benchmark_set_size": _metric(benchmark_samples, "metadata", "samples", "neutral"),
            "dataset_size": _metric(train_samples, "metadata", "samples", "neutral"),
            "task_family": _metric(task_family, "metadata"),
            "label_format": _metric(infer_label_format(meta, task_type=task_family) or canonical_label_format(task_family), "metadata"),
            "num_labels": _metric(infer_num_labels(meta, fallback=meta.get("num_classes")), "metadata"),
            "split_strategy": _metric(split_info["resolved"].get("strategy"), "metadata"),
            "split_strategy_requested": _metric(split_info["resolved"].get("requested_strategy"), "metadata"),
            "split_strategy_effective": _metric(split_info["resolved"].get("effective_strategy"), "metadata"),
            "data_distribution": _metric(split_info["resolved"].get("strategy"), "metadata"),
            "dataset_distribution_json": _metric(split_info["distribution_map"], "metadata"),
            "split_provenance_json": _metric(split_info["distribution_map"], "metadata"),
            "split_skew_axis": _metric(split_info["resolved"].get("requested_axis"), "metadata"),
            "split_skew_axis_effective": _metric(split_info["resolved"].get("effective_axis"), "metadata"),
            "split_bucket_spec_json": _metric(split_info["resolved"].get("bucket_spec"), "metadata"),
            "train_distribution_json": _metric(train_distribution, "metadata"),
            "benchmark_distribution_json": _metric(benchmark_distribution, "metadata"),
            "batch_size": _metric(_safe_int(self.config.get("batch_size")), "metadata", "samples", "neutral"),
            "learning_rate": _metric(_to_float(self.config.get("learning_rate")), "metadata", None, "neutral"),
            "epochs": _metric(_safe_int(self.config.get("training_epochs", self.config.get("epochs"))), "metadata", None, "neutral"),
            "device": _metric(resolved_device, "metadata"),
            "mixed_precision": _metric(bool(self.config.get("mixed_precision")), "metadata"),
            "precision_type": _metric(str(self.config.get("precision_type") or "fp16").strip().lower(), "metadata"),
            "gpu_requested_cpu_fallback_flag": _metric(bool(gpu_fallback_warning), "reliability", direction="lower_better"),
        }
        metrics.update(_stage_timing_metrics(stage_timings))
        if regression_controls is not None:
            metrics["regression_baselines"] = _metric(regression_controls, "metadata")
            for baseline, values in regression_controls["metrics"].items():
                for name, value in values.items():
                    metrics[f"baseline_{baseline}_{name}"] = _metric(value, "quality", direction="lower_better")
        if gpu_fallback_warning:
            metrics["gpu_fallback_warning"] = _metric(gpu_fallback_warning, "reliability")
        metrics.update(_hf_metadata_metrics(hf_metadata))
        metrics.update(perf_metrics)
        if secondary_name:
            metrics[secondary_name] = _metric(secondary, "quality", direction=_quality_direction(secondary_name))
        metrics.update(_specify_metric_dict(train_metrics))
        metrics.update(_specify_metric_dict(eval_qos))
        metrics.update(resource_metrics)
        metrics.update(explainability)
        metrics.update(reliability)
        metrics.update(update_signature_metrics)
        if accounting:
            metrics["dataset_accounting"] = _metric(accounting, "metadata")

        functional_attributes = {
            "task_family": task_family,
            "label_format": infer_label_format(meta, task_type=task_family) or canonical_label_format(task_family),
            "primary_metric": primary_name,
            "secondary_metric": secondary_name,
            "metric_score": primary_score,
            "input_schema": self.config.get("input_schema") or meta.get("input_schema"),
            "output_schema": self.config.get("output_schema") or meta.get("label_format"),
            "task_type": resolved_task_type,
            "hf_task": resolved_hf_task,
        }
        metadata = {
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "hardware_snapshot": hardware_snapshot,
            "loader_meta": _jsonable(meta),
            "hf_metadata": hf_metadata,
            "split_resolution": split_info["resolved"],
            "benchmark_sample_resolution": benchmark_sample_info,
            "split_provenance": split_info["distribution_map"],
            "resolved_device": resolved_device,
            "gpu_fallback_warning": gpu_fallback_warning,
            "service_source": self.config.get("service_source"),
            "fit_decision": self.config.get("fit_decision"),
            "fit_reason": self.config.get("fit_reason"),
            "resolved_task_type": resolved_task_type,
            "resolved_hf_task": resolved_hf_task,
        }
        record = self._service_record(
            status="completed",
            started_at=started_at,
            functional_attributes=functional_attributes,
            metadata=metadata,
            task_family=task_family,
            task_type=resolved_task_type,
            hf_task=resolved_hf_task,
            registry_metadata={**_registry_metadata(self.config), **hf_metadata},
        )
        if no_save_weights:
            # Debug observations intentionally have no reconstructable provider.
            # Return before checkpoint capture, compression, writes or verification.
            record["metadata_json"]["checkpoint_persistence"] = {
                "mode": "debug_no_weights", "weights_saved": False,
                "composition_compatible": False,
            }
            metrics["checkpoint_s"] = _metric(0.0, "runtime", "s", "lower_better")
            metrics["weights_saved"] = _metric(False, "metadata")
            split_provenance_rows.append({"split_name": "benchmark", "samples_count": benchmark_samples,
                "data_distribution": benchmark_distribution,
                "split_config": {"source": self.config.get("benchmark_split") or self.config.get("test_split"), "role": "benchmark"}})
            provenance = {
                "service_id": self.service_id, "provenance_version": PROVENANCE_VERSION,
                "architecture_id": identity["architecture_id"],
                "parameter_schema_hash": parameter_schema_hash,
                "label_map_hash": labels["label_map_hash"],
                "preprocessing_contract_hash": preprocessing["preprocessing_contract_hash"],
                "dataset_contract_hash": dataset_info["dataset_contract_hash"],
                "logical_dataset_contract_hash": dataset_info.get("logical_dataset_contract_hash"),
                "generation_mode": "debug_no_weights", "composition_compatible": 0,
                "provenance_json": {"mode": "debug_no_weights", "weights_saved": False,
                    "reason": "independent training/evaluation diagnostics; checkpoint persistence disabled"},
            }
            return record, metrics, [a for a in (perturbation_artifact, update_signature_artifact) if a], split_provenance_rows, provenance
        checkpoint_timer = _StageTimer()
        checkpoint_profile = self._checkpoint_profile = CheckpointProfile()
        if shared_parent and (not math.isfinite(float(primary)) or sha256_json(parameter_schema(model)) != parameter_schema_hash):
            raise ServiceExecutionError("client evaluation or parameter schema is invalid", failure_stage="client_validation")
        self._persist_progress("checkpointing", started_at=started_at)
        retain_trained_weights = True
        if isinstance(parent_state, TensorState):
            configured_retention = self.config.get("_retain_client_checkpoint")
            retain_trained_weights = (
                _config_bool(configured_retention, False)
                if configured_retention is not None
                else round_id == _safe_int(self.config.get("fedavg_rounds"))
            )
        if _config_bool(self.config.get("service_stage_timing_logging"), True):
            debug_print(
                f"[Checkpoint] service stage starts | service_id={self.service_id} "
                f"| compression={_checkpoint_compression(self.config).as_dict()}",
                flush=True,
            )
        service_patch: ServicePatchResult | None = None
        reduced_checkpoint = None
        final_artifact: dict[str, Any] | None = None
        fallback_target_state = None
        native_final_round_trip_verified = False
        if not retain_trained_weights:
            # Aggregation consumes this canonical state in memory. Its hash is
            # durable provenance, but no reconstructable client checkpoint is
            # published for a historical round.
            fallback_target_state = capture_model_tensor_state(model)
        if retain_trained_weights and lineage_base is not None and self.config.get("checkpoint_weight_format") in {"fp16", "int8"}:
            from ..reduced_checkpoint import persist_reduced_checkpoint, ReducedPrecisionRejected
            try:
                fallback_target_state = capture_model_tensor_state(model)
                reduced_base = parent_ref if isinstance(parent_ref, ContentObjectRef) else lineage_base.base
                if fallback_target_state.sha256 != reduced_base.content_sha256:
                    reduced_checkpoint = checkpoint_profile.call("reduced_save", persist_reduced_checkpoint,
                        provenance_root, self.service_id, lineage_id, fallback_target_state,
                        base_ref=parent_ref if isinstance(parent_ref, ContentObjectRef) else lineage_base.base,
                        parameter_schema_hash=parameter_schema_hash,
                        representation=self.config["checkpoint_weight_format"],
                        compression=CompressionSpec(self.config.get("checkpoint_reduced_compression", "none"),
                            self.config.get("checkpoint_reduced_compression_level", 1)),
                        measure_full=_config_bool(self.config.get("checkpoint_measure_full", False), False),
                        validate=lambda state: checkpoint_profile.call("reduced_quality", _validate_reduced_checkpoint,
                            self, model, fallback_target_state, state, x_test, y_test,
                            task_family=task_family, primary_name=primary_name))
                    final_artifact = reduced_checkpoint.artifact()
            except (UnsupportedTensorStateError, ReducedPrecisionRejected) as exc:
                checkpoint_fallback = {"requested": self.config["checkpoint_weight_format"], "selected": "exact", "reason": str(exc)}
        if retain_trained_weights and lineage_base is not None and reduced_checkpoint is None:
            try:
                if not shared_parent and not independent_family and isinstance(parent_state, TensorState) and isinstance(
                    parent_ref, ContentObjectRef
                ):
                    service_patch = checkpoint_profile.call("weights", persist_service_patch_from_base_state,
                        provenance_root,
                        self.service_id,
                        lineage_id,
                        model,
                        base_state=parent_state,
                        base_ref=parent_ref,
                        base_round_id=int(parent_round or 0),
                        parameter_schema_hash=parameter_schema_hash,
                        compression=checkpoint_compression,
                        operation_policy=self.config.get("checkpoint_patch_operation_policy", "adaptive"),
                        measure_full=_config_bool(self.config.get("checkpoint_measure_full", True), True),
                    )
                else:
                    service_patch = checkpoint_profile.call("weights", persist_service_patch,
                        provenance_root,
                        self.service_id,
                        lineage_id,
                        model,
                        parameter_schema_hash=parameter_schema_hash,
                        compression=checkpoint_compression,
                        operation_policy=self.config.get("checkpoint_patch_operation_policy", "adaptive"),
                        measure_full=_config_bool(self.config.get("checkpoint_measure_full", True), True),
                    )
                final_artifact = checkpoint_profile.call("artifact_metadata", service_patch.artifact)
            except UnsupportedTensorStateError as exc:
                checkpoint_fallback = _checkpoint_fallback_details(exc)
            except CheckpointStoreError as exc:
                if isinstance(parent_state, TensorState) and not shared_parent:
                    checkpoint_fallback = {
                        "required": True,
                        "policy": "full_object_v1",
                        "reason_code": "fedavg_exact_patch_round_trip_failed",
                        "detail": str(exc),
                    }
                else:
                    raise ServiceExecutionError(
                        f"exact checkpoint patch failed integrity verification: {exc}",
                        failure_stage="checkpoint_round_trip",
                    ) from exc
        if retain_trained_weights and service_patch is None and reduced_checkpoint is None:
            fallback_target_state, target_fallback = capture_model_tensor_state_or_fallback(model)
            if checkpoint_fallback is None and target_fallback is not None:
                checkpoint_fallback = target_fallback
            final_artifact = checkpoint_profile.call("native_save", save_model_artifact,
                provenance_root,
                self.service_id,
                "final",
                model,
            )
            if final_artifact is None:
                raise ServiceExecutionError(
                    "failed to persist either an exact tensor patch or a native final checkpoint",
                    failure_stage="checkpoint_persistence",
                )
            native_final_round_trip_verified = checkpoint_profile.call(
                "native_restore_verify", restore_model_artifact, final_artifact, model
            )
            if not native_final_round_trip_verified:
                raise ServiceExecutionError(
                    "native final checkpoint could not be restored into the trained model",
                    failure_stage="checkpoint_round_trip",
                )
            if fallback_target_state is not None:
                restored_target_state, restored_fallback = capture_model_tensor_state_or_fallback(model)
                if (
                    restored_fallback is not None
                    or restored_target_state is None
                    or restored_target_state.sha256 != fallback_target_state.sha256
                ):
                    raise ServiceExecutionError(
                        "native final checkpoint changed the canonical target tensor state",
                        failure_stage="checkpoint_round_trip",
                    )
            final_artifact.setdefault("metadata", {})["round_trip_verified"] = True
        if final_artifact is not None:
            generated_artifacts.append(final_artifact)
        support_artifacts = (
            checkpoint_profile.call("support_artifacts", save_reconstruction_support_artifacts,
                provenance_root,
                self.service_id,
                model,
            )
            if retain_trained_weights
            else []
        )
        generated_artifacts.extend(support_artifacts)
        provenance_documents = {
            "model_identity": identity,
            "parameter_schema": parameter_schema_document,
            "label_semantics": labels,
            "preprocessing": preprocessing,
            "dataset": dataset_info,
            "partition_manifest": {
                "tokenization_selection": meta.get("tokenization_selection"),
                "train_selected_rows": meta.get("selected_train_rows"),
                "test_selected_rows": meta.get("selected_test_rows"),
                "split_resolution": split_info["resolved"],
                "benchmark_sample_resolution": benchmark_sample_info,
                "fedavg_partition": (
                    dict(self.config.get("_prepared_train_partition", {}).get("partition_meta") or {})
                    if isinstance(self.config.get("_prepared_train_partition"), Mapping)
                    else None
                ),
            },
        }
        provenance_artifacts: dict[str, dict[str, Any]] = {}
        for name, document in provenance_documents.items():
            artifact = checkpoint_profile.call(
                "provenance_write", write_json_artifact, provenance_root, self.service_id, name, document
            )
            provenance_artifacts[name] = artifact
            generated_artifacts.append(artifact)
        service_checkpoint_storage = None
        if retain_trained_weights:
            checkpoint_storage = checkpoint_profile.call("storage_metadata", _checkpoint_storage_documents,
                service_id=self.service_id,
                lineage_id=lineage_id,
                parameter_schema_hash=parameter_schema_hash,
                base_artifact=base_artifact,
                lineage_base=lineage_base,
                final_artifact=final_artifact,
                service_patch=service_patch,
                reduced_checkpoint=reduced_checkpoint,
                fallback_target_state=fallback_target_state,
                support_artifacts=support_artifacts,
                provenance_artifacts=provenance_artifacts,
                fallback=checkpoint_fallback,
                compression=checkpoint_compression,
                provenance_root=provenance_root,
                native_final_round_trip_verified=native_final_round_trip_verified,
                parent_global_round=parent_round,
                explicit_parent_state_hash=(
                    parent_state.sha256
                    if isinstance(parent_state, TensorState)
                    else None
                ),
            )
            reconstruction_artifact = checkpoint_profile.call("reconstruction_manifest_write", write_json_artifact,
                provenance_root,
                self.service_id,
                "checkpoint_reconstruction",
                checkpoint_storage["reconstruction_manifest"],
            )
            generated_artifacts.append(reconstruction_artifact)
            checkpoint_storage["provenance"]["reconstruction_artifact"] = _artifact_reference(
                reconstruction_artifact
            )
            service_checkpoint_storage = _service_checkpoint_storage_row(
                service_id=self.service_id,
                lineage_id=lineage_id,
                parameter_schema_hash=parameter_schema_hash,
                final_artifact=final_artifact,
                service_patch=service_patch,
                reduced_checkpoint=reduced_checkpoint,
                lineage_base=lineage_base,
                fallback_target_state=fallback_target_state,
                reconstruction_artifact=reconstruction_artifact,
                fallback=checkpoint_fallback,
                native_final_round_trip_verified=native_final_round_trip_verified,
                parent_global_round=parent_round,
                explicit_parent_state_hash=(
                    parent_state.sha256
                    if isinstance(parent_state, TensorState)
                    else None
                ),
                explicit_parent_state_ref=(
                    str(parent_ref.manifest_path)
                    if isinstance(parent_ref, ContentObjectRef)
                    else None
                ),
            )
        else:
            checkpoint_storage = {
                "provenance": {
                    "storage_policy_version": "fedavg_historical_provenance_only_v1",
                    "storage_mode": "not_retained",
                    "client_weights_retained": False,
                    "target_state_sha256": fallback_target_state.sha256,
                    "parent_global_round": parent_round,
                    "parent_state_sha256": parent_state.sha256,
                    "reason": "only final-round FedAvg client weights are retained",
                },
                "reconstruction_manifest": None,
            }
        lineage_checkpoint_storage = _lineage_checkpoint_storage_row(
            lineage_id=lineage_id,
            parameter_schema_hash=parameter_schema_hash,
            base_artifact=(lineage_base.artifact() if lineage_base is not None else base_artifact),
            lineage_base=lineage_base,
            native_base_round_trip_verified=native_base_round_trip_verified,
            fallback=checkpoint_fallback,
        )
        artifacts = generated_artifacts
        if perturbation_artifact:
            artifacts.append(perturbation_artifact)
        if update_signature_artifact:
            artifacts.append(update_signature_artifact)
        checkpoint_elapsed_s = checkpoint_timer.elapsed_s()
        stage_timings["checkpoint_s"] = checkpoint_elapsed_s
        metrics.update(_checkpoint_profile_metrics(checkpoint_profile))
        metrics["checkpoint_s"] = _metric(
            checkpoint_elapsed_s, "runtime", "s", "lower_better"
        )
        _log_service_timing(
            self.config,
            self.service_id,
            "checkpoint",
            checkpoint_elapsed_s,
            detail=(
                f"compression={checkpoint_compression.as_dict()} "
                f"storage_mode={('patch' if service_patch is not None else 'full_object') if retain_trained_weights else 'historical_only'}"
            ),
        )
        split_provenance_rows.append(
            {
                "split_name": "benchmark",
                "samples_count": benchmark_samples,
                "data_distribution": benchmark_distribution,
                "split_config": {
                    "source": self.config.get("benchmark_split") or self.config.get("test_split"),
                    "role": "benchmark",
                },
            }
        )
        partition_hash = sha256_json(provenance_documents["partition_manifest"])
        final_hash = (
            service_patch.target_state_sha256
            if service_patch is not None
            else (
                fallback_target_state.sha256
                if fallback_target_state is not None
                else (final_artifact or {}).get("metadata", {}).get("sha256")
            )
        )
        full_provenance = {
            "version": PROVENANCE_VERSION,
            "documents": provenance_documents,
            "service_id": self.service_id,
            "created_at": started_at,
            "resolved_device": resolved_device,
            "identity": identity,
            "parameter_schema": parameter_schema_document,
            "labels": labels,
            "preprocessing": preprocessing,
            "dataset": dataset_info,
            "partition_manifest_hash": partition_hash,
            "base_model_hash": base_hash,
            "final_model_hash": final_hash,
            "checkpoint_storage": checkpoint_storage["provenance"],
            "checkpoint_profile": checkpoint_profile.as_dict(),
            "runtime": runtime_provenance(),
        }
        if self.config.get("task_contract"):
            from ..task_capabilities import resolve_capabilities
            full_provenance["task_contract"] = self.config["task_contract"]
            full_provenance["task_capabilities"] = resolve_capabilities(self.config, stage="publication",
                checks={"checkpoint_present": service_checkpoint_storage is not None,
                        "supervision": True, "training_completed": training_regime not in {"inference", "inference_only"},
                        "partition_support": True,
                        "checkpoint_integrity": bool(service_checkpoint_storage and service_checkpoint_storage.get("round_trip_status") == "passed")})
        if independent_family:
            full_provenance["independent_family"] = {
                key: self.config.get(key) for key in ("family_id", "initial_state_hash", "family_document_ref",
                    "family_document_hash", "manifest_reference", "compatibility_contract_hash", "training_seed")
            }
            full_provenance["independent_family"]["source_ids"] = split_info["resolved"]["partition_source_ids"]
        if shared_parent:
            full_provenance["shared_parent"] = {
                key: self.config.get(key) for key in ("family_id", "cohort_id", "parent_checkpoint_id",
                "parent_state_hash", "client_id", "client_recipe_hash", "training_seed", "compatibility_contract_hash")
            }
        if fl_run_id is not None:
            full_provenance["fedavg"] = {
                "generation_mode": "fedavg",
                "fl_run_id": fl_run_id,
                "client_id": client_id,
                "round_id": round_id,
                "run_partition_seed": self.config.get("run_partition_seed"),
                "client_partition_index": self.config.get("client_partition_index"),
                "benchmark_sample_seed": self.config.get("benchmark_sample_seed"),
                "fedavg_clients_per_run": self.config.get("fedavg_clients_per_run"),
                "fedavg_rounds": self.config.get("fedavg_rounds"),
                "participation_rate": self.config.get("fedavg_client_participation_rate"),
                "scheduled_to_participate": self.config.get("scheduled_to_participate"),
                "participation_schedule_seed": self.config.get("participation_schedule_seed"),
                "participation_schedule_version": self.config.get("participation_schedule_version"),
                "training_seed": self.config.get("training_seed"),
                "parent_global_round": parent_round,
                "parent_state_hash": base_hash,
                "compatibility_contract_hash": self.config.get("compatibility_contract_hash"),
                "client_weights_retained": retain_trained_weights,
            }
        generation_provenance = {
            "service_id": self.service_id,
            "provenance_version": PROVENANCE_VERSION,
            "architecture_id": identity["architecture_id"],
            "model_revision": identity.get("model_revision"),
            "dataset_revision": dataset_info.get("dataset_revision"),
            "parameter_schema_hash": parameter_schema_hash,
            "label_map_hash": labels["label_map_hash"],
            "preprocessing_contract_hash": preprocessing["preprocessing_contract_hash"],
            "dataset_contract_hash": dataset_info["dataset_contract_hash"],
            "logical_dataset_contract_hash": dataset_info.get("logical_dataset_contract_hash"),
            "partition_manifest_hash": partition_hash,
            "base_model_id": identity.get("model_id"),
            "base_model_hash": base_hash,
            "final_model_hash": final_hash,
            "lineage_id": lineage_id,
            "fl_run_id": fl_run_id,
            "client_id": client_id,
            "round_id": round_id,
            "compatibility_contract_hash": self.config.get("compatibility_contract_hash"),
            "parent_global_round": parent_round,
            "parent_state_hash": base_hash if fl_run_id is not None or shared_parent or independent_family else None,
            "generation_mode": self.config.get("generation_mode", "independent"),
            "family_id": self.config.get("family_id"),
            "parent_checkpoint_id": self.config.get("parent_checkpoint_id"),
            "cohort_id": self.config.get("cohort_id"),
            "training_universe_hash": self.config.get("training_universe_hash"),
            "benchmark_cohort_hash": self.config.get("benchmark_cohort_hash"),
            "scheduled_to_participate": (
                int(bool(self.config.get("scheduled_to_participate")))
                if fl_run_id is not None
                else None
            ),
            "participated": 1 if fl_run_id is not None else None,
            "provenance_json": full_provenance,
            "_lineage": {
                "lineage_id": lineage_id, "fl_run_id": fl_run_id,
                "base_model_id": identity.get("model_id"), "base_model_hash": base_hash,
                "initial_checkpoint_uri": (base_artifact or {}).get("artifact_uri"),
                "initial_checkpoint_sha256": (base_artifact or {}).get("metadata", {}).get("sha256"),
                "provenance_json": {"identity": identity, "runtime": full_provenance["runtime"]},
            },
            "_lineage_checkpoint_storage": lineage_checkpoint_storage,
            "_service_checkpoint_storage": service_checkpoint_storage,
        }
        final_capture_timer = _StageTimer()
        if independent_family:
            generation_provenance.update({key: self.config.get(key) for key in (
                "family_contract_version", "initial_state_hash", "family_document_ref", "family_document_hash", "manifest_reference")})
            generation_provenance["composition_compatible"] = 0
        if isinstance(parent_state, TensorState):
            self._final_tensor_state = (
                fallback_target_state if fallback_target_state is not None
                else capture_model_tensor_state(model)
            )
        metrics["final_state_capture_s"] = _metric(final_capture_timer.elapsed_s(), "runtime", "s", "lower_better")
        metrics["post_checkpoint_s"] = _metric(
            checkpoint_timer.elapsed_s() - checkpoint_elapsed_s,
            "runtime", "s", "lower_better",
        )
        return record, metrics, artifacts, split_provenance_rows, generation_provenance

    def _capture_update_signature(self, before: Any, after: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if not _config_bool(self.config.get("update_signature_enabled"), True):
            return {"update_signature_available": _metric(False, "metadata"),
                    "update_signature_disabled": _metric(True, "metadata")}, None
        if before is None or after is None:
            return {
                "update_signature_available": _metric(False, "metadata"),
                "update_signature_error": _metric("weights_unavailable", "metadata"),
            }, None

        output_dir = _update_signature_output_dir(self.config)
        try:
            metadata = compute_and_store_update_signature(
                before,
                after,
                output_dir=output_dir,
                run_id=self.service_id,
                round_idx=1,
                **{"cli" + "ent_" + "id": "service"},
                dim=_safe_int(self.config.get("update_signature_dim")) or 256,
                seed=_safe_int(self.config.get("seed")) or 42,
                max_source_elements=_safe_int(self.config.get("update_signature_max_source_elements")),
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "update_signature_available": _metric(False, "metadata"),
                "update_signature_error": _metric(type(exc).__name__, "metadata"),
            }, None

        if not metadata:
            return {
                "update_signature_available": _metric(False, "metadata"),
                "update_signature_error": _metric("no_comparable_weight_delta", "metadata"),
            }, None

        metrics = {
            "update_signature_available": _metric(True, "metadata"),
            "update_signature_id": _metric(metadata.get("update_signature_id"), "metadata"),
            "signature_dim": _metric(metadata.get("signature_dim"), "metadata", "dimensions"),
            "signature_norm": _metric(metadata.get("signature_norm"), "metadata"),
            "update_signature_path": _metric(metadata.get("update_signature_path"), "metadata"),
            "update_signature_method": _metric(metadata.get("update_signature_method"), "metadata"),
            "update_signature_source_dim": _metric(metadata.get("update_signature_source_dim"), "metadata", "parameters"),
            "update_signature_layer_count": _metric(metadata.get("update_signature_layer_count"), "metadata", "layers"),
        }
        artifact = {
            "artifact_type": "update_signature",
            "artifact_uri": str(metadata.get("update_signature_path")),
            "metadata": {
                "update_signature_id": metadata.get("update_signature_id"),
                "signature_dim": metadata.get("signature_dim"),
                "signature_norm": metadata.get("signature_norm"),
                "method": metadata.get("update_signature_method"),
            },
        }
        return metrics, artifact

    def _split_info_from_prepared_partition(self, prepared_partition: Mapping[str, Any], *, meta: Mapping[str, Any]) -> dict[str, Any]:
        x_train = prepared_partition.get("x_train")
        y_train = prepared_partition.get("y_train")
        split_resolution = dict(prepared_partition.get("split_resolution") or {})
        partition_meta = dict(prepared_partition.get("partition_meta") or {})
        hf_task = meta.get("hf_task") or self.config.get("hf_task")
        task_family = canonical_task_family(_resolved_task_type(self.config, meta), hf_task)
        train_distribution = _distribution_summary(x_train, y_train, meta=meta, task_family=task_family, hf_task=hf_task)
        strategy = str(split_resolution.get("strategy") or self.config.get("split_strategy") or "iid")
        resolved = {
            "strategy": strategy,
            "requested_strategy": split_resolution.get("requested_strategy", strategy),
            "effective_strategy": split_resolution.get("strategy", strategy),
            "distribution_type": self.config.get("distribution_type") or strategy,
            "distribution_param": self.config.get("distribution_param"),
            "sample_frac": self.config.get("sample_frac"),
            "sample_seed": partition_meta.get("sample_seed", self.config.get("sample_seed")),
            "requested_sample_size_total": self.config.get("sample_size"),
            "sample_strategy_effective": strategy,
            "requested_axis": split_resolution.get("requested_skew_axis") or split_resolution.get("skew_axis") or self.config.get("skew_axis"),
            "effective_axis": split_resolution.get("skew_axis") or partition_meta.get("skew_axis"),
            "axis_family": split_resolution.get("axis_family") or partition_meta.get("axis_family"),
            "bucket_spec": split_resolution.get("bucket_spec") or partition_meta.get("bucket_spec"),
            "source_fields": split_resolution.get("source_fields") or partition_meta.get("source_fields"),
            "fallback_reason": split_resolution.get("fallback_reason") or split_resolution.get("axis_fallback_reason") or partition_meta.get("fallback_reason"),
            "bucket_distribution": split_resolution.get("bucket_distribution") or partition_meta.get("bucket_histogram", {}),
            "effective_train_samples": _sample_count(x_train, y_train),
            "original_train_samples": partition_meta.get("source_pool_sample_count"),
            "provenance_only": False,
            "fedavg_partition": self.config.get("generation_mode") == "fedavg",
            "shared_parent_partition": self.config.get("generation_mode") == "shared_parent",
            "client_partition_index": prepared_partition.get("client_partition_index"),
            "run_partition_seed": prepared_partition.get("run_partition_seed"),
            "fl_run_id": prepared_partition.get("fl_run_id"),
            "client_key": partition_meta.get("client_key"),
            "partition_indices": partition_meta.get("indices"),
            "partition_source_ids": partition_meta.get("source_ids"),
            "partition_hash": partition_meta.get("partition_hash"),
        }
        partition_version = partition_meta.get("partition_generator_version") or partition_meta.get("version")
        if partition_version in SUPPORTED_PARTITION_VERSIONS:
            resolved.update({key: partition_meta[key] for key in (
                "source_pool_sample_count", "requested_quota", "effective_quota", "target_bucket_counts",
                "realised_bucket_counts", "capacity_reduction_count", "normalised_bucket_entropy",
                "cross_client_overlap_allowed")})
            resolved["partition_generator_version"] = partition_version
            for key in ("heterogeneity_intensity", "source_bucket_count", "selected_buckets",
                        "selected_bucket_count", "represented_bucket_count", "shard_expected_bucket_count",
                        "shard_support_quantile", "target_deviation_count", "resolution_reason",
                        "allocation_policy", "redistributed_sample_count", "selected_support_capacity",
                        "quantity_capacity_policy", "capacity_adjusted_quota", "quantity_scale_factor",
                        "regression_support_policy", "min_target_bin_fraction", "target_bin_coverage",
                        "target_extremes_represented", "controlled_specialist", "dirichlet_alpha",
                        "support_relaxation_count", "quantity_policy"):
                if key in partition_meta:
                    resolved[key] = partition_meta[key]
        if partition_version in {"joint_target_allocator_v2", "source_pool_sampler_v3"}:
            for key in (
                "partition_generator_version",
                "heterogeneity_intensity",
                "requested_partition_strategy",
                "requested_partition_params",
                "requested_quota",
                "effective_quota",
                "quota_shortfall",
                "initial_target_bucket_counts",
                "capacity_redraw_count",
                "source_pool_sample_count",
                "client_samples_used",
                "unused_source_samples",
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
                "partition_bucket_unit",
            ):
                resolved[key] = partition_meta.get(key)
        elif "backfill_count" in partition_meta:
            # Historical v1 provenance remains readable without introducing
            # its terminology into newly generated v2 records.
            resolved["backfill_count"] = partition_meta.get("backfill_count")
        distribution_map = {
            "train": train_distribution,
            "provenance": {
                "requested_strategy": resolved["requested_strategy"],
                "effective_strategy": resolved["effective_strategy"],
                "requested_axis": resolved["requested_axis"],
                "effective_axis": resolved["effective_axis"],
                "axis_family": resolved["axis_family"],
                "distribution_param": self.config.get("distribution_param"),
                "bucket_spec": resolved["bucket_spec"],
                "source_fields": resolved["source_fields"],
                "fallback_reason": resolved["fallback_reason"],
                "bucket_distribution": resolved["bucket_distribution"],
                "fedavg_partition": self.config.get("generation_mode") == "fedavg",
            "shared_parent_partition": self.config.get("generation_mode") == "shared_parent",
                "client_partition_index": resolved["client_partition_index"],
                "run_partition_seed": resolved["run_partition_seed"],
                "fl_run_id": resolved["fl_run_id"],
            },
        }
        if partition_version in {"joint_target_allocator_v2", "source_pool_sampler_v3"}:
            distribution_map["provenance"]["partition_summary"] = {
                key: resolved.get(key)
                for key in (
                    "partition_generator_version",
                    "heterogeneity_intensity",
                    "requested_partition_strategy",
                    "requested_partition_params",
                    "requested_quota",
                    "effective_quota",
                    "quota_shortfall",
                    "initial_target_bucket_counts",
                    "capacity_redraw_count",
                    "source_pool_sample_count",
                    "client_samples_used",
                    "unused_source_samples",
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
            }
        return {
            "x_train": x_train,
            "y_train": y_train,
            "resolved": resolved,
            "distribution_map": distribution_map,
            "provenance_rows": [
                {
                    "split_name": "train",
                    "samples_count": _sample_count(x_train, y_train),
                    "data_distribution": train_distribution,
                    "split_config": resolved,
                }
            ],
        }

    def _resolve_service_split(self, x_train, y_train, meta: Mapping[str, Any]) -> dict[str, Any]:
        selected_split = self.config.get("_pretokenization_split_info")
        if selected_split is not None:
            return selected_split
        if self.config.get("generation_mode", "independent") == "independent":
            from ..clients import partitioning
            partitioning.validate_partition_config(self.config)
            family = self.config.get("_family_document")
            hf_task = meta.get("hf_task") or self.config.get("hf_task")
            config = dict(self.config)
            if config.get("sample_seed") is None:
                config["sample_seed"] = _stable_sample_seed(config)
            config.setdefault("client_partition_index", 0)
            total = _sample_count(x_train, y_train)
            if config.get("sample_size") is None:
                config["sample_size"] = max(1, round(total * float(config.get("sample_frac") or 1)))
            config["split_strategy"] = _canonical_split_strategy(config.get("split_strategy") or config.get("distribution_type") or "iid")
            partition = partitioning.client_partition(x_train, y_train, config,
                source_ids=(family["reservation"]["training_ids"] if family else
                            [f"{meta.get('train_split') or 'train'}:{i}" for i in meta.get("selected_train_rows", range(total))]), meta=meta,
                task_family=canonical_task_family(_resolved_task_type(self.config, meta), hf_task))
            return self._split_info_from_prepared_partition(
                partitioning.materialize_partition(x_train, y_train, partition), meta=meta)
        requested_strategy = (
            self.config.get("split_strategy")
            or self.config.get("distribution_type")
            or self.config.get("data_distribution")
            or "iid"
        )
        strategy = _canonical_split_strategy(requested_strategy)
        if strategy not in {"iid", "dirichlet", "quantity_skew"}:
            raise ServiceExecutionError(
                f"service-local split strategy '{strategy}' is not supported; use iid or dirichlet",
                failure_stage="service_validation",
            )
        rng_seed = _safe_int(self.config.get("sample_seed"))
        if rng_seed is None:
            rng_seed = _stable_sample_seed(self.config)
        rng = np.random.default_rng(rng_seed)
        original_train_samples = _sample_count(x_train, y_train)
        hf_task = meta.get("hf_task") or self.config.get("hf_task")
        task_family = canonical_task_family(_resolved_task_type(self.config, meta), hf_task)
        axis = resolve_skew_axis(
            x_train,
            y_train,
            meta,
            split_name="train",
            task_family=task_family,
            hf_task=hf_task,
            requested_axis=self.config.get("skew_axis"),
            axis_config=_parse_jsonish(self.config.get("skew_axis_config")),
        )
        x_train, y_train, sample_info = _sample_service_rows(
            x_train,
            y_train,
            sample_size=self.config.get("sample_size"),
            sample_frac=self.config.get("sample_frac"),
            strategy=strategy,
            distribution_param=self.config.get("distribution_param"),
            axis=axis,
            rng=rng,
        )
        train_distribution = _distribution_summary(x_train, y_train, meta=meta, task_family=task_family, hf_task=hf_task)
        resolved = {
            "strategy": strategy,
            "requested_strategy": requested_strategy,
            "effective_strategy": sample_info.get("effective_strategy", strategy),
            "distribution_type": self.config.get("distribution_type") or strategy,
            "distribution_param": self.config.get("distribution_param"),
            "sample_frac": self.config.get("sample_frac"),
            "sample_seed": rng_seed,
            "requested_sample_size_total": sample_info.get("requested_sample_size_total"),
            "sample_strategy_effective": sample_info.get("sample_strategy_effective"),
            "sample_strategy_fallback_reason": sample_info.get("fallback_reason"),
            "requested_axis": axis.requested_axis,
            "effective_axis": sample_info.get("effective_axis", axis.effective_axis),
            "axis_family": axis.axis_family,
            "bucket_spec": axis.bucket_spec,
            "source_fields": axis.source_fields,
            "fallback_reason": sample_info.get("fallback_reason") or axis.fallback_reason,
            "bucket_distribution": sample_info.get("bucket_distribution", {}),
            "original_train_samples": original_train_samples,
            "effective_train_samples": _sample_count(x_train, y_train),
            "provenance_only": True,
        }
        if self.config.get("_family_document"):
            source = self.config["_family_document"]["reservation"]["training_ids"]
            indices = sample_info.get("_indices")
            resolved["partition_source_ids"] = list(source) if indices is None else [source[int(i)] for i in indices]
        distribution_map = {
            "train": train_distribution,
            "provenance": {
                "requested_strategy": requested_strategy,
                "effective_strategy": resolved["effective_strategy"],
                "requested_axis": axis.requested_axis,
                "effective_axis": resolved["effective_axis"],
                "axis_family": axis.axis_family,
                "distribution_param": self.config.get("distribution_param"),
                "bucket_spec": axis.bucket_spec,
                "source_fields": axis.source_fields,
                "fallback_reason": resolved["fallback_reason"],
                "bucket_distribution": resolved["bucket_distribution"],
            },
        }
        return {
            "x_train": x_train,
            "y_train": y_train,
            "resolved": resolved,
            "distribution_map": distribution_map,
            "provenance_rows": [
                {
                    "split_name": "train",
                    "samples_count": _sample_count(x_train, y_train),
                    "data_distribution": train_distribution,
                    "split_config": resolved,
                }
            ],
        }

    def _print_run_summary(
        self,
        *,
        task_family: str,
        split_info: Mapping[str, Any],
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        resolved_device: str,
    ) -> None:
        summary = self._build_run_summary(
            task_family=task_family,
            split_info=split_info,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            resolved_device=resolved_device,
        )
        if not debug_enabled(self.config):
            return  # Concise results are printed once after completion.
        print("========== CLIENT SETUP DEBUG ==========" if self.config.get("fl_run_id") else "========== SERVICE RUN SUMMARY ==========")
        for key, value in summary.items():
            print(f"{key}: {_format_summary_value(value)}")
        print("=========================================")

    def _build_run_summary(
        self,
        *,
        task_family: str,
        split_info: Mapping[str, Any],
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        resolved_device: str,
    ) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "case_name": self.config.get("case_name"),
            "task_family": task_family,
            "task_type": self.config.get("task_type"),
            "dataset_name": self.config.get("dataset_name") or _nested_get(self.config, "dataset_args", "dataset_name"),
            "dataset_config": self.config.get("dataset_config") or _nested_get(self.config, "dataset_args", "dataset_config"),
            "model_id": self.config.get("hf_model_id") or self.config.get("model_id") or getattr(model, "model_id", None) or self.config.get("model_type"),
            "training_regime": self.config.get("training_regime"),
            "resource_tier": self.config.get("resource_tier"),
            "dataset_variant": self.config.get("dataset_variant"),
            "split_variant": self.config.get("split_variant"),
            "knob_variant": self.config.get("knob_variant"),
            "split_strategy": _nested_get(split_info, "resolved", "strategy"),
            "split_strategy_effective": _nested_get(split_info, "resolved", "effective_strategy"),
            "split_skew_axis": _nested_get(split_info, "resolved", "effective_axis"),
            "split provenance": split_info.get("distribution_map"),
            "train samples": _sample_count(x_train, y_train),
            "benchmark samples": _sample_count(x_test, y_test),
            "effective model input samples": _sample_count(x_train, y_train),
            "batch size": self.config.get("batch_size"),
            "learning rate": self.config.get("learning_rate"),
            "epochs": self.config.get("training_epochs", self.config.get("epochs")),
            "device": resolved_device,
            "mixed_precision": self.config.get("mixed_precision"),
            "precision_type": self.config.get("precision_type"),
            "save_weights": _config_bool(self.config.get("save_weights"), False),
            "checkpoint_patch_compression": self.config.get("checkpoint_patch_compression"),
            "checkpoint_patch_compression_level": self.config.get("checkpoint_patch_compression_level"),
            "explainability_enabled": _config_bool(
                self.config.get("enable_perturbation_metrics", self.config.get("explainability_enabled")),
                True,
            ),
        }

    def _build_model(self, *, meta: Mapping[str, Any], task_family: str):
        input_shape = meta.get("input_shape")
        if input_shape is not None:
            input_shape = tuple(input_shape)
        return _create_model(
            input_shape=input_shape,
            num_classes=meta.get("num_classes"),
            hidden_layers=self.config.get("hidden_layers", [64]),
            learning_rate=float(self.config.get("learning_rate", 0.001) or 0.001),
            activation=self.config.get("activation", "relu"),
            weight_decay=float(self.config.get("weight_decay", 0.0) or 0.0),
            optimizer=self.config.get("optimizer", "adam"),
            task_type=task_family if task_family in {"classification", "regression", "clustering"} else self.config.get("task_type", task_family),
            model_type=self.config.get("model_type"),
            meta=dict(meta),
            hf_model_id=self.config.get("hf_model_id"),
            hf_model_revision=self.config.get("hf_model_revision"),
            dataset_name=self.config.get("dataset_name"),
            hf_task=self.config.get("hf_task"),
            max_length=self.config.get("max_length"),
            device=_device_arg_for_model(self.config.get("device")),
            mixed_precision=self.config.get("mixed_precision"),
            precision_type=self.config.get("precision_type"),
            mobilebert_head_normalization=self.config.get("mobilebert_head_normalization"),
            max_grad_norm=self.config.get("max_grad_norm"),
            batch_size=_effective_hf_batch_size(self.config, task_family),
            task_tag=self.config.get("task_tag"),
            clustering_k=self.config.get("clustering_k"),
            clustering_init=self.config.get("clustering_init", "k-means++"),
            clustering_n_init=self.config.get("clustering_n_init", 10),
            clustering_max_iter=self.config.get("clustering_max_iter", 300),
            clustering_tol=self.config.get("clustering_tol", 1e-4),
            seed=self.config.get("seed", 42),
        )

    def _train_model(self, model, x_train, y_train, *, task_family: str) -> dict[str, Any]:
        from ..clients.training import realized_training_recipe
        requested_config = self.config
        effective, adjustment = realized_training_recipe(requested_config, _sample_count(x_train, y_train))
        self._training_budget_adjustment = adjustment
        self.config = effective
        try:
            from ..clients.reviewed_opportunity import validate_realized_coverage, runtime_evidence
            validate_realized_coverage(effective, y_train)
            metrics = self._train_model_effective(model, x_train, y_train, task_family=task_family)
            metrics.update(runtime_evidence(effective, metrics, _sample_count(x_train, y_train)))
        finally:
            from ..execution_provenance import configuration_evidence
            self._effective_training_configuration = configuration_evidence(effective)
            if "training_memory_metadata" in effective:
                requested_config["training_memory_metadata"] = effective["training_memory_metadata"]
            self.config = requested_config
        if adjustment:
            metrics["training_budget_adjustment"] = adjustment
            metrics["effective_training_epochs"] = effective["training_epochs"]
            metrics["effective_gradient_accumulation_steps"] = metrics.get(
                "gradient_accumulation_steps_effective", effective["gradient_accumulation_steps"])
        return metrics

    def _train_model_effective(self, model, x_train, y_train, *, task_family: str) -> dict[str, Any]:
        from ..clients.training import validate_training_opportunity
        try:
            validate_training_opportunity(self.config, sample_size=_sample_count(x_train, y_train))
        except ValueError as exc:
            raise ServiceExecutionError(str(exc), failure_stage="client_training") from exc
        epochs = int(self.config.get("training_epochs", self.config.get("epochs", 1)) or 1)
        batch_size = int(self.config.get("batch_size", 32) or 32)
        learning_rate = float(self.config.get("learning_rate", 0.001) or 0.001)
        if task_family == "clustering" and hasattr(model, "fit"):
            model.fit(x_train)
            return {"training_epochs": epochs}
        if hasattr(model, "fit") and model.__class__.__name__.lower().startswith("transformers"):
            fit_kwargs = {
                "epochs": epochs,
                "lr": learning_rate,
                "optimizer": self.config.get("optimizer", "adamw"),
                "weight_decay": float(self.config.get("weight_decay", 0.0) or 0.0),
                "warmup_ratio": float(self.config.get("warmup_ratio", 0.0) or 0.0),
                "gradient_accumulation_steps": int(self.config.get("gradient_accumulation_steps", 1) or 1),
                "max_train_time_s": self.config.get("max_train_time_s", 60),
                "progress_log_interval": self.config.get("train_progress_log_interval", 10),
            }
            frozen_settings = self.config.get("generation_mode") == "shared_parent" or self.config.get("_family_document")
            fallback_kwargs = None if frozen_settings else {
                key: fit_kwargs[key]
                for key in ("epochs", "lr", "max_train_time_s", "progress_log_interval")
                if key in fit_kwargs
            }
            try:
                fit_kwargs = resolve_fit_kwargs(model, x_train, y_train, fit_kwargs, legacy_kwargs=fallback_kwargs)
            except TypeError as exc:
                if frozen_settings:
                    raise ServiceExecutionError("adapter cannot apply the frozen client training settings", failure_stage="client_training") from exc
                raise
            from ..execution_provenance import evidence_value
            self._adapter_fit_arguments = evidence_value(fit_kwargs)
            qos = _fit_transformers_with_memory_metadata(model, x_train, y_train, fit_kwargs, self.config)
            return dict(qos or {}, training_epochs=epochs)
        if ((self.config.get("generation_mode") == "shared_parent" or self.config.get("_family_document"))
                and callable(getattr(model, "compile", None)) and getattr(model, "optimizer", None) is not None):
            import tensorflow as tf
            started = time.perf_counter()
            limit = self.config.get("max_train_time_s")
            counts = {"steps": 0, "epochs": 0, "stopped": False}

            class LocalBudget(tf.keras.callbacks.Callback):
                last_progress_at = started

                def on_train_batch_end(self, batch, logs=None):
                    counts["steps"] += 1
                    now = time.perf_counter()
                    if counts["steps"] == 1 or now - self.last_progress_at >= 10:
                        print(f"Training: epoch={counts['epochs'] + 1}/{epochs} "
                              f"batch={batch + 1} elapsed_s={now - started:.1f}", flush=True)
                        self.last_progress_at = now
                    if limit is not None and time.perf_counter() - started >= float(limit):
                        counts["stopped"] = True
                        self.model.stop_training = True

                def on_epoch_end(self, epoch, logs=None):
                    if not counts["stopped"]:
                        counts["epochs"] += 1
                    print(f"Training: epoch={epoch + 1}/{epochs} finished "
                          f"loss={(logs or {}).get('loss')}", flush=True)

            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                                callbacks=[LocalBudget()])
            return {"training_epochs": epochs, "completed_training_epochs": counts["epochs"],
                    "optimizer_step_count": counts["steps"], "train_stopped_early": counts["stopped"],
                    "train_stopping_reason": "resource_limit" if counts["stopped"] else "epochs_completed",
                    "train_loss": float(history.history["loss"][-1])}
        train_local_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size, lr=learning_rate)
        return {"training_epochs": epochs}

    def _evaluate_model(self, model, x_test, y_test, *, inference_only: bool, task_family: str,
                        enforce_time_limit: bool = True) -> tuple[float, float, float, dict[str, Any]]:
        try:
            if self.config.get("task_contract") and task_family in {"token_classification", "segmentation", "generation"}:
                from ..tasks.contracts import TaskContract
                from ..tasks.runtime import evaluate_task_model
                contract = TaskContract.from_dict(self.config["task_contract"])
                result = evaluate_task_model(model, x_test, y_test, contract, [str(i) for i in range(len(y_test))],
                    max_eval_time_s=self.config.get("max_eval_time_s") if enforce_time_limit else None)
                metrics = result.metrics
                # The existing runner names primary/secondary independently; provide its historical order.
                primary, secondary = metrics[contract.evaluation["primary"]], metrics[contract.evaluation["secondary"]]
                qos = {key: value for key, value in metrics.items() if isinstance(value, (int, float, str))}
                # Keep already-computed prediction counts for collapse review.
                # Dropping the nested per-class table otherwise discards this
                # evidence and forces a second inference pass to recover it.
                for label, values in metrics.get("per_class", {}).items():
                    qos[f"metric_stat_class_{label}_pred_total"] = int(values["predicted"])
                    qos[f"metric_stat_class_{label}_target_total"] = int(values["support"])
                return metrics["loss"], primary, secondary, qos
            if hasattr(model, "evaluate") and model.__class__.__name__.lower().startswith("transformers"):
                loss, primary, secondary, qos = model.evaluate(
                    x_test,
                    y_test,
                    inference_only=inference_only,
                    max_eval_time_s=self.config.get("max_eval_time_s") if enforce_time_limit else None,
                    progress_log_interval=self.config.get("eval_progress_log_interval", 10),
                )
                return loss, primary, secondary, dict(qos or {})
            if task_family == "clustering" and hasattr(model, "evaluate"):
                loss, primary, secondary = model.evaluate(x_test, y_test)
                return loss, primary, secondary, {}
            if task_family == "regression":
                values = evaluate_regression_model(model, x_test, y_test)
                return values["loss"], values["rmse"], values["mae"], {
                    name: values[name]
                    for name in ("mse", "rmse", "mae")}
            loss, primary, secondary = evaluate_model(model, x_test, y_test, task_type="classification")
            return loss, primary, secondary, {}
        finally:
            core = getattr(model, "core", model)
            self.config["evaluation_memory_metadata"] = dict(getattr(core, "evaluation_memory_metadata", {}))


    def _service_record(
        self,
        *,
        status: str,
        started_at: str,
        functional_attributes: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        task_family: str | None = None,
        task_type: str | None = None,
        hf_task: str | None = None,
        registry_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        training_regime = self.config.get("training_regime")
        return {
            "service_id": self.service_id,
            "status": status,
            "case_name": self.config.get("case_name"),
            "task_family": task_family or self.config.get("task_type"),
            "task_type": task_type or self.config.get("task_type"),
            "modality": self.config.get("modality"),
            "input_schema": self.config.get("input_schema"),
            "output_schema": self.config.get("output_schema"),
            "dataset": self.config.get("dataset"),
            "dataset_name": self.config.get("dataset_name"),
            "dataset_config": self.config.get("dataset_config"),
            "train_split": self.config.get("train_split"),
            "benchmark_split": self.config.get("benchmark_split") or self.config.get("test_split"),
            "model_type": self.config.get("model_type"),
            "model_id": self.config.get("hf_model_id") or self.config.get("model_id"),
            "hf_task": hf_task or self.config.get("hf_task"),
            "training_regime": training_regime,
            "dataset_variant": self.config.get("dataset_variant"),
            "split_variant": self.config.get("split_variant"),
            "knob_variant": self.config.get("knob_variant"),
            "service_config_json": {**(self.config.get("service_config") or {}), **_service_config(self.config)},
            "registry_metadata_json": registry_metadata if registry_metadata is not None else _registry_metadata(self.config),
            "functional_attributes_json": functional_attributes or {},
            "metadata_json": {**(metadata or {"started_at": started_at}),
                "training_memory_metadata": self.config.get("training_memory_metadata"),
                "evaluation_memory_metadata": self.config.get("evaluation_memory_metadata"),
                "execution_owner": self.config.get("_execution_owner"),
                "execution_run_id": self.config.get("_execution_run_id"),
                "manifest_group_id": self.config.get("manifest_group_id")},
        }

    def _persist_progress(self, phase: str, *, started_at: str) -> None:
        """Track execution phase locally; publish only a finished service result."""
        self.config["_execution_phase"] = str(phase)
        self.config["_execution_started_at"] = started_at


@terminal_scope(lambda config: config)
def execute_service(config: Mapping[str, Any]) -> ServiceExecutionResult:
    try:
        instance = ServiceRunner(config)
    except BaseException as exc:
        resolved = {**CONFIG, **config}
        resolved["service_id"] = resolve_service_id(resolved)
        from ..debug_mode import apply_debug_mode
        apply_debug_mode(resolved)
        status = "failed" if isinstance(exc, Exception) else "interrupted"
        persist_terminal(resolved, status=status, reason=str(exc) or type(exc).__name__,
                         failure_stage="service_initialization", exc=exc)
        if status == "interrupted":
            raise
        return ServiceExecutionResult(service_id=resolved["service_id"], status="failed",
                                      db_path=str(resolved["db_path"]), metrics={}, error=bounded_text(str(exc)))
    return instance.run()


def persist_status_only_service(
    config: Mapping[str, Any], *, status: str, reason: str | None = None
) -> ServiceExecutionResult:
    """Persist a FedAvg observation that intentionally has no trained model."""
    resolved = dict(config)
    runner = ServiceRunner(resolved)
    db_path = str(resolved.get("db_path") or CONFIG.get("db_path") or "mlaas.db")
    now = datetime.now(timezone.utc).isoformat()
    record = runner._service_record(
        status=str(status),
        started_at=now,
        metadata={
            "started_at": now,
            "completed_at": now,
            "status_reason": reason,
            "fedavg": {
                "fl_run_id": resolved.get("fl_run_id"),
                "client_id": resolved.get("client_id"),
                "round_id": resolved.get("round_id"),
                "scheduled_to_participate": resolved.get("scheduled_to_participate"),
                "participation_schedule_seed": resolved.get("participation_schedule_seed"),
                "participation_schedule_version": resolved.get("participation_schedule_version"),
            },
        },
    )
    writer = make_writer("sqlite", db_path=db_path)
    try:
        writer.start()
        writer.write_service(record)
        writer.finish()
    except Exception:
        writer.abort()
        raise
    return ServiceExecutionResult(
        service_id=runner.service_id,
        status=str(status),
        db_path=db_path,
        metrics={},
        error=reason,
    )


def reset_model_to_weights(model: Any, weights: Any) -> bool:
    """Restore a model payload captured by _snapshot_model_weights."""
    set_weights = getattr(model, "set_weights", None)
    if not callable(set_weights) or weights is None:
        return False
    set_weights(_clone_weight_payload(weights))
    _clear_model_runtime_state(model)
    return True


def _effective_hf_batch_size(config: Mapping[str, Any], task_family: str | None = None) -> int:
    requested = int(config.get("batch_size", 16) or 16)
    model_id = str(config.get("hf_model_id") or config.get("model_id") or "").strip().lower()
    family = str(task_family or config.get("task_type") or "").strip().lower()
    if family == "segmentation" and model_id.startswith("nvidia/segformer-b5-"):
        return max(1, min(requested, 2))
    return requested


def snapshot_model_weights(model: Any) -> Any | None:
    return _snapshot_model_weights(model)


def _normalize_requested_device(value: Any) -> Any:
    if value is None:
        return "auto"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "nan", "auto", "gpu", "auto_gpu", "cuda_auto"}:
            return "auto"
        return value.strip()
    return value


def _device_arg_for_model(value: Any) -> Any:
    return None if _normalize_requested_device(value) == "auto" else value


def _gpu_fallback_warning(requested_device: Any, resolved_device: Any) -> str | None:
    requested = str(_normalize_requested_device(requested_device)).strip().lower()
    resolved = str(resolved_device or "").strip().lower()
    if requested in {"auto", "gpu", "auto_gpu", "cuda"} and resolved == "cpu":
        return "gpu_requested_but_cpu_resolved"
    return None


def _apply_runtime_knobs_to_model(model: Any, config: Mapping[str, Any]) -> None:
    batch_size = _safe_int(config.get("batch_size"))
    max_length = _safe_int(config.get("max_length"))
    max_grad_norm = config.get("max_grad_norm")
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
        if not np.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
    for candidate in (model, getattr(model, "core", None)):
        if candidate is None:
            continue
        if hasattr(candidate, "max_grad_norm"):
            candidate.max_grad_norm = max_grad_norm
        if batch_size is not None and hasattr(candidate, "batch_size"):
            try:
                setattr(candidate, "batch_size", int(batch_size))
            except Exception:
                pass
        if max_length is not None and hasattr(candidate, "max_length"):
            try:
                setattr(candidate, "max_length", int(max_length))
            except Exception:
                pass
        if hasattr(candidate, "mixed_precision"):
            try:
                setattr(candidate, "mixed_precision", bool(config.get("mixed_precision")))
            except Exception:
                pass
        if hasattr(candidate, "precision_type"):
            try:
                setattr(candidate, "precision_type", str(config.get("precision_type") or "fp16").strip().lower())
            except Exception:
                pass
        reconfigure = getattr(candidate, "_configure_precision_mode", None)
        if callable(reconfigure):
            try:
                reconfigure()
            except Exception:
                pass


def _clear_cuda_cache_if_available(model: Any = None) -> None:
    torch_mod = None
    core = getattr(model, "core", None) if model is not None else None
    for candidate in (model, core):
        if candidate is None:
            continue
        torch_mod = getattr(candidate, "torch", None) or getattr(candidate, "_torch", None)
        if torch_mod is not None:
            break
    if torch_mod is None:
        try:
            import torch as torch_mod  # type: ignore[no-redef]
        except Exception:
            torch_mod = None
    cuda = getattr(torch_mod, "cuda", None) if torch_mod is not None else None
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
        except Exception:
            pass


def _fit_transformers_with_memory_metadata(
    model: Any,
    x_train: Any,
    y_train: Any,
    fit_kwargs: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    # HFCore owns transactional microbatch recovery. Re-running model.fit here
    # could duplicate already committed updates and silently alter the recipe.
    try:
        return model.fit(x_train, y_train, **fit_kwargs)
    finally:
        core = getattr(model, "core", model)
        config["training_memory_metadata"] = dict(getattr(core, "training_memory_metadata", {}))


def _clear_model_runtime_state(model: Any) -> None:
    core = getattr(model, "core", None)
    torch = getattr(core, "torch", None)
    raw_model = getattr(core, "model", None) or getattr(model, "model", None) or model
    if raw_model is not None:
        zero_grad = getattr(raw_model, "zero_grad", None)
        if callable(zero_grad):
            try:
                zero_grad(set_to_none=True)
            except TypeError:
                try:
                    zero_grad()
                except Exception:
                    pass
            except Exception:
                pass
        eval_fn = getattr(raw_model, "eval", None)
        if callable(eval_fn):
            try:
                eval_fn()
            except Exception:
                pass
    if torch is not None:
        cuda = getattr(torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            try:
                empty_cache()
            except Exception:
                pass


def prepare_model_for_local_update_seed(config: Mapping[str, Any]) -> None:
    """Seed model construction, independently of later local-update resets."""
    seed = int(config.get("training_seed", config.get("seed", 42)))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if config.get("model_type") in {"mlp", "cnn", "logreg"}:
        try:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(seed)
        except ImportError:
            pass
    else:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass


def prepare_model_for_local_update(model: Any, config: Mapping[str, Any]) -> None:
    """Reset all training-only state without changing model parameters."""
    seed = _safe_int(config.get("training_seed"))
    if seed is None:
        seed = _safe_int(config.get("seed")) or 0
    random.seed(seed)
    np.random.seed(seed % (2**32))

    candidates = (model, getattr(model, "core", None), getattr(model, "model", None))
    keras_like = any(
        candidate is not None
        and any(
            token in str(getattr(candidate.__class__, "__module__", "")).lower()
            for token in ("keras", "tensorflow")
        )
        for candidate in candidates
    )
    if keras_like:
        try:
            import tensorflow as tf

            tf.keras.utils.set_random_seed(seed)
            # Existing Keras Dropout/random layers own generators whose counters
            # survive global seeding and recompilation. Reset those separately
            # from model weights so execution order cannot transfer randomness.
            for owner in candidates:
                flatten = getattr(owner, "_flatten_layers", None)
                if not callable(flatten):
                    continue
                for index, layer in enumerate(flatten(include_self=True, recursive=True)):
                    for offset, generator in enumerate(getattr(layer, "_seed_generators", ())):
                        generator.state.assign([(seed + 104729 * index + offset) % (2**31 - 1), 0])
        except (ImportError, AttributeError):
            pass
    torch_like = any(
        candidate is not None
        and (
            callable(getattr(candidate, "state_dict", None))
            or getattr(candidate, "torch", None) is not None
        )
        for candidate in candidates
    )
    if torch_like:
        try:
            import torch

            torch.manual_seed(seed)
            if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except (ImportError, AttributeError):
            pass

    _clear_model_runtime_state(model)
    prepare_keras_local_update(
        model,
        optimizer=str(config.get("optimizer") or "adam"),
        learning_rate=float(config.get("learning_rate") or 0.001),
        weight_decay=float(config.get("weight_decay") or 0.0),
    )
    for candidate in (model, getattr(model, "core", None)):
        if candidate is None:
            continue
        for name in ("optimizer", "scheduler", "scaler", "grad_scaler"):
            if hasattr(candidate, name) and "keras" not in str(
                getattr(candidate.__class__, "__module__", "")
            ).lower():
                try:
                    setattr(candidate, name, None)
                except Exception:
                    pass
    _apply_runtime_knobs_to_model(model, config)


def _resolve_execution_device(model) -> str:
    candidates = [model, getattr(model, "core", None), getattr(model, "model", None)]
    for obj in candidates:
        if obj is None:
            continue
        device = getattr(obj, "device", None)
        if device is not None:
            return str(device)
    for obj in candidates:
        parameters = getattr(obj, "parameters", None) if obj is not None else None
        if callable(parameters):
            try:
                return str(next(parameters()).device)
            except Exception:
                pass
    return "unknown"


def _parse_jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return value
    return value


def _cap_benchmark_split(x, y, *, config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    cap = (
        config.get("benchmark_sample_size")
        or config.get("max_benchmark_samples")
        or config.get("max_eval_samples")
    )
    requested = _safe_int(cap)
    total = _sample_count(x, y)
    if requested is None:
        return x, y, {
            "requested_sample_size_total": None,
            "effective_sample_size_total": total,
            "sample_strategy_effective": "all",
        }
    requested = max(0, min(total, int(requested)))
    if requested == total:
        return x, y, {
            "requested_sample_size_total": requested,
            "effective_sample_size_total": total,
            "sample_strategy_effective": "all",
        }

    seed = _safe_int(config.get("benchmark_sample_seed"))
    if seed is None:
        seed = _safe_int(config.get("sample_seed"))
    if seed is None:
        seed = _stable_sample_seed(config)
    benchmark_offset = 1 if config.get("benchmark_sample_seed") is None else 0
    rng = np.random.default_rng(int(seed) + benchmark_offset)
    idx = rng.choice(total, size=requested, replace=False)
    return _take_rows(x, idx), _take_rows(y, idx), {
        "requested_sample_size_total": requested,
        "effective_sample_size_total": requested,
        "sample_strategy_effective": "iid",
        "sample_seed": int(seed) + benchmark_offset,
        "benchmark_sample_seed": config.get("benchmark_sample_seed"),
    }


def _scalar_label_array(y) -> np.ndarray | None:
    if y is None:
        return None
    try:
        arr = np.asarray(y)
    except Exception:
        try:
            arr = np.asarray(y, dtype=object)
        except Exception:
            return None
    if arr.ndim != 1:
        return None
    if arr.size == 0:
        return arr
    first = arr[0]
    if isinstance(first, (list, tuple, dict, np.ndarray)):
        return None
    return arr


def _distribution_summary(x, y, *, meta: Mapping[str, Any], task_family: str | None, hf_task: str | None) -> Any:
    hf_task_norm = str(hf_task or "").strip().lower()
    try:
        if hf_task_norm in {"fill_mask", "masked_lm"}:
            return get_mlm_masked_token_stats(y, ignore_index=int(meta.get("ignore_index", -100)))
        if task_family == "retrieval" or hf_task_norm == "text_image_retrieval":
            return get_retrieval_pair_stats(x)
        if task_family == "vqa" or hf_task_norm == "visual_question_answering":
            return get_vqa_answer_stats(y, ignore_index=int(meta.get("ignore_index", -100)))
        if task_family == "generation" or hf_task_norm in {"token_classification", "causal_lm_generation", "seq2seq_generation"}:
            return get_token_label_stats(
                y,
                ignore_index=int(meta.get("ignore_index", -100)),
                pad_token_id=meta.get("pad_token_id"),
            )
        return get_data_distribution(
            y,
            num_classes=meta.get("num_classes"),
            bins=meta.get("distribution_bins") or 10,
            value_range=meta.get("distribution_range"),
            label_pad_value=int(meta.get("ignore_index", -100)),
        )
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}


def _format_summary_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return "" if value is None else value


def _fetch_hf_metadata(config: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    hf_model_id = config.get("hf_model_id") or meta.get("hf_model_id")
    dataset_family = str(meta.get("dataset_family") or config.get("dataset") or "").strip().lower()
    hf_dataset_id = None
    if dataset_family == "hf" or (config.get("dataset_name") and config.get("hf_model_id")):
        hf_dataset_id = config.get("dataset_name") or meta.get("dataset_name") or config.get("dataset")
    result: dict[str, Any] = {
        "hf_model_id": hf_model_id,
        "hf_dataset_id": hf_dataset_id,
    }
    if not hf_model_id and not hf_dataset_id:
        return result
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        result["hf_metadata_error"] = f"huggingface_hub import failed: {exc}"
        return result

    api = HfApi()
    if hf_model_id:
        try:
            model_info = api.model_info(str(hf_model_id))
            model_payload = _hf_info_payload(model_info)
            result.update(_normalise_hf_info(model_payload, model_info, prefix="hf_model"))
            result["downloads"] = result.get("hf_model_downloads")
            result["likes"] = result.get("hf_model_likes")
            result["pipeline_tag"] = result.get("hf_model_pipeline_tag")
            result["library_name"] = result.get("hf_model_library_name")
            result["license"] = result.get("hf_model_license")
            result["tags"] = result.get("hf_model_tags")
            result["last_modified"] = result.get("hf_model_last_modified")
            size = _extract_hf_model_size(model_payload, model_info)
            result["model_size"] = size
            result["params_count"] = size
        except Exception as exc:
            result["hf_model_metadata_error"] = str(exc)
    if hf_dataset_id:
        try:
            dataset_info = api.dataset_info(str(hf_dataset_id))
            dataset_payload = _hf_info_payload(dataset_info)
            result.update(_normalise_hf_info(dataset_payload, dataset_info, prefix="hf_dataset"))
            if result.get("downloads") is None:
                result["downloads"] = result.get("hf_dataset_downloads")
            if result.get("likes") is None:
                result["likes"] = result.get("hf_dataset_likes")
        except Exception as exc:
            result["hf_dataset_metadata_error"] = str(exc)
    result["hf_service_meta_json"] = json.dumps(result, ensure_ascii=False, default=str)
    return result


def _hf_info_payload(info: Any) -> dict[str, Any]:
    if info is None:
        return {}
    to_dict = getattr(info, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    data = getattr(info, "__dict__", None)
    return dict(data) if isinstance(data, dict) else {}


def _normalise_hf_info(payload: Mapping[str, Any], info: Any, *, prefix: str) -> dict[str, Any]:
    card_data = payload.get("cardData") or payload.get("card_data") or getattr(info, "cardData", None) or {}
    if not isinstance(card_data, Mapping):
        card_data = {}
    last_modified = payload.get("last_modified") if payload.get("last_modified") is not None else getattr(info, "last_modified", None)
    tags = payload.get("tags") if payload.get("tags") is not None else getattr(info, "tags", None)
    return {
        f"{prefix}_downloads": payload.get("downloads") if payload.get("downloads") is not None else getattr(info, "downloads", None),
        f"{prefix}_likes": payload.get("likes") if payload.get("likes") is not None else getattr(info, "likes", None),
        f"{prefix}_pipeline_tag": payload.get("pipeline_tag") or getattr(info, "pipeline_tag", None),
        f"{prefix}_library_name": payload.get("library_name") or getattr(info, "library_name", None),
        f"{prefix}_license": payload.get("license") or card_data.get("license") or getattr(info, "license", None),
        f"{prefix}_tags": tags,
        f"{prefix}_last_modified": _to_iso8601(last_modified),
    }


def _extract_hf_model_size(payload: Mapping[str, Any], info: Any) -> int | None:
    candidates = [
        payload.get("safetensors"),
        getattr(info, "safetensors", None),
        payload.get("cardData"),
        payload.get("card_data"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for key in ("total", "parameters", "params", "params_count", "model_size"):
            value = candidate.get(key)
            parsed = _safe_int(value)
            if parsed is not None and parsed > 0:
                return parsed
    return None


def _to_iso8601(value: Any) -> Any:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass
    return value


def _hf_metadata_metrics(metadata: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    fields = {
        "downloads": ("metadata", None, "neutral"),
        "likes": ("metadata", None, "neutral"),
        "model_size": ("resource", "parameters", "lower_better"),
        "params_count": ("resource", "parameters", "lower_better"),
        "pipeline_tag": ("metadata", None, "neutral"),
        "library_name": ("metadata", None, "neutral"),
        "license": ("metadata", None, "neutral"),
        "tags": ("metadata", None, "neutral"),
        "last_modified": ("metadata", None, "neutral"),
        "hf_model_id": ("metadata", None, "neutral"),
        "hf_dataset_id": ("metadata", None, "neutral"),
    }
    for key, (domain, unit, direction) in fields.items():
        if metadata.get(key) is not None:
            metrics[key] = _metric(metadata.get(key), domain, unit, direction)
    for key, value in metadata.items():
        if key.startswith("hf_model_") or key.startswith("hf_dataset_"):
            metrics[key] = _metric(value, metric_domain(key))
    return metrics


def _performance_alias_metrics(
    *,
    eval_qos: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    training_regime: str,
    train_runtime_s: float,
    eval_runtime_s: float,
    runtime_total_s: float,
    benchmark_samples: int,
) -> dict[str, Any]:
    latency = _first_number(
        eval_qos,
        "inference_latency_s",
        "latency_s_mean",
        "inference_latency_s_mean",
        "eval_latency_s_mean",
    )
    if latency is None:
        ms = _first_number(eval_qos, "inference_latency_ms_mean", "eval_latency_ms_mean")
        latency = None if ms is None else ms / 1000.0
    if latency is None and benchmark_samples > 0 and eval_runtime_s > 0:
        latency = float(eval_runtime_s) / float(benchmark_samples)

    tail_latency = _first_number(eval_qos, "inference_latency_s_p95", "tail_latency", "latency_s_p95")
    if tail_latency is None:
        ms = _first_number(eval_qos, "inference_latency_ms_p95", "eval_latency_ms_p95")
        tail_latency = None if ms is None else ms / 1000.0
    if tail_latency is None:
        tail_latency = latency

    throughput = _first_number(eval_qos, "throughput", "throughput_samples_s", "throughput_eps", "eval_throughput_eps", "examples_per_second")
    if throughput is None and benchmark_samples > 0 and eval_runtime_s > 0:
        throughput = float(benchmark_samples) / float(eval_runtime_s)
    avg_batch_time_ms = _first_number(
        train_metrics if str(training_regime).strip().lower() not in {"inference_only", "inference"} else eval_qos,
        "train_step_latency_ms_mean",
        "eval_latency_ms_mean",
        "inference_latency_ms_mean",
    )

    compute_time_s = float(train_runtime_s or 0.0) + float(eval_runtime_s or 0.0)
    return {
        "latency": _metric(latency, "latency", "s", "lower_better"),
        "tail_latency": _metric(tail_latency, "latency", "s", "lower_better"),
        "inference_latency_s": _metric(latency, "latency", "s", "lower_better"),
        "inference_latency_s_p95": _metric(tail_latency, "latency", "s", "lower_better"),
        "throughput": _metric(throughput, "runtime", "samples/s", "higher_better"),
        "throughput_samples_s": _metric(throughput, "runtime", "samples/s", "higher_better"),
        "avg_batch_time_ms": _metric(avg_batch_time_ms, "runtime", "ms", "lower_better"),
        "compute_time_s": _metric(compute_time_s, "runtime", "s", "lower_better"),
        "runtime_s": _metric(runtime_total_s, "runtime", "s", "lower_better"),
    }


def _first_number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in mapping:
            value = _to_float(mapping.get(key))
            if not math.isnan(value):
                return value
    return None


def _service_perturbation_metrics(model, x_eval, y_eval, *, config: Mapping[str, Any], meta: Mapping[str, Any], task_family: str):
    # ``explainability_enabled`` is retained as the user-facing compatibility
    # switch. Either explicit switch enables the probe; a disabled default must
    # not erase an explicitly enabled legacy manifest field.
    enabled = _config_bool(config.get("enable_perturbation_metrics"), False) or _config_bool(
        config.get("explainability_enabled"), False
    )
    if not enabled:
        return {
            "perturbation_enabled_flag": _metric(False, "explainability"),
            "explainability_supported_flag": _metric(False, "explainability"),
            "faithfulness_alignment_supported_flag": _metric(False, "explainability"),
            "explainability_support_status": _metric("disabled", "explainability"),
        }, None
    log_stage = _config_bool(config.get("perturbation_stage_logging"), True)
    start = time.perf_counter()
    if log_stage:
        debug_print(
            f"[Perturbation] service stage starts | service_id={config.get('service_id') or 'unknown'} "
            f"| task={task_family or 'unknown'} | samples={_sample_count(x_eval, y_eval)}",
            flush=True,
        )
    try:
        values = run_perturbation_stage(
            model,
            x_eval,
            y_eval,
            task_family=task_family,
            hf_task=config.get("hf_task") or meta.get("hf_task"),
            config=dict(config),
            meta=dict(meta),
            service_id=config.get("service_id"),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        values = {
            "perturbation_enabled_flag": True,
            "perturbation_supported_flag": False,
            "explainability_supported_flag": False,
            "faithfulness_alignment_supported_flag": False,
            "explainability_support_status": "unsupported_probe_error",
            "perturbation_error": f"{type(exc).__name__}: {exc}",
            "perturbation_duration_s": elapsed,
        }
    if isinstance(values, dict):
        values.setdefault("perturbation_duration_s", time.perf_counter() - start)
    if log_stage:
        debug_print(
            f"[Perturbation] service stage ends | service_id={config.get('service_id') or 'unknown'} "
            f"| supported={(values or {}).get('perturbation_supported_flag')} "
            f"| samples={(values or {}).get('perturbation_sample_count', 0)} "
            f"| truncated={(values or {}).get('perturbation_truncated_flag', False)} "
            f"| duration_s={(values or {}).get('perturbation_duration_s', time.perf_counter() - start):.2f}",
            flush=True,
        )
    artifact = None
    samples = values.pop("perturbation_samples", None) if isinstance(values, dict) else None
    if samples:
        artifact = {
            "artifact_type": "perturbation_samples",
            "artifact_uri": f"service://{config.get('service_id')}/perturbation_samples",
            "metadata": {"perturbation_samples": samples},
        }
    return {key: _metric(value, metric_domain(key)) for key, value in (values or {}).items()}, artifact


def _log_service_timing(config: Mapping[str, Any], service_id: str, stage: str, elapsed_s: float, *, detail: str | None = None) -> None:
    if not _config_bool(config.get("service_stage_timing_logging"), True):
        return
    model_id = config.get("hf_model_id") or config.get("model_id") or config.get("model_type") or "unknown"
    message = f"[ServiceTiming] service_id={service_id} | model={model_id} | stage={stage} | elapsed_s={elapsed_s:.3f}"
    if detail:
        message = f"{message} | {detail}"
    debug_print(message, flush=True)


def _stage_timing_metrics(stage_timings: Mapping[str, float]) -> dict[str, Any]:
    return {
        key: _metric(float(value), "runtime", "s", "lower_better")
        for key, value in (stage_timings or {}).items()
        if value is not None
    }


def _preprocessor_stage_timings(meta: Mapping[str, Any]) -> dict[str, float]:
    timings = meta.get("preprocess_timing_s") if isinstance(meta, Mapping) else None
    if not isinstance(timings, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in timings.items():
        try:
            out[f"preprocess_{key}_s"] = float(value)
        except Exception:
            continue
    return out


def _load_dataset(name: str, **kwargs):
    global load_dataset
    if load_dataset is None:
        from ..data.master_loader import load_dataset as imported

        load_dataset = imported
    return load_dataset(name, **kwargs)


def _assert_strict_cohort_access(config: Mapping[str, Any]) -> None:
    """Guard prepared strict runs from receiving audit examples.

    The composition validator passes only ``training`` or
    ``catalogue_metric`` IDs into a service execution.  Failing closed here
    prevents a future caller from accidentally evaluating catalogue metrics on
    the held-out audit partition.
    """
    plan = config.get("_strict_cohort_plan")
    if plan is None:
        return
    if not isinstance(plan, Mapping):
        raise ServiceExecutionError("_strict_cohort_plan must be a mapping", failure_stage="cohort_isolation")
    role = str(config.get("_strict_cohort_role") or "").strip().lower()
    if role not in {"training", "catalogue_metric"}:
        raise ServiceExecutionError("strict service runs must declare training or catalogue_metric role", failure_stage="cohort_isolation")
    supplied = {str(item) for item in (config.get("_strict_cohort_example_ids") or [])}
    expected = {str(item) for item in (plan.get(f"{role}_ids") or [])}
    audit = {str(item) for item in (plan.get("audit_ids") or [])}
    if not supplied or supplied != expected:
        raise ServiceExecutionError("strict run example IDs do not exactly match its declared cohort role", failure_stage="cohort_isolation")
    if supplied.intersection(audit):
        raise ServiceExecutionError("audit examples were supplied to a service execution", failure_stage="cohort_isolation")


def _create_model(**kwargs):
    global create_model
    if create_model is None:
        from ..models.builders import create_model as imported

        create_model = imported
    return create_model(**kwargs)


def _service_config(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "mobilebert_head_normalization",
        "max_grad_norm",
        "smoke_test",
        "miopen_find_mode",
        "miopen_process_isolated",
        "training_memory_metadata",
        "evaluation_memory_metadata",
        "no_save_weights",
        "enable_perturbation_metrics",
        "explainability_enabled",
        "update_signature_enabled",
        "precision_policy",
        "resource_tier",
        "batch_size",
        "learning_rate",
        "training_epochs",
        "optimizer",
        "weight_decay",
        "momentum",
        "warmup_ratio",
        "gradient_accumulation_steps",
        "sample_size",
        "sample_seed",
        "max_length",
        "device",
        "mixed_precision",
        "precision_type",
        "checkpoint_patch_compression",
        "checkpoint_patch_compression_level",
        "num_workers",
        "timeout_s",
        "max_train_time_s",
        "max_eval_time_s",
    )
    return {key: config.get(key) for key in keys if config.get(key) is not None}


def _registry_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "resource_tier",
        "model_resource_tier",
        "dataset_resource_tier",
        "fit_quality_score",
        "service_source",
        "model_role",
        "fit_decision",
        "fit_reason",
        "realism_score",
        "domain_alignment",
        "dataset_hint",
        "hf_pipeline_tag",
        "hf_downloads",
        "hf_likes",
        "hf_model_id",
        "hf_dataset_id",
        "downloads",
        "likes",
        "model_size",
        "params_count",
        "pipeline_tag",
        "library_name",
        "license",
        "tags",
        "last_modified",
        "hf_model_downloads",
        "hf_model_likes",
        "hf_model_pipeline_tag",
        "hf_model_library_name",
        "hf_model_license",
        "hf_model_tags",
        "hf_model_last_modified",
        "hf_dataset_downloads",
        "hf_dataset_likes",
        "hf_dataset_license",
        "hf_dataset_tags",
        "hf_dataset_last_modified",
        "hf_author",
        "hf_url",
        "hf_service_meta_json",
    )
    return {key: config.get(key) for key in keys if config.get(key) is not None}


def _count_model_params(model) -> int:
    for candidate in (model, getattr(model, "core", None), getattr(model, "model", None)):
        if candidate is None:
            continue
        count_params = getattr(candidate, "count_params", None)
        if callable(count_params):
            try:
                count = int(count_params())
                if count >= 0:
                    return count
            except Exception:
                pass
        parameters = getattr(candidate, "parameters", None)
        if callable(parameters):
            try:
                return int(sum(p.numel() for p in parameters()))
            except Exception:
                pass
    return 0


def _resource_metrics(*, runtime_total_s, workload_runtime_s, train_metrics, eval_qos, model_size, usage) -> dict[str, Any]:
    memory_mb = usage.memory_used_mb or usage.peak_host_ram_mb or 0.0
    gpu_mb = usage.gpu_memory_used_mb or usage.peak_vram_mb or 0.0
    comm_bytes = float(eval_qos.get("comm_bytes", 0.0) or 0.0)
    raw_resource_cost = (
        float(runtime_total_s or 0.0)
        + float(workload_runtime_s or 0.0)
        + float(memory_mb or 0.0) / 1024.0
        + float(gpu_mb or 0.0) / 1024.0
        + float(comm_bytes or 0.0) / 1073741824.0
        + (float(model_size or 0.0) / 1_000_000.0 if model_size else 0.0)
    )
    metrics = {
        "cpu_time_s": _metric(usage.cpu_time_s, "resource", "s", "lower_better"),
        "cpu_utilization": _metric(usage.cpu_utilization, "resource", "percent", "lower_better"),
        "memory_used_mb": _metric(usage.memory_used_mb, "resource", "MB", "lower_better"),
        "memory_utilization": _metric(usage.memory_utilization, "resource", "percent", "lower_better"),
        "gpu_utilization": _metric(usage.gpu_utilization, "resource", "percent", "lower_better"),
        "gpu_memory_used_mb": _metric(usage.gpu_memory_used_mb, "resource", "MB", "lower_better"),
        "peak_vram_mb": _metric(usage.peak_vram_mb, "resource", "MB", "lower_better"),
        "peak_gpu_memory_mb": _metric(usage.peak_vram_mb, "resource", "MB", "lower_better"),
        "avg_vram_mb": _metric(usage.avg_vram_mb, "resource", "MB", "lower_better"),
        "peak_host_ram_mb": _metric(usage.peak_host_ram_mb, "resource", "MB", "lower_better"),
        "avg_host_ram_mb": _metric(usage.avg_host_ram_mb, "resource", "MB", "lower_better"),
        "raw_resource_cost": _metric(raw_resource_cost, "cost", None, "lower_better"),
        "resource_cost_score": _metric(1.0 / (1.0 + raw_resource_cost), "cost", "score", "higher_better"),
    }
    return metrics


def _explainability_metrics(model, config: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    if not _config_bool(config.get("explainability_enabled"), True):
        return {"explainability_supported_flag": _metric(False, "explainability"), "explainability_score": _metric(0.0, "explainability")}
    declared = config.get("explainability_method") or _nested_get(meta, "explainability", "preferred_methods")
    supported = _nested_get(meta, "explainability", "supported")
    if supported is None:
        supported = _has_importance_signal(model)
    score = _importance_proxy_score(model) if supported else 0.0
    return {
        "explainability_supported_flag": _metric(bool(supported), "explainability", direction="higher_better"),
        "explainability_method": _metric(declared[0] if isinstance(declared, list) and declared else declared or "metadata_or_importance_proxy", "explainability"),
        "explainability_score": _metric(score, "explainability", "score", "higher_better"),
    }


def _has_importance_signal(model) -> bool:
    return any(hasattr(model, attr) for attr in ("feature_importances_", "coef_", "cluster_centers_", "get_weights"))


def _importance_proxy_score(model) -> float:
    arrays = []
    for attr in ("feature_importances_", "coef_", "cluster_centers_"):
        if hasattr(model, attr):
            try:
                arrays.append(np.asarray(getattr(model, attr), dtype="float64").reshape(-1))
            except Exception:
                pass
    get_weights = getattr(model, "get_weights", None)
    if callable(get_weights):
        try:
            arrays.extend(np.asarray(w, dtype="float64").reshape(-1) for w in get_weights())
        except Exception:
            pass
    arrays = [arr for arr in arrays if arr.size]
    if not arrays:
        return 0.0
    values = np.abs(np.concatenate(arrays))
    values = values[np.isfinite(values)]
    if values.size == 0 or float(values.sum()) <= 0.0:
        return 0.0
    probs = values / float(values.sum())
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(values.size)) if values.size > 1 else 1.0
    return float(np.clip(1.0 - (entropy / max(max_entropy, 1e-12)), 0.0, 1.0))


def _reliability_metrics(*, status: str, eval_qos: Mapping[str, Any]) -> dict[str, Any]:
    failed = status != "completed"
    reliability_score = 0.0 if failed else 1.0
    if eval_qos.get("label_space_mismatch"):
        reliability_score = min(reliability_score, 0.75)
    if eval_qos.get("truncation_rate") is not None:
        try:
            reliability_score = min(reliability_score, max(0.0, 1.0 - float(eval_qos["truncation_rate"])))
        except Exception:
            pass
    return {
        "service_success_flag": _metric(not failed, "reliability", direction="higher_better"),
        "reliability_score": _metric(reliability_score, "reliability", "score", "higher_better"),
    }


def _specify_metric_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (values or {}).items():
        if value is None:
            continue
        normalized = str(key).strip().lower()
        if normalized.endswith("_ms_mean"):
            out[normalized.replace("_ms_mean", "_s_mean")] = _metric(_to_float(value) / 1000.0, "latency", "s", "lower_better")
        elif normalized.endswith("_ms_p95"):
            out[normalized.replace("_ms_p95", "_s_p95")] = _metric(_to_float(value) / 1000.0, "latency", "s", "lower_better")
        elif normalized in {"mse", "rmse", "mae"}:
            out[normalized] = _metric(value, "quality", direction="lower_better")
        else:
            out[normalized] = _metric(value, metric_domain(normalized))
    return out


def _metric(value: Any, domain: str, unit: str | None = None, direction: str = "neutral") -> dict[str, Any]:
    return {"value": value, "domain": domain, "unit": unit, "direction": direction}


def _quality_direction(metric_name: str) -> str:
    return "lower_better" if str(metric_name).lower() in {"loss", "mse", "rmse", "mae", "perplexity"} else "higher_better"


def _resolved_task_type(config: Mapping[str, Any], meta: Mapping[str, Any]) -> str | None:
    hf_task = meta.get("hf_task") or config.get("hf_task")
    hf_task_norm = str(hf_task or "").strip().lower().replace("-", "_")
    meta_task_type = meta.get("task_type")
    meta_task_text = None if _is_blank(meta_task_type) else str(meta_task_type).strip().lower()
    if hf_task_norm == "sentence_similarity" and (
        bool(meta.get("is_regression")) or meta_task_text == "regression"
    ):
        return "regression"
    if meta_task_text and canonical_task_family(meta_task_text, hf_task) != "unknown":
        return "classification" if canonical_task_family(meta_task_text, hf_task) == "classification" else meta_task_text
    config_task_type = config.get("task_type")
    if _is_blank(config_task_type):
        return None
    config_task_text = str(config_task_type).strip().lower()
    config_task_family = canonical_task_family(config_task_text, hf_task)
    if hf_task_norm == "sentence_similarity" and config_task_family == "regression":
        return "regression"
    return "classification" if config_task_family == "classification" else config_task_text


def _validate_service_compatibility(
    *,
    task_family: str,
    training_regime: str,
    hf_model_id: Any,
    batch_size: int | None,
    train_samples: int,
    benchmark_samples: int,
    task_contract: Mapping[str, Any] | None = None,
) -> None:
    if training_regime in {"inference_only", "inference"}:
        return
    minimums = {
        "detection": (32, 16),
        "segmentation": (32, 16),
    }
    if task_contract:
        from ..tasks import TaskContract
        contract = TaskContract.from_dict(task_contract)
        minimums[task_family] = (int(contract.training.get("minimum_examples", 1)),
                                int(contract.evaluation.get("minimum_examples", 1)))
    if task_family in minimums:
        min_train, min_benchmark = minimums[task_family]
        if train_samples < min_train or benchmark_samples < min_benchmark:
            raise ServiceExecutionError(
                (
                    f"{task_family} finetune services require at least {min_train} train samples "
                    f"and {min_benchmark} benchmark samples; got train={train_samples}, "
                    f"benchmark={benchmark_samples}"
                ),
                failure_stage="service_validation",
            )
    model_id = str(hf_model_id or "").strip().lower()
    if task_family == "segmentation" and model_id.startswith("openmmlab/upernet-") and int(batch_size or 0) < 2:
        raise ServiceExecutionError(
            f"{hf_model_id} requires batch_size >= 2 for finetune segmentation runs",
            failure_stage="service_validation",
        )


def _required_secondary_metrics(task_family: str, hf_task: str | None, primary_name: str | None) -> tuple[str, ...]:
    metric = str(primary_name or "").strip().lower()
    hf_task_norm = str(hf_task or "").strip().lower().replace("-", "_")
    if hf_task_norm == "sentence_similarity" and task_family == "regression":
        return ("spearman",)
    if hf_task_norm == "sentence_similarity" and task_family == "classification":
        return ("f1",)
    if task_family == "detection" and metric == "map":
        return ("map@0.5",)
    if task_family == "segmentation" and metric == "iou":
        return ("dice",)
    if task_family == "generation" and metric == "loss":
        return ("perplexity",)
    return ()


def _is_finite_numeric_metric(value: Any) -> bool:
    try:
        numeric = float(value)
    except Exception:
        return False
    return bool(np.isfinite(numeric))


def _is_sentence_similarity_metric_fallback(
    *,
    hf_task: str | None,
    primary_name: str | None,
    primary_value: Any,
    secondary_name: str | None,
    secondary_value: Any,
    metric_score: Any,
) -> bool:
    hf_task_norm = str(hf_task or "").strip().lower().replace("-", "_")
    primary_norm = str(primary_name or "").strip().lower()
    secondary_norm = str(secondary_name or "").strip().lower()
    return (
        hf_task_norm == "sentence_similarity"
        and primary_norm == "pearson"
        and secondary_norm == "spearman"
        and not _is_finite_numeric_metric(primary_value)
        and _is_finite_numeric_metric(secondary_value)
        and _is_finite_numeric_metric(metric_score)
    )


def _service_metric_score_value(
    *,
    task_family: str,
    hf_task: str | None,
    primary_name: str | None,
    primary_value: Any,
    secondary_name: str | None,
    secondary_value: Any,
) -> float:
    score = metric_score_value(task_family, primary_name, primary_value)
    if _is_finite_numeric_metric(score):
        return score
    hf_task_norm = str(hf_task or "").strip().lower().replace("-", "_")
    if (
        hf_task_norm == "sentence_similarity"
        and str(primary_name or "").strip().lower() == "pearson"
        and str(secondary_name or "").strip().lower() == "spearman"
        and _is_finite_numeric_metric(secondary_value)
    ):
        return metric_score_value(task_family, primary_name, secondary_value)
    return score


def _validate_service_metrics(
    *,
    task_family: str,
    hf_task: str | None,
    primary_name: str | None,
    primary_value: Any,
    secondary_name: str | None,
    secondary_value: Any,
    metric_score: Any,
    loss_value: Any = None,
) -> None:
    if task_family == "generation" and not _is_finite_numeric_metric(loss_value):
        raise ServiceExecutionError(
            f"nonfinite_evaluation_loss: supervised generation loss is missing or non-finite: {loss_value!r}",
            failure_stage="metric_validation",
        )
    if not _is_finite_numeric_metric(primary_value):
        if _is_sentence_similarity_metric_fallback(
            hf_task=hf_task,
            primary_name=primary_name,
            primary_value=primary_value,
            secondary_name=secondary_name,
            secondary_value=secondary_value,
            metric_score=metric_score,
        ):
            return
        raise ServiceExecutionError(
            f"Primary metric '{primary_name}' is not a finite numeric value: {primary_value!r}",
            failure_stage="metric_validation",
        )
    if not _is_finite_numeric_metric(metric_score):
        raise ServiceExecutionError(
            f"metric_score is not a finite numeric value for primary metric '{primary_name}': {metric_score!r}",
            failure_stage="metric_validation",
        )
    secondary_metric_values = {}
    if secondary_name:
        secondary_metric_values[str(secondary_name).strip().lower()] = secondary_value
    for required_name in _required_secondary_metrics(task_family, hf_task, primary_name):
        required_value = secondary_metric_values.get(required_name)
        if not _is_finite_numeric_metric(required_value):
            raise ServiceExecutionError(
                f"Required secondary metric '{required_name}' is not a finite numeric value: {required_value!r}",
                failure_stage="metric_validation",
            )


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _config_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _snapshot_model_weights(model: Any) -> Any | None:
    get_weights = getattr(model, "get_weights", None)
    if not callable(get_weights):
        return None
    try:
        weights = get_weights()
    except Exception:
        return None
    return _clone_weight_payload(weights)


def _clone_weight_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone_weight_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_weight_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_weight_payload(item) for item in value)
    try:
        return np.array(value, copy=True)
    except Exception:
        return value


def _update_signature_output_dir(config: Mapping[str, Any]) -> str:
    configured = config.get("update_signature_dir")
    if configured:
        return str(configured)
    db_path = str(config.get("db_path") or CONFIG.get("db_path") or "")
    db_folder = os.path.dirname(db_path)
    if db_folder:
        return os.path.join(db_folder, "update_signatures")
    return os.path.join("outputs", "update_signatures")


def _provenance_artifact_root(config: Mapping[str, Any]) -> str:
    configured = config.get("provenance_artifact_dir")
    if configured:
        return str(configured)
    db_path = str(config.get("db_path") or CONFIG.get("db_path") or "")
    folder = os.path.dirname(db_path)
    return os.path.join(folder or "outputs", "service_artifacts")


def _checkpoint_compression(config: Mapping[str, Any]) -> CompressionSpec:
    algorithm = str(config.get("checkpoint_patch_compression") or "zlib").strip().lower()
    level = _safe_int(config.get("checkpoint_patch_compression_level"))
    return CompressionSpec(algorithm=algorithm, level=1 if level is None else level)


def _checkpoint_fallback_details(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, UnsupportedTensorStateError):
        return exc.as_fallback_signal()
    return {
        "required": True,
        "policy": "full_object_v1",
        "reason_code": "exact_patch_round_trip_failed",
        "detail": f"{type(exc).__name__}: {exc}",
    }


def _artifact_reference(artifact: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not artifact:
        return None
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), Mapping) else {}
    return {
        "artifact_type": artifact.get("artifact_type"),
        "artifact_uri": artifact.get("artifact_uri"),
        "artifact_sha256": metadata.get("sha256"),
        "content_address": metadata.get("content_address"),
        "size_bytes": metadata.get("size_bytes"),
        "format": metadata.get("format"),
        "role": metadata.get("role"),
        "serialization": metadata.get("serialization"),
        "immutable": metadata.get("immutable"),
    }


def _compression_settings(compression: CompressionSpec) -> dict[str, Any]:
    return {
        key: value
        for key, value in compression.as_dict().items()
        if key != "algorithm"
    }


def _validate_reduced_checkpoint(service, model, source, reloaded, x, y, *, task_family, primary_name):
    """Compare both representations on the same deterministic catalogue subset."""
    from ..data.fedavg_partition import _take_rows
    count = min(_sample_count(x, y), int(service.config.get("checkpoint_validation_samples", 256)))
    if count < 1:
        raise ValueError("checkpoint_validation_samples must be positive")
    tolerance = float(service.config.get("checkpoint_quality_tolerance", 0.01))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("checkpoint_quality_tolerance must be finite and nonnegative")
    seed = int(service.config.get("seed", 42))
    indexes = np.random.default_rng(seed).choice(_sample_count(x, y), count, replace=False)
    subset = (_take_rows(x, indexes), _take_rows(y, indexes))
    try:
        _, reference, _, _ = service._evaluate_model(model, *subset, inference_only=False,
            task_family=task_family, enforce_time_limit=False)
        restore_model_tensor_state(model, reloaded, verify=True)
        _, candidate, _, _ = service._evaluate_model(model, *subset, inference_only=False,
            task_family=task_family, enforce_time_limit=False)
    finally:
        restore_model_tensor_state(model, source, verify=True)
    lower = primary_name in {"mae", "mse", "rmse", "loss", "perplexity"}
    permitted = tolerance * max(abs(float(reference)), 1e-6) if lower else tolerance
    degradation = float(candidate) - float(reference) if lower else float(reference) - float(candidate)
    return {"status": "passed" if math.isfinite(degradation) and degradation <= permitted else "failed",
        "scope": "deterministic catalogue subset; original catalogue metrics unchanged",
        "sample_count": count, "sample_index_hash": sha256_json(indexes.tolist()), "seed": seed,
        "metric": primary_name, "reference": float(reference), "reloaded": float(candidate),
        "degradation": degradation, "permitted_degradation": permitted,
        "tolerance_kind": "relative" if lower else "absolute"}


def _checkpoint_storage_documents(
    *,
    service_id: str,
    lineage_id: str,
    parameter_schema_hash: str,
    base_artifact: Mapping[str, Any],
    lineage_base: LineageBaseResult | None,
    final_artifact: Mapping[str, Any],
    service_patch: ServicePatchResult | None,
    fallback_target_state: Any,
    support_artifacts: list[dict[str, Any]],
    provenance_artifacts: Mapping[str, Mapping[str, Any]],
    fallback: Mapping[str, Any] | None,
    compression: CompressionSpec,
    provenance_root: str,
    native_final_round_trip_verified: bool,
    parent_global_round: int | None = None,
    explicit_parent_state_hash: str | None = None,
    reduced_checkpoint=None,
) -> dict[str, Any]:
    del provenance_root  # Artefact references already contain their resolved store paths.
    exact_patch = service_patch is not None
    measurements = dict(service_patch.measurements) if exact_patch else {}
    target_state_hash = (
        service_patch.target_state_sha256
        if service_patch is not None
        else getattr(fallback_target_state, "sha256", None)
    )
    document = {
        "format": "mlaas-checkpoint-reconstruction-v1",
        "storage_policy_version": (
            "fedavg_round_base_exact_patch_v1"
            if exact_patch and parent_global_round is not None
            else ("shared_base_exact_patch_v1" if exact_patch else "full_object_v1")
        ),
        "storage_mode": "shared_base_exact_patch" if exact_patch else "full_object",
        "service_id": service_id,
        "lineage_id": lineage_id,
        "tensor_state_hash_specification": TENSOR_STATE_FORMAT,
        "parameter_schema": {
            "hash": parameter_schema_hash,
            "ownership": "separate_generator_provenance",
        },
        "base": {
            "artifact": _artifact_reference(base_artifact),
            "tensor_state_sha256": (
                service_patch.base.content_sha256
                if service_patch is not None
                else (
                    explicit_parent_state_hash
                    or (lineage_base.tensor_state_sha256 if lineage_base is not None else None)
                )
            ),
            "global_round": parent_global_round,
        },
        "target": {
            "artifact": _artifact_reference(final_artifact),
            "tensor_state_sha256": target_state_hash,
        },
        "patch_semantics": (
            {
                "format": TENSOR_PATCH_FORMAT,
                "operations": (
                    "ordered changed-tensor byte XOR or REPLACE; omitted tensors inherit "
                    + ("the round-global base" if parent_global_round is not None else "the lineage base")
                ),
                "mode": (
                    "stored_patch" if service_patch.patch is not None else "base_reference_only"
                ),
                "content_sha256": (
                    service_patch.patch.content_sha256
                    if service_patch.patch is not None
                    else None
                ),
            }
            if service_patch is not None
            else None
        ),
        "storage_encoding": {
            "compression": compression.as_dict() if exact_patch else {"algorithm": "native"},
            "independent_of_patch_semantics": True,
        },
        "operation_selection": dict(service_patch.operation_selection) if exact_patch else None,
        "support_artifacts": [
            reference
            for reference in (_artifact_reference(item) for item in support_artifacts)
            if reference is not None
        ],
        "provenance_artifacts": {
            name: reference
            for name, reference in (
                (name, _artifact_reference(artifact))
                for name, artifact in sorted(provenance_artifacts.items())
            )
            if reference is not None
        },
        "storage_measurements": measurements,
        "measurement_scope": (
            "model-state objects and required reference manifests; common support artefacts "
            "are recorded separately and excluded from both alternatives"
        ),
        "round_trip_verified": bool(
            service_patch is not None or native_final_round_trip_verified
        ),
        "fallback": dict(fallback) if fallback else None,
    }
    if reduced_checkpoint is not None:
        reduced = reduced_checkpoint.document
        target_state_hash = reduced["reloaded_state_sha256"]
        document.update(storage_policy_version="reduced_precision_v1", storage_mode="full_object",
            round_trip_verified=True, source_state_sha256=reduced["source_state_sha256"],
            lossy=True, quality_validation=reduced["quality_validation"],
            storage_encoding={"container": "safetensors", "representation": reduced["representation"],
                              "compression": reduced["compression"]})
        document["target"]["tensor_state_sha256"] = target_state_hash
        document["target"]["source_tensor_state_sha256"] = reduced["source_state_sha256"]
    return {
        "reconstruction_manifest": document,
        "provenance": {
            "storage_policy_version": document["storage_policy_version"],
            "storage_mode": document["storage_mode"],
            "tensor_state_hash_specification": TENSOR_STATE_FORMAT,
            "parameter_schema_hash": parameter_schema_hash,
            "base_state_sha256": document["base"]["tensor_state_sha256"],
            "target_state_sha256": target_state_hash,
            "patch_semantics": document["patch_semantics"],
            "storage_encoding": document["storage_encoding"],
            "operation_selection": document["operation_selection"],
            "storage_measurements": measurements,
            "fallback": document["fallback"],
            **({"source_state_sha256": reduced_checkpoint.document["source_state_sha256"],
                "lossy": True, "quality_validation": reduced_checkpoint.document["quality_validation"]} if reduced_checkpoint else {}),
        },
    }


def _lineage_checkpoint_storage_row(
    *,
    lineage_id: str,
    parameter_schema_hash: str,
    base_artifact: Mapping[str, Any],
    lineage_base: LineageBaseResult | None,
    native_base_round_trip_verified: bool,
    fallback: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = base_artifact.get("metadata") if isinstance(base_artifact.get("metadata"), Mapping) else {}
    if not metadata.get("sha256"):
        raise ServiceExecutionError(
            "initial checkpoint is missing its artefact hash",
            failure_stage="checkpoint_persistence",
        )
    if lineage_base is not None:
        compression = lineage_base.base.compression
        return {
            "lineage_id": lineage_id,
            "storage_policy_version": "shared_base_exact_patch_v1",
            "state_hash_spec_version": TENSOR_STATE_FORMAT,
            "parameter_schema_hash": parameter_schema_hash,
            "base_state_sha256": lineage_base.tensor_state_sha256,
            "base_artifact_ref": str(lineage_base.manifest_path),
            "base_artifact_sha256": metadata.get("sha256"),
            "base_compression_codec": compression.algorithm,
            "base_compression_settings_json": _compression_settings(compression),
            "base_logical_bytes": lineage_base.base.logical_size_bytes,
            "base_physical_bytes": (
                lineage_base.base.physical_size_bytes + lineage_base.manifest_path.stat().st_size
            ),
            "metadata_json": {
                "created": lineage_base.created,
                "restored_existing_base": lineage_base.restored_existing_base,
                "semantic_state_and_parameter_schema_are_separate": True,
            },
        }
    return {
        "lineage_id": lineage_id,
        "storage_policy_version": "full_object_v1",
        "state_hash_spec_version": TENSOR_STATE_FORMAT,
        "parameter_schema_hash": parameter_schema_hash,
        "base_state_sha256": (
            lineage_base.tensor_state_sha256 if lineage_base is not None else None
        ),
        "base_artifact_ref": str(base_artifact.get("artifact_uri")),
        "base_artifact_sha256": metadata.get("sha256"),
        "base_compression_codec": "native",
        "base_compression_settings_json": {},
        "base_logical_bytes": None,
        "base_physical_bytes": _safe_int(metadata.get("size_bytes")),
        "metadata_json": {
            "fallback": dict(fallback) if fallback else None,
            "round_trip_verified": bool(native_base_round_trip_verified),
            "semantic_state_and_parameter_schema_are_separate": True,
        },
    }


def _service_checkpoint_storage_row(
    *,
    service_id: str,
    lineage_id: str,
    parameter_schema_hash: str,
    final_artifact: Mapping[str, Any],
    service_patch: ServicePatchResult | None,
    lineage_base: LineageBaseResult | None,
    fallback_target_state: Any,
    reconstruction_artifact: Mapping[str, Any],
    fallback: Mapping[str, Any] | None,
    native_final_round_trip_verified: bool,
    parent_global_round: int | None = None,
    explicit_parent_state_hash: str | None = None,
    explicit_parent_state_ref: str | None = None,
    reduced_checkpoint=None,
) -> dict[str, Any]:
    final_metadata = final_artifact.get("metadata") if isinstance(final_artifact.get("metadata"), Mapping) else {}
    reconstruction_metadata = (
        reconstruction_artifact.get("metadata")
        if isinstance(reconstruction_artifact.get("metadata"), Mapping)
        else {}
    )
    reconstruction_size = int(reconstruction_metadata.get("size_bytes") or 0)
    if not reconstruction_metadata.get("sha256"):
        raise ServiceExecutionError(
            "checkpoint reconstruction manifest is missing its artefact hash",
            failure_stage="checkpoint_persistence",
        )
    if reduced_checkpoint is not None:
        reduced = reduced_checkpoint.document
        reference_bytes = reconstruction_size + reduced_checkpoint.manifest_path.stat().st_size
        return {
            "service_id": service_id, "lineage_id": lineage_id,
            "storage_policy_version": "reduced_precision_v1", "storage_mode": "full_object",
            "state_hash_spec_version": TENSOR_STATE_FORMAT, "parameter_schema_hash": parameter_schema_hash,
            "base_state_sha256": reduced["base_state_sha256"], "target_state_sha256": reduced["reloaded_state_sha256"],
            "parent_global_round": parent_global_round,
            "explicit_base_state_ref": explicit_parent_state_ref or str(lineage_base.base.manifest_path),
            "full_artifact_ref": str(reduced_checkpoint.payload_path), "full_artifact_sha256": reduced["payload_sha256"],
            "reconstruction_artifact_ref": str(reconstruction_artifact["artifact_uri"]),
            "reconstruction_artifact_sha256": reconstruction_metadata["sha256"],
            "target_logical_bytes": reduced["source_tensor_bytes"],
            "full_checkpoint_physical_bytes": (reduced["full_comparison_payload_bytes"] + reference_bytes
                if reduced.get("full_comparison_payload_bytes") is not None else None),
            "reconstruction_physical_bytes": reference_bytes,
            "selected_physical_bytes": reduced["payload_bytes"] + reference_bytes,
            "round_trip_status": "passed", "round_trip_verified_at": datetime.now(timezone.utc).isoformat(),
            "metadata_json": {"source_state_sha256": reduced["source_state_sha256"],
                "reloaded_state_sha256": reduced["reloaded_state_sha256"], "lossy": True,
                "representation": reduced["representation_version"], "compression": reduced["compression"],
                "verification_scope": "stored representation and sampled performance; not source bit equality",
                "weight_error": reduced["weight_error"], "quality_validation": reduced["quality_validation"]},
        }
    if service_patch is not None:
        measurements = dict(service_patch.measurements)
        compression_data = dict(measurements.get("compression") or {})
        patch_reference_size = service_patch.manifest_path.stat().st_size
        reference_bytes = patch_reference_size + reconstruction_size
        patch_physical = int(measurements["patch_physical_size_bytes"])
        full_physical = measurements["full_state_baseline_physical_size_bytes"]
        selected_physical = patch_physical + reference_bytes
        full_comparison_physical = int(full_physical) + reference_bytes if full_physical is not None else None
        return {
            "service_id": service_id,
            "lineage_id": lineage_id,
            "storage_policy_version": (
                "fedavg_round_base_exact_patch_v1"
                if parent_global_round is not None
                else "shared_base_exact_patch_v1"
            ),
            "storage_mode": "shared_base_exact_patch",
            "state_hash_spec_version": TENSOR_STATE_FORMAT,
            "parameter_schema_hash": parameter_schema_hash,
            "base_state_sha256": service_patch.base.content_sha256,
            "parent_global_round": parent_global_round,
            "explicit_base_state_ref": str(service_patch.base.manifest_path),
            "target_state_sha256": service_patch.target_state_sha256,
            "patch_semantics": (
                "ordered_changed_tensor_byte_xor_or_replace_with_round_global_base_inheritance"
                if parent_global_round is not None
                else "ordered_changed_tensor_byte_xor_or_replace_with_base_inheritance"
            ),
            "patch_semantics_version": TENSOR_PATCH_FORMAT,
            "patch_compression_codec": compression_data.get("algorithm"),
            "patch_compression_settings_json": {
                key: value for key, value in compression_data.items() if key != "algorithm"
            },
            "patch_artifact_ref": (
                str(service_patch.patch.manifest_path)
                if service_patch.patch is not None
                else None
            ),
            "patch_artifact_sha256": (
                sha256_file(service_patch.patch.manifest_path)
                if service_patch.patch is not None
                else None
            ),
            "full_artifact_ref": None,
            "full_artifact_sha256": None,
            "reconstruction_artifact_ref": str(reconstruction_artifact["artifact_uri"]),
            "reconstruction_artifact_sha256": reconstruction_metadata.get("sha256"),
            "target_logical_bytes": measurements["full_state_baseline_logical_size_bytes"],
            "full_checkpoint_physical_bytes": full_comparison_physical,
            "patch_logical_bytes": measurements["patch_logical_size_bytes"],
            "patch_physical_bytes": patch_physical,
            "reconstruction_physical_bytes": reference_bytes,
            "selected_physical_bytes": selected_physical,
            "round_trip_status": "passed",
            "round_trip_verified_at": datetime.now(timezone.utc).isoformat(),
            "metadata_json": {
                "semantic_patch_content_sha256": (
                    service_patch.patch.content_sha256
                    if service_patch.patch is not None
                    else None
                ),
                "patch_mode": (
                    "stored_patch" if service_patch.patch is not None else "base_reference_only"
                ),
                "operation_counts": dict(service_patch.operation_counts),
                "operation_selection": dict(service_patch.operation_selection),
                "raw_measurements": measurements,
                "common_support_artifacts_excluded_from_both_alternatives": True,
            },
        }
    final_size = int(final_metadata.get("size_bytes") or 0)
    selected_physical = final_size + reconstruction_size
    target_hash = getattr(fallback_target_state, "sha256", None)
    return {
        "service_id": service_id,
        "lineage_id": lineage_id,
        "storage_policy_version": "full_object_v1",
        "storage_mode": "full_object",
        "state_hash_spec_version": TENSOR_STATE_FORMAT,
        "parameter_schema_hash": parameter_schema_hash,
        "base_state_sha256": (
            explicit_parent_state_hash
            or (lineage_base.tensor_state_sha256 if lineage_base is not None else None)
        ),
        "parent_global_round": parent_global_round,
        "explicit_base_state_ref": explicit_parent_state_ref,
        "target_state_sha256": target_hash,
        "patch_semantics": None,
        "patch_semantics_version": None,
        "patch_compression_codec": None,
        "patch_compression_settings_json": None,
        "patch_artifact_ref": None,
        "patch_artifact_sha256": None,
        "full_artifact_ref": str(final_artifact.get("artifact_uri")),
        "full_artifact_sha256": final_metadata.get("sha256"),
        "reconstruction_artifact_ref": str(reconstruction_artifact["artifact_uri"]),
        "reconstruction_artifact_sha256": reconstruction_metadata.get("sha256"),
        "target_logical_bytes": (
            getattr(fallback_target_state, "raw_tensor_size_bytes", None)
            if fallback_target_state is not None
            else None
        ),
        "full_checkpoint_physical_bytes": selected_physical,
        "patch_logical_bytes": None,
        "patch_physical_bytes": None,
        "reconstruction_physical_bytes": reconstruction_size,
        "selected_physical_bytes": selected_physical,
        "round_trip_status": (
            "passed" if native_final_round_trip_verified else "not_run"
        ),
        "round_trip_verified_at": (
            datetime.now(timezone.utc).isoformat()
            if native_final_round_trip_verified
            else None
        ),
        "round_trip_error": None,
        "metadata_json": {
            "fallback": dict(fallback) if fallback else None,
            "common_support_artifacts_excluded_from_both_alternatives": True,
        },
    }


def _resolve_hf_revisions(config: dict[str, Any]) -> None:
    """Resolve mutable Hub names once; failures remain visible in provenance."""
    model_id = config.get("hf_model_id") or config.get("model_id")
    dataset_id = config.get("dataset_name") if str(config.get("dataset") or "").lower() in {"hf", "huggingface"} else None
    if not model_id and not dataset_id:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
    except Exception as exc:  # Network/auth failures should not prevent a service run.
        config.setdefault("hf_revision_resolution_error", f"{type(exc).__name__}: {exc}")
        return

    failures: list[str] = []
    identities = (
        ("model", model_id, "hf_model_revision", api.model_info),
        ("dataset", dataset_id, "hf_dataset_revision", api.dataset_info),
    )
    for kind, repo_id, revision_key, resolver in identities:
        if not repo_id:
            continue
        requested = config.get(revision_key)
        if _is_hf_commit_sha(requested):
            continue
        try:
            kwargs = {"revision": str(requested)} if not _is_blank(requested) else {}
            resolved = getattr(resolver(str(repo_id), **kwargs), "sha", None)
            if not resolved:
                raise RuntimeError("Hub response did not include a commit SHA")
            if requested is not None:
                config[f"{revision_key}_requested"] = requested
            config[revision_key] = resolved
        except Exception as exc:  # Network/auth failures remain explicit provenance.
            failures.append(f"{kind} {repo_id}: {type(exc).__name__}: {exc}")
    if failures:
        config["hf_revision_resolution_error"] = "; ".join(failures)


def _is_hf_commit_sha(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return len(text) in {40, 64} and all(character in "0123456789abcdef" for character in text)


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _jsonable(value: Any) -> Any:
    from ..diagnostic_json import diagnostic_value
    return diagnostic_value(value)
