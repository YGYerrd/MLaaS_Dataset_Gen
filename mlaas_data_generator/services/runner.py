from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
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
from ..models.train_eval import evaluate_model, train_local_model
from ..federated.update_signature import compute_and_store_update_signature
from ..storage.writer import make_writer
from .perturbation import run_perturbation_stage
from .system_metrics import ResourceTracker, capture_hardware_snapshot
from .taxonomy import canonical_label_format, canonical_metric_names, canonical_task_family, metric_domain, metric_score_value

load_dataset = None
create_model = None


@dataclass
class ServiceExecutionResult:
    service_id: str
    status: str
    db_path: str
    metrics: dict[str, Any]
    error: str | None = None


class ServiceExecutionError(RuntimeError):
    def __init__(self, message: str, *, failure_stage: str = "service_execution"):
        super().__init__(message)
        self.failure_stage = failure_stage


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


class ServiceRunner:
    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(CONFIG)
        self.config.update(dict(config or {}))
        self.config["device"] = _normalize_requested_device(self.config.get("device"))
        self.service_id = resolve_service_id(self.config)

    def run(self) -> ServiceExecutionResult:
        service_id = self.service_id
        db_path = str(self.config.get("db_path") or CONFIG["db_path"])
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        writer = make_writer("sqlite", db_path=db_path)
        writer.start()

        try:
            record, metrics, artifacts, split_provenance_rows = self._execute_service(started_at=started_at)
            db_timer = _StageTimer()
            writer.write_service(record)
            writer.write_service_metrics(service_id, metrics)
            for row in split_provenance_rows:
                writer.write_service_split_provenance(service_id, **row)
            for artifact in artifacts:
                writer.write_service_artifact(service_id, **artifact)
            db_write_pre_commit_s = db_timer.elapsed_s()
            metrics["db_write_s"] = _metric(db_write_pre_commit_s, "runtime", "s", "lower_better")
            writer.write_service_metric(
                service_id,
                "db_write_s",
                metrics["db_write_s"],
            )
            writer.finish()
            _log_service_timing(
                self.config,
                service_id,
                "DB write",
                db_timer.elapsed_s(),
                detail=f"db_path={db_path}",
            )
            return ServiceExecutionResult(service_id=service_id, status="success", db_path=db_path, metrics=metrics)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            failure_stage = getattr(exc, "failure_stage", "service_execution")
            failure_record = self._service_record(
                status="failed",
                started_at=started_at,
                metadata={
                    "runtime_total_s": elapsed,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "failure_stage": failure_stage,
                },
            )
            try:
                db_timer = _StageTimer()
                writer.write_service(failure_record)
                writer.write_service_failure(
                    service_id=service_id,
                    row_index=_safe_int(self.config.get("row_index")),
                    case_name=_safe_str(self.config.get("case_name")),
                    manifest_group_id=_safe_str(self.config.get("manifest_group_id")),
                    failure_stage=failure_stage,
                    error_message=str(exc),
                    resolved_config_json=json.dumps(self.config, default=str),
                    traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip(),
                )
                writer.finish()
                _log_service_timing(
                    self.config,
                    service_id,
                    "DB write",
                    db_timer.elapsed_s(),
                    detail=f"db_path={db_path} status=failed",
                )
            except Exception:
                writer.abort()
            return ServiceExecutionResult(service_id=service_id, status="failed", db_path=db_path, metrics={}, error=str(exc))

    def _execute_service(self, *, started_at: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        service_start = time.perf_counter()
        stage_timings: dict[str, float] = {}
        training_regime = str(self.config.get("training_regime") or "finetune_transfer").strip().lower()
        prepared_dataset = self.config.get("_prepared_dataset")
        if prepared_dataset is None:
            dataset_args = dict(self.config.get("dataset_args") or {})
            dataset_args.setdefault("task", self.config.get("task") or self.config.get("task_type"))
            dataset_args.setdefault("inference_only", training_regime in {"inference_only", "inference"})
            dataset_timer = _StageTimer()
            train, test, meta = _load_dataset(self.config.get("dataset", "hf"), **dataset_args)
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
        tracker.start()
        train_metrics: dict[str, Any] = {}
        eval_qos: dict[str, Any] = {}
        loss = primary = secondary = math.nan
        train_runtime_s = 0.0
        update_signature_metrics: dict[str, Any] = {}
        update_signature_artifact: dict[str, Any] | None = None

        workload_start = time.perf_counter()
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
            weights_before_training = _snapshot_model_weights(model)
            train_start = time.perf_counter()
            train_metrics = self._train_model(model, x_train, y_train, task_family=task_family)
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
                _snapshot_model_weights(model),
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
            task_family=task_family,
            hf_task=resolved_hf_task,
            primary_name=primary_name,
            primary_value=primary,
            secondary_name=secondary_name,
            secondary_value=secondary,
            metric_score=primary_score,
        )
        explainability, perturbation_artifact = _service_perturbation_metrics(
            model,
            x_test,
            y_test,
            config={**self.config, "service_id": self.service_id},
            meta=meta,
            task_family=task_family,
        )
        tracked_runtime_s = time.perf_counter() - workload_start
        usage = tracker.stop(tracked_runtime_s)
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
        hardware_snapshot = capture_hardware_snapshot() if _config_bool(self.config.get("measure_system_metrics"), True) else None
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
        artifacts = []
        if perturbation_artifact:
            artifacts.append(perturbation_artifact)
        if update_signature_artifact:
            artifacts.append(update_signature_artifact)
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
        return record, metrics, artifacts, split_provenance_rows

    def _capture_update_signature(self, before: Any, after: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if not _config_bool(self.config.get("update_signature_enabled"), True):
            return {}, None
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

    def _resolve_service_split(self, x_train, y_train, meta: Mapping[str, Any]) -> dict[str, Any]:
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
        sample_cap = self.config.get("sample_size")
        if sample_cap is None:
            sample_cap = self.config.get("max_samples")
        x_train, y_train, sample_info = _sample_service_rows(
            x_train,
            y_train,
            sample_size=sample_cap,
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
        print("========== SERVICE RUN SUMMARY ==========")
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
            "model_id": self.config.get("hf_model_id") or self.config.get("model_id") or getattr(model, "model_id", None),
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
            hf_task=self.config.get("hf_task"),
            max_length=self.config.get("max_length"),
            device=_device_arg_for_model(self.config.get("device")),
            mixed_precision=self.config.get("mixed_precision"),
            precision_type=self.config.get("precision_type"),
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
            try:
                qos = _fit_transformers_with_cuda_oom_retry(model, x_train, y_train, fit_kwargs, self.config)
            except TypeError:
                fallback_kwargs = {
                    key: fit_kwargs[key]
                    for key in ("epochs", "lr", "max_train_time_s", "progress_log_interval")
                    if key in fit_kwargs
                }
                qos = _fit_transformers_with_cuda_oom_retry(model, x_train, y_train, fallback_kwargs, self.config)
            return dict(qos or {}, training_epochs=epochs)
        train_local_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size, lr=learning_rate)
        return {"training_epochs": epochs}

    def _evaluate_model(self, model, x_test, y_test, *, inference_only: bool, task_family: str) -> tuple[float, float, float, dict[str, Any]]:
        if hasattr(model, "evaluate") and model.__class__.__name__.lower().startswith("transformers"):
            loss, primary, secondary, qos = model.evaluate(
                x_test,
                y_test,
                inference_only=inference_only,
                max_eval_time_s=self.config.get("max_eval_time_s"),
                progress_log_interval=self.config.get("eval_progress_log_interval", 10),
            )
            return loss, primary, secondary, dict(qos or {})
        if task_family == "clustering" and hasattr(model, "evaluate"):
            loss, primary, secondary = model.evaluate(x_test, y_test)
            return loss, primary, secondary, {}
        loss, primary, secondary = evaluate_model(model, x_test, y_test, task_type=("regression" if task_family == "regression" else "classification"))
        return loss, primary, secondary, {}

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
            "service_config_json": self.config.get("service_config") or _service_config(self.config),
            "registry_metadata_json": registry_metadata if registry_metadata is not None else _registry_metadata(self.config),
            "functional_attributes_json": functional_attributes or {},
            "metadata_json": metadata or {"started_at": started_at},
        }


def execute_service(config: Mapping[str, Any]) -> ServiceExecutionResult:
    return ServiceRunner(config).run()


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
    for candidate in (model, getattr(model, "core", None)):
        if candidate is None:
            continue
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


def _is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda out of memory" in text or (
        exc.__class__.__name__.lower() == "outofmemoryerror" and "cuda" in text
    )


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


def _fit_transformers_with_cuda_oom_retry(
    model: Any,
    x_train: Any,
    y_train: Any,
    fit_kwargs: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    try:
        return model.fit(x_train, y_train, **fit_kwargs)
    except Exception as exc:
        if not _is_cuda_oom_error(exc):
            raise
        current_batch_size = _safe_int(config.get("batch_size")) or _safe_int(getattr(model, "batch_size", None)) or 1
        if current_batch_size <= 1 or bool(config.get("_cuda_oom_retry_attempted")):
            raise
        retry_batch_size = max(1, int(current_batch_size) // 2)
        config["_cuda_oom_retry_attempted"] = True
        config["batch_size"] = retry_batch_size
        _apply_runtime_knobs_to_model(model, config)
        _clear_cuda_cache_if_available(model)
        print(
            "[HFCore.finetune] CUDA OOM retry starts | "
            f"batch_size={current_batch_size} -> {retry_batch_size}"
        )
        return model.fit(x_train, y_train, **fit_kwargs)


def _clear_model_runtime_state(model: Any) -> None:
    core = getattr(model, "core", None)
    torch = getattr(core, "torch", None)
    raw_model = getattr(core, "model", None) or getattr(model, "model", None)
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


def _canonical_split_strategy(value: Any) -> str:
    strategy = str(value or "iid").strip().lower().replace("-", "_")
    aliases = {
        "niid": "dirichlet",
        "non_iid": "dirichlet",
        "non_iid_dirichlet": "dirichlet",
        "shards": "shard",
        "quantity": "quantity_skew",
        "quantity_skewed": "quantity_skew",
    }
    return aliases.get(strategy, strategy)


def _stable_sample_seed(config: Mapping[str, Any]) -> int:
    parts = [
        config.get("seed", 42),
        config.get("service_id"),
        config.get("hf_model_id") or config.get("model_type"),
        config.get("dataset_name") or config.get("dataset"),
        config.get("dataset_config"),
        config.get("dataset_variant"),
        config.get("split_variant"),
        config.get("knob_variant"),
        config.get("split_strategy") or config.get("distribution_type"),
    ]
    raw = "|".join("" if value is None else str(value) for value in parts)
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8], 16)


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


def _sample_service_rows(
    x,
    y,
    *,
    sample_size: Any,
    sample_frac: Any,
    strategy: str = "iid",
    distribution_param: Any = None,
    axis=None,
    rng,
) -> tuple[Any, Any, dict[str, Any]]:
    total = _sample_count(x, y)
    requested = _safe_int(sample_size)
    if requested is None and sample_frac is not None:
        try:
            requested = int(round(total * float(sample_frac)))
        except Exception:
            requested = None
    if requested is None:
        return x, y, {
            "requested_sample_size_total": None,
            "sample_strategy_effective": "all",
            "effective_strategy": "all",
            "effective_axis": getattr(axis, "effective_axis", None),
            "bucket_distribution": bucket_distribution(getattr(axis, "bucket_ids", None), getattr(axis, "bucket_labels", None)),
        }
    requested = max(0, min(total, int(requested)))
    if requested == total:
        return x, y, {
            "requested_sample_size_total": requested,
            "sample_strategy_effective": "all",
            "effective_strategy": "all",
            "effective_axis": getattr(axis, "effective_axis", None),
            "bucket_distribution": bucket_distribution(getattr(axis, "bucket_ids", None), getattr(axis, "bucket_labels", None)),
        }
    strategy = _canonical_split_strategy(strategy)
    info = {
        "requested_sample_size_total": requested,
        "sample_strategy_effective": strategy,
        "effective_strategy": strategy,
        "effective_axis": getattr(axis, "effective_axis", None),
        "bucket_distribution": bucket_distribution(getattr(axis, "bucket_ids", None), getattr(axis, "bucket_labels", None)),
    }
    if requested <= 0:
        idx = np.asarray([], dtype=int)
    elif strategy in {"dirichlet", "quantity_skew"}:
        effective_strategy = "dirichlet" if strategy == "quantity_skew" else strategy
        info["effective_strategy"] = effective_strategy
        if strategy == "quantity_skew":
            info["fallback_reason"] = "quantity_skew is a service-local compatibility alias for dirichlet on the resolved skew axis"
        compatible, reason = axis_supports_strategy(axis, effective_strategy) if axis is not None else (False, "missing skew axis")
        if not compatible or getattr(axis, "bucket_ids", None) is None:
            idx = rng.choice(total, size=requested, replace=False)
            info["sample_strategy_effective"] = "iid"
            info["effective_strategy"] = "iid"
            info["fallback_reason"] = reason or info.get("fallback_reason") or f"strategy='{strategy}' requires a resolvable skew axis"
        else:
            idx, skew_info = _strategy_sample_indices(
                getattr(axis, "bucket_ids"),
                total=total,
                requested=requested,
                strategy=effective_strategy,
                distribution_param=distribution_param,
                bucket_labels=getattr(axis, "bucket_labels", None),
                rng=rng,
            )
            info.update(skew_info)
    else:
        idx = rng.choice(total, size=requested, replace=False)
    return _take_rows(x, idx), _take_rows(y, idx), info


def _cap_benchmark_split(x, y, *, config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    cap = (
        config.get("benchmark_sample_size")
        or config.get("max_benchmark_samples")
        or config.get("max_eval_samples")
        or config.get("max_samples")
        or config.get("sample_size")
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

    seed = _safe_int(config.get("sample_seed"))
    if seed is None:
        seed = _stable_sample_seed(config)
    rng = np.random.default_rng(int(seed) + 1)
    idx = rng.choice(total, size=requested, replace=False)
    return _take_rows(x, idx), _take_rows(y, idx), {
        "requested_sample_size_total": requested,
        "effective_sample_size_total": requested,
        "sample_strategy_effective": "iid",
        "sample_seed": int(seed) + 1,
    }


def _strategy_sample_indices(bucket_ids, *, total: int, requested: int, strategy: str, distribution_param: Any, bucket_labels: Mapping[str, Any] | None, rng) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(bucket_ids, dtype=np.int64).reshape(-1)
    if len(labels) != total:
        raise ValueError("bucket_ids length does not match sample count")
    unique_labels = np.asarray(sorted(np.unique(labels).tolist()), dtype=labels.dtype)
    if len(unique_labels) <= 1:
        return (
            rng.choice(total, size=requested, replace=False),
            {
                "sample_strategy_effective": "iid",
                "fallback_reason": f"strategy='{strategy}' requires at least two axis buckets",
            },
        )

    alpha = _to_float(distribution_param)
    if math.isnan(alpha) or alpha <= 0:
        alpha = 0.2
    concentration = max(0.05, min(float(alpha), 10.0))

    weights = rng.dirichlet(np.full(len(unique_labels), concentration, dtype="float64"))
    counts = _counts_from_weights(weights, requested, labels, unique_labels)
    chosen: list[int] = []
    for label, count in zip(unique_labels, counts):
        if count <= 0:
            continue
        pool = np.where(labels == label)[0]
        if len(pool) == 0:
            continue
        chosen.extend(rng.choice(pool, size=min(int(count), len(pool)), replace=False).tolist())

    if len(chosen) < requested:
        remaining = np.setdiff1d(np.arange(total, dtype=int), np.asarray(chosen, dtype=int), assume_unique=False)
        if len(remaining) > 0:
            top_up = rng.choice(remaining, size=min(requested - len(chosen), len(remaining)), replace=False)
            chosen.extend(top_up.tolist())

    idx = np.asarray(chosen[:requested], dtype=int)
    rng.shuffle(idx)
    return idx, {
        "sample_strategy_effective": strategy,
        "effective_strategy": strategy,
        "label_sampling_alpha": float(alpha),
        "label_sampling_concentration": float(concentration),
        "label_sampling_weights": {
            str((bucket_labels or {}).get(str(int(label)), int(label))): float(weight)
            for label, weight in zip(unique_labels.tolist(), weights.tolist())
        },
        "bucket_distribution": bucket_distribution(labels[idx], bucket_labels),
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


def _counts_from_weights(weights: np.ndarray, requested: int, labels: np.ndarray, unique_labels: np.ndarray) -> np.ndarray:
    capacities = np.asarray([int(np.sum(labels == label)) for label in unique_labels], dtype=int)
    raw = np.asarray(weights, dtype="float64") * float(requested)
    counts = np.floor(raw).astype(int)
    counts = np.minimum(counts, capacities)
    while counts.sum() < requested and np.any(counts < capacities):
        remaining_capacity = capacities - counts
        fractional = raw - np.floor(raw)
        scores = np.where(remaining_capacity > 0, fractional + weights, -1.0)
        idx = int(np.argmax(scores))
        if scores[idx] < 0:
            break
        counts[idx] += 1
    while counts.sum() > requested:
        idx = int(np.argmax(counts))
        counts[idx] -= 1
    return counts


def _take_rows(value, idx):
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: _take_rows(child, idx) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        return value[idx]
    if isinstance(value, tuple):
        idx_list = idx.tolist() if isinstance(idx, np.ndarray) else list(idx)
        return tuple(value[i] for i in idx_list)
    if isinstance(value, list):
        idx_list = idx.tolist() if isinstance(idx, np.ndarray) else list(idx)
        return [value[i] for i in idx_list]
    try:
        return np.asarray(value, dtype=object)[idx]
    except Exception:
        return value


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
    enabled = _config_bool(config.get("enable_perturbation_metrics", config.get("explainability_enabled")), True)
    if not enabled:
        return {
            "perturbation_enabled_flag": _metric(False, "explainability"),
            "explainability_supported_flag": _metric(False, "explainability"),
            "explainability_score": _metric(0.0, "explainability", "score", "higher_better"),
        }, None
    log_stage = _config_bool(config.get("perturbation_stage_logging"), True)
    start = time.perf_counter()
    if log_stage:
        print(
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
            "explainability_score": 0.0,
            "perturbation_error": f"{type(exc).__name__}: {exc}",
            "perturbation_duration_s": elapsed,
        }
    if isinstance(values, dict):
        values.setdefault("perturbation_duration_s", time.perf_counter() - start)
    if log_stage:
        print(
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
    print(message, flush=True)


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


def _create_model(**kwargs):
    global create_model
    if create_model is None:
        from ..models.builders import create_model as imported

        create_model = imported
    return create_model(**kwargs)


def _service_config(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "resource_tier",
        "batch_size",
        "learning_rate",
        "training_epochs",
        "optimizer",
        "weight_decay",
        "momentum",
        "warmup_ratio",
        "gradient_accumulation_steps",
        "max_samples",
        "sample_size",
        "sample_seed",
        "max_length",
        "device",
        "mixed_precision",
        "precision_type",
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


def _sample_count(x, y=None) -> int:
    source = y if y is not None else x
    if source is None:
        return 0
    if isinstance(source, Mapping):
        for value in source.values():
            try:
                return int(len(value))
            except Exception:
                continue
        return 0
    try:
        return int(len(source))
    except Exception:
        return 0


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
        else:
            out[normalized] = _metric(value, metric_domain(normalized))
    return out


def _metric(value: Any, domain: str, unit: str | None = None, direction: str = "neutral") -> dict[str, Any]:
    return {"value": value, "domain": domain, "unit": unit, "direction": direction}


def _quality_direction(metric_name: str) -> str:
    return "lower_better" if str(metric_name).lower() in {"loss", "rmse", "mae", "perplexity"} else "higher_better"


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
        return meta_task_text
    config_task_type = config.get("task_type")
    if _is_blank(config_task_type):
        return None
    config_task_text = str(config_task_type).strip().lower()
    config_task_family = canonical_task_family(config_task_text, hf_task)
    if hf_task_norm == "sentence_similarity" and config_task_family == "regression":
        return "regression"
    return config_task_text


def _validate_service_compatibility(
    *,
    task_family: str,
    training_regime: str,
    hf_model_id: Any,
    batch_size: int | None,
    train_samples: int,
    benchmark_samples: int,
) -> None:
    if training_regime in {"inference_only", "inference"}:
        return
    minimums = {
        "detection": (32, 16),
        "segmentation": (32, 16),
    }
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
    if hf_task_norm == "sentence_similarity":
        return ("spearman",)
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
) -> None:
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


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


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


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return value
    except Exception:
        return str(value)
