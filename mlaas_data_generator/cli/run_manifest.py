from __future__ import annotations

from ..diagnostic_json import diagnostic_json, bounded_text
from ..execution_state import (persist_terminal, queue_services, recover_service_states,
                               register_manifest_roster, reconcile_manifest_roster, FEDAVG_OBSERVATION_STATUSES)


from mlaas_data_generator.terminal import debug_print, debug_enabled, terminal_scope, print_results

import argparse
import concurrent.futures
import importlib
import json
import multiprocessing
import os
import queue
import shutil
import sqlite3
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..fedavg_run import execute_fedavg_run, recover_abandoned_fedavg_runs
from ..fedavg_manifest import PARTITION_GENERATOR_VERSION
from ..compatibility import known_bad_row_reason
from ..family_preflight import manifest_family_error
from ..config import CONFIG, FAILED_MANIFEST_PATH, FAILURE_LOG_PATH, MANIFEST_RESULTS_PATH
from ..data.sources.generic import KERAS_DATASETS
from ..data.runtime_cache import preparation_cache_scope, start_preparation_cache
from ..hf_auth import load_hf_token_from_file
from ..registry.datasets import DATASET_REGISTRY
from ..models.label_schema import infer_label_format, infer_num_labels
from ..services.runner import execute_service, resolve_service_id
from ..services import runner as service_runner
from ..services.taxonomy import canonical_task_family
from ..runtime_compat import is_cuda_poison_error
from ..storage.writer import make_writer

BASE_DEFAULTS: dict[str, Any] = {
    "training_regime": "finetune_transfer",
    "training_epochs": 1,
    "batch_size": 16,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "mixed_precision": False,
    "precision_type": "fp16",
    "seed": 42,
    "dataset_variant": 0,
    "split_variant": 0,
    "knob_variant": 0,
}

BOOL_COLUMNS = {
    "checkpoint_measure_full",
    "enabled",
    "measure_system_metrics",
    "mixed_precision",
    "explainability_enabled",
    "enable_perturbation_metrics",
    "perturbation_stage_logging",
    "perturbation_progress_logging",
    "save_weights",
    "no_save_weights",
    "smoke_test",
    "update_signature_enabled",
    "dynamic_padding",
    "report_decode_errors",
    "scheduled_to_participate",
}
INT_COLUMNS = {
    "checkpoint_reduced_compression_level", "checkpoint_validation_samples",
    "initialization_seed",
    "evaluation_seed",
    "effective_batch_size",
    "cohort_size",
    "max_client_quantity",
    "seed",
    "sample_seed",
    "training_epochs",
    "batch_size",
    "sample_size",
    "max_length",
    "num_workers",
    "timeout_s",
    "max_train_time_s",
    "max_eval_time_s",
    "source_max_length",
    "target_max_length",
    "train_examples",
    "benchmark_examples",
    "vqa_answer_vocab_size",
    "dataset_variant",
    "split_variant",
    "knob_variant",
    "perturbation_sample_count",
    "perturbation_candidate_units",
    "perturbation_target_units",
    "perturbation_trust_trials",
    "perturbation_progress_sample_interval",
    "clustering_k",
    "clustering_n_init",
    "clustering_max_iter",
    "update_signature_dim",
    "update_signature_max_source_elements",
    "gradient_accumulation_steps",
    "checkpoint_patch_compression_level",
    "round_id",
    "run_partition_seed",
    "benchmark_sample_seed",
    "client_partition_index",
    "fedavg_run_index",
    "benchmark_sample_size",
    "run_train_pool_size",
    "fedavg_source_pool_cap",
    "requested_total_samples",
    "training_seed",
    "participation_schedule_seed",
    "fedavg_clients_per_run",
    "fedavg_rounds",
    "requested_quota",
}
FLOAT_COLUMNS = {
    "max_grad_norm",
    "checkpoint_quality_tolerance",
    "learning_rate",
    "weight_decay",
    "momentum",
    "warmup_ratio",
    "mlm_probability",
    "distribution_param",
    "clustering_tol",
    "realism_score",
    "perturbation_random_strength",
    "fedavg_client_participation_rate",
    "heterogeneity_intensity",
    "shard_support_quantile",
    "min_target_bin_fraction",
}
ENUM_COLUMNS = {"training_regime", "resource_tier", "optimizer", "device", "model_type", "hf_task", "task_type", "modality", "split_strategy", "distribution_type", "skew_axis", "precision_type"}
JSON_COLUMNS = {
    "preprocessing_identity",
    "processor_size",
    "task_contract", "task_source_groups", "generation_config",
    "checkpoint_storage_options",
    "_frozen_manifest_defaults",
    "training_policy",
    "client_recipe",
    "hidden_layers",
    "column_mapping",
    "service_config",
    "custom_distributions",
    "skew_axis_config",
    "requested_partition_params",
}
LINEAGE_COLUMNS = {"fl_run_id", "client_id", "round_id", "lineage_id"}

MULTIMODAL_TASKS = {"image_captioning", "text_image_retrieval", "visual_question_answering"}
MULTIMODAL_AUDIT_PATH = MANIFEST_RESULTS_PATH.parent / "service_manifest_multimodal_audit.csv"
MISSING_MULTIMODAL_MANIFEST_PATH = MANIFEST_RESULTS_PATH.parent / "service_manifest_missing_multimodal.csv"
FEDAVG_RUN_RESULTS_PATH = MANIFEST_RESULTS_PATH.parent / "fedavg_run_results.csv"

DATASET_ARG_COLUMNS = {
    "source_group_column",
    "processor_size",
    "dataset_name",
    "dataset_config",
    "hf_dataset_revision",
    "hf_model_id",
    "hf_model_revision",
    "hf_task",
    "max_length",
    "train_split",
    "test_split",
    "benchmark_split",
    "label_column",
    "mask_column",
    "text_column",
    "image_column",
    "question_column",
    "answer_column",
    "ranking_label_column",
    "modality",
    "missing_pair_handling",
    "on_decode_error",
    "report_decode_errors",
    "vqa_label_mode", "label_mode",
    "vqa_answer_vocab_size",
    "vqa_unseen_answer_policy",
    "retrieval_positive_policy",
    "benchmark_sample_size",
    "benchmark_sample_seed",
    "source_max_length",
    "target_max_length",
    "dynamic_padding",
    "mlm_probability",
    "column_mapping",
    "task_tag",
    "task",
    "training_regime",
    "service_source",
    "model_role",
    "input_schema",
    "fit_decision",
    "fit_reason",
    "dataset_hint",
    "train_examples",
    "benchmark_examples",
    "hf_pipeline_tag",
    "hf_downloads",
    "hf_likes",
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
    "hf_author",
    "hf_url",
    "hf_service_meta_json",
    "explainability_enabled",
    "explainability_method",
    "explainability_target",
}

REQUIRED_COLUMNS = {"dataset", "model_type", "task_type"}
BLANK_STRINGS = {"", "na", "n/a", "nan", "null", "none", "not applicable", "not_applicable"}
RESOURCE_TIER_VALUES = {"smoketest", "light", "medium", "heavy", "stress_test"}
KERAS_MODEL_TYPES = {"cnn", "mobilenetv2", "mlp", "logreg"}

COLUMN_ALIASES = {
    "manifest group id": "manifest_group_id",
    "training regime": "training_regime",
    "dataset variant": "dataset_variant",
    "split variant": "split_variant",
    "knob variant": "knob_variant",
    "service config": "service_config",
    "batch size": "batch_size",
    "learning rate": "learning_rate",
    "learing_rate": "learning_rate",
    "earning_rate": "learning_rate",
    "mlm probability": "mlm_probability",
    "training epochs": "training_epochs",
    "epochs": "training_epochs",
    "model type": "model_type",
    "task type": "task_type",
    "dataset name": "dataset_name",
    "dataset config": "dataset_config",
    "hf model id": "hf_model_id",
    "train split": "train_split",
    "test split": "test_split",
    "benchmark split": "benchmark_split",
    "label column": "label_column",
    "mask column": "mask_column",
    "text column": "text_column",
    "image column": "image_column",
    "task tag": "task_tag",
    "dataset task": "task",
    "db": "db_path",
    "database": "db_path",
    "database path": "db_path",
    "sql db": "db_path",
    "sql db path": "db_path",
    "sqlite db": "db_path",
    "sqlite db path": "db_path",
    "sample count": "sample_size",
    "sample_count": "sample_size",
    "samples": "sample_size",
    "split strategy": "split_strategy",
    "distribution type": "distribution_type",
    "distribution param": "distribution_param",
    "skew axis": "skew_axis",
    "skew axis config": "skew_axis_config",
    "custom distributions": "custom_distributions",
    "save weights": "save_weights",
    "precision type": "precision_type",
}

FEDERATED_COLUMNS = {
    "external_run_id",
    "run_group_id",
    "run_group",
    "num_" + "".join(["c", "lients"]),
    "num_" + "rounds",
    "rounds",
    "".join(["c", "lients"]),
    "".join(["c", "lient"]) + "_" + "participation_rate",
    "".join(["c", "lient"]) + "_" + "dropout_rate",
    "aggregation",
    "aggregator",
    "aggregation_" + "weight",
    "aggregation_" + "weight_unit",
    "aggregation_" + "weight_value",
    "global_" + "model",
    "local_epochs",
}


@dataclass
class RowValidation:
    ok: bool
    error: str = ""


class ManifestPreflightError(RuntimeError):
    """Raised when manifest execution prerequisites are not available."""


@dataclass
class ManifestEntry:
    idx: int
    ordinal: int
    resolved: dict[str, Any]
    validation: RowValidation


@dataclass
class ManifestProgressTracker:
    total: int
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    _running: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self._running is None:
            self._running = {}
        self._last_render = None
        self._finished = False

    def start(self, worker_label: str, description: str) -> None:
        self._running[worker_label] = description
        self.render()

    def clear(self, worker_label: str) -> None:
        if worker_label in self._running:
            self._running.pop(worker_label, None)
            self.render()

    def record(self, status: str, *, render: bool = True) -> None:
        self.completed += 1
        normalized = str(status).strip().lower()
        if normalized == "success":
            self.succeeded += 1
        elif normalized in {"skipped", "not_participating"}:
            self.skipped += 1
        else:
            self.failed += 1
        if render:
            self.render()

    def render(self, *, force: bool = False) -> None:
        if self._finished:
            return
        now = time.monotonic()
        if not force and self._last_render is not None and now - self._last_render < 10:
            return
        self._last_render = now
        queued = max(self.total - self.completed - len(self._running), 0)
        print(f"Manifest progress: {self.completed}/{self.total} complete | "
              f"running={len(self._running)} queued={queued} "
              f"success={self.succeeded} skipped={self.skipped} failed={self.failed}", flush=True)

    def finish(self) -> None:
        if not self._finished:
            self.render(force=True)
            self._finished = True


_ACTIVE_MANIFEST_PROGRESS: ManifestProgressTracker | None = None
_WORKER_PROGRESS_QUEUE: Any = None
_WORKER_LABEL = "main"


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict)):
        return False
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in BLANK_STRINGS
    return False


def _worker_label_for_gpu(gpu_id: int | None) -> str:
    return "cpu" if gpu_id is None else f"gpu{int(gpu_id)}"


def _describe_entry(entry: ManifestEntry) -> str:
    resolved = entry.resolved
    total = int(resolved.get("_manifest_total") or entry.ordinal)
    return f"{resolved.get('service_id')} ({entry.ordinal}/{total})"


def _describe_group(entries: list[ManifestEntry]) -> str:
    first = entries[0].resolved
    return (
        f"{len(entries)} rows "
        f"{first.get('hf_model_id') or first.get('model_type')} "
        f"{first.get('dataset_name') or ''}"
    ).strip()


def _manifest_progress() -> ManifestProgressTracker | None:
    return _ACTIVE_MANIFEST_PROGRESS


def _manifest_progress_start(worker_label: str, description: str) -> None:
    if _WORKER_PROGRESS_QUEUE is not None:
        _WORKER_PROGRESS_QUEUE.put({"event": "start", "worker_label": _WORKER_LABEL, "description": description})
        return
    tracker = _manifest_progress()
    if tracker is not None:
        tracker.start(worker_label, description)


def _manifest_progress_clear(worker_label: str) -> None:
    tracker = _manifest_progress()
    if tracker is not None:
        tracker.clear(worker_label)


def _record_manifest_progress_result(result: dict[str, Any]) -> None:
    render = not result.pop("_terminal_defer_progress", False)
    if _WORKER_PROGRESS_QUEUE is not None:
        try:
            _WORKER_PROGRESS_QUEUE.put(
                {
                    "event": "result",
                    "worker_label": _WORKER_LABEL,
                    "status": result.get("status"),
                    "render": render,
                }
            )
        except Exception:
            pass
        return

    tracker = _manifest_progress()
    if tracker is not None:
        tracker.record(str(result.get("status") or "failed"), render=render)


def _drain_progress_queue(progress_queue: Any) -> None:
    tracker = _manifest_progress()
    if tracker is None or progress_queue is None:
        return
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            return
        except Exception:
            return
        if event.get("event") == "start":
            tracker.start(str(event.get("worker_label")), str(event.get("description")))
        elif event.get("event") == "result":
            tracker.record(str(event.get("status") or "failed"), render=event.get("render", True))


def _normalize_value(value: Any) -> Any:
    if _is_blank(value):
        return None
    return value.strip() if isinstance(value, str) else value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_by_column(column: str, value: Any) -> Any:
    if column == "optimizer" and isinstance(value, str) and value.strip().lower() == "none":
        return "none"  # An explicit estimator setting, not a missing CSV value.
    value = _normalize_value(value)
    if value is None:
        return None
    if column in BOOL_COLUMNS:
        return _to_bool(value)
    if column in INT_COLUMNS:
        return int(float(value))
    if column in FLOAT_COLUMNS:
        return float(value)
    if column in JSON_COLUMNS and isinstance(value, str):
        return json.loads(value)
    if column in ENUM_COLUMNS and isinstance(value, str):
        return value.strip().lower()
    return value


def _normalize_column_name(column: Any) -> Any:
    if not isinstance(column, str):
        return column
    normalized = column.strip()
    lower = normalized.lower()
    return COLUMN_ALIASES.get(lower, lower.replace(" ", "_"))


def _normalize_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_column_name(col) for col in df.columns]
    return df


def _extract_defaults_row(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if "service_id" not in df.columns:
        return {}, df
    marker = df["service_id"].astype(str).str.strip().str.lower() == "defaults"
    if not marker.any():
        return {}, df
    defaults: dict[str, Any] = {}
    defaults_row = df.loc[marker].iloc[0]
    for column, raw in defaults_row.items():
        if column == "service_id":
            continue
        value = _coerce_by_column(column, raw)
        if value is not None:
            defaults[column] = value
    return defaults, df.loc[~marker].reset_index(drop=True)


def load_manifest(file_path: Path, sheet: str = "services") -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = _normalize_manifest_columns(pd.read_csv(file_path))
        if "family_ref" in df:
            from .manifest.review import expand_review_manifest
            return expand_review_manifest(df, file_path)
        defaults, rows = _extract_defaults_row(df)
        return rows, defaults
    if suffix in {".xlsx", ".xls"}:
        workbook = pd.read_excel(file_path, sheet_name=None)
        service_rows_df = _normalize_manifest_columns(workbook.get(sheet) if sheet in workbook else next(iter(workbook.values())))
        if "family_ref" in service_rows_df:
            from .manifest.review import expand_review_manifest
            return expand_review_manifest(service_rows_df, file_path)
        defaults: dict[str, Any] = {}
        if "defaults" in workbook:
            defaults_df = _normalize_manifest_columns(workbook["defaults"])
            if not defaults_df.empty:
                defaults = {
                    key: _coerce_by_column(key, value)
                    for key, value in defaults_df.iloc[0].to_dict().items()
                    if _coerce_by_column(key, value) is not None
                }
        csv_defaults, service_rows_df = _extract_defaults_row(service_rows_df)
        defaults.update(csv_defaults)
        return service_rows_df, defaults
    raise ValueError(f"Unsupported file extension '{suffix}'. Use .csv or .xlsx")


def _database_candidates(db_path: Any) -> list[Path]:
    """Return the configured SQLite file and any GPU shards derived from it."""
    raw = str(db_path or CONFIG.get("db_path") or "").strip()
    if not raw:
        return []
    path = Path(raw)
    candidates = [path]
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    candidates.extend(sorted(path.parent.glob(f"{stem}.gpu*{suffix}")))
    return list(dict.fromkeys(candidates))


def _find_existing_service_database(
    db_path: Any,
    service_id: Any,
    cache: dict[str, set[str]],
) -> str | None:
    """Find a service by primary ID without creating or migrating a database."""
    wanted = str(service_id)
    for candidate in _database_candidates(db_path):
        if not candidate.is_file():
            continue
        cache_key = str(candidate.resolve())
        if cache_key not in cache:
            with sqlite3.connect(str(candidate)) as conn:
                table_exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='services'"
                ).fetchone()
                columns = {row[1] for row in conn.execute("PRAGMA table_info(services)")}
                cache[cache_key] = (
                    {str(row[0]) for row in conn.execute("SELECT service_id FROM services WHERE status='completed'")}
                    if table_exists and "status" in columns
                    else set()
                )
        if wanted in cache[cache_key]:
            return str(candidate)
    return None


def _resolve_row(row: pd.Series, manifest_defaults: dict[str, Any]) -> dict[str, Any]:
    # Compact manifests snapshot defaults when generated. Failed-row exports
    # retain that snapshot as a JSON column, so they remain self-contained.
    frozen_key = "_frozen_manifest_defaults"
    frozen = _coerce_by_column(frozen_key, row.get(frozen_key))
    if frozen is None:
        frozen = manifest_defaults.get(frozen_key)
    if frozen is not None and not isinstance(frozen, dict):
        raise ValueError("frozen manifest defaults must be an object")
    resolved = dict(frozen) if frozen is not None else {
        **CONFIG, **BASE_DEFAULTS,
        # Old expanded manifests did not snapshot these fields. New exports do.
        "checkpoint_patch_operation_policy": "adaptive",
        "checkpoint_measure_full": True, "checkpoint_weight_format": "native",
    }
    resolved.update({key: value for key, value in manifest_defaults.items() if key != frozen_key})
    for column, raw in row.items():
        if column == frozen_key:
            continue
        value = _coerce_by_column(column, raw)
        if value is not None:
            resolved[column] = value

    # Retired source caps in old manifests must not limit client sample_size.
    resolved.pop("max_samples", None)
    resolved.pop("fedavg_source_pool_cap", None)

    if resolved.get("generation_mode") == "shared_parent" and isinstance(resolved.get("client_recipe"), dict):
        for key, value in resolved["client_recipe"].items():
            if value is None:
                resolved[key] = None
    if (resolved.get("optimizer") == "none" and isinstance(resolved.get("training_policy"), dict)
            and resolved["training_policy"].get("optimizers") == ["none"]
            and _is_blank(row.get("learning_rate"))):
        resolved["learning_rate"] = None  # Preserve the frozen non-gradient policy's null rate.
    if resolved.get("benchmark_split") and not resolved.get("test_split"):
        resolved["test_split"] = resolved["benchmark_split"]
    if resolved.get("test_split") and not resolved.get("benchmark_split"):
        resolved["benchmark_split"] = resolved["test_split"]
    resolved["precision_type"] = _normalize_precision_type(resolved.get("precision_type"))
    if str(_normalize_requested_device(resolved.get("device"))).strip().lower() == "cpu":
        resolved["mixed_precision"] = False
        resolved["precision_type"] = "fp16"
    _apply_final_multimodal_runtime_defaults(resolved)
    from ..task_contract import resolve_sentence_pair_contract
    from ..precision_policy import apply_precision_policy
    resolve_sentence_pair_contract(resolved)
    apply_precision_policy(resolved)
    from ..segmentation_policy import apply_segmentation_label_policy
    apply_segmentation_label_policy(resolved)
    if _is_blank(resolved.get("service_id")):
        resolved["service_id"] = resolve_service_id(resolved)
    resolved["dataset_args"] = _build_dataset_args(resolved)
    from ..debug_mode import apply_debug_mode
    apply_debug_mode(resolved)
    return resolved


def _resolve_execution_row(row: pd.Series, manifest_defaults: dict[str, Any]) -> dict[str, Any]:
    """Retain a row identity so malformed settings can be recorded as failed."""
    try:
        return _resolve_row(row, manifest_defaults)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        config = {**CONFIG, **BASE_DEFAULTS, **manifest_defaults}
        config.update({key: value for key, value in row.items() if not _is_blank(value)})
        config["service_id"] = resolve_service_id(config)
        from ..debug_mode import apply_debug_mode
        apply_debug_mode(config)
        config["_row_resolution_error"] = bounded_text(f"Invalid manifest settings: {exc}")
        return config


def _apply_final_multimodal_runtime_defaults(resolved: dict[str, Any]) -> None:
    hf_task = str(resolved.get("hf_task") or "").strip().lower().replace("-", "_")
    model_id = str(resolved.get("hf_model_id") or "").strip().lower()
    generative_vqa = hf_task == "visual_question_answering" and (
        model_id.startswith("salesforce/blip")
        or model_id.startswith("microsoft/git-")
        or "blip" in model_id
        or "git" in model_id
    )
    if hf_task == "image_captioning" or generative_vqa:
        resolved["batch_size"] = 1
        service_config = resolved.get("service_config")
        if isinstance(service_config, dict):
            service_config["batch_size"] = 1


def _build_dataset_args(resolved: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"seed": resolved.get("seed", 42)}
    for key in DATASET_ARG_COLUMNS:
        value = resolved.get(key)
        if value is not None:
            args[key] = value
    if args.get("benchmark_split") and not args.get("test_split"):
        args["test_split"] = args["benchmark_split"]
    if resolved.get("partition_generator_version") == PARTITION_GENERATOR_VERSION:
        args["benchmark_sample_size"] = resolved.get("benchmark_sample_size")
        args["benchmark_sample_seed"] = resolved.get("benchmark_sample_seed")
    return args


def _resolve_manifest_hf_revisions(
    resolved: dict[str, Any],
    cache: dict[tuple[str, str, str | None], tuple[str | None, str | None]],
) -> None:
    """Resolve each Hub identity once before grouped dataset/model loading."""
    if not _is_hf_resolved(resolved):
        return
    requests = (
        ("model", "hf_model_id", "hf_model_revision"),
        ("dataset", "dataset_name", "hf_dataset_revision"),
    )
    errors: list[str] = []
    for kind, identity_key, revision_key in requests:
        identity = resolved.get(identity_key)
        if _is_blank(identity):
            continue
        requested = None if _is_blank(resolved.get(revision_key)) else str(resolved[revision_key])
        cache_key = (kind, str(identity), requested)
        if cache_key not in cache:
            probe: dict[str, Any]
            if kind == "model":
                probe = {"hf_model_id": identity, revision_key: requested}
            else:
                probe = {
                    "dataset": "hf",
                    "dataset_name": identity,
                    revision_key: requested,
                }
            service_runner._resolve_hf_revisions(probe)
            cache[cache_key] = (
                probe.get(revision_key),
                probe.get("hf_revision_resolution_error"),
            )
        revision, error = cache[cache_key]
        if revision:
            resolved[revision_key] = revision
        if error:
            errors.append(f"{kind}: {error}")
    if errors:
        resolved["hf_revision_resolution_error"] = "; ".join(errors)


def _validate_row(resolved: dict[str, Any]) -> RowValidation:
    from ..debug_mode import debug_mode_error
    if error := debug_mode_error(resolved):
        return RowValidation(False, error)
    if str(resolved.get("log_mode") or "concise").strip().lower() not in {"concise", "debug"}:
        return RowValidation(False, "log_mode must be concise or debug")
    federated_columns = [
        key
        for key, value in resolved.items()
        if (key in FEDERATED_COLUMNS or str(key).startswith("global_")) and key not in LINEAGE_COLUMNS and not _is_blank(value)
    ]
    if federated_columns:
        return RowValidation(
            False,
            "Federated columns are not accepted in service manifests: "
            + ", ".join(sorted(federated_columns)),
        )
    generation_mode = str(resolved.get("generation_mode") or "independent").strip().lower()
    if generation_mode == "fedavg" or not _is_blank(resolved.get("fl_run_id")):
        return RowValidation(False, "FedAvg generation is retired; use select-parent-configs -> train-parents -> cohort-manifest")
    if generation_mode not in {"independent", "shared_parent"}:
        return RowValidation(False, "generation_mode must be independent or shared_parent")
    if generation_mode == "shared_parent":
        from ..cohort_manifest import validate_client_recipe
        try:
            validate_client_recipe(resolved)
        except (ValueError, TypeError, KeyError) as exc:
            return RowValidation(False, str(exc))
    elif any(not _is_blank(resolved.get(key)) for key in ("client_id", "round_id")):
        return RowValidation(False, "client_id requires shared_parent generation; round_id is retired")
    elif not _is_blank(resolved.get("recipe_sampler_version")):
        from ..clients.training import validate_training_recipe
        from ..clients.partitioning import SUPPORTED_PARTITION_VERSIONS, validate_partition_config
        try:
            validate_training_recipe(resolved)
            if resolved.get("partition_generator_version") not in SUPPORTED_PARTITION_VERSIONS:
                raise ValueError("client recipe requires its versioned partitioner")
            validate_partition_config(resolved)
        except (ValueError, TypeError, KeyError) as exc:
            return RowValidation(False, str(exc))

    for col in REQUIRED_COLUMNS:
        if _is_blank(resolved.get(col)):
            return RowValidation(False, f"Missing required column '{col}'")
    if _is_blank(resolved.get("service_id")):
        return RowValidation(False, "Missing or unresolved service_id")
    training_regime = str(resolved.get("training_regime") or "").strip().lower()
    if training_regime not in {"finetune_transfer", "inference_only", "generic"}:
        return RowValidation(False, "training_regime must be one of finetune_transfer, inference_only, or generic")
    resource_tier = str(resolved.get("resource_tier") or "").strip().lower()
    if resource_tier and resource_tier not in RESOURCE_TIER_VALUES:
        return RowValidation(False, "resource_tier must be one of smoketest, light, medium, heavy, or stress_test")
    if int(resolved.get("training_epochs", 1) or 1) <= 0 and training_regime != "inference_only":
        return RowValidation(False, "training_epochs must be > 0 for trainable services")
    if int(resolved.get("batch_size", 0) or 0) <= 0:
        return RowValidation(False, "batch_size must be > 0")
    precision_type = _normalize_precision_type(resolved.get("precision_type"))
    if precision_type not in {"fp16", "bf16", "fp32"}:
        return RowValidation(False, "precision_type must be one of fp16 or bf16")
    from ..weight_precision import normalize_weight_format
    try:
        normalize_weight_format(resolved.get("checkpoint_weight_format"))
        service_runner._checkpoint_compression(resolved)
        from ..checkpoint_store import CompressionSpec
        CompressionSpec(resolved.get("checkpoint_reduced_compression", "none"), resolved.get("checkpoint_reduced_compression_level", 1))
        if int(resolved.get("checkpoint_validation_samples", 256)) < 1:
            raise ValueError("checkpoint_validation_samples must be positive")
        tolerance = float(resolved.get("checkpoint_quality_tolerance", 0.01))
        if not 0 <= tolerance < float("inf"):
            raise ValueError("checkpoint_quality_tolerance must be finite and nonnegative")
    except (TypeError, ValueError) as exc:
        return RowValidation(False, str(exc))
    if resolved.get("checkpoint_patch_operation_policy", "adaptive") not in {"adaptive", "fixed_xor", "fixed_replace"}:
        return RowValidation(False, "invalid checkpoint_patch_operation_policy")
    task_family = canonical_task_family(resolved.get("task_type"), resolved.get("hf_task"))
    vision_error = _validate_vision_row_requirements(resolved, task_family=task_family, training_regime=training_regime)
    if vision_error:
        return RowValidation(False, vision_error)
    blocked_reason = known_bad_row_reason(resolved)
    if blocked_reason:
        return RowValidation(False, blocked_reason)
    family_error = manifest_family_error(resolved)
    if family_error:
        return RowValidation(False, family_error)
    final_blocked_reason = _final_multimodal_row_reason(resolved)
    if final_blocked_reason:
        return RowValidation(False, final_blocked_reason)

    if str(resolved.get("dataset")).strip().lower() == "hf":
        if _is_blank(resolved.get("hf_model_id")):
            return RowValidation(False, "HF service rows require hf_model_id")
        hf_task = str(resolved.get("hf_task") or "").strip().lower().replace("-", "_")
        if _is_blank(hf_task):
            return RowValidation(False, "HF service rows require hf_task")
        if hf_task in {"seq2seq_generation", "text2text_generation"}:
            column_mapping = resolved.get("column_mapping") if isinstance(resolved.get("column_mapping"), dict) else {}
            source_col = column_mapping.get("source") or resolved.get("source_column") or resolved.get("text_column")
            target_col = column_mapping.get("target") or resolved.get("target_column") or resolved.get("label_column")
            if _is_blank(source_col) or _is_blank(target_col) or str(source_col) == str(target_col):
                return RowValidation(False, "seq2seq_generation requires distinct source and target columns")

    if str(resolved.get("modality") or "").strip().lower() == "multimodal":
        if _is_blank(resolved.get("image_column")):
            return RowValidation(False, "Multimodal service rows require image_column")
        if _is_blank(resolved.get("text_column")):
            return RowValidation(False, "Multimodal service rows require text_column")
    return RowValidation(True)


def _final_multimodal_row_reason(resolved: Mapping[str, Any]) -> str | None:
    hf_task = str(resolved.get("hf_task") or resolved.get("task") or "").strip().lower().replace("-", "_")
    training_regime = str(resolved.get("training_regime") or "").strip().lower()
    if training_regime == "inference_only":
        return None
    model_id = str(resolved.get("hf_model_id") or "").strip().lower()
    if hf_task == "image_captioning" and model_id.startswith("microsoft/git-"):
        return (
            "blocked final-run multimodal row: fine-tuned GIT image-captioning rows produced "
            "very low CIDEr metrics in this project"
        )
    if hf_task == "visual_question_answering" and (
        model_id.startswith("salesforce/blip")
        or model_id.startswith("microsoft/git-")
        or "blip" in model_id
        or "git" in model_id
    ):
        return (
            "blocked final-run multimodal row: fine-tuned generative VQA rows produced zero "
            "exact-match metrics in this project"
        )
    return None


def _manifest_requires_hf_preflight(enabled_df: pd.DataFrame) -> bool:
    for _, row in enabled_df.iterrows():
        dataset = str(_normalize_value(row.get("dataset")) or "").strip().lower()
        model_type = str(_normalize_value(row.get("model_type")) or "").strip().lower()
        hf_task = str(_normalize_value(row.get("hf_task")) or "").strip().lower()
        if dataset == "hf" or model_type.startswith("hf") or hf_task:
            return True
    return False


def _row_requires_keras_preflight(row: pd.Series | dict[str, Any]) -> bool:
    dataset = str(_normalize_value(row.get("dataset")) or "").strip().lower()
    model_type = str(_normalize_value(row.get("model_type")) or "").strip().lower()
    if dataset in KERAS_DATASETS:
        return True
    if dataset in {"hf", "huggingface"} or model_type.startswith("hf"):
        return False
    return model_type in KERAS_MODEL_TYPES


def _manifest_requires_keras_preflight(enabled_df: pd.DataFrame) -> bool:
    for _, row in enabled_df.iterrows():
        if _row_requires_keras_preflight(row):
            return True
    return False


def _ensure_manifest_preflight(enabled_df: pd.DataFrame) -> None:
    required_imports = {}
    if _manifest_requires_hf_preflight(enabled_df):
        required_imports.update(
            {
                "datasets": "Hugging Face dataset loading requires the 'datasets' package. Install it with: pip install datasets",
                "transformers": "Manifest preflight failed: required HF dependency 'transformers' is not importable. Install it with: pip install transformers",
            }
        )
    if _manifest_requires_keras_preflight(enabled_df):
        required_imports.update(
            {
                "tensorflow": "Keras-backed generic service rows require TensorFlow. Install it with: pip install tensorflow",
                "keras": "Keras-backed generic service rows require Keras. Install it with: pip install keras",
            }
        )
    for package_name, message in required_imports.items():
        try:
            importlib.import_module(package_name)
        except Exception as exc:  # noqa: BLE001
            raise ManifestPreflightError(message) from exc


def _lookup_registry_dataset_counts(resolved: dict[str, Any]) -> tuple[int | None, int | None]:
    dataset_name = str(resolved.get("dataset_name") or "").strip()
    dataset_config = resolved.get("dataset_config")
    if not dataset_name:
        return None, None
    for spec in DATASET_REGISTRY.values():
        if str(spec.get("dataset_name") or "").strip() != dataset_name:
            continue
        if spec.get("dataset_config") != dataset_config:
            continue
        train_examples = spec.get("train_examples")
        benchmark_examples = spec.get("benchmark_examples")
        return (
            int(train_examples) if train_examples is not None else None,
            int(benchmark_examples) if benchmark_examples is not None else None,
        )
    return None, None


def _validate_vision_row_requirements(resolved: dict[str, Any], *, task_family: str, training_regime: str) -> str | None:
    if training_regime == "inference_only":
        return None
    minimums = {
        "detection": (32, 16),
        "segmentation": (32, 16),
    }
    required = minimums.get(task_family)
    if required is not None:
        train_examples = resolved.get("train_examples")
        benchmark_examples = resolved.get("benchmark_examples")
        if train_examples is None or benchmark_examples is None:
            train_examples, benchmark_examples = _lookup_registry_dataset_counts(resolved)
        if train_examples is not None and benchmark_examples is not None:
            min_train, min_benchmark = required
            if int(train_examples) < min_train or int(benchmark_examples) < min_benchmark:
                return (
                    f"{task_family} finetune services require at least {min_train} train examples and "
                    f"{min_benchmark} benchmark examples; got train={train_examples}, "
                    f"benchmark={benchmark_examples}"
                )
    model_id = str(resolved.get("hf_model_id") or "").strip().lower()
    if task_family == "segmentation" and model_id.startswith("openmmlab/upernet-") and int(resolved.get("batch_size", 0) or 0) < 2:
        return f"{resolved.get('hf_model_id')} requires batch_size >= 2 for finetune segmentation runs"
    return None


def _is_enabled(row: pd.Series) -> bool:
    if "enabled" not in row.index:
        return True
    raw = _normalize_value(row.get("enabled"))
    return True if raw is None else _to_bool(raw)


def _format_traceback(exc) -> str | None:
    if exc is None:
        return None
    return bounded_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip())


def _write_failure_log(log_path: Path, *, row_index, service_id, case_name, manifest_group_id, failure_stage, error_message, resolved, exc=None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 100,
        f"timestamp: {datetime.now().isoformat()}",
        f"row_index: {row_index}",
        f"service_id: {service_id}",
        f"case_name: {case_name}",
        f"manifest_group_id: {manifest_group_id}",
        f"failure_stage: {failure_stage}",
        f"error_message: {bounded_text(str(error_message))}",
        "resolved_config:",
        diagnostic_json(resolved),
    ]
    if exc is not None:
        lines.extend(["traceback:", _format_traceback(exc) or ""])
    lines.append("")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_failure_db(resolved: dict[str, Any], *, row_index, service_id, case_name, manifest_group_id, failure_stage, error_message, exc=None) -> None:
    if service_id is None or not str(service_id):
        # Manifest-level preflight errors have no service identity.
        return
    config = {**resolved, "service_id": str(service_id), "row_index": _safe_int(row_index),
              "case_name": case_name, "manifest_group_id": manifest_group_id}
    persist_terminal(config, status="failed", reason=error_message,
                     failure_stage=str(failure_stage), exc=exc)


def _clear_cuda_runtime_state() -> None:
    try:
        import torch
    except Exception:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return
    try:
        if not cuda.is_available():
            return
    except Exception:
        return
    for name in ("synchronize", "empty_cache", "ipc_collect"):
        fn = getattr(cuda, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _record_entry_failure(entry: ManifestEntry, *, failure_stage: str, error_message: Any, exc=None,
                          write_log: bool = True) -> dict[str, Any]:
    resolved = entry.resolved
    service_id = resolved.get("service_id")
    error_text = str(error_message) if error_message is not None else ""
    if write_log:
        _write_failure_log(
            FAILURE_LOG_PATH,
            row_index=entry.idx,
            service_id=service_id,
            case_name=resolved.get("case_name"),
            manifest_group_id=resolved.get("manifest_group_id"),
            failure_stage=failure_stage,
            error_message=error_text,
            resolved=resolved,
            exc=exc,
        )
    _write_failure_db(
        resolved,
        row_index=entry.idx,
        service_id=service_id,
        case_name=resolved.get("case_name"),
        manifest_group_id=resolved.get("manifest_group_id"),
        failure_stage=failure_stage,
        error_message=error_text,
        exc=exc,
    )
    result = _result_row(resolved, entry.idx, "failed", error_text, service_id=str(service_id or ""))
    _record_manifest_progress_result(result)
    return result


def _manifest_csv_value(value: Any) -> Any:
    value = _normalize_value(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return value


def _failed_result_row_indexes(results: list[dict[str, Any]]) -> list[int]:
    failed: list[int] = []
    seen: set[int] = set()
    for result in _sort_result_rows(results):
        if str(result.get("status") or "").strip().lower() in {"success", "skipped"}:
            continue
        row_index = result.get("row_index")
        if row_index is None or _is_blank(row_index):
            continue
        try:
            parsed = int(row_index)
        except Exception:
            continue
        if parsed not in seen:
            failed.append(parsed)
            seen.add(parsed)
    return failed


def _resolved_from_result(result: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("resolved_config_json")
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write_failed_manifest_csv(
    *,
    rows_df: pd.DataFrame,
    manifest_defaults: dict[str, Any],
    results: list[dict[str, Any]],
    output_path: Path | None = None,
    db_path: str | None = None,
) -> Path | None:
    output_path = output_path or FAILED_MANIFEST_PATH
    failed_indexes = _failed_result_row_indexes(results)
    if failed_indexes and "cohort_id" in rows_df:
        cohorts = set(rows_df.loc[rows_df.index.intersection(failed_indexes), "cohort_id"].dropna())
        failed_indexes = sorted(set(failed_indexes) | set(rows_df.index[rows_df["cohort_id"].isin(cohorts)]))
    if not failed_indexes:
        if output_path.exists():
            output_path.unlink()
            print(f"Removed stale failed-row manifest: {output_path}")
        return None

    result_by_index: dict[int, dict[str, Any]] = {}
    for result in results:
        row_index = result.get("row_index")
        if row_index is None or _is_blank(row_index):
            continue
        try:
            result_by_index[int(row_index)] = result
        except Exception:
            continue

    columns = list(rows_df.columns)
    family_reference_fields = ("family_document_ref", "family_document_hash", "hf_model_revision", "hf_dataset_revision")
    if any(_resolved_from_result(result).get("family_document_ref") for result in results):
        columns.extend(field for field in family_reference_fields if field not in columns)
    for column in manifest_defaults:
        if column not in columns:
            columns.append(column)
    if "service_id" not in columns:
        columns.insert(0, "service_id")
    if db_path is not None and "db_path" not in columns:
        columns.append("db_path")

    retry_rows: list[dict[str, Any]] = []
    for row_index in failed_indexes:
        if row_index not in rows_df.index:
            continue
        source = rows_df.loc[row_index].to_dict()
        resolved = _resolved_from_result(result_by_index.get(row_index, {}))
        retry_row: dict[str, Any] = {}
        for column in columns:
            value = source.get(column)
            if column in family_reference_fields and _is_blank(value):
                value = resolved.get(column)
            if _is_blank(value) and column in manifest_defaults:
                value = manifest_defaults[column]
            if column == "service_id" and _is_blank(value):
                value = resolved.get("service_id") or result_by_index.get(row_index, {}).get("service_id")
            if column == "db_path" and _is_blank(value) and db_path is not None:
                value = db_path
            retry_row[column] = _manifest_csv_value(value)
        retry_rows.append(retry_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(retry_rows, columns=columns).to_csv(output_path, index=False)
    print(f"Wrote failed-row retry manifest: {output_path} ({len(retry_rows)} rows)")
    return output_path


def _multimodal_task(value: Any) -> str:
    task = str(value or "").strip().lower().replace("-", "_")
    return task if task in MULTIMODAL_TASKS else ""


def _row_multimodal_task(row: Mapping[str, Any]) -> str:
    for key in ("hf_task", "task", "task_type"):
        task = _multimodal_task(row.get(key))
        if task:
            return task
    if str(row.get("modality") or "").strip().lower() == "multimodal":
        return _multimodal_task(row.get("task_type"))
    return ""


def _service_ids_from_multimodal_rows(rows_df: pd.DataFrame, manifest_defaults: dict[str, Any]) -> set[str]:
    service_ids: set[str] = set()
    for _, row in rows_df.iterrows():
        source = row.to_dict()
        merged = dict(manifest_defaults)
        merged.update({key: value for key, value in source.items() if not _is_blank(value)})
        if not _row_multimodal_task(merged):
            continue
        service_id = merged.get("service_id")
        if not _is_blank(service_id):
            service_ids.add(str(service_id))
    return service_ids


def _fetch_multimodal_service_audit_rows(db_path: str | None, service_ids: set[str] | None = None) -> list[dict[str, Any]]:
    if not db_path:
        return []
    path = Path(str(db_path))
    if not path.exists():
        return []
    placeholders = ""
    params: list[Any] = []
    service_filter = ""
    if service_ids:
        placeholders = ",".join("?" for _ in service_ids)
        service_filter = f" AND s.service_id IN ({placeholders})"
        params.extend(sorted(service_ids))
    sql = f"""
    WITH m AS (
        SELECT
            service_id,
            metric_name,
            COALESCE(value_num, CAST(value_int AS REAL), CAST(value_bool AS REAL)) AS num_value
        FROM service_metrics
    ),
    metric_values AS (
        SELECT
            service_id,
            MAX(CASE WHEN metric_name = 'metric_score' THEN num_value END) AS metric_score,
            MAX(CASE WHEN metric_name = 'r@1' THEN num_value END) AS recall_at_1,
            MAX(CASE WHEN metric_name = 'r@5' THEN num_value END) AS recall_at_5,
            MAX(CASE WHEN metric_name = 'cider' THEN num_value END) AS cider,
            MAX(CASE WHEN metric_name = 'bleu' THEN num_value END) AS bleu,
            MAX(CASE WHEN metric_name = 'exact_match' THEN num_value END) AS exact_match,
            MAX(CASE WHEN metric_name = 'answer_token_accuracy' THEN num_value END) AS answer_token_accuracy
        FROM m
        GROUP BY service_id
    ),
    latest_failures AS (
        SELECT sf.service_id, sf.error_message
        FROM service_failures sf
        JOIN (
            SELECT service_id, MAX(failure_id) AS failure_id
            FROM service_failures
            WHERE service_id IS NOT NULL
            GROUP BY service_id
        ) latest ON latest.failure_id = sf.failure_id
    )
    SELECT
        s.service_id,
        s.status,
        s.case_name,
        s.hf_task,
        s.model_id,
        s.dataset_name,
        s.training_regime,
        mv.metric_score,
        mv.recall_at_1,
        mv.recall_at_5,
        mv.cider,
        mv.bleu,
        mv.exact_match,
        mv.answer_token_accuracy,
        lf.error_message
    FROM services s
    LEFT JOIN metric_values mv ON mv.service_id = s.service_id
    LEFT JOIN latest_failures lf ON lf.service_id = s.service_id
    WHERE s.hf_task IN ('image_captioning', 'text_image_retrieval', 'visual_question_answering')
    {service_filter}
    ORDER BY s.hf_task, s.status, s.service_id
    """
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(sql, params)]
        conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to audit multimodal services from SQLite: {exc}")
        return []
    return rows


def _multimodal_audit_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    status = str(row.get("status") or "").strip().lower()
    task = _multimodal_task(row.get("hf_task"))
    if status != "completed":
        reasons.append("status_not_completed")
    metric_score = row.get("metric_score")
    if metric_score is not None:
        try:
            if float(metric_score) <= 0.05:
                reasons.append("metric_score_lte_0.05")
        except Exception:
            pass
    if task == "text_image_retrieval" and row.get("recall_at_1") is not None:
        try:
            if float(row.get("recall_at_1")) <= 0.02:
                reasons.append("r@1_lte_0.02")
        except Exception:
            pass
    if task == "visual_question_answering" and row.get("exact_match") is not None:
        try:
            if float(row.get("exact_match")) == 0.0:
                reasons.append("exact_match_eq_0")
        except Exception:
            pass
    return reasons


def _write_multimodal_audit_csv(
    *,
    rows_df: pd.DataFrame,
    manifest_defaults: dict[str, Any],
    db_path: str | None,
    output_path: Path | None = None,
) -> Path | None:
    output_path = output_path or MULTIMODAL_AUDIT_PATH
    service_ids = _service_ids_from_multimodal_rows(rows_df, manifest_defaults)
    audit_rows = []
    for row in _fetch_multimodal_service_audit_rows(db_path, service_ids or None):
        reasons = _multimodal_audit_reasons(row)
        if not reasons:
            continue
        audit_rows.append({**row, "audit_reasons": ",".join(reasons)})
    if not audit_rows:
        if output_path.exists():
            output_path.unlink()
            print(f"Removed stale multimodal audit report: {output_path}")
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(output_path, index=False)
    print(f"Wrote multimodal audit report: {output_path} ({len(audit_rows)} rows)")
    return output_path


def _write_missing_multimodal_manifest_csv(
    *,
    rows_df: pd.DataFrame,
    manifest_defaults: dict[str, Any],
    db_path: str | None,
    output_path: Path | None = None,
) -> Path | None:
    output_path = output_path or MISSING_MULTIMODAL_MANIFEST_PATH
    service_rows = {
        str(row.get("service_id")): row
        for row in _fetch_multimodal_service_audit_rows(db_path, _service_ids_from_multimodal_rows(rows_df, manifest_defaults) or None)
        if row.get("service_id") is not None
    }
    missing_rows: list[dict[str, Any]] = []
    for _, row in rows_df.iterrows():
        source = row.to_dict()
        merged = dict(manifest_defaults)
        merged.update({key: value for key, value in source.items() if not _is_blank(value)})
        if not _row_multimodal_task(merged):
            continue
        service_id = str(merged.get("service_id") or "")
        db_row = service_rows.get(service_id)
        if db_row is not None and str(db_row.get("status") or "").strip().lower() == "completed":
            continue
        missing_rows.append({column: _manifest_csv_value(source.get(column, manifest_defaults.get(column))) for column in rows_df.columns})
    if not missing_rows:
        if output_path.exists():
            output_path.unlink()
            print(f"Removed stale missing multimodal manifest: {output_path}")
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing_rows, columns=list(rows_df.columns)).to_csv(output_path, index=False)
    print(f"Wrote missing multimodal manifest: {output_path} ({len(missing_rows)} rows)")
    return output_path


@terminal_scope(lambda *args, **kwargs: {"log_mode": kwargs.get("log_mode")})
@preparation_cache_scope()
def run_manifest(
    file: str,
    sheet: str = "services",
    dry_run: bool = False,
    db_path: str | None = None,
    *,
    grouped_hf: bool = True,
    workers: int = 1,
    rerun_existing: bool = False,
    log_mode: str | None = None,
    no_save_weights: bool = False,
    smoke_test: bool = False,
) -> Path:
    global _ACTIVE_MANIFEST_PROGRESS
    manifest_path = Path(file)
    load_hf_token_from_file()
    print(f"Loading manifest file: {manifest_path}")
    rows_df, manifest_defaults = load_manifest(manifest_path, sheet=sheet)
    if smoke_test:
        rows_df = rows_df.copy()
        rows_df["smoke_test"] = True
        print("Smoke test: perturbation and update signatures disabled; service IDs use __smoke_test.")
    if no_save_weights:
        rows_df = rows_df.copy()
        rows_df["no_save_weights"] = True
        print("Debug mode: no weight checkpoints; metrics retained; service IDs use __debug_no_weights.")
    shared_manifest = "generation_mode" in rows_df and rows_df["generation_mode"].eq("shared_parent").any()
    results_path = manifest_path.with_name(manifest_path.stem + "_results.csv") if shared_manifest else MANIFEST_RESULTS_PATH
    failed_path = manifest_path.with_name(manifest_path.stem + "_failed.csv") if shared_manifest else FAILED_MANIFEST_PATH

    enabled_df = rows_df[rows_df.apply(_is_enabled, axis=1)].copy()
    print(f"Loaded manifest: {manifest_path}")
    print(f"Total rows: {len(rows_df)}")
    print(f"Enabled services: {len(enabled_df)}")

    manifest_group_id = str(uuid.uuid4())
    results: list[dict[str, Any]] = []
    entries: list[ManifestEntry] = []
    _ACTIVE_MANIFEST_PROGRESS = ManifestProgressTracker(total=len(enabled_df))
    recovery_db_paths = _fedavg_recovery_db_paths(
        enabled_df,
        manifest_defaults,
        db_path_override=db_path,
    )

    try:
        if not dry_run:
            for recovery_path in _service_recovery_paths(recovery_db_paths):
                recover_service_states(recovery_path)
                reconcile_manifest_roster(recovery_path)
            _recover_fedavg_databases(
                recovery_db_paths,
                reason="interrupted process detected during manifest startup recovery",
            )
        if not dry_run:
            roster = []
            for idx, row in enabled_df.iterrows():
                config = _resolve_execution_row(row, manifest_defaults)
                config.update(db_path=db_path or config.get("db_path"), manifest_group_id=manifest_group_id,
                              _execution_run_id=manifest_group_id, row_index=int(idx))
                roster.append(config)
            register_manifest_roster(roster, run_id=manifest_group_id, manifest_path=manifest_path, rerun_existing=rerun_existing)
        if not dry_run:
            try:
                _ensure_manifest_preflight(enabled_df)
            except ManifestPreflightError as exc:
                preflight_cache = {}
                for index, (idx, row) in enumerate(enabled_df.iterrows()):
                    resolved = _resolve_execution_row(row, manifest_defaults)
                    resolved.update(db_path=db_path or resolved.get("db_path"), manifest_group_id=manifest_group_id,
                                    _execution_run_id=manifest_group_id, row_index=int(idx))
                    existing = None if rerun_existing else _find_existing_service_database(resolved["db_path"], resolved["service_id"], preflight_cache)
                    if existing:
                        results.append(_result_row(resolved, int(idx), "skipped", "already completed", service_id=resolved["service_id"]))
                        continue
                    entry = ManifestEntry(idx=int(idx), ordinal=index + 1, resolved=resolved, validation=RowValidation(False, str(exc)))
                    results.append(_record_entry_failure(entry, failure_stage="manifest_preflight", error_message=str(exc), exc=exc, write_log=index == 0))
                output_path = results_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(results).to_csv(output_path, index=False)
                _write_failed_manifest_csv(
                    rows_df=rows_df,
                    manifest_defaults=manifest_defaults,
                    results=results,
                    output_path=failed_path,
                    db_path=db_path,
                )
                for recovery_path in _service_recovery_paths(recovery_db_paths):
                    reconcile_manifest_roster(recovery_path, run_id=manifest_group_id)
                print(f"Manifest preflight failed: {exc}")
                print(f"Wrote results: {output_path}")
                return output_path

        entries: list[ManifestEntry] = []
        hf_revision_cache: dict[tuple[str, str, str | None], tuple[str | None, str | None]] = {}
        family_reference_cache = {}
        existing_service_cache: dict[str, set[str]] = {}
        preparation_update = time.monotonic()
        print(f"Preparing manifest: 0/{len(enabled_df)} rows checked", flush=True)
        for i, (idx, row) in enumerate(enabled_df.iterrows(), start=1):
            if time.monotonic() - preparation_update >= 10:
                print(f"Preparing manifest: {i - 1}/{len(enabled_df)} rows checked", flush=True)
                preparation_update = time.monotonic()
            resolved = _resolve_execution_row(row, manifest_defaults)
            if log_mode is not None:
                resolved["log_mode"] = log_mode
            resolved["_execution_run_id"] = manifest_group_id
            resolved["row_index"] = int(idx)
            resolved["_manifest_total"] = len(enabled_df)
            resolved["_manifest_ordinal"] = i
            if db_path is not None:
                resolved["db_path"] = db_path
            if _is_blank(resolved.get("manifest_group_id")):
                resolved["manifest_group_id"] = manifest_group_id
            if not _is_blank(resolved.get("fl_run_id")) and _is_blank(resolved.get("lineage_id")):
                resolved["lineage_id"] = f"lineage__{resolved['fl_run_id']}"
            if not dry_run and not rerun_existing:
                existing_db = _find_existing_service_database(
                    resolved.get("db_path"), resolved.get("service_id"), existing_service_cache
                )
                if existing_db is not None:
                    resolved["_existing_service_db_path"] = existing_db
                    entries.append(ManifestEntry(idx=int(idx), ordinal=i, resolved=resolved, validation=RowValidation(True)))
                    continue
            from .manifest.preparation import bind_manifest, freeze_manifest_revisions
            try:
                if resolved.get("_row_resolution_error"):
                    raise ValueError(resolved["_row_resolution_error"])
                if not dry_run:
                    bind_manifest(resolved, file, cache=family_reference_cache)
                if not dry_run and "_existing_service_db_path" not in resolved and resolved.get("generation_mode") != "shared_parent":
                    if not resolved.get("_family_reference") and not resolved.get("_frozen_revisions"):
                        _resolve_manifest_hf_revisions(resolved, hf_revision_cache)
                    freeze_manifest_revisions(resolved)
                    resolved["dataset_args"] = _build_dataset_args(resolved)
            except (ValueError, OSError, KeyError, RuntimeError) as exc:
                entries.append(ManifestEntry(idx=int(idx), ordinal=i, resolved=resolved, validation=RowValidation(False, str(exc))))
                continue
            entries.append(ManifestEntry(idx=int(idx), ordinal=i, resolved=resolved, validation=_validate_row(resolved)))

        print(f"Preparing manifest: {len(enabled_df)}/{len(enabled_df)} rows checked", flush=True)
        if shared_manifest:
            from ..cohort_manifest import validate_cohort_entries
            shared_entries = [e for e in entries if e.resolved.get("generation_mode") == "shared_parent"
                              and "_existing_service_db_path" not in e.resolved]
            for _, group in _group_entries(shared_entries, lambda row: row.get("cohort_id")):
                try:
                    validate_cohort_entries(group)
                except (ValueError, KeyError, TypeError) as exc:
                    for entry in group:
                        entry.validation = RowValidation(False, str(exc))

        # FedAvg is stateful across every client-round row. Avoid executing an
        # incomplete run when any member was protected by the existence check.
        existing_fedavg_runs = {
            str(entry.resolved.get("fl_run_id"))
            for entry in entries
            if "_existing_service_db_path" in entry.resolved
            and not _is_blank(entry.resolved.get("fl_run_id"))
        }
        for entry in entries:
            fl_run_id = str(entry.resolved.get("fl_run_id") or "")
            if fl_run_id in existing_fedavg_runs:
                entry.resolved.setdefault(
                    "_existing_service_db_path",
                    f"FedAvg run {fl_run_id} already has service rows in SQLite",
                )

        valid_entries: list[ManifestEntry] = []
        for entry in entries:
            resolved = entry.resolved
            service_id = resolved["service_id"]
            debug_print(f"\nService {entry.ordinal}/{len(enabled_df)}: {service_id}")
            debug_print(
                f"dataset={resolved.get('dataset')} "
                f"task={resolved.get('task_type')} "
                f"model={resolved.get('model_type')} "
                f"training_regime={resolved.get('training_regime')} "
                f"db={resolved.get('db_path')}"
            )

            if not entry.validation.ok:
                _manifest_progress_start("main", f"validate row {entry.ordinal}/{len(enabled_df)}")
                result = _result_row(resolved, entry.idx, "failed", entry.validation.error, service_id=service_id)
                results.append(result)
                _record_manifest_progress_result(result)
                _manifest_progress_clear("main")
                _write_failure_log(
                    FAILURE_LOG_PATH,
                    row_index=entry.idx,
                    service_id=service_id,
                    case_name=resolved.get("case_name"),
                    manifest_group_id=resolved.get("manifest_group_id"),
                    failure_stage="validation_failed",
                    error_message=entry.validation.error,
                    resolved=resolved,
                )
                if not dry_run:
                    _write_failure_db(
                        resolved,
                        row_index=entry.idx,
                        service_id=service_id,
                        case_name=resolved.get("case_name"),
                        manifest_group_id=resolved.get("manifest_group_id"),
                        failure_stage="validation_failed",
                        error_message=entry.validation.error,
                    )
                print(f"Skipping row {entry.idx}: {entry.validation.error}")
                continue

            existing_db = resolved.get("_existing_service_db_path")
            if existing_db is not None:
                _manifest_progress_start("main", f"skip existing row {entry.ordinal}/{len(enabled_df)}")
                message = f"Service already exists in SQLite database: {existing_db}"
                result = _result_row(resolved, entry.idx, "skipped", message, service_id=service_id)
                results.append(result)
                _record_manifest_progress_result(result)
                _manifest_progress_clear("main")
                print(f"Skipping existing row {entry.ordinal}: {existing_db}")
                debug_print(f"service_id={service_id}")
                continue

            if dry_run:
                _manifest_progress_start("main", f"dry-run row {entry.ordinal}/{len(enabled_df)}")
                print(diagnostic_json(resolved))
                result = _result_row(resolved, entry.idx, "success", "", service_id=service_id)
                results.append(result)
                _record_manifest_progress_result(result)
                _manifest_progress_clear("main")
                continue

            valid_entries.append(entry)

        if not dry_run:
            fedavg_entries = [entry for entry in valid_entries if not _is_blank(entry.resolved.get("fl_run_id"))]
            cohort_entries = [entry for entry in valid_entries if entry.resolved.get("generation_mode") == "shared_parent"]
            other_entries = [entry for entry in valid_entries if _is_blank(entry.resolved.get("fl_run_id")) and entry.resolved.get("generation_mode") != "shared_parent"]
            if cohort_entries:
                from ..shared_parent import execute_cohort
                for _, group in _group_entries(cohort_entries, lambda row: row.get("cohort_id")):
                    try:
                        cohort_results = execute_cohort(group)
                        by_id = {r["service_id"]: r for r in cohort_results}
                        for entry in group:
                            outcome = by_id[entry.resolved["service_id"]]
                            results.append(_result_row(entry.resolved, entry.idx, outcome["status"], outcome["error_message"], service_id=outcome["service_id"]))
                            _record_manifest_progress_result(results[-1])
                    except Exception as exc:
                        for entry in group:
                            results.append(_record_entry_failure(entry, failure_stage="cohort_preparation", error_message=str(exc), exc=exc))
                        print(f"Cohort failed preparation: {exc}")
            fedavg_summaries: list[dict[str, Any]] = []
            if fedavg_entries:
                fedavg_results, fedavg_summaries = _execute_fedavg_entries(fedavg_entries)
                results.extend(fedavg_results)
            if other_entries:
                results.extend(
                    _execute_entries_grouped_hf(other_entries, workers=workers)
                    if grouped_hf
                    else _execute_entries_row_local(other_entries, workers=workers)
                )
            if fedavg_summaries:
                FEDAVG_RUN_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(fedavg_summaries).to_csv(FEDAVG_RUN_RESULTS_PATH, index=False)
                print(f"Wrote fedavg run summaries: {FEDAVG_RUN_RESULTS_PATH}")

        if not dry_run:
            _finalize_manifest_results(entries, results)
            for recovery_path in _service_recovery_paths(recovery_db_paths):
                reconcile_manifest_roster(recovery_path, run_id=manifest_group_id)
        output_path = results_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_results = _sort_result_rows(results)
        pd.DataFrame(sorted_results).to_csv(output_path, index=False)
        _write_failed_manifest_csv(
            rows_df=rows_df,
            manifest_defaults=manifest_defaults,
            results=sorted_results,
            output_path=failed_path,
            db_path=db_path,
        )
        if not dry_run and not shared_manifest:
            effective_db_path = db_path or CONFIG.get("db_path")
            _write_multimodal_audit_csv(
                rows_df=rows_df,
                manifest_defaults=manifest_defaults,
                db_path=effective_db_path,
            )
            _write_missing_multimodal_manifest_csv(
                rows_df=rows_df,
                manifest_defaults=manifest_defaults,
                db_path=effective_db_path,
            )
        print(f"Wrote results: {output_path}")
        return output_path
    except BaseException as exc:
        if not dry_run:
            for recovery_path in _service_recovery_paths(recovery_db_paths):
                for recover in (recover_service_states, reconcile_manifest_roster):
                    try:
                        recover(recovery_path, run_id=manifest_group_id, reason=f"manifest shutdown: {type(exc).__name__}")
                    except Exception as recovery_error:
                        # Retain the roster/journals for restart and the original
                        # exception; continue recovery in the other databases.
                        print(f"Shutdown recovery deferred for {recovery_path}: {bounded_text(str(recovery_error))}", flush=True)
            _recover_fedavg_databases(
                recovery_db_paths,
                reason=f"manifest shutdown interrupted execution: {type(exc).__name__}",
                best_effort=True,
            )
        raise
    finally:
        if _ACTIVE_MANIFEST_PROGRESS is not None:
            _ACTIVE_MANIFEST_PROGRESS.finish()
        _ACTIVE_MANIFEST_PROGRESS = None


def _service_recovery_paths(paths):
    return sorted({str(candidate) for path in paths for candidate in _database_candidates(path)})


def _finalize_manifest_results(entries, results):
    reported = {r["service_id"]: r for r in results}
    for entry in entries:
        outcome = reported.get(entry.resolved["service_id"])
        if outcome is None:
            results.append(_record_entry_failure(entry, failure_stage="missing_terminal_result",
                           error_message="executor returned without a terminal result"))
            continue
        if outcome["status"] == "skipped":
            continue
        config = {**entry.resolved, **_resolved_from_result(outcome)}
        expected = "completed" if outcome["status"] == "success" else outcome["status"]
        db = config.get("db_path") or entry.resolved.get("db_path")
        actual = None
        if db and Path(db).is_file():
            with sqlite3.connect(db) as conn:
                if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='services'").fetchone():
                    actual = conn.execute("SELECT status FROM services WHERE service_id=?", (outcome["service_id"],)).fetchone()
        if actual is None or actual[0] != expected:
            if expected in FEDAVG_OBSERVATION_STATUSES:
                service_runner.persist_status_only_service(config, status=expected, reason=outcome.get("error_message"))
            else:
                persist_terminal({**config, "service_id": outcome["service_id"]}, status=expected,
                                 reason=outcome.get("error_message"), failure_stage="executor_result")


def _fedavg_recovery_db_paths(
    rows_df: pd.DataFrame,
    manifest_defaults: Mapping[str, Any],
    *,
    db_path_override: str | None,
) -> list[str]:
    """Resolve every database this manifest can write before execution starts."""
    if db_path_override is not None:
        return [str(db_path_override)]
    fallback = manifest_defaults.get("db_path") or CONFIG.get("db_path") or "mlaas.db"
    paths: set[str] = set()
    for _, row in rows_df.iterrows():
        row_path = row.get("db_path") if "db_path" in rows_df.columns else None
        paths.add(str(fallback if _is_blank(row_path) else row_path))
    if not paths:
        paths.add(str(fallback))
    return sorted(paths)


def _recover_fedavg_databases(
    paths: list[str],
    *,
    reason: str,
    best_effort: bool = False,
) -> None:
    for path in paths:
        try:
            recovered = recover_abandoned_fedavg_runs(path, reason=reason)
        except Exception as exc:  # shutdown cleanup must not mask the original exit
            if not best_effort:
                raise
            print(
                f"[FedAvg recovery] db={path} | cleanup_error={type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        if recovered["runs"] or recovered["rounds"]:
            print(
                "[FedAvg recovery] "
                f"db={path} | interrupted_runs={recovered['runs']} "
                f"| interrupted_rounds={recovered['rounds']} | reason={reason}",
                flush=True,
            )


def _execute_entries_row_local(entries: list[ManifestEntry], *, workers: int = 1) -> list[dict[str, Any]]:
    worker_count = _resolve_parallel_worker_count(workers)
    if worker_count <= 1 or len(entries) <= 1:
        results: list[dict[str, Any]] = []
        for entry in entries:
            _manifest_progress_start("main", _describe_entry(entry))
            results.append(_execute_entry_row_local(entry))
            _manifest_progress_clear("main")
        return results

    gpu_slots = _detect_cuda_device_ids()[:worker_count]
    auto_gpu_entries = [entry for entry in entries if _entry_supports_auto_gpu_affinity(entry.resolved)]
    fixed_entries = [entry for entry in entries if not _entry_supports_auto_gpu_affinity(entry.resolved)]

    if not gpu_slots or len(auto_gpu_entries) <= 1:
        results = []
        for entry in entries:
            _manifest_progress_start("main", _describe_entry(entry))
            results.append(_execute_entry_row_local(entry))
            _manifest_progress_clear("main")
        return results

    print(
        "Parallel manifest execution enabled: "
        f"workers={len(gpu_slots)} visible_gpus={','.join(str(gpu) for gpu in gpu_slots)} "
        f"auto_gpu_rows={len(auto_gpu_entries)} fixed_rows={len(fixed_entries)}"
    )
    results = _execute_entries_with_gpu_affinity(auto_gpu_entries, gpu_slots)
    for entry in fixed_entries:
        _manifest_progress_start("main", _describe_entry(entry))
        results.append(_execute_entry_row_local(entry))
        _manifest_progress_clear("main")
    return results


def _resolve_parallel_worker_count(workers: int | None) -> int:
    try:
        parsed = int(workers or 1)
    except Exception:
        return 1
    return max(1, parsed)


def _detect_cuda_device_ids() -> list[int]:
    try:
        import torch
    except Exception:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        count = int(torch.cuda.device_count() or 0)
    except Exception:
        return []
    return list(range(max(0, count)))


def _normalize_requested_device(value: Any) -> str:
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan"}:
        return "auto"
    return text


def _entry_supports_auto_gpu_affinity(resolved: dict[str, Any]) -> bool:
    requested = _normalize_requested_device(resolved.get("device"))
    return requested in {"auto", "gpu", "auto_gpu", "cuda_auto", "cuda"}


def _worker_initializer(cuda_visible_devices: int | None, progress_queue: Any = None) -> None:
    global _WORKER_PROGRESS_QUEUE, _WORKER_LABEL
    start_preparation_cache()
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    _WORKER_PROGRESS_QUEUE = progress_queue
    _WORKER_LABEL = _worker_label_for_gpu(cuda_visible_devices)


def _worker_entry_payload(entry: ManifestEntry) -> dict[str, Any]:
    return {
        "idx": int(entry.idx),
        "ordinal": int(entry.ordinal),
        "resolved": dict(entry.resolved),
    }


def _db_path_for_gpu(db_path: Any, gpu_id: int) -> str:
    raw = str(db_path or CONFIG.get("db_path") or "").strip()
    if not raw:
        return raw
    path = Path(raw)
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    gpu_name = f"{stem}.gpu{int(gpu_id)}{suffix}"
    return str(path.with_name(gpu_name))


def _entry_with_gpu_db_path(entry: ManifestEntry, gpu_id: int) -> ManifestEntry:
    resolved = dict(entry.resolved)
    resolved["db_path"] = _db_path_for_gpu(resolved.get("db_path"), gpu_id)
    return ManifestEntry(
        idx=int(entry.idx),
        ordinal=int(entry.ordinal),
        resolved=resolved,
        validation=entry.validation,
    )


def _execute_entry_worker(payload: dict[str, Any]) -> dict[str, Any]:
    entry = ManifestEntry(
        idx=int(payload["idx"]),
        ordinal=int(payload.get("ordinal", 0)),
        resolved=dict(payload.get("resolved") or {}),
        validation=RowValidation(True),
    )
    return _execute_entry_row_local(entry)


def _execute_entries_with_gpu_affinity(entries: list[ManifestEntry], gpu_slots: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    mp_context = multiprocessing.get_context("spawn")
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    executors = [
        concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp_context,
            initializer=_worker_initializer,
            initargs=(gpu_id, progress_queue),
        )
        for gpu_id in gpu_slots
    ]
    pending_entries = list(entries)
    future_map: dict[concurrent.futures.Future, tuple[ManifestEntry, int]] = {}
    try:
        for executor, gpu_id in zip(executors, gpu_slots, strict=False):
            if not pending_entries:
                break
            entry = pending_entries.pop(0)
            gpu_entry = _entry_with_gpu_db_path(entry, gpu_id)
            _manifest_progress_start(_worker_label_for_gpu(gpu_id), _describe_entry(gpu_entry))
            future = executor.submit(_execute_entry_worker, _worker_entry_payload(gpu_entry))
            future_map[future] = (gpu_entry, gpu_id)
        while future_map:
            done, _ = concurrent.futures.wait(
                list(future_map),
                timeout=0.2,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            _drain_progress_queue(progress_queue)
            if not done:
                continue
            for future in done:
                entry, gpu_id = future_map.pop(future)
                worker_label = _worker_label_for_gpu(gpu_id)
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    _manifest_progress_clear(worker_label)
                    _manifest_progress_start("main", _describe_entry(entry))
                    results.append(_execute_entry_row_local(entry))
                    _manifest_progress_clear("main")
                    print(f"Parallel worker failed for row {entry.idx}; retried in main process: {exc}")
                else:
                    _manifest_progress_clear(worker_label)
                if pending_entries:
                    next_entry = pending_entries.pop(0)
                    gpu_entry = _entry_with_gpu_db_path(next_entry, gpu_id)
                    _manifest_progress_start(worker_label, _describe_entry(gpu_entry))
                    next_future = executors[gpu_slots.index(gpu_id)].submit(_execute_entry_worker, _worker_entry_payload(gpu_entry))
                    future_map[next_future] = (gpu_entry, gpu_id)
        _drain_progress_queue(progress_queue)
    finally:
        for executor in executors:
            executor.shutdown(wait=True)
        manager.shutdown()
    return results


def _execute_entry_row_local(entry: ManifestEntry) -> dict[str, Any]:
    from .miopen_worker import should_isolate
    if should_isolate(entry.resolved):
        return _execute_isolated_miopen([entry], grouped=False)[0]
    resolved = entry.resolved
    service_id = resolved["service_id"]
    queue_services([resolved])
    try:
        from .manifest.preparation import applies, execute_entries
        if applies(resolved):
            outcome = execute_entries([entry])[0]
            result = _result_row(resolved, entry.idx, outcome["status"], outcome["error_message"], service_id=service_id)
            _record_manifest_progress_result(result)
            return result
        summary = execute_service(resolved)
        status = "success" if summary.status == "success" else "failed"
        error = summary.error or ""
        if status != "success":
            print(f"Service failed for row {entry.idx}: {error}")
        result = _result_row(resolved, entry.idx, status, error, service_id=summary.service_id)
        _record_manifest_progress_result(result)
        return result
    except Exception as exc:  # noqa: BLE001
        _write_failure_log(
            FAILURE_LOG_PATH,
            row_index=entry.idx,
            service_id=service_id,
            case_name=resolved.get("case_name"),
            manifest_group_id=resolved.get("manifest_group_id"),
            failure_stage="runtime_exception",
            error_message=str(exc),
            resolved=resolved,
            exc=exc,
        )
        _write_failure_db(
            resolved,
            row_index=entry.idx,
            service_id=service_id,
            case_name=resolved.get("case_name"),
            manifest_group_id=resolved.get("manifest_group_id"),
            failure_stage="runtime_exception",
            error_message=str(exc),
            exc=exc,
        )
        print(f"Service failed for row {entry.idx}: {exc}")
        result = _result_row(resolved, entry.idx, "failed", str(exc), service_id=service_id)
        _record_manifest_progress_result(result)
        return result
    finally:
        if _is_hf_resolved(resolved):
            _clear_cuda_runtime_state()


def _execute_fedavg_entries(entries: list[ManifestEntry]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for _, run_entries in _group_entries(entries, lambda resolved: str(resolved.get("fl_run_id") or "")):
        first = run_entries[0].resolved
        debug_print(
            "\nfedavg run: "
            f"fl_run_id={first.get('fl_run_id')} "
            f"model={first.get('hf_model_id') or first.get('model_type')} "
            f"clients={first.get('fedavg_clients_per_run')} "
            f"rounds={first.get('fedavg_rounds')} client_rounds={len(run_entries)}"
        )
        _manifest_progress_start("main", f"fedavg run {first.get('fl_run_id')}")
        run_results, summary = execute_fedavg_run(
            run_entries,
            execute_entry=_fedavg_execute_entry,
            build_dataset_args=_build_dataset_args,
            load_dataset=service_runner._load_dataset,
            build_model=_build_fedavg_hf_model if _is_hf_resolved(first) else _build_fedavg_generic_model,
        )
        results.extend(run_results)
        summaries.append(summary)
        _manifest_progress_clear("main")
    return results, summaries


def _build_fedavg_generic_model(config: dict[str, Any], meta: Any, task_family: str):
    return service_runner.ServiceRunner(config)._build_model(meta=dict(meta or {}), task_family=task_family)


def _build_fedavg_hf_model(config: dict[str, Any], meta: Any, task_family: str):
    del task_family
    return _build_prepared_hf_model(config, meta)


def _fedavg_execute_entry(
    entry: ManifestEntry,
    *,
    prepared: dict[str, Any] | None,
    failure_message: str | None = None,
    status_override: str | None = None,
) -> dict[str, Any]:
    resolved = dict(entry.resolved)
    service_id = resolved["service_id"]
    if failure_message:
        status = status_override or "failed"
        try:
            service_runner.persist_status_only_service(
                resolved, status=status, reason=failure_message
            )
        except Exception as exc:  # noqa: BLE001
            failure_message = f"{failure_message}; status-only persistence failed: {exc}"
        if status == "failed":
            _write_failure_log(
                FAILURE_LOG_PATH,
                row_index=entry.idx,
                service_id=service_id,
                case_name=resolved.get("case_name"),
                manifest_group_id=resolved.get("manifest_group_id"),
                failure_stage="fedavg_run",
                error_message=failure_message,
                resolved=resolved,
            )
            _write_failure_db(
                resolved,
                row_index=entry.idx,
                service_id=service_id,
                case_name=resolved.get("case_name"),
                manifest_group_id=resolved.get("manifest_group_id"),
                failure_stage="fedavg_run",
                error_message=failure_message,
            )
        result = _result_row(resolved, entry.idx, status, failure_message, service_id=service_id)
        result["_terminal_defer_progress"] = not debug_enabled(resolved)
        _record_manifest_progress_result(result)
        return result
    if prepared:
        resolved.update(prepared)
    _manifest_progress_start(_WORKER_LABEL,
        f"round {resolved.get('round_id')}/{resolved.get('fedavg_rounds')} {resolved.get('client_id')}")
    resolved.setdefault("_client_started_perf", time.perf_counter())
    try:
        summary = execute_service(resolved)
        status = "success" if summary.status == "success" else "failed"
        error = summary.error or ""
        if status != "success":
            print(f"fedavg client failed for row {entry.idx}: {error}")
        result = _result_row(resolved, entry.idx, status, error, service_id=summary.service_id)
        result["_terminal_metrics"] = summary.metrics or {}
        result["_terminal_weights_saved"] = bool(getattr(summary, "weights_saved", False))
        # Training can reduce the effective batch size after an OOM retry.
        result["_terminal_runtime_config"] = {
            key: metric.get("value") for key, metric in (summary.metrics or {}).items()
            if key in {"batch_size", "learning_rate"} and isinstance(metric, dict)
        }
        if status == "success" and prepared and prepared.get("_prepared_model") is not None:
            result["_final_tensor_state"] = getattr(summary, "final_tensor_state", None)
            if result["_final_tensor_state"] is None:
                result["_final_tensor_state"] = service_runner.capture_model_tensor_state(prepared["_prepared_model"])
            result["_checkpoint_integrity_verified"] = True
        result["_terminal_defer_progress"] = not debug_enabled(resolved)
        _record_manifest_progress_result(result)
        return result
    except Exception as exc:  # noqa: BLE001
        _write_failure_log(
            FAILURE_LOG_PATH,
            row_index=entry.idx,
            service_id=service_id,
            case_name=resolved.get("case_name"),
            manifest_group_id=resolved.get("manifest_group_id"),
            failure_stage="fedavg_runtime_exception",
            error_message=str(exc),
            resolved=resolved,
            exc=exc,
        )
        _write_failure_db(
            resolved,
            row_index=entry.idx,
            service_id=service_id,
            case_name=resolved.get("case_name"),
            manifest_group_id=resolved.get("manifest_group_id"),
            failure_stage="fedavg_runtime_exception",
            error_message=str(exc),
            exc=exc,
        )
        print(f"fedavg client failed for row {entry.idx}: {exc}")
        result = _result_row(resolved, entry.idx, "failed", str(exc), service_id=service_id)
        result["_terminal_defer_progress"] = not debug_enabled(resolved)
        _record_manifest_progress_result(result)
        return result
    finally:
        if _is_hf_resolved(resolved):
            _clear_cuda_runtime_state()


def _execute_entries_grouped_hf(entries: list[ManifestEntry], *, workers: int = 1) -> list[dict[str, Any]]:
    hf_entries = [entry for entry in entries if _is_groupable_hf_entry(entry.resolved)]
    local_entries = [entry for entry in entries if not _is_groupable_hf_entry(entry.resolved)]
    results: list[dict[str, Any]] = []
    if local_entries:
        results.extend(_execute_entries_row_local(local_entries, workers=workers))
    model_groups = []
    for _, model_entries in _group_entries(hf_entries, _hf_model_group_key):
        model_entries = sorted(model_entries, key=_entry_group_sort_key)
        model_groups.append(model_entries)
    results.extend(_execute_hf_model_groups(model_groups, workers=workers))
    return results


def _execute_hf_model_groups(model_groups: list[list[ManifestEntry]], *, workers: int = 1) -> list[dict[str, Any]]:
    if not model_groups:
        return []

    worker_count = _resolve_parallel_worker_count(workers)
    gpu_slots = _detect_cuda_device_ids()[:worker_count]
    if worker_count <= 1 or len(gpu_slots) <= 1 or len(model_groups) <= 1:
        results: list[dict[str, Any]] = []
        for model_entries in model_groups:
            first = model_entries[0].resolved
            print(
                "\nGrouped HF model: "
                f"model={first.get('hf_model_id')} task={first.get('hf_task')} rows={len(model_entries)}"
            )
            _manifest_progress_start("main", _describe_group(model_entries))
            results.extend(_execute_hf_model_group(model_entries))
            _manifest_progress_clear("main")
        return results

    print(
        "Parallel grouped HF execution enabled: "
        f"workers={len(gpu_slots)} visible_gpus={','.join(str(gpu) for gpu in gpu_slots)} "
        f"groups={len(model_groups)}"
    )
    return _execute_hf_groups_with_gpu_affinity(model_groups, gpu_slots)


def _worker_group_payload(entries: list[ManifestEntry]) -> list[dict[str, Any]]:
    return [_worker_entry_payload(entry) for entry in entries]


def _execute_hf_group_worker(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = [
        ManifestEntry(
            idx=int(item["idx"]),
            ordinal=int(item.get("ordinal", 0)),
            resolved=dict(item.get("resolved") or {}),
            validation=RowValidation(True),
        )
        for item in payload
    ]
    return _execute_hf_model_group(entries)


def _execute_hf_groups_with_gpu_affinity(model_groups: list[list[ManifestEntry]], gpu_slots: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    mp_context = multiprocessing.get_context("spawn")
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    executors = [
        concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp_context,
            initializer=_worker_initializer,
            initargs=(gpu_id, progress_queue),
        )
        for gpu_id in gpu_slots
    ]
    executor_map = {gpu_id: executors[index] for index, gpu_id in enumerate(gpu_slots)}
    pending_groups = list(model_groups)
    future_map: dict[concurrent.futures.Future, tuple[list[ManifestEntry], int]] = {}
    try:
        for gpu_id in gpu_slots:
            if not pending_groups:
                break
            model_entries = pending_groups.pop(0)
            gpu_entries = [_entry_with_gpu_db_path(entry, gpu_id) for entry in model_entries]
            first = gpu_entries[0].resolved
            print(
                "\nGrouped HF model: "
                f"model={first.get('hf_model_id')} task={first.get('hf_task')} "
                f"rows={len(gpu_entries)} assigned_gpu={gpu_id} db={first.get('db_path')}"
            )
            _manifest_progress_start(_worker_label_for_gpu(gpu_id), _describe_group(gpu_entries))
            future = executor_map[gpu_id].submit(_execute_hf_group_worker, _worker_group_payload(gpu_entries))
            future_map[future] = (gpu_entries, gpu_id)
        while future_map:
            done, _ = concurrent.futures.wait(
                list(future_map),
                timeout=0.2,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            _drain_progress_queue(progress_queue)
            if not done:
                continue
            for future in done:
                model_entries, gpu_id = future_map.pop(future)
                worker_label = _worker_label_for_gpu(gpu_id)
                try:
                    results.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    _manifest_progress_clear(worker_label)
                    first = model_entries[0].resolved
                    if is_cuda_poison_error(exc):
                        print(
                            "Parallel grouped HF worker hit a CUDA poison error; recording group failures: "
                            f"model={first.get('hf_model_id')} task={first.get('hf_task')} error={exc}"
                        )
                        _clear_cuda_runtime_state()
                        for entry in model_entries:
                            results.append(
                                _record_entry_failure(
                                    entry,
                                    failure_stage="grouped_worker_cuda_poison",
                                    error_message=exc,
                                    exc=exc,
                                )
                            )
                    else:
                        _manifest_progress_start("main", _describe_group(model_entries))
                        print(
                            "Parallel grouped HF worker failed; retrying in main process: "
                            f"model={first.get('hf_model_id')} task={first.get('hf_task')} error={exc}"
                        )
                        results.extend(_execute_hf_model_group(model_entries))
                        _manifest_progress_clear("main")
                else:
                    _manifest_progress_clear(worker_label)
                if pending_groups:
                    next_group = pending_groups.pop(0)
                    gpu_entries = [_entry_with_gpu_db_path(entry, gpu_id) for entry in next_group]
                    first = gpu_entries[0].resolved
                    print(
                        "\nGrouped HF model: "
                        f"model={first.get('hf_model_id')} task={first.get('hf_task')} "
                        f"rows={len(gpu_entries)} assigned_gpu={gpu_id} db={first.get('db_path')}"
                    )
                    _manifest_progress_start(worker_label, _describe_group(gpu_entries))
                    next_future = executor_map[gpu_id].submit(_execute_hf_group_worker, _worker_group_payload(gpu_entries))
                    future_map[next_future] = (gpu_entries, gpu_id)
        _drain_progress_queue(progress_queue)
    finally:
        for executor in executors:
            executor.shutdown(wait=True)
        manager.shutdown()
    return results


def _execute_isolated_miopen(entries, *, grouped):
    from .miopen_worker import run_isolated
    print("[MIOpen] isolated vision worker | MIOPEN_FIND_MODE=FAST | "
          f"model={entries[0].resolved.get('hf_model_id')} | services={len(entries)}", flush=True)
    try:
        results = run_isolated(_worker_group_payload(entries), grouped=grouped)
    except Exception as exc:
        results = []
        for entry in entries:
            # A worker may exit after committing some clients. Preserve them.
            cfg = entry.resolved
            path = Path(cfg.get("db_path") or CONFIG["db_path"])
            completed = False
            if path.is_file():
                with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
                    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='services'").fetchone():
                        row = conn.execute("SELECT status FROM services WHERE service_id=?", (cfg["service_id"],)).fetchone()
                        completed = row is not None and row[0] == "completed"
            if completed:
                result = _result_row(cfg, entry.idx, "success", "", service_id=cfg["service_id"])
                _record_manifest_progress_result(result)
            else:
                result = _record_entry_failure(entry, failure_stage="miopen_worker",
                                               error_message=str(exc), exc=exc)
            results.append(result)
        return results
    for result in results:
        _record_manifest_progress_result(result)
    return results


@terminal_scope(lambda entries: entries[0].resolved if entries else {})
def _execute_hf_model_group(entries: list[ManifestEntry]) -> list[dict[str, Any]]:
    from .miopen_worker import should_isolate
    if entries and should_isolate(entries[0].resolved):
        return _execute_isolated_miopen(entries, grouped=True)
    queue_services([entry.resolved for entry in entries])
    from .manifest.preparation import applies, execute_entries, group_key
    if entries and applies(entries[0].resolved):
        results = []
        for _, group in _group_entries(entries, group_key):
            reported = set()
            def on_start(entry):
                _manifest_progress_start(_WORKER_LABEL, _describe_entry(entry))
            def on_result(entry, outcome):
                result = _result_row(entry.resolved, entry.idx, outcome["status"], outcome["error_message"], service_id=outcome["service_id"])
                reported.add(entry.idx)
                results.append(result)
                _record_manifest_progress_result(result)
            try:
                execute_entries(group, on_start=on_start, on_result=on_result)
            except Exception as exc:
                first = group[0].resolved
                print(f"Family preparation failed: model={first.get('hf_model_id')} "
                      f"dataset={first.get('dataset_name')} affected_clients={len(group)} | {exc}")
                remaining = [entry for entry in group if entry.idx not in reported]
                results.extend(_record_entry_failure(entry, failure_stage="family_preparation",
                    error_message=str(exc), exc=exc, write_log=index == 0) for index, entry in enumerate(remaining))
        return results
    results: list[dict[str, Any]] = []
    model_cache: dict[str, tuple[Any, Any]] = {}
    for _, dataset_entries in _group_entries(entries, _hf_dataset_group_key):
        prepared_dataset = None
        dataset_entries = sorted(dataset_entries, key=_entry_group_sort_key)
        first = dataset_entries[0].resolved
        dataset_args = _dataset_args_for_group(dataset_entries)
        print(
            "Grouped HF dataset: "
            f"dataset={first.get('dataset_name')} config={first.get('dataset_config')} "
            f"train_split={dataset_args.get('train_split')} test_split={dataset_args.get('test_split')} "
            f"rows={len(dataset_entries)}"
        )
        try:
            dataset_load_start = time.perf_counter()
            from ..data.tokenization_selection import training_row_selection
            with training_row_selection([entry.resolved for entry in dataset_entries]) as selection:
                prepared_dataset = service_runner._load_dataset(first.get("dataset", "hf"), **dataset_args)
            dataset_load_s = float(time.perf_counter() - dataset_load_start)
            debug_print(
                "[ServiceTiming] "
                f"model={first.get('hf_model_id') or first.get('model_type')} "
                f"| stage=dataset load | elapsed_s={dataset_load_s:.3f} "
                f"| dataset={first.get('dataset_name')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Grouped HF dataset load failed; falling back to row-local execution: {exc}")
            results.extend(_execute_entries_row_local(dataset_entries))
            continue

        meta = prepared_dataset[2]
        model_key = _hf_prepared_model_key(first, meta)
        if model_key not in model_cache:
            try:
                model = _build_prepared_hf_model(first, meta)
                base_weights = service_runner.snapshot_model_weights(model)
            except Exception as exc:  # noqa: BLE001
                print(f"Grouped HF model build failed; falling back to row-local execution: {exc}")
                prepared_dataset = None
                results.extend(_execute_entries_row_local(dataset_entries))
                continue
            if base_weights is None and any(_is_trainable_entry(entry.resolved) for entry in dataset_entries):
                print("Grouped HF model did not expose resettable weights; falling back to row-local execution.")
                prepared_dataset = None
                results.extend(_execute_entries_row_local(dataset_entries))
                continue
            model_cache[model_key] = (model, base_weights)

        model, base_weights = model_cache[model_key]
        for entry_pos, entry in enumerate(dataset_entries):
            model_reset_s = None
            if base_weights is not None:
                reset_start = time.perf_counter()
                try:
                    service_runner.reset_model_to_weights(model, base_weights)
                except Exception as exc:  # noqa: BLE001
                    model_cache.pop(model_key, None)
                    if is_cuda_poison_error(exc):
                        _clear_cuda_runtime_state()
                    result = _record_entry_failure(
                        entry,
                        failure_stage=(
                            "model_reset_cuda_poison"
                            if is_cuda_poison_error(exc)
                            else "model_reset_failed"
                        ),
                        error_message=exc,
                        exc=exc,
                    )
                    results.append(result)
                    _clear_cuda_runtime_state()
                    try:
                        model = _build_prepared_hf_model(first, meta)
                        base_weights = service_runner.snapshot_model_weights(model)
                        model_cache[model_key] = (model, base_weights)
                    except Exception as rebuild_exc:  # noqa: BLE001
                        print(
                            "Grouped HF model rebuild failed after reset error; "
                            f"remaining rows in dataset group will fall back row-local: {rebuild_exc}"
                        )
                        remaining = dataset_entries[entry_pos + 1 :]
                        results.extend(_execute_entries_row_local(remaining))
                        model_cache.clear()
                        _clear_cuda_runtime_state()
                        return results
                    continue
                model_reset_s = float(time.perf_counter() - reset_start)
                debug_print(
                    "[ServiceTiming] "
                    f"service_id={entry.resolved.get('service_id')} "
                    f"| model={entry.resolved.get('hf_model_id') or entry.resolved.get('model_type')} "
                    f"| stage=model reset | elapsed_s={model_reset_s:.3f}",
                    flush=True,
                )
            _manifest_progress_start(_WORKER_LABEL, _describe_entry(entry))
            prepared_config = dict(entry.resolved)
            prepared_config["_prepared_dataset"] = prepared_dataset
            selected_split = selection.materialize(entry.resolved["service_id"], prepared_dataset[0])
            if selected_split is not None:
                prepared_config["_pretokenization_split_info"] = selected_split
            prepared_config["_prepared_model"] = model
            prepared_config["_dataset_load_s"] = dataset_load_s
            if model_reset_s is not None:
                prepared_config["_model_reset_s"] = model_reset_s
            try:
                summary = service_runner.ServiceRunner(prepared_config).run()
            except Exception as exc:  # noqa: BLE001
                model_cache.pop(model_key, None)
                if is_cuda_poison_error(exc):
                    _clear_cuda_runtime_state()
                result = _record_entry_failure(
                    entry,
                    failure_stage=(
                        "service_execution_cuda_poison"
                        if is_cuda_poison_error(exc)
                        else "service_execution"
                    ),
                    error_message=exc,
                    exc=exc,
                )
                results.append(result)
                _clear_cuda_runtime_state()
                if is_cuda_poison_error(exc):
                    try:
                        model = _build_prepared_hf_model(first, meta)
                        base_weights = service_runner.snapshot_model_weights(model)
                        model_cache[model_key] = (model, base_weights)
                    except Exception:
                        pass
                continue
            finally:
                prepared_config.pop("_prepared_dataset", None)
                prepared_config.pop("_pretokenization_split_info", None)
                selected_split = None
            status = "success" if summary.status == "success" else "failed"
            error = summary.error or ""
            if status != "success":
                print(f"Service failed for row {entry.idx}: {error}")
                if is_cuda_poison_error(error):
                    model_cache.pop(model_key, None)
                    _clear_cuda_runtime_state()
            result = _result_row(entry.resolved, entry.idx, status, error, service_id=summary.service_id)
            results.append(result)
            _record_manifest_progress_result(result)
            _clear_cuda_runtime_state()
        # Release the previous expanded pool before allocating the next dataset.
        prepared_dataset = None
    model_cache.clear()
    _clear_cuda_runtime_state()
    return results


def _build_prepared_hf_model(config: dict[str, Any], meta: Any):
    meta_dict = dict(meta or {})
    task_family = canonical_task_family(meta_dict.get("task_type") or config.get("task_type"), meta_dict.get("hf_task") or config.get("hf_task"))
    return service_runner.ServiceRunner(config)._build_model(meta=meta_dict, task_family=task_family)


def _is_hf_resolved(resolved: dict[str, Any]) -> bool:
    dataset = str(resolved.get("dataset") or "").strip().lower()
    model_type = str(resolved.get("model_type") or "").strip().lower()
    return dataset in {"hf", "huggingface"} and (
        model_type.startswith("hf") or model_type.startswith("transformers")
    )


def _requires_row_local_hf_entry(resolved: dict[str, Any]) -> bool:
    hf_task = str(resolved.get("hf_task") or "").strip().lower().replace("-", "_")
    model_id = str(resolved.get("hf_model_id") or "").strip().lower()
    if hf_task == "image_captioning":
        return True
    if hf_task == "visual_question_answering" and (
        model_id.startswith("salesforce/blip")
        or model_id.startswith("microsoft/git-")
        or "blip" in model_id
        or "git" in model_id
    ):
        return True
    return False


def _is_groupable_hf_entry(resolved: dict[str, Any]) -> bool:
    if not _is_blank(resolved.get("fl_run_id")):
        return False
    return _is_hf_resolved(resolved) and bool(resolved.get("hf_model_id")) and not _requires_row_local_hf_entry(resolved)


def _is_trainable_entry(resolved: dict[str, Any]) -> bool:
    return str(resolved.get("training_regime") or "").strip().lower() not in {"inference_only", "inference"}


def _group_entries(entries: list[ManifestEntry], key_fn):
    groups: dict[str, list[ManifestEntry]] = {}
    order: list[str] = []
    for entry in entries:
        key = key_fn(entry.resolved)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(entry)
    for key in order:
        yield key, groups[key]


def _hf_model_group_key(resolved: dict[str, Any]) -> str:
    from .manifest.preparation import applies, group_key
    if applies(resolved):
        return group_key(resolved)
    dataset_args = dict(resolved.get("dataset_args") or {})
    return _json_key(
        {
            "hf_model_id": resolved.get("hf_model_id"),
            "hf_model_revision": resolved.get("hf_model_revision"),
            "hf_task": resolved.get("hf_task"),
            "model_type": resolved.get("model_type"),
            "device": resolved.get("device"),
            "mixed_precision": resolved.get("mixed_precision"),
            "precision_type": resolved.get("precision_type"),
            "no_save_weights": resolved.get("no_save_weights", False),
            "loader_template": dataset_args.get("loader_template"),
            "task_tag": resolved.get("task_tag"),
        }
    )


def _hf_dataset_group_key(resolved: dict[str, Any]) -> str:
    args = dict(resolved.get("dataset_args") or {})
    args.pop("max_samples", None)
    args["inference_only"] = not _is_trainable_entry(resolved)
    return _json_key({
        "dataset": resolved.get("dataset"),
        "hf_dataset_revision": resolved.get("hf_dataset_revision"),
        "dataset_args": args,
    })


def _dataset_args_for_group(entries: list[ManifestEntry]) -> dict[str, Any]:
    args = dict(entries[0].resolved.get("dataset_args") or {})
    args.pop("max_samples", None)  # Accept old prepared rows without applying their cap.
    args["inference_only"] = not _is_trainable_entry(entries[0].resolved)
    return args


def _hf_prepared_model_key(config: dict[str, Any], meta: Any) -> str:
    meta_dict = dict(meta or {})
    return _json_key(
        {
            "hf_model_id": config.get("hf_model_id") or meta_dict.get("hf_model_id"),
            "hf_model_revision": config.get("hf_model_revision") or meta_dict.get("hf_model_revision"),
            "hf_task": config.get("hf_task") or meta_dict.get("hf_task"),
            "model_type": config.get("model_type"),
            "device": config.get("device"),
            "mixed_precision": config.get("mixed_precision"),
            "precision_type": config.get("precision_type"),
            "num_labels": infer_num_labels(meta_dict, fallback=meta_dict.get("num_classes")),
            "label_format": infer_label_format(meta_dict, task_type=meta_dict.get("task_type") or config.get("task_type")),
        }
    )


def _normalize_precision_type(value: Any) -> str:
    text = str(value or "fp16").strip().lower()
    return text if text in {"fp16", "bf16", "fp32"} else "fp16"


def _entry_group_sort_key(entry: ManifestEntry) -> tuple[Any, ...]:
    resolved = entry.resolved
    return (
        str(resolved.get("dataset_name") or ""),
        str(resolved.get("dataset_config") or ""),
        int(resolved.get("split_variant") or 0),
        int(resolved.get("knob_variant") or 0),
        entry.idx,
    )


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _sort_result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda row: (row.get("row_index") is None, row.get("row_index") if row.get("row_index") is not None else -1))


def _result_row(resolved: dict[str, Any], idx: int, status: str, error_message: str, *, service_id: str) -> dict[str, Any]:
    public_resolved = {
        key: value for key, value in resolved.items() if not str(key).startswith("_")
    }
    return {
        "service_id": service_id,
        "row_index": int(idx),
        "manifest_group_id": resolved.get("manifest_group_id"),
        "case_name": resolved.get("case_name"),
        "status": status,
        "error_message": error_message,
        "resolved_config_json": diagnostic_json(public_resolved),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute manifest service rows")
    parser.add_argument("--file", required=True, help="Path to manifest file (.csv or .xlsx)")
    parser.add_argument("--sheet", default="services", help="Sheet name for service rows (xlsx only)")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Resolve and validate rows without executing services")
    parser.add_argument("--db", "--db-path", dest="db_path", default=None, help="SQLite database path for service records")
    parser.add_argument("--log-mode", choices=("concise", "debug"), default=None, help="Terminal output detail (default: concise)")
    parser.add_argument("--workers", type=int, default=1, help="Number of row-level worker processes to run")
    parser.add_argument("--rerun-existing", action="store_true", help="Execute rows even when their service_id already exists in SQLite")
    parser.add_argument("--no-save-weights", action="store_true", help="Independent debugging: train/evaluate without saving weight checkpoints")
    parser.add_argument("--smoke-test", action="store_true", help="Skip perturbation and update signatures; use separate smoke-test service IDs")
    parser.add_argument("--no-grouped-hf", dest="grouped_hf", action="store_false", help="Disable grouped Hugging Face execution")
    parser.set_defaults(grouped_hf=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_manifest(
        file=args.file,
        sheet=args.sheet,
        dry_run=args.dry_run,
        db_path=args.db_path,
        grouped_hf=args.grouped_hf,
        workers=args.workers,
        rerun_existing=args.rerun_existing,
        log_mode=args.log_mode,
        no_save_weights=args.no_save_weights,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
