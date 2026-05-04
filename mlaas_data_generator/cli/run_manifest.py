from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import multiprocessing
import os
import queue
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..compatibility import known_bad_row_reason
from ..config import CONFIG, FAILED_MANIFEST_PATH, FAILURE_LOG_PATH, MANIFEST_RESULTS_PATH
from ..hf_auth import load_hf_token_from_file
from ..registry.datasets import DATASET_REGISTRY
from ..models.label_schema import infer_label_format, infer_num_labels
from ..services.runner import execute_service, resolve_service_id
from ..services import runner as service_runner
from ..services.taxonomy import canonical_task_family
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
    "enabled",
    "measure_system_metrics",
    "mixed_precision",
    "explainability_enabled",
    "enable_perturbation_metrics",
    "perturbation_stage_logging",
    "perturbation_progress_logging",
    "save_weights",
    "update_signature_enabled",
    "dynamic_padding",
    "report_decode_errors",
}
INT_COLUMNS = {
    "seed",
    "sample_seed",
    "training_epochs",
    "batch_size",
    "sample_size",
    "max_samples",
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
}
FLOAT_COLUMNS = {
    "learning_rate",
    "weight_decay",
    "momentum",
    "warmup_ratio",
    "mlm_probability",
    "distribution_param",
    "clustering_tol",
    "realism_score",
    "perturbation_random_strength",
}
ENUM_COLUMNS = {"training_regime", "resource_tier", "optimizer", "device", "model_type", "hf_task", "task_type", "modality", "split_strategy", "distribution_type", "skew_axis", "precision_type"}
JSON_COLUMNS = {"column_mapping", "service_config", "custom_distributions", "skew_axis_config"}

DATASET_ARG_COLUMNS = {
    "dataset_name",
    "dataset_config",
    "hf_model_id",
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
    "vqa_label_mode",
    "vqa_answer_vocab_size",
    "vqa_unseen_answer_policy",
    "retrieval_positive_policy",
    "max_samples",
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
    _running: dict[str, str] | None = None
    _initialized: bool = False
    _lines_rendered: int = 0

    def __post_init__(self) -> None:
        if self._running is None:
            self._running = {}

    def start(self, worker_label: str, description: str) -> None:
        self._running[worker_label] = description
        self.render()

    def clear(self, worker_label: str) -> None:
        if worker_label in self._running:
            self._running.pop(worker_label, None)
            self.render()

    def record(self, status: str) -> None:
        self.completed += 1
        if str(status).strip().lower() == "success":
            self.succeeded += 1
        else:
            self.failed += 1
        self.render()

    def render(self) -> None:
        if not sys.stdout.isatty():
            queued = max(self.total - self.completed - len(self._running), 0)
            print(
                "Manifest progress: "
                f"{self.completed}/{self.total} complete | "
                f"running={len(self._running)} queued={queued} "
                f"success={self.succeeded} failed={self.failed}"
            )
            return

        lines = self._build_lines()
        if not self._initialized:
            sys.stdout.write("\n" * len(lines))
            self._initialized = True
            self._lines_rendered = len(lines)
        sys.stdout.write(f"\x1b[{self._lines_rendered}A")
        for line in lines:
            sys.stdout.write("\x1b[2K")
            sys.stdout.write(line + "\n")
        extra = self._lines_rendered - len(lines)
        for _ in range(max(0, extra)):
            sys.stdout.write("\x1b[2K\n")
        self._lines_rendered = len(lines)
        sys.stdout.flush()

    def finish(self) -> None:
        self.render()
        if sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _build_lines(self) -> list[str]:
        running_count = len(self._running)
        queued = max(self.total - self.completed - running_count, 0)
        width = shutil.get_terminal_size((120, 20)).columns
        percent = 100.0 if self.total <= 0 else (self.completed / self.total) * 100.0
        header = (
            "Manifest progress "
            f"[{self.completed}/{self.total} {percent:5.1f}%] "
            f"running={running_count} queued={queued} "
            f"success={self.succeeded} failed={self.failed}"
        )
        running_items = " | ".join(f"{label}:{desc}" for label, desc in sorted(self._running.items())) or "idle"
        running_line = f"Workers: {running_items}"
        return [header[:width], running_line[:width]]


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
    return f"row {entry.ordinal}/{total} {resolved.get('service_id')}"


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
    tracker = _manifest_progress()
    if tracker is not None:
        tracker.start(worker_label, description)


def _manifest_progress_clear(worker_label: str) -> None:
    tracker = _manifest_progress()
    if tracker is not None:
        tracker.clear(worker_label)


def _record_manifest_progress_result(result: dict[str, Any]) -> None:
    if _WORKER_PROGRESS_QUEUE is not None:
        try:
            _WORKER_PROGRESS_QUEUE.put(
                {
                    "event": "result",
                    "worker_label": _WORKER_LABEL,
                    "status": result.get("status"),
                }
            )
        except Exception:
            pass
        return

    tracker = _manifest_progress()
    if tracker is not None:
        tracker.record(str(result.get("status") or "failed"))


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
        if event.get("event") == "result":
            tracker.record(str(event.get("status") or "failed"))


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


def _coerce_by_column(column: str, value: Any) -> Any:
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
        defaults, rows = _extract_defaults_row(df)
        return rows, defaults
    if suffix in {".xlsx", ".xls"}:
        workbook = pd.read_excel(file_path, sheet_name=None)
        service_rows_df = _normalize_manifest_columns(workbook.get(sheet) if sheet in workbook else next(iter(workbook.values())))
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


def _resolve_row(row: pd.Series, manifest_defaults: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(CONFIG)
    resolved.update(BASE_DEFAULTS)
    resolved.update(manifest_defaults)
    for column, raw in row.items():
        value = _coerce_by_column(column, raw)
        if value is not None:
            resolved[column] = value

    if resolved.get("benchmark_split") and not resolved.get("test_split"):
        resolved["test_split"] = resolved["benchmark_split"]
    if resolved.get("test_split") and not resolved.get("benchmark_split"):
        resolved["benchmark_split"] = resolved["test_split"]
    resolved["precision_type"] = _normalize_precision_type(resolved.get("precision_type"))
    if str(_normalize_requested_device(resolved.get("device"))).strip().lower() == "cpu":
        resolved["mixed_precision"] = False
        resolved["precision_type"] = "fp16"
    if _is_blank(resolved.get("service_id")):
        resolved["service_id"] = resolve_service_id(resolved)
    resolved["dataset_args"] = _build_dataset_args(resolved)
    return resolved


def _build_dataset_args(resolved: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for key in DATASET_ARG_COLUMNS:
        value = resolved.get(key)
        if value is not None:
            args[key] = value
    if args.get("benchmark_split") and not args.get("test_split"):
        args["test_split"] = args["benchmark_split"]
    return args


def _validate_row(resolved: dict[str, Any]) -> RowValidation:
    federated_columns = [
        key
        for key, value in resolved.items()
        if (key in FEDERATED_COLUMNS or str(key).startswith("global_")) and not _is_blank(value)
    ]
    if federated_columns:
        return RowValidation(
            False,
            "Federated columns are not accepted in service manifests: "
            + ", ".join(sorted(federated_columns)),
        )

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
    if precision_type not in {"fp16", "bf16"}:
        return RowValidation(False, "precision_type must be one of fp16 or bf16")
    task_family = canonical_task_family(resolved.get("task_type"), resolved.get("hf_task"))
    vision_error = _validate_vision_row_requirements(resolved, task_family=task_family, training_regime=training_regime)
    if vision_error:
        return RowValidation(False, vision_error)
    blocked_reason = known_bad_row_reason(resolved)
    if blocked_reason:
        return RowValidation(False, blocked_reason)

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


def _manifest_requires_hf_preflight(enabled_df: pd.DataFrame) -> bool:
    for _, row in enabled_df.iterrows():
        dataset = str(_normalize_value(row.get("dataset")) or "").strip().lower()
        model_type = str(_normalize_value(row.get("model_type")) or "").strip().lower()
        hf_task = str(_normalize_value(row.get("hf_task")) or "").strip().lower()
        if dataset == "hf" or model_type.startswith("hf") or hf_task:
            return True
    return False


def _ensure_manifest_preflight(enabled_df: pd.DataFrame) -> None:
    if not _manifest_requires_hf_preflight(enabled_df):
        return
    required_imports = {
        "datasets": "Hugging Face dataset loading requires the 'datasets' package. Install it with: pip install datasets",
        "transformers": "Manifest preflight failed: required HF dependency 'transformers' is not importable. Install it with: pip install transformers",
    }
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
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


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
        f"error_message: {error_message}",
        "resolved_config:",
        json.dumps(resolved, indent=2, default=str),
    ]
    if exc is not None:
        lines.extend(["traceback:", _format_traceback(exc) or ""])
    lines.append("")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_failure_db(resolved: dict[str, Any], *, row_index, service_id, case_name, manifest_group_id, failure_stage, error_message, exc=None) -> None:
    db_path = resolved.get("db_path") or CONFIG.get("db_path")
    try:
        writer = make_writer("sqlite", db_path=db_path)
        writer.start()
        writer.write_service_failure(
            service_id=str(service_id) if service_id is not None else None,
            row_index=int(row_index) if row_index is not None else None,
            case_name=str(case_name) if case_name is not None else None,
            manifest_group_id=str(manifest_group_id) if manifest_group_id is not None else None,
            failure_stage=str(failure_stage),
            error_message=str(error_message) if error_message is not None else None,
            resolved_config_json=json.dumps(resolved, default=str),
            traceback_text=_format_traceback(exc),
        )
        writer.finish()
    except Exception as db_exc:  # noqa: BLE001
        print(f"Warning: failed to persist service failure to SQLite: {db_exc}")


def _manifest_csv_value(value: Any) -> Any:
    value = _normalize_value(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return value


def _failed_result_row_indexes(results: list[dict[str, Any]]) -> list[int]:
    failed: list[int] = []
    seen: set[int] = set()
    for result in _sort_result_rows(results):
        if str(result.get("status") or "").strip().lower() == "success":
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


def run_manifest(
    file: str,
    sheet: str = "services",
    dry_run: bool = False,
    db_path: str | None = None,
    *,
    grouped_hf: bool = True,
    workers: int = 1,
) -> Path:
    global _ACTIVE_MANIFEST_PROGRESS
    manifest_path = Path(file)
    load_hf_token_from_file()
    print(f"Loading manifest file: {manifest_path}")
    rows_df, manifest_defaults = load_manifest(manifest_path, sheet=sheet)

    enabled_df = rows_df[rows_df.apply(_is_enabled, axis=1)].copy()
    print(f"Loaded manifest: {manifest_path}")
    print(f"Total rows: {len(rows_df)}")
    print(f"Enabled services: {len(enabled_df)}")

    manifest_group_id = str(uuid.uuid4())
    results: list[dict[str, Any]] = []
    _ACTIVE_MANIFEST_PROGRESS = ManifestProgressTracker(total=len(enabled_df))

    try:
        if not dry_run:
            try:
                _ensure_manifest_preflight(enabled_df)
            except ManifestPreflightError as exc:
                resolved = {
                    "db_path": db_path or CONFIG.get("db_path"),
                    "manifest_group_id": manifest_group_id,
                    "manifest_path": str(manifest_path),
                    "sheet": sheet,
                }
                results.append(
                    {
                        "service_id": "",
                        "row_index": None,
                        "manifest_group_id": manifest_group_id,
                        "case_name": "__manifest_preflight__",
                        "status": "failed",
                        "error_message": str(exc),
                        "resolved_config_json": json.dumps(resolved, default=str),
                    }
                )
                _write_failure_log(
                    FAILURE_LOG_PATH,
                    row_index=None,
                    service_id=None,
                    case_name="__manifest_preflight__",
                    manifest_group_id=manifest_group_id,
                    failure_stage="manifest_preflight",
                    error_message=str(exc),
                    resolved=resolved,
                    exc=exc,
                )
                _write_failure_db(
                    resolved,
                    row_index=None,
                    service_id=None,
                    case_name="__manifest_preflight__",
                    manifest_group_id=manifest_group_id,
                    failure_stage="manifest_preflight",
                    error_message=str(exc),
                    exc=exc,
                )
                output_path = MANIFEST_RESULTS_PATH
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(results).to_csv(output_path, index=False)
                _write_failed_manifest_csv(
                    rows_df=rows_df,
                    manifest_defaults=manifest_defaults,
                    results=results,
                    db_path=db_path,
                )
                print(f"Manifest preflight failed: {exc}")
                print(f"Wrote results: {output_path}")
                return output_path

        entries: list[ManifestEntry] = []
        for i, (idx, row) in enumerate(enabled_df.iterrows(), start=1):
            resolved = _resolve_row(row, manifest_defaults)
            resolved["row_index"] = int(idx)
            resolved["_manifest_total"] = len(enabled_df)
            if db_path is not None:
                resolved["db_path"] = db_path
            if _is_blank(resolved.get("manifest_group_id")):
                resolved["manifest_group_id"] = manifest_group_id
            entries.append(ManifestEntry(idx=int(idx), ordinal=i, resolved=resolved, validation=_validate_row(resolved)))

        valid_entries: list[ManifestEntry] = []
        for entry in entries:
            resolved = entry.resolved
            service_id = resolved["service_id"]
            print(f"\nService {entry.ordinal}/{len(enabled_df)}: {service_id}")
            print(
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

            if dry_run:
                _manifest_progress_start("main", f"dry-run row {entry.ordinal}/{len(enabled_df)}")
                print(json.dumps(resolved, indent=2, default=str))
                result = _result_row(resolved, entry.idx, "success", "", service_id=service_id)
                results.append(result)
                _record_manifest_progress_result(result)
                _manifest_progress_clear("main")
                continue

            valid_entries.append(entry)

        if not dry_run:
            results.extend(
                _execute_entries_grouped_hf(valid_entries, workers=workers)
                if grouped_hf
                else _execute_entries_row_local(valid_entries, workers=workers)
            )

        output_path = MANIFEST_RESULTS_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_results = _sort_result_rows(results)
        pd.DataFrame(sorted_results).to_csv(output_path, index=False)
        _write_failed_manifest_csv(
            rows_df=rows_df,
            manifest_defaults=manifest_defaults,
            results=sorted_results,
            db_path=db_path,
        )
        print(f"Wrote results: {output_path}")
        return output_path
    finally:
        if _ACTIVE_MANIFEST_PROGRESS is not None:
            _ACTIVE_MANIFEST_PROGRESS.finish()
        _ACTIVE_MANIFEST_PROGRESS = None


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
    resolved = entry.resolved
    service_id = resolved["service_id"]
    try:
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
                    _manifest_progress_start("main", _describe_group(model_entries))
                    first = model_entries[0].resolved
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


def _execute_hf_model_group(entries: list[ManifestEntry]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    model_cache: dict[str, tuple[Any, Any]] = {}
    for _, dataset_entries in _group_entries(entries, _hf_dataset_group_key):
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
            prepared_dataset = service_runner._load_dataset(first.get("dataset", "hf"), **dataset_args)
            dataset_load_s = float(time.perf_counter() - dataset_load_start)
            print(
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

        _, _, meta = prepared_dataset
        model_key = _hf_prepared_model_key(first, meta)
        if model_key not in model_cache:
            try:
                model = _build_prepared_hf_model(first, meta)
                base_weights = service_runner.snapshot_model_weights(model)
            except Exception as exc:  # noqa: BLE001
                print(f"Grouped HF model build failed; falling back to row-local execution: {exc}")
                results.extend(_execute_entries_row_local(dataset_entries))
                continue
            if base_weights is None and any(_is_trainable_entry(entry.resolved) for entry in dataset_entries):
                print("Grouped HF model did not expose resettable weights; falling back to row-local execution.")
                results.extend(_execute_entries_row_local(dataset_entries))
                continue
            model_cache[model_key] = (model, base_weights)

        model, base_weights = model_cache[model_key]
        for entry in dataset_entries:
            model_reset_s = None
            if base_weights is not None:
                reset_start = time.perf_counter()
                service_runner.reset_model_to_weights(model, base_weights)
                model_reset_s = float(time.perf_counter() - reset_start)
                print(
                    "[ServiceTiming] "
                    f"service_id={entry.resolved.get('service_id')} "
                    f"| model={entry.resolved.get('hf_model_id') or entry.resolved.get('model_type')} "
                    f"| stage=model reset | elapsed_s={model_reset_s:.3f}",
                    flush=True,
                )
            prepared_config = dict(entry.resolved)
            prepared_config["_prepared_dataset"] = prepared_dataset
            prepared_config["_prepared_model"] = model
            prepared_config["_dataset_load_s"] = dataset_load_s
            if model_reset_s is not None:
                prepared_config["_model_reset_s"] = model_reset_s
            summary = service_runner.ServiceRunner(prepared_config).run()
            status = "success" if summary.status == "success" else "failed"
            error = summary.error or ""
            if status != "success":
                print(f"Service failed for row {entry.idx}: {error}")
            result = _result_row(entry.resolved, entry.idx, status, error, service_id=summary.service_id)
            results.append(result)
            _record_manifest_progress_result(result)
    return results


def _build_prepared_hf_model(config: dict[str, Any], meta: Any):
    meta_dict = dict(meta or {})
    task_family = canonical_task_family(meta_dict.get("task_type") or config.get("task_type"), meta_dict.get("hf_task") or config.get("hf_task"))
    return service_runner.ServiceRunner(config)._build_model(meta=meta_dict, task_family=task_family)


def _is_groupable_hf_entry(resolved: dict[str, Any]) -> bool:
    dataset = str(resolved.get("dataset") or "").strip().lower()
    model_type = str(resolved.get("model_type") or "").strip().lower()
    return dataset in {"hf", "huggingface"} and bool(resolved.get("hf_model_id")) and (
        model_type.startswith("hf") or model_type.startswith("transformers")
    )


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
    dataset_args = dict(resolved.get("dataset_args") or {})
    return _json_key(
        {
            "hf_model_id": resolved.get("hf_model_id"),
            "hf_task": resolved.get("hf_task"),
            "model_type": resolved.get("model_type"),
            "device": resolved.get("device"),
            "mixed_precision": resolved.get("mixed_precision"),
            "precision_type": resolved.get("precision_type"),
            "loader_template": dataset_args.get("loader_template"),
            "task_tag": resolved.get("task_tag"),
        }
    )


def _hf_dataset_group_key(resolved: dict[str, Any]) -> str:
    args = dict(resolved.get("dataset_args") or {})
    args.pop("max_samples", None)
    args["inference_only"] = not _is_trainable_entry(resolved)
    return _json_key({"dataset": resolved.get("dataset"), "dataset_args": args})


def _dataset_args_for_group(entries: list[ManifestEntry]) -> dict[str, Any]:
    args = dict(entries[0].resolved.get("dataset_args") or {})
    max_samples = [
        int(entry.resolved["max_samples"])
        for entry in entries
        if entry.resolved.get("max_samples") is not None
    ]
    if max_samples:
        args["max_samples"] = max(max_samples)
    args["inference_only"] = not _is_trainable_entry(entries[0].resolved)
    return args


def _hf_prepared_model_key(config: dict[str, Any], meta: Any) -> str:
    meta_dict = dict(meta or {})
    return _json_key(
        {
            "hf_model_id": config.get("hf_model_id") or meta_dict.get("hf_model_id"),
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
    return "bf16" if text == "bf16" else "fp16"


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
    return {
        "service_id": service_id,
        "row_index": int(idx),
        "manifest_group_id": resolved.get("manifest_group_id"),
        "case_name": resolved.get("case_name"),
        "status": status,
        "error_message": error_message,
        "resolved_config_json": json.dumps(resolved, default=str),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute manifest service rows")
    parser.add_argument("--file", required=True, help="Path to manifest file (.csv or .xlsx)")
    parser.add_argument("--sheet", default="services", help="Sheet name for service rows (xlsx only)")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Resolve and validate rows without executing services")
    parser.add_argument("--db", "--db-path", dest="db_path", default=None, help="SQLite database path for service records")
    parser.add_argument("--workers", type=int, default=1, help="Number of row-level worker processes to run")
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
    )


if __name__ == "__main__":
    main()
