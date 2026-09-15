"""FedAvg generation eligibility for the active service manifest pipeline."""

from __future__ import annotations

from typing import Any, Mapping

FEDAVG_HF_TASK_KEYS = frozenset(
    {
        "text_classification",
        "token_classification",
        "image_classification",
        "sentence_similarity",
    }
)

FEDAVG_GENERIC_TASK_KEYS = frozenset(
    {
        "keras_image_classification",
        "tabular_regression",
    }
)

FEDAVG_ALLOWED_MODEL_TYPES = frozenset(
    {
        "hf_finetune",
        "cnn",
        "mlp",
        "logreg",
    }
)

FEDAVG_EXCLUDED_MODEL_TYPES = frozenset(
    {
        "randomforest",
        "kmeans",
        "hf",
    }
)


def fedavg_benchmark_sample_seed(fl_run_id: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"{fl_run_id}:benchmark".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def fedavg_run_partition_seed(fl_run_id: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"{fl_run_id}:partition".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def is_fedavg_eligible_row(row: Mapping[str, Any]) -> bool:
    training_regime = str(row.get("training_regime") or "").strip().lower()
    if training_regime in {"inference_only", "inference"}:
        return False

    model_type = str(row.get("model_type") or "").strip().lower()
    if model_type in FEDAVG_EXCLUDED_MODEL_TYPES:
        return False
    if model_type not in FEDAVG_ALLOWED_MODEL_TYPES:
        return False

    task_key = str(row.get("task_key") or row.get("task") or "").strip()
    hf_task = str(row.get("hf_task") or "").strip().lower().replace("-", "_")

    if str(row.get("dataset") or "").strip().lower() == "hf":
        if task_key and task_key not in FEDAVG_HF_TASK_KEYS:
            return False
        if hf_task and hf_task not in {
            "sequence_classification",
            "text_classification",
            "token_classification",
            "image_classification",
            "sentence_similarity",
        }:
            return False
        return True

    if model_type in {"cnn", "mlp", "logreg"} and training_regime == "generic":
        if model_type == "logreg":
            task_type = str(row.get("task_type") or "").strip().lower()
            return task_type in {"classification", "binary_classification", "multiclass_classification"}
        return task_key in FEDAVG_GENERIC_TASK_KEYS or model_type in {"cnn", "mlp"}
    return False


def fedavg_compatibility_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("dataset_name") or ""),
        str(row.get("dataset_config") or ""),
        str(row.get("hf_dataset_revision") or ""),
        str(row.get("train_split") or ""),
        str(row.get("test_split") or ""),
        str(row.get("model_type") or ""),
        str(row.get("hf_model_id") or ""),
        str(row.get("hf_model_revision") or ""),
        str(row.get("hf_task") or ""),
        str(row.get("task_type") or ""),
        str(row.get("training_regime") or ""),
        str(row.get("max_length") or ""),
        str(row.get("input_schema") or ""),
    )
