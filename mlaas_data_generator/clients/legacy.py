"""Explicit variant expansion and historical service-local sampling."""
from __future__ import annotations

import hashlib
import math
import numpy as np
from typing import Mapping
from ..data.skew_axes import axis_supports_strategy, bucket_distribution
from typing import Any
from .legacy_profiles import ALIASES as LEGACY_PROFILE_ALIASES
from .resources import (
    _as_float, _estimated_model_params_m, _unique_ints,
    _batch_sizes_for, _learning_rates_for, _epochs_for,
    _weight_decays_for, _optimizers_for, _warmup_ratios_for,
    _mlm_probabilities_for, _gradient_accumulation_steps_for,
)


def _sample_seed_for_variant(
    *,
    base_seed: int,
    model_id: Any,
    dataset_name: Any,
    dataset_config: Any,
    dataset_variant: int,
    split_variant: int,
    knob_variant: int,
) -> int:
    raw = "|".join(
        str(value)
        for value in (
            base_seed,
            model_id,
            dataset_name,
            dataset_config,
            dataset_variant,
            split_variant,
            knob_variant,
        )
    )
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8], 16)



def _default_skew_axis_for_task(task_key: str) -> str | None:
    mapping = {
        "text_classification": "class_label",
        "token_classification": "entity_present_sentence",
        "sentence_similarity": "score_bin",
        "fill_mask": "raw_text_length_bucket",
        "text_generation": "supervised_token_bucket",
        "text2text_generation": "supervised_token_bucket",
        "image_classification": "class_label",
        "image_captioning": "supervised_token_bucket",
        "text_image_retrieval": "query_length_bucket",
        "visual_question_answering": "answer_vocab",
    }
    return mapping.get(task_key)



def _skew_axis_config_for_task(task_key: str) -> dict[str, Any] | None:
    if task_key == "sentence_similarity":
        return {"num_bins": 5}
    if task_key in {"text_generation", "text2text_generation"}:
        return {"num_bins": 5}
    return None



def _split_knobs_for_variant(task_key: str, knob_variant: int) -> tuple[str, float | None, str | None, dict[str, Any] | None]:
    skew_axis = _default_skew_axis_for_task(task_key)
    skew_axis_config = _skew_axis_config_for_task(task_key)
    if skew_axis is None or task_key in {"object_detection", "image_segmentation"}:
        return "iid", None, skew_axis, skew_axis_config
    options: tuple[tuple[str, float | None], ...] = (("iid", None), ("dirichlet", 0.2))
    split_strategy, distribution_param = options[int(knob_variant) % len(options)]
    return split_strategy, distribution_param, skew_axis, skew_axis_config



def _training_knobs_for_variant(
    *,
    task_key: str,
    model: dict[str, Any],
    training_regime: str,
    knob_variant: int,
) -> dict[str, Any]:
    batch_sizes = _batch_sizes_for(task_key, model)
    split_strategy, distribution_param, skew_axis, skew_axis_config = _split_knobs_for_variant(task_key, knob_variant)
    mixed_precision, precision_type = _precision_knobs_for_variant(knob_variant)
    if training_regime == "inference_only":
        return {
            "training_epochs": 0,
            "batch_size": batch_sizes[int(knob_variant) % len(batch_sizes)],
            "learning_rate": None,
            "optimizer": "none",
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "mlm_probability": _mlm_probabilities_for(task_key)[int(knob_variant) % len(_mlm_probabilities_for(task_key))] if _mlm_probabilities_for(task_key) else None,
            "gradient_accumulation_steps": 1,
            "momentum": 0.0,
            "split_strategy": split_strategy,
            "skew_axis": skew_axis,
            "skew_axis_config": skew_axis_config,
            "distribution_type": split_strategy,
            "distribution_param": distribution_param,
            "mixed_precision": mixed_precision,
            "precision_type": precision_type,
        }

    learning_rates = _learning_rates_for(task_key, model)
    epochs = _epochs_for(task_key)
    weight_decays = _weight_decays_for(task_key)
    optimizers = _optimizers_for(task_key)
    warmup_ratios = _warmup_ratios_for(task_key)
    mlm_probabilities = _mlm_probabilities_for(task_key)
    grad_accum_steps = _gradient_accumulation_steps_for(task_key, model)
    batch_idx = int(knob_variant) % len(batch_sizes)
    lr_idx = int(knob_variant) % len(learning_rates)
    optimizer_idx = int(knob_variant) % len(optimizers)
    epoch_idx = int(knob_variant) % len(epochs)
    wd_idx = int(knob_variant) % len(weight_decays)
    warmup_idx = int(knob_variant) % len(warmup_ratios)
    mlm_idx = int(knob_variant) % len(mlm_probabilities) if mlm_probabilities else 0
    grad_accum_idx = int(knob_variant) % len(grad_accum_steps)
    return {
        "training_epochs": epochs[epoch_idx],
        "batch_size": batch_sizes[batch_idx],
        "learning_rate": learning_rates[lr_idx],
        "optimizer": optimizers[optimizer_idx],
        "weight_decay": weight_decays[wd_idx],
        "warmup_ratio": warmup_ratios[warmup_idx],
        "mlm_probability": mlm_probabilities[mlm_idx] if mlm_probabilities else None,
        "gradient_accumulation_steps": grad_accum_steps[grad_accum_idx],
        "momentum": 0.0,
        "split_strategy": split_strategy,
        "skew_axis": skew_axis,
        "skew_axis_config": skew_axis_config,
        "distribution_type": split_strategy,
        "distribution_param": distribution_param,
        "mixed_precision": mixed_precision,
        "precision_type": precision_type,
    }



def _precision_knobs_for_variant(knob_variant: int) -> tuple[bool, str]:
    enabled = int(knob_variant) % 4 != 3
    if not enabled:
        return False, "fp16"
    # ROCm runs in this project have been materially more stable with fp16 than bf16.
    return True, "fp16"



def _generic_split_knobs_for_variant(
    case: dict[str, Any],
    split_variant: int,
) -> tuple[str, float | None, str | None, dict[str, Any] | None]:
    task_key = str(case.get("task_key") or "")
    task_type = str(case.get("task_type") or "")
    if task_type == "regression":
        skew_axis = "score_bin"
        skew_axis_config = {"num_bins": 5 + (int(split_variant) % 4)}
    elif task_type in {"classification", "clustering"}:
        skew_axis = "class_label"
        skew_axis_config = None
    else:
        skew_axis = _default_skew_axis_for_task(task_key)
        skew_axis_config = _skew_axis_config_for_task(task_key)

    options: tuple[tuple[str, float | None], ...] = (
        ("iid", None),
        ("dirichlet", 0.05),
        ("dirichlet", 0.1),
        ("dirichlet", 0.2),
        ("dirichlet", 0.5),
        ("dirichlet", 1.0),
    )
    split_strategy, distribution_param = options[int(split_variant) % len(options)]
    return split_strategy, distribution_param, skew_axis, skew_axis_config



def _generic_training_knobs_for_variant(
    case: dict[str, Any],
    knob_variant: int,
) -> dict[str, Any]:
    task_key = str(case.get("task_key") or "")
    model_type = str(case.get("model_type") or "").lower()
    variant = int(knob_variant)

    if model_type in {"randomforest", "kmeans"}:
        learning_rates = (None,)
        optimizers = ("none",)
        weight_decays = (0.0,)
        momentum_values = (0.0,)
    elif task_key == "keras_image_classification":
        learning_rates = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
        optimizers = ("adam", "sgd", "rmsprop", "adamw")
        weight_decays = (0.0, 1e-5, 1e-4, 1e-3)
        momentum_values = (0.0, 0.5, 0.9)
    elif task_key == "tabular_regression":
        learning_rates = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
        optimizers = ("adam", "sgd", "rmsprop", "adamw", "adagrad")
        weight_decays = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
        momentum_values = (0.0, 0.5, 0.9)
    else:
        learning_rates = (case.get("learning_rate"),)
        optimizers = (case.get("optimizer") or "adam",)
        weight_decays = (0.0,)
        momentum_values = (0.0,)

    base_batch_sizes = _batch_sizes_for(task_key, case)
    case_batch = case.get("batch_size")
    if case_batch and task_key != "text_image_retrieval":
        candidates = tuple(sorted({1, 2, 4, 8, 16, 32, int(case_batch)}))
        batch_sizes = _unique_ints(
            tuple(value for value in candidates if value <= max(int(case_batch), max(base_batch_sizes)))
        )
    else:
        batch_sizes = base_batch_sizes

    if model_type == "kmeans":
        epochs = (1,)
    else:
        epochs = (1, 2, 3)

    return {
        "training_epochs": epochs[variant % len(epochs)],
        "batch_size": batch_sizes[variant % len(batch_sizes)],
        "learning_rate": learning_rates[(variant * 2 + variant // 3) % len(learning_rates)],
        "optimizer": optimizers[(variant * 3 + variant // 2) % len(optimizers)],
        "weight_decay": weight_decays[(variant * 5 + variant // 4) % len(weight_decays)],
        "momentum": momentum_values[(variant * 7 + variant // 5) % len(momentum_values)],
        "clustering_k": (2, 3, 4, 5, 6)[variant % 5] if model_type == "kmeans" else case.get("clustering_k"),
        "clustering_n_init": (5, 10, 20)[variant % 3] if model_type == "kmeans" else None,
        "clustering_max_iter": (100, 200, 300, 500)[variant % 4] if model_type == "kmeans" else None,
        "clustering_tol": (1e-3, 1e-4, 1e-5)[variant % 3] if model_type == "kmeans" else None,
    }



# Versionless service-local sampling, retained for saved manifests.

def _canonical_split_strategy(value: Any) -> str:
    strategy = str(value or "iid").strip().lower().replace("-", "_")
    aliases = {
        "niid": "dirichlet",
        "non_iid": "dirichlet",
        "non_iid_dirichlet": "dirichlet",
        "shards": "shard",
        **LEGACY_PROFILE_ALIASES,
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
    elif strategy == "dirichlet":
        effective_strategy = strategy
        info["effective_strategy"] = effective_strategy
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
    info["_indices"] = idx
    return _take_rows(x, idx), _take_rows(y, idx), info



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
    from ..data.image_store import DiskArrayStore
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: _take_rows(child, idx) for key, child in value.items()}
    if isinstance(value, DiskArrayStore):
        return value.take(idx)
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

