from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mlaas_data_generator.clients import recipes, training
from mlaas_data_generator.provenance import sha256_json, canonical_json
from mlaas_data_generator.family_preflight import hf_dataset_error, manifest_family_error
from mlaas_data_generator.evaluation_sampling import evaluation_settings, task_benchmark_sample_size
from mlaas_data_generator.evaluation_protocol import PLAN_CONFIG_KEYS, manifest_evaluation_defaults, resolve_evaluation_plan

from mlaas_data_generator.clients.resources import (
    _as_float, _estimated_model_params_m, _unique_ints,
    _batch_sizes_for, _learning_rates_for, _epochs_for,
    _weight_decays_for, _optimizers_for, _warmup_ratios_for,
    _mlm_probabilities_for, _gradient_accumulation_steps_for,
)
from mlaas_data_generator.clients.legacy import (
    _sample_seed_for_variant, _default_skew_axis_for_task,
    _skew_axis_config_for_task, _split_knobs_for_variant, _training_knobs_for_variant,
    _precision_knobs_for_variant, _generic_split_knobs_for_variant, _generic_training_knobs_for_variant,
)

from mlaas_data_generator.compatibility import known_bad_manifest_combo_reason, known_bad_path_reason
from mlaas_data_generator.config import CNN_HIDDEN_LAYER_POOL, DEFAULT_MANIFEST_PATH
from mlaas_data_generator.fedavg_eligibility import is_fedavg_eligible_row
from mlaas_data_generator.fedavg_manifest import (
    expand_fedavg_run,
    fedavg_client_knobs,
    stratified_heterogeneity_intensities,
)
from mlaas_data_generator.registry import DATASET_REGISTRY, MODEL_REGISTRY


@dataclass(frozen=True)
class TaskSpec:
    pipeline_tag: str
    hf_task: str
    modality: str
    task_type: str
    task_label: str
    task_tag: str | None = None


@dataclass(frozen=True)
class ManifestProfile:
    name: str
    default_avg_sample_size: int


TASK_SPECS: dict[str, TaskSpec] = {
    "text_classification": TaskSpec("text-classification", "sequence_classification", "text", "classification", "textcls"),
    "token_classification": TaskSpec("token-classification", "token_classification", "text", "classification", "tokencls"),
    "sentence_similarity": TaskSpec("sentence-similarity", "sentence_similarity", "text", "regression", "pairscore"),
    "fill_mask": TaskSpec("fill-mask", "fill_mask", "text", "classification", "fillmask"),
    "text_generation": TaskSpec("text-generation", "causal_lm_generation", "text", "generation", "textgen", "language-modeling"),
    "text2text_generation": TaskSpec("text2text-generation", "seq2seq_generation", "text", "generation", "text2text", "summarization"),
    "image_classification": TaskSpec("image-classification", "image_classification", "image", "classification", "imgcls"),
    "object_detection": TaskSpec("object-detection", "image_detection", "image", "detection", "objdet"),
    "image_segmentation": TaskSpec("image-segmentation", "image_segmentation", "image", "segmentation", "imgseg"),
    "image_captioning": TaskSpec("image-to-text", "image_captioning", "multimodal", "generation", "imgcap", "captioning"),
    "text_image_retrieval": TaskSpec("zero-shot-image-classification", "text_image_retrieval", "multimodal", "retrieval", "imgtxtret", "retrieval"),
    "visual_question_answering": TaskSpec("visual-question-answering", "visual_question_answering", "multimodal", "vqa", "vqa", "vqa"),
}


MANIFEST_PROFILES: dict[str, ManifestProfile] = {
    "test": ManifestProfile("test", default_avg_sample_size=128),
    "balanced": ManifestProfile("balanced", default_avg_sample_size=768),
    "benchmark": ManifestProfile("benchmark", default_avg_sample_size=1600),
}


MANIFEST_COLUMNS = [
    *PLAN_CONFIG_KEYS,
    "checkpoint_patch_operation_policy", "checkpoint_measure_full", "checkpoint_weight_format",
    "checkpoint_reduced_compression", "checkpoint_reduced_compression_level", "checkpoint_validation_samples", "checkpoint_quality_tolerance",
    "recipe_sampler_version", "training_policy", "training_policy_hash",
    "label_format", "label_mode", "num_labels", "num_classes", "task_resolution_source", "precision_policy",
    "mobilebert_head_normalization", "max_grad_norm",
    "calibration_band", "calibration_template_id",
    "effective_batch_size", "client_partition_index", "max_client_quantity",
    "unconstrained_sample_size", "quantity_capacity_policy",
    "service_id",
    "enabled",
    "case_name",
    "notes",
    "dataset",
    "dataset_name",
    "dataset_config",
    "hf_dataset_revision",
    "train_split",
    "test_split",
    "benchmark_split",
    "model_type",
    "hidden_layers",
    "hf_task",
    "hf_model_id",
    "hf_model_revision",
    "task_type",
    "task",
    "task_tag",
    "modality",
    "input_schema",
    "output_schema",
    "training_regime",
    "dataset_variant",
    "split_variant",
    "knob_variant",
    "service_config",
    "split_strategy",
    "skew_axis",
    "skew_axis_config",
    "distribution_type",
    "distribution_param",
    "custom_distributions",
    "training_epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "weight_decay",
    "momentum",
    "clustering_k",
    "clustering_n_init",
    "clustering_max_iter",
    "clustering_tol",
    "warmup_ratio",
    "mlm_probability",
    "gradient_accumulation_steps",
    "seed",
    "sample_seed",
    "initialization_seed",
    "evaluation_seed",
    "sample_size",
    "max_length",
    "timeout_s",
    "max_train_time_s",
    "max_eval_time_s",
    "device",
    "mixed_precision",
    "precision_type",
    "save_weights",
    "checkpoint_patch_compression",
    "checkpoint_patch_compression_level",
    "num_workers",
    "text_column",
    "image_column",
    "label_column",
    "mask_column",
    "question_column",
    "answer_column",
    "ranking_label_column",
    "vqa_label_mode",
    "vqa_answer_vocab_size",
    "vqa_unseen_answer_policy",
    "retrieval_positive_policy",
    "missing_pair_handling",
    "on_decode_error",
    "report_decode_errors",
    "source_max_length",
    "target_max_length",
    "dynamic_padding",
    "column_mapping",
    "explainability_enabled",
    "enable_perturbation_metrics",
    "perturbation_stage_logging",
    "perturbation_progress_logging",
    "perturbation_sample_count",
    "perturbation_candidate_units",
    "perturbation_target_units",
    "perturbation_trust_trials",
    "perturbation_progress_sample_interval",
    "perturbation_random_strength",
    "explainability_method",
    "explainability_target",
    "service_source",
    "model_role",
    "fit_decision",
    "fit_reason",
    "fit_quality_score",
    "realism_score",
    "domain_alignment",
    "dataset_hint",
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
    "fl_run_id",
    "client_id",
    "round_id",
    "lineage_id",
    "generation_mode",
    "run_partition_seed",
    "benchmark_sample_seed",
    "benchmark_sample_size",
    "run_train_pool_size",
    "fedavg_source_pool_cap",
    "requested_total_samples",
    "training_seed",
    "scheduled_to_participate",
    "participation_schedule_seed",
    "participation_schedule_version",
    "fedavg_run_index",
    "fedavg_clients_per_run",
    "fedavg_rounds",
    "fedavg_client_participation_rate",
    "heterogeneity_intensity",
    "shard_support_quantile",
    "regression_support_policy",
    "min_target_bin_fraction",
    "partition_generator_version",
    "requested_partition_params",
    "requested_quota",
    "original_fl_run_id",
]


GENERIC_MANIFEST_CASES: tuple[dict[str, Any], ...] = (
    {
        "task_key": "keras_image_classification",
        "task_label": "keras_imgcls",
        "dataset": "cifar10",
        "dataset_name": "cifar10",
        "task_type": "classification",
        "task": "classification",
        "modality": "image",
        "model_type": "cnn",
        "input_schema": "single_image",
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "adam",
    },
    {
        "task_key": "sklearn_image_classification",
        "task_label": "sk_imgcls",
        "dataset": "digits",
        "dataset_name": "digits",
        "task_type": "classification",
        "task": "classification",
        "modality": "image",
        "model_type": "randomforest",
        "input_schema": "single_image_flattened",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "none",
    },
    {
        "task_key": "tabular_regression",
        "task_label": "tabreg",
        "dataset": "synthetic",
        "dataset_name": "synthetic",
        "task_type": "regression",
        "task": "regression",
        "modality": "tabular",
        "model_type": "mlp",
        "input_schema": "tabular_features",
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "adam",
    },
    {
        "task_key": "tabular_regression",
        "task_label": "tabreg_rf",
        "dataset": "synthetic",
        "dataset_name": "synthetic",
        "task_type": "regression",
        "task": "regression",
        "modality": "tabular",
        "model_type": "randomforest",
        "input_schema": "tabular_features",
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "none",
    },
    {
        "task_key": "tabular_regression",
        "task_label": "tabreg",
        "dataset": "uci_wine_quality",
        "dataset_name": "uci_wine_quality",
        "task_type": "regression",
        "task": "regression",
        "modality": "tabular",
        "model_type": "randomforest",
        "input_schema": "tabular_features",
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "none",
    },
    {
        "task_key": "clustering",
        "task_label": "cluster",
        "dataset": "synthetic",
        "dataset_name": "synthetic",
        "task_type": "clustering",
        "task": "clustering",
        "modality": "tabular",
        "model_type": "kmeans",
        "input_schema": "tabular_features",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "none",
        "clustering_k": 3,
    },
)
GENERIC_MANIFEST_TASK_KEYS = frozenset(case["task_key"] for case in GENERIC_MANIFEST_CASES)


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _resolve_manifest_profile(profile_name: str | None) -> ManifestProfile:
    return MANIFEST_PROFILES.get(str(profile_name or "balanced").strip().lower(), MANIFEST_PROFILES["balanced"])


def _normalise_positive_int(value: int | None, *, minimum: int = 1) -> int:
    if value is None:
        return minimum
    return max(minimum, int(value))


def _resolve_manifest_seed(seed: int | None) -> int:
    if seed is None:
        # The global seed also reaches NumPy/sklearn, which require 32 bits.
        return secrets.randbits(32)
    return int(seed)


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _service_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    safe_prefix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in prefix.lower()).strip("_") or "svc"
    return f"{safe_prefix}_{digest}"


def _training_regime_defaults(training_regime: str) -> dict[str, str]:
    if training_regime == "inference_only":
        return {"model_type": "hf", "model_role": "service"}
    return {"model_type": "hf_finetune", "model_role": "task_head"}


def _service_config(row: dict[str, Any]) -> str:
    payload = {
        **{key: row[key] for key in (
            "hidden_layers",
            "checkpoint_patch_operation_policy", "checkpoint_measure_full", "checkpoint_weight_format",
            "checkpoint_reduced_compression", "checkpoint_reduced_compression_level",
            "checkpoint_validation_samples", "checkpoint_quality_tolerance",
            "checkpoint_patch_compression", "checkpoint_patch_compression_level",
        ) if key in row},
        "label_mode": row.get("label_mode"),
        "label_format": row.get("label_format"),
        "task_type": row.get("task_type"),
        "num_labels": row.get("num_labels"),
        "precision_policy": row.get("precision_policy"),
        "mobilebert_head_normalization": row.get("mobilebert_head_normalization"),
        "max_grad_norm": row.get("max_grad_norm"),
        "training_epochs": row.get("training_epochs"),
        "split_strategy": row.get("split_strategy"),
        "skew_axis": row.get("skew_axis"),
        "skew_axis_config": row.get("skew_axis_config"),
        "distribution_type": row.get("distribution_type"),
        "distribution_param": row.get("distribution_param"),
        "batch_size": row.get("batch_size"),
        "learning_rate": row.get("learning_rate"),
        "optimizer": row.get("optimizer"),
        "weight_decay": row.get("weight_decay"),
        "warmup_ratio": row.get("warmup_ratio"),
        "mlm_probability": row.get("mlm_probability"),
        "gradient_accumulation_steps": row.get("gradient_accumulation_steps"),
        "momentum": row.get("momentum"),
        "clustering_k": row.get("clustering_k"),
        "clustering_n_init": row.get("clustering_n_init"),
        "clustering_max_iter": row.get("clustering_max_iter"),
        "clustering_tol": row.get("clustering_tol"),
        "sample_size": row.get("sample_size"),
        "sample_seed": row.get("sample_seed"),
        "initialization_seed": row.get("initialization_seed"),
        "evaluation_seed": row.get("evaluation_seed"),
        "benchmark_sample_size": row.get("benchmark_sample_size"),
        "training_seed": row.get("training_seed"),
        "max_length": row.get("max_length"),
        "timeout_s": row.get("timeout_s"),
        "max_train_time_s": row.get("max_train_time_s"),
        "max_eval_time_s": row.get("max_eval_time_s"),
        "device": row.get("device"),
        "mixed_precision": row.get("mixed_precision"),
        "precision_type": row.get("precision_type"),
        "save_weights": row.get("save_weights"),
        "enable_perturbation_metrics": row.get("enable_perturbation_metrics"),
        "explainability_enabled": row.get("explainability_enabled"),
        "perturbation_sample_count": row.get("perturbation_sample_count"),
    }
    payload.update({key: row[key] for key in PLAN_CONFIG_KEYS if key in row})
    if str(row.get("generation_mode") or "independent").strip().lower() == "fedavg":
        payload.update(
            {
                "training_seed": row.get("training_seed"),
                "generation_mode": "fedavg",
                "fl_run_id": row.get("fl_run_id"),
                "client_id": row.get("client_id"),
                "round_id": row.get("round_id"),
                "scheduled_to_participate": row.get("scheduled_to_participate"),
                "fedavg_client_participation_rate": row.get("fedavg_client_participation_rate"),
                "participation_schedule_seed": row.get("participation_schedule_seed"),
                "participation_schedule_version": row.get("participation_schedule_version"),
                "heterogeneity_intensity": row.get("heterogeneity_intensity"),
                "partition_generator_version": row.get("partition_generator_version"),
                "requested_partition_params": row.get("requested_partition_params"),
                "requested_quota": row.get("requested_quota"),
                "fedavg_source_pool_cap": row.get("fedavg_source_pool_cap"),
                "requested_total_samples": row.get("requested_total_samples"),
                "original_fl_run_id": row.get("original_fl_run_id"),
            }
        )
    if row.get("recipe_sampler_version"):
        payload.update({key: row.get(key) for key in ("recipe_sampler_version", "training_policy", "training_policy_hash",
            "effective_batch_size", "partition_generator_version", "requested_quota", "max_client_quantity",
            "unconstrained_sample_size", "quantity_capacity_policy",
            "client_partition_index", "heterogeneity_intensity", "shard_support_quantile",
            "calibration_band", "calibration_template_id",
            "regression_support_policy", "min_target_bin_fraction")})
    return json.dumps({k: v for k, v in payload.items() if v is not None}, sort_keys=True)


def _split_value(dataset_spec: dict[str, Any], key: str, variant: int) -> Any:
    value = dataset_spec.get(key)
    if isinstance(value, (list, tuple)) and value:
        return value[variant % len(value)]
    return value


def _target_sample_size(
    *,
    profile: ManifestProfile,
    avg_sample_size: int | None,
) -> int:
    average = int(profile.default_avg_sample_size if avg_sample_size is None else avg_sample_size)
    recipes.sample_quantity_bounds(average)
    return average


def _max_length(dataset_spec: dict[str, Any], model: dict[str, Any], task_key: str) -> int | None:
    base = _as_int(dataset_spec.get("max_length")) or _as_int(model.get("max_length"))
    if base is None:
        return None
    caps = {"text_generation": 192, "text2text_generation": 192,
            "image_captioning": 64, "text_image_retrieval": 64, "visual_question_answering": 48}
    cap = caps.get(task_key, 128)
    return max(1, min(int(base), int(cap)))


def _task_specific_inference_model(task_key: str, model: dict[str, Any]) -> bool:
    model_id = str(model.get("hf_model_id") or "").lower()
    if task_key == "token_classification":
        return any(token in model_id for token in ("ner", "conll", "wnut"))
    if task_key == "sentence_similarity":
        return "sentence-transformers" in model_id or "sentence-similarity" in model_id
    if task_key == "text_classification":
        return any(token in model_id for token in ("sst", "sentiment", "mnli", "qqp", "qnli", "cola", "ag-news"))
    return False


def _quality_allows_combo(
    *,
    task_key: str,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    training_regime: str,
    strict_inference_dataset_match: bool,
) -> bool:
    if known_bad_manifest_combo_reason(task_key=task_key, model=model, dataset_spec=dataset_spec):
        return False
    allowed = set(model.get("allowed_training_regimes") or [])
    if allowed and training_regime not in allowed:
        return False
    if training_regime != "inference_only":
        if not _vision_dataset_meets_minimum_examples(task_key, dataset_spec):
            return False
        family = str(model.get("family") or "").strip().lower()
        if task_key == "image_captioning" and family == "git":
            return False
        if task_key == "visual_question_answering" and family in {"blip", "blip2", "git"}:
            return False
        if task_key in {"image_captioning", "text_image_retrieval", "visual_question_answering"}:
            return bool(model.get("finetune_validated"))
        return True

    dataset_key = str(dataset_spec.get("registry_key") or "")
    inference_keys = set(model.get("inference_dataset_keys") or [])
    if inference_keys:
        return dataset_key in inference_keys
    if strict_inference_dataset_match:
        return task_key in {"fill_mask", "text_generation", "text2text_generation", "image_captioning", "text_image_retrieval", "visual_question_answering"}
    if task_key in {"text_classification", "sentence_similarity", "token_classification"}:
        return _task_specific_inference_model(task_key, model)
    if task_key == "image_classification":
        expected = _as_int(model.get("inference_num_labels"))
        actual = _as_int(dataset_spec.get("num_classes"))
        return bool(expected is not None and actual is not None and expected == actual)
    return True


def _vision_dataset_meets_minimum_examples(task_key: str, dataset_spec: dict[str, Any]) -> bool:
    minimums = {
        "object_detection": (32, 16),
        "image_segmentation": (32, 16),
    }
    required = minimums.get(task_key)
    if required is None:
        return True
    train_examples = _as_int(dataset_spec.get("train_examples"))
    benchmark_examples = _as_int(dataset_spec.get("benchmark_examples"))
    if train_examples is None or benchmark_examples is None:
        return True
    min_train, min_benchmark = required
    return train_examples >= min_train and benchmark_examples >= min_benchmark


def _row_is_manifest_eligible(row: dict[str, Any], *, check_family: bool = True) -> bool:
    if check_family and manifest_family_error(row):
        return False
    training_regime = str(row.get("training_regime") or "").strip().lower()
    blocked_reason = known_bad_path_reason(
        task_key=row.get("task"),
        hf_task=row.get("hf_task"),
        task_tag=row.get("task_tag"),
        model_id=row.get("hf_model_id"),
        dataset_name=row.get("dataset_name"),
        dataset_config=row.get("dataset_config"),
    )
    if blocked_reason:
        return False
    if training_regime == "inference_only":
        return True
    model_id = str(row.get("hf_model_id") or "").strip().lower()
    task_type = str(row.get("task_type") or "").strip().lower()
    hf_task = str(row.get("hf_task") or row.get("task") or "").strip().lower().replace("-", "_")
    meta = row.get("hf_service_meta_json")
    family = ""
    if isinstance(meta, str) and meta.strip():
        try:
            family = str((json.loads(meta) or {}).get("model_family") or "").strip().lower()
        except Exception:
            family = ""
    if hf_task == "image_captioning" and (family == "git" or model_id.startswith("microsoft/git-")):
        return False
    if hf_task == "visual_question_answering" and (
        family in {"blip", "blip2", "git"}
        or model_id.startswith("salesforce/blip")
        or model_id.startswith("microsoft/git-")
    ):
        return False
    batch_size = _as_int(row.get("batch_size")) or 0
    if task_type == "segmentation" and model_id.startswith("openmmlab/upernet-") and batch_size < 2:
        return False
    return True


def _fit_quality_score(task_key: str, model: dict[str, Any], dataset_spec: dict[str, Any], training_regime: str) -> int:
    score = 60
    if model.get("finetune_validated"):
        score += 15
    if model.get("inference_dataset_keys") and dataset_spec.get("registry_key") in set(model.get("inference_dataset_keys") or []):
        score += 20
    if training_regime == "inference_only" and _task_specific_inference_model(task_key, model):
        score += 15
    if training_regime == "inference_only" and task_key in {"text_classification", "sentence_similarity", "token_classification", "image_classification"}:
        if not _task_specific_inference_model(task_key, model) and not model.get("inference_dataset_keys"):
            score -= 40
    params_m = _estimated_model_params_m(model)
    if params_m is not None and params_m <= 85:
        score += 5
    return max(0, score)


def _sort_models(models: list[dict[str, Any]], *, task_key: str,
                 selected_training_regimes: list[str], rng: random.Random) -> list[dict[str, Any]]:
    eligible = []
    for model in models:
        if any(_quality_allows_model_regime(model, regime) for regime in selected_training_regimes):
            eligible.append({**model, "_estimated_params_m": _estimated_model_params_m(model),
                             "_selection_jitter": rng.random()})
    # Uniform random priorities: size and quality metadata must not determine
    # which compatible architectures reach the population limit.
    return sorted(eligible, key=lambda item: item["_selection_jitter"])


def _quality_allows_model_regime(model: dict[str, Any], training_regime: str) -> bool:
    allowed = set(model.get("allowed_training_regimes") or [])
    return not allowed or training_regime in allowed


def _model_selection_score(task_key: str, model: dict[str, Any], selected_training_regimes: list[str]) -> int:
    score = 50
    params_m = _estimated_model_params_m(model)
    if params_m is not None and params_m <= 85:
        score += 10
    if model.get("finetune_validated"):
        score += 20
    if model.get("inference_dataset_keys"):
        score += 10
    if "inference_only" in selected_training_regimes and _task_specific_inference_model(task_key, model):
        score += 15
    return score


def _sort_datasets(datasets: list[dict[str, Any]], *, task_key: str,
                   rng: random.Random) -> list[dict[str, Any]]:
    eligible = [{**dataset, "_selection_jitter": rng.random()} for dataset in datasets]
    return sorted(eligible, key=lambda item: item["_selection_jitter"])


def _row_from_registry(
    *,
    task_key: str,
    task_spec: TaskSpec,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    budgets: dict[str, int],
    target_sample_size: int,
    training_regime: str,
    dataset_variant: int,
    split_variant: int,
    knob_variant: int,
    seed: int,
) -> dict[str, Any]:
    defaults = _training_regime_defaults(training_regime)
    knobs = _training_knobs_for_variant(
        task_key=task_key,
        model=model,
        training_regime=training_regime,
        knob_variant=knob_variant,
    )
    max_length = _max_length(dataset_spec, model, task_key)
    source_max_length = dataset_spec.get("source_max_length")
    target_max_length = dataset_spec.get("target_max_length")
    if task_key == "text2text_generation" and max_length is not None:
        source_max_length = source_max_length or max_length
        target_max_length = target_max_length or min(max_length, 96)
    fit_quality_score = _fit_quality_score(task_key, model, dataset_spec, training_regime)
    estimated_params_m = model.get("_estimated_params_m") or _estimated_model_params_m(model)
    estimated_params_count = None if estimated_params_m is None else int(float(estimated_params_m) * 1_000_000)
    sample_size = (
        None
        if training_regime == "inference_only"
        else target_sample_size
    )
    sample_seed = _sample_seed_for_variant(
        base_seed=seed,
        model_id=model.get("hf_model_id"),
        dataset_name=dataset_spec.get("dataset_name"),
        dataset_config=dataset_spec.get("dataset_config"),
        dataset_variant=dataset_variant,
        split_variant=split_variant,
        knob_variant=knob_variant,
    )
    row = {
        "enabled": True,
        "dataset": "hf",
        "dataset_name": dataset_spec.get("dataset_name"),
        "dataset_config": dataset_spec.get("dataset_config"),
        "train_split": _split_value(dataset_spec, "train_split", split_variant),
        "test_split": _split_value(dataset_spec, "test_split", split_variant),
        "benchmark_split": _split_value(dataset_spec, "test_split", split_variant),
        "model_type": defaults["model_type"],
        "hf_task": task_spec.hf_task,
        "hf_model_id": model.get("hf_model_id"),
        "task_type": task_spec.task_type,
        "task": task_key,
        "task_tag": task_spec.task_tag,
        "modality": task_spec.modality,
        "input_schema": dataset_spec.get("input_schema") or model.get("input_schema"),
        "output_schema": dataset_spec.get("label_format"),
        "training_regime": training_regime,
        "dataset_variant": dataset_variant,
        "split_variant": split_variant,
        "knob_variant": knob_variant,
        "seed": seed,
        "split_strategy": knobs["split_strategy"],
        "skew_axis": knobs["skew_axis"],
        "skew_axis_config": json.dumps(knobs["skew_axis_config"], sort_keys=True) if knobs.get("skew_axis_config") else None,
        "distribution_type": knobs["distribution_type"],
        "distribution_param": knobs["distribution_param"],
        "custom_distributions": None,
        "training_epochs": knobs["training_epochs"],
        "batch_size": knobs["batch_size"],
        "learning_rate": knobs["learning_rate"],
        "optimizer": knobs["optimizer"],
        "weight_decay": knobs["weight_decay"],
        "warmup_ratio": knobs["warmup_ratio"],
        "mlm_probability": knobs["mlm_probability"],
        "gradient_accumulation_steps": knobs["gradient_accumulation_steps"],
        "momentum": knobs["momentum"],
        "sample_seed": sample_seed,
        "sample_size": sample_size,
        "benchmark_sample_size": task_benchmark_sample_size({"hf_task": task_spec.hf_task}),
        "max_length": max_length,
        "timeout_s": budgets["timeout_s"],
        "max_train_time_s": budgets["max_train_time_s"],
        "max_eval_time_s": budgets["max_eval_time_s"],
        "device": "auto",
        "mixed_precision": knobs["mixed_precision"],
        "precision_type": knobs["precision_type"],
        "save_weights": False,
        "num_workers": 0,
        "text_column": dataset_spec.get("text_column"),
        "image_column": dataset_spec.get("image_column"),
        "label_column": dataset_spec.get("label_column"),
        "mask_column": dataset_spec.get("mask_column"),
        "question_column": dataset_spec.get("question_column"),
        "answer_column": dataset_spec.get("answer_column"),
        "ranking_label_column": dataset_spec.get("ranking_label_column"),
        "vqa_label_mode": dataset_spec.get("vqa_label_mode"),
        "vqa_answer_vocab_size": dataset_spec.get("vqa_answer_vocab_size"),
        "vqa_unseen_answer_policy": dataset_spec.get("vqa_unseen_answer_policy"),
        "retrieval_positive_policy": dataset_spec.get("retrieval_positive_policy"),
        "missing_pair_handling": dataset_spec.get("missing_pair_handling"),
        "on_decode_error": dataset_spec.get("on_decode_error"),
        "report_decode_errors": dataset_spec.get("report_decode_errors"),
        "source_max_length": source_max_length,
        "target_max_length": target_max_length,
        "dynamic_padding": dataset_spec.get("dynamic_padding") if dataset_spec.get("dynamic_padding") is not None else task_spec.modality == "text",
        "column_mapping": json.dumps(dataset_spec.get("column_mapping"), sort_keys=True) if dataset_spec.get("column_mapping") else None,
        "explainability_enabled": bool((model.get("explainability") or dataset_spec.get("explainability") or {}).get("supported", True)),
        "enable_perturbation_metrics": True,
        "perturbation_stage_logging": True,
        "perturbation_progress_logging": False,
        **manifest_evaluation_defaults(),
        "perturbation_progress_sample_interval": 1,
        "perturbation_random_strength": 0.02,
        "explainability_method": _preferred_explainability(model, dataset_spec),
        "explainability_target": (model.get("explainability") or dataset_spec.get("explainability") or {}).get("target_type"),
        "service_source": "hf_registry",
        "model_role": defaults["model_role"],
        "fit_decision": "compatible",
        "fit_reason": (
            f"quality-filtered registry pair task={task_key} training_regime={training_regime} "
        ),
        "fit_quality_score": fit_quality_score,
        "realism_score": model.get("realism_score"),
        "domain_alignment": dataset_spec.get("domain_alignment"),
        "dataset_hint": dataset_spec.get("dataset_hint"),
        "hf_pipeline_tag": model.get("pipeline_tag") or task_spec.pipeline_tag,
        "hf_downloads": model.get("downloads"),
        "hf_likes": model.get("likes"),
        "model_size": model.get("model_size") or estimated_params_count,
        "params_count": model.get("params_count") or estimated_params_count,
        "hf_author": model.get("author"),
        "hf_url": model.get("url"),
        "hf_service_meta_json": json.dumps(
            {
                "model_family": model.get("family"),
                "dataset_registry_key": dataset_spec.get("registry_key"),
                "model_registry_key": model.get("registry_key"),
                "training_regime": training_regime,
                "estimated_model_params_m": estimated_params_m,
                "fit_quality_score": fit_quality_score,
                "dataset_variant": dataset_variant,
                "split_variant": split_variant,
                "knob_variant": knob_variant,
            },
            sort_keys=True,
        ),
    }
    from ...task_contract import resolve_sentence_pair_contract
    resolve_sentence_pair_contract(row, dataset_spec=dataset_spec)
    from ...precision_policy import apply_precision_policy
    apply_precision_policy(row, model_family=model.get("family"))
    from ...segmentation_policy import apply_segmentation_label_policy
    apply_segmentation_label_policy(row)
    from ...mobilebert_policy import apply_mobilebert_generation_defaults
    apply_mobilebert_generation_defaults(row, model_family=model.get("family"))
    row["case_name"] = (
        f"{task_spec.task_label}__{model.get('hf_model_id')}__{dataset_spec.get('dataset_name')}"
        f"__{training_regime}__d{dataset_variant}__s{split_variant}__k{knob_variant}"
    )
    row["notes"] = "Reviewed service row generated from model and dataset registries"
    row["service_id"] = _service_id(
        f"hf_{task_spec.task_label}",
        {
            "model": row["hf_model_id"],
            "dataset": row["dataset_name"],
            "dataset_config": row["dataset_config"],
            "training_regime": training_regime,
            "budgets": budgets,
            "dataset_variant": dataset_variant,
            "split_variant": split_variant,
            "knob_variant": knob_variant,
            "sample_seed": sample_seed,
        },
    )
    row["service_config"] = _service_config(row)
    return row


def _preferred_explainability(model: dict[str, Any], dataset_spec: dict[str, Any]) -> str | None:
    payload = model.get("explainability") or dataset_spec.get("explainability") or {}
    methods = payload.get("preferred_methods")
    if isinstance(methods, list) and methods:
        return str(methods[0])
    return None


def _model_candidates(task_key: str) -> list[dict[str, Any]]:
    candidates = []
    for registry_key, model in MODEL_REGISTRY.items():
        if model.get("task_key") == task_key:
            copied = dict(model)
            copied["registry_key"] = registry_key
            candidates.append(copied)
    return candidates


def _dataset_candidates(task_key: str, model: dict[str, Any], *, training_regime: str) -> list[dict[str, Any]]:
    allowed = set(model.get("dataset_keys") or [])
    inference_allowed = set(model.get("inference_dataset_keys") or [])
    if training_regime == "inference_only" and inference_allowed:
        allowed = inference_allowed
    candidates = []
    for registry_key, dataset in DATASET_REGISTRY.items():
        if dataset.get("task_key") != task_key:
            continue
        if allowed and registry_key not in allowed:
            continue
        copied = dict(dataset)
        copied["registry_key"] = registry_key
        candidates.append(copied)
    return candidates


def _selected_generic_cases(requested_task_keys: list[str]) -> list[dict[str, Any]]:
    requested = set(requested_task_keys)
    return [case for case in GENERIC_MANIFEST_CASES if case["task_key"] in requested]


def _sample_cnn_hidden_layers(case: dict[str, Any], *, seed: int) -> list[int]:
    # A separate family RNG keeps the draw independent of client count, task
    # ordering and the draws used for training recipes.
    contract = {key: case.get(key) for key in (
        "task_key", "model_type", "dataset", "dataset_name", "dataset_config",
    )}
    rng = random.Random(f"cnn_hidden_layers_v1:{seed}:{sha256_json(contract)}")
    return list(rng.choice(CNN_HIDDEN_LAYER_POOL))


def _row_from_generic_case(
    *,
    case: dict[str, Any],
    profile: ManifestProfile,
    budgets: dict[str, int],
    target_sample_size: int,
    dataset_variant: int,
    split_variant: int,
    knob_variant: int,
    seed: int,
) -> dict[str, Any]:
    del profile
    split_strategy, distribution_param, skew_axis, skew_axis_config = _generic_split_knobs_for_variant(case, split_variant)
    training_knobs = _generic_training_knobs_for_variant(case, knob_variant)
    row = {
        **case,
        "enabled": True,
        "case_name": f"{case['task_label']}__{case['dataset_name']}__generic__d{dataset_variant}__s{split_variant}__k{knob_variant}",
        "notes": "Reviewed generic service row",
        "training_regime": "generic",
        "dataset_variant": dataset_variant,
        "split_variant": split_variant,
        "knob_variant": knob_variant,
        "split_strategy": split_strategy,
        "skew_axis": skew_axis,
        "skew_axis_config": json.dumps(skew_axis_config, sort_keys=True) if skew_axis_config else None,
        "distribution_type": split_strategy,
        "distribution_param": distribution_param,
        "custom_distributions": None,
        "training_epochs": training_knobs["training_epochs"],
        "batch_size": training_knobs["batch_size"],
        "learning_rate": training_knobs["learning_rate"],
        "optimizer": training_knobs["optimizer"],
        "weight_decay": training_knobs["weight_decay"],
        "momentum": training_knobs["momentum"],
        "sample_size": target_sample_size,
        "benchmark_sample_size": evaluation_settings({})["benchmark_sample_size"],
        "timeout_s": budgets["timeout_s"],
        "max_train_time_s": budgets["max_train_time_s"],
        "max_eval_time_s": budgets["max_eval_time_s"],
        "mixed_precision": False,
        "precision_type": "fp16",
        "num_workers": 0,
        "service_source": "generic_registry",
        "model_role": "service",
        "fit_decision": "compatible",
        "fit_reason": "generic manifest-compatible task runner",
        "fit_quality_score": 70,
        "explainability_enabled": True,
        "enable_perturbation_metrics": True,
        "perturbation_stage_logging": True,
        "perturbation_progress_logging": False,
        **manifest_evaluation_defaults(),
        "perturbation_progress_sample_interval": 1,
        "perturbation_random_strength": 0.02,
        "device": "auto",
        "save_weights": False,
    }
    for clustering_key in ("clustering_k", "clustering_n_init", "clustering_max_iter", "clustering_tol"):
        if training_knobs.get(clustering_key) is not None:
            row[clustering_key] = training_knobs[clustering_key]
    row["service_id"] = _service_id(
        f"gen_{case['task_label']}",
        {
            "model": row["model_type"],
            "dataset": row["dataset_name"],
            "training_regime": "generic",
            "budgets": budgets,
            "dataset_variant": dataset_variant,
            "split_variant": split_variant,
            "knob_variant": knob_variant,
            "seed": seed,
        },
    )
    row["service_config"] = _service_config(row)
    return row


def _default_fedavg_runs(profile: ManifestProfile) -> int:
    return {"test": 2, "balanced": 50, "benchmark": 200}.get(profile.name, 50)


def _fedavg_client_knobs_for_row(*, base_row: dict[str, Any], client_index: int, task_key: str) -> dict[str, Any]:
    model = {"hf_model_id": base_row.get("hf_model_id")}
    return fedavg_client_knobs(
        base_row=base_row,
        client_index=client_index,
        task_key=task_key,
        profile_batch_sizes=_batch_sizes_for(task_key, model),
        profile_learning_rates=_learning_rates_for(task_key, model),
        profile_epochs=_epochs_for(task_key),
        profile_optimizers=_optimizers_for(task_key),
    )


def _collect_fedavg_run_seed_rows(
    *,
    requested_task_keys: list[str],
    selected_training_regimes: list[str],
    models_per_task: int,
    datasets_per_model: int,
    dataset_variants_per_pair: int,
    split_variants_per_pair: int,
    budgets: dict[str, int],
    target_sample_size: int,
    seed: int,
    max_models_per_family: int | None,
    strict_inference_dataset_match: bool,
    rng: random.Random,
) -> list[dict[str, Any]]:
    run_seeds: list[dict[str, Any]] = []
    for task_key in requested_task_keys:
        task_spec = TASK_SPECS.get(task_key)
        if task_spec is not None:
            models = _model_candidates(task_key)
            models = _sort_models(
                models,
                task_key=task_key,
                selected_training_regimes=selected_training_regimes,
                rng=rng,
            )
            if max_models_per_family:
                models = _cap_models_per_family(models, int(max_models_per_family))
            model_limit = _normalise_positive_int(models_per_task)
            for model in models[:model_limit]:
                for training_regime in selected_training_regimes:
                    if training_regime == "inference_only":
                        continue
                    if not _quality_allows_model_regime(model, training_regime):
                        continue
                    datasets = _dataset_candidates(task_key, model, training_regime=training_regime)
                    datasets = [
                        dataset
                        for dataset in datasets
                        if _quality_allows_combo(
                            task_key=task_key,
                            model=model,
                            dataset_spec=dataset,
                            training_regime=training_regime,
                            strict_inference_dataset_match=strict_inference_dataset_match,
                        )
                    ]
                    datasets = _sort_datasets(
                        datasets,
                        task_key=task_key,
                        rng=rng,
                    )
                    dataset_limit = _normalise_positive_int(datasets_per_model)
                    for dataset in datasets[:dataset_limit]:
                        for dataset_variant in range(_normalise_positive_int(dataset_variants_per_pair)):
                            for split_variant in range(_normalise_positive_int(split_variants_per_pair)):
                                row = _row_from_registry(
                                    task_key=task_key,
                                    task_spec=task_spec,
                                    model=model,
                                    dataset_spec=dataset,
                                    budgets=budgets,
                                    target_sample_size=target_sample_size,
                                    training_regime=training_regime,
                                    dataset_variant=dataset_variant,
                                    split_variant=split_variant,
                                    knob_variant=0,
                                    seed=seed,
                                )
                                if _row_is_manifest_eligible(row, check_family=False) and is_fedavg_eligible_row(row):
                                    run_seeds.append(row)
            continue

        for case in _selected_generic_cases([task_key]):
            for dataset_variant in range(_normalise_positive_int(dataset_variants_per_pair)):
                for split_variant in range(_normalise_positive_int(split_variants_per_pair)):
                    row = _row_from_generic_case(
                        case=case,
                        profile=_resolve_manifest_profile("balanced"),
                        budgets=budgets,
                        target_sample_size=target_sample_size,
                        dataset_variant=dataset_variant,
                        split_variant=split_variant,
                        knob_variant=0,
                        seed=seed,
                    )
                    if is_fedavg_eligible_row(row):
                        run_seeds.append(row)
    return run_seeds


def _independent_clients(base: dict, *, count: int, seed: int, model: dict | None = None,
                         heterogeneity_intensity: float = 0.5, average_samples: int,
                         regression_support_policy: str = "general", min_target_bin_fraction: float = 0.6) -> list[dict]:
    """Attach shared recipes to a selected registry contract, without loading it."""
    from ...tasks.contracts import uses_task_protocol, VERSION as TASK_CONTRACT_VERSION
    if uses_task_protocol(base):
        base = {**base, "task_contract_version": TASK_CONTRACT_VERSION}
    context = {**(model or {}), **base}
    policy = training.build_training_policy({"config": context}, task=base["task"])
    namespace = sha256_json({"base": base["service_id"], "seed": seed, "count": count,
                             "intensity": heterogeneity_intensity, "policy": sha256_json(policy)})
    sampled = recipes.sample_client_recipes(policy, count=count, seed=namespace,
        average_samples=average_samples, quantity_seed=str(seed),
        regression_support_policy=regression_support_policy, min_target_bin_fraction=min_target_bin_fraction,
        heterogeneity_intensity=heterogeneity_intensity)
    rows = []
    for client in sampled:
        client = {key: value for key, value in client.items() if key != "client_id"}
        row = {**base, **client, "knob_variant": client["client_partition_index"],
               "distribution_type": client["split_strategy"], "distribution_param": None}
        training.validate_training_recipe(row)
        if not _row_is_manifest_eligible(row):
            raise ValueError(f"shared client policy produced an unsupported recipe for {base['case_name']}")
        row["service_id"] = _service_id("svc", {key: value for key, value in row.items() if key != "service_config"})
        row["service_config"] = _service_config(row)
        rows.append(row)
    return rows


def build_hf_manifest(
    json_path: str | None = None,
    task_keys: list[str] | None = None,
    models_per_task: int = 10,
    datasets_per_model: int = 1,
    training_regimes: list[str] | None = None,
    dataset_variants_per_pair: int | None = None,
    split_variants_per_pair: int | None = None,
    knob_variants_per_pair: int | None = None,
    total_services: int | None = None,
    seed: int | None = None,
    manifest_profile: str = "balanced",
    timeout_s: int = 1800,
    max_train_time_s: int = 120,
    max_eval_time_s: int = 120,
    avg_sample_size: int | None = None,
    max_models_per_family: int | None = None,
    strict_inference_dataset_match: bool = True,
    generation_mode: str = "independent",
    fedavg_runs: int | None = None,
    fedavg_clients_per_run: int = 10,
    fedavg_rounds: int = 3,
    fedavg_client_participation_rate: float = 1.0,
    fedavg_heterogeneity_intensity: float | None = None,
    fedavg_source_pool_cap: int | None = None,
    clients_per_family: int | None = None,
    initialization_seed: int | None = None,
    evaluation_seed: int | None = None,
    metric_evaluation_profile: str = "benchmark",
    heterogeneity_intensity: float = 0.5,
    average_samples: int | None = None,
    regression_support_policy: str = "general",
    min_target_bin_fraction: float = 0.6,
) -> pd.DataFrame:
    del json_path
    if not 0 <= heterogeneity_intensity <= 1:
        raise ValueError("heterogeneity intensity must be in [0,1]")
    variants_explicit = any(v is not None for v in (dataset_variants_per_pair, split_variants_per_pair, knob_variants_per_pair))
    if clients_per_family is not None and variants_explicit:
        raise ValueError("--clients-per-family cannot be combined with explicit variant counts")
    if clients_per_family is not None and clients_per_family < 1:
        raise ValueError("--clients-per-family must be positive")
    dataset_variants_per_pair = dataset_variants_per_pair or 1
    split_variants_per_pair = split_variants_per_pair or 1
    knob_variants_per_pair = (knob_variants_per_pair or 1) if variants_explicit or generation_mode == "fedavg" else (clients_per_family or 100)
    seed = _resolve_manifest_seed(seed)
    initialization_seed = seed if initialization_seed is None else int(initialization_seed)
    evaluation_seed = evaluation_settings({"evaluation_seed": evaluation_seed})["evaluation_seed"]
    resolve_evaluation_plan({"metric_evaluation_profile": metric_evaluation_profile})
    rng = random.Random(seed)
    profile = _resolve_manifest_profile(manifest_profile)
    budgets = {"timeout_s": timeout_s, "max_train_time_s": max_train_time_s, "max_eval_time_s": max_eval_time_s}
    for key, value in budgets.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if average_samples is not None:
        if avg_sample_size is not None and avg_sample_size != average_samples:
            raise ValueError("average_samples and avg_sample_size must agree")
        avg_sample_size = average_samples
    target_sample_size = _target_sample_size(
        profile=profile,
        avg_sample_size=avg_sample_size,
    )
    requested_task_keys = task_keys or (list(TASK_SPECS) + sorted(GENERIC_MANIFEST_TASK_KEYS))
    selected_training_regimes = training_regimes or ["finetune_transfer"]
    selected_training_regimes = [str(item).strip().lower() for item in selected_training_regimes if str(item).strip()]
    mode = str(generation_mode or "independent").strip().lower()

    if mode == "fedavg":
        if total_services is not None:
            raise ValueError("--total-services is independent-mode only; use --fedavg-runs")
        if int(fedavg_clients_per_run) < 4:
            raise ValueError("fedavg_clients_per_run must be at least 4")
        if fedavg_source_pool_cap is not None and int(fedavg_source_pool_cap) < 1:
            raise ValueError("fedavg_source_pool_cap must be positive or omitted")
        if int(fedavg_rounds) < 2:
            raise ValueError("fedavg_rounds must be at least 2")
        if not 0.0 < float(fedavg_client_participation_rate) <= 1.0:
            raise ValueError("fedavg_client_participation_rate must be in (0, 1]")
        run_seeds = _collect_fedavg_run_seed_rows(
            requested_task_keys=requested_task_keys,
            selected_training_regimes=selected_training_regimes,
            models_per_task=models_per_task,
            datasets_per_model=datasets_per_model,
            dataset_variants_per_pair=dataset_variants_per_pair,
            split_variants_per_pair=split_variants_per_pair,
            budgets=budgets,
            target_sample_size=target_sample_size,
            seed=seed,
            max_models_per_family=max_models_per_family,
            strict_inference_dataset_match=strict_inference_dataset_match,
            rng=rng,
        )
        run_limit = fedavg_runs
        if run_limit is None:
            run_limit = _default_fedavg_runs(profile)
        run_seeds = run_seeds[: max(0, int(run_limit))]
        if fedavg_heterogeneity_intensity is None:
            intensities = stratified_heterogeneity_intensities(len(run_seeds), seed=seed)
        else:
            intensity = float(fedavg_heterogeneity_intensity)
            if not 0.0 <= intensity <= 1.0:
                raise ValueError("fedavg_heterogeneity_intensity must be in [0, 1]")
            intensities = [intensity] * len(run_seeds)
        rows: list[dict[str, Any]] = []
        for run_index, (base_row, intensity) in enumerate(zip(run_seeds, intensities)):
            base_row = dict(base_row)
            base_row["fedavg_source_pool_cap"] = fedavg_source_pool_cap
            base_row["evaluation_seed"] = evaluation_seed
            base_row.update(manifest_evaluation_defaults(metric_evaluation_profile))
            rows.extend(
                expand_fedavg_run(
                    base_row,
                    run_index=run_index,
                    clients_per_run=int(fedavg_clients_per_run),
                    rounds=int(fedavg_rounds),
                    participation_rate=float(fedavg_client_participation_rate),
                    seed=seed,
                    heterogeneity_intensity=intensity,
                    average_samples=target_sample_size,
                    service_id_fn=_service_id,
                    service_config_fn=_service_config,
                    client_knobs_fn=_fedavg_client_knobs_for_row,
                )
            )
        df = pd.DataFrame(rows)
        return df.reindex(columns=MANIFEST_COLUMNS)

    rows: list[dict[str, Any]] = []
    exclusions: dict[tuple, dict[str, Any]] = {}

    def family_datasets(datasets, model, training_regime):
        if training_regime == "inference_only":
            return datasets
        accepted = []
        for dataset in datasets:
            valid_variant = False
            for variant in range(_normalise_positive_int(split_variants_per_pair)):
                selected = {**dataset, "train_split": _split_value(dataset, "train_split", variant),
                            "test_split": _split_value(dataset, "test_split", variant)}
                error = hf_dataset_error(selected, dataset_spec=dataset)
                if error is None:
                    valid_variant = True
                    continue
                key = (task_key, dataset.get("registry_key"), selected["train_split"], selected["test_split"], error)
                record = exclusions.setdefault(key, {
                    "task": task_key, "dataset_name": dataset.get("dataset_name"),
                    "dataset_config": dataset.get("dataset_config"),
                    "train_split": selected["train_split"], "test_split": selected["test_split"],
                    "reason": error, "model_ids": [],
                })
                if model.get("hf_model_id") not in record["model_ids"]:
                    record["model_ids"].append(model.get("hf_model_id"))
            if valid_variant:
                accepted.append(dataset)
        return accepted

    for task_key in requested_task_keys:
        task_spec = TASK_SPECS.get(task_key)
        if task_spec is None:
            continue
        models = _model_candidates(task_key)
        models = _sort_models(
            models,
            task_key=task_key,
            selected_training_regimes=selected_training_regimes,
            rng=rng,
        )
        model_limit = _normalise_positive_int(models_per_task)
        emitted_models = 0
        emitted_families: dict[str, int] = {}
        for model in models:
            if emitted_models >= model_limit:
                break
            architecture = str(model.get("family") or model.get("registry_key") or "unknown")
            if max_models_per_family and emitted_families.get(architecture, 0) >= int(max_models_per_family):
                continue
            rows_before_model = len(rows)
            for training_regime in selected_training_regimes:
                if not _quality_allows_model_regime(model, training_regime):
                    continue
                datasets = _dataset_candidates(task_key, model, training_regime=training_regime)
                datasets = [
                    dataset
                    for dataset in datasets
                    if _quality_allows_combo(
                        task_key=task_key,
                        model=model,
                        dataset_spec=dataset,
                        training_regime=training_regime,
                        strict_inference_dataset_match=strict_inference_dataset_match,
                    )
                ]
                datasets = family_datasets(datasets, model, training_regime)
                datasets = _sort_datasets(
                    datasets,
                    task_key=task_key,
                    rng=rng,
                )
                dataset_limit = _normalise_positive_int(datasets_per_model)
                for dataset in datasets[:dataset_limit]:
                    for dataset_variant in range(_normalise_positive_int(dataset_variants_per_pair)):
                        for split_variant in range(_normalise_positive_int(split_variants_per_pair)):
                            if training_regime != "inference_only" and hf_dataset_error({
                                **dataset, "train_split": _split_value(dataset, "train_split", split_variant),
                                "test_split": _split_value(dataset, "test_split", split_variant),
                            }, dataset_spec=dataset):
                                continue
                            effective_knob_variants = (
                                1 if training_regime == "inference_only"
                                else _normalise_positive_int(knob_variants_per_pair)
                            )
                            emitted = 0
                            candidate_count = effective_knob_variants if variants_explicit else 8
                            for knob_variant in range(candidate_count):
                                row = _row_from_registry(
                                    task_key=task_key,
                                    task_spec=task_spec,
                                    model=model,
                                    dataset_spec=dataset,
                                    budgets=budgets,
                                    target_sample_size=target_sample_size,
                                    training_regime=training_regime,
                                    dataset_variant=dataset_variant,
                                    split_variant=split_variant,
                                    knob_variant=knob_variant,
                                    seed=seed,
                                )
                                if not variants_explicit and _mlm_probabilities_for(task_key):
                                    # Masking is prepared data: freeze it within the
                                    # family rather than changing the shared source.
                                    row["mlm_probability"] = _mlm_probabilities_for(task_key)[0]
                                    row["service_config"] = _service_config(row)
                                if _row_is_manifest_eligible(row):
                                    if not variants_explicit and training_regime != "inference_only":
                                        rows.extend(_independent_clients(row, count=effective_knob_variants, seed=seed,
                                            model=model, heterogeneity_intensity=heterogeneity_intensity, average_samples=target_sample_size,
                                            regression_support_policy=regression_support_policy, min_target_bin_fraction=min_target_bin_fraction))
                                        break
                                    rows.append(row)
                                    emitted += 1
                                    if emitted == effective_knob_variants:
                                        break
            if len(rows) > rows_before_model:
                emitted_models += 1
                emitted_families[architecture] = emitted_families.get(architecture, 0) + 1

    for case in _selected_generic_cases(requested_task_keys):
        if case.get("model_type") == "cnn":
            case = {**case, "hidden_layers": _sample_cnn_hidden_layers(case, seed=seed)}
        for dataset_variant in range(_normalise_positive_int(dataset_variants_per_pair)):
            for split_variant in range(_normalise_positive_int(split_variants_per_pair)):
                for knob_variant in range(knob_variants_per_pair if variants_explicit else 1):
                    row = _row_from_generic_case(case=case, profile=profile,
                        budgets=budgets, target_sample_size=target_sample_size,
                        dataset_variant=dataset_variant, split_variant=split_variant,
                        knob_variant=knob_variant, seed=seed)
                    if variants_explicit:
                        rows.append(row)
                    else:
                        rows.extend(_independent_clients(row, count=knob_variants_per_pair, seed=seed,
                            heterogeneity_intensity=heterogeneity_intensity, average_samples=target_sample_size,
                            regression_support_policy=regression_support_policy, min_target_bin_fraction=min_target_bin_fraction))

    if total_services is not None and total_services >= 0:
        rows = _balanced_family_limit(rows, int(total_services), seed=seed)
    if variants_explicit or total_services is not None:
        groups = {}
        for row in rows:
            if row["training_regime"] == "inference_only":
                continue
            key = tuple(row.get(k) for k in ("task", "hf_model_id", "model_type", "dataset_name", "dataset_config", "training_regime"))
            groups.setdefault(key, []).append(row)
        for group in groups.values():
            quantities = recipes.stratified_sample_quantities(target_sample_size, count=len(group), seed=str(seed))
            if (all(row.get("recipe_sampler_version") == training.SAMPLER_VERSION for row in group)
                    and len({row["training_policy_hash"] for row in group}) == 1):
                # Truncation must not accidentally remove the intentional band
                # allocation. Resample knobs for the final roster, not outcomes.
                knobs = training.sample_training_knobs(group[0]["training_policy"],
                    [row["split_strategy"] for row in group], seed=f"{seed}:final_roster")
                for index, row in enumerate(group):
                    row.pop("calibration_band", None)
                    row.pop("calibration_template_id", None)
                    row.update(knobs[index])
            recipes.assign_shard_support_quantiles(group, seed=str(seed))
            for row, quantity in zip(group, quantities):
                quantity *= row.get("training_policy", {}).get("sample_quantity_multiplier", 1)
                from ...clients.capacity import known_training_capacity, scale_quantity, CAPACITY_POLICY
                maximum = recipes.sample_quantity_bounds(target_sample_size)[1] * row.get("training_policy", {}).get("sample_quantity_multiplier", 1)
                row["unconstrained_sample_size"] = quantity
                quantity, row["max_client_quantity"] = scale_quantity(quantity, maximum, known_training_capacity(row))
                row["quantity_capacity_policy"] = CAPACITY_POLICY
                row["sample_size"] = quantity
                if row.get("recipe_sampler_version"):
                    row["requested_quota"] = quantity
                    training.constrain_training_opportunity(row, seed=f"{seed}:{row['service_id']}")
                    training.validate_training_recipe(row)
                row["service_id"] = _service_id("svc", {k: v for k, v in row.items() if k != "service_config"})
                row["service_config"] = _service_config(row)
    for row in rows:
        row.update(manifest_evaluation_defaults(metric_evaluation_profile))
        row["service_config"] = _service_config(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["seed"] = seed
        df["initialization_seed"] = int(initialization_seed)
        df["evaluation_seed"] = int(evaluation_seed)
        if "sample_seed" not in df:
            df["sample_seed"] = None
        missing_seed = df["sample_seed"].isna()
        if missing_seed.any():
            df.loc[missing_seed, "sample_seed"] = df.loc[missing_seed, "service_id"].map(lambda sid: int(hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()[:8], 16))
        if "training_seed" not in df:
            df["training_seed"] = df["sample_seed"]
        else:
            df["training_seed"] = df["training_seed"].fillna(df["sample_seed"])
        evaluation_plan = resolve_evaluation_plan({"metric_evaluation_profile": metric_evaluation_profile, "evaluation_seed": evaluation_seed})
        df["service_id"] = df["service_id"].map(lambda sid: _service_id("svc", {"recipe": sid,
            "initialization_seed": initialization_seed, "evaluation_seed": evaluation_seed, "metric_evaluation_plan": evaluation_plan}))
        df["service_config"] = df.apply(lambda row: json.dumps({
            **json.loads(row["service_config"]),
            **{key: int(row[key]) for key in ("seed", "sample_seed", "initialization_seed", "evaluation_seed", "training_seed")},
        }, sort_keys=True), axis=1)
    if total_services is not None and total_services >= 0:
        df = df.head(int(total_services))
    from ...config import CONFIG
    checkpoint_defaults = {key: CONFIG[key] for key in (
        "checkpoint_patch_operation_policy", "checkpoint_measure_full", "checkpoint_weight_format",
        "checkpoint_reduced_compression", "checkpoint_reduced_compression_level",
        "checkpoint_validation_samples", "checkpoint_quality_tolerance",
        "checkpoint_patch_compression", "checkpoint_patch_compression_level")}
    for key, value in checkpoint_defaults.items():
        df[key] = value
    if not df.empty:
        df["service_config"] = df["service_config"].map(lambda value: json.dumps({**json.loads(value), **checkpoint_defaults}, sort_keys=True))
    df = df.reindex(columns=MANIFEST_COLUMNS)
    df.attrs["family_preflight_exclusions"] = list(exclusions.values())
    return df


def _balanced_family_limit(rows: list[dict[str, Any]], limit: int, *, seed: int) -> list[dict[str, Any]]:
    """Select families in random task-balanced rounds, retaining client groups.

    Only the last selected family may be partial. Exhausted tasks leave the
    rotation; a task with more registered families cannot dominate early slots.
    """
    if limit >= len(rows):
        return rows
    tasks: dict[str, dict[tuple, list[dict[str, Any]]]] = {}
    for row in rows:
        task = str(row.get("task_key") or row.get("task"))
        family = tuple(row.get(k) for k in (
            "task", "hf_model_id", "model_type", "dataset_name", "dataset_config",
            "training_regime", "dataset_variant", "split_variant"))
        tasks.setdefault(task, {}).setdefault(family, []).append(row)
    rng = random.Random(f"uniform_family_selection_v1:{seed}")
    queues = {task: list(families.values()) for task, families in sorted(tasks.items())}
    for families in queues.values():
        rng.shuffle(families)
    selected = []
    while queues and len(selected) < limit:
        order = list(queues)
        rng.shuffle(order)
        for task in order:
            family = queues[task].pop()
            selected.extend(family[:limit - len(selected)])
            if not queues[task]:
                del queues[task]
            if len(selected) == limit:
                break
    return selected


def _cap_models_per_family(models: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    selected = []
    for model in models:
        family = str(model.get("family") or model.get("registry_key") or "unknown")
        if counts.get(family, 0) >= limit:
            continue
        counts[family] = counts.get(family, 0) + 1
        selected.append(model)
    return selected


def save_manifest(df: pd.DataFrame, output_path: Path, sheet_name: str = "services", *,
                  manifest_format: str = "review") -> None:
    if manifest_format not in {"review", "expanded"}:
        raise ValueError("manifest_format must be review or expanded")
    if output_path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        raise ValueError("Output path must end with .csv or .xlsx")
    exclusions = df.attrs.get("family_preflight_exclusions", [])
    if "family_preflight_exclusions" in df.attrs:
        from ...task_capabilities import resolve_capabilities
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_groups = {}
        fields = ("task", "hf_model_id", "model_type", "dataset_name", "dataset_config", "max_length")
        for row in df.to_dict("records"):
            context = {k: None if pd.isna(row.get(k)) else row.get(k) for k in fields}
            entry = recipe_groups.setdefault(sha256_json(context), {"context": context, "rows": []})
            entry["rows"].append(row)
        output_path.with_suffix(".preflight.json").write_text(json.dumps({
            "version": "independent_family_preflight_v1",
            "scope": "offline registry checks; loaded data is checked again during preparation",
            "excluded_dataset_contracts": exclusions,
            "task_capabilities": [{**g["context"], "capabilities": resolve_capabilities(g["rows"][0])}
                for g in recipe_groups.values()],
            "recipe_diversity": [{**g["context"], **recipes.recipe_summary(g["rows"])} for g in recipe_groups.values()],
        }, indent=2), encoding="utf-8")
    # Historical federated/parent documents retain their embedded recipe format.
    historical = "generation_mode" in df and df["generation_mode"].isin(["fedavg", "shared_parent"]).any()
    if manifest_format == "review" and not historical:
        from .review import save_review_manifest
        save_review_manifest(df, output_path, sheet_name=sheet_name)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.map(lambda value: canonical_json(value) if isinstance(value, (dict, list)) else value)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        return
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        with pd.ExcelWriter(output_path) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            pd.DataFrame([{"enabled": True}]).to_excel(writer, sheet_name="defaults", index=False)
        return
    raise ValueError("Output path must end with .csv or .xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reviewed MLaaS service manifest rows")
    parser.add_argument("--input-json")
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--sheet", default="services")
    parser.add_argument("--manifest-format", choices=("review", "expanded"), default="review",
                        help="Compact review table with shared definitions (default), or legacy expanded rows")
    parser.add_argument("--task-keys")
    parser.add_argument("--models-per-task", type=int, default=10)
    parser.add_argument("--max-models-per-family", type=int)
    parser.add_argument("--datasets-per-model", type=int, default=1)
    parser.add_argument("--training-regimes")
    parser.add_argument("--dataset-variants-per-pair", type=int)
    parser.add_argument("--split-variants-per-pair", type=int)
    parser.add_argument("--knob-variants-per-pair", type=int)
    parser.add_argument("--clients-per-family", type=int)
    parser.add_argument("--initialization-seed", type=int, help="Optional initialization override; defaults to --seed.")
    parser.add_argument("--evaluation-seed", type=int, help="Optional evaluation override; defaults to --seed.")
    parser.add_argument("--metric-evaluation-profile", choices=("benchmark", "diagnostic", "pilot", "production"), default="benchmark")
    parser.add_argument("--heterogeneity-intensity", type=float, default=0.5)
    parser.add_argument("--regression-support-policy", choices=("general", "specialist"), default="general")
    parser.add_argument("--min-target-bin-fraction", type=float, default=0.6)
    parser.add_argument("--total-services", type=int)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--max-train-time-s", type=int, default=120)
    parser.add_argument("--max-eval-time-s", type=int, default=120)
    parser.add_argument("--manifest-profile", choices=sorted(MANIFEST_PROFILES), default="balanced")
    parser.add_argument("--average-samples", "--avg-sample-size", dest="avg_sample_size", type=int)
    parser.add_argument("--seed", type=int, help="Global seed for recipes, initialization, evaluation, and derived client seeds. Omit to generate and record a fresh seed.")
    args = parser.parse_args()

    df = build_hf_manifest(
        json_path=args.input_json,
        task_keys=_parse_csv_arg(args.task_keys),
        models_per_task=args.models_per_task,
        datasets_per_model=args.datasets_per_model,
        training_regimes=_parse_csv_arg(args.training_regimes),
        dataset_variants_per_pair=args.dataset_variants_per_pair,
        split_variants_per_pair=args.split_variants_per_pair,
        knob_variants_per_pair=args.knob_variants_per_pair,
        total_services=args.total_services,
        seed=args.seed,
        clients_per_family=args.clients_per_family,
        initialization_seed=args.initialization_seed,
        evaluation_seed=args.evaluation_seed,
        metric_evaluation_profile=args.metric_evaluation_profile,
        heterogeneity_intensity=args.heterogeneity_intensity,
        regression_support_policy=args.regression_support_policy,
        min_target_bin_fraction=args.min_target_bin_fraction,
        manifest_profile=args.manifest_profile,
        timeout_s=args.timeout_s, max_train_time_s=args.max_train_time_s, max_eval_time_s=args.max_eval_time_s,
        avg_sample_size=args.avg_sample_size,
        max_models_per_family=args.max_models_per_family,
    )
    save_manifest(df, Path(args.output), sheet_name=args.sheet, manifest_format=args.manifest_format)
    print(f"Wrote {len(df)} service rows to {args.output}")


if __name__ == "__main__":
    main()
