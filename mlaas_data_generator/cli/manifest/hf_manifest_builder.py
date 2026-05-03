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

from mlaas_data_generator.compatibility import known_bad_manifest_combo_reason, known_bad_path_reason
from mlaas_data_generator.config import DEFAULT_MANIFEST_PATH
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
    training_epochs: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    learning_rates: tuple[float, ...]
    timeout_s: int


@dataclass(frozen=True)
class ResourceTierSpec:
    name: str
    rank: int
    default_avg_sample_size: int
    timeout_s: int
    max_train_time_s: int
    max_eval_time_s: int
    perturbation_sample_count: int = 1


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
    "test": ManifestProfile("test", default_avg_sample_size=128, training_epochs=(1,), batch_sizes=(4, 8), learning_rates=(5e-5, 1e-4), timeout_s=900),
    "balanced": ManifestProfile("balanced", default_avg_sample_size=768, training_epochs=(1, 2), batch_sizes=(8, 16), learning_rates=(2e-5, 5e-5, 1e-4), timeout_s=1800),
    "benchmark": ManifestProfile("benchmark", default_avg_sample_size=1600, training_epochs=(1, 2, 3), batch_sizes=(8, 16, 32), learning_rates=(2e-5, 5e-5, 1e-4), timeout_s=3600),
}

RESOURCE_TIERS: dict[str, ResourceTierSpec] = {
    "smoketest": ResourceTierSpec("smoketest", rank=0, default_avg_sample_size=32, timeout_s=900, max_train_time_s=45, max_eval_time_s=60),
    "light": ResourceTierSpec("light", rank=0, default_avg_sample_size=128, timeout_s=900, max_train_time_s=45, max_eval_time_s=60),
    "medium": ResourceTierSpec("medium", rank=1, default_avg_sample_size=768, timeout_s=1800, max_train_time_s=120, max_eval_time_s=120),
    "heavy": ResourceTierSpec("heavy", rank=2, default_avg_sample_size=1600, timeout_s=3600, max_train_time_s=300, max_eval_time_s=240),
    "stress_test": ResourceTierSpec(
        "stress_test",
        rank=3,
        default_avg_sample_size=4000,
        timeout_s=7200,
        max_train_time_s=900,
        max_eval_time_s=600,
        perturbation_sample_count=2,
    ),
}

PROFILE_DEFAULT_RESOURCE_TIERS = {
    "test": "light",
    "balanced": "medium",
    "benchmark": "heavy",
}

RESOURCE_TIER_ALIASES = {
    "smoke": "smoketest",
    "smoke_test": "smoketest",
    "smoke-test": "smoketest",
    "stress": "stress_test",
    "stress-test": "stress_test",
    "stresstest": "stress_test",
    "benchmark": "heavy",
    "test": "light",
}


MANIFEST_COLUMNS = [
    "service_id",
    "enabled",
    "case_name",
    "notes",
    "dataset",
    "dataset_name",
    "dataset_config",
    "train_split",
    "test_split",
    "benchmark_split",
    "model_type",
    "hf_task",
    "hf_model_id",
    "task_type",
    "task",
    "task_tag",
    "modality",
    "input_schema",
    "output_schema",
    "training_regime",
    "resource_tier",
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
    "warmup_ratio",
    "mlm_probability",
    "gradient_accumulation_steps",
    "sample_seed",
    "sample_size",
    "max_samples",
    "max_length",
    "timeout_s",
    "max_train_time_s",
    "max_eval_time_s",
    "device",
    "mixed_precision",
    "precision_type",
    "save_weights",
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
    "model_resource_tier",
    "dataset_resource_tier",
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
        "max_samples": 1200,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "adam",
    },
    {
        "task_key": "sklearn_image_classification",
        "task_label": "sk_imgcls",
        "dataset": "cifar10",
        "dataset_name": "cifar10",
        "task_type": "classification",
        "task": "classification",
        "modality": "image",
        "model_type": "randomforest",
        "input_schema": "single_image_flattened",
        "max_samples": 1000,
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
        "max_samples": 1200,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "adam",
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
        "max_samples": 1600,
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
        "max_samples": 1200,
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


def _resolve_resource_tier(resource_tier: str | None, profile: ManifestProfile) -> tuple[ResourceTierSpec, bool]:
    explicit = resource_tier is not None
    default_name = PROFILE_DEFAULT_RESOURCE_TIERS.get(profile.name, "medium")
    raw = str(resource_tier or default_name).strip().lower()
    name = RESOURCE_TIER_ALIASES.get(raw, raw)
    if name not in RESOURCE_TIERS:
        valid = ", ".join(RESOURCE_TIERS)
        raise ValueError(f"Unknown resource_tier '{resource_tier}'. Expected one of: {valid}")
    return RESOURCE_TIERS[name], explicit


def _resource_tier_rank(tier_name: str | None) -> int:
    name = RESOURCE_TIER_ALIASES.get(str(tier_name or "medium").strip().lower(), str(tier_name or "medium").strip().lower())
    spec = RESOURCE_TIERS.get(name)
    return spec.rank if spec else RESOURCE_TIERS["medium"].rank


def _resource_tier_by_rank(rank: int) -> ResourceTierSpec:
    clamped = max(0, min(int(rank), max(spec.rank for spec in RESOURCE_TIERS.values())))
    ranked = sorted(RESOURCE_TIERS.values(), key=lambda spec: (spec.rank, spec.default_avg_sample_size))
    selected = None
    for spec in ranked:
        if spec.rank == clamped:
            selected = spec
    return selected or RESOURCE_TIERS["medium"]


def _is_smoketest_tier(resource_tier: ResourceTierSpec) -> bool:
    return resource_tier.name == "smoketest"


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _normalise_positive_int(value: int | None, *, minimum: int = 1) -> int:
    if value is None:
        return minimum
    return max(minimum, int(value))


def _resolve_manifest_seed(seed: int | None) -> int:
    if seed is None:
        return secrets.randbits(63)
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
        "resource_tier": row.get("resource_tier"),
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
        "max_samples": row.get("max_samples"),
        "sample_size": row.get("sample_size"),
        "sample_seed": row.get("sample_seed"),
        "max_length": row.get("max_length"),
        "timeout_s": row.get("timeout_s"),
        "max_train_time_s": row.get("max_train_time_s"),
        "max_eval_time_s": row.get("max_eval_time_s"),
        "device": row.get("device"),
        "mixed_precision": row.get("mixed_precision"),
        "precision_type": row.get("precision_type"),
        "save_weights": row.get("save_weights"),
        "enable_perturbation_metrics": row.get("enable_perturbation_metrics"),
        "perturbation_sample_count": row.get("perturbation_sample_count"),
    }
    return json.dumps({k: v for k, v in payload.items() if v is not None}, sort_keys=True)


def _split_value(dataset_spec: dict[str, Any], key: str, variant: int) -> Any:
    value = dataset_spec.get(key)
    if isinstance(value, (list, tuple)) and value:
        return value[variant % len(value)]
    return value


def _estimated_model_params_m(model: dict[str, Any]) -> float | None:
    explicit = _as_float(model.get("estimated_params_m"))
    if explicit is not None:
        return explicit
    for key in ("params_count", "model_size"):
        value = _as_float(model.get(key))
        if value is None:
            continue
        return value / 1_000_000.0 if value > 100_000 else value

    model_id = str(model.get("hf_model_id") or model.get("registry_key") or "").lower()
    exact_estimates = {
        "sshleifer/tiny-gpt2": 0.1,
        "roneneldan/tinystories-1m": 1.0,
        "roneneldan/tinystories-33m": 33.0,
        "huawei-noah/tinybert_general_4l_312d": 14.0,
        "google/electra-small-discriminator": 14.0,
        "google/electra-small-generator": 14.0,
        "albert/albert-base-v2": 12.0,
        "microsoft/minilm-l12-h384-uncased": 33.0,
        "google/mobilebert-uncased": 25.0,
        "squeezebert/squeezebert-uncased": 51.0,
        "distilbert-base-uncased": 66.0,
        "distilbert-base-cased": 66.0,
        "distilroberta-base": 82.0,
        "distilgpt2": 82.0,
        "eleutherai/pythia-70m": 70.0,
        "gpt2": 124.0,
        "microsoft/dialogpt-small": 117.0,
        "microsoft/dialogpt-medium": 345.0,
        "eleutherai/gpt-neo-125m": 125.0,
        "facebook/opt-125m": 125.0,
        "bert-base-uncased": 110.0,
        "bert-base-cased": 110.0,
        "dslim/bert-base-ner": 110.0,
        "roberta-base": 125.0,
        "microsoft/deberta-base": 139.0,
        "google/electra-base-discriminator": 110.0,
        "google/electra-base-generator": 110.0,
        "google/flan-t5-small": 80.0,
        "t5-small": 60.0,
        "salesforce/codet5-small": 60.0,
        "google/flan-t5-base": 250.0,
        "t5-base": 220.0,
        "google/byt5-small": 300.0,
        "google/mt5-small": 300.0,
        "allenai/led-base-16384": 162.0,
        "sshleifer/distilbart-cnn-12-6": 306.0,
        "sshleifer/distilbart-xsum-12-6": 306.0,
        "google/vit-base-patch16-224": 86.0,
        "apple/mobilevit-small": 5.6,
        "microsoft/resnet-50": 25.0,
        "facebook/deit-tiny-patch16-224": 5.7,
        "facebook/convnext-tiny-224": 28.0,
        "microsoft/swin-tiny-patch4-window7-224": 28.0,
        "google/efficientnet-b0": 5.3,
        "google/mobilenet_v2_1.0_224": 3.5,
        "facebook/regnet-y-040": 21.0,
        "facebook/levit-128s": 7.8,
        "facebook/detr-resnet-50": 41.0,
        "facebook/detr-resnet-50-dc5": 41.0,
        "hustvl/yolos-small": 31.0,
        "hustvl/yolos-tiny": 6.0,
        "hustvl/yolos-base": 85.0,
        "pekingu/rtdetr_r18vd_coco_o365": 20.0,
        "pekingu/rtdetr_v2_r18vd": 20.0,
        "microsoft/conditional-detr-resnet-50": 44.0,
        "sensetime/deformable-detr": 40.0,
        "microsoft/table-transformer-detection": 28.0,
        "nvidia/segformer-b0-finetuned-ade-512-512": 3.8,
        "nvidia/segformer-b1-finetuned-ade-512-512": 14.0,
        "nvidia/segformer-b2-finetuned-ade-512-512": 25.0,
        "nvidia/segformer-b3-finetuned-ade-512-512": 45.0,
        "nvidia/segformer-b4-finetuned-ade-512-512": 62.0,
        "nvidia/segformer-b5-finetuned-ade-640-640": 82.0,
        "intel/dpt-large-ade": 344.0,
        "openmmlab/upernet-convnext-tiny": 60.0,
        "openmmlab/upernet-swin-tiny": 60.0,
        "mattmdjaga/segformer_b2_clothes": 25.0,
        "salesforce/blip-image-captioning-base": 224.0,
        "salesforce/blip-image-captioning-large": 446.0,
        "microsoft/git-base": 177.0,
        "microsoft/git-base-coco": 177.0,
        "microsoft/git-base-vatex": 177.0,
        "microsoft/git-base-textcaps": 177.0,
        "microsoft/git-large": 430.0,
        "microsoft/git-large-coco": 430.0,
        "microsoft/git-large-vatex": 430.0,
        "microsoft/git-large-textcaps": 430.0,
        "openai/clip-vit-base-patch32": 151.0,
        "openai/clip-vit-base-patch16": 150.0,
        "openai/clip-vit-large-patch14": 427.0,
        "openai/clip-vit-large-patch14-336": 427.0,
        "patrickjohncyh/fashion-clip": 151.0,
        "wkcn/tinyclip-vit-8m-16-text-3m-yfcc15m": 11.0,
        "google/siglip-base-patch16-224": 203.0,
        "google/siglip2-base-patch16-224": 203.0,
        "google/siglip-so400m-patch14-384": 878.0,
        "baai/altclip": 322.0,
        "salesforce/blip-vqa-base": 224.0,
        "salesforce/blip-vqa-capfilt-large": 446.0,
        "salesforce/blip2-opt-2.7b": 2700.0,
        "salesforce/blip2-opt-2.7b-coco": 2700.0,
        "microsoft/git-base-vqav2": 177.0,
        "microsoft/git-large-vqav2": 430.0,
        "dandelin/vilt-b32-finetuned-vqa": 113.0,
        "bingsu/temp_vilt_vqa": 113.0,
        "jeney/vilt-b32-finetuned-vqa": 113.0,
        "jmonas/vilt-33m-vqa": 33.0,
    }
    if model_id in exact_estimates:
        return exact_estimates[model_id]

    family = str(model.get("family") or "").lower()
    family_estimates = {
        "tinybert": 14.0,
        "albert": 12.0,
        "electra": 60.0,
        "mobilebert": 25.0,
        "distilbert": 66.0,
        "distilroberta": 82.0,
        "bert": 110.0,
        "roberta": 125.0,
        "deberta": 139.0,
        "gpt2": 124.0,
        "t5": 120.0,
        "bart": 306.0,
        "vit": 86.0,
        "resnet": 25.0,
        "segformer": 45.0,
        "clip": 200.0,
        "blip": 224.0,
        "git": 177.0,
        "vilt": 113.0,
    }
    return family_estimates.get(family)


def _resource_tier_for_model(model: dict[str, Any]) -> str:
    params_m = _estimated_model_params_m(model)
    if params_m is None:
        return "medium"
    if params_m <= 85:
        return "light"
    if params_m <= 250:
        return "medium"
    if params_m <= 700:
        return "heavy"
    return "stress_test"


def _dataset_cost_score(dataset_spec: dict[str, Any], task_key: str) -> float:
    samples = float(_as_int(dataset_spec.get("max_samples")) or 0)
    max_length = float(_as_int(dataset_spec.get("max_length")) or 128)
    num_classes = float(_as_int(dataset_spec.get("num_classes")) or 1)
    if task_key in {"object_detection", "image_segmentation"}:
        return samples * 4.0
    if task_key == "image_classification":
        return samples * max(1.0, min(num_classes, 200.0) / 50.0)
    if task_key in {"image_captioning", "text_image_retrieval", "visual_question_answering"}:
        return samples * 2.0
    return samples * max(1.0, max_length / 128.0)


def _resource_tier_for_dataset(dataset_spec: dict[str, Any], task_key: str) -> str:
    score = _dataset_cost_score(dataset_spec, task_key)
    if score <= 1000:
        return "light"
    if score <= 3000:
        return "medium"
    if score <= 8000:
        return "heavy"
    return "stress_test"


def _model_aware_runtime_tier(
    *,
    task_key: str,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    training_regime: str,
    resource_tier: ResourceTierSpec,
) -> ResourceTierSpec:
    effective_rank = int(resource_tier.rank)
    family = str(model.get("family") or "").strip().lower()
    model_id = str(model.get("hf_model_id") or "").strip().lower()
    params_m = _estimated_model_params_m(model) or 0.0

    if training_regime == "inference_only" and task_key == "text2text_generation":
        if family == "led" or model_id == "allenai/led-base-16384":
            effective_rank = max(effective_rank, RESOURCE_TIERS["heavy"].rank)
        elif family == "mt5":
            effective_rank = max(effective_rank, RESOURCE_TIERS["heavy"].rank)
        elif family == "t5" and (params_m >= 180.0 or model_id in {"t5-base", "google/flan-t5-base"}):
            effective_rank = max(effective_rank, RESOURCE_TIERS["heavy"].rank)
        elif family == "bart" and params_m >= 250.0:
            effective_rank = max(effective_rank, RESOURCE_TIERS["heavy"].rank)

    return _resource_tier_by_rank(effective_rank)


def _target_sample_size(
    *,
    profile: ManifestProfile,
    resource_tier: ResourceTierSpec,
    resource_tier_explicit: bool,
    avg_sample_size: int | None,
) -> int:
    if avg_sample_size is not None:
        return max(1, int(avg_sample_size))
    if resource_tier_explicit:
        return int(resource_tier.default_avg_sample_size)
    return min(int(profile.default_avg_sample_size), int(resource_tier.default_avg_sample_size))


def _max_samples(dataset_spec: dict[str, Any], target_sample_size: int, resource_tier: ResourceTierSpec, task_key: str) -> int:
    if _is_smoketest_tier(resource_tier):
        smoke_mins = {
            "object_detection": 32,
            "image_segmentation": 32,
            "image_captioning": 16,
            "text_image_retrieval": 16,
            "visual_question_answering": 16,
        }
        dataset_cap = _as_int(dataset_spec.get("max_samples")) or target_sample_size
        return max(1, min(int(dataset_cap), int(smoke_mins.get(task_key, 8))))
    task_caps = {
        "object_detection": (48, 120, 240, 480),
        "image_segmentation": (48, 96, 192, 384),
        "image_captioning": (128, 768, 2000, 4000),
        "text_image_retrieval": (128, 768, 2000, 4000),
        "visual_question_answering": (128, 768, 2000, 4000),
        "image_classification": (256, 1000, 2200, 5000),
    }
    caps = task_caps.get(task_key, (128, 768, 1600, 4000))
    tier_cap = caps[min(resource_tier.rank, len(caps) - 1)]
    dataset_cap = _as_int(dataset_spec.get("max_samples")) or target_sample_size
    requested = min(int(target_sample_size), int(tier_cap))
    return max(1, min(int(dataset_cap), requested))


def _model_aware_max_samples(
    *,
    dataset_spec: dict[str, Any],
    model: dict[str, Any],
    target_sample_size: int,
    resource_tier: ResourceTierSpec,
    task_key: str,
    training_regime: str,
) -> int:
    base = _max_samples(dataset_spec, target_sample_size, resource_tier, task_key)
    if training_regime != "inference_only" or task_key != "text2text_generation":
        return base

    family = str(model.get("family") or "").strip().lower()
    model_id = str(model.get("hf_model_id") or "").strip().lower()
    if model_id in {"t5-base", "google/flan-t5-base"}:
        return min(base, 384)
    if family in {"t5", "mt5"}:
        return min(base, 512)
    if family == "led" or model_id == "allenai/led-base-16384":
        return min(base, 320)
    if family == "bart" and (_estimated_model_params_m(model) or 0.0) >= 250.0:
        return min(base, 512)
    return base


def _max_length(dataset_spec: dict[str, Any], model: dict[str, Any], resource_tier: ResourceTierSpec, task_key: str) -> int | None:
    base = _as_int(dataset_spec.get("max_length")) or _as_int(model.get("max_length"))
    if base is None:
        return None
    caps = {
        "text_generation": (128, 192, 256, 512),
        "text2text_generation": (128, 192, 256, 512),
        "fill_mask": (96, 128, 192, 256),
        "token_classification": (96, 128, 192, 256),
        "image_captioning": (48, 64, 96, 128),
        "text_image_retrieval": (48, 64, 96, 128),
        "visual_question_answering": (32, 48, 64, 96),
    }
    default_caps = (96, 128, 192, 256)
    cap = caps.get(task_key, default_caps)[min(resource_tier.rank, 3)]
    return max(1, min(int(base), int(cap)))


def _unique_ints(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        value = max(1, int(value))
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _batch_sizes_for(task_key: str, model: dict[str, Any], resource_tier: ResourceTierSpec) -> tuple[int, ...]:
    rank = resource_tier.rank
    if task_key in {"object_detection", "image_segmentation", "image_captioning", "visual_question_answering", "text_image_retrieval"}:
        table = ((1, 2), (2, 4), (4, 8), (8, 16))
    elif task_key in {"text_generation", "text2text_generation", "token_classification"}:
        table = ((2, 4), (4, 8), (8, 16), (8, 16, 32))
    elif task_key == "image_classification":
        table = ((4, 8), (8, 16), (16, 32), (32, 64))
    else:
        table = ((4, 8), (8, 16), (16, 32), (16, 32, 64))

    params_m = _estimated_model_params_m(model)
    values = table[min(rank, len(table) - 1)]
    if params_m is not None and params_m > 700:
        values = tuple(max(1, value // 4) for value in values)
    elif params_m is not None and params_m > 250:
        values = tuple(max(1, value // 2) for value in values)
    return _unique_ints(values)


def _learning_rates_for(task_key: str, model: dict[str, Any], resource_tier: ResourceTierSpec) -> tuple[float, ...]:
    params_m = _estimated_model_params_m(model) or 100.0
    tiny_model = params_m <= 35.0 or str(model.get("family") or "").lower() in {"tinybert", "tinystories", "tinyclip", "albert"}
    if task_key in {"object_detection", "image_segmentation"}:
        rates = (5e-6, 1e-5, 2e-5)
    elif task_key == "text_image_retrieval":
        rates = (1e-6, 5e-6, 1e-5)
    elif task_key in {"text_generation", "text2text_generation", "image_captioning", "visual_question_answering"}:
        rates = (1e-5, 2e-5, 3e-5)
    elif task_key == "image_classification":
        rates = (1e-5, 2e-5, 5e-5)
    else:
        rates = (2e-5, 3e-5, 5e-5)
        if tiny_model:
            rates = (3e-5, 5e-5, 1e-4)
    if resource_tier.rank <= 0:
        return rates[:2]
    if resource_tier.rank >= 3 and tiny_model and rates[-1] < 1e-4:
        return (*rates, 1e-4)
    return rates


def _epochs_for(task_key: str, resource_tier: ResourceTierSpec) -> tuple[int, ...]:
    if task_key in {"object_detection", "image_segmentation", "text_generation", "text2text_generation", "image_captioning", "visual_question_answering", "text_image_retrieval"}:
        table = ((1,), (1,), (1, 2), (2, 3))
    else:
        table = ((1,), (1, 2), (2, 3), (3, 4))
    return table[min(resource_tier.rank, len(table) - 1)]


def _weight_decays_for(task_key: str, resource_tier: ResourceTierSpec) -> tuple[float, ...]:
    if resource_tier.rank == 0:
        return (0.0, 0.001)
    if task_key in {"text_generation", "text2text_generation", "image_captioning", "visual_question_answering"}:
        return (0.0, 0.001, 0.01)
    return (0.0, 0.001, 0.01, 0.05)


def _optimizers_for(task_key: str, resource_tier: ResourceTierSpec) -> tuple[str, ...]:
    if task_key in {"object_detection", "image_segmentation", "image_captioning", "visual_question_answering", "text_image_retrieval"}:
        return ("adamw", "adam")
    if resource_tier.rank <= 0:
        return ("adamw", "adam")
    return ("adamw", "adam", "sgd")


def _warmup_ratios_for(task_key: str, resource_tier: ResourceTierSpec) -> tuple[float, ...]:
    if task_key in {"object_detection", "image_segmentation"}:
        return (0.0, 0.03)
    if resource_tier.rank <= 0:
        return (0.0, 0.05)
    return (0.0, 0.03, 0.06, 0.1)


def _mlm_probabilities_for(task_key: str) -> tuple[float, ...]:
    if task_key == "fill_mask":
        return (0.10, 0.15, 0.20, 0.30)
    return ()


def _gradient_accumulation_steps_for(task_key: str, model: dict[str, Any], resource_tier: ResourceTierSpec) -> tuple[int, ...]:
    params_m = _estimated_model_params_m(model) or 0.0
    if task_key in {"object_detection", "image_segmentation", "image_captioning", "visual_question_answering", "text_image_retrieval"}:
        values = (1, 2, 4) if resource_tier.rank >= 1 else (1, 2)
    elif task_key in {"text_generation", "text2text_generation", "token_classification"}:
        values = (1, 2, 4) if resource_tier.rank >= 1 else (1, 2)
    else:
        values = (1, 2, 4) if resource_tier.rank >= 2 else (1, 2)
    if params_m > 250:
        values = tuple(value for value in values if value >= 2) or (2,)
    return _unique_ints(values)


def _sample_size_for_variant(max_samples: int, knob_variant: int) -> int:
    cap = max(1, int(max_samples))
    if cap <= 2:
        return 1
    if cap <= 4:
        return cap - 1
    fractions = (0.5, 0.65, 0.8, 0.95)
    frac = fractions[int(knob_variant) % len(fractions)]
    return max(1, min(cap - 1, int(round(float(cap) * frac))))


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
        "fill_mask": "masked_token_id",
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


def _split_knobs_for_variant(task_key: str, knob_variant: int, resource_tier: ResourceTierSpec) -> tuple[str, float | None, str | None, dict[str, Any] | None]:
    skew_axis = _default_skew_axis_for_task(task_key)
    skew_axis_config = _skew_axis_config_for_task(task_key)
    if resource_tier.rank == 0 or skew_axis is None or task_key in {"object_detection", "image_segmentation"}:
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
    resource_tier: ResourceTierSpec,
) -> dict[str, Any]:
    batch_sizes = _batch_sizes_for(task_key, model, resource_tier)
    split_strategy, distribution_param, skew_axis, skew_axis_config = _split_knobs_for_variant(task_key, knob_variant, resource_tier)
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

    learning_rates = _learning_rates_for(task_key, model, resource_tier)
    epochs = _epochs_for(task_key, resource_tier)
    weight_decays = _weight_decays_for(task_key, resource_tier)
    optimizers = _optimizers_for(task_key, resource_tier)
    warmup_ratios = _warmup_ratios_for(task_key, resource_tier)
    mlm_probabilities = _mlm_probabilities_for(task_key)
    grad_accum_steps = _gradient_accumulation_steps_for(task_key, model, resource_tier)
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


def _row_is_manifest_eligible(row: dict[str, Any]) -> bool:
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


def _sort_models_for_tier(
    models: list[dict[str, Any]],
    *,
    task_key: str,
    resource_tier: ResourceTierSpec,
    selected_training_regimes: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    eligible = []
    for model in models:
        if not any(_quality_allows_model_regime(model, regime) for regime in selected_training_regimes):
            continue
        model_tier = _resource_tier_for_model(model)
        if not _is_smoketest_tier(resource_tier) and _resource_tier_rank(model_tier) > resource_tier.rank:
            continue
        copied = dict(model)
        copied["_model_resource_tier"] = model_tier
        copied["_estimated_params_m"] = _estimated_model_params_m(model)
        copied["_selection_jitter"] = rng.random()
        eligible.append(copied)
    if not eligible:
        return []
    target_rank = min(
        resource_tier.rank,
        max(_resource_tier_rank(str(item.get("_model_resource_tier"))) for item in eligible),
    )
    return sorted(
        eligible,
        key=lambda item: (
            abs(_resource_tier_rank(str(item.get("_model_resource_tier"))) - target_rank),
            -_model_selection_score(task_key, item, selected_training_regimes),
            _estimated_model_params_m(item) or 1_000_000.0,
            item["_selection_jitter"],
        ),
    )


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


def _sort_datasets_for_tier(
    datasets: list[dict[str, Any]],
    *,
    task_key: str,
    resource_tier: ResourceTierSpec,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not datasets:
        return []
    eligible = []
    for dataset in datasets:
        copied = dict(dataset)
        copied["_dataset_resource_tier"] = _resource_tier_for_dataset(dataset, task_key)
        copied["_selection_jitter"] = rng.random()
        eligible.append(copied)

    if _is_smoketest_tier(resource_tier):
        return sorted(
            eligible,
            key=lambda item: (
                _dataset_cost_score(item, task_key),
                item["_selection_jitter"],
            ),
        )

    within_tier = [item for item in eligible if _resource_tier_rank(str(item.get("_dataset_resource_tier"))) <= resource_tier.rank]
    source = within_tier or eligible
    target_rank = min(
        resource_tier.rank,
        max(_resource_tier_rank(str(item.get("_dataset_resource_tier"))) for item in source),
    )
    return sorted(
        source,
        key=lambda item: (
            abs(_resource_tier_rank(str(item.get("_dataset_resource_tier"))) - target_rank),
            _dataset_cost_score(item, task_key),
            item["_selection_jitter"],
        ),
    )


def _row_from_registry(
    *,
    task_key: str,
    task_spec: TaskSpec,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    resource_tier: ResourceTierSpec,
    target_sample_size: int,
    training_regime: str,
    dataset_variant: int,
    split_variant: int,
    knob_variant: int,
    seed: int,
) -> dict[str, Any]:
    defaults = _training_regime_defaults(training_regime)
    runtime_tier = _model_aware_runtime_tier(
        task_key=task_key,
        model=model,
        dataset_spec=dataset_spec,
        training_regime=training_regime,
        resource_tier=resource_tier,
    )
    knobs = _training_knobs_for_variant(
        task_key=task_key,
        model=model,
        training_regime=training_regime,
        knob_variant=knob_variant,
        resource_tier=runtime_tier,
    )
    max_length = _max_length(dataset_spec, model, runtime_tier, task_key)
    source_max_length = dataset_spec.get("source_max_length")
    target_max_length = dataset_spec.get("target_max_length")
    if task_key == "text2text_generation" and max_length is not None:
        source_max_length = source_max_length or max_length
        target_max_length = target_max_length or min(max_length, 96 if runtime_tier.rank <= 1 else 128)
    fit_quality_score = _fit_quality_score(task_key, model, dataset_spec, training_regime)
    model_resource_tier = str(model.get("_model_resource_tier") or _resource_tier_for_model(model))
    dataset_resource_tier = str(dataset_spec.get("_dataset_resource_tier") or _resource_tier_for_dataset(dataset_spec, task_key))
    estimated_params_m = model.get("_estimated_params_m") or _estimated_model_params_m(model)
    estimated_params_count = None if estimated_params_m is None else int(float(estimated_params_m) * 1_000_000)
    max_samples = _model_aware_max_samples(
        dataset_spec=dataset_spec,
        model=model,
        target_sample_size=target_sample_size,
        resource_tier=resource_tier,
        task_key=task_key,
        training_regime=training_regime,
    )
    sample_size = (
        None
        if training_regime == "inference_only"
        else _sample_size_for_variant(max_samples, knob_variant)
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
        "resource_tier": resource_tier.name,
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
        "max_samples": max_samples,
        "max_length": max_length,
        "timeout_s": runtime_tier.timeout_s,
        "max_train_time_s": runtime_tier.max_train_time_s,
        "max_eval_time_s": runtime_tier.max_eval_time_s,
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
        "enable_perturbation_metrics": bool((model.get("explainability") or dataset_spec.get("explainability") or {}).get("supported", True)),
        "perturbation_stage_logging": True,
        "perturbation_progress_logging": False,
        "perturbation_sample_count": resource_tier.perturbation_sample_count,
        "perturbation_candidate_units": 4,
        "perturbation_target_units": 1,
        "perturbation_trust_trials": 2,
        "perturbation_progress_sample_interval": 1,
        "perturbation_random_strength": 0.02,
        "explainability_method": _preferred_explainability(model, dataset_spec),
        "explainability_target": (model.get("explainability") or dataset_spec.get("explainability") or {}).get("target_type"),
        "service_source": "hf_registry",
        "model_role": defaults["model_role"],
        "fit_decision": "compatible",
        "fit_reason": (
            f"quality-filtered registry pair task={task_key} training_regime={training_regime} "
            f"resource_tier={resource_tier.name}"
        ),
        "fit_quality_score": fit_quality_score,
        "model_resource_tier": model_resource_tier,
        "dataset_resource_tier": dataset_resource_tier,
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
                "resource_tier": resource_tier.name,
                "model_resource_tier": model_resource_tier,
                "dataset_resource_tier": dataset_resource_tier,
                "estimated_model_params_m": estimated_params_m,
                "fit_quality_score": fit_quality_score,
                "dataset_variant": dataset_variant,
                "split_variant": split_variant,
                "knob_variant": knob_variant,
            },
            sort_keys=True,
        ),
    }
    row["case_name"] = (
        f"{task_spec.task_label}__{model.get('hf_model_id')}__{dataset_spec.get('dataset_name')}"
        f"__{training_regime}__{resource_tier.name}__d{dataset_variant}__s{split_variant}__k{knob_variant}"
    )
    row["notes"] = "Reviewed service row generated from model and dataset registries"
    row["service_id"] = _service_id(
        f"hf_{task_spec.task_label}",
        {
            "model": row["hf_model_id"],
            "dataset": row["dataset_name"],
            "dataset_config": row["dataset_config"],
            "training_regime": training_regime,
            "resource_tier": resource_tier.name,
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


def _row_from_generic_case(
    *,
    case: dict[str, Any],
    profile: ManifestProfile,
    resource_tier: ResourceTierSpec,
    target_sample_size: int,
    dataset_variant: int,
    split_variant: int,
    knob_variant: int,
    seed: int,
) -> dict[str, Any]:
    split_strategy, distribution_param, skew_axis, skew_axis_config = _split_knobs_for_variant(case["task_key"], knob_variant, resource_tier)
    epochs = _epochs_for(case["task_key"], resource_tier)[int(knob_variant) % len(_epochs_for(case["task_key"], resource_tier))]
    case_batch = case.get("batch_size")
    if case_batch:
        batch_sizes = _unique_ints((max(1, min(int(case_batch), size)) for size in _batch_sizes_for(case["task_key"], case, resource_tier)))
    else:
        batch_sizes = profile.batch_sizes
    batch_size = batch_sizes[int(knob_variant) % len(batch_sizes)]
    max_samples = min(int(case.get("max_samples", target_sample_size)), int(target_sample_size))
    row = {
        **case,
        "enabled": True,
        "case_name": f"{case['task_label']}__{case['dataset_name']}__generic__{resource_tier.name}__d{dataset_variant}__s{split_variant}__k{knob_variant}",
        "notes": "Reviewed generic service row",
        "training_regime": "generic",
        "resource_tier": resource_tier.name,
        "dataset_variant": dataset_variant,
        "split_variant": split_variant,
        "knob_variant": knob_variant,
        "split_strategy": split_strategy,
        "skew_axis": skew_axis,
        "skew_axis_config": json.dumps(skew_axis_config, sort_keys=True) if skew_axis_config else None,
        "distribution_type": split_strategy,
        "distribution_param": distribution_param,
        "custom_distributions": None,
        "training_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": case.get("learning_rate"),
        "optimizer": case.get("optimizer"),
        "weight_decay": 0.0,
        "momentum": 0.0,
        "sample_size": None,
        "max_samples": max_samples,
        "timeout_s": resource_tier.timeout_s,
        "max_train_time_s": resource_tier.max_train_time_s,
        "max_eval_time_s": resource_tier.max_eval_time_s,
        "mixed_precision": False,
        "precision_type": "fp16",
        "num_workers": 0,
        "service_source": "generic_registry",
        "model_role": "service",
        "fit_decision": "compatible",
        "fit_reason": f"generic manifest-compatible task runner resource_tier={resource_tier.name}",
        "fit_quality_score": 70,
        "model_resource_tier": "light" if case.get("model_type") in {"randomforest", "kmeans"} else resource_tier.name,
        "dataset_resource_tier": resource_tier.name,
        "explainability_enabled": True,
        "enable_perturbation_metrics": True,
        "perturbation_stage_logging": True,
        "perturbation_progress_logging": False,
        "perturbation_sample_count": resource_tier.perturbation_sample_count,
        "perturbation_candidate_units": 4,
        "perturbation_target_units": 1,
        "perturbation_trust_trials": 2,
        "perturbation_progress_sample_interval": 1,
        "perturbation_random_strength": 0.02,
        "device": "auto",
        "save_weights": False,
    }
    row["service_id"] = _service_id(
        f"gen_{case['task_label']}",
        {
            "model": row["model_type"],
            "dataset": row["dataset_name"],
            "training_regime": "generic",
            "resource_tier": resource_tier.name,
            "dataset_variant": dataset_variant,
            "split_variant": split_variant,
            "knob_variant": knob_variant,
            "seed": seed,
        },
    )
    row["service_config"] = _service_config(row)
    return row


def build_hf_manifest(
    json_path: str | None = None,
    task_keys: list[str] | None = None,
    models_per_task: int = 10,
    datasets_per_model: int = 1,
    training_regimes: list[str] | None = None,
    dataset_variants_per_pair: int = 1,
    split_variants_per_pair: int = 1,
    knob_variants_per_pair: int = 1,
    total_services: int | None = None,
    seed: int | None = None,
    manifest_profile: str = "balanced",
    resource_tier: str | None = None,
    avg_sample_size: int | None = None,
    max_models_per_family: int | None = None,
    strict_inference_dataset_match: bool = True,
) -> pd.DataFrame:
    del json_path
    seed = _resolve_manifest_seed(seed)
    rng = random.Random(seed)
    profile = _resolve_manifest_profile(manifest_profile)
    resolved_resource_tier, resource_tier_explicit = _resolve_resource_tier(resource_tier, profile)
    smoketest = _is_smoketest_tier(resolved_resource_tier)
    target_sample_size = _target_sample_size(
        profile=profile,
        resource_tier=resolved_resource_tier,
        resource_tier_explicit=resource_tier_explicit,
        avg_sample_size=avg_sample_size,
    )
    requested_task_keys = task_keys or (list(TASK_SPECS) if smoketest else list(TASK_SPECS) + sorted(GENERIC_MANIFEST_TASK_KEYS))
    selected_training_regimes = training_regimes or ["finetune_transfer"]
    selected_training_regimes = [str(item).strip().lower() for item in selected_training_regimes if str(item).strip()]

    rows: list[dict[str, Any]] = []
    for task_key in requested_task_keys:
        task_spec = TASK_SPECS.get(task_key)
        if task_spec is None:
            continue
        models = _model_candidates(task_key)
        models = _sort_models_for_tier(
            models,
            task_key=task_key,
            resource_tier=resolved_resource_tier,
            selected_training_regimes=selected_training_regimes,
            rng=rng,
        )
        if max_models_per_family:
            models = _cap_models_per_family(models, int(max_models_per_family))
        model_limit = len(models) if smoketest else _normalise_positive_int(models_per_task)
        for model in models[:model_limit]:
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
                datasets = _sort_datasets_for_tier(
                    datasets,
                    task_key=task_key,
                    resource_tier=resolved_resource_tier,
                    rng=rng,
                )
                dataset_limit = len(datasets) if smoketest else _normalise_positive_int(datasets_per_model)
                for dataset in datasets[:dataset_limit]:
                    for dataset_variant in range(_normalise_positive_int(dataset_variants_per_pair)):
                        for split_variant in range(_normalise_positive_int(split_variants_per_pair)):
                            effective_knob_variants = (
                                1 if training_regime == "inference_only"
                                else _normalise_positive_int(knob_variants_per_pair)
                            )
                            for knob_variant in range(effective_knob_variants):
                                row = _row_from_registry(
                                    task_key=task_key,
                                    task_spec=task_spec,
                                    model=model,
                                    dataset_spec=dataset,
                                    resource_tier=resolved_resource_tier,
                                    target_sample_size=target_sample_size,
                                    training_regime=training_regime,
                                    dataset_variant=dataset_variant,
                                    split_variant=split_variant,
                                    knob_variant=knob_variant,
                                    seed=seed,
                                )
                                if _row_is_manifest_eligible(row):
                                    rows.append(row)

    for case in _selected_generic_cases(requested_task_keys):
        for dataset_variant in range(_normalise_positive_int(dataset_variants_per_pair)):
            for split_variant in range(_normalise_positive_int(split_variants_per_pair)):
                for knob_variant in range(_normalise_positive_int(knob_variants_per_pair)):
                    rows.append(
                        _row_from_generic_case(
                            case=case,
                            profile=profile,
                            resource_tier=resolved_resource_tier,
                            target_sample_size=target_sample_size,
                            dataset_variant=dataset_variant,
                            split_variant=split_variant,
                            knob_variant=knob_variant,
                            seed=seed,
                        )
                    )

    df = pd.DataFrame(rows)
    if total_services is not None and total_services >= 0:
        df = df.head(int(total_services))
    return df.reindex(columns=MANIFEST_COLUMNS)


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


def save_manifest(df: pd.DataFrame, output_path: Path, sheet_name: str = "services") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--task-keys")
    parser.add_argument("--models-per-task", type=int, default=10)
    parser.add_argument("--max-models-per-family", type=int)
    parser.add_argument("--datasets-per-model", type=int, default=1)
    parser.add_argument("--training-regimes")
    parser.add_argument("--dataset-variants-per-pair", type=int, default=1)
    parser.add_argument("--split-variants-per-pair", type=int, default=1)
    parser.add_argument("--knob-variants-per-pair", type=int, default=1)
    parser.add_argument("--total-services", type=int)
    parser.add_argument("--manifest-profile", choices=sorted(MANIFEST_PROFILES), default="balanced")
    parser.add_argument("--resource-tier", choices=sorted(RESOURCE_TIERS), help="Workload budget: smoketest, light, medium, heavy, or stress_test. Defaults from --manifest-profile.")
    parser.add_argument("--avg-sample-size", type=int)
    parser.add_argument("--seed", type=int, help="Optional seed. Omit for a fresh randomized manifest on each run.")
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
        manifest_profile=args.manifest_profile,
        resource_tier=args.resource_tier,
        avg_sample_size=args.avg_sample_size,
        max_models_per_family=args.max_models_per_family,
    )
    save_manifest(df, Path(args.output), sheet_name=args.sheet)
    print(f"Wrote {len(df)} service rows to {args.output}")


if __name__ == "__main__":
    main()
