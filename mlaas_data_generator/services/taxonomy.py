from __future__ import annotations

import math

import numpy as np

from ..hf_tasks import normalize_hf_task as shared_normalize_hf_task


def normalize_hf_task(hf_task: str | None) -> str:
    return shared_normalize_hf_task(hf_task, default="unknown", unknown="unknown")


def canonical_task_family(task_type: str | None, hf_task: str | None = None) -> str:
    base = (task_type or "").strip().lower()
    hf = normalize_hf_task(hf_task)

    if base in {"image_classification"}:
        return "classification"
    if base in {"object_detection", "image_detection", "detection"}:
        return "detection"
    if base in {"image_segmentation", "semantic_segmentation", "segmentation"}:
        return "segmentation"
    if base in {"generation", "text_generation", "text2text_generation", "image_captioning"}:
        return "generation"
    if base in {"retrieval", "text_image_retrieval", "image_text_retrieval"}:
        return "retrieval"
    if base in {"vqa", "visual_question_answering", "visual_qa"}:
        return "vqa"
    if base in {"classification", "regression", "clustering"}:
        if base == "classification":
            if hf == "sentence_similarity":
                return "regression"
            if hf == "token_classification":
                return "token_classification"
            if hf == "fill_mask":
                return "fill_mask"
            if hf in {"image_detection"}:
                return "detection"
            if hf in {"image_segmentation"}:
                return "segmentation"
            if hf in {"causal_lm_generation", "seq2seq_generation", "image_captioning"}:
                return "generation"
            if hf == "text_image_retrieval":
                return "retrieval"
            if hf == "visual_question_answering":
                return "vqa"
        return base
    return "unknown"


def canonical_label_format(task_family: str) -> str:
    return {
        "classification": "single_label",
        "token_classification": "token_labels",
        "fill_mask": "token_labels",
        "regression": "continuous",
        "generation": "token_labels",
        "clustering": "cluster_id",
        "detection": "bbox_coco",
        "segmentation": "mask",
        "retrieval": "paired_rank",
        "vqa": "answer_text",
    }.get(task_family, "unknown")


def canonical_metric_names(task_family: str, metric_key: str | None = None, *, hf_task: str | None = None, task_tag: str | None = None) -> tuple[str, str | None]:
    hf = normalize_hf_task(hf_task)
    if hf == "sentence_similarity" and task_family in {"classification", "regression"}:
        return ("pearson", "spearman")
    if task_family == "classification":
        return ("accuracy", "f1")
    if task_family == "token_classification":
        return ("f1", "accuracy")
    if task_family == "fill_mask":
        return ("masked_accuracy", "perplexity_proxy")
    if task_family == "regression":
        return ("rmse", "mae")
    if task_family == "generation":
        if hf == "causal_lm_generation":
            return ("loss", "perplexity")
        tag = (task_tag or "").strip().lower().replace("-", "_")
        if tag == "summarization":
            return ("rouge1", "rouge2")
        if tag == "translation":
            return ("sacrebleu", None)
        if tag in {"captioning", "image_captioning"}:
            return ("cider", "bleu")
        return ("loss", "perplexity")
    if task_family == "clustering":
        return ("silhouette", "inertia")
    if task_family == "detection":
        return ("map", "map@0.5")
    if task_family == "segmentation":
        return ("iou", "dice")
    if task_family == "retrieval":
        return ("r@1", "r@5")
    if task_family == "vqa":
        return ("exact_match", None)
    return ((metric_key or "metric").lower(), None)


def metric_score_value(task_family: str, metric_name: str | None, metric_value: float | None) -> float:
    try:
        value = float(metric_value)
    except Exception:
        return math.nan
    if not np.isfinite(value):
        return math.nan
    metric = (metric_name or "").lower()
    if metric == "pearson":
        return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))
    if task_family == "regression" or metric in {"loss", "rmse", "mae", "perplexity", "cross_entropy_loss"}:
        return float(1.0 / (1.0 + max(0.0, value)))
    return float(value)


def metric_domain(metric_name: str) -> str:
    name = str(metric_name).lower()
    if "latency" in name:
        return "latency"
    if any(part in name for part in ("runtime", "duration", "time_s", "throughput")):
        return "runtime"
    if any(part in name for part in ("memory", "vram", "ram", "cpu", "gpu", "tokens", "params", "model_size")):
        return "resource"
    if "cost" in name or "efficiency" in name:
        return "cost"
    if "explain" in name:
        return "explainability"
    if any(part in name for part in ("reliability", "trust", "failure", "error", "retry", "oom", "nan")):
        return "reliability"
    if any(part in name for part in ("accuracy", "f1", "loss", "rmse", "mae", "perplexity", "rouge", "bleu", "cider", "map", "iou", "dice", "silhouette", "metric_score", "exact_match", "pearson", "spearman")):
        return "quality"
    return "metadata"
