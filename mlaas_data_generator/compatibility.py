from __future__ import annotations

from typing import Any, Mapping


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


_BLOCKED_IMAGE_CLASSIFICATION_RUNTIME_PAIRS = {
    ("google/efficientnet-b0", "zalando-datasets/fashion_mnist", "fashion_mnist"),
    ("google/efficientnet-b0", "ylecun/mnist", "mnist"),
    ("facebook/regnet-y-040", "timm/oxford-iiit-pet", ""),
    ("facebook/regnet-y-040", "ufldl-stanford/svhn", "cropped_digits"),
    ("google/mobilenet_v2_1.0_224", "timm/oxford-iiit-pet", ""),
    ("google/mobilenet_v2_1.0_224", "microsoft/cats_vs_dogs", ""),
    ("google/mobilenet_v2_1.0_224", "zalando-datasets/fashion_mnist", ""),
    ("google/mobilenet_v2_1.0_224", "zalando-datasets/fashion_mnist", "default"),
    ("google/mobilenet_v2_1.0_224", "ufldl-stanford/svhn", "cropped_digits"),
    ("microsoft/resnet-50", "cifar10", ""),
    ("microsoft/resnet-50", "microsoft/cats_vs_dogs", ""),
}


def known_bad_path_reason(
    *,
    task_key: str | None = None,
    hf_task: str | None = None,
    task_tag: str | None = None,
    model_id: Any = None,
    model_family: Any = None,
    dataset_name: Any = None,
    dataset_config: Any = None,
) -> str | None:
    task = _norm(task_key)
    hf = _norm(hf_task).replace("-", "_")
    tag = _norm(task_tag).replace("-", "_")
    model = _norm(model_id)
    family = _norm(model_family)
    dataset = _norm(dataset_name)
    dataset_cfg = _norm(dataset_config)

    if model == "squeezebert/squeezebert-uncased":
        return (
            "blocked known-bad path: squeezebert/squeezebert-uncased repeatedly failed model loading "
            "in this project"
        )

    if model == "salesforce/codet5-small" and task == "text2text_generation":
        return (
            "blocked known-bad path: Salesforce/codet5-small repeatedly collapsed to rouge1=0 "
            "for text2text_generation runs in this project"
        )

    if task == "text2text_generation" and tag == "summarization" and family == "codet5":
        return (
            "blocked semantic pairing: codet5-family models are disabled for summarization-style "
            "text2text generation in this project"
        )

    if task == "image_segmentation" and dataset == "buddhi19/syntheticgenv5":
        return (
            "blocked known-bad path: buddhi19/SyntheticGenV5 segmentation rows are disabled until "
            "the image/mask schema is explicitly verified"
        )

    if task == "image_segmentation" and dataset == "zhoubolei/scene_parse_150":
        return (
            "blocked known-bad runtime path: zhoubolei/scene_parse_150 requires a dataset loading "
            "path unsupported by the current datasets runtime"
        )

    if task == "image_classification" and (model, dataset, dataset_cfg) in _BLOCKED_IMAGE_CLASSIFICATION_RUNTIME_PAIRS:
        return (
            "blocked known-bad runtime path: this image-classification model/dataset pair repeatedly "
            "hit miopenStatusUnknownError in this project"
        )

    if hf in {"seq2seq_generation", "text2text_generation"} and tag == "summarization" and model == "salesforce/codet5-small":
        return (
            "blocked semantic pairing: Salesforce/codet5-small is disabled for summarization-style "
            "seq2seq runs in this project"
        )

    return None


def known_bad_manifest_combo_reason(
    *,
    task_key: str,
    model: Mapping[str, Any],
    dataset_spec: Mapping[str, Any],
) -> str | None:
    return known_bad_path_reason(
        task_key=task_key,
        hf_task=model.get("hf_task") or model.get("pipeline_tag"),
        task_tag=dataset_spec.get("task_tag"),
        model_id=model.get("hf_model_id"),
        model_family=model.get("family"),
        dataset_name=dataset_spec.get("dataset_name"),
        dataset_config=dataset_spec.get("dataset_config"),
    )


def known_bad_row_reason(resolved: Mapping[str, Any]) -> str | None:
    task_key = resolved.get("task")
    if not task_key:
        hf_task = _norm(resolved.get("hf_task")).replace("-", "_")
        task_type = _norm(resolved.get("task_type"))
        if hf_task == "fill_mask":
            task_key = "fill_mask"
        elif hf_task == "seq2seq_generation":
            task_key = "text2text_generation"
        elif hf_task == "causal_lm_generation":
            task_key = "text_generation"
        elif hf_task == "image_segmentation":
            task_key = "image_segmentation"
        elif hf_task == "image_classification":
            task_key = "image_classification"
        elif hf_task == "image_detection":
            task_key = "object_detection"
        elif hf_task == "visual_question_answering":
            task_key = "visual_question_answering"
        elif task_type:
            task_key = task_type

    return known_bad_path_reason(
        task_key=str(task_key or ""),
        hf_task=resolved.get("hf_task"),
        task_tag=resolved.get("task_tag"),
        model_id=resolved.get("hf_model_id") or resolved.get("model_id"),
        model_family=resolved.get("model_family"),
        dataset_name=resolved.get("dataset_name"),
        dataset_config=resolved.get("dataset_config"),
    )
