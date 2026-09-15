"""Task/model training limits, independent of manifest population size or profile."""
from __future__ import annotations

from typing import Any, Mapping


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None



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



def _unique_ints(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        value = max(1, int(value))
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)



def _batch_sizes_for(task_key: str, model: dict[str, Any]) -> tuple[int, ...]:
    family = str(model.get("family") or "").strip().lower()
    if task_key == "text_image_retrieval":
        from ..contrastive_batch import retrieval_batch_target
        model_id = str(model.get("hf_model_id") or "").lower()
        image_size = 384 if "384" in model_id else 336 if "336" in model_id else 224
        target = retrieval_batch_target(_estimated_model_params_m(model) or 427.0, image_size)
        return (target,)
    if task_key == "image_captioning" or (task_key == "visual_question_answering" and family in {"blip", "blip2", "git"}):
        return (1,)
    if task_key in {"object_detection", "image_segmentation", "image_captioning", "visual_question_answering", "text_image_retrieval"}:
        values = (2, 4)
    elif task_key in {"text_generation", "text2text_generation", "token_classification"}:
        values = (4, 8)
    elif task_key == "image_classification":
        values = (8, 16)
    else:
        values = (8, 16)

    params_m = _estimated_model_params_m(model)
    model_id = str(model.get("hf_model_id") or "").strip().lower()
    if task_key == "object_detection" and (family in {"detr", "conditional-detr", "deformable-detr"} or "detr" in model_id):
        values = tuple(min(2, value) for value in values)
    if params_m is not None and params_m > 700:
        values = tuple(max(1, value // 4) for value in values)
    elif params_m is not None and params_m > 250:
        values = tuple(max(1, value // 2) for value in values)
    return _unique_ints(values)



def _learning_rates_for(task_key: str, model: dict[str, Any]) -> tuple[float, ...]:
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
    return rates



def _epochs_for(task_key: str) -> tuple[int, ...]:
    # Calibration may extend this exploration grid for supported recipes.
    return (1, 2)


def _weight_decays_for(task_key: str) -> tuple[float, ...]:
    if task_key in {"text_generation", "text2text_generation", "image_captioning", "visual_question_answering"}:
        return (0.0, 0.001, 0.01)
    return (0.0, 0.001, 0.01, 0.05)



def _optimizers_for(task_key: str) -> tuple[str, ...]:
    if task_key in {"object_detection", "image_segmentation", "image_captioning", "visual_question_answering", "text_image_retrieval"}:
        return ("adamw", "adam")
    return ("adamw", "adam", "sgd")



def _warmup_ratios_for(task_key: str) -> tuple[float, ...]:
    if task_key in {"object_detection", "image_segmentation"}:
        return (0.0, 0.03)
    return (0.0, 0.03, 0.06, 0.1)



def _mlm_probabilities_for(task_key: str) -> tuple[float, ...]:
    if task_key == "fill_mask":
        return (0.10, 0.15, 0.20, 0.30)
    return ()



def _gradient_accumulation_steps_for(task_key: str, model: dict[str, Any]) -> tuple[int, ...]:
    params_m = _estimated_model_params_m(model) or 0.0
    if task_key in {"object_detection", "image_segmentation", "image_captioning", "visual_question_answering", "text_image_retrieval"}:
        values = (1, 2, 4)
    elif task_key in {"text_generation", "text2text_generation", "token_classification"}:
        values = (1, 2, 4)
    else:
        values = (1, 2)
    if params_m > 250:
        values = tuple(value for value in values if value >= 2) or (2,)
    return _unique_ints(values)



def task_key(config: Mapping[str, Any]) -> str:
    task = str(config.get("hf_task") or "").replace("-", "_")
    if task == "sequence_classification":
        return "text_classification"
    if task:
        return task
    if config.get("model_type") == "cnn":
        return "keras_image_classification"
    return "tabular_regression" if config.get("task_type") == "regression" else "classification"

