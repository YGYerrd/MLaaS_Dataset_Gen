from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HF_TASK = "sequence_classification"
UNKNOWN_HF_TASK = "unknown"

CANONICAL_HF_TASKS = frozenset({
    "sequence_classification",
    "token_classification",
    "sentence_similarity",
    "fill_mask",
    "causal_lm_generation",
    "seq2seq_generation",
    "image_classification",
    "image_detection",
    "image_segmentation",
    "image_captioning",
    "text_image_retrieval",
    "visual_question_answering",
    "multimodal",
})

HF_TASK_ALIASES = {
    "text_classification": "sequence_classification",
    "seq_cls": "sequence_classification",
    "sequence_cls": "sequence_classification",
    "token_cls": "token_classification",
    "ner": "token_classification",
    "masked_lm": "fill_mask",
    "mlm": "fill_mask",
    "text_generation": "causal_lm_generation",
    "causal_lm": "causal_lm_generation",
    "text2text": "seq2seq_generation",
    "text2text_generation": "seq2seq_generation",
    "vision_classification": "image_classification",
    "image_cls": "image_classification",
    "object_detection": "image_detection",
    "detection": "image_detection",
    "semantic_segmentation": "image_segmentation",
    "segmentation": "image_segmentation",
    "image_to_text": "image_captioning",
    "image_caption": "image_captioning",
    "captioning": "image_captioning",
    "image_text_retrieval": "text_image_retrieval",
    "retrieval": "text_image_retrieval",
    "vqa": "visual_question_answering",
    "visual_qa": "visual_question_answering",
}

HF_TASK_MODALITY = {
    "sequence_classification": "text",
    "token_classification": "text",
    "sentence_similarity": "text",
    "fill_mask": "text",
    "causal_lm_generation": "text",
    "seq2seq_generation": "text",
    "image_classification": "image",
    "image_detection": "image",
    "image_segmentation": "image",
    "image_captioning": "multimodal",
    "text_image_retrieval": "multimodal",
    "visual_question_answering": "multimodal",
    "multimodal": "multimodal",
}

IMAGE_TASK_TYPES = {
    "image_classification": "classification",
    "image_detection": "detection",
    "image_segmentation": "segmentation",
}

MULTIMODAL_TASK_TAGS = {
    "image_captioning": "captioning",
    "text_image_retrieval": "retrieval",
    "visual_question_answering": "vqa",
}

PIPELINE_TAG_TO_HF_TASK = {
    "text-classification": "sequence_classification",
    "token-classification": "token_classification",
    "sentence-similarity": "sentence_similarity",
    "fill-mask": "fill_mask",
    "text-generation": "causal_lm_generation",
    "text2text-generation": "seq2seq_generation",
    "image-classification": "image_classification",
    "object-detection": "image_detection",
    "image-segmentation": "image_segmentation",
    "image-to-text": "image_captioning",
    "zero-shot-image-classification": "text_image_retrieval",
    "visual-question-answering": "visual_question_answering",
}

MODEL_TEMPLATE_TO_HF_TASK = {
    "hf_sequence_classification": "sequence_classification",
    "hf_token_classification": "token_classification",
    "hf_sentence_similarity": "sentence_similarity",
    "hf_fill_mask": "fill_mask",
    "hf_causal_lm": "causal_lm_generation",
    "hf_seq2seq": "seq2seq_generation",
    "auto_image_to_text": "image_captioning",
    "auto_clip_retrieval": "text_image_retrieval",
    "auto_vqa": "visual_question_answering",
    "auto_sequence_classification": "sequence_classification",
    "auto_token_classification": "token_classification",
    "auto_masked_lm": "fill_mask",
    "auto_causal_lm": "causal_lm_generation",
    "auto_seq2seq_lm": "seq2seq_generation",
}

DATASET_TEMPLATE_TO_HF_TASK = {
    "hf_text_classification": "sequence_classification",
    "hf_token_classification": "token_classification",
    "hf_sentence_pair_classification": "sentence_similarity",
    "hf_masked_lm": "fill_mask",
    "hf_causal_lm": "causal_lm_generation",
    "hf_seq2seq": "seq2seq_generation",
    "hf_text_sequence": "sequence_classification",
    "hf_text_token": "token_classification",
    "hf_text_similarity": "sentence_similarity",
    "hf_text_fill_mask": "fill_mask",
    "hf_text_generation": None,
    "hf_image_captioning": "image_captioning",
    "hf_image_text_retrieval": "text_image_retrieval",
    "hf_visual_question_answering": "visual_question_answering",
}

@dataclass(frozen=True)
class HfTaskSpec:
    hf_task: str
    modality: str
    task_type: str | None = None
    task_tag: str | None = None


def normalize_hf_task(hf_task: str | None, *, default: str = DEFAULT_HF_TASK, unknown: str | None = None) -> str:
    task = str(hf_task or "").strip().lower().replace("-", "_")
    if not task:
        return default
    normalized = HF_TASK_ALIASES.get(task, task)
    if normalized in CANONICAL_HF_TASKS:
        return normalized
    return unknown if unknown is not None else normalized


def hf_task_modality(hf_task: str | None, *, default: str | None = None) -> str | None:
    return HF_TASK_MODALITY.get(normalize_hf_task(hf_task), default)


def hf_task_type(hf_task: str | None, *, default: str | None = None) -> str | None:
    return IMAGE_TASK_TYPES.get(normalize_hf_task(hf_task), default)


def hf_task_tag(hf_task: str | None, *, default: str | None = None) -> str | None:
    return MULTIMODAL_TASK_TAGS.get(normalize_hf_task(hf_task), default)


def resolve_hf_task_spec(hf_task: str | None, *, default: str = DEFAULT_HF_TASK, unknown: str | None = None) -> HfTaskSpec:
    task = normalize_hf_task(hf_task, default=default, unknown=unknown)
    return HfTaskSpec(
        hf_task=task,
        modality=HF_TASK_MODALITY.get(task, "text" if task != unknown else "unknown"),
        task_type=IMAGE_TASK_TYPES.get(task),
        task_tag=MULTIMODAL_TASK_TAGS.get(task),
    )


def normalize_loader_template(loader_template: str | None) -> str | None:
    if not loader_template:
        return None
    return str(loader_template).strip().lower()


def resolve_model_hf_task(loader_template: str | None = None, hf_task: str | None = None) -> str:
    template = normalize_loader_template(loader_template)
    mapped = MODEL_TEMPLATE_TO_HF_TASK.get(template)
    if mapped:
        return mapped
    return normalize_hf_task(hf_task)


def resolve_dataset_hf_task(loader_template: str | None = None, hf_task: str | None = None, *, task_tag: str | None = None, pipeline_tag: str | None = None) -> str:
    template = normalize_loader_template(loader_template)
    if template == "hf_text_generation":
        normalized_task_tag = str(task_tag or "").strip().lower().replace("-", "_")
        normalized_pipeline_tag = str(pipeline_tag or "").strip().lower()
        if normalized_task_tag in {"summarization", "translation", "seq2seq", "text2text"} or normalized_pipeline_tag == "text2text-generation":
            return "seq2seq_generation"
        return "causal_lm_generation"
    mapped = DATASET_TEMPLATE_TO_HF_TASK.get(template)
    if mapped:
        return mapped
    return normalize_hf_task(hf_task)
