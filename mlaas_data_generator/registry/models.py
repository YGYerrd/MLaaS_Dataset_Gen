from __future__ import annotations

from copy import deepcopy

ModelSpec = dict[str, object]
RegistryEntry = tuple[str, ModelSpec]


FINETUNE_AND_INFERENCE = ["finetune_transfer", "inference_only"]
FINETUNE_ONLY = ["finetune_transfer"]
INFERENCE_ONLY = ["inference_only"]

TEXT_CLASSIFICATION_DATASETS = ["glue_sst2", "ag_news", "imdb"]
TEXT_CLASSIFICATION_EXTENDED_DATASETS = [
    *TEXT_CLASSIFICATION_DATASETS,
    "rotten_tomatoes",
    "emotion",
    "glue_cola",
    "glue_qnli",
    "glue_qqp",
    "glue_rte",
    "glue_mnli",
]
TOKEN_CLASSIFICATION_DATASETS = [
    "conll2003",
    "wnut_17",
    "wikiann_en",
    "conll2002_es",
    "conll2002_nl",
    "conllpp",
    "ncbi_disease",
    "wikiann_de",
    "wikiann_fr",
    "wikiann_es",
]
SENTENCE_SIMILARITY_DATASETS = [
    "glue_stsb",
    "glue_mrpc",
    "glue_qqp_similarity",
    "glue_rte_similarity",
    "glue_qnli_similarity",
    "glue_mnli_similarity",
    "glue_wnli_similarity",
    "paws_labeled_final",
    "pawsx_en",
    "stsb_sentence_transformers",
]
FILL_MASK_DATASETS = [
    "wikitext2",
    "ag_news_fillmask",
    "imdb_fillmask",
    "rotten_tomatoes_fillmask",
    "emotion_fillmask",
    "cnn_dailymail_fillmask",
    "xsum_fillmask",
    "tinystories_fillmask",
    "billsum_fillmask",
    "samsum_fillmask",
]
TEXT_GENERATION_DATASETS = [
    "wikitext2_lm",
    "ag_news_lm",
    "imdb_lm",
    "rotten_tomatoes_lm",
    "emotion_lm",
    "cnn_dailymail_lm",
    "xsum_lm",
    "tinystories_lm",
    "billsum_lm",
    "arxiv_summarization_lm",
]
TEXT2TEXT_GENERATION_DATASETS = [
    "cnn_dailymail",
    "xsum",
    "billsum",
    "samsum",
    "arxiv_summarization",
    "pubmed_summarization",
    "govreport_summarization",
    "scitldr_aic",
    "scitldr_abstract",
    "dolly_15k",
]
IMAGE_CLASSIFICATION_DATASETS = [
    "beans",
    "cifar10",
    "food101",
    "mnist",
    "fashion_mnist",
    "cifar100",
    "svhn_cropped_digits",
    "cats_vs_dogs",
    "oxford_iiit_pet",
    "tiny_imagenet",
]
OBJECT_DETECTION_DATASETS = [
    "coco_detection",
    "cppe5_detection",
    "fashionpedia_4cat_detection",
    "license_plate_detection",
    "hard_hat_detection",
    "forklift_detection",
    "german_traffic_sign_detection",
    "blood_cell_detection",
    "table_extraction_detection",
    "plane_detection",
]
IMAGE_SEGMENTATION_DATASETS = [
    "scene_parse_150",
    "ade20k_mini_segmentation",
    "ade20k_tiny_segmentation",
    "hot_building_segmentation",
    "foodseg103_segmentation",
    "human_parsing_segmentation",
    "crater_binary_segmentation",
    "conequest_segmentation",
    "apple_dms_materials_segmentation",
    "syntheticgenv5_segmentation",
]
IMAGE_CAPTIONING_DATASETS = ["flickr8k_captioning"]
RETRIEVAL_DATASETS = ["flickr8k_retrieval"]
VQA_DATASETS = ["vqav2"]

TEXT_CLASSIFICATION_EXPLAINABILITY = {
    "supported": True,
    "preferred_methods": ["integrated_gradients", "attention_rollout"],
    "target_type": "class_label",
    "input_type": "single_text",
    "attribution_level": "token",
    "requires_gradients": True,
    "requires_attention": True,
    "supports_gradients": True,
    "supports_attention_rollout": True,
    "supports_token_attribution": True,
}

TOKEN_CLASSIFICATION_EXPLAINABILITY = {
    "supported": True,
    "preferred_methods": ["integrated_gradients", "attention_rollout"],
    "target_type": "token_label",
    "input_type": "token_sequence",
    "attribution_level": "token",
    "requires_gradients": True,
    "requires_attention": True,
    "supports_gradients": True,
    "supports_attention_rollout": True,
    "supports_token_attribution": True,
}

SENTENCE_SIMILARITY_EXPLAINABILITY = {
    "supported": True,
    "preferred_methods": ["integrated_gradients", "attention_rollout"],
    "target_type": "similarity_score",
    "input_type": "text_pair",
    "attribution_level": "token",
    "requires_gradients": True,
    "requires_attention": True,
    "supports_gradients": True,
    "supports_attention_rollout": True,
    "supports_token_attribution": True,
}

MINILM_SIMILARITY_EXPLAINABILITY = {
    "supported": True,
    "preferred_methods": ["integrated_gradients", "token_saliency"],
    "target_type": "similarity_score",
    "input_type": "text_pair",
    "attribution_level": "token",
    "requires_gradients": True,
    "requires_attention": False,
    "supports_gradients": True,
    "supports_attention_rollout": False,
    "supports_token_attribution": True,
}

FILL_MASK_EXPLAINABILITY = {
    "supported": True,
    "preferred_methods": ["integrated_gradients", "attention_rollout"],
    "target_type": "masked_token",
    "input_type": "single_text",
    "attribution_level": "token",
    "requires_gradients": True,
    "requires_attention": True,
    "supports_gradients": True,
    "supports_attention_rollout": True,
    "supports_token_attribution": True,
}

TEXT_GENERATION_EXPLAINABILITY = {
    "supported": True,
    "preferred_methods": ["integrated_gradients", "token_saliency"],
    "target_type": "generated_token",
    "input_type": "single_text",
    "attribution_level": "token",
    "requires_gradients": True,
    "requires_attention": False,
    "supports_gradients": True,
    "supports_attention_rollout": False,
    "supports_token_attribution": True,
}

TEXT2TEXT_GENERATION_EXPLAINABILITY = {
    "supported": True,
    "preferred_methods": ["integrated_gradients", "attention_rollout"],
    "target_type": "summary_token",
    "input_type": "single_text",
    "attribution_level": "token",
    "requires_gradients": True,
    "requires_attention": True,
    "supports_gradients": True,
    "supports_attention_rollout": True,
    "supports_token_attribution": True,
}

IMAGE_ATTN_EXPLAINABILITY = {
    "supports_gradients": True,
    "supports_attention_rollout": True,
    "supports_token_attribution": False,
}

IMAGE_NO_ATTN_EXPLAINABILITY = {
    "supports_gradients": True,
    "supports_attention_rollout": False,
    "supports_token_attribution": False,
}

CAPTIONING_EXPLAINABILITY = {
    "supports_gradients": True,
    "supports_attention_rollout": True,
    "supports_token_attribution": True,
}

RETRIEVAL_EXPLAINABILITY = {
    "supports_gradients": True,
    "supports_attention_rollout": False,
    "supports_token_attribution": True,
}

VQA_EXPLAINABILITY = {
    "supports_gradients": True,
    "supports_attention_rollout": True,
    "supports_token_attribution": True,
}

VILT_VQA_EXPLAINABILITY = {
    "supported": True,
    "preferred_methods": ["integrated_gradients", "attention_rollout"],
    "target_type": "summary_token",
    "input_type": "single_text",
    "attribution_level": "token",
    "requires_gradients": True,
    "requires_attention": True,
    "supports_gradients": True,
    "supports_attention_rollout": True,
    "supports_token_attribution": True,
}


def _copy_explainability(template: dict[str, object]) -> dict[str, object]:
    return deepcopy(template)


def _registry_entry(
    registry_id: str,
    *,
    hf_model_id: str,
    task_key: str,
    pipeline_tag: str,
    family: str,
    modality: str,
    dataset_keys: list[str],
    loader_template: str,
    explainability: dict[str, object],
    allowed_training_regimes: list[str],
    model_role: str = "task_head",
    **extra: object,
) -> RegistryEntry:
    spec: ModelSpec = {
        "hf_model_id": hf_model_id,
        "task_key": task_key,
        "pipeline_tag": pipeline_tag,
        "family": family,
        "modality": modality,
        "model_role": model_role,
        "allowed_training_regimes": list(allowed_training_regimes),
        "dataset_keys": list(dataset_keys),
        "loader_template": loader_template,
        "explainability": _copy_explainability(explainability),
    }
    spec.update(extra)
    return registry_id, spec


def _text_classification_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    dataset_keys: list[str],
    allowed_training_regimes: list[str],
) -> RegistryEntry:
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="text_classification",
        pipeline_tag="text-classification",
        family=family,
        modality="text",
        dataset_keys=dataset_keys,
        loader_template="hf_sequence_classification",
        explainability=TEXT_CLASSIFICATION_EXPLAINABILITY,
        allowed_training_regimes=allowed_training_regimes,
    )


def _token_classification_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    dataset_keys: list[str],
    allowed_training_regimes: list[str],
) -> RegistryEntry:
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="token_classification",
        pipeline_tag="token-classification",
        family=family,
        modality="text",
        dataset_keys=dataset_keys,
        loader_template="hf_token_classification",
        explainability=TOKEN_CLASSIFICATION_EXPLAINABILITY,
        allowed_training_regimes=allowed_training_regimes,
    )


def _sentence_similarity_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    allowed_training_regimes: list[str],
    explainability: dict[str, object] = SENTENCE_SIMILARITY_EXPLAINABILITY,
) -> RegistryEntry:
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="sentence_similarity",
        pipeline_tag="sentence-similarity",
        family=family,
        modality="text",
        dataset_keys=SENTENCE_SIMILARITY_DATASETS,
        loader_template="hf_sentence_similarity",
        explainability=explainability,
        allowed_training_regimes=allowed_training_regimes,
    )


def _fill_mask_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    allowed_training_regimes: list[str],
) -> RegistryEntry:
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="fill_mask",
        pipeline_tag="fill-mask",
        family=family,
        modality="text",
        dataset_keys=FILL_MASK_DATASETS,
        loader_template="hf_fill_mask",
        explainability=FILL_MASK_EXPLAINABILITY,
        allowed_training_regimes=allowed_training_regimes,
    )


def _text_generation_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    allowed_training_regimes: list[str],
    inference_dataset_keys: list[str] | None = None,
) -> RegistryEntry:
    extra: dict[str, object] = {}
    if inference_dataset_keys is not None:
        extra["inference_dataset_keys"] = list(inference_dataset_keys)
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="text_generation",
        pipeline_tag="text-generation",
        family=family,
        modality="text",
        dataset_keys=TEXT_GENERATION_DATASETS,
        loader_template="hf_causal_lm",
        explainability=TEXT_GENERATION_EXPLAINABILITY,
        allowed_training_regimes=allowed_training_regimes,
        **extra,
    )


def _text2text_generation_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    allowed_training_regimes: list[str],
    inference_dataset_keys: list[str] | None = None,
) -> RegistryEntry:
    extra: dict[str, object] = {}
    if inference_dataset_keys is not None:
        extra["inference_dataset_keys"] = list(inference_dataset_keys)
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="text2text_generation",
        pipeline_tag="text2text-generation",
        family=family,
        modality="text",
        dataset_keys=TEXT2TEXT_GENERATION_DATASETS,
        loader_template="hf_seq2seq",
        explainability=TEXT2TEXT_GENERATION_EXPLAINABILITY,
        allowed_training_regimes=allowed_training_regimes,
        **extra,
    )


def _image_classification_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    explainability: dict[str, object],
    inference_num_labels: int,
    allowed_training_regimes: list[str],
) -> RegistryEntry:
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="image_classification",
        pipeline_tag="image-classification",
        family=family,
        modality="image",
        dataset_keys=IMAGE_CLASSIFICATION_DATASETS,
        loader_template="auto_image_classification",
        explainability=explainability,
        allowed_training_regimes=allowed_training_regimes,
        inference_num_labels=inference_num_labels,
    )


def _object_detection_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    explainability: dict[str, object],
    allowed_training_regimes: list[str],
    inference_dataset_keys: list[str] | None = None,
) -> RegistryEntry:
    extra: dict[str, object] = {}
    if inference_dataset_keys is not None:
        extra["inference_dataset_keys"] = list(inference_dataset_keys)
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="object_detection",
        pipeline_tag="object-detection",
        family=family,
        modality="image",
        dataset_keys=OBJECT_DETECTION_DATASETS,
        loader_template="auto_object_detection",
        explainability=explainability,
        allowed_training_regimes=allowed_training_regimes,
        **extra,
    )


def _image_segmentation_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    allowed_training_regimes: list[str],
    inference_dataset_keys: list[str] | None = None,
) -> RegistryEntry:
    extra: dict[str, object] = {}
    if inference_dataset_keys is not None:
        extra["inference_dataset_keys"] = list(inference_dataset_keys)
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="image_segmentation",
        pipeline_tag="image-segmentation",
        family=family,
        modality="image",
        dataset_keys=IMAGE_SEGMENTATION_DATASETS,
        loader_template="auto_image_segmentation",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        allowed_training_regimes=allowed_training_regimes,
        **extra,
    )


def _image_captioning_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    allowed_training_regimes: list[str],
    inference_dataset_keys: list[str] | None = None,
    finetune_validated: bool = False,
) -> RegistryEntry:
    extra: dict[str, object] = {}
    if inference_dataset_keys is not None:
        extra["inference_dataset_keys"] = list(inference_dataset_keys)
    if finetune_validated:
        extra["finetune_validated"] = True
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="image_captioning",
        pipeline_tag="image-to-text",
        family=family,
        modality="multimodal",
        dataset_keys=IMAGE_CAPTIONING_DATASETS,
        loader_template="auto_image_to_text",
        explainability=CAPTIONING_EXPLAINABILITY,
        allowed_training_regimes=allowed_training_regimes,
        **extra,
    )


def _retrieval_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    allowed_training_regimes: list[str],
    finetune_validated: bool = False,
) -> RegistryEntry:
    extra: dict[str, object] = {}
    if finetune_validated:
        extra["finetune_validated"] = True
        extra["retrieval_positive_policy"] = "diagonal_in_batch"
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="text_image_retrieval",
        pipeline_tag="zero-shot-image-classification",
        family=family,
        modality="multimodal",
        dataset_keys=RETRIEVAL_DATASETS,
        loader_template="auto_clip_retrieval",
        explainability=RETRIEVAL_EXPLAINABILITY,
        allowed_training_regimes=allowed_training_regimes,
        model_role="dual_encoder",
        **extra,
    )


def _vqa_model(
    registry_id: str,
    hf_model_id: str,
    *,
    family: str,
    explainability: dict[str, object],
    allowed_training_regimes: list[str],
    inference_dataset_keys: list[str] | None = None,
    finetune_validated: bool = False,
    vqa_label_mode: str | None = None,
) -> RegistryEntry:
    extra: dict[str, object] = {}
    if inference_dataset_keys is not None:
        extra["inference_dataset_keys"] = list(inference_dataset_keys)
    if finetune_validated:
        extra["finetune_validated"] = True
    if vqa_label_mode is not None:
        extra["vqa_label_mode"] = str(vqa_label_mode)
    return _registry_entry(
        registry_id,
        hf_model_id=hf_model_id,
        task_key="visual_question_answering",
        pipeline_tag="visual-question-answering",
        family=family,
        modality="multimodal",
        dataset_keys=VQA_DATASETS,
        loader_template="auto_vqa",
        explainability=explainability,
        allowed_training_regimes=allowed_training_regimes,
        **extra,
    )


def _build_registry(*entries: RegistryEntry) -> dict[str, dict[str, object]]:
    registry: dict[str, dict[str, object]] = {}
    for registry_id, spec in entries:
        if registry_id in registry:
            raise ValueError(f"Duplicate model registry key: {registry_id}")
        registry[registry_id] = spec
    return registry


TEXT_CLASSIFICATION_MODELS: tuple[RegistryEntry, ...] = (
    _text_classification_model(
        "distilbert-base-uncased_textcls",
        "distilbert-base-uncased",
        family="bert",
        dataset_keys=TEXT_CLASSIFICATION_DATASETS,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_classification_model(
        "roberta-base_textcls",
        "roberta-base",
        family="roberta",
        dataset_keys=["ag_news", "imdb"],
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_classification_model(
        "bert-base-uncased_textcls",
        "bert-base-uncased",
        family="bert",
        dataset_keys=TEXT_CLASSIFICATION_DATASETS,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_classification_model(
        "albert-base-v2_textcls",
        "albert/albert-base-v2",
        family="albert",
        dataset_keys=TEXT_CLASSIFICATION_EXTENDED_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _text_classification_model(
        "electra-small-discriminator_textcls",
        "google/electra-small-discriminator",
        family="electra",
        dataset_keys=TEXT_CLASSIFICATION_EXTENDED_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _text_classification_model(
        "distilroberta-base_textcls",
        "distilroberta-base",
        family="distilroberta",
        dataset_keys=TEXT_CLASSIFICATION_EXTENDED_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _text_classification_model(
        "mobilebert-uncased_textcls",
        "google/mobilebert-uncased",
        family="mobilebert",
        dataset_keys=TEXT_CLASSIFICATION_EXTENDED_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _text_classification_model(
        "tinybert-general-4l_textcls",
        "huawei-noah/TinyBERT_General_4L_312D",
        family="tinybert",
        dataset_keys=TEXT_CLASSIFICATION_EXTENDED_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _text_classification_model(
        "deberta-base_textcls",
        "microsoft/deberta-base",
        family="deberta",
        dataset_keys=TEXT_CLASSIFICATION_EXTENDED_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _text_classification_model(
        "squeezebert-uncased_textcls",
        "squeezebert/squeezebert-uncased",
        family="squeezebert",
        dataset_keys=TEXT_CLASSIFICATION_EXTENDED_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
)

TOKEN_CLASSIFICATION_MODELS: tuple[RegistryEntry, ...] = (
    _token_classification_model(
        "bert-base-cased_tokencls",
        "bert-base-cased",
        family="bert",
        dataset_keys=["conll2003", "wnut_17"],
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _token_classification_model(
        "distilbert-base-cased_tokencls",
        "distilbert-base-cased",
        family="distilbert",
        dataset_keys=["conll2003"],
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _token_classification_model(
        "distilbert-base-uncased_tokencls",
        "distilbert-base-uncased",
        family="distilbert",
        dataset_keys=["wnut_17"],
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _token_classification_model(
        "dslim-bert-base-ner_tokencls",
        "dslim/bert-base-NER",
        family="bert",
        dataset_keys=TOKEN_CLASSIFICATION_DATASETS,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _token_classification_model(
        "roberta-base_tokencls",
        "roberta-base",
        family="roberta",
        dataset_keys=TOKEN_CLASSIFICATION_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _token_classification_model(
        "distilroberta-base_tokencls",
        "distilroberta-base",
        family="distilroberta",
        dataset_keys=TOKEN_CLASSIFICATION_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _token_classification_model(
        "electra-small-discriminator_tokencls",
        "google/electra-small-discriminator",
        family="electra",
        dataset_keys=TOKEN_CLASSIFICATION_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _token_classification_model(
        "electra-base-discriminator_tokencls",
        "google/electra-base-discriminator",
        family="electra",
        dataset_keys=TOKEN_CLASSIFICATION_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _token_classification_model(
        "mobilebert-uncased_tokencls",
        "google/mobilebert-uncased",
        family="mobilebert",
        dataset_keys=TOKEN_CLASSIFICATION_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _token_classification_model(
        "tinybert-general-4l_tokencls",
        "huawei-noah/TinyBERT_General_4L_312D",
        family="tinybert",
        dataset_keys=TOKEN_CLASSIFICATION_DATASETS,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
)

SENTENCE_SIMILARITY_MODELS: tuple[RegistryEntry, ...] = (
    _sentence_similarity_model(
        "distilbert-base-uncased_similarity",
        "distilbert-base-uncased",
        family="distilbert",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _sentence_similarity_model(
        "roberta-base_similarity",
        "roberta-base",
        family="roberta",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _sentence_similarity_model(
        "MiniLM_similarity",
        "microsoft/MiniLM-L12-H384-uncased",
        family="MiniLM",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        explainability=MINILM_SIMILARITY_EXPLAINABILITY,
    ),
    _sentence_similarity_model(
        "bert-base-uncased_similarity",
        "bert-base-uncased",
        family="bert",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _sentence_similarity_model(
        "distilroberta-base_similarity",
        "distilroberta-base",
        family="distilroberta",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _sentence_similarity_model(
        "electra-small-discriminator_similarity",
        "google/electra-small-discriminator",
        family="electra",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _sentence_similarity_model(
        "mobilebert-uncased_similarity",
        "google/mobilebert-uncased",
        family="mobilebert",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _sentence_similarity_model(
        "tinybert-general-4l_similarity",
        "huawei-noah/TinyBERT_General_4L_312D",
        family="tinybert",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _sentence_similarity_model(
        "deberta-base_similarity",
        "microsoft/deberta-base",
        family="deberta",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _sentence_similarity_model(
        "squeezebert-uncased_similarity",
        "squeezebert/squeezebert-uncased",
        family="squeezebert",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
)

FILL_MASK_MODELS: tuple[RegistryEntry, ...] = (
    _fill_mask_model(
        "distilroberta-base_fillmask",
        "distilroberta-base",
        family="roberta",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _fill_mask_model(
        "bert-base-uncased_fillmask",
        "bert-base-uncased",
        family="bert",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _fill_mask_model(
        "distilbert-base-uncased_fillmask",
        "distilbert-base-uncased",
        family="distilbert",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _fill_mask_model(
        "roberta-base_fillmask",
        "roberta-base",
        family="roberta",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _fill_mask_model(
        "bert-base-cased_fillmask",
        "bert-base-cased",
        family="bert",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _fill_mask_model(
        "electra-small-generator_fillmask",
        "google/electra-small-generator",
        family="electra",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _fill_mask_model(
        "electra-base-generator_fillmask",
        "google/electra-base-generator",
        family="electra",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _fill_mask_model(
        "deberta-base_fillmask",
        "microsoft/deberta-base",
        family="deberta",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _fill_mask_model(
        "mobilebert-uncased_fillmask",
        "google/mobilebert-uncased",
        family="mobilebert",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _fill_mask_model(
        "squeezebert-uncased_fillmask",
        "squeezebert/squeezebert-uncased",
        family="squeezebert",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
)

TEXT_GENERATION_MODELS: tuple[RegistryEntry, ...] = (
    _text_generation_model(
        "distilgpt2_textgen",
        "distilgpt2",
        family="gpt2",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_generation_model(
        "gpt2_textgen",
        "gpt2",
        family="gpt2",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_generation_model(
        "tiny-gpt2_textgen",
        "sshleifer/tiny-gpt2",
        family="gpt2",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_generation_model(
        "dialogpt-small_textgen",
        "microsoft/DialoGPT-small",
        family="dialogpt",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_generation_model(
        "dialogpt-medium_textgen",
        "microsoft/DialoGPT-medium",
        family="dialogpt",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_generation_model(
        "gpt-neo-125m_textgen",
        "EleutherAI/gpt-neo-125m",
        family="gpt-neo",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_generation_model(
        "pythia-70m_textgen",
        "EleutherAI/pythia-70m",
        family="pythia",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_generation_model(
        "opt-125m_textgen",
        "facebook/opt-125m",
        family="opt",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text_generation_model(
        "tinystories-1m_textgen",
        "roneneldan/TinyStories-1M",
        family="tinystories",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["tinystories_lm"],
    ),
    _text_generation_model(
        "tinystories-33m_textgen",
        "roneneldan/TinyStories-33M",
        family="tinystories",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["tinystories_lm"],
    ),
)

TEXT2TEXT_GENERATION_MODELS: tuple[RegistryEntry, ...] = (
    _text2text_generation_model(
        "flan-t5-small_text2text",
        "google/flan-t5-small",
        family="t5",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text2text_generation_model(
        "t5-small_text2text",
        "t5-small",
        family="t5",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text2text_generation_model(
        "flan-t5-base_text2text",
        "google/flan-t5-base",
        family="t5",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text2text_generation_model(
        "t5-base_text2text",
        "t5-base",
        family="t5",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text2text_generation_model(
        "distilbart-cnn-12-6_text2text",
        "sshleifer/distilbart-cnn-12-6",
        family="bart",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["cnn_dailymail"],
    ),
    _text2text_generation_model(
        "distilbart-xsum-12-6_text2text",
        "sshleifer/distilbart-xsum-12-6",
        family="bart",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["xsum"],
    ),
    _text2text_generation_model(
        "byt5-small_text2text",
        "google/byt5-small",
        family="byt5",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text2text_generation_model(
        "mt5-small_text2text",
        "google/mt5-small",
        family="mt5",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text2text_generation_model(
        "led-base-16384_text2text",
        "allenai/led-base-16384",
        family="led",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _text2text_generation_model(
        "codet5-small_text2text",
        "Salesforce/codet5-small",
        family="codet5",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
)

IMAGE_CLASSIFICATION_MODELS: tuple[RegistryEntry, ...] = (
    _image_classification_model(
        "vit-base-patch16-224_imgcls",
        "google/vit-base-patch16-224",
        family="vit",
        explainability=IMAGE_ATTN_EXPLAINABILITY,
        inference_num_labels=1000,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _image_classification_model(
        "mobilevit-small_imgcls",
        "apple/mobilevit-small",
        family="mobilevit",
        explainability=IMAGE_ATTN_EXPLAINABILITY,
        inference_num_labels=1000,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _image_classification_model(
        "resnet-50_imgcls",
        "microsoft/resnet-50",
        family="resnet",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        inference_num_labels=1000,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
    ),
    _image_classification_model(
        "deit-tiny-patch16-224_imgcls",
        "facebook/deit-tiny-patch16-224",
        family="deit",
        explainability=IMAGE_ATTN_EXPLAINABILITY,
        inference_num_labels=1000,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _image_classification_model(
        "convnext-tiny-224_imgcls",
        "facebook/convnext-tiny-224",
        family="convnext",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        inference_num_labels=1000,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _image_classification_model(
        "swin-tiny-patch4-window7-224_imgcls",
        "microsoft/swin-tiny-patch4-window7-224",
        family="swin",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        inference_num_labels=1000,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _image_classification_model(
        "efficientnet-b0_imgcls",
        "google/efficientnet-b0",
        family="efficientnet",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        inference_num_labels=1000,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _image_classification_model(
        "mobilenet-v2-1-0-224_imgcls",
        "google/mobilenet_v2_1.0_224",
        family="mobilenet",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        inference_num_labels=1000,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _image_classification_model(
        "regnet-y-040_imgcls",
        "facebook/regnet-y-040",
        family="regnet",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        inference_num_labels=1000,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _image_classification_model(
        "levit-128s_imgcls",
        "facebook/levit-128S",
        family="levit",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        inference_num_labels=1000,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
)

OBJECT_DETECTION_MODELS: tuple[RegistryEntry, ...] = (
    _object_detection_model(
        "detr-resnet-50_objdet",
        "facebook/detr-resnet-50",
        family="detr",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["coco_detection"],
    ),
    _object_detection_model(
        "yolos-small_objdet",
        "hustvl/yolos-small",
        family="yolos",
        explainability=IMAGE_ATTN_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["coco_detection"],
    ),
    _object_detection_model(
        "rtdetr-r18vd_objdet",
        "PekingU/rtdetr_r18vd_coco_o365",
        family="rtdetr",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        allowed_training_regimes=INFERENCE_ONLY,
        inference_dataset_keys=["coco_detection"],
    ),
    _object_detection_model(
        "yolos-tiny_objdet",
        "hustvl/yolos-tiny",
        family="yolos",
        explainability=IMAGE_ATTN_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _object_detection_model(
        "yolos-base_objdet",
        "hustvl/yolos-base",
        family="yolos",
        explainability=IMAGE_ATTN_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _object_detection_model(
        "detr-resnet-50-dc5_objdet",
        "facebook/detr-resnet-50-dc5",
        family="detr",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _object_detection_model(
        "conditional-detr-resnet-50_objdet",
        "microsoft/conditional-detr-resnet-50",
        family="conditional-detr",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _object_detection_model(
        "deformable-detr_objdet",
        "SenseTime/deformable-detr",
        family="deformable-detr",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _object_detection_model(
        "rtdetr-v2-r18vd_objdet",
        "PekingU/rtdetr_v2_r18vd",
        family="rtdetr-v2",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _object_detection_model(
        "table-transformer-detection_objdet",
        "microsoft/table-transformer-detection",
        family="table-transformer",
        explainability=IMAGE_NO_ATTN_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_ONLY,
    ),
)

IMAGE_SEGMENTATION_MODELS: tuple[RegistryEntry, ...] = (
    _image_segmentation_model(
        "segformer-b0_seg",
        "nvidia/segformer-b0-finetuned-ade-512-512",
        family="segformer",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["scene_parse_150", "ade20k_mini_segmentation", "ade20k_tiny_segmentation"],
    ),
    _image_segmentation_model(
        "segformer-b2_seg",
        "nvidia/segformer-b2-finetuned-ade-512-512",
        family="segformer",
        allowed_training_regimes=INFERENCE_ONLY,
        inference_dataset_keys=["scene_parse_150", "ade20k_mini_segmentation", "ade20k_tiny_segmentation"],
    ),
    _image_segmentation_model(
        "segformer-b1_seg",
        "nvidia/segformer-b1-finetuned-ade-512-512",
        family="segformer",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["scene_parse_150", "ade20k_mini_segmentation", "ade20k_tiny_segmentation"],
    ),
    _image_segmentation_model(
        "segformer-b3_seg",
        "nvidia/segformer-b3-finetuned-ade-512-512",
        family="segformer",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["scene_parse_150", "ade20k_mini_segmentation", "ade20k_tiny_segmentation"],
    ),
    _image_segmentation_model(
        "segformer-b4_seg",
        "nvidia/segformer-b4-finetuned-ade-512-512",
        family="segformer",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["scene_parse_150", "ade20k_mini_segmentation", "ade20k_tiny_segmentation"],
    ),
    _image_segmentation_model(
        "segformer-b5_seg",
        "nvidia/segformer-b5-finetuned-ade-640-640",
        family="segformer",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["scene_parse_150", "ade20k_mini_segmentation", "ade20k_tiny_segmentation"],
    ),
    _image_segmentation_model(
        "dpt-large-ade_seg",
        "Intel/dpt-large-ade",
        family="dpt",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["scene_parse_150", "ade20k_mini_segmentation", "ade20k_tiny_segmentation"],
    ),
    _image_segmentation_model(
        "upernet-convnext-tiny_seg",
        "openmmlab/upernet-convnext-tiny",
        family="upernet",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _image_segmentation_model(
        "upernet-swin-tiny_seg",
        "openmmlab/upernet-swin-tiny",
        family="upernet",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
    _image_segmentation_model(
        "segformer-b2-clothes_seg",
        "mattmdjaga/segformer_b2_clothes",
        family="segformer-clothes",
        allowed_training_regimes=FINETUNE_ONLY,
    ),
)

IMAGE_CAPTIONING_MODELS: tuple[RegistryEntry, ...] = (
    _image_captioning_model(
        "blip-caption-base",
        "Salesforce/blip-image-captioning-base",
        family="blip",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        finetune_validated=True,
    ),
    _image_captioning_model(
        "blip-caption-large",
        "Salesforce/blip-image-captioning-large",
        family="blip",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _image_captioning_model(
        "git-base-caption",
        "microsoft/git-base",
        family="git",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        finetune_validated=True,
    ),
    _image_captioning_model(
        "git-base-coco-caption",
        "microsoft/git-base-coco",
        family="git",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        finetune_validated=True,
    ),
    _image_captioning_model(
        "git-large-caption",
        "microsoft/git-large",
        family="git",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _image_captioning_model(
        "git-large-coco-caption",
        "microsoft/git-large-coco",
        family="git",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _image_captioning_model(
        "git-base-vatex-caption",
        "microsoft/git-base-vatex",
        family="git",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _image_captioning_model(
        "git-large-vatex-caption",
        "microsoft/git-large-vatex",
        family="git",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _image_captioning_model(
        "git-base-textcaps-caption",
        "microsoft/git-base-textcaps",
        family="git",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _image_captioning_model(
        "git-large-textcaps-caption",
        "microsoft/git-large-textcaps",
        family="git",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
)

RETRIEVAL_MODELS: tuple[RegistryEntry, ...] = (
    _retrieval_model(
        "clip-vit-base-patch32_retrieval",
        "openai/clip-vit-base-patch32",
        family="clip",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        finetune_validated=True,
    ),
    _retrieval_model(
        "clip-vit-large-patch14_retrieval",
        "openai/clip-vit-large-patch14",
        family="clip",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        finetune_validated=True,
    ),
    _retrieval_model(
        "clip-vit-large-patch14-336_retrieval",
        "openai/clip-vit-large-patch14-336",
        family="clip",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        finetune_validated=True,
    ),
    _retrieval_model(
        "clip-vit-base-patch16_retrieval",
        "openai/clip-vit-base-patch16",
        family="clip",
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        finetune_validated=True,
    ),
    _retrieval_model(
        "fashion-clip_retrieval",
        "patrickjohncyh/fashion-clip",
        family="clip",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _retrieval_model(
        "siglip-base-patch16-224_retrieval",
        "google/siglip-base-patch16-224",
        family="siglip",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _retrieval_model(
        "siglip-so400m-patch14-384_retrieval",
        "google/siglip-so400m-patch14-384",
        family="siglip",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _retrieval_model(
        "siglip2-base-patch16-224_retrieval",
        "google/siglip2-base-patch16-224",
        family="siglip2",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _retrieval_model(
        "tinyclip-vit-8m-16-text-3m_retrieval",
        "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        family="tinyclip",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _retrieval_model(
        "altclip_retrieval",
        "BAAI/AltCLIP",
        family="altclip",
        allowed_training_regimes=INFERENCE_ONLY,
    ),
)

VQA_MODELS: tuple[RegistryEntry, ...] = (
    _vqa_model(
        "blip-vqa-base",
        "Salesforce/blip-vqa-base",
        family="blip",
        explainability=VQA_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["vqav2"],
        finetune_validated=True,
        vqa_label_mode="generation",
    ),
    _vqa_model(
        "vilt-b32-vqa",
        "dandelin/vilt-b32-finetuned-vqa",
        family="vilt",
        explainability=VILT_VQA_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["vqav2"],
        finetune_validated=True,
        vqa_label_mode="classification",
    ),
    _vqa_model(
        "blip-vqa-capfilt-large",
        "Salesforce/blip-vqa-capfilt-large",
        family="blip",
        explainability=VQA_EXPLAINABILITY,
        allowed_training_regimes=INFERENCE_ONLY,
        inference_dataset_keys=["vqav2"],
    ),
    _vqa_model(
        "blip2-opt-2-7b-vqa",
        "Salesforce/blip2-opt-2.7b",
        family="blip2",
        explainability=VQA_EXPLAINABILITY,
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _vqa_model(
        "blip2-opt-2-7b-coco-vqa",
        "Salesforce/blip2-opt-2.7b-coco",
        family="blip2",
        explainability=VQA_EXPLAINABILITY,
        allowed_training_regimes=INFERENCE_ONLY,
    ),
    _vqa_model(
        "git-base-vqav2",
        "microsoft/git-base-vqav2",
        family="git",
        explainability=VQA_EXPLAINABILITY,
        allowed_training_regimes=FINETUNE_AND_INFERENCE,
        inference_dataset_keys=["vqav2"],
        finetune_validated=True,
        vqa_label_mode="generation",
    ),
    _vqa_model(
        "git-large-vqav2",
        "microsoft/git-large-vqav2",
        family="git",
        explainability=VQA_EXPLAINABILITY,
        allowed_training_regimes=INFERENCE_ONLY,
        inference_dataset_keys=["vqav2"],
    ),
    _vqa_model(
        "bingsu-temp-vilt-vqa",
        "Bingsu/temp_vilt_vqa",
        family="vilt",
        explainability=VILT_VQA_EXPLAINABILITY,
        allowed_training_regimes=INFERENCE_ONLY,
        inference_dataset_keys=["vqav2"],
    ),
    _vqa_model(
        "jeney-vilt-b32-vqa",
        "Jeney/vilt-b32-finetuned-vqa",
        family="vilt",
        explainability=VILT_VQA_EXPLAINABILITY,
        allowed_training_regimes=INFERENCE_ONLY,
        inference_dataset_keys=["vqav2"],
    ),
    _vqa_model(
        "vilt-33m-vqa",
        "jmonas/ViLT-33M-vqa",
        family="vilt",
        explainability=VILT_VQA_EXPLAINABILITY,
        allowed_training_regimes=INFERENCE_ONLY,
        inference_dataset_keys=["vqav2"],
    ),
)

MODEL_REGISTRY: dict[str, dict[str, object]] = _build_registry(
    *TEXT_CLASSIFICATION_MODELS,
    *TOKEN_CLASSIFICATION_MODELS,
    *SENTENCE_SIMILARITY_MODELS,
    *FILL_MASK_MODELS,
    *TEXT_GENERATION_MODELS,
    *TEXT2TEXT_GENERATION_MODELS,
    *IMAGE_CLASSIFICATION_MODELS,
    *OBJECT_DETECTION_MODELS,
    *IMAGE_SEGMENTATION_MODELS,
    *IMAGE_CAPTIONING_MODELS,
    *RETRIEVAL_MODELS,
    *VQA_MODELS,
)
