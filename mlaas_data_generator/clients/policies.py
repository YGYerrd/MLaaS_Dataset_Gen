"""Broad fallback/exploration ranges; calibration adds observed joint recipes.

Model size affects resource limits in resources.py. Family evidence selection
lives in calibration.py, leaving unsupported tasks on these explicit defaults.
"""
from __future__ import annotations

from .resources import task_key

POLICY_VERSION = "catalogue_training_policies_v2"

RANDOM_INITIALIZATION_EPOCHS = (5, 10, 20)
RANDOM_INITIALIZATION_SAMPLE_MULTIPLIER = 2


def uses_small_random_initialization(parent):
    """Local neural baselines, excluding clients resuming a trained parent."""
    config = parent["config"]
    return (config.get("training_regime") == "generic"
            and config.get("model_type") in {"cnn", "mlp", "logreg"}
            and not parent.get("parent_state_hash")
            and not config.get("parent_state_hash")
            and config.get("generation_mode") != "shared_parent")

# These are broad defaults, not per-family optima or promises of model quality.
LEARNING_RATES = {
    "hf_text": {
        "adam": [1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
        "adamw": [1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
        "sgd": [5e-5, 1e-3, 3e-3, 1e-2],
        "rmsprop": [1e-5, 1.5e-5, 2e-5],
    },
    "hf_image": {
        "adam": [5e-6, 1e-5, 2e-5, 5e-5], "adamw": [5e-6, 1e-5, 2e-5, 5e-5],
        "sgd": [5e-4, 1e-3, 2e-3, 5e-3], "rmsprop": [2.5e-6, 5e-6, 1e-5, 2e-5],
    },
    "hf_generation": {
        "adam": [1e-5, 2e-5, 3e-5], "adamw": [1e-5, 2e-5, 3e-5],
        "sgd": [1e-3, 2e-3, 3e-3], "rmsprop": [5e-6, 1e-5, 1.5e-5],
    },
    "hf_multimodal": {
        "adam": [1e-6, 5e-6, 1e-5, 2e-5], "adamw": [1e-6, 5e-6, 1e-5, 2e-5],
        "sgd": [1e-4, 5e-4, 1e-3, 2e-3], "rmsprop": [5e-7, 2.5e-6, 5e-6, 1e-5],
    },
    "generic_classification": {
        "adam": [1e-4, 3e-4, 1e-3, 3e-3], "adamw": [1e-4, 3e-4, 1e-3, 3e-3],
        "sgd": [1e-3, 3e-3, 1e-2], "rmsprop": [1e-4, 3e-4, 1e-3],
        "adagrad": [1e-3, 3e-3, 1e-2],
    },
    "generic_regression": {
        "adam": [1e-3, 3e-3], "adamw": [1e-3, 3e-3], "sgd": [1e-2, 3e-2],
        "rmsprop": [1e-3, 3e-3], "adagrad": [1e-2, 3e-2],
    },
    "generic_non_gradient": {"none": [None]},
}


def policy_key(config, *, task=None):
    """Select by backend/task only; no dataset names, checkpoints or scores."""
    task = task or task_key(config)
    if config.get("training_regime") == "generic":
        if config.get("model_type") in {"randomforest", "kmeans"}:
            return "generic_non_gradient"
        return "generic_regression" if config.get("task_type") == "regression" else "generic_classification"
    if task in {"image_classification", "object_detection", "image_segmentation"}:
        return "hf_image"
    if task in {"image_captioning", "visual_question_answering", "text_image_retrieval"}:
        return "hf_multimodal"
    if task in {"text_generation", "text2text_generation"}:
        return "hf_generation"
    return "hf_text"
