from .hf_strategy import HFStrategy
from .keras_strategy import ClassificationStrategy, RegressionStrategy
from .clustering import ClusteringStrategy
from .base import TaskStrategy, canonical_task_family


def make_task_strategy(task_type: str, meta: dict, knobs: dict, config: dict, x_test, y_test, metric_key: str, save_weights: bool) -> TaskStrategy:
    mt = (config.get("model_type") or "").lower()
    is_hf_model = mt in ("hf", "hf_text", "transformers", "hf_finetune", "hf_train", "transformers_finetune")
    hf_task = (config.get("hf_task") or (config.get("dataset_args", {}) or {}).get("hf_task"))
    task_family = canonical_task_family(task_type, hf_task)

    if is_hf_model and task_family in {
        "classification",
        "regression",
        "token_classification",
        "fill_mask",
        "detection",
        "segmentation",
        "generation",
        "retrieval",
        "vqa",
    }:
        return HFStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)

    if task_type == "classification":
        return ClassificationStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)

    if task_type == "regression":
        return RegressionStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)

    if task_type == "clustering":
        return ClusteringStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)

    raise ValueError(f"Unknown task type: {task_type}")
