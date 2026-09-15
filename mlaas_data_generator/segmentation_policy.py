"""Dataset semantics established by the ADE20K-mini mask audit."""
from collections.abc import Mapping

ADE20K_MINI = "qubvel-hf/ade20k-mini"
ADE20K_MINI_LABELS = {"background": 0, "person": 1, "car": 2}
ADE20K_MINI_CONTRACT_VERSION = "ade20k_mini_three_labels_v1"


def dataset_id(config):
    args = config.get("dataset_args") or {}
    return config.get("hf_id") or config.get("dataset_name") or (
        args.get("dataset_name") if isinstance(args, Mapping) else None)


def is_ade20k_mini(config):
    return str(dataset_id(config) or "").strip().lower() == ADE20K_MINI


def apply_segmentation_label_policy(config):
    if is_ade20k_mini(config) and not config.get("inference_only") and config.get(
        "training_regime") not in {"inference", "inference_only"}:
        config.update(num_labels=3, num_classes=3)
    return config


def validate_ade20k_mini_metadata(meta, config=None):
    if not (is_ade20k_mini(meta) or is_ade20k_mini(config or {})):
        return
    if (meta.get("num_labels") != 3 or meta.get("num_classes") != 3
            or meta.get("label_mapping") != ADE20K_MINI_LABELS
            or meta.get("segmentation_reduce_labels") is not False):
        raise ValueError("ADE20K-mini fine-tuning requires background=0/person=1/car=2 "
                         "and unreduced three-label masks; regenerate legacy preparations")


def validate_ade20k_mini_preparation(document):
    """Reject legacy production reuse while leaving historical artifacts readable."""
    if not is_ade20k_mini(document.get("config", {})):
        return
    labels = document.get("contracts", {}).get("labels", {})
    expected = [{"index": index, "label": name}
                for name, index in ADE20K_MINI_LABELS.items()]
    if (labels.get("num_labels") != 3 or labels.get("labels") != expected
            or labels.get("special_semantics", {}).get("segmentation_reduce_labels") is not False):
        raise ValueError("legacy ADE20K-mini family has an incompatible label contract; "
                         "regenerate with a new service ID and manifest")
