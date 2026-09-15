"""Temporary full-precision policy for model families with unstable AMP gradients."""

FP32_REASON = "mobile_models_fp32_stability_policy_v2"
T5_FP32_REASON = "t5_fp32_train_eval_stability_policy_v1"
SEGFORMER_MINI_FP32_REASON = "segformer_ade20k_mini_fp32_stability_v1"


def fp32_policy_reason(model_id=None, model_family=None, *, dataset_id=None):
    name = str(model_id or "").lower().replace("-", "_").split("/")[-1]
    family = str(model_family or "").lower().replace("-", "_")
    from .segmentation_policy import ADE20K_MINI
    if str(dataset_id or "").strip().lower() == ADE20K_MINI and (
        name.startswith("segformer_") or family == "segformer"
    ):
        return SEGFORMER_MINI_FP32_REASON
    if family in {"t5", "mt5", "umt5", "longt5"} or any(
        name == marker or name.startswith(marker + "_")
        for marker in ("t5", "flan_t5", "byt5", "mt5", "umt5", "longt5", "codet5")
    ):
        return T5_FP32_REASON
    if requires_fp32(model_id, model_family):
        return FP32_REASON
    return None


def requires_fp32(model_id=None, model_family=None):
    name = str(model_id or "").lower().replace("-", "_")
    family = str(model_family or "").lower().replace("-", "_")
    return any(marker in name or family.startswith(marker) for marker in ("mobilenet", "efficientnet", "mobilevit", "mobilebert"))


def apply_precision_policy(config, *, model_family=None):
    from .segmentation_policy import dataset_id
    reason = fp32_policy_reason(config.get("hf_model_id") or config.get("model_id"), model_family,
                               dataset_id=dataset_id(config))
    if reason:
        config.update(mixed_precision=False, precision_type="fp32", precision_policy=reason)
    return config
