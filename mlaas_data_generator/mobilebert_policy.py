"""Explicit defaults for newly generated MobileBERT classification services."""

NORMALIZATION = "layer_norm_v1"


def apply_mobilebert_generation_defaults(config, *, model_family=None):
    name = str(config.get("hf_model_id") or "").lower()
    family = str(model_family or "").lower()
    task = config.get("hf_task") or config.get("task")
    if ("mobilebert" in name or family == "mobilebert") and task in {
        "sequence_classification", "text_classification", "sentence_similarity"
    } and config.get("training_regime") not in {"inference", "inference_only"}:
        config.update(mobilebert_head_normalization=NORMALIZATION, max_grad_norm=1.0,
                      mixed_precision=False, precision_type="fp32")
    return config
