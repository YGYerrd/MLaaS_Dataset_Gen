"""Checkpoint representation selection is independent of training precision."""
def normalize_weight_format(value):
    value = str(value or "native").strip().lower()
    if value not in {"native", "fp16", "int8"}:
        raise ValueError("checkpoint_weight_format must be native, fp16 or int8")
    return value
