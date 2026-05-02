def infer_num_labels(meta: dict | None = None, fallback: int | None = None) -> int | None:
    """Single source of truth for label-count inference."""
    if isinstance(meta, dict):
        for key in ("num_labels", "num_classes"):
            value = meta.get(key)
            if value is not None:
                return int(value)
    if fallback is None:
        return None
    return int(fallback)


def infer_label_format(meta: dict | None = None, task_type: str | None = None) -> str:
    if isinstance(meta, dict) and meta.get("label_format"):
        return str(meta["label_format"]).lower()

    base_task = (task_type or (meta or {}).get("task_type") or "").lower()
    if base_task == "regression":
        return "continuous"
    if base_task == "clustering":
        return "cluster_id"
    return "single_index"


def infer_ignore_index(meta: dict | None = None, default: int = -100) -> int:
    if isinstance(meta, dict):
        for key in ("ignore_index", "label_pad_value"):
            if meta.get(key) is not None:
                return int(meta[key])
    return int(default)
