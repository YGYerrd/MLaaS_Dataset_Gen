import numpy as np


def attach_label_schema(meta: dict, y_train, *, default_num_labels: int | None = None, ignore_index: int | None = None) -> dict:
    """Attach shared label contract to metadata emitted by preprocessors."""
    meta2 = dict(meta)
    task_type = str(meta2.get("task_type", "classification")).lower()

    if task_type == "regression":
        meta2["label_format"] = "continuous"
        meta2["num_labels"] = 1
        return meta2

    if task_type == "clustering":
        meta2["label_format"] = "single_index"
        meta2["num_labels"] = int(default_num_labels or meta2.get("num_classes") or 0)
        return meta2

    if meta2.get("label_granularity") in {"token", "span"}:
        meta2["label_format"] = "token_index"
        if ignore_index is not None:
            meta2["ignore_index"] = int(ignore_index)
    elif bool(meta2.get("is_multilabel")):
        meta2["label_format"] = "multihot"
    else:
        arr = np.asarray(y_train)
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
            row_sums = arr.sum(axis=1)
            is_binary = np.isin(arr, [0, 1]).all()
            meta2["label_format"] = "onehot" if is_binary and np.all(row_sums == 1) else "multihot"
        else:
            meta2["label_format"] = "single_index"

    resolved = default_num_labels
    if resolved is None:
        resolved = meta2.get("num_classes")
    if resolved is None:
        resolved = meta2.get("num_labels")
    if resolved is not None:
        meta2["num_labels"] = int(resolved)

    return meta2