import numpy as np

from .label_schema import attach_label_schema


def _resolve_pair_columns(text_column):
    if isinstance(text_column, str):
        stripped = text_column.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            parts = [p.strip().strip("\"'") for p in stripped[1:-1].split(",") if p.strip()]
            if len(parts) == 2:
                return parts
    return text_column


def _is_continuous_labels(labels):
    arr = np.asarray(labels)
    if arr.size == 0:
        return False
    if np.issubdtype(arr.dtype, np.floating):
        return True
    if np.issubdtype(arr.dtype, np.integer):
        return False

    for value in labels:
        if value is None:
            continue
        if isinstance(value, float) and not float(value).is_integer():
            return True
    return False


def _normalize_regression_targets(train_labels, test_labels):
    train_arr = np.asarray(train_labels, dtype="float32")
    test_arr = np.asarray(test_labels, dtype="float32")

    lo = float(np.min(train_arr)) if train_arr.size else 0.0
    hi = float(np.max(train_arr)) if train_arr.size else 0.0

    if hi > lo:
        train_norm = (train_arr - lo) / (hi - lo)
        test_norm = (test_arr - lo) / (hi - lo)
        norm_mode = "minmax"
    else:
        train_norm = np.zeros_like(train_arr, dtype="float32")
        test_norm = np.zeros_like(test_arr, dtype="float32")
        norm_mode = "constant"

    return train_norm.astype("float32"), test_norm.astype("float32"), lo, hi, norm_mode


def _encode_categorical_targets(train_labels, test_labels, label_feat):
    try:
        from datasets import ClassLabel
    except Exception:
        ClassLabel = None

    num_classes = None
    label_mapping = None

    if ClassLabel is not None and isinstance(label_feat, ClassLabel):
        num_classes = int(label_feat.num_classes)
        label_mapping = {str(name): int(i) for i, name in enumerate(getattr(label_feat, "names", []))}

    uniq = {}
    y_train = []
    for value in train_labels:
        if value not in uniq:
            uniq[value] = len(uniq)
        y_train.append(uniq[value])

    y_test = []
    for value in test_labels:
        if value not in uniq:
            raise ValueError(f"Unseen label in test split: {value!r}")
        y_test.append(uniq[value])

    y_train = np.asarray(y_train, dtype="int32")
    y_test = np.asarray(y_test, dtype="int32")

    if num_classes is None:
        num_classes = int(len(uniq))
    if not label_mapping:
        label_mapping = {str(k): int(v) for k, v in uniq.items()}

    return y_train, y_test, num_classes, label_mapping


def preprocess_hf_text_similarity(
    train,
    test,
    meta,
    *,
    hf_model_id,
    text_column,
    label_column="label",
    label_mode="auto",
    dynamic_padding=False,
):
    ds_train, _ = train
    ds_test, _ = test
    text_column = _resolve_pair_columns(text_column)

    cols = set(ds_train.column_names)
    if not (isinstance(text_column, (list, tuple)) and len(text_column) == 2):
        raise ValueError("sentence_similarity requires text_column=[text_a, text_b]")

    text_col_a, text_col_b = text_column[0], text_column[1]
    if text_col_a not in cols or text_col_b not in cols:
        raise ValueError(
            f"Missing text pair columns {text_column} in dataset '{meta.get('hf_id')}'. Available: {sorted(cols)}"
        )

    if label_column not in cols:
        raise ValueError(f"Missing label_column '{label_column}' in dataset '{meta.get('hf_id')}'")

    train_labels = list(ds_train[label_column])
    test_labels = list(ds_test[label_column])
    label_feat = ds_train.features.get(label_column)

    mode = str(label_mode or "auto").lower()
    if mode not in {"auto", "regression", "categorical"}:
        raise ValueError("label_mode must be one of: auto, regression, categorical")

    is_regression = (mode == "regression") or (mode == "auto" and _is_continuous_labels(train_labels))

    if is_regression:
        y_train, y_test, y_min, y_max, norm_mode = _normalize_regression_targets(train_labels, test_labels)
        num_classes = None
        label_mapping = None
    else:
        y_train, y_test, num_classes, label_mapping = _encode_categorical_targets(
            train_labels,
            test_labels,
            label_feat,
        )
        y_min = y_max = None
        norm_mode = None

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise ImportError(
            "HF text preprocessing requires 'transformers'. Install with: pip install transformers"
        ) from e

    max_length = int(meta.get("max_length", 128))
    dynamic_padding = bool(dynamic_padding)
    padding_mode = "dynamic" if dynamic_padding else "max_length"
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=False)

    enc_train = tokenizer(
        list(ds_train[text_col_a]),
        list(ds_train[text_col_b]),
        truncation=True,
        padding=False if dynamic_padding else "max_length",
        max_length=max_length,
    )
    enc_test = tokenizer(
        list(ds_test[text_col_a]),
        list(ds_test[text_col_b]),
        truncation=True,
        padding=False if dynamic_padding else "max_length",
        max_length=max_length,
    )

    if dynamic_padding:
        train_max_len = max((len(ids) for ids in enc_train["input_ids"]), default=0)
        test_max_len = max((len(ids) for ids in enc_test["input_ids"]), default=0)
        pad_to = max(train_max_len, test_max_len)
    else:
        pad_to = max_length

    enc_train = tokenizer.pad(enc_train, padding="max_length", max_length=pad_to, return_tensors="np")
    enc_test = tokenizer.pad(enc_test, padding="max_length", max_length=pad_to, return_tensors="np")

    X_train = {
        "input_ids": enc_train["input_ids"].astype("int32"),
        "attention_mask": enc_train["attention_mask"].astype("int32"),
    }
    X_test = {
        "input_ids": enc_test["input_ids"].astype("int32"),
        "attention_mask": enc_test["attention_mask"].astype("int32"),
    }

    if "token_type_ids" in enc_train:
        X_train["token_type_ids"] = enc_train["token_type_ids"].astype("int32")
        X_test["token_type_ids"] = enc_test["token_type_ids"].astype("int32")

    task_type = "regression" if is_regression else "classification"
    meta2 = dict(meta)
    meta2.update({
        "input_shape": (pad_to,),
        "num_classes": None if is_regression else int(num_classes),
        "label_mapping": label_mapping,
        "text_column": list(text_column),
        "label_column": label_column,
        "hf_model_id": hf_model_id,
        "x_format": "dict",
        "x_keys": list(X_train.keys()),
        "label_granularity": "pair_sequence",
        "hf_task": "sentence_similarity",
        "modality": "text",
        "task_type": task_type,
        "is_regression": bool(is_regression),
        "dynamic_padding": dynamic_padding,
        "padding_mode": padding_mode,
    })

    if is_regression:
        meta2.update({
            "label_normalization": norm_mode,
            "label_min": y_min,
            "label_max": y_max,
            "regression_target": "similarity_score",
            "num_labels": 1,
        })

    meta2 = attach_label_schema(meta2, y_train, default_num_labels=(None if is_regression else num_classes))

    return (X_train, y_train), (X_test, y_test), meta2