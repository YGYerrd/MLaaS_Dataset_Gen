import numpy as np

from .label_schema import attach_label_schema
from ..accounting import append_accounting_stage, finalize_accounting
from ...models.adapters.hf_cache import get_cached_tokenizer


def _resolve_text_column_spec(text_column):
    if isinstance(text_column, str):
        stripped = text_column.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            parts = [p.strip().strip("\"'") for p in stripped[1:-1].split(",") if p.strip()]
            if len(parts) == 2:
                return parts
        return text_column
    return text_column


def _dataset_len(ds, fallback_column):
    try:
        return int(len(ds))
    except Exception:
        return int(len(ds[fallback_column]))


def _is_multilabel_sample(value):
    return isinstance(value, (list, tuple, set, np.ndarray))


def _encode_multilabel_targets(train_labels, test_labels, label_feat):
    try:
        from datasets import ClassLabel, Sequence
    except Exception:
        ClassLabel = None
        Sequence = None

    num_classes = None
    label_mapping = None

    if (
        ClassLabel is not None
        and Sequence is not None
        and isinstance(label_feat, Sequence)
        and isinstance(getattr(label_feat, "feature", None), ClassLabel)
    ):
        class_label = label_feat.feature
        num_classes = int(class_label.num_classes)
        if getattr(class_label, "names", None):
            label_mapping = {str(name): int(i) for i, name in enumerate(class_label.names)}

    if num_classes is None:
        max_label = -1
        for ys in (train_labels, test_labels):
            for row in ys:
                for idx in row:
                    max_label = max(max_label, int(idx))
        num_classes = max_label + 1

    if num_classes <= 0:
        raise ValueError("Could not infer num_classes for multi-label targets")

    def to_multi_hot(rows):
        out = np.zeros((len(rows), num_classes), dtype="float32")
        for i, row in enumerate(rows):
            for idx in row:
                j = int(idx)
                if j < 0 or j >= num_classes:
                    raise ValueError(f"Label index out of range for multi-label target: {j}")
                out[i, j] = 1.0
        return out

    y_train = to_multi_hot(train_labels)
    y_test = to_multi_hot(test_labels)

    if label_mapping is None:
        label_mapping = {str(i): i for i in range(num_classes)}

    return y_train, y_test, num_classes, label_mapping

def preprocess_hf_text_sequence(
    train,
    test,
    meta,
    *,
    hf_model_id,
    text_column="text",
    label_column="label",
    dynamic_padding=False,
):
    ds_train, _ = train
    ds_test, _ = test
    text_column = _resolve_text_column_spec(text_column)


    cols = set(ds_train.column_names)

    is_pair = False
    if isinstance(text_column, str):
        if text_column not in cols:
            raise ValueError(f"Missing text_column '{text_column}' in dataset '{meta.get('hf_id')}'")
        text_col_1 = text_column
        text_col_2 = None
    elif isinstance(text_column, (list, tuple)) and len(text_column) == 2:
        is_pair = True
        text_col_1 = text_column[0]
        text_col_2 = text_column[1]
        if text_col_1 not in cols or text_col_2 not in cols:
            raise ValueError(
                f"Missing text_column pair {text_column} in dataset '{meta.get('hf_id')}'. "
                f"Available: {sorted(cols)}"
            )
    else:
        raise ValueError("text_column must be a string or a list/tuple of length 2 for text pairs.")

    if label_column not in cols:
        raise ValueError(f"Missing label_column '{label_column}' in dataset '{meta.get('hf_id')}'")

    try:
        from datasets import ClassLabel
    except Exception:
        ClassLabel = None

    label_feat = ds_train.features.get(label_column)

    train_labels = list(ds_train[label_column])
    test_labels = list(ds_test[label_column])

    first_non_null = next((v for v in train_labels if v is not None), None)
    is_multilabel = _is_multilabel_sample(first_non_null)

    if is_multilabel:
        y_train, y_test, num_classes, label_mapping = _encode_multilabel_targets(
            train_labels,
            test_labels,
            label_feat,
        )
    elif ClassLabel is not None and isinstance(label_feat, ClassLabel):
        num_classes = int(label_feat.num_classes)
        y_train = np.asarray(train_labels, dtype="int32")
        y_test = np.asarray(test_labels, dtype="int32")
        label_mapping = {str(i): i for i in range(num_classes)}
    else:
        uniq = {}
        y_train_list = []
        for v in train_labels:
            if v not in uniq:
                uniq[v] = len(uniq)
            y_train_list.append(uniq[v])

        y_test_list = []
        for v in ds_test[label_column]:
            if v not in uniq:
                raise ValueError(f"Unseen label in test split: {v!r}")
            y_test_list.append(uniq[v])

        y_train = np.asarray(y_train_list, dtype="int32")
        y_test = np.asarray(y_test_list, dtype="int32")
        num_classes = int(len(uniq))
        label_mapping = {str(k): int(v) for k, v in uniq.items()}

    try:
        import transformers
    except Exception as e:
        raise ImportError(
            "HF text preprocessing requires 'transformers'. Install with: pip install transformers"
        ) from e

    max_length = int(meta.get("max_length", 128))
    dynamic_padding = bool(dynamic_padding)
    padding_mode = "dynamic" if dynamic_padding else "max_length"
    tokenizer, _, _ = get_cached_tokenizer(
        hf_model_id=hf_model_id,
        task="sequence_classification",
        device="cpu",
        transformers_module=transformers,
    )

    tokenize_padding = False if dynamic_padding else "max_length"

    if not is_pair:

        enc_train = tokenizer(
            list(ds_train[text_col_1]),
            truncation=True,
            padding=tokenize_padding,
            max_length=max_length,
        )
        enc_test = tokenizer(
            list(ds_test[text_col_1]),
            truncation=True,
            padding=tokenize_padding,
            max_length=max_length,
        )
    else:

        enc_train = tokenizer(
            list(ds_train[text_col_1]),
            list(ds_train[text_col_2]),
            truncation=True,
            padding=tokenize_padding,
            max_length=max_length,
        )
        enc_test = tokenizer(
            list(ds_test[text_col_1]),
            list(ds_test[text_col_2]),
            truncation=True,
            padding=tokenize_padding,
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

    meta2 = dict(meta)
    meta2.update({
        "input_shape": (max_length,),
        "num_classes": num_classes,
        "label_mapping": label_mapping,
        "text_column": text_column,
        "label_column": label_column,
        "hf_model_id": hf_model_id,
        "x_format": "dict",
        "x_keys": list(X_train.keys()),
        "label_granularity": "sequence",
        "hf_task": "sequence_classification",
        "is_multilabel": bool(is_multilabel),
        "classification_type": "multilabel" if is_multilabel else "single_label",
        "modality": "text",
        "dynamic_padding": dynamic_padding,
        "padding_mode": padding_mode,
    })
    meta2["input_shape"] = (pad_to,)
    meta2 = attach_label_schema(meta2, y_train, default_num_labels=num_classes)
    train_sequences = int(X_train["input_ids"].shape[0])
    test_sequences = int(X_test["input_ids"].shape[0])
    train_count = _dataset_len(ds_train, text_col_1)
    test_count = _dataset_len(ds_test, text_col_1)
    meta2 = append_accounting_stage(
        meta2,
        stage="hf_text_sequence",
        split="train",
        input_record_count=train_count,
        post_filter_record_count=train_count,
        tokenized_record_count=train_sequences,
        emitted_record_count=train_sequences,
        sequence_count=train_sequences,
        metric_instance_count=train_sequences,
    )
    meta2 = append_accounting_stage(
        meta2,
        stage="hf_text_sequence",
        split="test",
        input_record_count=test_count,
        post_filter_record_count=test_count,
        tokenized_record_count=test_sequences,
        emitted_record_count=test_sequences,
        sequence_count=test_sequences,
        metric_instance_count=test_sequences,
    )
    meta2 = finalize_accounting(meta2)

    return (X_train, y_train), (X_test, y_test), meta2
