import numpy as np

from .label_schema import attach_label_schema
from ..accounting import append_accounting_stage, finalize_accounting
from ...models.adapters.hf_cache import get_cached_tokenizer


def _extract_class_label_names(label_feature):
    if hasattr(label_feature, "names"):
        return list(label_feature.names)

    nested = getattr(label_feature, "feature", None)
    if nested is not None:
        names = _extract_class_label_names(nested)
        if names:
            return names

    if isinstance(label_feature, (list, tuple)) and len(label_feature) == 1:
        return _extract_class_label_names(label_feature[0])

    return None


def preprocess_hf_text_token(train, test, meta, *, hf_model_id, tokens_column, label_column, dynamic_padding=False):
    ds_train, _ = train
    ds_test, _ = test

    cols = set(ds_train.column_names)
    if not tokens_column:
        raise ValueError("token_classification requires tokens_column=<column_name>")
    if tokens_column not in cols:
        raise ValueError(f"Missing tokens_column '{tokens_column}' in dataset '{meta.get('hf_id')}'")
    if label_column not in cols:
        raise ValueError(f"Missing label_column '{label_column}' in dataset '{meta.get('hf_id')}'")

    label_feat = ds_train.features.get(label_column)
    label_names = _extract_class_label_names(label_feat)
    if not label_names:
        raise ValueError("Token classification requires Sequence(ClassLabel) style labels.")
    num_classes = int(len(label_names))
    label_mapping = {name: idx for idx, name in enumerate(label_names)}

    try:
        import transformers
    except Exception as e:
        raise ImportError(
            "HF token preprocessing requires 'transformers'. Install with: pip install transformers"
        ) from e

    max_length = int(meta.get("max_length", 128))
    dynamic_padding = bool(dynamic_padding)
    padding_mode = "dynamic" if dynamic_padding else "max_length"
    tokenizer, _, _ = get_cached_tokenizer(
        hf_model_id=hf_model_id,
        task="token_classification",
        device="cpu",
        transformers_module=transformers,
    )

    def _encode_tokens_and_labels(tokens_list, tags_list):
        enc = tokenizer(
            tokens_list,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )

        labels = np.full((len(tokens_list), max_length), -100, dtype="int32")
        for i in range(len(tokens_list)):
            word_ids = enc.word_ids(batch_index=i)
            prev_word = None
            for j, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id != prev_word and word_id < len(tags_list[i]):
                    labels[i, j] = int(tags_list[i][word_id])
                prev_word = word_id

        return {
            "input_ids": enc["input_ids"].astype("int32"),
            "attention_mask": enc["attention_mask"].astype("int32"),
        }, labels

    X_train, y_train = _encode_tokens_and_labels(list(ds_train[tokens_column]), list(ds_train[label_column]))
    X_test, y_test = _encode_tokens_and_labels(list(ds_test[tokens_column]), list(ds_test[label_column]))

    meta2 = dict(meta)
    meta2.update({
        "input_shape": (max_length,),
        "num_classes": num_classes,
        "label_mapping": label_mapping,
        "tokens_column": tokens_column,
        "label_column": label_column,
        "hf_model_id": hf_model_id,
        "x_format": "dict",
        "x_keys": ["input_ids", "attention_mask"],
        "label_granularity": "token",
        "hf_task": "token_classification",
        "modality": "text",
        "label_pad_value": -100,
        "dynamic_padding": dynamic_padding,
        "padding_mode": padding_mode,
    })
    meta2 = attach_label_schema(meta2, y_train, default_num_labels=num_classes, ignore_index=-100)
    train_sequences = int(X_train["input_ids"].shape[0])
    test_sequences = int(X_test["input_ids"].shape[0])
    train_supervised_tokens = int(np.count_nonzero(y_train != -100))
    test_supervised_tokens = int(np.count_nonzero(y_test != -100))
    meta2 = append_accounting_stage(
        meta2,
        stage="hf_text_token",
        split="train",
        input_record_count=len(ds_train),
        post_filter_record_count=len(ds_train),
        tokenized_record_count=train_sequences,
        emitted_record_count=train_sequences,
        sequence_count=train_sequences,
        supervised_token_count=train_supervised_tokens,
        metric_instance_count=train_sequences,
    )
    meta2 = append_accounting_stage(
        meta2,
        stage="hf_text_token",
        split="test",
        input_record_count=len(ds_test),
        post_filter_record_count=len(ds_test),
        tokenized_record_count=test_sequences,
        emitted_record_count=test_sequences,
        sequence_count=test_sequences,
        supervised_token_count=test_supervised_tokens,
        metric_instance_count=test_sequences,
    )
    meta2 = finalize_accounting(meta2)

    return (X_train, y_train), (X_test, y_test), meta2
