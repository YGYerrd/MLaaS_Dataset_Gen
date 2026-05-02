import numpy as np

from .label_schema import attach_label_schema
from ...models.adapters.hf_cache import get_cached_tokenizer


def _tokenize_texts(tokenizer, texts, max_length, dynamic_padding=False):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=False if dynamic_padding else "max_length",
        max_length=max_length,
        return_special_tokens_mask=True,
    )


def _build_mlm_labels(encodings, tokenizer, mlm_probability, label_pad_value, rng):
    input_ids = encodings["input_ids"].astype("int32").copy()
    attention_mask = encodings["attention_mask"].astype("int32")

    if "special_tokens_mask" in encodings:
        special_tokens_mask = encodings["special_tokens_mask"].astype(bool)
    else:
        special_tokens_mask = np.zeros_like(input_ids, dtype=bool)

    if tokenizer.mask_token_id is None:
        raise ValueError("fill_mask preprocessing requires a tokenizer with mask_token_id")

    candidate_mask = (attention_mask == 1) & (~special_tokens_mask)

    masked_flags = rng.random(input_ids.shape) < float(mlm_probability)
    masked_flags &= candidate_mask

    labels = np.full_like(input_ids, int(label_pad_value), dtype="int32")
    labels[masked_flags] = input_ids[masked_flags]

    if np.any(masked_flags):
        replace_prob = rng.random(input_ids.shape)

        replace_with_mask = masked_flags & (replace_prob < 0.8)
        replace_with_random = masked_flags & (replace_prob >= 0.8) & (replace_prob < 0.9)

        input_ids[replace_with_mask] = int(tokenizer.mask_token_id)

        if np.any(replace_with_random):
            vocab_size = int(tokenizer.vocab_size)
            input_ids[replace_with_random] = rng.integers(0, vocab_size, size=int(replace_with_random.sum()), dtype=np.int32)

    features = {
        "input_ids": input_ids.astype("int32"),
        "attention_mask": attention_mask.astype("int32"),
    }
    if "token_type_ids" in encodings:
        features["token_type_ids"] = encodings["token_type_ids"].astype("int32")

    return features, labels


def preprocess_hf_text_fill_mask(
    train,
    test,
    meta,
    *,
    hf_model_id,
    text_column="text",
    mlm_probability=0.15,
    label_pad_value=-100,
    dynamic_padding=False,
):
    ds_train, _ = train
    ds_test, _ = test

    cols = set(ds_train.column_names)
    if text_column not in cols:
        raise ValueError(f"Missing text_column '{text_column}' in dataset '{meta.get('hf_id')}'")

    try:
        import transformers
    except Exception as e:
        raise ImportError(
            "HF fill-mask preprocessing requires 'transformers'. Install with: pip install transformers"
        ) from e

    max_length = int(meta.get("max_length", 128))
    dynamic_padding = bool(dynamic_padding)
    padding_mode = "dynamic" if dynamic_padding else "max_length"
    mlm_probability = float(mlm_probability)
    label_pad_value = int(label_pad_value)

    if mlm_probability <= 0.0 or mlm_probability >= 1.0:
        raise ValueError("mlm_probability must be in the open interval (0, 1)")

    tokenizer, _, _ = get_cached_tokenizer(
        hf_model_id=hf_model_id,
        task="fill_mask",
        device="cpu",
        transformers_module=transformers,
    )

    seed = int(meta.get("seed", 42))
    train_rng = np.random.default_rng(seed)
    test_rng = np.random.default_rng(seed + 1)

    enc_train = _tokenize_texts(tokenizer, ds_train[text_column], max_length=max_length, dynamic_padding=dynamic_padding)
    enc_test = _tokenize_texts(tokenizer, ds_test[text_column], max_length=max_length, dynamic_padding=dynamic_padding)

    if dynamic_padding:
        train_max_len = max((len(ids) for ids in enc_train["input_ids"]), default=0)
        test_max_len = max((len(ids) for ids in enc_test["input_ids"]), default=0)
        pad_to = max(train_max_len, test_max_len)
        enc_train = tokenizer.pad(enc_train, padding="max_length", max_length=pad_to, return_tensors="np")
        enc_test = tokenizer.pad(enc_test, padding="max_length", max_length=pad_to, return_tensors="np")
    else:
        pad_to = max_length
        enc_train = tokenizer.pad(enc_train, padding="max_length", max_length=pad_to, return_tensors="np")
        enc_test = tokenizer.pad(enc_test, padding="max_length", max_length=pad_to, return_tensors="np")

    X_train, y_train = _build_mlm_labels(enc_train, tokenizer, mlm_probability, label_pad_value, train_rng)
    X_test, y_test = _build_mlm_labels(enc_test, tokenizer, mlm_probability, label_pad_value, test_rng)

    meta2 = dict(meta)
    meta2.update({
        "input_shape": (pad_to,),
        "num_classes": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "text_column": text_column,
        "hf_model_id": hf_model_id,
        "x_format": "dict",
        "x_keys": list(X_train.keys()),
        "label_granularity": "token",
        "hf_task": "fill_mask",
        "modality": "text",
        "label_pad_value": label_pad_value,
        "mlm_probability": mlm_probability,
        "hf_tokenizer_required": "AutoTokenizer",
        "hf_model_required": "AutoModelForMaskedLM",
        "dynamic_padding": dynamic_padding,
        "padding_mode": padding_mode,
    })
    meta2 = attach_label_schema(
        meta2,
        y_train,
        default_num_labels=meta2["num_classes"],
        ignore_index=label_pad_value,
    )

    return (X_train, y_train), (X_test, y_test), meta2