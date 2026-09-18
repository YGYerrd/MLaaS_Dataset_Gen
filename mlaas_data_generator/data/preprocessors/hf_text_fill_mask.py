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


def _source_mlm_labels(encodings, tokenizer, probability, ignore, *, seed, source_ids):
    """Keep masking independent of client union, row order and padding width."""
    from ...provenance import sha256_json
    features = {key: np.asarray(value, dtype="int32").copy() for key, value in encodings.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}}
    targets = np.full_like(features["input_ids"], ignore)
    for index, source_id in enumerate(source_ids):
        mask = np.asarray(encodings["attention_mask"][index]).astype(bool)
        row = {key: np.asarray(value[index])[mask][None, :] for key, value in encodings.items()}
        row_seed = int(sha256_json(["source_row_mask_v1", seed, source_id])[:16], 16)
        prepared, labels = _build_mlm_labels(row, tokenizer, probability, ignore, np.random.default_rng(row_seed))
        for key in features:
            features[key][index, mask] = prepared[key][0]
        targets[index, mask] = labels[0]
    return features, targets


def _prepare_split(ds, tokenizer, text_column, max_length, probability, ignore, *,
                   seed, source_rows, split, batch_size=256):
    """Bound tokenizer/native encoding memory to one batch of source rows."""
    features = None
    targets = np.empty((len(ds), max_length), dtype="int32")
    longest = 0
    for start in range(0, len(ds), batch_size):
        end = min(start + batch_size, len(ds))
        # Slice the dataset before accessing the column: ds[text_column] can
        # materialize the entire corpus as Python strings.
        enc = _tokenize_texts(tokenizer, ds[start:end][text_column], max_length,
                              dynamic_padding=True)
        longest = max(longest, max((len(row) for row in enc["input_ids"]), default=0))
        enc = tokenizer.pad(enc, padding="max_length", max_length=max_length, return_tensors="np")
        batch_x, batch_y = _source_mlm_labels(
            enc, tokenizer, probability, ignore, seed=seed,
            source_ids=[f"{split}:{source_rows[i]}" for i in range(start, end)])
        if features is None:
            features = {key: np.empty((len(ds), max_length), dtype="int32") for key in batch_x}
        for key, value in batch_x.items():
            features[key][start:end] = value
        targets[start:end] = batch_y
        if len(ds) > batch_size and (start == 0 or end == len(ds) or (start // batch_size) % 100 == 0):
            print(f"Preparation: fill_mask {split} {end:,}/{len(ds):,} rows", flush=True)
    if features is None:
        features = {key: np.empty((0, max_length), dtype="int32")
                    for key in ("input_ids", "attention_mask")}
    return features, targets, longest


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

    from ..raw_text_axis import with_raw_text_lengths
    for split, dataset in (("train", ds_train), ("test", ds_test)):
        meta = with_raw_text_lengths(dataset, meta, split=split, column=text_column)

    tokenizer, _, _ = get_cached_tokenizer(
        hf_model_id=hf_model_id,
        task="fill_mask",
        device="cpu",
        transformers_module=transformers,
    )

    seed = int(meta.get("seed", 42))

    X_train, y_train, train_max_len = _prepare_split(
        ds_train, tokenizer, text_column, max_length, mlm_probability, label_pad_value,
        seed=seed, source_rows=meta.get("selected_train_rows", range(len(ds_train))),
        split=meta.get("train_split") or "train")
    X_test, y_test, test_max_len = _prepare_split(
        ds_test, tokenizer, text_column, max_length, mlm_probability, label_pad_value,
        seed=seed, source_rows=meta.get("selected_test_rows", range(len(ds_test))),
        split=meta.get("test_split") or "test")
    pad_to = max(train_max_len, test_max_len) if dynamic_padding else max_length
    if pad_to < max_length:
        # Preserve both left- and right-padding semantics without retokenizing.
        columns = slice(max_length - pad_to, None) if getattr(tokenizer, "padding_side", "right") == "left" else slice(0, pad_to)
        X_train = {key: value[:, columns] for key, value in X_train.items()}
        X_test = {key: value[:, columns] for key, value in X_test.items()}
        y_train, y_test = y_train[:, columns], y_test[:, columns]

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
        "masking_policy": "source_row_mask_v1",
        "masking_seed": seed,
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

    from ..tokenization_selection import filter_token_supervision
    return filter_token_supervision((X_train, y_train), (X_test, y_test), meta2)
