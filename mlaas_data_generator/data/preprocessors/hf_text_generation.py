import json
import time

from ..accounting import append_accounting_stage, finalize_accounting

import numpy as np


_GENERATION_BATCH_ROWS = 32


def _encode_generation_split(ds, encode_batch, *, widths, pad_id, ignore_index, dynamic_padding, split, accepted_positions=None):
    """Read/tokenize bounded raw batches and avoid corpus-sized Python token lists."""
    count = len(ds)
    ids = np.empty((count, widths[0]), dtype="int32")
    masks = np.empty_like(ids)
    labels = np.empty((count, widths[1]), dtype="int32")
    written = source_width = target_width = 0
    started = last = time.monotonic()
    print(f"Preparing generation {split}: 0/{count:,} rows", flush=True)
    for start in range(0, count, _GENERATION_BATCH_ROWS):
        stop = min(start + _GENERATION_BATCH_ROWS, count)
        if hasattr(ds, "select"):
            # Select before reading columns: long documents stay in Arrow until needed.
            batch = ds.select(range(start, stop))
        else:
            batch = {column: ds[column][start:stop] for column in ds.column_names}
        try:
            encoded = encode_batch(batch)
        except ValueError as exc:
            if not str(exc).startswith("insufficient_supervision:"):
                raise
            encoded = None
        if encoded is not None:
            features, targets = encoded[:2]
            if accepted_positions is not None:
                positions = encoded[-1]
                accepted_positions.extend(start + int(i) for i in positions)
            size = len(targets)
            sw, tw = features["input_ids"].shape[1], targets.shape[1]
            end = written + size
            ids[written:end].fill(pad_id)
            masks[written:end].fill(0)
            labels[written:end].fill(ignore_index)
            ids[written:end, :sw] = features["input_ids"]
            masks[written:end, :sw] = features["attention_mask"]
            labels[written:end, :tw] = targets
            written = end
            source_width, target_width = max(source_width, sw), max(target_width, tw)
        now = time.monotonic()
        if start == 0 or stop == count or now - last >= 10:
            print(f"Preparing generation {split}: {stop:,}/{count:,} rows | elapsed={now-started:.1f}s", flush=True)
            last = now
    if not written:
        raise ValueError(f"insufficient_supervision: generation {split} split has no targets")
    sw, tw = (source_width, target_width) if dynamic_padding else widths
    return {"input_ids": ids[:written, :sw], "attention_mask": masks[:written, :sw]}, labels[:written, :tw], sw, tw


# Seq2seq auto-mapping priority is ordered from the most explicit generic schema
# names to common summarization/translation aliases. The first match wins so the
# behavior stays deterministic whenever multiple plausible columns are present.
SEQ2SEQ_SOURCE_CANDIDATES = [
    "source_text",
    "input",
    "prompt",
    "instruction",
    "text",
    "source",
    "article",
    "document",
    "dialogue",
    "report",
    "context",
    "question",
    "src",
    "input_text",
    "email_body",
]

SEQ2SEQ_TARGET_CANDIDATES = [
    "target_text",
    "label",
    "output",
    "completion",
    "response",
    "target",
    "highlights",
    "summary",
    "abstract",
    "answer",
    "subject",
    "subject_line",
    "tldr",
    "tgt",
    "output_text",
]


def _first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def _normalize_explicit_column(value, columns):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            if item in columns:
                return item
        return None
    return value if value in columns else None


def resolve_generation_columns(ds_train, *, column_mapping=None, text_column=None, label_column=None, hf_task="causal_lm_generation"):
    cols = set(ds_train.column_names)
    mapping = dict(column_mapping or {})
    explicit_text_col = _normalize_explicit_column(text_column, cols)
    explicit_label_col = _normalize_explicit_column(label_column, cols)

    if hf_task == "seq2seq_generation":
        source_col = mapping.get("source") or explicit_text_col or _first_existing(cols, SEQ2SEQ_SOURCE_CANDIDATES)
        target_col = mapping.get("target") or explicit_label_col or _first_existing(cols, SEQ2SEQ_TARGET_CANDIDATES)
        mode = "source_target"
    else:
        prompt_col = mapping.get("prompt") or explicit_text_col or _first_existing(cols, [
            "prompt", "instruction", "input", "source_text", "text", "source",
        ])
        target_col = mapping.get("target") or explicit_label_col or _first_existing(cols, [
            "completion", "response", "output", "target_text", "label", "target",
        ])
        source_col = prompt_col
        mode = "prompt_target"

        if explicit_text_col and explicit_label_col and explicit_text_col == explicit_label_col:
            source_col = explicit_text_col
            target_col = explicit_label_col
            mode = "single_text"
        elif target_col is None:
            single_text_col = _first_existing(cols, [
                "text", "content", "document", "body", "passage", "article",
            ])
            if single_text_col is None and len(cols) == 1:
                single_text_col = next(iter(cols))
            if single_text_col is not None:
                source_col = single_text_col
                target_col = single_text_col
                mode = "single_text"

    if not source_col:
        raise ValueError(
            f"Could not resolve source/prompt column for task '{hf_task}'. "
            f"Provide dataset_args.column_mapping. Available columns: {sorted(cols)}"
        )
    if not target_col:
        raise ValueError(
            f"Could not resolve target column for task '{hf_task}'. "
            f"Provide dataset_args.column_mapping. Available columns: {sorted(cols)}"
        )

    return {"source": source_col, "target": target_col, "mode": mode}


def _to_text_list(values):
    out = []
    for v in values:
        if v is None:
            out.append("")
        elif isinstance(v, (list, tuple)):
            out.append(" ".join(str(item) for item in v if item is not None))
        else:
            out.append(str(v))
    return out


def _strip_trailing_eos(token_ids, tokenizer):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return list(token_ids)
    trimmed = list(token_ids)
    if trimmed and int(trimmed[-1]) == int(eos_id):
        trimmed = trimmed[:-1]
    return trimmed


def _pad_encodings(enc, pad_to):
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    padded_ids = np.zeros((len(input_ids), pad_to), dtype="int32")
    padded_mask = np.zeros_like(padded_ids)
    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        width = min(len(ids), pad_to)
        padded_ids[i, :width] = ids[:width]
        padded_mask[i, :width] = mask[:width]
    return padded_ids, padded_mask


def _supervised_token_count(labels, ignore_index):
    return sum(int(np.count_nonzero(labels[start:start + 1024] != int(ignore_index)))
               for start in range(0, len(labels), 1024))


def _dataset_size(ds, fallback_column):
    try:
        return int(len(ds))
    except Exception:
        return int(len(ds[fallback_column]))


def _load_auto_tokenizer(hf_model_id):
    from ..runtime_cache import cached_preparation
    return cached_preparation("tokenizer", hf_model_id,
                              lambda: _load_auto_tokenizer_uncached(hf_model_id))[0]


def _load_auto_tokenizer_uncached(hf_model_id):
    import transformers

    fast_error = None
    try:
        return transformers.AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    except Exception as exc:
        fast_error = exc

    slow_error = None
    try:
        return transformers.AutoTokenizer.from_pretrained(hf_model_id, use_fast=False)
    except Exception as exc:
        slow_error = exc

    model_id = str(hf_model_id or "").strip().lower()
    added_token_error = "addedtoken" in str(fast_error).lower() or "addedtoken" in str(slow_error).lower()
    if model_id.startswith("salesforce/codet5-") and added_token_error:
        return _load_codet5_roberta_tokenizer(transformers, hf_model_id)
    if slow_error is not None:
        raise slow_error
    raise fast_error


def _sanitize_token_spec(value):
    if isinstance(value, dict):
        content = value.get("content")
        return content if content is not None else None
    if isinstance(value, list):
        return [item for item in (_sanitize_token_spec(item) for item in value) if item is not None]
    return value


def _load_codet5_roberta_tokenizer(transformers, hf_model_id):
    from huggingface_hub import hf_hub_download

    roberta_tokenizer = getattr(transformers, "RobertaTokenizer", None)
    if roberta_tokenizer is None:
        raise ValueError(f"RobertaTokenizer is unavailable for {hf_model_id}")

    vocab_file = hf_hub_download(hf_model_id, "vocab.json")
    merges_file = hf_hub_download(hf_model_id, "merges.txt")
    tokenizer_kwargs = {}
    try:
        tokenizer_config_path = hf_hub_download(hf_model_id, "tokenizer_config.json")
        with open(tokenizer_config_path, "r", encoding="utf-8") as handle:
            tokenizer_config = json.load(handle)
        tokenizer_kwargs["add_prefix_space"] = bool(tokenizer_config.get("add_prefix_space", False))
        model_max_length = tokenizer_config.get("model_max_length")
        if model_max_length is not None:
            tokenizer_kwargs["model_max_length"] = int(model_max_length)
        for key in ("unk_token", "bos_token", "eos_token", "sep_token", "cls_token", "pad_token", "mask_token"):
            token_value = _sanitize_token_spec(tokenizer_config.get(key))
            if token_value is not None:
                tokenizer_kwargs[key] = token_value
    except Exception:
        tokenizer_config = {}

    tokenizer = roberta_tokenizer(vocab_file=vocab_file, merges_file=merges_file, **tokenizer_kwargs)

    try:
        special_tokens_map_path = hf_hub_download(hf_model_id, "special_tokens_map.json")
        with open(special_tokens_map_path, "r", encoding="utf-8") as handle:
            special_tokens_map = json.load(handle)
    except Exception:
        special_tokens_map = {}

    additional_special_tokens = _sanitize_token_spec(special_tokens_map.get("additional_special_tokens"))
    if additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": list(additional_special_tokens)})
    return tokenizer


def preprocess_hf_text_causal_lm_generation(
    train,
    test,
    meta,
    *,
    hf_model_id,
    column_mapping=None,
    text_column=None,
    label_column=None,
    max_length=None,
    source_max_length=None,
    target_max_length=None,
    prompt_loss_only=True,
    ignore_index=-100,
    dynamic_padding=False,
):
    ds_train, _ = train
    ds_test, _ = test

    tokenizer = _load_auto_tokenizer(hf_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    resolved = resolve_generation_columns(
        ds_train,
        column_mapping=column_mapping,
        text_column=text_column,
        label_column=label_column,
        hf_task="causal_lm_generation",
    )
    prompt_col, target_col = resolved["source"], resolved["target"]
    generation_mode = resolved.get("mode", "prompt_target")

    max_len = int(max_length or meta.get("max_length", 512))
    src_max = int(source_max_length or max_len)
    tgt_max = int(target_max_length or max_len)

    def _encode_batch(ds):
        pad_id = int(tokenizer.pad_token_id)
        eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id)

        ids_list, mask_list, label_list, row_positions = [], [], [], []
        if generation_mode == "single_text":
            texts = _to_text_list(ds[prompt_col])
            tok = tokenizer(texts, truncation=True, padding=False, max_length=max_len, add_special_tokens=True)

            for position, (text, ids) in enumerate(zip(texts, tok["input_ids"])):
                if not text.strip():
                    continue
                full_ids = _strip_trailing_eos(ids, tokenizer)[:max_len]
                if not full_ids:
                    full_ids = [eos_id]
                if eos_id is not None and full_ids:
                    if len(full_ids) < max_len and full_ids[-1] != eos_id:
                        full_ids = full_ids + [eos_id]
                    elif full_ids[-1] != eos_id and len(full_ids) == max_len:
                        full_ids[-1] = eos_id
                labels = full_ids.copy()
                ids_list.append(full_ids)
                mask_list.append([1] * len(full_ids))
                label_list.append(labels)
                row_positions.append(position)
        else:
            prompts = _to_text_list(ds[prompt_col])
            targets = _to_text_list(ds[target_col])

            p_tok = tokenizer(prompts, truncation=True, padding=False, max_length=src_max, add_special_tokens=True)
            t_tok = tokenizer(targets, truncation=True, padding=False, max_length=tgt_max, add_special_tokens=False)

            for position, (p_ids, t_ids) in enumerate(zip(p_tok["input_ids"], t_tok["input_ids"])):
                prompt_ids = _strip_trailing_eos(p_ids, tokenizer)
                full_ids = (prompt_ids + list(t_ids) + [eos_id])[:max_len]
                labels = full_ids.copy()
                if prompt_loss_only:
                    prompt_len = min(len(prompt_ids), len(full_ids))
                    labels[:prompt_len] = [int(ignore_index)] * prompt_len

                ids_list.append(full_ids)
                mask_list.append([1] * len(full_ids))
                label_list.append(labels)
                row_positions.append(position)

        # Causal models shift labels by one position. An EOS-only blank row or
        # a prompt that fills the entire context has no next-token supervision.
        supervised_rows = [
            (ids, mask, labels, position)
            for ids, mask, labels, position in zip(ids_list, mask_list, label_list, row_positions)
            if any(label != int(ignore_index) for label in labels[1:])
        ]
        if not supervised_rows:
            raise ValueError("insufficient_supervision: causal LM split has no next-token targets")
        ids_list, mask_list, label_list, row_positions = map(list, zip(*supervised_rows))
        pad_to = max(len(x) for x in ids_list) if dynamic_padding else max_len
        padded_ids, padded_mask, padded_labels = [], [], []
        for ids, mask, labels in zip(ids_list, mask_list, label_list):
            ids = ids[:pad_to]
            mask = mask[:pad_to]
            labels = labels[:pad_to]
            pad_len = max(0, pad_to - len(ids))
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_mask.append(mask + [0] * pad_len)
            padded_labels.append(labels + [int(ignore_index)] * pad_len)

        X = {
            "input_ids": np.asarray(padded_ids, dtype="int32"),
            "attention_mask": np.asarray(padded_mask, dtype="int32"),
        }
        y = np.asarray(padded_labels, dtype="int32")
        return X, y, pad_to, row_positions

    accepted = {"train": [], "evaluation": []}
    cache = meta.get("_preparation_cache", {})
    def _encode_split(ds, split):
        if split == "evaluation" and "causal_evaluation" in cache:
            result, positions = cache["causal_evaluation"]
            accepted[split].extend(positions)
            return dict(result[0]), result[1], result[2]
        x, y, width, _ = _encode_generation_split(ds, _encode_batch,
            widths=(max_len, max_len), pad_id=int(tokenizer.pad_token_id), ignore_index=ignore_index,
            dynamic_padding=dynamic_padding, split=split, accepted_positions=accepted[split])
        result = x, y, width
        if split == "evaluation":
            cache["causal_evaluation"] = ((dict(x), y, width), list(accepted[split]))
        return result

    X_train, y_train, train_len = _encode_split(ds_train, "train")
    X_test, y_test, test_len = _encode_split(ds_test, "evaluation")

    pad_len = max(train_len, test_len) if dynamic_padding else max_len
    if dynamic_padding and train_len != test_len:
        # keep shapes aligned for downstream tensorization consistency
        X_train["input_ids"], X_train["attention_mask"] = _pad_encodings(X_train, pad_len)
        X_test["input_ids"], X_test["attention_mask"] = _pad_encodings(X_test, pad_len)
        y_train = np.pad(y_train, ((0, 0), (0, pad_len - y_train.shape[1])), constant_values=int(ignore_index))
        y_test = np.pad(y_test, ((0, 0), (0, pad_len - y_test.shape[1])), constant_values=int(ignore_index))

    meta2 = dict(meta)
    meta2["accepted_train_positions"] = accepted["train"]
    from ..tokenization_selection import select_metadata
    for split, key in (("train", "train"), ("evaluation", "test")):
        meta2 = select_metadata(meta2, accepted[split], len(ds_train if key == "train" else ds_test), split=key)
    meta2.update({
        "hf_task": "causal_lm_generation",
        "hf_model_id": hf_model_id,
        "x_format": "dict",
        "x_keys": list(X_train.keys()),
        "input_shape": (pad_len,),
        "label_granularity": "token",
        "modality": "text",
        "num_classes": int(meta.get("num_classes", 1) or 1),
        "column_mapping": (
            {"text": prompt_col}
            if generation_mode == "single_text"
            else {"prompt": prompt_col, "target": target_col}
        ),
        "generation_mode": generation_mode,
        "prompt_loss_only": bool(prompt_loss_only),
        "ignore_index": int(ignore_index),
        "source_max_length": src_max,
        "target_max_length": tgt_max,
        "dynamic_padding": bool(dynamic_padding),
        "pad_token_id": int(tokenizer.pad_token_id),
        "padding_side": str(getattr(tokenizer, "padding_side", "right")),
    })
    train_input_count = _dataset_size(ds_train, prompt_col)
    test_input_count = _dataset_size(ds_test, prompt_col)
    train_sequences = int(X_train["input_ids"].shape[0])
    test_sequences = int(X_test["input_ids"].shape[0])
    meta2 = append_accounting_stage(
        meta2,
        stage="hf_text_generation",
        split="train",
        input_record_count=train_input_count,
        post_filter_record_count=train_sequences,
        tokenized_record_count=train_sequences,
        emitted_record_count=train_sequences,
        sequence_count=train_sequences,
        supervised_token_count=_supervised_token_count(y_train, ignore_index),
        metric_instance_count=train_sequences,
    )
    meta2 = append_accounting_stage(
        meta2,
        stage="hf_text_generation",
        split="test",
        input_record_count=test_input_count,
        post_filter_record_count=test_sequences,
        tokenized_record_count=test_sequences,
        emitted_record_count=test_sequences,
        sequence_count=test_sequences,
        supervised_token_count=_supervised_token_count(y_test, ignore_index),
        metric_instance_count=test_sequences,
    )
    meta2 = finalize_accounting(meta2)
    return (X_train, y_train), (X_test, y_test), meta2


def preprocess_hf_text_seq2seq_generation(
    train,
    test,
    meta,
    *,
    hf_model_id,
    column_mapping=None,
    text_column=None,
    label_column=None,
    max_length=None,
    source_max_length=None,
    target_max_length=None,
    ignore_index=-100,
    dynamic_padding=False,
):
    ds_train, _ = train
    ds_test, _ = test

    tokenizer = _load_auto_tokenizer(hf_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    resolved = resolve_generation_columns(
        ds_train,
        column_mapping=column_mapping,
        text_column=text_column,
        label_column=label_column,
        hf_task="seq2seq_generation",
    )
    source_col, target_col = resolved["source"], resolved["target"]

    max_len = int(max_length or meta.get("max_length", 512))
    src_max = int(source_max_length or max_len)
    tgt_max = int(target_max_length or max_len)

    def _encode_batch(ds):
        src = _to_text_list(ds[source_col])
        tgt = _to_text_list(ds[target_col])

        source_enc = tokenizer(src, truncation=True, padding=False, max_length=src_max, return_attention_mask=True)
        target_enc = tokenizer(text_target=tgt, truncation=True, padding=False, max_length=tgt_max, return_attention_mask=False)

        src_pad_to = max(len(x) for x in source_enc["input_ids"]) if dynamic_padding else src_max
        tgt_pad_to = max(len(x) for x in target_enc["input_ids"]) if dynamic_padding else tgt_max

        pad_id = int(tokenizer.pad_token_id)

        X_ids, X_mask, y = [], [], []
        for s_ids, s_mask, t_ids in zip(source_enc["input_ids"], source_enc["attention_mask"], target_enc["input_ids"]):
            s_ids = list(s_ids)[:src_pad_to]
            s_mask = list(s_mask)[:src_pad_to]
            t_ids = list(t_ids)[:tgt_pad_to]

            s_pad = max(0, src_pad_to - len(s_ids))
            t_pad = max(0, tgt_pad_to - len(t_ids))

            X_ids.append(s_ids + [pad_id] * s_pad)
            X_mask.append(s_mask + [0] * s_pad)
            y.append(t_ids + [int(ignore_index)] * t_pad)

        X = {
            "input_ids": np.asarray(X_ids, dtype="int32"),
            "attention_mask": np.asarray(X_mask, dtype="int32"),
        }
        labels = np.asarray(y, dtype="int32")
        return X, labels, src_pad_to, tgt_pad_to, list(range(len(src)))

    def _encode_split(ds, split):
        return _encode_generation_split(ds, _encode_batch,
            widths=(src_max, tgt_max), pad_id=int(tokenizer.pad_token_id), ignore_index=ignore_index,
            dynamic_padding=dynamic_padding, split=split)

    X_train, y_train, train_src_len, train_tgt_len = _encode_split(ds_train, "train")
    X_test, y_test, test_src_len, test_tgt_len = _encode_split(ds_test, "evaluation")

    src_pad_to = max(train_src_len, test_src_len) if dynamic_padding else src_max
    tgt_pad_to = max(train_tgt_len, test_tgt_len) if dynamic_padding else tgt_max
    if dynamic_padding:
        if X_train["input_ids"].shape[1] != src_pad_to:
            X_train["input_ids"], X_train["attention_mask"] = _pad_encodings(X_train, src_pad_to)
        if X_test["input_ids"].shape[1] != src_pad_to:
            X_test["input_ids"], X_test["attention_mask"] = _pad_encodings(X_test, src_pad_to)

        if y_train.shape[1] != tgt_pad_to:
            y_train = np.pad(y_train, ((0, 0), (0, tgt_pad_to - y_train.shape[1])), constant_values=int(ignore_index))
        if y_test.shape[1] != tgt_pad_to:
            y_test = np.pad(y_test, ((0, 0), (0, tgt_pad_to - y_test.shape[1])), constant_values=int(ignore_index))

    meta2 = dict(meta)
    meta2.update({
        "hf_task": "seq2seq_generation",
        "hf_model_id": hf_model_id,
        "x_format": "dict",
        "x_keys": list(X_train.keys()),
        "input_shape": (src_pad_to,),
        "label_granularity": "token",
        "modality": "text",
        "num_classes": int(meta.get("num_classes", 1) or 1),
        "column_mapping": {"source": source_col, "target": target_col},
        "ignore_index": int(ignore_index),
        "source_max_length": src_max,
        "target_max_length": tgt_max,
        "dynamic_padding": bool(dynamic_padding),
        "target_shape": (tgt_pad_to,),
    })

    train_input_count = _dataset_size(ds_train, source_col)
    test_input_count = _dataset_size(ds_test, source_col)
    train_sequences = int(X_train["input_ids"].shape[0])
    test_sequences = int(X_test["input_ids"].shape[0])
    meta2 = append_accounting_stage(
        meta2,
        stage="hf_text_generation",
        split="train",
        input_record_count=train_input_count,
        post_filter_record_count=train_input_count,
        tokenized_record_count=train_sequences,
        emitted_record_count=train_sequences,
        sequence_count=train_sequences,
        supervised_token_count=_supervised_token_count(y_train, ignore_index),
        metric_instance_count=train_sequences,
    )
    meta2 = append_accounting_stage(
        meta2,
        stage="hf_text_generation",
        split="test",
        input_record_count=test_input_count,
        post_filter_record_count=test_input_count,
        tokenized_record_count=test_sequences,
        emitted_record_count=test_sequences,
        sequence_count=test_sequences,
        supervised_token_count=_supervised_token_count(y_test, ignore_index),
        metric_instance_count=test_sequences,
    )
    meta2 = finalize_accounting(meta2)
    return (X_train, y_train), (X_test, y_test), meta2
