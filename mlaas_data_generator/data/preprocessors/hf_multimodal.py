from ..accounting import append_accounting_stage, finalize_accounting
from ..hf_cache_paths import (
    load_image_from_bytes,
    load_image_from_path,
    with_hf_image_decode_disabled,
)
from ..multimodal_columns import resolve_existing_column
import numpy as np
from collections import Counter
import re


_IMAGE_COLUMN_ALIASES = ("image", "img", "images", "pixel_values")
_TEXT_COLUMN_ALIASES = (
    "text",
    "caption",
    "captions",
    "sentence",
    "sentences",
    "description",
    "descriptions",
    "question",
)

_TASK_COLUMN_DEFAULTS = {
    "visual_question_answering": {"text_column": "question", "label_column": "answer"},
    "text_image_retrieval": {"text_column": "caption", "label_column": None},
    "image_captioning": {"text_column": "caption", "label_column": None},
}


def _has_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict) and any(k in value for k in ("array", "bytes", "path")):
        return any(_has_value(value.get(k)) for k in ("array", "bytes", "path"))
    return True


def _is_pil_image(value):
    try:
        from PIL import Image
    except Exception:
        return False
    return isinstance(value, Image.Image)


def _coerce_image_input(value):
    if _is_pil_image(value):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("array") is not None:
            return np.asarray(value.get("array"))
        if value.get("bytes") is not None:
            return load_image_from_bytes(value.get("bytes"))
        if value.get("path"):
            return load_image_from_path(value.get("path"))
    if isinstance(value, (bytes, bytearray)):
        return load_image_from_bytes(value)
    if isinstance(value, str):
        return load_image_from_path(value)
    return value


def _infer_image_hw(value):
    if _is_pil_image(value):
        width, height = value.size
        return int(height), int(width)
    if isinstance(value, dict):
        height = value.get("height")
        width = value.get("width")
        if height is not None and width is not None:
            return int(height), int(width)
        if value.get("array") is not None:
            return _infer_image_hw(np.asarray(value.get("array")))
        if value.get("path"):
            try:
                image = load_image_from_path(value.get("path"))
                width, height = image.size
                return int(height), int(width)
            except Exception:
                return None, None
    arr = np.asarray(value) if hasattr(value, "__array__") or hasattr(value, "__array_interface__") else None
    if arr is not None and arr.ndim >= 2:
        return int(arr.shape[0]), int(arr.shape[1])
    return None, None


def _resolve_image_target_hw(image_processor):
    for attr in ("crop_size", "size"):
        size = getattr(image_processor, attr, None)
        if not isinstance(size, dict):
            continue
        height = size.get("height")
        width = size.get("width")
        if height and width:
            return int(height), int(width)
        shortest = size.get("shortest_edge")
        if shortest:
            edge = int(shortest)
            return edge, edge
    return None


def _resize_chw_float32(chw, target_hw):
    arr = np.asarray(chw, dtype=np.float32)
    if target_hw is None or arr.ndim != 3:
        return arr
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if arr.shape[1] == target_h and arr.shape[2] == target_w:
        return arr

    try:
        from PIL import Image
    except Exception:
        return np.resize(arr, (arr.shape[0], target_h, target_w)).astype(np.float32, copy=False)

    channels = []
    for channel in arr:
        im = Image.fromarray(channel.astype(np.float32), mode="F")
        im = im.resize((target_w, target_h), resample=Image.BILINEAR)
        channels.append(np.asarray(im, dtype=np.float32))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def _normalize_multimodal_pixel_array(pixel_values, *, target_hw=None):
    arr = np.asarray(pixel_values, dtype=np.float32)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] != 3 and arr.shape[-1] == 3:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(
            f"Multimodal image encoding must produce CHW with 3 channels, got shape={arr.shape}"
        )
    return _resize_chw_float32(arr, target_hw).astype(np.float32, copy=False)


def _stack_multimodal_pixel_values(pixel_values, *, target_hw=None):
    if not pixel_values:
        return np.zeros((0, 3, 0, 0), dtype=np.float32)

    resolved_target_hw = target_hw
    if resolved_target_hw is None:
        shape_counts = Counter(
            tuple(np.asarray(pix, dtype=np.float32).shape[-2:])
            for pix in pixel_values
            if np.asarray(pix).ndim >= 3
        )
        if shape_counts:
            resolved_target_hw = shape_counts.most_common(1)[0][0]

    normalized = [
        _normalize_multimodal_pixel_array(pix, target_hw=resolved_target_hw)
        for pix in pixel_values
    ]
    shapes = {tuple(arr.shape) for arr in normalized}
    if len(shapes) != 1:
        raise ValueError(
            "Multimodal pixel stack remains ragged after normalization: "
            + ", ".join(str(shape) for shape in sorted(shapes))
        )
    return np.stack(normalized, axis=0).astype(np.float32, copy=False)


def _pick_answer_text(values):
    cleaned = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    return max(cleaned, key=lambda v: (counts[v], -cleaned.index(v)))


def _coerce_vqa_answer(value):
    if value is None:
        return ""

    if isinstance(value, dict):
        for key in ("multiple_choice_answer", "answer", "answers", "label", "labels"):
            if key in value:
                return _coerce_vqa_answer(value.get(key))
        return _pick_answer_text(value.values())

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        answers = [_coerce_vqa_answer(item) for item in value]
        return _pick_answer_text(answers)

    return str(value).strip()


_VQA_ARTICLES = {"a", "an", "the"}


def _normalize_vqa_answer(text):
    txt = str(text or "").lower().strip()
    txt = re.sub(r"[^\w\s]", " ", txt)
    parts = [p for p in txt.split() if p not in _VQA_ARTICLES]
    return " ".join(parts)


def _is_placeholder_label(text):
    return bool(re.fullmatch(r"label[_\-\s]?\d+", str(text or "").strip().lower()))


def _config_label_maps(config):
    if config is None:
        return {}, {}

    raw_id2label = getattr(config, "id2label", None)
    raw_label2id = getattr(config, "label2id", None)

    id2label = {}
    if isinstance(raw_id2label, dict):
        for raw_idx, raw_label in raw_id2label.items():
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            label = str(raw_label)
            if label.strip():
                id2label[idx] = label

    if id2label and all(_is_placeholder_label(label) for label in id2label.values()):
        id2label = {}

    label2id = {}
    if isinstance(raw_label2id, dict):
        for raw_label, raw_idx in raw_label2id.items():
            label = str(raw_label)
            if _is_placeholder_label(label):
                continue
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            normalized = _normalize_vqa_answer(label)
            if normalized:
                label2id[normalized] = idx
                id2label.setdefault(idx, label)

    for idx, label in id2label.items():
        normalized = _normalize_vqa_answer(label)
        if normalized:
            label2id.setdefault(normalized, int(idx))

    return label2id, id2label


def _resolve_vqa_label_mode(requested_mode, *, config, hf_model_id):
    requested = str(requested_mode or "auto").strip().lower().replace("-", "_")
    if requested in {"class", "classes", "classification", "vqa_class_index"}:
        return "classification"
    if requested in {"generative", "generation", "token", "token_index", "vqa_token_index"}:
        return "generation"
    if requested not in {"", "auto"}:
        raise ValueError("vqa_label_mode must be one of ['auto', 'classification', 'generation']")

    model_type = str(getattr(config, "model_type", "") or "").strip().lower()
    model_id = str(hf_model_id or "").strip().lower()
    family_hint = f"{model_type} {model_id}"
    if "vilt" in family_hint:
        return "classification"
    if "blip" in family_hint or "git" in family_hint:
        return "generation"

    label2id, _ = _config_label_maps(config)
    if label2id:
        return "classification"
    return "answer_text"


def _build_vqa_train_vocab(answer_texts, *, max_vocab_size=None):
    counts = Counter(
        _normalize_vqa_answer(value)
        for value in np.asarray(answer_texts, dtype=object).reshape(-1)
    )
    counts.pop("", None)

    limit = _bounded_positive_int(max_vocab_size)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ranked = ranked[:limit]

    label2id = {label: idx for idx, (label, _) in enumerate(ranked)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label, counts


def _encode_vqa_class_labels(answer_texts, label2id, *, unseen_policy="ignore"):
    policy = str(unseen_policy or "ignore").strip().lower()
    if policy not in {"ignore", "error"}:
        raise ValueError("vqa_unseen_answer_policy must be one of ['ignore', 'error']")

    labels = []
    unseen = 0
    for value in np.asarray(answer_texts, dtype=object).reshape(-1):
        normalized = _normalize_vqa_answer(value)
        if normalized in label2id:
            labels.append(int(label2id[normalized]))
            continue
        unseen += 1
        if policy == "error":
            raise ValueError(f"VQA answer '{value}' is not present in the answer vocabulary")
        labels.append(-100)
    return np.asarray(labels, dtype=np.int64), unseen


def _encode_vqa_token_labels(tokenizer, answer_texts, *, max_length, ignore_index=-100):
    labels = []
    for value in np.asarray(answer_texts, dtype=object).reshape(-1):
        label_enc = tokenizer(
            str(value),
            truncation=True,
            padding="max_length",
            max_length=int(max_length),
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors=None,
        )
        label_ids = np.asarray(label_enc["input_ids"], dtype=np.int64)
        label_mask = np.asarray(label_enc["attention_mask"], dtype=np.int64)
        special_mask = np.asarray(label_enc.get("special_tokens_mask", np.zeros_like(label_ids)), dtype=np.int64)
        if label_ids.ndim == 2:
            label_ids = label_ids[0]
        if label_mask.ndim == 2:
            label_mask = label_mask[0]
        if special_mask.ndim == 2:
            special_mask = special_mask[0]
        row = label_ids.copy()
        row[(label_mask == 0) | (special_mask != 0)] = int(ignore_index)
        special_ids = getattr(tokenizer, "all_special_ids", None)
        if special_ids:
            row[np.isin(row, np.asarray(special_ids, dtype=np.int64))] = int(ignore_index)
        labels.append(row)
    return np.asarray(labels, dtype=np.int64)


def _bounded_positive_int(value):
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0 or parsed >= 1_000_000:
        return None
    return parsed


def _config_text_length_limits(config):
    limits = []
    configs = [config]
    for attr in ("text_config", "encoder", "decoder"):
        nested = getattr(config, attr, None)
        if nested is not None:
            configs.append(nested)

    for cfg in configs:
        for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            candidate = _bounded_positive_int(getattr(cfg, attr, None))
            if candidate is not None:
                limits.append(candidate)
    return limits


def _resolve_text_max_length(tokenizer, hf_model_id, requested_max_length, auto_config_cls=None, config=None):
    requested = _bounded_positive_int(requested_max_length) or 128
    limits = []

    tokenizer_limit = _bounded_positive_int(getattr(tokenizer, "model_max_length", None))
    if tokenizer_limit is not None:
        limits.append(tokenizer_limit)

    if config is not None:
        limits.extend(_config_text_length_limits(config))
    elif auto_config_cls is not None:
        try:
            config = auto_config_cls.from_pretrained(hf_model_id)
            limits.extend(_config_text_length_limits(config))
        except Exception:
            pass

    model_limit = min(limits) if limits else None
    effective = min(requested, model_limit) if model_limit is not None else requested
    return effective, requested, model_limit


def _load_auto_tokenizer(AutoTokenizer, hf_model_id):
    try:
        return AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(hf_model_id, use_fast=False)


def _validate_pair_alignment(ds, *, image_column, text_column, split_name, missing_pair_handling):
    valid_indices = []
    missing_pairs = []
    decode_error_rows = []

    for idx in range(len(ds)):
        try:
            row = ds[idx]
        except Exception:
            decode_error_rows.append(idx)
            continue
        has_image = _has_value(row.get(image_column))
        has_text = _has_value(row.get(text_column))
        if has_image and has_text:
            valid_indices.append(idx)
            continue
        if has_image != has_text:
            missing_pairs.append(idx)

    if missing_pairs and missing_pair_handling == "error":
        raise ValueError(
            f"HF multimodal split '{split_name}' has {len(missing_pairs)} rows with broken image/text pairs. "
            "Use missing_pair_handling='drop' to filter rows with missing counterparts."
        )

    filtered = ds if not missing_pairs or missing_pair_handling == "error" else ds.select(valid_indices)
    report = {
        "split": split_name,
        "missing_pair_rows": len(missing_pairs),
        "decode_error_rows": len(decode_error_rows),
        "aligned_rows": len(valid_indices),
        "output_rows": len(filtered),
        "policy": missing_pair_handling,
    }
    return filtered, report


def _encode_split(
    ds,
    *,
    hf_task,
    tokenizer,
    image_processor,
    image_column,
    text_column,
    label_column,
    max_length,
    split_name="split",
    on_decode_error="skip",
    report_decode_errors=False,
):
    input_ids = []
    attention_masks = []
    pixel_values = []
    labels = []
    caption_lengths = []
    image_sizes = []
    decode_errors = []
    image_target_hw = _resolve_image_target_hw(image_processor)

    def _process_image(image_value):
        candidate_kwargs = [
            {"return_tensors": None, "do_resize": True, "do_normalize": True},
            {"return_tensors": None},
            {},
        ]
        last_err = None
        image_input = _coerce_image_input(image_value)
        for kwargs in candidate_kwargs:
            try:
                return image_processor(image_input, **kwargs)
            except TypeError as e:
                last_err = e
                continue
            except ValueError as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        return image_processor(image_input)

    for idx in range(len(ds)):
        try:
            row = ds[idx]
            text_val = row.get(text_column)
            image_val = row.get(image_column)

            if not _has_value(text_val) or image_val is None:
                continue
            image_h, image_w = _infer_image_hw(image_val)

            text_enc = tokenizer(
                str(text_val),
                truncation=True,
                padding="max_length",
                max_length=int(max_length),
                return_attention_mask=True,
                return_tensors=None,
            )
            image_enc = _process_image(image_val)

            ids = np.asarray(text_enc["input_ids"], dtype=np.int64)
            mask = np.asarray(text_enc["attention_mask"], dtype=np.int64)
            pix = image_enc.get("pixel_values", image_enc)

            if ids.ndim == 2:
                ids = ids[0]
            if mask.ndim == 2:
                mask = mask[0]
            pix = _normalize_multimodal_pixel_array(pix, target_hw=image_target_hw)

            row_label = None
            if hf_task == "image_captioning":
                label_text = row.get(label_column) if label_column is not None else text_val
                if not _has_value(label_text):
                    label_text = text_val
                label_enc = tokenizer(
                    str(label_text),
                    truncation=True,
                    padding="max_length",
                    max_length=int(max_length),
                    return_attention_mask=True,
                    return_tensors=None,
                )
                label_ids = np.asarray(label_enc["input_ids"], dtype=np.int64)
                label_mask = np.asarray(label_enc["attention_mask"], dtype=np.int64)
                if label_ids.ndim == 2:
                    label_ids = label_ids[0]
                if label_mask.ndim == 2:
                    label_mask = label_mask[0]
                row_label = label_ids.copy()
                row_label[label_mask == 0] = -100
            elif label_column is not None:
                row_label = row.get(label_column)
                if hf_task == "visual_question_answering":
                    row_label = _coerce_vqa_answer(row_label)
        except Exception as e:
            decode_errors.append({"index": idx, "error": str(e)})
            if on_decode_error == "raise":
                raise
            continue

        input_ids.append(ids)
        attention_masks.append(mask)
        pixel_values.append(pix)
        if hf_task == "text_image_retrieval":
            caption_lengths.append(len(str(text_val).split()))
            image_sizes.append([image_h or 0, image_w or 0])
        if hf_task == "image_captioning":
            labels.append(row_label)
        elif label_column is not None:
            labels.append(row_label)

    if not input_ids:
        preview = "; ".join(f"row {item['index']}: {item['error']}" for item in decode_errors[:3])
        suffix = f" First errors: {preview}" if preview else ""
        raise ValueError(f"HF multimodal split '{split_name}' produced zero valid examples after decoding.{suffix}")

    x = {
        "input_ids": np.asarray(input_ids, dtype=np.int64),
        "attention_mask": np.asarray(attention_masks, dtype=np.int64),
        "pixel_values": _stack_multimodal_pixel_values(pixel_values, target_hw=image_target_hw),
    }
    if hf_task == "text_image_retrieval":
        x["caption_lengths"] = np.asarray(caption_lengths, dtype=np.int64)
        x["image_sizes"] = np.asarray(image_sizes, dtype=np.int64)
    y = np.asarray(labels, dtype=object) if labels and hf_task == "visual_question_answering" else (
        np.asarray(labels) if labels else np.zeros((len(input_ids),), dtype=np.int64)
    )

    if len(x["input_ids"]) != len(x["pixel_values"]):
        raise ValueError("Multimodal alignment check failed: token and image batch lengths differ")

    if label_column is not None and len(y) != len(x["input_ids"]):
        raise ValueError("Multimodal alignment check failed: label length does not match paired inputs")

    report = {
        "split": split_name,
        "policy": on_decode_error,
        "failed": len(decode_errors),
        "survived": len(input_ids),
    }
    if report_decode_errors:
        report["errors"] = decode_errors

    return x, y, len(input_ids), report


def _resolve_multimodal_columns(
    hf_task,
    image_column,
    text_column,
    label_column,
    question_column,
    answer_column,
    ranking_label_column,
):
    task = str(hf_task or "multimodal").strip().lower().replace("-", "_")
    defaults = _TASK_COLUMN_DEFAULTS.get(task, {})

    resolved_text_column = text_column
    resolved_label_column = label_column

    if task == "visual_question_answering":
        resolved_text_column = question_column or (
            defaults.get("text_column", "question") if text_column in {None, "", "text"} else text_column
        )
        resolved_label_column = answer_column or (
            defaults.get("label_column", "answer") if label_column in {None, "", "label"} else label_column
        )
    elif task == "text_image_retrieval":
        resolved_text_column = text_column or defaults.get("text_column", "text")
        resolved_label_column = ranking_label_column
    elif task == "image_captioning":
        resolved_text_column = text_column or defaults.get("text_column", "text")
        resolved_label_column = label_column

    return task, image_column, resolved_text_column, resolved_label_column


def preprocess_hf_multimodal(
    train,
    test,
    meta,
    *,
    hf_model_id,
    hf_task="multimodal",
    image_column="image",
    text_column="text",
    label_column=None,
    max_length=128,
    missing_pair_handling="drop",
    on_decode_error="skip",
    report_decode_errors=False,
    question_column=None,
    answer_column=None,
    ranking_label_column=None,
    vqa_label_mode="auto",
    vqa_answer_vocab_size=None,
    vqa_unseen_answer_policy="ignore",
):
    try:
        from transformers import AutoTokenizer, AutoImageProcessor
        try:
            from transformers import AutoConfig
        except Exception:
            AutoConfig = None
    except Exception as e:
        raise ImportError("HF multimodal preprocessing requires transformers[vision]") from e

    policy = str(missing_pair_handling or "drop").strip().lower()
    if policy not in {"drop", "error"}:
        raise ValueError("missing_pair_handling must be one of ['drop', 'error']")
    decode_policy = str(on_decode_error or "skip").strip().lower()
    if decode_policy not in {"skip", "raise", "report"}:
        raise ValueError("on_decode_error must be one of ['skip', 'raise', 'report']")

    ds_train, _ = train
    ds_test, _ = test

    hf_task, image_column, text_column, label_column = _resolve_multimodal_columns(
        hf_task,
        image_column,
        text_column,
        label_column,
        question_column,
        answer_column,
        ranking_label_column,
    )
    train_columns = list(getattr(ds_train, "column_names", []) or [])
    train_column_set = set(train_columns)
    if train_columns:
        text_column = resolve_existing_column(
            text_column,
            train_columns,
            aliases=_TEXT_COLUMN_ALIASES,
            numbered_alias_bases=("caption", "captions", "sentence", "sentences", "description", "descriptions"),
        )
        image_column = resolve_existing_column(image_column, train_columns, aliases=_IMAGE_COLUMN_ALIASES)
        if hf_task == "image_captioning":
            label_column = resolve_existing_column(
                label_column,
                train_columns,
                aliases=(text_column,),
                numbered_alias_bases=("caption", "captions", "sentence", "sentences", "description", "descriptions"),
            )
            if label_column not in train_column_set:
                label_column = text_column
        elif label_column is not None:
            label_column = resolve_existing_column(label_column, train_columns)
            if label_column not in train_column_set:
                label_column = None

    ds_train = with_hf_image_decode_disabled(ds_train, image_column)
    ds_test = with_hf_image_decode_disabled(ds_test, image_column)

    model_config = None
    if AutoConfig is not None:
        try:
            model_config = AutoConfig.from_pretrained(hf_model_id)
        except Exception:
            model_config = None

    tokenizer = _load_auto_tokenizer(AutoTokenizer, hf_model_id)
    image_processor = AutoImageProcessor.from_pretrained(hf_model_id)
    text_max_length, requested_max_length, model_text_max_length = _resolve_text_max_length(
        tokenizer,
        hf_model_id,
        max_length,
        auto_config_cls=AutoConfig,
        config=model_config,
    )

    ds_train, train_report = _validate_pair_alignment(
        ds_train,
        image_column=image_column,
        text_column=text_column,
        split_name="train",
        missing_pair_handling=policy,
    )
    ds_test, test_report = _validate_pair_alignment(
        ds_test,
        image_column=image_column,
        text_column=text_column,
        split_name="test",
        missing_pair_handling=policy,
    )

    x_train, y_train, train_survived, train_decode_report = _encode_split(
        ds_train,
        hf_task=hf_task,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_column=image_column,
        text_column=text_column,
        label_column=label_column,
        max_length=text_max_length,
        split_name="train",
        on_decode_error=decode_policy,
        report_decode_errors=bool(report_decode_errors),
    )
    x_test, y_test, test_survived, test_decode_report = _encode_split(
        ds_test,
        hf_task=hf_task,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_column=image_column,
        text_column=text_column,
        label_column=label_column,
        max_length=text_max_length,
        split_name="test",
        on_decode_error=decode_policy,
        report_decode_errors=bool(report_decode_errors),
    )

    vqa_meta = {}
    if hf_task == "visual_question_answering":
        if bool(meta.get("inference_only")):
            resolved_vqa_mode = "answer_text"
        else:
            resolved_vqa_mode = _resolve_vqa_label_mode(
                vqa_label_mode,
                config=model_config,
                hf_model_id=hf_model_id,
            )
        vqa_meta["vqa_label_mode"] = resolved_vqa_mode
        vqa_meta["vqa_unseen_answer_policy"] = str(vqa_unseen_answer_policy or "ignore").strip().lower()

        if resolved_vqa_mode == "classification":
            model_label2id, model_id2label = _config_label_maps(model_config)
            model_train_labels, model_train_unseen = (
                _encode_vqa_class_labels(y_train, model_label2id, unseen_policy="ignore")
                if model_label2id
                else (None, len(y_train))
            )
            use_model_vocab = bool(model_label2id) and model_train_labels is not None and int(model_train_unseen) < len(y_train)
            if use_model_vocab:
                label2id = model_label2id
                id2label = model_id2label
                vocab_source = "model_config"
            else:
                label2id, id2label, answer_counts = _build_vqa_train_vocab(
                    y_train,
                    max_vocab_size=vqa_answer_vocab_size,
                )
                vocab_source = "train_split"
                vqa_meta["vqa_answer_frequency_top"] = dict(answer_counts.most_common(10))

            y_train, train_unseen = _encode_vqa_class_labels(
                y_train,
                label2id,
                unseen_policy=vqa_unseen_answer_policy,
            )
            y_test, test_unseen = _encode_vqa_class_labels(
                y_test,
                label2id,
                unseen_policy=vqa_unseen_answer_policy,
            )
            num_labels = (max(id2label) + 1) if id2label else len(label2id)
            vqa_meta.update(
                {
                    "label_format": "vqa_class_index",
                    "num_classes": int(num_labels),
                    "num_labels": int(num_labels),
                    "label2id": {str(k): int(v) for k, v in label2id.items()},
                    "id2label": {str(int(k)): str(v) for k, v in id2label.items()},
                    "vqa_answer_vocab_source": vocab_source,
                    "vqa_answer_vocab_size": int(len(label2id)),
                    "vqa_train_unseen_answer_count": int(train_unseen),
                    "vqa_test_unseen_answer_count": int(test_unseen),
                }
            )
        elif resolved_vqa_mode == "generation":
            y_train = _encode_vqa_token_labels(tokenizer, y_train, max_length=text_max_length, ignore_index=-100)
            y_test = _encode_vqa_token_labels(tokenizer, y_test, max_length=text_max_length, ignore_index=-100)
            vqa_meta.update(
                {
                    "label_format": "vqa_token_index",
                    "ignore_index": -100,
                    "label_pad_value": -100,
                    "num_classes": int(getattr(tokenizer, "vocab_size", 0) or 0),
                    "num_labels": int(getattr(tokenizer, "vocab_size", 0) or 0),
                }
            )
        else:
            vqa_meta.update({"label_format": "answer_text"})

    meta = append_accounting_stage(
        meta,
        stage="hf_multimodal",
        split="train",
        input_record_count=train_report.get("aligned_rows", len(ds_train)),
        post_filter_record_count=train_survived,
        tokenized_record_count=train_survived,
        emitted_record_count=train_survived,
        sequence_count=train_survived,
        metric_instance_count=len(y_train),
    )
    meta = append_accounting_stage(
        meta,
        stage="hf_multimodal",
        split="test",
        input_record_count=test_report.get("aligned_rows", len(ds_test)),
        post_filter_record_count=test_survived,
        tokenized_record_count=test_survived,
        emitted_record_count=test_survived,
        sequence_count=test_survived,
        metric_instance_count=len(y_test),
    )
    meta = finalize_accounting(meta)
    inferred_input_shape = ()
    pixel_values = x_train.get("pixel_values") if isinstance(x_train, dict) else None
    if pixel_values is not None and len(pixel_values) > 0:
        inferred_input_shape = tuple(getattr(pixel_values[0], "shape", ()))
    label_format = vqa_meta.get("label_format")
    if not label_format:
        if hf_task == "image_captioning":
            label_format = "token_index"
        elif hf_task == "text_image_retrieval":
            label_format = "paired_rank"
        else:
            label_format = "single_index"
    meta.update(
        {
            "input_shape": inferred_input_shape,
            "modality": "multimodal",
            "hf_task": hf_task,
            "hf_processor": hf_model_id,
            "max_length": int(text_max_length),
            "requested_max_length": int(requested_max_length),
            "model_text_max_length": (
                int(model_text_max_length) if model_text_max_length is not None else None
            ),
            "max_length_adjusted": bool(text_max_length != requested_max_length),
            "image_column": image_column,
            "text_column": text_column,
            "label_column": label_column,
            "label_format": label_format,
            "retrieval_positive_policy": (
                "diagonal_in_batch" if hf_task == "text_image_retrieval" else None
            ),
            "x_keys": ["input_ids", "attention_mask", "pixel_values"],
            **vqa_meta,
            "schema": {
                "image_column": image_column,
                "text_column": text_column,
                "label_column": label_column,
                "task": hf_task,
                "label_format": label_format,
                "vqa": vqa_meta if hf_task == "visual_question_answering" else None,
                "retrieval_positive_policy": (
                    "diagonal_in_batch" if hf_task == "text_image_retrieval" else None
                ),
                "text_max_length": int(text_max_length),
                "requested_max_length": int(requested_max_length),
                "model_text_max_length": (
                    int(model_text_max_length) if model_text_max_length is not None else None
                ),
                "max_length_adjusted": bool(text_max_length != requested_max_length),
                "pair_validation": {
                    "missing_pair_handling": policy,
                    "train": train_report,
                    "test": test_report,
                },
                "decode_report": {
                    "on_decode_error": decode_policy,
                    "train": train_decode_report,
                    "test": test_decode_report,
                },
                "batch_contract": {
                    "text_keys": ["input_ids", "attention_mask"],
                    "image_keys": ["pixel_values"],
                    "combined_keys": ["input_ids", "attention_mask", "pixel_values"],
                },
            },
        }
    )

    return (x_train, y_train), (x_test, y_test), meta
