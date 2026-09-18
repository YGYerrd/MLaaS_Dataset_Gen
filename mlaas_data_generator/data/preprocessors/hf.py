from .hf_text_sequence import preprocess_hf_text_sequence
from pathlib import Path
from .hf_text_similarity import preprocess_hf_text_similarity
from .hf_text_fill_mask import preprocess_hf_text_fill_mask
from .hf_text_token import preprocess_hf_text_token
from .hf_text_generation import (
    preprocess_hf_text_causal_lm_generation,
    preprocess_hf_text_seq2seq_generation,
)
from .hf_image import preprocess_hf_image
from .hf_multimodal import preprocess_hf_multimodal
from ...hf_tasks import (
    HF_TASK_MODALITY,
    IMAGE_TASK_TYPES,
    normalize_hf_task,
    resolve_dataset_hf_task,
    resolve_hf_task_spec,
)


_MULTIMODAL_TASKS = frozenset(task for task, modality in HF_TASK_MODALITY.items() if modality == "multimodal" and task != "multimodal")
_IMAGE_TASKS = frozenset(IMAGE_TASK_TYPES)


_EXPECTED_BATCH_KEYS = {
    "sequence_classification": {"input_ids", "attention_mask"},
    "token_classification": {"input_ids", "attention_mask"},
    "sentence_similarity": {"input_ids", "attention_mask"},
    "fill_mask": {"input_ids", "attention_mask"},
    "causal_lm_generation": {"input_ids", "attention_mask"},
    "seq2seq_generation": {"input_ids", "attention_mask"},
    "image_classification": {"pixel_values"},
    "image_detection": {"pixel_values"},
    "image_segmentation": {"pixel_values"},
    "multimodal": {"input_ids", "attention_mask", "pixel_values"},
    "text_image_retrieval": {"input_ids", "attention_mask", "pixel_values"},
    "visual_question_answering": {"input_ids", "attention_mask", "pixel_values"},
    "image_captioning": {"input_ids", "attention_mask", "pixel_values"},
}

def _resolve_dataset_loader_template(meta, dataset_args):
    dataset_meta = meta.get("dataset_args") if isinstance(meta.get("dataset_args"), dict) else {}
    template = dataset_args.get("loader_template") or meta.get("loader_template") or dataset_meta.get("loader_template")
    return str(template).strip().lower() if template else None


def _resolve_text_hf_task(meta, dataset_args):
    return resolve_dataset_hf_task(
        loader_template=_resolve_dataset_loader_template(meta, dataset_args),
        hf_task=meta.get("hf_task", dataset_args.get("hf_task", "sequence_classification")),
        task_tag=meta.get("task_tag") or dataset_args.get("task_tag"),
        pipeline_tag=meta.get("pipeline_tag") or dataset_args.get("pipeline_tag"),
    )


def _validate_hf_preprocessor_output(train, test, meta):
    x_train, y_train = train
    x_test, y_test = test
    hf_task = str(meta.get("hf_task", "")).strip().lower().replace("-", "_")

    if not isinstance(x_train, dict) or not isinstance(x_test, dict):
        raise TypeError(f"HF task '{hf_task}' requires dict features; got {type(x_train)} / {type(x_test)}")

    expected = _EXPECTED_BATCH_KEYS.get(hf_task, {"input_ids", "attention_mask"})
    missing_train = sorted(expected - set(x_train.keys()))
    missing_test = sorted(expected - set(x_test.keys()))
    if missing_train or missing_test:
        raise ValueError(
            f"HF preprocessor output validation failed for task '{hf_task}'. "
            f"Missing train keys={missing_train}, test keys={missing_test}."
        )

    if y_train is None or y_test is None:
        raise ValueError(f"HF preprocessor output validation failed for task '{hf_task}': missing labels.")

    train_count = len(next(iter(x_train.values())))
    test_count = len(next(iter(x_test.values())))
    if train_count == 0 or test_count == 0:
        train_split = meta.get("train_split", "train")
        test_split = meta.get("test_split", "test")
        raise ValueError(
            f"HF preprocessor output validation failed for task '{hf_task}': "
            f"zero samples detected "
            f"(split='{train_split}', count={train_count}; split='{test_split}', count={test_count}). "
            f"Expected keys={sorted(expected)}."
        )
    if train_count != len(y_train) or test_count != len(y_test):
        raise ValueError(
            f"HF preprocessor output validation failed for task '{hf_task}': feature/label batch mismatch."
        )

    meta["x_keys"] = list(x_train.keys())
    from ..tokenization_selection import finalize_selected_accounting
    meta = finalize_selected_accounting(meta, train)
    return train, test, meta


def prepare_hf_candidates(selection, train, test, meta, dataset_args, prepare):
    """Fit cheap raw label metadata, then prepare only selected candidates."""
    task = _resolve_text_hf_task(meta, dataset_args)
    raw_labels = train[1]
    from ..raw_text_axis import AXIS, configured_skew_axis, with_raw_text_lengths
    if any(c.get("_recorded_source_ids") is None
           and (c.get("split_strategy") or c.get("distribution_type")) in {"dirichlet", "shard"}
           and configured_skew_axis(c) == AXIS for c in selection.configs):
        column = dataset_args.get("text_column", "text")
        print(f"Preparation: raw text length buckets | {len(train[0]):,} training rows", flush=True)
        meta = with_raw_text_lengths(train[0], {**meta, "text_column": column}, split="train", column=column)
    column = dataset_args.get("label_column", "label")
    if task == "image_classification" and column in train[0].column_names:
        import numpy as np
        raw_labels = np.asarray(train[0][column])
        meta = {**meta, "num_classes": int(np.max(raw_labels)) + 1}
    if task == "visual_question_answering":
        # Fitting an answer vocabulary needs only raw labels, never decoded
        # images or tokens. Keep that fit stable across replacement batches.
        from .hf_multimodal import _resolve_multimodal_columns, _coerce_vqa_answer
        _, _, _, answer_column = _resolve_multimodal_columns(task,
            dataset_args.get("image_column"), dataset_args.get("text_column"), dataset_args.get("label_column"),
            dataset_args.get("question_column"), dataset_args.get("answer_column"), dataset_args.get("ranking_label_column"))
        if answer_column in train[0].column_names:
            meta = {**meta, "_vqa_source_answers": [_coerce_vqa_answer(a) for a in train[0][answer_column]]}
    return selection.prepare((train[0], raw_labels), test, meta, prepare)


def preprocess_hf(train, test, meta, **dataset_args):
    from ..tokenization_selection import active_selection
    selection = active_selection()
    task = _resolve_text_hf_task(meta, dataset_args)
    # Scalar text labels are fitted by their existing preprocessors before the
    # row-preserving selection hook. All other tasks select raw candidates here.
    if selection is not None and task not in {"sequence_classification", "sentence_similarity"}:
        return prepare_hf_candidates(selection, train, test, meta, dataset_args,
                                     lambda tr, te, m: preprocess_hf(tr, te, m, **dataset_args))
    requested_modality = str(meta.get("modality", "text")).strip().lower()
    requested_task = meta.get("hf_task", dataset_args.get("hf_task", meta.get("task_type")))
    hf_task = normalize_hf_task(requested_task)
    canonical_modality = HF_TASK_MODALITY.get(hf_task)
    modality = requested_modality if requested_modality in {"image", "multimodal"} else (canonical_modality or requested_modality)

    hf_model_id = dataset_args.get("hf_model_id")
    if not hf_model_id:
        raise ValueError("HF preprocessing requires hf_model_id in dataset_args")
    hf_model_revision = dataset_args.get("hf_model_revision") or dataset_args.get("model_revision")
    if hf_model_revision and not Path(str(dataset_args.get("hf_model_id") or "")).is_dir():
        try:
            from huggingface_hub import snapshot_download
            hf_model_id = snapshot_download(repo_id=str(hf_model_id), revision=str(hf_model_revision))
        except Exception as exc:
            raise RuntimeError(f"Unable to resolve pinned Hugging Face model revision {dataset_args.get('hf_model_id')}@{hf_model_revision}") from exc
    meta["hf_model_revision"] = hf_model_revision
    from ..tokenization_selection import report_tokenization_policy
    report_tokenization_policy(meta, dataset_args)

    if modality == "image":
        if hf_task not in _IMAGE_TASKS:
            task_type = str(meta.get("task_type", "classification")).strip().lower()
            hf_task = normalize_hf_task(f"image_{task_type}")
        task_spec = resolve_hf_task_spec(hf_task)
        if task_spec.hf_task not in _IMAGE_TASKS:
            raise ValueError(f"Unsupported HF image task: {hf_task}")
        hf_task = task_spec.hf_task
        task_type = task_spec.task_type
        meta["hf_task"] = hf_task
        meta["modality"] = "image"
        meta["task_type"] = task_type
        out = preprocess_hf_image(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            image_column=dataset_args.get("image_column", "image"),
            label_column=dataset_args.get("label_column", "label"),
            boxes_column=dataset_args.get("boxes_column"),
            classes_column=dataset_args.get("classes_column"),
            mask_column=dataset_args.get("mask_column"),
            task_type=task_type,
            training_augmentations=dataset_args.get("training_augmentations", True),
            eval_augmentations=dataset_args.get("eval_augmentations", False),
            on_decode_error=dataset_args.get("on_decode_error", "skip"),
            report_decode_errors=dataset_args.get("report_decode_errors", True),
            **({"processor_size": dataset_args["processor_size"]} if dataset_args.get("processor_size") is not None else {}),
        )
        return _validate_hf_preprocessor_output(*out)

    if modality == "multimodal":
        if hf_task not in _MULTIMODAL_TASKS:
            hf_task = "multimodal"
        meta["hf_task"] = hf_task
        meta["modality"] = "multimodal"
        out = preprocess_hf_multimodal(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            hf_task=hf_task,
            image_column=dataset_args.get("image_column", "image"),
            text_column=dataset_args.get("text_column", "text"),
            label_column=dataset_args.get("label_column"),
            max_length=dataset_args.get("max_length", meta.get("max_length", 128)),
            missing_pair_handling=dataset_args.get("missing_pair_handling", "drop"),
            on_decode_error=dataset_args.get("on_decode_error", "skip"),
            report_decode_errors=dataset_args.get("report_decode_errors", True),
            question_column=dataset_args.get("question_column"),
            answer_column=dataset_args.get("answer_column"),
            ranking_label_column=dataset_args.get("ranking_label_column"),
            vqa_label_mode=dataset_args.get("vqa_label_mode", "auto"),
            vqa_answer_vocab_size=dataset_args.get("vqa_answer_vocab_size"),
            vqa_unseen_answer_policy=dataset_args.get("vqa_unseen_answer_policy", "ignore"),
        )
        return _validate_hf_preprocessor_output(*out)

    if modality != "text":
        raise NotImplementedError(f"HF modality '{modality}' not implemented")

    hf_task = _resolve_text_hf_task(meta, dataset_args)
    meta["hf_task"] = hf_task
    meta["modality"] = "text"

    if hf_task == "sequence_classification":
        out = preprocess_hf_text_sequence(
            train, test, meta,
            hf_model_id=hf_model_id,
            text_column=dataset_args.get("text_column", "text"),
            label_column=dataset_args.get("label_column", "label"),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "token_classification":
        out = preprocess_hf_text_token(
            train, test, meta,
            hf_model_id=hf_model_id,
            tokens_column=dataset_args.get("tokens_column") or dataset_args.get("text_column"),
            label_column=dataset_args.get("label_column"),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "sentence_similarity":
        from ...task_contract import resolve_sentence_pair_contract
        contract = resolve_sentence_pair_contract({**meta, **dataset_args})
        out = preprocess_hf_text_similarity(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            text_column=dataset_args.get("text_column", ["sentence1", "sentence2"]),
            label_column=dataset_args.get("label_column", "label"),
            label_mode=contract.get("label_mode", "auto"),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "fill_mask":
        out = preprocess_hf_text_fill_mask(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            text_column=dataset_args.get("text_column", "text"),
            mlm_probability=dataset_args.get("mlm_probability", 0.15),
            label_pad_value=dataset_args.get("label_pad_value", -100),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "causal_lm_generation":
        out = preprocess_hf_text_causal_lm_generation(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            column_mapping=dataset_args.get("column_mapping"),
            text_column=dataset_args.get("text_column"),
            label_column=dataset_args.get("label_column"),
            max_length=dataset_args.get("max_length"),
            source_max_length=dataset_args.get("source_max_length"),
            target_max_length=dataset_args.get("target_max_length"),
            prompt_loss_only=dataset_args.get("prompt_loss_only", True),
            ignore_index=dataset_args.get("label_pad_value", -100),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "seq2seq_generation":
        out = preprocess_hf_text_seq2seq_generation(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            column_mapping=dataset_args.get("column_mapping"),
            text_column=dataset_args.get("text_column"),
            label_column=dataset_args.get("label_column"),
            max_length=dataset_args.get("max_length"),
            source_max_length=dataset_args.get("source_max_length"),
            target_max_length=dataset_args.get("target_max_length"),
            ignore_index=dataset_args.get("label_pad_value", -100),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
        )
        return _validate_hf_preprocessor_output(*out)

    raise ValueError(f"Unsupported HF text task: {hf_task}")
