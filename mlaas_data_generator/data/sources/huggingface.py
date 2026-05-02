from ..accounting import append_accounting_stage, update_accounting
from ..hf_cache_paths import with_hf_image_decode_disabled
from ..multimodal_columns import resolve_existing_column


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


def _bool_arg(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def load_huggingface_source(**kwargs):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError(
            "Hugging Face dataset loading requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from e

    dataset_name = kwargs.get("dataset_name")
    if not dataset_name:
        raise ValueError("HF source requires dataset_name=<repo_id> in dataset_args.")

    dataset_config = kwargs.get("dataset_config", None)
    train_split = kwargs.get("train_split", "train")
    requested_test_split = kwargs.get("test_split", "test")

    max_samples = kwargs.get("max_samples", None)
    seed = int(kwargs.get("seed", 42))
    max_length = int(kwargs.get("max_length", 128))
    inference_only = _bool_arg(kwargs.get("inference_only", False))

    task_type = kwargs.get("task", "classification")
    modality = str(kwargs.get("modality", "text")).strip().lower()
    if modality not in {"text", "image", "multimodal"}:
        raise ValueError(f"Unsupported HF modality '{modality}'. Expected one of ['text', 'image', 'multimodal']")
    hf_task = kwargs.get("hf_task", "sequence_classification")

    ds_train = load_dataset(dataset_name, dataset_config, split=train_split)
    raw_train_count = len(ds_train)

    def _try_load_split(split_name):
        return load_dataset(dataset_name, dataset_config, split=split_name)

    def _is_unlabelled(ds):
        if modality == "multimodal":
            return False
        label_column = kwargs.get("label_column", "label")
        try:
            ys = ds[label_column]
        except Exception:
            return True
        if ys is None or len(ys) == 0:
            return True
        try:
            return all(int(v) == -1 for v in ys)
        except Exception:
            return False

    ds_test = None
    chosen_test_split = None
    raw_test_count = 0
    for candidate in [requested_test_split, "validation", "val", "dev"]:
        try:
            tmp = _try_load_split(candidate)
        except Exception:
            continue
        if _is_unlabelled(tmp):
            continue
        ds_test = tmp
        chosen_test_split = candidate
        raw_test_count = len(tmp)
        break

    if ds_test is None:
        test_size = float(kwargs.get("test_size", 0.2))
        split = ds_train.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        ds_train = split["train"]
        ds_test = split["test"]
        chosen_test_split = "train_test_split"
        raw_train_count = len(ds_train) + len(ds_test)
        raw_test_count = len(ds_test)

    post_split_train_count = len(ds_train)
    post_split_test_count = len(ds_test)

    if max_samples:
        n = int(max_samples)
        if len(ds_train) > 1:
            ds_train = ds_train.shuffle(seed=seed)
        if len(ds_test) > 1:
            ds_test = ds_test.shuffle(seed=seed + 1)
        ds_train = ds_train.select(range(min(n, len(ds_train))))
        test_n = n if inference_only else max(1, n // 5)
        ds_test = ds_test.select(range(min(test_n, len(ds_test))))

    schema = None
    def _has_value(value):
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict) and any(k in value for k in ("array", "bytes", "path")):
            return any(_has_value(value.get(k)) for k in ("array", "bytes", "path"))
        return True

    def _apply_pair_integrity(ds, *, split_name, image_column, text_column, policy):
        if policy not in {"drop", "error"}:
            raise ValueError("missing_pair_handling must be one of ['drop', 'error']")

        raw_ds = with_hf_image_decode_disabled(ds, image_column)
        original_count = len(raw_ds)
        image_values = raw_ds[image_column]
        text_values = raw_ds[text_column]

        has_image_mask = [_has_value(value) for value in image_values]
        has_text_mask = [_has_value(value) for value in text_values]

        valid_mask = [has_image and has_text for has_image, has_text in zip(has_image_mask, has_text_mask)]
        missing_pair_mask = [has_image != has_text for has_image, has_text in zip(has_image_mask, has_text_mask)]

        valid = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
        missing_pair_rows = [idx for idx, is_missing_pair in enumerate(missing_pair_mask) if is_missing_pair]

        if missing_pair_rows and policy == "error":
            raise ValueError(
                f"HF multimodal pair integrity failed for split='{split_name}': "
                f"{len(missing_pair_rows)} rows have missing image/text counterparts. "
                f"Set missing_pair_handling='drop' to filter invalid rows."
            )

        if missing_pair_rows and policy == "drop":
            raw_ds = raw_ds.select(valid)

        return raw_ds, {
            "split": split_name,
            "policy": policy,
            "dropped_rows": len(missing_pair_rows) if policy == "drop" else 0,
            "missing_pair_rows": len(missing_pair_rows),
            "aligned_pairs": len(valid),
            "original_rows": original_count,
            "output_rows": len(raw_ds),
        }

    if modality == "image":
        image_column = kwargs.get("image_column", "image")
        label_column = kwargs.get("label_column", "label")
        boxes_column = kwargs.get("boxes_column")
        classes_column = kwargs.get("classes_column")
        mask_column = kwargs.get("mask_column")
        if task_type == "segmentation" and not mask_column:
            mask_column = label_column

        train_cols = set(getattr(ds_train, "column_names", []) or [])
        if image_column not in train_cols:
            raise ValueError(
                f"HF image modality requires image_column '{image_column}' to exist in dataset '{dataset_name}'. "
                f"Available columns: {sorted(train_cols)}"
            )

        schema = {
            "image_column": image_column,
            "label_column": label_column if label_column in train_cols else None,
            "detection": {
                "boxes_column": boxes_column if boxes_column in train_cols else None,
                "classes_column": classes_column if classes_column in train_cols else None,
            },
            "segmentation": {
                "mask_column": mask_column if mask_column in train_cols else None,
            },
        }

        if task_type == "classification" and schema["label_column"] is None:
            raise ValueError(
                f"HF image classification requires label_column '{label_column}'. "
                f"Available columns: {sorted(train_cols)}"
            )

        if task_type == "detection":
            missing = [c for c in (boxes_column, classes_column) if c and c not in train_cols]
            if missing:
                raise ValueError(f"HF image detection requested columns not found: {missing}")

        if task_type == "segmentation":
            if not mask_column:
                raise ValueError(
                    f"HF image segmentation requires a mask column. "
                    f"Tried mask_column={kwargs.get('mask_column')} and label_column={label_column}."
                )
            if mask_column not in train_cols:
                raise ValueError(f"HF image segmentation mask_column '{mask_column}' not found in dataset")

    if modality == "multimodal":
        image_column = kwargs.get("image_column", "image")
        text_column = kwargs.get("text_column", "text")
        label_column = kwargs.get("label_column")
        missing_pair_handling = str(kwargs.get("missing_pair_handling", "drop")).strip().lower()

        train_columns = list(getattr(ds_train, "column_names", []) or [])
        train_cols = set(train_columns)
        image_column = resolve_existing_column(
            image_column,
            train_columns,
            aliases=_IMAGE_COLUMN_ALIASES,
        )
        text_column = resolve_existing_column(
            text_column,
            train_columns,
            aliases=_TEXT_COLUMN_ALIASES,
            numbered_alias_bases=("caption", "captions", "sentence", "sentences", "description", "descriptions"),
        )
        for required_col, name in ((image_column, "image_column"), (text_column, "text_column")):
            if required_col not in train_cols:
                raise ValueError(
                    f"HF multimodal modality requires {name} '{required_col}' to exist in dataset '{dataset_name}'. "
                    f"Available columns: {sorted(train_cols)}"
                )

        resolved_label_column = label_column
        if resolved_label_column not in train_cols:
            if str(hf_task).strip().lower().replace("-", "_") == "image_captioning":
                resolved_label_column = text_column
            else:
                resolved_label_column = None

        ds_train, train_pair_report = _apply_pair_integrity(
            ds_train,
            split_name="train",
            image_column=image_column,
            text_column=text_column,
            policy=missing_pair_handling,
        )
        ds_test, test_pair_report = _apply_pair_integrity(
            ds_test,
            split_name=chosen_test_split,
            image_column=image_column,
            text_column=text_column,
            policy=missing_pair_handling,
        )

        schema = {
            "image_column": image_column,
            "text_column": text_column,
            "label_column": resolved_label_column,
            "pair_validation": {
                "missing_pair_handling": missing_pair_handling,
                "train": train_pair_report,
                "test": test_pair_report,
            },
        }

    dataset_args = dict(kwargs)
    dataset_args.pop("preprocessors", None)

    meta = {
        "dataset_family": "hf",
        "hf_id": dataset_name,
        "hf_subset": dataset_config,
        "train_split": train_split,
        "test_split": chosen_test_split,
        "seed": seed,
        "max_length": max_length,
        "task_type": task_type,
        "modality": modality,
        "hf_task": hf_task,
        "inference_only": inference_only,
        "schema": schema,
        "loader_template": dataset_args.get("loader_template"),
        "dataset_args": dataset_args,
    }

    meta = update_accounting(meta, raw_record_count=raw_train_count, post_filter_record_count=len(ds_train))
    meta = append_accounting_stage(
        meta,
        stage="hf_source",
        split="train",
        raw_record_count=raw_train_count,
        post_filter_record_count=len(ds_train),
        emitted_record_count=len(ds_train),
        sequence_count=len(ds_train),
        metric_instance_count=len(ds_train),
    )
    meta = append_accounting_stage(
        meta,
        stage="hf_source",
        split="test",
        raw_record_count=raw_test_count,
        post_filter_record_count=len(ds_test),
        emitted_record_count=len(ds_test),
        sequence_count=len(ds_test),
        metric_instance_count=len(ds_test),
    )

    return (ds_train, None), (ds_test, None), meta
