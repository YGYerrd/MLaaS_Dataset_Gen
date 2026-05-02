from ..accounting import append_accounting_stage, finalize_accounting
from ..hf_cache_paths import resolve_hf_cache_path, with_hf_columns_decode_disabled
import io
import json
import os
import inspect
import logging
import re

import numpy as np

LOGGER = logging.getLogger(__name__)


def _is_pil_image(value):
    try:
        from PIL import Image
    except Exception:
        return False
    return isinstance(value, Image.Image)


_DETECTION_LABEL_ALIASES = {
    "tv": ("tv monitor",),
    "couch": ("sofa",),
    "cell phone": ("mobile phone",),
    "hair drier": ("hair dryer",),
}


def _normalize_label_name(name):
    txt = str(name or "").strip().lower()
    txt = re.sub(r"[^a-z0-9]+", " ", txt).strip()
    return txt


def _to_numpy_rgb(image_like):
    if image_like is None:
        raise ValueError("image is None")

    if isinstance(image_like, np.ndarray):
        arr = image_like
    elif isinstance(image_like, dict):
        if "array" in image_like and image_like["array"] is not None:
            arr = np.asarray(image_like["array"])
        elif "bytes" in image_like and image_like["bytes"] is not None:
            data = image_like["bytes"]
            arr = _decode_bytes(data)
        elif "path" in image_like and image_like["path"]:
            arr = _decode_path(image_like["path"])
        else:
            raise ValueError("unsupported HF image dict payload")
    elif isinstance(image_like, (bytes, bytearray)):
        arr = _decode_bytes(image_like)
    elif isinstance(image_like, str):
        arr = _decode_path(image_like)
    elif hasattr(image_like, "__array__") or hasattr(image_like, "__array_interface__"):
        arr = np.asarray(image_like)
    else:
        raise TypeError(f"unsupported image payload type={type(image_like)}")

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"expected HWC image with ndim=3, got shape={arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"expected channel-last with 1/3/4 channels, got shape={arr.shape}")

    return arr


def _decode_path(path):
    path = resolve_hf_cache_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        from PIL import Image
    except Exception as e:
        raise ImportError("Image decoding from path requires Pillow") from e
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _decode_bytes(data):
    try:
        from PIL import Image
    except Exception as e:
        raise ImportError("Image decoding from bytes requires Pillow") from e
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def _normalise_dataset_id(dataset_id):
    if not dataset_id:
        return ""
    return str(dataset_id).strip().lower().split("@", 1)[0]


def _decode_dataset_specific_rgb_mask(arr, *, dataset_id=None):
    dataset_id = _normalise_dataset_id(dataset_id)
    if dataset_id != "qubvel-hf/ade20k-mini" or arr.ndim != 3:
        return arr

    # qubvel ADE20K mini annotations encode class id in R and instance id in G.
    if arr.shape[-1] in {3, 4}:
        return np.asarray(arr[..., 0], dtype=np.int64)
    if arr.shape[0] in {3, 4}:
        return np.asarray(arr[0], dtype=np.int64)
    return arr


def _to_numpy_mask(mask_like, *, dataset_id=None):
    if mask_like is None:
        raise ValueError("segmentation mask is missing")

    if isinstance(mask_like, np.ndarray):
        arr = mask_like
    elif isinstance(mask_like, dict):
        if "array" in mask_like and mask_like["array"] is not None:
            arr = np.asarray(mask_like["array"])
        elif "bytes" in mask_like and mask_like["bytes"] is not None:
            try:
                from PIL import Image
            except Exception as e:
                raise ImportError("Segmentation mask decoding from bytes requires Pillow") from e
            with Image.open(io.BytesIO(mask_like["bytes"])) as im:
                arr = np.asarray(im)
        elif "path" in mask_like and mask_like["path"]:
            try:
                from PIL import Image
            except Exception as e:
                raise ImportError("Segmentation mask decoding from path requires Pillow") from e
            with Image.open(resolve_hf_cache_path(mask_like["path"])) as im:
                arr = np.asarray(im)
        else:
            raise ValueError("unsupported HF segmentation mask dict payload")
    elif isinstance(mask_like, (bytes, bytearray)):
        try:
            from PIL import Image
        except Exception as e:
            raise ImportError("Segmentation mask decoding from bytes requires Pillow") from e
        with Image.open(io.BytesIO(mask_like)) as im:
            arr = np.asarray(im)
    elif isinstance(mask_like, str):
        resolved = resolve_hf_cache_path(mask_like)
        if not os.path.exists(resolved):
            raise FileNotFoundError(mask_like)
        try:
            from PIL import Image
        except Exception as e:
            raise ImportError("Segmentation mask decoding from path requires Pillow") from e
        with Image.open(resolved) as im:
            arr = np.asarray(im)
    elif _is_pil_image(mask_like):
        arr = np.asarray(mask_like)
    elif hasattr(mask_like, "__array__") or hasattr(mask_like, "__array_interface__"):
        arr = np.asarray(mask_like)
    else:
        raise TypeError(f"unsupported segmentation mask payload type={type(mask_like)}")

    arr = np.asarray(arr)
    arr = _decode_dataset_specific_rgb_mask(arr, dataset_id=dataset_id)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"expected 2D segmentation mask, got shape={arr.shape}")

    arr = arr.astype(np.int64, copy=False)
    if arr.size:
        unique = np.unique(arr)
        if unique.size <= 2 and set(int(v) for v in unique.tolist()).issubset({0, 255}):
            arr = (arr > 0).astype(np.int64, copy=False)

    return arr


def _sample_segmentation_mask_range(ds, mask_column, *, limit=None, dataset_id=None):
    if not mask_column:
        return None, None

    observed_min = None
    observed_max = None
    sample_count = len(ds) if limit is None else min(int(limit), len(ds))
    for idx in range(sample_count):
        try:
            arr = _to_numpy_mask(ds[idx].get(mask_column), dataset_id=dataset_id)
        except Exception:
            continue
        if arr.size == 0:
            continue
        current_min = int(np.min(arr))
        current_max = int(np.max(arr))
        observed_min = current_min if observed_min is None else min(observed_min, current_min)
        observed_max = current_max if observed_max is None else max(observed_max, current_max)

    return observed_min, observed_max


def _infer_segmentation_num_classes(labels, *, ignore_index=None):
    observed_max = None
    for item in labels or []:
        arr = np.asarray(item)
        if arr.size == 0:
            continue
        if ignore_index is not None:
            arr = arr[arr != int(ignore_index)]
        if arr.size == 0:
            continue
        current_max = int(np.max(arr))
        observed_max = current_max if observed_max is None else max(observed_max, current_max)

    if observed_max is None:
        return None
    return int(observed_max + 1)


def _normalise_detection_item(boxes, classes):
    if boxes is None:
        boxes = []
    if classes is None:
        classes = []
    out_boxes = np.asarray(boxes, dtype=np.float32)
    out_classes = np.asarray(classes, dtype=np.int64)
    if out_boxes.size == 0:
        out_boxes = np.zeros((0, 4), dtype=np.float32)
    elif out_boxes.ndim == 1:
        if out_boxes.shape[0] != 4:
            raise ValueError("detection boxes must be Nx4")
        out_boxes = out_boxes.reshape(1, 4)
    elif out_boxes.ndim != 2 or out_boxes.shape[1] != 4:
        raise ValueError(f"detection boxes must be Nx4, got shape={out_boxes.shape}")

    if out_classes.ndim == 0:
        out_classes = out_classes.reshape(1)
    else:
        out_classes = out_classes.reshape(-1)

    if out_boxes.shape[0] != out_classes.shape[0]:
        n = min(int(out_boxes.shape[0]), int(out_classes.shape[0]))
        out_boxes = out_boxes[:n]
        out_classes = out_classes[:n]
    return {"boxes": out_boxes, "classes": out_classes}


def _decode_detection_annotation_payload(annotation):
    if not isinstance(annotation, str):
        return annotation

    text = annotation.strip()
    if not text or text[0] not in "[{":
        return annotation
    try:
        return json.loads(text)
    except Exception:
        return annotation


def _extract_detection_records(records):
    boxes = []
    classes = []
    candidate_boxes_keys = ("boxes", "bbox", "bboxes")
    candidate_classes_keys = ("classes", "class_labels", "labels", "label", "category", "category_id", "category_ids")

    for record in records:
        if not isinstance(record, dict):
            continue
        box = None
        for key in candidate_boxes_keys:
            if key in record:
                box = record.get(key)
                break
        cls = None
        for key in candidate_classes_keys:
            if key in record:
                cls = record.get(key)
                break
        if box is None or cls is None:
            continue
        boxes.append(box)
        classes.append(cls)

    return _normalise_detection_item(boxes, classes)


def _extract_detection_annotations(row, *, label_column=None, boxes_column=None, classes_column=None):
    boxes = row.get(boxes_column) if boxes_column else None
    classes = row.get(classes_column) if classes_column else None

    if boxes is not None or classes is not None:
        return _normalise_detection_item(boxes, classes)

    annotation = row.get(label_column) if label_column else None
    annotation = _decode_detection_annotation_payload(annotation)
    if isinstance(annotation, (list, tuple)):
        return _extract_detection_records(annotation)
    if not isinstance(annotation, dict):
        return _normalise_detection_item([], [])

    for container_key in ("objects", "annotations", "targets"):
        nested = annotation.get(container_key)
        nested = _decode_detection_annotation_payload(nested)
        if isinstance(nested, (list, tuple)):
            return _extract_detection_records(nested)
        if isinstance(nested, dict):
            annotation = nested
            break

    candidate_boxes_keys = ("boxes", "bbox", "bboxes")
    candidate_classes_keys = ("classes", "class_labels", "labels", "label", "category", "category_id", "category_ids")

    extracted_boxes = None
    for key in candidate_boxes_keys:
        if key in annotation:
            extracted_boxes = annotation.get(key)
            break

    extracted_classes = None
    for key in candidate_classes_keys:
        if key in annotation:
            extracted_classes = annotation.get(key)
            break

    return _normalise_detection_item(extracted_boxes, extracted_classes)


def _infer_detection_box_format(boxes, *, image_h, image_w):
    out_boxes = np.asarray(boxes, dtype=np.float32)
    if out_boxes.size == 0:
        return None
    if out_boxes.ndim == 1:
        if out_boxes.shape[0] != 4:
            return None
        out_boxes = out_boxes.reshape(1, 4)
    if out_boxes.ndim != 2 or out_boxes.shape[1] != 4:
        return None

    tol_w = float(image_w) * 1.05
    tol_h = float(image_h) * 1.05

    x1, y1, x2_or_w, y2_or_h = out_boxes.T

    xyxy_violations = (
        (x1 < 0).sum()
        + (y1 < 0).sum()
        + (x2_or_w < x1).sum()
        + (y2_or_h < y1).sum()
        + (x2_or_w > tol_w).sum()
        + (y2_or_h > tol_h).sum()
    )

    xywh_violations = (
        (x1 < 0).sum()
        + (y1 < 0).sum()
        + (x2_or_w < 0).sum()
        + (y2_or_h < 0).sum()
        + ((x1 + x2_or_w) > tol_w).sum()
        + ((y1 + y2_or_h) > tol_h).sum()
    )

    if int(xyxy_violations) < int(xywh_violations):
        return "xyxy"
    if int(xywh_violations) < int(xyxy_violations):
        return "xywh"

    # Ambiguous absolute coordinates: default to xywh because many HF object
    # detection datasets publish [x, y, width, height].
    return "xywh"


def _is_likely_contiguous_zero_based_detection_labels(labels):
    all_classes = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        classes = item.get("classes")
        if classes is None:
            continue
        arr = np.asarray(classes, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            continue
        all_classes.append(arr)

    if not all_classes:
        return False

    cls = np.concatenate(all_classes, axis=0)
    unique = np.unique(cls)
    if unique.size == 0 or int(unique[0]) != 0:
        return False

    # Allow a small amount of sparsity in sampled subsets while still
    # recognizing zero-based contiguous id spaces (e.g. COCO 0..79).
    max_id = int(unique[-1])
    uniq_count = int(unique.size)
    slack = max(10, int(np.ceil(0.15 * max(1, uniq_count))))
    return max_id <= int((uniq_count - 1) + slack)


def _extract_detection_category_names(ds, label_column):
    features = getattr(ds, "features", None) or {}
    label_feature = features.get(label_column) if isinstance(features, dict) else None
    if label_feature is None:
        return None

    nested = getattr(label_feature, "feature", None)
    if isinstance(nested, dict):
        for key in ("category", "category_id", "category_ids", "class_labels", "labels"):
            category_feature = nested.get(key)
            names = getattr(category_feature, "names", None)
            if names:
                return [str(name) for name in names]
    names = getattr(label_feature, "names", None)
    if names:
        return [str(name) for name in names]
    return None


def _build_detection_class_id_map(*, hf_model_id, category_names):
    if not category_names:
        return None
    try:
        from transformers import AutoConfig
    except Exception:
        return None

    try:
        cfg = AutoConfig.from_pretrained(hf_model_id)
        id2label = getattr(cfg, "id2label", None)
    except Exception:
        return None

    if not isinstance(id2label, dict) or not id2label:
        return None

    model_name_to_ids = {}
    for raw_id, raw_name in id2label.items():
        try:
            mid = int(raw_id)
        except Exception:
            continue
        normalized = _normalize_label_name(raw_name)
        if not normalized or normalized in {"n a", "na"}:
            continue
        model_name_to_ids.setdefault(normalized, []).append(mid)

    resolved = []
    for name in category_names:
        normalized = _normalize_label_name(name)
        candidates = [normalized]
        candidates.extend(_normalize_label_name(alias) for alias in _DETECTION_LABEL_ALIASES.get(normalized, ()))
        matched_id = None
        for candidate in candidates:
            ids = model_name_to_ids.get(candidate) or []
            if len(ids) == 1:
                matched_id = int(ids[0])
                break
        if matched_id is None:
            return None
        resolved.append(matched_id)
    return np.asarray(resolved, dtype=np.int64)


def _process_split(
    ds,
    *,
    split_name,
    image_processor,
    image_column,
    task_type,
    training,
    label_column=None,
    boxes_column=None,
    classes_column=None,
    mask_column=None,
    dataset_id=None,
    on_decode_error="skip",
    report_decode_errors=False,
):
    if task_type == "detection":
        LOGGER.info(
            "[detection preprocessing] entering _process_split split=%s len=%d training=%s",
            split_name,
            len(ds),
            bool(training),
        )
    images = []
    labels = []
    decode_errors = []
    processor_call = getattr(image_processor, "__call__", None)
    accepts_kwargs = False
    accepted_params = set()
    if callable(processor_call):
        try:
            sig = inspect.signature(processor_call)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            accepted_params = set(sig.parameters)
        except (TypeError, ValueError):
            accepts_kwargs = True
    preprocess_call = getattr(image_processor, "preprocess", None)
    preprocess_accepts_kwargs = False
    preprocess_params = set()
    if callable(preprocess_call):
        try:
            preprocess_sig = inspect.signature(preprocess_call)
            preprocess_accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in preprocess_sig.parameters.values()
            )
            preprocess_params = set(preprocess_sig.parameters)
        except (TypeError, ValueError):
            preprocess_accepts_kwargs = True

    def _build_processor_kwargs(*, training_enabled):
        base_kwargs = {
            "return_tensors": None,
            "do_resize": True,
            "do_normalize": True,
        }
        supports_do_augment = ("do_augment" in accepted_params) or ("do_augment" in preprocess_params)
        if supports_do_augment:
            base_kwargs["do_augment"] = bool(training_enabled)

        if not accepts_kwargs and accepted_params:
            candidate_kwargs = {k: v for k, v in base_kwargs.items() if k in accepted_params}
        elif accepts_kwargs and preprocess_params and not preprocess_accepts_kwargs:
            candidate_kwargs = {k: v for k, v in base_kwargs.items() if k in preprocess_params}
        elif accepts_kwargs and preprocess_params:
            candidate_kwargs = {k: v for k, v in base_kwargs.items() if k in preprocess_params}
        else:
            candidate_kwargs = dict(base_kwargs)

        call_attempts = [dict(candidate_kwargs)]
        if "do_augment" in candidate_kwargs:
            call_attempts.append({k: v for k, v in candidate_kwargs.items() if k != "do_augment"})
        call_attempts.extend(
            [
                {k: v for k, v in candidate_kwargs.items() if k in {"return_tensors"}},
                {},
            ]
        )

        return call_attempts

    def _process_image(image_array, *, training_enabled):
        call_attempts = _build_processor_kwargs(training_enabled=training_enabled)
        last_err = None
        for kwargs in call_attempts:
            try:
                return image_processor(image_array, **kwargs)
            except TypeError as e:
                last_err = e
                continue
            except ValueError as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        return image_processor(image_array)

    def _process_segmentation(image_array, mask_array, *, training_enabled):
        call_attempts = _build_processor_kwargs(training_enabled=training_enabled)
        last_err = None

        for kwargs in call_attempts:
            try:
                return image_processor([image_array], segmentation_maps=[mask_array], **kwargs)
            except TypeError as e:
                last_err = e
            except ValueError as e:
                last_err = e
            try:
                return image_processor(images=[image_array], segmentation_maps=[mask_array], **kwargs)
            except TypeError as e:
                last_err = e
            except ValueError as e:
                last_err = e

        if last_err is not None:
            raise last_err
        return image_processor([image_array], segmentation_maps=[mask_array])

    for idx in range(len(ds)):
        if task_type == "detection" and (idx == 0 or idx % 25 == 0):
            LOGGER.info(
                "[detection preprocessing] split=%s progress idx=%d/%d",
                split_name,
                idx,
                len(ds),
            )
        try:
            row = ds[idx]
            if task_type == "detection":
                LOGGER.info("[detection preprocessing] split=%s idx=%d before image decode", split_name, idx)
            image = _to_numpy_rgb(row.get(image_column))
            if task_type == "detection":
                LOGGER.info(
                    "[detection preprocessing] split=%s idx=%d after image decode shape=%s",
                    split_name,
                    idx,
                    tuple(getattr(image, "shape", ())),
                )
                LOGGER.info("[detection preprocessing] split=%s idx=%d before processor call", split_name, idx)
            if task_type == "segmentation":
                mask = _to_numpy_mask(row.get(mask_column), dataset_id=dataset_id)
                proc = _process_segmentation(image, mask, training_enabled=training)
                pix = proc.get("pixel_values", proc)
                label = proc.get("labels", mask)
            else:
                proc = _process_image(image, training_enabled=training)
                pix = proc.get("pixel_values", proc)
            if task_type == "detection":
                LOGGER.info("[detection preprocessing] split=%s idx=%d after processor call", split_name, idx)
                LOGGER.info(
                    "[detection preprocessing] split=%s idx=%d before np.asarray(pixel_values)",
                    split_name,
                    idx,
                )
            pix = np.asarray(pix, dtype=np.float32)
            if pix.ndim == 4:
                pix = pix[0]
            if pix.ndim != 3:
                raise ValueError(f"processor output must be CHW/HWC 3D, got {pix.shape}")
            if pix.shape[0] != 3 and pix.shape[-1] == 3:
                pix = np.transpose(pix, (2, 0, 1))
            if pix.shape[0] != 3:
                raise ValueError(f"processor output must have 3 channels, got {pix.shape}")

            if task_type == "classification":
                label = int(row.get(label_column))
            elif task_type == "detection":
                LOGGER.info(
                    "[detection preprocessing] split=%s idx=%d before np.asarray(boxes/classes)",
                    split_name,
                    idx,
                )
                label = _extract_detection_annotations(
                    row,
                    label_column=label_column,
                    boxes_column=boxes_column,
                    classes_column=classes_column,
                )
                # Preserve the raw decoded image size so downstream box
                # normalization stays aligned with dataset-native coordinates
                # even when the processor resizes/pads pixel values.
                label["image_size"] = np.asarray([image.shape[0], image.shape[1]], dtype=np.int64)
                label["box_format"] = _infer_detection_box_format(
                    label.get("boxes", []),
                    image_h=image.shape[0],
                    image_w=image.shape[1],
                )
            elif task_type == "segmentation":
                label = np.asarray(label)
                if label.ndim == 3 and label.shape[0] == 1:
                    label = label[0]
                elif label.ndim == 3 and label.shape[-1] == 1:
                    label = label[..., 0]
                if label.ndim != 2:
                    raise ValueError(f"segmentation labels must be 2D after preprocessing, got {label.shape}")
                label = label.astype(np.int64, copy=False)
            else:
                label = None
        except Exception as e:
            if task_type == "detection":
                LOGGER.exception(
                    "[detection preprocessing] split=%s idx=%d failed during preprocessing",
                    split_name,
                    idx,
                )
            decode_errors.append({"index": idx, "error": str(e)})
            if on_decode_error == "raise":
                raise
            if on_decode_error == "report":
                images.append(None)
                labels.append(None)
            continue

        images.append(pix)
        labels.append(label)

    if on_decode_error != "report":
        if not images:
            if decode_errors:
                preview = "; ".join(f"row {item['index']}: {item['error']}" for item in decode_errors[:3])
                raise ValueError(
                    f"all samples failed preprocessing for split='{split_name}'. "
                    f"sample errors: {preview}"
                )
            raise ValueError(f"no samples survived preprocessing for split='{split_name}'")
        # Detection and segmentation datasets can include many high-resolution images.
        # Stacking the full split into one contiguous NCHW tensor eagerly allocates
        # all pixel storage at once and can exhaust host RAM before batching.
        #
        # Keep pixel values as a per-sample list and let the training loop
        # materialize tensor batches lazily in HFCore._batch_iter/encode_batch.
        if task_type in {"detection", "segmentation"}:
            x = {"pixel_values": images}
        else:
            try:
                stacked_images = np.stack(images, axis=0).astype(np.float32, copy=False)
            except ValueError:
                unique_shapes = sorted({tuple(np.asarray(img).shape) for img in images})
                channel_counts = {shape[0] for shape in unique_shapes if len(shape) == 3}
                if channel_counts != {3}:
                    raise ValueError(
                        "pixel values have inconsistent non-CHW shapes after preprocessing. "
                        f"observed shapes={unique_shapes}"
                    )

                max_h = max(shape[1] for shape in unique_shapes)
                max_w = max(shape[2] for shape in unique_shapes)
                padded_images = []
                for image in images:
                    arr = np.asarray(image, dtype=np.float32)
                    if arr.shape[1] == max_h and arr.shape[2] == max_w:
                        padded_images.append(arr)
                        continue
                    pad_h = max_h - arr.shape[1]
                    pad_w = max_w - arr.shape[2]
                    padded_images.append(np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant"))
                stacked_images = np.stack(padded_images, axis=0).astype(np.float32, copy=False)
            x = {"pixel_values": stacked_images}
    else:
        x = {"pixel_values": images}
    if task_type == "classification":
        y = np.asarray(labels, dtype=np.int64)
    else:
        y = labels

    report = {"total": len(ds), "failed": len(decode_errors), "survived": len(images)}
    if report_decode_errors:
        report["errors"] = decode_errors
    return x, y, report


def preprocess_hf_image(
    train,
    test,
    meta,
    *,
    hf_model_id,
    image_column="image",
    label_column="label",
    boxes_column=None,
    classes_column=None,
    mask_column=None,
    task_type=None,
    training_augmentations=True,
    eval_augmentations=False,
    on_decode_error="skip",
    report_decode_errors=False,
):
    if (task_type or meta.get("task_type", "classification")).strip().lower() == "detection":
        LOGGER.info("[detection preprocessing] entering detection preprocessing in preprocess_hf_image")
    try:
        from transformers import AutoImageProcessor
    except Exception as e:
        raise ImportError("HF image preprocessing requires transformers[vision]") from e

    if on_decode_error not in {"skip", "raise", "report"}:
        raise ValueError("on_decode_error must be one of ['skip', 'raise', 'report']")

    ds_train, _ = train
    ds_test, _ = test
    task_type = (task_type or meta.get("task_type", "classification")).strip().lower()
    detection_class_id_map = None
    if task_type == "detection":
        LOGGER.info(
            "[detection preprocessing] split lengths train=%d test=%d",
            len(ds_train),
            len(ds_test),
        )
        category_names = _extract_detection_category_names(ds_train, label_column)
        detection_class_id_map = _build_detection_class_id_map(
            hf_model_id=hf_model_id,
            category_names=category_names,
        )

    resolved_mask_column = mask_column
    if task_type == "segmentation" and not resolved_mask_column:
        resolved_mask_column = label_column
    dataset_id = meta.get("hf_id") or (meta.get("dataset_args") or {}).get("dataset_name")
    media_columns = [image_column]
    if task_type == "segmentation" and resolved_mask_column:
        media_columns.append(resolved_mask_column)
    ds_train = with_hf_columns_decode_disabled(ds_train, *media_columns)
    ds_test = with_hf_columns_decode_disabled(ds_test, *media_columns)

    processor_kwargs = {"use_fast": False} if task_type == "segmentation" else {}
    segmentation_model_num_labels = None
    segmentation_ignore_index = None
    if task_type == "segmentation":
        try:
            from transformers import AutoConfig
        except Exception:
            AutoConfig = None
        if AutoConfig is not None:
            try:
                cfg = AutoConfig.from_pretrained(hf_model_id)
                raw_num_labels = getattr(cfg, "num_labels", None)
                if raw_num_labels is not None:
                    segmentation_model_num_labels = int(raw_num_labels)
                raw_ignore_index = getattr(cfg, "semantic_loss_ignore_index", None)
                if raw_ignore_index is not None:
                    segmentation_ignore_index = int(raw_ignore_index)
            except Exception:
                segmentation_model_num_labels = None
                segmentation_ignore_index = None

    processor = AutoImageProcessor.from_pretrained(hf_model_id, **processor_kwargs)
    if task_type == "segmentation" and hasattr(processor, "do_reduce_labels"):
        # Scan the selected training split instead of a tiny prefix sample.
        # The previous probe could miss rare max-label ids and cause the same
        # dataset/model pair to flip between aligned (150 labels) and misaligned
        # (151 labels) runs depending on shuffle order.
        raw_min, raw_max = _sample_segmentation_mask_range(
            ds_train,
            resolved_mask_column,
            dataset_id=dataset_id,
        )
        if (
            raw_min == 0
            and raw_max is not None
            and segmentation_model_num_labels is not None
            and int(raw_max) == int(segmentation_model_num_labels)
        ):
            processor.do_reduce_labels = True

    x_train, y_train, train_report = _process_split(
        ds_train,
        split_name="train",
        image_processor=processor,
        image_column=image_column,
        task_type=task_type,
        training=bool(training_augmentations),
        label_column=label_column,
        boxes_column=boxes_column,
        classes_column=classes_column,
        mask_column=resolved_mask_column,
        dataset_id=dataset_id,
        on_decode_error=on_decode_error,
        report_decode_errors=report_decode_errors,
    )
    x_test, y_test, test_report = _process_split(
        ds_test,
        split_name="test",
        image_processor=processor,
        image_column=image_column,
        task_type=task_type,
        training=bool(eval_augmentations),
        label_column=label_column,
        boxes_column=boxes_column,
        classes_column=classes_column,
        mask_column=resolved_mask_column,
        dataset_id=dataset_id,
        on_decode_error=on_decode_error,
        report_decode_errors=report_decode_errors,
    )

    if task_type == "detection":
        likely_contiguous = _is_likely_contiguous_zero_based_detection_labels(list(y_train) + list(y_test))
        if detection_class_id_map is not None:
            for item in y_train:
                if isinstance(item, dict):
                    item["class_id_map"] = detection_class_id_map
            for item in y_test:
                if isinstance(item, dict):
                    item["class_id_map"] = detection_class_id_map
        if likely_contiguous:
            for item in y_train:
                if isinstance(item, dict):
                    item["force_contiguous_label_remap"] = True
            for item in y_test:
                if isinstance(item, dict):
                    item["force_contiguous_label_remap"] = True

    meta = append_accounting_stage(
        meta,
        stage="hf_image",
        split="train",
        input_record_count=len(ds_train),
        post_filter_record_count=int(train_report["survived"]),
        emitted_record_count=len(x_train["pixel_values"]),
        sequence_count=len(x_train["pixel_values"]),
        metric_instance_count=len(y_train),
    )
    meta = append_accounting_stage(
        meta,
        stage="hf_image",
        split="test",
        input_record_count=len(ds_test),
        post_filter_record_count=int(test_report["survived"]),
        emitted_record_count=len(x_test["pixel_values"]),
        sequence_count=len(x_test["pixel_values"]),
        metric_instance_count=len(y_test),
    )
    meta = finalize_accounting(meta)
    inferred_input_shape = ()
    pixel_values = x_train.get("pixel_values") if isinstance(x_train, dict) else None
    if pixel_values is not None and len(pixel_values) > 0:
        inferred_input_shape = tuple(getattr(pixel_values[0], "shape", ()))
    inferred_num_classes = None
    if task_type == "classification" and len(y_train) > 0:
        inferred_num_classes = int(np.unique(np.asarray(y_train)).size)
    feature_num_classes = None
    if task_type == "classification":
        try:
            label_feature = (getattr(ds_train, "features", None) or {}).get(label_column)
            if label_feature is not None:
                class_names = getattr(label_feature, "names", None)
                if class_names:
                    feature_num_classes = int(len(class_names))
                else:
                    num_classes_attr = getattr(label_feature, "num_classes", None)
                    if num_classes_attr is not None:
                        feature_num_classes = int(num_classes_attr)
        except Exception:
            feature_num_classes = None

    resolved_num_classes = meta.get("num_classes")
    if task_type == "classification":
        candidates = []
        for candidate in (resolved_num_classes, feature_num_classes, inferred_num_classes):
            if candidate is None:
                continue
            try:
                value = int(candidate)
            except Exception:
                continue
            if value > 0:
                candidates.append(value)
        resolved_num_classes = int(max(candidates)) if candidates else None
    elif task_type == "segmentation":
        inferred_segmentation_num_classes = _infer_segmentation_num_classes(
            list(y_train) + list(y_test),
            ignore_index=segmentation_ignore_index,
        )
        candidates = []
        for candidate in (resolved_num_classes, segmentation_model_num_labels, inferred_segmentation_num_classes):
            if candidate is None:
                continue
            try:
                value = int(candidate)
            except Exception:
                continue
            if value > 0:
                candidates.append(value)
        resolved_num_classes = int(max(candidates)) if candidates else None
    meta.update(
        {
            "input_shape": inferred_input_shape,
            "image_column": image_column,
            "label_column": label_column,
            "boxes_column": boxes_column,
            "classes_column": classes_column,
            "mask_column": resolved_mask_column,
            "task_type": task_type,
            "modality": "image",
            "hf_processor": hf_model_id,
            "channel_order": "CHW",
            "num_classes": resolved_num_classes,
            "num_labels": resolved_num_classes,
            "training_augmentations": bool(training_augmentations),
            "eval_augmentations": bool(eval_augmentations),
            "decode_error_policy": on_decode_error,
            "decode_report": {"train": train_report, "test": test_report},
            "segmentation_reduce_labels": bool(getattr(processor, "do_reduce_labels", False)) if task_type == "segmentation" else None,
            "schema": {
                "image_column": image_column,
                "label_column": label_column if task_type == "classification" else None,
                "detection": {"boxes_column": boxes_column, "classes_column": classes_column},
                "segmentation": {"mask_column": resolved_mask_column},
            },
        }
    )
    if task_type == "segmentation":
        if segmentation_ignore_index is not None:
            meta["label_pad_value"] = segmentation_ignore_index
            meta["ignore_index"] = segmentation_ignore_index
    else:
        meta.pop("label_pad_value", None)
        meta.pop("ignore_index", None)
    if task_type == "detection":
        meta["detection_label_id_space"] = (
            "contiguous_zero_based"
            if _is_likely_contiguous_zero_based_detection_labels(list(y_train) + list(y_test))
            else "dataset_native"
        )
        if detection_class_id_map is not None:
            meta["detection_class_id_map"] = detection_class_id_map.tolist()

    return (x_train, y_train), (x_test, y_test), meta
