import sys
import types

import numpy as np

import mlaas_data_generator.data.preprocessors.hf_image as hf_image_module
from mlaas_data_generator.data.preprocessors.hf import preprocess_hf
from mlaas_data_generator.data.preprocessors.hf_image import (
    _build_detection_class_id_map,
    _extract_detection_annotations,
    _to_numpy_mask,
)


class DummySplit:
    def __init__(self, rows, *, features=None):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []
        self.features = features or {}

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, item):
        if isinstance(item, str):
            return [r.get(item) for r in self._rows]
        return self._rows[item]


class BrokenRowSplit(DummySplit):
    def __init__(self, rows, *, broken_indices=None, features=None):
        super().__init__(rows, features=features)
        self._broken_indices = set(broken_indices or ())

    def __getitem__(self, item):
        if isinstance(item, str):
            return [r.get(item) for r in self._rows]
        if item in self._broken_indices:
            raise FileNotFoundError(f"missing-image-{item}.jpg")
        return self._rows[item]


class DecodeToggleSplit(DummySplit):
    def __init__(self, rows, *, required_columns=None, features=None):
        super().__init__(rows, features=features)
        self._required_columns = tuple(required_columns or ())
        self._raw_columns = set()

    def disable_decode(self, *columns):
        clone = DecodeToggleSplit(self._rows, required_columns=self._required_columns, features=self.features)
        clone._raw_columns = self._raw_columns | {column for column in columns if column}
        return clone

    def __getitem__(self, item):
        if isinstance(item, str):
            return [r.get(item) for r in self._rows]
        missing = [column for column in self._required_columns if column not in self._raw_columns]
        if missing:
            raise FileNotFoundError(f"decoded access failed before raw-cast for columns={missing}")
        return self._rows[item]


class FakeImageProcessor:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, image, return_tensors=None, do_resize=True, do_normalize=True, do_augment=False):
        arr = np.asarray(image, dtype=np.float32)
        if do_normalize and arr.max() > 1:
            arr = arr / 255.0
        if do_augment:
            arr = arr + 0.1
        # return HWC to exercise channel-order conversion.
        return {"pixel_values": arr}


class FakeImageProcessorNoAugmentArg:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, image, return_tensors=None, do_resize=True, do_normalize=True):
        arr = np.asarray(image, dtype=np.float32)
        if do_normalize and arr.max() > 1:
            arr = arr / 255.0
        return {"pixel_values": arr}


class FakeImageProcessorCallKwargsButStrictPreprocess:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, image, **kwargs):
        return self.preprocess(image, **kwargs)

    def preprocess(self, image, return_tensors=None, do_resize=True, do_normalize=True):
        arr = np.asarray(image, dtype=np.float32)
        if do_normalize and arr.max() > 1:
            arr = arr / 255.0
        return {"pixel_values": arr}


class FakeSegmentationImageProcessor:
    last_from_pretrained_kwargs = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_from_pretrained_kwargs = dict(kwargs)
        return cls()

    def __init__(self):
        self.do_reduce_labels = False

    def __call__(self, images, segmentation_maps=None, return_tensors=None, do_resize=True, do_normalize=True, do_augment=False):
        image = images[0] if isinstance(images, (list, tuple)) else images
        mask = segmentation_maps[0] if isinstance(segmentation_maps, (list, tuple)) else segmentation_maps

        image_arr = np.asarray(image, dtype=np.float32)
        if image_arr.ndim == 3 and image_arr.shape[0] != 3 and image_arr.shape[-1] == 3:
            image_arr = np.transpose(image_arr, (2, 0, 1))
        if do_resize:
            fill = 0.1 if do_augment else 0.0
            image_arr = np.full((1, 3, 4, 4), fill, dtype=np.float32)
        else:
            image_arr = image_arr[None, ...]

        mask_arr = np.asarray(mask, dtype=np.int64)
        if self.do_reduce_labels:
            mask_arr = mask_arr.copy()
            mask_arr[mask_arr == 0] = 255
            valid = mask_arr != 255
            mask_arr[valid] -= 1
        if do_resize:
            mask_fill = int(mask_arr.max()) if mask_arr.size else 0
            mask_arr = np.full((1, 4, 4), mask_fill, dtype=np.int64)
        else:
            mask_arr = mask_arr[None, ...]

        return {"pixel_values": image_arr, "labels": mask_arr}


class FakeClassLabel:
    def __init__(self, names):
        self.names = list(names)


def _install_fake_transformers():
    fake_mod = types.SimpleNamespace(AutoImageProcessor=FakeImageProcessor)
    sys.modules["transformers"] = fake_mod


def _install_fake_transformers_no_augment_arg():
    fake_mod = types.SimpleNamespace(AutoImageProcessor=FakeImageProcessorNoAugmentArg)
    sys.modules["transformers"] = fake_mod


def _install_fake_transformers_call_kwargs_strict_preprocess():
    fake_mod = types.SimpleNamespace(AutoImageProcessor=FakeImageProcessorCallKwargsButStrictPreprocess)
    sys.modules["transformers"] = fake_mod


def _install_fake_transformers_segmentation():
    fake_mod = types.SimpleNamespace(AutoImageProcessor=FakeSegmentationImageProcessor)
    sys.modules["transformers"] = fake_mod


def test_image_classification_routing_and_deterministic_eval():
    _install_fake_transformers()
    train_rows = [
        {"image": np.zeros((4, 4, 3), dtype=np.uint8), "label": 0},
        {"image": np.ones((4, 4, 3), dtype=np.uint8) * 255, "label": 1},
    ]
    test_rows = [{"image": np.ones((4, 4, 3), dtype=np.uint8), "label": 1}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "sequence_classification", "modality": "image", "task_type": "classification", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        training_augmentations=True,
        eval_augmentations=False,
    )

    x_train, y_train = train
    x_test, y_test = test

    assert meta["hf_task"] == "image_classification"
    assert x_train["pixel_values"].shape == (2, 3, 4, 4)
    assert x_test["pixel_values"].shape == (1, 3, 4, 4)
    assert np.isclose(float(x_train["pixel_values"][0, 0, 0, 0]), 0.1)
    assert not np.isclose(float(x_test["pixel_values"][0, 0, 0, 0]), 0.1)
    assert y_train.dtype == np.int64
    assert y_test.dtype == np.int64


def test_image_decode_error_skip_and_report():
    _install_fake_transformers()
    train_rows = [
        {"image": np.zeros((2, 2, 3), dtype=np.uint8), "label": 0},
        {"image": object(), "label": 1},
    ]
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "label": 0}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "sequence_classification", "modality": "image", "task_type": "classification", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        on_decode_error="skip",
        report_decode_errors=True,
    )

    x_train, y_train = train
    assert x_train["pixel_values"].shape[0] == 1
    assert y_train.shape[0] == 1
    assert meta["decode_report"]["train"]["failed"] == 1
    assert meta["accounting"]["post_filter_record_count"] == 1
    assert meta["accounting"]["sequence_count"] == 1


def test_image_decode_error_skip_when_row_fetch_fails():
    _install_fake_transformers()
    train_rows = [
        {"image": np.zeros((2, 2, 3), dtype=np.uint8), "label": 0},
        {"image": np.ones((2, 2, 3), dtype=np.uint8), "label": 1},
    ]
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "label": 0}]

    train, test, meta = preprocess_hf(
        (BrokenRowSplit(train_rows, broken_indices={0}), None),
        (DummySplit(test_rows), None),
        {"hf_task": "sequence_classification", "modality": "image", "task_type": "classification", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        on_decode_error="skip",
        report_decode_errors=True,
    )

    x_train, y_train = train
    assert x_train["pixel_values"].shape[0] == 1
    assert y_train.tolist() == [1]
    assert meta["decode_report"]["train"]["failed"] == 1
    assert meta["accounting"]["post_filter_record_count"] == 1


def test_image_detection_schema_passthrough():
    _install_fake_transformers()
    train_rows = [
        {"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [[0, 0, 1, 1]], "classes": [2]},
    ]
    test_rows = [
        {"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [], "classes": []},
    ]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "sequence_classification", "modality": "image", "task_type": "detection", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        boxes_column="boxes",
        classes_column="classes",
    )

    _, y_train = train
    assert meta["schema"]["detection"]["boxes_column"] == "boxes"
    assert y_train[0]["boxes"].shape == (1, 4)
    assert y_train[0]["classes"].shape == (1,)
    assert y_train[0]["image_size"].tolist() == [2, 2]
    assert y_train[0]["box_format"] in {"xyxy", "xywh"}


def test_image_task_dispatch_uses_hf_task_when_modality_missing():
    _install_fake_transformers()
    train_rows = [{"image": np.zeros((3, 3, 3), dtype=np.uint8), "label": 1}]
    test_rows = [{"image": np.ones((3, 3, 3), dtype=np.uint8), "label": 0}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "image_classification", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
    )

    x_train, y_train = train
    x_test, y_test = test
    assert meta["modality"] == "image"
    assert meta["task_type"] == "classification"
    assert meta["hf_task"] == "image_classification"
    assert x_train["pixel_values"].shape == (1, 3, 3, 3)
    assert x_test["pixel_values"].shape == (1, 3, 3, 3)
    assert y_train.tolist() == [1]
    assert y_test.tolist() == [0]


def test_image_classification_preserves_existing_num_classes_metadata():
    _install_fake_transformers()
    train_rows = [{"image": np.zeros((3, 3, 3), dtype=np.uint8), "label": 0}]
    test_rows = [{"image": np.ones((3, 3, 3), dtype=np.uint8), "label": 0}]

    _, _, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "image_classification", "modality": "image", "task_type": "classification", "num_classes": 3, "hf_id": "dummy"},
        hf_model_id="dummy/vision",
    )

    assert meta["num_classes"] == 3


def test_image_classification_uses_feature_classes_when_meta_is_too_small():
    _install_fake_transformers()
    label_feature = FakeClassLabel(["healthy", "angular_leaf_spot", "bean_rust"])
    train_rows = [{"image": np.zeros((3, 3, 3), dtype=np.uint8), "label": 0}]
    test_rows = [{"image": np.ones((3, 3, 3), dtype=np.uint8), "label": 1}]

    _, _, meta = preprocess_hf(
        (DummySplit(train_rows, features={"label": label_feature}), None),
        (DummySplit(test_rows, features={"label": label_feature}), None),
        {"hf_task": "image_classification", "modality": "image", "task_type": "classification", "num_classes": 1, "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        label_column="label",
    )

    assert meta["num_classes"] == 3


def test_image_task_dispatch_normalizes_detection_alias_with_wrong_modality():
    _install_fake_transformers()
    train_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [[0, 0, 1, 1]], "classes": [2]}]
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [], "classes": []}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "object_detection", "modality": "text", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        boxes_column="boxes",
        classes_column="classes",
    )

    _, y_train = train
    _, y_test = test
    assert meta["modality"] == "image"
    assert meta["task_type"] == "detection"
    assert meta["hf_task"] == "image_detection"
    assert y_train[0]["boxes"].shape == (1, 4)
    assert y_test[0]["boxes"].shape == (0, 4)


def test_image_preprocessor_handles_processors_without_do_augment_arg():
    _install_fake_transformers_no_augment_arg()
    train_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [[0, 0, 1, 1]], "classes": [1]}]
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [[0, 0, 1, 1]], "classes": [1]}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "object_detection", "modality": "image", "task_type": "detection", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        boxes_column="boxes",
        classes_column="classes",
    )

    x_train, y_train = train
    x_test, y_test = test
    assert len(x_train["pixel_values"]) == 1
    assert len(x_test["pixel_values"]) == 1
    assert len(y_train) == 1
    assert len(y_test) == 1
    assert meta["decode_report"]["train"]["failed"] == 0
    assert meta["decode_report"]["test"]["failed"] == 0


def test_image_detection_extracts_boxes_and_classes_from_annotation_column():
    _install_fake_transformers()
    train_rows = [
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "annotation": {"objects": {"bbox": [[0, 0, 1, 1]], "category": [3]}},
        }
    ]
    test_rows = [
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "annotation": {"objects": {"bbox": [[0, 0, 1, 1]], "category_id": [4]}},
        }
    ]

    train, test, _ = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "object_detection", "modality": "image", "task_type": "detection", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        label_column="annotation",
    )

    _, y_train = train
    _, y_test = test
    assert y_train[0]["boxes"].shape == (1, 4)
    assert y_train[0]["classes"].tolist() == [3]
    assert y_test[0]["boxes"].shape == (1, 4)
    assert y_test[0]["classes"].tolist() == [4]


def test_image_detection_marks_contiguous_zero_based_label_space_for_forced_remap():
    _install_fake_transformers()
    train_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [[0, 0, 1, 1]], "classes": [0, 2, 3]}]
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [[0, 0, 1, 1]], "classes": [2]}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "object_detection", "modality": "image", "task_type": "detection", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        boxes_column="boxes",
        classes_column="classes",
    )

    _, y_train = train
    _, y_test = test
    assert meta["detection_label_id_space"] == "contiguous_zero_based"
    assert bool(y_train[0]["force_contiguous_label_remap"]) is True
    assert bool(y_test[0]["force_contiguous_label_remap"]) is True


def test_image_preprocessor_filters_kwargs_against_preprocess_signature():
    _install_fake_transformers_call_kwargs_strict_preprocess()
    train_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "label": 1}]
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "label": 1}]

    (x_train, y_train), (x_test, y_test), meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "image_classification", "modality": "image", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
    )

    assert x_train["pixel_values"].shape[0] == 1
    assert x_test["pixel_values"].shape[0] == 1
    assert y_train.tolist() == [1]
    assert y_test.tolist() == [1]
    assert meta["decode_report"]["train"]["failed"] == 0


def test_image_classification_preprocessor_drops_null_ignore_index_metadata():
    _install_fake_transformers()
    train_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "label": 1}]
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "label": 1}]

    (_, _), (_, _), meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {
            "hf_task": "image_classification",
            "modality": "image",
            "hf_id": "dummy",
            "label_pad_value": None,
            "ignore_index": None,
        },
        hf_model_id="dummy/vision",
    )

    assert "label_pad_value" not in meta
    assert "ignore_index" not in meta


def test_image_segmentation_uses_label_column_as_mask_fallback_and_slow_processor():
    _install_fake_transformers_segmentation()
    train_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "annotation": np.asarray([[0, 1], [2, 3]], dtype=np.uint8)}]
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "annotation": np.asarray([[1, 1], [1, 1]], dtype=np.uint8)}]

    (x_train, y_train), (x_test, y_test), meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "image_segmentation", "modality": "image", "task_type": "segmentation", "hf_id": "dummy"},
        hf_model_id="dummy/segmentation",
        label_column="annotation",
    )

    assert FakeSegmentationImageProcessor.last_from_pretrained_kwargs == {"use_fast": False}
    assert meta["mask_column"] == "annotation"
    assert meta["schema"]["segmentation"]["mask_column"] == "annotation"
    assert len(x_train["pixel_values"]) == 1
    assert len(x_test["pixel_values"]) == 1
    assert x_train["pixel_values"][0].shape == (3, 4, 4)
    assert x_test["pixel_values"][0].shape == (3, 4, 4)
    assert y_train[0].shape == (4, 4)
    assert y_test[0].shape == (4, 4)
    assert meta["num_classes"] == 4
    assert meta["num_labels"] == 4
    assert meta["decode_report"]["train"]["failed"] == 0


def test_image_segmentation_reduce_labels_scans_beyond_small_prefix():
    class _AutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return types.SimpleNamespace(num_labels=3, semantic_loss_ignore_index=255)

    fake_mod = types.SimpleNamespace(
        AutoImageProcessor=FakeSegmentationImageProcessor,
        AutoConfig=_AutoConfig,
    )
    sys.modules["transformers"] = fake_mod

    train_rows = []
    for _ in range(16):
        train_rows.append(
            {"image": np.zeros((2, 2, 3), dtype=np.uint8), "annotation": np.asarray([[0, 1], [1, 1]], dtype=np.uint8)}
        )
    train_rows.append(
        {"image": np.zeros((2, 2, 3), dtype=np.uint8), "annotation": np.asarray([[0, 3], [1, 2]], dtype=np.uint8)}
    )
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "annotation": np.asarray([[0, 1], [2, 3]], dtype=np.uint8)}]

    (_, _), (_, _), meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "image_segmentation", "modality": "image", "task_type": "segmentation", "hf_id": "dummy"},
        hf_model_id="dummy/segmentation",
        label_column="annotation",
    )

    assert meta["segmentation_reduce_labels"] is True
    assert meta["num_labels"] == 3


def test_qubvel_ade20k_mini_rgb_annotation_decodes_to_2d_semantic_mask():
    rgb_mask = np.asarray(
        [
            [[0, 0, 0], [1, 1, 0]],
            [[2, 3, 0], [1, 2, 0]],
        ],
        dtype=np.uint8,
    )

    mask = _to_numpy_mask(rgb_mask, dataset_id="qubvel-hf/ade20k-mini")

    assert mask.tolist() == [[0, 1], [2, 1]]


def test_image_segmentation_preprocesses_qubvel_ade20k_mini_rgb_annotations():
    _install_fake_transformers_segmentation()
    train_rows = [
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "annotation": np.asarray(
                [
                    [[0, 0, 0], [1, 1, 0]],
                    [[2, 3, 0], [1, 2, 0]],
                ],
                dtype=np.uint8,
            ),
        }
    ]
    test_rows = [
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "annotation": np.asarray(
                [
                    [[0, 0, 0], [2, 1, 0]],
                    [[2, 2, 0], [0, 0, 0]],
                ],
                dtype=np.uint8,
            ),
        }
    ]

    (x_train, y_train), (x_test, y_test), meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {
            "hf_task": "image_segmentation",
            "modality": "image",
            "task_type": "segmentation",
            "hf_id": "qubvel-hf/ade20k-mini",
        },
        hf_model_id="dummy/segmentation",
        label_column="annotation",
        mask_column="annotation",
    )

    assert len(x_train["pixel_values"]) == 1
    assert len(x_test["pixel_values"]) == 1
    assert y_train[0].shape == (4, 4)
    assert y_test[0].shape == (4, 4)
    assert int(np.max(y_train[0])) == 2
    assert meta["decode_report"]["train"]["failed"] == 0


def test_image_segmentation_disables_hf_decode_before_row_access(monkeypatch):
    _install_fake_transformers_segmentation()

    def _disable_decode(ds, *columns):
        return ds.disable_decode(*columns)

    monkeypatch.setattr(hf_image_module, "with_hf_columns_decode_disabled", _disable_decode)

    train_rows = [
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "annotation": np.zeros((2, 2), dtype=np.uint8),
        }
    ]
    test_rows = [
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "annotation": np.zeros((2, 2), dtype=np.uint8),
        }
    ]

    (x_train, y_train), (_x_test, _y_test), meta = preprocess_hf(
        (DecodeToggleSplit(train_rows, required_columns=("image", "annotation")), None),
        (DecodeToggleSplit(test_rows, required_columns=("image", "annotation")), None),
        {
            "hf_task": "image_segmentation",
            "modality": "image",
            "task_type": "segmentation",
            "hf_id": "dummy",
        },
        hf_model_id="dummy/segmentation",
        label_column="annotation",
        mask_column="annotation",
    )

    assert len(x_train["pixel_values"]) == 1
    assert y_train[0].shape == (4, 4)
    assert meta["decode_report"]["train"]["failed"] == 0


def test_segmentation_binary_255_masks_are_mapped_to_foreground_class():
    mask = _to_numpy_mask(np.asarray([[0, 255], [255, 0]], dtype=np.uint8))

    assert mask.tolist() == [[0, 1], [1, 0]]


def test_detection_annotations_accept_json_records_and_label_key():
    json_row = {
        "annotations": '[{"bbox": [1, 2, 3, 4], "category_id": 7}, {"bbox": [5, 6, 7, 8], "category_id": 8}]'
    }
    parsed = _extract_detection_annotations(json_row, label_column="annotations")

    assert parsed["boxes"].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert parsed["classes"].tolist() == [7, 8]

    label_row = {"objects": {"bbox": [[1, 2, 3, 4]], "label": [4]}}
    parsed = _extract_detection_annotations(label_row, label_column="objects")

    assert parsed["boxes"].tolist() == [[1, 2, 3, 4]]
    assert parsed["classes"].tolist() == [4]


def test_detection_class_id_map_matches_names_to_model_ids():
    class _AutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return types.SimpleNamespace(
                id2label={
                    0: "N/A",
                    1: "person",
                    2: "bicycle",
                    12: "street sign",
                    13: "stop sign",
                }
            )

    sys.modules["transformers"] = types.SimpleNamespace(AutoConfig=_AutoConfig)

    mapping = _build_detection_class_id_map(
        hf_model_id="dummy/model",
        category_names=["person", "stop sign"],
    )

    assert mapping.tolist() == [1, 13]
