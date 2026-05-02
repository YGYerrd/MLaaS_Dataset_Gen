import numpy as np
import torch

from mlaas_data_generator.models.adapters.hf_task import (
    ImageClassificationSpec,
    ObjectDetectionSpec,
    ImageSegmentationSpec,
)


def test_image_classification_metrics_from_statistics_topk():
    spec = ImageClassificationSpec()
    out = spec.metrics_from_statistics(
        {
            "top1_correct": 7,
            "top5_correct": 9,
            "total": 10,
            "class_0_tp": 3,
            "class_0_pred_total": 4,
            "class_0_target_total": 5,
            "class_1_tp": 4,
            "class_1_pred_total": 6,
            "class_1_target_total": 5,
        }
    )
    assert np.isclose(out["primary"], 0.7)
    assert np.isclose(out["secondary"], np.mean([(2 * 3) / 9, (2 * 4) / 11]))
    assert np.isclose(out["named_metrics"]["accuracy"], 0.7)
    assert np.isclose(out["named_metrics"]["top1_accuracy"], 0.7)
    assert np.isclose(out["named_metrics"]["f1"], out["secondary"])
    assert np.isclose(out["named_metrics"]["macro_f1"], out["secondary"])
    assert np.isclose(out["named_metrics"]["top5_accuracy"], 0.9)


def test_image_classification_metrics_exposes_accuracy_alias():
    spec = ImageClassificationSpec()
    out = spec.metrics(y_true=[0, 1, 2], y_pred=[0, 0, 2])
    assert np.isclose(out["primary"], 2.0 / 3.0)
    assert np.isclose(out["secondary"], np.mean([2.0 / 3.0, 0.0, 1.0]))
    assert np.isclose(out["named_metrics"]["accuracy"], 2.0 / 3.0)
    assert np.isclose(out["named_metrics"]["top1_accuracy"], 2.0 / 3.0)
    assert np.isclose(out["named_metrics"]["f1"], out["secondary"])
    assert np.isclose(out["named_metrics"]["macro_f1"], out["secondary"])


def test_object_detection_metrics_from_statistics_map_summary():
    spec = ObjectDetectionSpec()

    class _Metric:
        def compute(self):
            return {
                "map": torch.tensor(0.31),
                "map_50": torch.tensor(0.52),
                "map_75": torch.tensor(0.29),
                "mar_1": torch.tensor(0.18),
                "mar_10": torch.tensor(0.41),
                "mar_100": torch.tensor(0.48),
            }

        def reset(self):
            return None

    out = spec.metrics_from_statistics(
        {
            "__kind__": "torchmetrics_map",
            "metric": _Metric(),
            "num_updates": 1.0,
        }
    )
    assert "map" in out["named_metrics"]
    assert "map@0.5" in out["named_metrics"]
    assert np.isclose(out["primary"], 0.31)
    assert np.isclose(out["secondary"], 0.52)


def test_segmentation_metrics_from_statistics_iou_and_dice():
    spec = ImageSegmentationSpec()
    out = spec.metrics_from_statistics(
        {
            "class_0_intersection": 3,
            "class_0_pred_total": 4,
            "class_0_target_total": 5,
            "class_1_intersection": 2,
            "class_1_pred_total": 3,
            "class_1_target_total": 3,
        }
    )
    expected_iou = np.mean([3 / (4 + 5 - 3), 2 / (3 + 3 - 2)])
    expected_dice = np.mean([(2 * 3) / (4 + 5), (2 * 2) / (3 + 3)])
    assert np.isclose(out["primary"], expected_iou)
    assert np.isclose(out["secondary"], expected_dice)
    assert np.isclose(out["named_metrics"]["pixel_accuracy"], 5 / 8)


def test_segmentation_metrics_ignore_index_and_use_classwise_miou():
    spec = ImageSegmentationSpec()
    y_true = np.asarray([[0, 1], [1, 255]], dtype=np.int64)
    y_pred = np.asarray([[0, 1], [0, 0]], dtype=np.int64)
    out = spec.metrics(y_true, y_pred, y_extra={"ignore_index": 255})
    expected_iou = np.mean([1 / 2, 1 / 2])
    expected_dice = np.mean([2 / 3, 2 / 3])
    assert np.isclose(out["primary"], expected_iou)
    assert np.isclose(out["secondary"], expected_dice)


def test_image_specs_do_not_require_tokenizer():
    assert ImageClassificationSpec.requires_tokenizer is False
    assert ObjectDetectionSpec.requires_tokenizer is False
    assert ImageSegmentationSpec.requires_tokenizer is False


def test_object_detection_does_not_require_num_labels_and_builds_without_it():
    class _AutoModelForObjectDetection:
        called_kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.called_kwargs = kwargs
            return {"model_id": model_id, "kwargs": kwargs}

    class _Transformers:
        AutoModelForObjectDetection = _AutoModelForObjectDetection

    spec = ObjectDetectionSpec()
    assert spec.requires_num_labels is False
    model = spec.build_model(_Transformers, "fake/model", num_labels=None)
    assert model["model_id"] == "fake/model"
    assert "num_labels" not in _AutoModelForObjectDetection.called_kwargs


def test_object_detection_encode_batch_converts_absolute_xywh_to_normalized_cxcywh():
    spec = ObjectDetectionSpec()
    xb = {"pixel_values": [np.zeros((3, 100, 200), dtype=np.float32)]}
    yb = [{"boxes": [[20, 10, 40, 30]], "classes": [2], "box_format": "xywh"}]

    _, labels_t, _ = spec.encode_batch(
        tokenizer=None,
        xb=xb,
        yb=yb,
        max_length=0,
        torch=torch,
        device=torch.device("cpu"),
    )

    boxes = labels_t[0]["boxes"].detach().cpu().numpy()
    # xywh absolute [20,10,40,30] on (h=100,w=200) -> xyxy norm [0.1,0.1,0.3,0.4] -> cxcywh [0.2,0.25,0.2,0.3]
    assert np.allclose(boxes, np.asarray([[0.2, 0.25, 0.2, 0.3]], dtype=np.float32), atol=1e-6)


def test_object_detection_encode_batch_uses_original_image_size_when_present():
    spec = ObjectDetectionSpec()
    xb = {"pixel_values": [np.zeros((3, 800, 1200), dtype=np.float32)]}
    yb = [
        {
            "boxes": [[300, 200, 500, 350]],
            "classes": [2],
            "box_format": "xyxy",
            "image_size": [400, 600],  # original decoded image size before resizing
        }
    ]

    _, labels_t, _ = spec.encode_batch(
        tokenizer=None,
        xb=xb,
        yb=yb,
        max_length=0,
        torch=torch,
        device=torch.device("cpu"),
    )

    boxes = labels_t[0]["boxes"].detach().cpu().numpy()
    expected = np.asarray([[2.0 / 3.0, 0.6875, 1.0 / 3.0, 0.375]], dtype=np.float32)
    assert np.allclose(boxes, expected, atol=1e-6)


def test_object_detection_encode_batch_emits_pixel_mask_for_padded_batches():
    spec = ObjectDetectionSpec()
    xb = {
        "pixel_values": [
            np.zeros((3, 4, 5), dtype=np.float32),
            np.zeros((3, 2, 3), dtype=np.float32),
        ]
    }

    enc, labels_t, _ = spec.encode_batch(
        tokenizer=None,
        xb=xb,
        yb=[{"boxes": [], "classes": []}, {"boxes": [], "classes": []}],
        max_length=0,
        torch=torch,
        device=torch.device("cpu"),
    )

    assert labels_t is not None
    pixel_mask = enc["pixel_mask"].detach().cpu().numpy()
    assert pixel_mask.shape == (2, 4, 5)
    assert np.all(pixel_mask[0] == 1)
    assert np.all(pixel_mask[1, :2, :3] == 1)
    assert np.all(pixel_mask[1, 2:, :] == 0)
    assert np.all(pixel_mask[1, :, 3:] == 0)


def test_object_detection_batch_metric_statistics_from_outputs_emits_torchmetrics_payload():
    spec = ObjectDetectionSpec(score_threshold=0.05)
    labels_t = [
        {
            "class_labels": torch.tensor([1], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
        }
    ]

    class _Outputs:
        logits = torch.tensor([[[0.1, 5.0, -4.0]]], dtype=torch.float32)  # class 1 is top score, final index is no-object
        pred_boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]], dtype=torch.float32)

    stats = spec.batch_metric_statistics_from_outputs(torch, _Outputs(), labels_t, {"score_threshold": 0.05})
    payload = stats["__map_batch__"]
    assert np.isclose(payload["metric_instance_count"], 1.0)
    assert payload["targets"][0]["labels"].tolist() == [1]
    assert payload["preds"][0]["labels"].tolist() == [1]
    assert payload["preds"][0]["boxes"].shape == (1, 4)


def test_object_detection_batch_metric_statistics_keeps_low_confidence_predictions_for_ap_ranking():
    spec = ObjectDetectionSpec(score_threshold=0.05)
    labels_t = [
        {
            "class_labels": torch.tensor([1], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
        }
    ]

    class _Outputs:
        # Class-1 confidence stays below 0.5 while box/class match perfectly.
        logits = torch.tensor([[[0.0, 0.1, 0.0]]], dtype=torch.float32)
        pred_boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]], dtype=torch.float32)

    stats = spec.batch_metric_statistics_from_outputs(torch, _Outputs(), labels_t, {"score_threshold": 0.05})
    payload = stats["__map_batch__"]
    assert payload["preds"][0]["scores"].shape[0] == 1
    assert float(payload["preds"][0]["scores"][0]) > 0.0


def test_object_detection_batch_metric_statistics_remaps_predicted_contiguous_ids_for_coco_models():
    spec = ObjectDetectionSpec(score_threshold=0.05)
    spec._model_valid_class_ids = [1, 2]
    labels_t = [
        {
            "class_labels": torch.tensor([1], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
        }
    ]

    class _Outputs:
        # Predicted class index 0 should be remapped to model class id 1 when valid ids start at 1.
        logits = torch.tensor([[[5.0, 0.1, -4.0]]], dtype=torch.float32)
        pred_boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]], dtype=torch.float32)

    stats = spec.batch_metric_statistics_from_outputs(torch, _Outputs(), labels_t, {"score_threshold": 0.05})
    payload = stats["__map_batch__"]
    assert payload["preds"][0]["labels"].tolist() == [1]


def test_object_detection_batch_metric_statistics_does_not_remap_detr_sparse_label_indices():
    spec = ObjectDetectionSpec(score_threshold=0.05)
    # Mimic DETR-style sparse COCO ids where valid class ids are non-contiguous
    # and logits still include the sparse index space (plus no-object).
    spec._model_valid_class_ids = [1, 2, 3, 5]
    labels_t = [
        {
            "class_labels": torch.tensor([1], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
        }
    ]

    class _Outputs:
        # Argmax class index is 1 and should stay 1 (not remapped to 2).
        logits = torch.tensor([[[0.1, 5.0, 0.2, 0.1, 0.0, -4.0]]], dtype=torch.float32)
        pred_boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]], dtype=torch.float32)

    stats = spec.batch_metric_statistics_from_outputs(torch, _Outputs(), labels_t, {"score_threshold": 0.05})
    payload = stats["__map_batch__"]
    assert payload["preds"][0]["labels"].tolist() == [1]


def test_object_detection_metric_accumulator_updates_torchmetrics(monkeypatch):
    spec = ObjectDetectionSpec()
    seen = {}

    class _Metric:
        def update(self, preds, targets):
            seen["preds"] = preds
            seen["targets"] = targets

        def compute(self):
            return {
                "map": torch.tensor(0.2),
                "map_50": torch.tensor(0.4),
                "map_75": torch.tensor(0.1),
                "mar_1": torch.tensor(0.1),
                "mar_10": torch.tensor(0.2),
                "mar_100": torch.tensor(0.3),
            }

        def reset(self):
            seen["reset"] = True

    monkeypatch.setattr(spec, "_build_map_metric", lambda: (_Metric(), "faster_coco_eval"))

    acc = spec.init_metric_accumulator()
    batch = {
        "__map_batch__": {
            "preds": [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)}],
            "targets": [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}],
            "metric_instance_count": 3.0,
        }
    }
    acc = spec.accumulate_metric_statistics(acc, batch)
    summary = spec.metric_statistics_summary(acc)
    out = spec.metrics_from_statistics(acc)

    assert "preds" in seen
    assert summary["metric_instance_count"] == 3.0
    assert summary["num_updates"] == 1.0
    assert np.isclose(out["primary"], 0.2)
    assert np.isclose(out["secondary"], 0.4)
    assert seen["reset"] is True


def test_object_detection_metric_accumulator_falls_back_on_int32_overflow(monkeypatch):
    spec = ObjectDetectionSpec()

    class _Metric:
        def update(self, preds, targets):
            raise RuntimeError("value cannot be converted to type int32 without overflow")

        def compute(self):
            raise AssertionError("fallback path should not compute backend metric")

    monkeypatch.setattr(spec, "_build_map_metric", lambda: (_Metric(), "faster_coco_eval"))

    acc = spec.init_metric_accumulator()
    batch = {
        "__map_batch__": {
            "preds": [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
                    "scores": torch.tensor([0.9], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.long),
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.long),
                }
            ],
            "metric_instance_count": 1.0,
            "fallback_stats": {"gt": 1.0, "tp_0.5": 1.0, "fp_0.5": 0.0, "tp_0.75": 1.0, "fp_0.75": 0.0, "tp_0.95": 1.0, "fp_0.95": 0.0},
        }
    }
    acc = spec.accumulate_metric_statistics(acc, batch)
    out = spec.metrics_from_statistics(acc)

    assert acc["fallback_updates"] == 1.0
    assert np.isclose(out["primary"], 1.0)
    assert np.isclose(out["secondary"], 1.0)


def test_object_detection_encode_batch_remaps_contiguous_coco_ids_when_model_uses_na_zero_slot():
    spec = ObjectDetectionSpec()
    # Mimic COCO-style id2label where index 0 is "N/A" and real classes start at 1.
    spec._model_valid_class_ids = list(range(1, 91))
    xb = {"pixel_values": [np.zeros((3, 8, 8), dtype=np.float32)]}
    yb = [{"boxes": [[0, 0, 1, 1]], "classes": [0, 2]}]

    _, labels_t, _ = spec.encode_batch(
        tokenizer=None,
        xb=xb,
        yb=yb,
        max_length=0,
        torch=torch,
        device=torch.device("cpu"),
    )

    assert labels_t[0]["class_labels"].detach().cpu().numpy().tolist() == [1, 3]


def test_object_detection_encode_batch_can_force_contiguous_coco_remap_without_class_zero_in_batch():
    spec = ObjectDetectionSpec()
    spec._model_valid_class_ids = list(range(1, 91))
    xb = {"pixel_values": [np.zeros((3, 8, 8), dtype=np.float32)]}
    yb = [{"boxes": [[0, 0, 1, 1]], "classes": [2], "force_contiguous_label_remap": True}]

    _, labels_t, _ = spec.encode_batch(
        tokenizer=None,
        xb=xb,
        yb=yb,
        max_length=0,
        torch=torch,
        device=torch.device("cpu"),
    )

    assert labels_t[0]["class_labels"].detach().cpu().numpy().tolist() == [3]


def test_object_detection_encode_batch_uses_explicit_class_id_map_when_available():
    spec = ObjectDetectionSpec()
    xb = {"pixel_values": [np.zeros((3, 8, 8), dtype=np.float32)]}
    yb = [{"boxes": [[0, 0, 1, 1]], "classes": [0, 2], "class_id_map": [11, 13, 17]}]

    _, labels_t, _ = spec.encode_batch(
        tokenizer=None,
        xb=xb,
        yb=yb,
        max_length=0,
        torch=torch,
        device=torch.device("cpu"),
    )

    assert labels_t[0]["class_labels"].detach().cpu().numpy().tolist() == [11, 17]


def test_segmentation_loss_downsamples_labels_to_logits_instead_of_upsampling_logits():
    spec = ImageSegmentationSpec()
    logits = torch.randn(1, 3, 2, 2)
    labels = torch.tensor(
        [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1]]],
        dtype=torch.long,
    )

    forward_inputs = spec.build_forward_inputs({"pixel_values": torch.zeros(1, 3, 4, 4)}, labels_t=labels)
    loss = spec.loss_fn(torch, logits, labels, {"ignore_index": -100})
    stats = spec.batch_metric_statistics(torch, logits, labels, {"ignore_index": -100})

    assert "labels" not in forward_inputs
    assert loss.ndim == 0
    assert stats["metric_instance_count"] == 1.0
