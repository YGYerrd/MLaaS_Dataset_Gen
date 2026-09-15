from types import SimpleNamespace

import numpy as np
import pytest

from mlaas_data_generator.data.preprocessors.hf_image import (
    _declared_segmentation_label_mapping, _to_numpy_mask, preprocess_hf_image,
)


LABELS = {"background": 0, "person": 1, "car": 2}


class Split:
    column_names = ["image", "annotation", "semantic_class_to_id"]
    features = {}

    def __init__(self, mappings):
        self.rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8),
                      "annotation": np.array([[[0, 0, 0], [1, 7, 0]], [[2, 12, 0], [0, 0, 0]]]),
                      "semantic_class_to_id": mapping} for mapping in mappings]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return [row[key] for row in self.rows] if isinstance(key, str) else self.rows[key]


def test_rgb_mini_mask_preserves_red_class_ids_and_ignores_instance_channel():
    raw = Split([LABELS])[0]["annotation"]
    assert not np.array_equal(raw[..., 0], raw[..., 1])
    actual = _to_numpy_mask(raw, dataset_id="qubvel-hf/ade20k-mini")
    np.testing.assert_array_equal(actual, [[0, 1], [2, 0]])


@pytest.mark.parametrize("mappings", [[LABELS, {"background": 0, "person": 2, "car": 1}],
                                        [{"background": 0, "person": 4}], [{}]])
def test_invalid_or_changing_label_schemas_fail(mappings):
    with pytest.raises(ValueError, match="semantic_class_to_id"):
        _declared_segmentation_label_mapping(Split(mappings))


@pytest.mark.parametrize("inference_only,num_classes", [(False, 3), (True, 150)])
def test_declared_segmentation_labels_control_finetune_head_and_keep_background(monkeypatch, inference_only, num_classes):
    import transformers
    class Processor:
        do_reduce_labels = True
        def to_dict(self):
            return {"do_reduce_labels": self.do_reduce_labels}
        def __call__(self, images, segmentation_maps, **kwargs):
            assert not self.do_reduce_labels
            return {"pixel_values": np.stack(images).transpose(0, 3, 1, 2),
                    "labels": np.stack(segmentation_maps)}
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda *a, **k: Processor())
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(
        num_labels=150, semantic_loss_ignore_index=255))
    split = Split([LABELS])
    train, test, meta = preprocess_hf_image(
        (split, None), (split, None), {"hf_id": "qubvel-hf/ade20k-mini", "inference_only": inference_only},
        hf_model_id="test-local", task_type="segmentation", label_column="annotation",
        training_augmentations=False,
    )
    assert meta["num_labels"] == num_classes
    assert meta["label_mapping"] == LABELS
    assert not meta["segmentation_reduce_labels"]
    assert set(np.unique(train[1])) == {0, 1, 2}
