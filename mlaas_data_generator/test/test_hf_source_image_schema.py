import sys
import types

import pytest

from mlaas_data_generator.data.sources.huggingface import load_huggingface_source


class DummyDS:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r.get(key) for r in self.rows]
        return self.rows[key]

    def select(self, idxs):
        return DummyDS([self.rows[i] for i in idxs])

    def train_test_split(self, test_size=0.2, seed=42, shuffle=True):
        n_test = max(1, int(len(self.rows) * test_size))
        return {"train": DummyDS(self.rows[:-n_test]), "test": DummyDS(self.rows[-n_test:])}


def test_hf_source_image_schema_classification(monkeypatch):
    ds = DummyDS([
        {"image": object(), "label": 0, "boxes": [[1, 2, 3, 4]], "classes": [1], "mask": [[1]]},
        {"image": object(), "label": 1, "boxes": [[1, 2, 3, 4]], "classes": [1], "mask": [[0]]},
    ])

    fake_mod = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: ds)
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    _, _, meta = load_huggingface_source(
        dataset_name="dummy",
        modality="image",
        task="classification",
        image_column="image",
        label_column="label",
        boxes_column="boxes",
        classes_column="classes",
        mask_column="mask",
    )

    assert meta["schema"]["image_column"] == "image"
    assert meta["schema"]["label_column"] == "label"
    assert meta["schema"]["detection"]["boxes_column"] == "boxes"
    assert meta["schema"]["segmentation"]["mask_column"] == "mask"


def test_hf_source_image_requires_label_for_classification(monkeypatch):
    ds = DummyDS([{"image": object()}])
    fake_mod = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: ds)
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    with pytest.raises(ValueError):
        load_huggingface_source(dataset_name="dummy", modality="image", task="classification", image_column="image")


def test_hf_source_image_segmentation_falls_back_to_label_column_for_mask(monkeypatch):
    ds = DummyDS([{"image": object(), "annotation": [[1]]}])
    fake_mod = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: ds)
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    _, _, meta = load_huggingface_source(
        dataset_name="dummy",
        modality="image",
        task="segmentation",
        image_column="image",
        label_column="annotation",
    )

    assert meta["schema"]["segmentation"]["mask_column"] == "annotation"
