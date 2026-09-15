"""Segmentation preparation must not retain the expanded source pool in RAM."""
from types import SimpleNamespace
import gc
import tracemalloc

import numpy as np
import pytest

from mlaas_data_generator.data.image_store import (
    DiskArraySequence, DiskImageArray, close_image_stores,
)
from mlaas_data_generator.data.preprocessors import hf_image
from mlaas_data_generator.shared_parent import prepared_data_hash


class Rows:
    features = {}
    column_names = ["image", "mask"]

    def __init__(self, count=6, size=8, bad=()):
        self.count, self.size, self.bad = count, size, bad

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if index >= self.count:
            raise IndexError(index)
        return {"image": None if index in self.bad else np.full(
            (self.size, self.size, 3), index, dtype=np.uint8),
            "mask": np.full((self.size, self.size), index % 3, dtype=np.uint8)}


class Processor:
    do_reduce_labels = False

    def to_dict(self):
        return {"do_reduce_labels": self.do_reduce_labels}

    def __call__(self, images, segmentation_maps, **kwargs):
        pixels = np.asarray(images, dtype=np.float32).transpose(0, 3, 1, 2) / 255
        labels = np.asarray(segmentation_maps, dtype=np.int64)
        labels[:, 0, 0] = 255
        return {"pixel_values": pixels, "labels": labels}


def prepare(rows, **kwargs):
    return hf_image._process_split(
        rows, split_name="train", image_processor=Processor(), image_column="image",
        mask_column="mask", task_type="segmentation", training=False, **kwargs)


@pytest.fixture(autouse=True)
def cache_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("MLAAS_IMAGE_CACHE_DIR", str(tmp_path))


@pytest.mark.parametrize("policy", ["skip", "report", "raise"])
def test_spill_preserves_values_hashes_and_decode_alignment(monkeypatch, policy):
    rows = Rows(bad=() if policy == "raise" else (0, 3))
    monkeypatch.setattr(hf_image, "IMAGE_MEMORY_LIMIT_BYTES", 10**9)
    expected = prepare(rows, on_decode_error=policy, report_decode_errors=True)
    monkeypatch.setattr(hf_image, "IMAGE_MEMORY_LIMIT_BYTES", 1)
    x, y, report = actual = prepare(rows, on_decode_error=policy, report_decode_errors=True)
    try:
        assert isinstance(x["pixel_values"], DiskArraySequence)
        assert isinstance(y, DiskArraySequence)
        assert report == expected[2]
        assert prepared_data_hash(actual[:2]) == prepared_data_hash(expected[:2])
        for index, label in enumerate(expected[1]):
            if label is None:
                assert y[index] is None and x["pixel_values"][index] is None
            else:
                np.testing.assert_array_equal(y[index], label)
                assert y[index].dtype == np.int64
                assert x["pixel_values"][index].dtype == np.float32
    finally:
        close_image_stores((x, y))


@pytest.mark.parametrize("model_id,size", [
    ("Intel/dpt-large-ade", 384),
    ("openmmlab/upernet-convnext-tiny", 512),
])
def test_full_preprocessor_peak_is_bounded_including_class_inference(monkeypatch, model_id, size):
    import transformers
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda *a, **k: Processor())
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k:
                        SimpleNamespace(num_labels=3, semantic_loss_ignore_index=255))
    monkeypatch.setattr(hf_image, "IMAGE_MEMORY_LIMIT_BYTES", 1024**2)
    rows = Rows(count=40, size=size)
    gc.collect()
    tracemalloc.start()
    prepared = None
    try:
        prepared = hf_image.preprocess_hf_image(
            (rows, None), (rows, None), {}, hf_model_id=model_id,
            task_type="segmentation", mask_column="mask", training_augmentations=False)
        train, test, meta = prepared
        prepared_data_hash((train, test))  # Hashing must stream too.
        _, peak = tracemalloc.get_traced_memory()
        expanded_bytes = 2 * len(rows) * size * size * (3 * 4 + 8)
        print(f"{model_id}: expanded={expanded_bytes / 2**20:.1f} MiB, peak={peak / 2**20:.1f} MiB")
        assert peak < expanded_bytes / 5
        assert meta["num_labels"] == 3
        assert meta["ignore_index"] == 255
        assert isinstance(train[1], DiskArraySequence) and isinstance(test[1], DiskArraySequence)
    finally:
        tracemalloc.stop()
        if prepared is not None:
            close_image_stores(prepared[:2])


def test_selections_stay_lazy_and_training_batches_match(monkeypatch):
    import torch
    from mlaas_data_generator.data.fedavg_partition import _take_rows
    from mlaas_data_generator.data.splitters import _take, split_data
    from mlaas_data_generator.clients.legacy import _take_rows as legacy_take
    from mlaas_data_generator.models.adapters.hf_core import HFCore
    from mlaas_data_generator.models.adapters.hf_task import ImageSegmentationSpec
    from mlaas_data_generator.tasks.contracts import resolve_task_contract, target_fingerprints
    from mlaas_data_generator.services.perturbation import _get_sample
    from mlaas_data_generator.data.distributions import get_data_distribution

    monkeypatch.setattr(hf_image, "IMAGE_MEMORY_LIMIT_BYTES", 1)
    x, y, _ = prepare(Rows())
    try:
        contract = resolve_task_contract({"hf_task": "image_segmentation", "num_labels": 3,
                                          "label_pad_value": 255})
        assert contract.supervision_count(y) == 6 * 63
        assert len(target_fingerprints(y)) == 6
        assert get_data_distribution(y, num_classes=3, label_pad_value=255) == {0: 126, 1: 126, 2: 126}
        original_get = DiskArraySequence.__getitem__
        with monkeypatch.context() as patch:
            def no_read(*args):
                raise AssertionError("selection must not read pixels or masks")
            patch.setattr(DiskArraySequence, "__getitem__", no_read)
            for take in (_take_rows, _take, legacy_take):
                view = take(y, np.array([4, 1, 4]))
                assert isinstance(view, DiskArraySequence)
                assert view._file is y._file
            split_data(x, y, num_clients=2, strategy="iid", task_family="segmentation")
        np.testing.assert_array_equal(_get_sample(y, 2), original_get(y, 2))
        core = HFCore.__new__(HFCore)
        core.batch_size, core.label_pad_value = 2, 255
        core.task_spec = ImageSegmentationSpec()
        order = np.array([5, 1, 3, 0, 4, 2])
        for batch_index, (xb, yb) in enumerate(core._batch_iter(x, y, indices=order)):
            ids = order[batch_index * 2:batch_index * 2 + 2]
            assert isinstance(xb["pixel_values"], list) and isinstance(yb, list)
            np.testing.assert_array_equal(yb, [y[i] for i in ids])
            enc, labels, extra = core.task_spec.encode_batch(None, xb, yb, 0, torch, "cpu", ignore_index=255)
            logits = torch.nn.Conv2d(3, 3, 1)(enc["pixel_values"])
            loss = core.task_spec.loss_fn(torch, logits, labels, extra)
            loss.backward()
            assert torch.isfinite(loss)
        assert sum(len(yb) for _, yb in core._batch_iter(x, y)) == len(y)
    finally:
        close_image_stores((x, y))


def test_ragged_masks_views_lifetime_and_classification_padding():
    store = DiskArraySequence(dtype="int64", sample_ndim=2)
    samples = [np.arange(6).reshape(2, 3), np.full((4, 2), 255)]
    for sample in samples:
        store.append(sample)
    view = store.take([1, 0, 1]).take(slice(1, None))
    backing = store._file
    del store
    gc.collect()
    assert not backing.closed
    assert [row.shape for row in view[:]] == [(2, 3), (4, 2)]
    assert prepared_data_hash(view) == prepared_data_hash(samples)
    with pytest.raises(TypeError, match="batches"):
        np.asarray(view)
    close_image_stores(view)
    assert backing.closed
    dense = DiskImageArray()
    try:
        dense.append(np.ones((3, 2, 3)))
        dense.append(np.ones((3, 4, 2)))
        assert dense[:].shape == (2, 3, 4, 3)
        assert prepared_data_hash(dense) == prepared_data_hash(dense[:])
    finally:
        dense.close()


def test_spilled_files_close_on_decode_or_write_failure(monkeypatch):
    monkeypatch.setattr(hf_image, "IMAGE_MEMORY_LIMIT_BYTES", 1)
    stores = []
    class TrackingStore(DiskArraySequence):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            stores.append(self)
    monkeypatch.setattr(hf_image, "DiskArraySequence", TrackingStore)
    with pytest.raises(ValueError):
        prepare(Rows(bad=(2,)), on_decode_error="raise")
    assert len(stores) == 2 and all(store._file.closed for store in stores)
    stores.clear()
    def fail_write(self, image):
        raise OSError("disk full")
    monkeypatch.setattr(TrackingStore, "append", fail_write)
    with pytest.raises(OSError, match="disk full"):
        prepare(Rows(), on_decode_error="skip")
    assert len(stores) == 2 and all(store._file.closed for store in stores)


def test_train_cache_closes_when_test_split_fails(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda *a, **k: Processor())
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k:
                        SimpleNamespace(num_labels=3, semantic_loss_ignore_index=255))
    monkeypatch.setattr(hf_image, "IMAGE_MEMORY_LIMIT_BYTES", 1)
    stores = []
    class TrackingStore(DiskArraySequence):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            stores.append(self)
    monkeypatch.setattr(hf_image, "DiskArraySequence", TrackingStore)
    with pytest.raises(ValueError):
        hf_image.preprocess_hf_image(
            (Rows(), None), (Rows(bad=(0,)), None), {}, hf_model_id="local-test",
            task_type="segmentation", mask_column="mask", on_decode_error="raise")
    assert len(stores) == 2 and all(store._file.closed for store in stores)
