"""Temporary image storage with bounded reads and cheap row-selection views.

No memory mapping: scanning or hashing a split must not fault the entire expanded
image pool into the process working set. Files live until their last view dies.
"""
from __future__ import annotations

import os
import operator
import tempfile
import threading

import numpy as np


class DiskArrayStore:
    dtype = np.dtype("float32")
    ndim = 4
    _pad_to_split = True

    def __init__(self, *, dtype="float32", sample_ndim=3):
        self.dtype = np.dtype(dtype)
        self.ndim = sample_ndim + 1
        directory = os.environ.get("MLAAS_IMAGE_CACHE_DIR") or None
        self._file = tempfile.TemporaryFile(prefix="mlaas-images-", suffix=".bin", dir=directory)
        self._lock = threading.RLock()
        self._records = []
        self._image_shape = (3, 0, 0) if self._pad_to_split else (0,) * sample_ndim
        self._indices = None
        self._owner = None

    def append(self, image):
        if self._owner is not None:
            raise ValueError("cannot append to an image selection")
        if image is None and not self._pad_to_split:
            self._records.append(None)
            return
        image = np.ascontiguousarray(image, dtype=self.dtype)
        if image.ndim != self.ndim - 1:
            raise ValueError("disk array sample has the wrong number of dimensions")
        if self._pad_to_split and (image.ndim != 3 or image.shape[0] != 3):
            raise ValueError("disk image storage requires CHW RGB images")
        with self._lock:
            self._file.seek(0, 2)
            offset = self._file.tell()
            try:
                self._file.write(memoryview(image).cast("B"))
            except OSError as exc:
                raise OSError("Unable to write temporary image cache; check free disk space "
                              "or set MLAAS_IMAGE_CACHE_DIR to a larger drive") from exc
            self._records.append((offset, image.shape))
            self._image_shape = tuple(max(a, b) for a, b in zip(self._image_shape, image.shape))

    def __len__(self):
        return len(self._records) if self._indices is None else len(self._indices)

    @property
    def shape(self):
        return (len(self), *self._image_shape)

    def take(self, indices):
        """Select rows without reading pixels; all views share file ownership."""
        selected = np.arange(*indices.indices(len(self))) if isinstance(indices, slice) else np.asarray(indices)
        if selected.dtype.kind == "b":
            if selected.shape != (len(self),):
                raise IndexError("boolean image selection must match the pool length")
            selected = np.flatnonzero(selected)
        elif selected.size and selected.dtype.kind not in "iu":
            raise IndexError("image selection requires integer indices")
        selected = np.asarray(selected, dtype=np.int64).reshape(-1)
        # Validate bounds and resolve nested views with only O(rows) metadata.
        selected = np.where(selected < 0, selected + len(self), selected)
        if np.any((selected < 0) | (selected >= len(self))):
            raise IndexError("image selection is out of bounds")
        resolved = selected if self._indices is None else self._indices[selected]
        view = object.__new__(type(self))
        view._file, view._lock, view._records = self._file, self._lock, self._records
        view._image_shape = self._image_shape
        view.dtype, view.ndim = self.dtype, self.ndim
        view._indices = resolved
        view._owner = self if self._owner is None else self._owner
        return view

    def __getitem__(self, key):
        if not np.isscalar(key):
            selection = self.take(key)
            if not self._pad_to_split:
                return [selection[i] for i in range(len(selection))]
            batch = np.empty(selection.shape, dtype=self.dtype)
            for i in range(len(selection)):
                batch[i] = selection[i]
            return batch
        index = operator.index(key)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        actual = index if self._indices is None else int(self._indices[index])
        record = self._records[actual]
        if record is None:
            return None
        offset, shape = record
        count = int(np.prod(shape))
        with self._lock:
            self._file.seek(offset)
            raw = self._file.read(count * self.dtype.itemsize)
        if len(raw) != count * self.dtype.itemsize:
            raise OSError("temporary image cache is truncated")
        image = np.frombuffer(raw, dtype=self.dtype).reshape(shape)
        if not self._pad_to_split:
            return image.copy()
        # Match the original split-wide zero-padding, even for subset views.
        output = np.zeros(self._image_shape, dtype=self.dtype)
        output[:, :shape[1], :shape[2]] = image
        return output

    def __array__(self, dtype=None, copy=None):
        raise TypeError("disk arrays must be read in batches, not converted as a full split")

    def close(self):
        """Close a root pool after its consumers have finished (also closes views)."""
        self._file.close()


class DiskImageArray(DiskArrayStore):
    """Classification pixels with the original split-wide zero-padding."""


class DiskArraySequence(DiskArrayStore):
    """List-compatible batches of unpadded arrays, including report placeholders.

    Unlike classification's dense pool, structured images and masks retain each
    sample's exact shape for the task's batch encoder.
    """
    _pad_to_split = False


def close_image_stores(value):
    if isinstance(value, DiskArrayStore):
        value.close()
    elif isinstance(value, dict):
        for child in value.values():
            close_image_stores(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            close_image_stores(child)
