"""Tokenizer-independent character lengths for raw fill-mask partitions."""
from collections.abc import Mapping

import numpy as np

AXIS = "raw_text_length_bucket"
LENGTH_FIELD = "raw_text_char_length"
UPPER_BOUNDS = (0, 64, 256, 1024, 4096)
BUCKET_LABELS = {"0": "empty", "1": "1-64 chars", "2": "65-256 chars",
                 "3": "257-1024 chars", "4": "1025-4096 chars", "5": "4097+ chars"}


def configured_skew_axis(config):
    axis = config.get("skew_axis")
    task = str(config.get("hf_task") or config.get("task") or "").lower().replace("-", "_")
    return AXIS if not axis and task in {"fill_mask", "masked_lm"} else axis


def text_column(meta):
    return meta.get("text_column") or (meta.get("dataset_args") or {}).get("text_column") or "text"


def raw_text_lengths(dataset, column, *, batch_size=256):
    """Scan only the text column in bounded batches; never create model tokens."""
    if hasattr(dataset, "column_names"):
        if column not in dataset.column_names:
            return None
        count = len(dataset)
        if hasattr(dataset, "select_columns"):
            dataset = dataset.select_columns([column])
        batch = lambda start, end: dataset[start:end][column]
    elif isinstance(dataset, Mapping) and column in dataset:
        count = len(dataset[column])
        batch = lambda start, end: dataset[column][start:end]
    else:
        return None
    lengths = np.empty(count, dtype=np.int64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        values = batch(start, end)
        if len(values) != end - start:
            raise ValueError(f"{AXIS} raw text batches do not align with source rows")
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f"{AXIS} requires string values in text_column='{column}'")
        lengths[start:start + len(values)] = [len(value) for value in values]
    return lengths


def with_raw_text_lengths(dataset, meta, *, split, column):
    """Reuse shared raw counts and carry selected counts through preparation."""
    sidecars = meta.get("split_sidecars") or {}
    payload = sidecars.get(split) or {}
    lengths = payload.get(LENGTH_FIELD)
    if lengths is None or meta.get("raw_text_length_column") != column:
        lengths = raw_text_lengths(dataset, column)
    if lengths is None or len(lengths) != len(dataset):
        raise ValueError(f"{AXIS} requires an aligned raw text_column='{column}'")
    return {**meta, "raw_text_length_column": column, "split_sidecars": {
        **sidecars, split: {**payload, LENGTH_FIELD: lengths}}}
