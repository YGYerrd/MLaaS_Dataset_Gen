"""Capacity facts and deterministic quantity scaling; no dataset downloads."""
from __future__ import annotations

import math

CAPACITY_POLICY = "capacity_scaled_quantity_v1"
REALIZED_TRAINING_POLICY = "realized_training_opportunity_v1"
SYNTHETIC_SAMPLE_COUNT = 2000


def known_training_capacity(config):
    from ..family_preflight import registered_dataset_spec

    name = config.get("dataset_name") or config.get("dataset")
    if name == "synthetic":
        test_size = (config.get("dataset_args") or {}).get("test_size", config.get("test_size", 0.2))
        test_count = math.ceil(SYNTHETIC_SAMPLE_COUNT * test_size) if isinstance(test_size, float) else int(test_size)
        return max(0, SYNTHETIC_SAMPLE_COUNT - test_count)
    spec = registered_dataset_spec(config)
    # Client quantities and train_examples are workload hints, not cardinality facts.
    count = (spec.get("split_sizes") or {}).get(config.get("train_split") or "train")
    return int(count) if count is not None else None


def scale_quantity(quantity, maximum, capacity):
    if capacity is None or capacity >= maximum:
        return int(quantity), int(maximum)
    if capacity < 1:
        raise ValueError("no usable training examples are available")
    return max(1, min(int(capacity), round(quantity * capacity / maximum))), int(capacity)
