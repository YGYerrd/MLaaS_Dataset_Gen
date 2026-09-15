"""Conservative retrieval starting batches for a 16 GiB training budget.

These are starting points, not a VRAM guarantee. Actual activations, precision,
optimizer state and other allocations determine whether runtime fallback is needed.
"""
from math import ceil


MIN_CONTRASTIVE_PAIRS = 2


def retrieval_batch_target(params_m=427.0, image_size=224):
    # Leave room for FP32 weights/gradients/Adam moments on the 16 GiB baseline.
    baseline = 4 if params_m > 700 else 8 if params_m > 250 else 16
    target = baseline
    area_factor = max(1.0, (float(image_size) / 224.0) ** 2)
    while target > MIN_CONTRASTIVE_PAIRS and target * area_factor > baseline:
        target //= 2
    return target


def contrastive_ranges(count, capacity):
    """Partition without dropping, repeating or forwarding a singleton pair."""
    if count < MIN_CONTRASTIVE_PAIRS or capacity < MIN_CONTRASTIVE_PAIRS or ceil(count / capacity) > count // MIN_CONTRASTIVE_PAIRS:
        raise ValueError(
            "insufficient_contrastive_batch: at least two pairs are required per "
            f"forward; cannot partition {count} pairs with capacity {capacity}. "
            "Use a larger physical batch or a partition with an even number of pairs."
        )
    ranges = []
    start = 0
    while start < count:
        stop = min(count, start + capacity)
        if count - stop == 1:
            stop -= 1
        ranges.append((start, stop))
        start = stop
    return ranges
