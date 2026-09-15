"""Sample quantities are an independent, fixed-spread client axis."""
from __future__ import annotations

import numpy as np

from ..provenance import sha256_json

SAMPLE_QUANTITY_SPREAD = 0.75
QUANTITY_SAMPLER_VERSION = "stratified_uniform_quantity_v1"


def sample_quantity_bounds(average_samples: int) -> tuple[int, int]:
    if average_samples < 1:
        raise ValueError("average_samples must be positive")
    # Tiny averages still need nonempty client partitions.
    return (max(1, round(average_samples * (1 - SAMPLE_QUANTITY_SPREAD))),
            max(1, round(average_samples * (1 + SAMPLE_QUANTITY_SPREAD))))


def stratified_sample_quantities(average_samples: int, *, count: int, seed: str) -> list[int]:
    """Round each equal-width stratum's midpoint, then shuffle across client IDs.

    Midpoints give uniform coverage centred on the requested average without
    finite-cohort drift. Integer rounding may repeat quantities in small ranges.
    """
    minimum, maximum = sample_quantity_bounds(average_samples)
    if count < 1:
        raise ValueError("client count must be positive")
    if count == 1:
        return [int(average_samples)]
    quantities = [round(minimum + (i + 0.5) * (maximum - minimum) / count)
                  for i in range(count)]
    rng = np.random.default_rng(int(sha256_json([seed, QUANTITY_SAMPLER_VERSION])[:8], 16))
    rng.shuffle(quantities)
    return quantities
