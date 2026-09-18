"""One deterministic dataset-example budget for every evaluation task."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .config import CONFIG


# These generation-heavy evaluations are intentionally bounded below the
# global evaluation budget.  Keep the policy here so every manifest generator
# derives its task cap from the same global setting.
QUARTER_BENCHMARK_SAMPLE_TASKS = frozenset({
    "seq2seq_generation",
    "text2text_generation",
    "image_captioning",
    "captioning",
    "visual_question_answering",
    "vqa",
})
_BENCHMARK_CAP_KEYS = ("benchmark_sample_size", "max_benchmark_samples", "max_eval_samples")
_TASK_KEYS = ("hf_task", "task", "task_key", "task_type")


def _validated_integer(name: str, value, minimum: int) -> int:
    try:
        number = int(value)
        valid = not isinstance(value, bool) and float(value) == number and number >= minimum
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return number


def _uses_quarter_benchmark_cap(config: Mapping) -> bool:
    nested = config.get("dataset_args") or {}
    task = next((source.get(key) for key in _TASK_KEYS
                 for source in (config, nested) if source.get(key)), "")
    normalized_task = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized_task in QUARTER_BENCHMARK_SAMPLE_TASKS


def task_benchmark_sample_size(
    config: Mapping,
    benchmark_sample_size: int | None = None,
) -> int:
    """Return the effective task cap derived from the global evaluation cap."""
    if benchmark_sample_size is None:
        nested = config.get("dataset_args") or {}
        benchmark_sample_size = next((source[key] for source in (config, nested)
                                      for key in _BENCHMARK_CAP_KEYS
                                      if source.get(key) is not None), CONFIG["benchmark_sample_size"])
    cap = _validated_integer("benchmark_sample_size", benchmark_sample_size, 1)
    return max(1, cap // 4) if _uses_quarter_benchmark_cap(config) else cap


def evaluation_settings(config: Mapping) -> dict:
    """Resolve the cap and evaluation seed independently of training settings."""
    nested = config.get("dataset_args") or {}
    explicit_cap = next((source[key] for source in (config, nested)
                         for key in _BENCHMARK_CAP_KEYS
                         if source.get(key) is not None), None)
    cap = CONFIG["benchmark_sample_size"] if explicit_cap is None else explicit_cap
    seed = next((source[key] for source in (config, nested)
                 for key in ("evaluation_seed", "benchmark_sample_seed")
                 if source.get(key) is not None), CONFIG["evaluation_seed"])
    cap = _validated_integer("benchmark_sample_size", cap, 1)
    seed = _validated_integer("evaluation_seed", seed, 0)
    # Saved manifests may contain the old full global budget. Enforce the
    # current task ceiling while retaining smaller explicit caps. Taking min
    # also makes repeated resolution idempotent.
    if _uses_quarter_benchmark_cap(config):
        global_cap = _validated_integer("benchmark_sample_size", CONFIG["benchmark_sample_size"], 1)
        cap = min(cap, max(1, global_cap // 4))
    return {"benchmark_sample_size": cap, "evaluation_seed": seed}


def evaluation_indices(count: int, config: Mapping) -> tuple[np.ndarray, dict]:
    settings = evaluation_settings(config)
    cap, seed = settings["benchmark_sample_size"], settings["evaluation_seed"]
    # Preserve source order, including the complete split when it fits the cap.
    indices = (np.sort(np.random.default_rng(seed).choice(count, size=cap, replace=False))
               if count > cap else np.arange(count, dtype=int))
    return indices, {
        "requested_sample_size_total": cap,
        "available_sample_size_total": count,
        "effective_sample_size_total": len(indices),
        "sample_strategy_effective": "iid" if count > cap else "all",
        "sample_seed": seed,
        "evaluation_seed": seed,
    }
