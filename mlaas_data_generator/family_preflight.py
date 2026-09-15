"""Shared, offline dataset checks for independent checkpoint families.

Registry counts are early exclusion evidence, not a replacement for checking the
loaded evaluation set. These rules describe the family protocol, not model IDs.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

# Benchmark sizes are preferences: use the available evaluation source. Only
# require enough rows for non-empty, disjoint catalogue and audit partitions.
MIN_EVALUATION_EXAMPLES = 2


def _text(value: Any, default: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return default if text.lower() in {"", "none", "nan"} else text


def independent_family_applies(config: Mapping[str, Any]) -> bool:
    return (
        _text(config.get("generation_mode"), "independent") == "independent"
        and _text(config.get("training_regime"), "finetune_transfer") not in {"inference", "inference_only"}
        and config.get("model_type") in {"hf_finetune", "hf_train", "transformers_finetune", "mlp", "cnn", "logreg"}
    )


def evaluation_size_error(count: int, *, minimum: int = MIN_EVALUATION_EXAMPLES,
                          split: str = "test", stage: str = "") -> str | None:
    if count >= minimum:
        return None
    return (
        f"at least {minimum} evaluation examples are required for catalogue and audit data; "
        f"split={split} contains {count}" + (f" {stage}" if stage else "")
    )


def hf_split_error(train_split: Any = None, test_split: Any = None) -> str | None:
    train = _text(train_split, "train")
    test = _text(test_split, "test")
    if not re.fullmatch(r"[\w.-]+", train) or train == "train_test_split":
        return "compatible HF services require the full official train split"
    sources = [part.strip().split("[", 1)[0] for part in test.split("+")]
    if test == "train_test_split" or train in sources:
        return "compatible HF services require evaluation data from a separate official split; training and evaluation sources overlap"
    if len(sources) != len(set(sources)):
        return "compatible HF services cannot repeat an evaluation source; use one disjoint evaluation split"
    return None


def hf_dataset_error(config: Mapping[str, Any], *, dataset_spec: Mapping[str, Any] | None = None) -> str | None:
    train = _text(config.get("train_split"), "train")
    test = _text(config.get("test_split"), _text(config.get("benchmark_split"), "test"))
    error = hf_split_error(train, test)
    if error:
        return error
    # Only use counts for the actual selected split; never infer capacity from
    # client sample_size, which is a requested quantity rather than a source cardinality.
    spec = dataset_spec or {}
    if test in spec.get("unlabeled_splits", ()):
        return f"compatible HF services require labeled evaluation data; split={test} is declared unlabeled"
    counts = spec.get("split_sizes") or {}
    count = counts.get(test)
    if count is None and test == _text(spec.get("test_split")):
        count = spec.get("benchmark_examples")
    if count is not None:
        return evaluation_size_error(int(count), split=test, stage="in registry metadata")
    return None


def registered_dataset_spec(config: Mapping[str, Any]) -> Mapping[str, Any]:
    from .registry import DATASET_REGISTRY

    name = _text(config.get("dataset_name")).lower()
    subset = _text(config.get("dataset_config")).lower()
    candidates = [spec for spec in DATASET_REGISTRY.values()
                  if _text(spec.get("dataset_name")).lower() == name
                  and _text(spec.get("dataset_config")).lower() == subset]
    # A dataset can have several task views. Prefer its matching declared view.
    return next((s for s in candidates if s.get("task_key") == config.get("task")),
                candidates[0] if candidates else {})


def manifest_family_error(config: Mapping[str, Any]) -> str | None:
    if not independent_family_applies(config) or _text(config.get("dataset")).lower() != "hf":
        return None
    return hf_dataset_error(config, dataset_spec=registered_dataset_spec(config))
