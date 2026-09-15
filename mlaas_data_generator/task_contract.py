"""Resolve the label semantics of sentence pairs from their dataset contract."""
from __future__ import annotations

from .tasks import TaskContract, TaskContractError, resolve_task_contract


def resolve_sentence_pair_contract(config, meta=None, *, dataset_spec=None):
    from .family_preflight import registered_dataset_spec

    spec = registered_dataset_spec(config) if dataset_spec is None else dataset_spec
    template = spec.get("loader_template") or config.get("loader_template")
    pair = (spec.get("task_key") == "sentence_similarity"
            or str(config.get("hf_task", "")).replace("-", "_") == "sentence_similarity"
            or template in {"hf_sentence_pair_classification", "hf_text_similarity"})
    if not pair:
        return config
    declared = spec.get("task_type")
    if declared is None and template == "hf_sentence_pair_classification":
        declared = "classification"
    loaded = (meta or {}).get("task_type")
    if declared and loaded and loaded != declared:
        raise ValueError(f"sentence-pair dataset contract is {declared}, but preprocessing produced {loaded} labels")
    task_type = declared or loaded
    if task_type not in {"classification", "regression"}:
        return config  # Unregistered pairs are resolved once the loader has a schema.
    regression = task_type == "regression"
    values = {"task_type": task_type, "label_format": "continuous" if regression else "single_index",
              "output_schema": "continuous" if regression else "single_index",
              "label_mode": "regression" if regression else "categorical",
              "skew_axis": "score_bin" if regression else "class_label",
              "task_resolution_source": "dataset_contract" if declared else "loaded_label_schema"}
    if regression:
        values.update(num_labels=1, num_classes=None)
    else:
        count = spec.get("num_classes") or (meta or {}).get("num_classes") or (meta or {}).get("num_labels")
        if count is not None:
            values.update(num_classes=int(count), num_labels=int(count))
        values["skew_axis_config"] = None
    config.update(values)
    if isinstance(config.get("dataset_args"), dict):
        config["dataset_args"].update({key: values[key] for key in ("task_type", "label_mode", "label_format")})
    if meta is not None:
        meta.update({key: value for key, value in values.items() if key not in {"skew_axis", "skew_axis_config"}})
    return config
