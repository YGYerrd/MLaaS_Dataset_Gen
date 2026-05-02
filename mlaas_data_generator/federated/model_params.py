from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def extract_model_parameters(
    *,
    model: Any | None = None,
    payload: Any | None = None,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSON-ready final model parameters grouped by parameter type."""
    buckets = {
        "weights": {},
        "biases": {},
        "alphas": {},
        "betas": {},
        "other_parameters": {},
    }

    if payload is not None:
        _collect_from_payload(payload, buckets)
        source = "payload"
    else:
        source = _collect_from_model(model, buckets)

    optimizer_parameters = _optimizer_parameters(model)
    training_parameters = _training_parameters(config, optimizer_parameters)

    return {
        "parameter_source": source,
        "metadata": dict(metadata or {}),
        "training_parameters": training_parameters,
        "optimizer_parameters": optimizer_parameters,
        **buckets,
        "summary": _summary(buckets),
    }


def write_final_model_parameters(
    *,
    output_dir: str | Path,
    run_id: str,
    model_role: str,
    model_id: str,
    round_idx: int | None,
    model_type: str | None,
    task_type: str | None,
    model: Any | None = None,
    payload: Any | None = None,
    pre_extracted: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    """Write one final model artifact and return its path.

    Hugging Face adapters are saved as checkpoint directories via
    ``save_pretrained``. Other models fall back to the legacy compact JSON
    parameter export.
    """
    checkpoint_path = _try_write_pretrained_checkpoint(
        output_dir=output_dir,
        run_id=run_id,
        model_role=model_role,
        model_id=model_id,
        round_idx=round_idx,
        model_type=model_type,
        task_type=task_type,
        model=model,
        config=config,
        metadata=metadata,
    )
    if checkpoint_path:
        return checkpoint_path

    params = dict(pre_extracted or extract_model_parameters(
        model=model,
        payload=payload,
        config=config,
        metadata=metadata,
    ))

    if not _has_exported_parameters(params):
        return None

    run_folder = Path(output_dir) / _safe_name(str(run_id))
    run_folder.mkdir(parents=True, exist_ok=True)

    role = _safe_name(str(model_role)).lower()
    if role == "global":
        path = run_folder / "global.json"
    elif role == "client":
        client_folder = run_folder / "clients"
        client_folder.mkdir(parents=True, exist_ok=True)
        path = client_folder / f"{_safe_name(str(model_id))}.json"
    else:
        path = run_folder / f"{role}__{_safe_name(str(model_id))}.json"
    document = {
        "run_id": run_id,
        "model_role": model_role,
        "model_id": model_id,
        "round": round_idx,
        "model_type": model_type,
        "task_type": task_type,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **params,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return str(path)


def _try_write_pretrained_checkpoint(
    *,
    output_dir: str | Path,
    run_id: str,
    model_role: str,
    model_id: str,
    round_idx: int | None,
    model_type: str | None,
    task_type: str | None,
    model: Any | None,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    pretrained_model = _find_pretrained_model(model)
    if pretrained_model is None:
        return None

    run_folder = Path(output_dir) / _safe_name(str(run_id))
    role = _safe_name(str(model_role)).lower()
    if role == "global":
        checkpoint_dir = run_folder / "global"
    elif role == "client":
        checkpoint_dir = run_folder / "clients" / _safe_name(str(model_id))
    else:
        checkpoint_dir = run_folder / f"{role}__{_safe_name(str(model_id))}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not _save_pretrained(pretrained_model, checkpoint_dir):
        return None

    saved_components = ["model"]
    for component_name, component in _pretrained_components(model):
        if _save_pretrained(component, checkpoint_dir):
            saved_components.append(component_name)

    document = {
        "run_id": run_id,
        "model_role": model_role,
        "model_id": model_id,
        "round": round_idx,
        "model_type": model_type,
        "task_type": task_type,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "artifact_type": "huggingface_pretrained",
        "format": "save_pretrained",
        "safe_serialization_requested": True,
        "saved_components": saved_components,
        "metadata": dict(metadata or {}),
        "training_parameters": _training_parameters(config, _optimizer_parameters(model)),
    }
    sidecar = checkpoint_dir / "_mlaas_metadata.json"
    sidecar.write_text(json.dumps(document, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return str(checkpoint_dir)


def _find_pretrained_model(model: Any | None) -> Any | None:
    if model is None:
        return None
    for candidate in _model_candidates(model):
        save_pretrained = getattr(candidate, "save_pretrained", None)
        if not callable(save_pretrained):
            continue
        # Tokenizers and processors also expose save_pretrained; prefer actual
        # models by requiring a state_dict-like or parameters-like surface.
        if callable(getattr(candidate, "state_dict", None)) or callable(getattr(candidate, "parameters", None)):
            return candidate
    return None


def _pretrained_components(model: Any | None) -> list[tuple[str, Any]]:
    if model is None:
        return []

    components: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def add(name: str, candidate: Any | None):
        if candidate is None or id(candidate) in seen:
            return
        if not callable(getattr(candidate, "save_pretrained", None)):
            return
        components.append((name, candidate))
        seen.add(id(candidate))

    core = getattr(model, "core", None)
    task_spec = getattr(core, "task_spec", None)
    for owner in (model, core, task_spec):
        if owner is None:
            continue
        add("tokenizer", getattr(owner, "tokenizer", None))
        add("processor", getattr(owner, "processor", None))
        add("image_processor", getattr(owner, "image_processor", None))
        add("image_processor", getattr(owner, "_image_processor", None))
        add("feature_extractor", getattr(owner, "feature_extractor", None))
    return components


def _save_pretrained(component: Any, checkpoint_dir: Path) -> bool:
    save_pretrained = getattr(component, "save_pretrained", None)
    if not callable(save_pretrained):
        return False
    try:
        save_pretrained(str(checkpoint_dir), safe_serialization=True)
        return True
    except Exception:
        # Older Transformers components may not accept safe_serialization, and
        # some environments may lack safetensors. A normal HF checkpoint is
        # still preferable to expanding tensors into JSON.
        try:
            save_pretrained(str(checkpoint_dir))
            return True
        except Exception:
            return False


def write_final_model_manifest(
    *,
    output_dir: str | Path,
    run_id: str,
    files: list[Mapping[str, Any]],
) -> str | None:
    if not files:
        return None
    run_folder = Path(output_dir) / _safe_name(str(run_id))
    run_folder.mkdir(parents=True, exist_ok=True)
    path = run_folder / "manifest.json"
    document = {
        "run_id": run_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return str(path)


def _collect_from_model(model: Any | None, buckets: dict[str, dict[str, Any]]) -> str:
    if model is None:
        return "none"

    if _collect_keras_variables(model, buckets):
        return "keras_variables"

    candidates = _model_candidates(model)
    for candidate in candidates:
        if _collect_state_dict(candidate, buckets):
            return "state_dict"

    for candidate in candidates:
        if _collect_estimator_attributes(candidate, buckets):
            return "estimator_attributes"

    if _collect_get_weights(model, buckets):
        return "get_weights"

    return "none"


def _model_candidates(model: Any) -> list[Any]:
    candidates = []
    seen = set()

    def add(candidate):
        if candidate is None or id(candidate) in seen:
            return
        candidates.append(candidate)
        seen.add(id(candidate))

    add(model)
    add(getattr(model, "core", None))
    add(getattr(getattr(model, "core", None), "model", None))
    add(getattr(model, "model", None))
    add(getattr(model, "estimator", None))
    add(getattr(model, "km", None))
    return candidates


def _collect_from_payload(payload: Any, buckets: dict[str, dict[str, Any]]) -> bool:
    if isinstance(payload, Mapping):
        found = False
        for key, value in payload.items():
            found = _add_parameter(str(key), value, buckets) or found
        return found

    if isinstance(payload, (list, tuple)):
        found = False
        for idx, value in enumerate(payload):
            found = _add_parameter(f"layer_{idx}", value, buckets) or found
        return found

    return _add_parameter("value", payload, buckets)


def _collect_keras_variables(model: Any, buckets: dict[str, dict[str, Any]]) -> bool:
    variables = getattr(model, "weights", None)
    if not variables:
        return False

    found = False
    for idx, variable in enumerate(list(variables)):
        name = getattr(variable, "name", None) or f"variable_{idx}"
        value = _variable_value(variable)
        found = _add_parameter(str(name), value, buckets) or found
    return found


def _collect_state_dict(model: Any, buckets: dict[str, dict[str, Any]]) -> bool:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        return False
    try:
        values = state_dict()
    except Exception:
        return False
    if not isinstance(values, Mapping):
        return False

    found = False
    for key, value in values.items():
        found = _add_parameter(str(key), value, buckets) or found
    return found


def _collect_estimator_attributes(model: Any, buckets: dict[str, dict[str, Any]]) -> bool:
    attrs = (
        "coef_",
        "intercept_",
        "cluster_centers_",
        "_centers",
        "feature_importances_",
        "classes_",
        "n_features_in_",
    )
    found = False
    for attr in attrs:
        if hasattr(model, attr):
            try:
                value = getattr(model, attr)
            except Exception:
                continue
            found = _add_parameter(attr, value, buckets) or found
    return found


def _collect_get_weights(model: Any, buckets: dict[str, dict[str, Any]]) -> bool:
    get_weights = getattr(model, "get_weights", None)
    if not callable(get_weights):
        return False
    try:
        weights = get_weights()
    except Exception:
        return False
    return _collect_from_payload(weights, buckets)


def _add_parameter(name: str, value: Any, buckets: dict[str, dict[str, Any]]) -> bool:
    arr = _to_numpy(value)
    if arr is None:
        return False
    bucket = _bucket_for(name, arr)
    buckets[bucket][name] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "values": arr.tolist(),
    }
    return True


def _to_numpy(value: Any) -> np.ndarray | None:
    value = _variable_value(value)
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    try:
        arr = arr.astype(np.float64, copy=False)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _variable_value(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach()
        except Exception:
            pass

    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        try:
            value = cpu()
        except Exception:
            pass

    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        try:
            return numpy_fn()
        except Exception:
            pass
    return value


def _bucket_for(name: str, arr: np.ndarray) -> str:
    lower = str(name or "").lower()
    if "bias" in lower or lower.endswith("/b:0") or lower.endswith(".b"):
        return "biases"
    if "beta" in lower:
        return "betas"
    if "alpha" in lower:
        return "alphas"
    if any(token in lower for token in ("weight", "kernel", "coef", "embedding", "center")):
        return "weights"
    if lower.startswith("layer_") and arr.ndim == 1:
        return "biases"
    if lower.startswith("layer_"):
        return "weights"
    if arr.ndim == 1:
        return "biases"
    return "weights"


def _optimizer_parameters(model: Any | None) -> dict[str, Any]:
    optimizer = _find_optimizer(model)
    if optimizer is None:
        return {}

    params = {}
    for key in ("learning_rate", "lr", "beta_1", "beta_2", "epsilon", "momentum", "weight_decay"):
        if not hasattr(optimizer, key):
            continue
        try:
            value = getattr(optimizer, key)
        except Exception:
            continue
        scalar = _scalar(value)
        if scalar is not None:
            params[key] = scalar
    return params


def _find_optimizer(model: Any | None) -> Any | None:
    if model is None:
        return None
    for candidate in _model_candidates(model):
        optimizer = getattr(candidate, "optimizer", None)
        if optimizer is not None:
            return optimizer
    return None


def _training_parameters(
    config: Mapping[str, Any] | None,
    optimizer_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    config = config or {}
    learning_rate = _first_scalar(
        optimizer_parameters.get("learning_rate"),
        optimizer_parameters.get("lr"),
        config.get("learning_rate"),
        config.get("lr"),
    )
    weight_decay = _first_scalar(
        optimizer_parameters.get("weight_decay"),
        config.get("weight_decay"),
    )
    beta_value = _first_scalar(
        weight_decay,
        optimizer_parameters.get("beta_1"),
        config.get("beta"),
    )

    out = {
        "alpha": learning_rate,
        "learning_rate": learning_rate,
        "beta": beta_value,
        "weight_decay": weight_decay,
        "optimizer": config.get("optimizer"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _first_scalar(*values: Any) -> Any:
    for value in values:
        scalar = _scalar(value)
        if scalar is not None:
            return scalar
    return None


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    value = _variable_value(value)
    try:
        arr = np.asarray(value)
    except Exception:
        return value if isinstance(value, (str, bool, int, float)) else None
    if arr.shape != ():
        return None
    item = arr.item()
    if isinstance(item, (np.integer,)):
        return int(item)
    if isinstance(item, (np.floating,)):
        item = float(item)
        return item if np.isfinite(item) else None
    if isinstance(item, (bool, int, float, str)):
        return item
    return None


def _summary(buckets: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts = {}
    total_tensors = 0
    total_elements = 0
    for bucket, values in buckets.items():
        count = len(values)
        counts[f"{bucket}_count"] = count
        total_tensors += count
        for entry in values.values():
            shape = entry.get("shape") or []
            elements = 1
            for dim in shape:
                elements *= int(dim)
            total_elements += int(elements)
    counts["total_tensors"] = total_tensors
    counts["total_elements"] = total_elements
    return counts


def _has_exported_parameters(params: Mapping[str, Any]) -> bool:
    summary = params.get("summary") if isinstance(params, Mapping) else None
    if isinstance(summary, Mapping) and int(summary.get("total_tensors", 0) or 0) > 0:
        return True
    return bool(params.get("training_parameters") or params.get("optimizer_parameters"))


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._")[:160] or "model"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
