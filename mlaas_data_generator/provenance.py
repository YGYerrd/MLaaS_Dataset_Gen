"""Immutable, generator-side provenance and artefact helpers.

This module deliberately records facts about one generated service.  It does
not decide whether two services can be combined; that policy belongs to a
consumer of the catalogue.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROVENANCE_VERSION = 1

_HF_WEIGHT_FILE = re.compile(
    r"^(?:pytorch_model(?:-\d{5}-of-\d{5})?\.bin|"
    r"model(?:-\d{5}-of-\d{5})?\.safetensors|"
    r"tf_model(?:\.h5)?|flax_model\.msgpack|"
    r"adapter_model(?:\.bin|\.safetensors)|model\.keras)$"
)


def _normalise(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, np.ndarray):
        return _normalise(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(_normalise(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_integrity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_file():
        return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size, "format": path.suffix.lstrip(".") or "file"}
    files = []
    total = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        size = child.stat().st_size
        total += size
        files.append({"path": child.relative_to(path).as_posix(), "sha256": sha256_file(child), "size_bytes": size})
    return {"sha256": sha256_json(files), "size_bytes": total, "format": "directory", "files": files}


def _cas_root(root: str | Path) -> Path:
    return Path(root).resolve() / "objects" / "sha256"


def _staging_root(root: str | Path) -> Path:
    path = Path(root).resolve() / "objects" / ".staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _new_stage(root: str | Path, prefix: str) -> Path:
    from .storage_overflow import choose_storage_root

    root = choose_storage_root(root)
    return Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=str(_staging_root(root))))


def _safe_suffix(value: str) -> str:
    value = value if value.startswith(".") else f".{value}"
    if not re.fullmatch(r"\.[A-Za-z0-9._-]+", value):
        raise ValueError(f"unsafe CAS file suffix: {value!r}")
    return value


def _publish_file_to_cas(root: str | Path, staged_file: Path, *, suffix: str) -> tuple[Path, dict[str, Any]]:
    """Atomically publish a file at an immutable path derived from its bytes."""
    integrity = path_integrity(staged_file)
    digest = str(integrity["sha256"])
    # Staging may be on the overflow volume. Publish on that same volume.
    root = staged_file.parent.parent.parent.parent
    target = _cas_root(root) / digest[:2] / f"{digest}{_safe_suffix(suffix)}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if path_integrity(target)["sha256"] != digest:
            raise RuntimeError(f"content-address collision at {target}")
        staged_file.unlink()
    else:
        try:
            # A hard-link gives create-if-absent semantics and never exposes a
            # partially copied target.  Both paths live beneath the same root.
            os.link(staged_file, target)
            staged_file.unlink()
        except FileExistsError:
            staged_file.unlink()
        except OSError:
            # Some filesystems do not permit hard links. os.replace still
            # publishes the completed staging file as one directory entry.
            try:
                os.replace(staged_file, target)
            except OSError:
                if not target.exists():
                    raise
                staged_file.unlink(missing_ok=True)
    published = path_integrity(target)
    if published["sha256"] != digest:
        raise RuntimeError(f"CAS verification failed after publishing {target}")
    return target, published | {"content_address": f"sha256:{digest}", "immutable": True}


def _publish_directory_to_cas(root: str | Path, staged_directory: Path) -> tuple[Path, dict[str, Any]]:
    """Atomically publish a directory tree under its canonical tree hash."""
    integrity = path_integrity(staged_directory)
    digest = str(integrity["sha256"])
    root = staged_directory.parent.parent.parent.parent
    target = _cas_root(root) / digest[:2] / f"{digest}.tree"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if path_integrity(target)["sha256"] != digest:
            raise RuntimeError(f"content-address collision at {target}")
        shutil.rmtree(staged_directory)
    else:
        try:
            os.replace(staged_directory, target)
        except OSError:
            if not target.exists():
                raise
            if path_integrity(target)["sha256"] != digest:
                raise RuntimeError(f"concurrent CAS publication conflict at {target}")
            shutil.rmtree(staged_directory, ignore_errors=True)
    published = path_integrity(target)
    if published["sha256"] != digest:
        raise RuntimeError(f"CAS verification failed after publishing {target}")
    return target, published | {"content_address": f"sha256:{digest}", "immutable": True}


def _artifact(
    artifact_type: str,
    path: Path,
    integrity: Mapping[str, Any],
    *,
    role: str,
    **metadata_values: Any,
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "artifact_uri": str(path),
        "metadata": dict(integrity) | {"role": role, **metadata_values},
    }


def _atomic_write_once(path: Path, payload: bytes) -> None:
    """Create a small immutable reference, accepting an identical existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable reference already contains different bytes: {path}")
        return
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            pass
        except OSError:
            try:
                exclusive = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                pass
            else:
                with os.fdopen(exclusive, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable reference publication conflict: {path}")
    finally:
        temp_path.unlink(missing_ok=True)


def _lineage_reference_path(root: str | Path, lineage_id: str) -> Path:
    # Keep user-provided identifiers out of path traversal while retaining a
    # readable prefix for operators inspecting the artefact tree.
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", str(lineage_id)).strip("._")[:48] or "lineage"
    identity = hashlib.sha256(str(lineage_id).encode("utf-8")).hexdigest()[:16]
    return Path(root).resolve() / "lineages" / f"{readable}-{identity}" / "initial-artifact.json"


def _read_lineage_reference(root: str | Path, lineage_id: str) -> dict[str, Any] | None:
    path = _lineage_reference_path(root, lineage_id)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    artifact = value.get("artifact") if isinstance(value, Mapping) else None
    if not isinstance(artifact, Mapping):
        raise RuntimeError(f"invalid lineage checkpoint reference: {path}")
    uri = Path(str(artifact.get("artifact_uri") or ""))
    expected = (artifact.get("metadata") or {}).get("sha256")
    if not uri.exists() or not expected or path_integrity(uri)["sha256"] != expected:
        raise RuntimeError(f"lineage checkpoint reference failed integrity validation: {path}")
    return dict(artifact)


def _write_lineage_reference(root: str | Path, lineage_id: str, artifact: Mapping[str, Any]) -> None:
    document = {"version": 1, "lineage_id": str(lineage_id), "artifact": _normalise(dict(artifact))}
    _atomic_write_once(_lineage_reference_path(root, lineage_id), canonical_json(document).encode("utf-8"))


def _model_candidates(model: Any) -> list[Any]:
    candidates, seen = [], set()
    for candidate in (model, getattr(model, "core", None), getattr(getattr(model, "core", None), "model", None), getattr(model, "model", None)):
        if candidate is not None and id(candidate) not in seen:
            candidates.append(candidate)
            seen.add(id(candidate))
    return candidates


def parameter_schema(model: Any) -> dict[str, Any]:
    """Return ordered state layout metadata without any tensor values."""
    for candidate in _model_candidates(model):
        state_dict = getattr(candidate, "state_dict", None)
        named_parameters = getattr(candidate, "named_parameters", None)
        if callable(state_dict):
            try:
                parameter_map = dict(named_parameters()) if callable(named_parameters) else {}
                parameter_names = set(parameter_map)
                entries = []
                for index, (name, value) in enumerate(state_dict().items()):
                    entries.append({
                        "index": index,
                        "name": str(name),
                        "kind": "parameter" if name in parameter_names else "buffer",
                        "shape": [int(dim) for dim in getattr(value, "shape", ())],
                        "dtype": str(getattr(value, "dtype", "unknown")),
                        "numel": int(getattr(value, "numel", lambda: np.asarray(value).size)()),
                        "trainable": bool(
                            getattr(parameter_map.get(name), "requires_grad", False)
                        ),
                    })
                return {"framework": "pytorch", "entries": entries}
            except Exception:
                pass
    for candidate in _model_candidates(model):
        weights = getattr(candidate, "weights", None)
        if weights:
            entries = []
            for index, value in enumerate(list(weights)):
                entries.append({
                    "index": index,
                    "name": str(
                        getattr(value, "path", None)
                        or getattr(value, "name", None)
                        or f"weight_{index}"
                    ),
                    "kind": "variable",
                    "shape": [int(dim) for dim in getattr(value, "shape", ())],
                    "dtype": str(getattr(value, "dtype", "unknown")),
                    "numel": int(np.prod(getattr(value, "shape", ()) or (1,))),
                    "trainable": bool(getattr(value, "trainable", False)),
                })
            return {"framework": "keras", "entries": entries}
    get_weights = getattr(model, "get_weights", None)
    if callable(get_weights):
        try:
            return {"framework": "array", "entries": [
                {"index": i, "name": f"weight_{i}", "kind": "array", "shape": list(np.asarray(value).shape),
                 "dtype": str(np.asarray(value).dtype), "numel": int(np.asarray(value).size), "trainable": None}
                for i, value in enumerate(get_weights())
            ]}
        except Exception:
            pass
    return {"framework": "unknown", "entries": []}


def model_state_schema(model: Any) -> dict[str, Any]:
    """Backward-compatible name for :func:`parameter_schema`.

    The result is a schema only. Canonical model-state values and their hashes
    are owned by ``checkpoint_store`` and are never included here.
    """
    return parameter_schema(model)


def model_identity(model: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _model_candidates(model)[-1] if _model_candidates(model) else model
    model_config = getattr(candidate, "config", None)
    to_dict = getattr(model_config, "to_dict", None)
    try:
        config_value = to_dict() if callable(to_dict) else dict(getattr(model_config, "__dict__", {}) or {})
    except Exception:
        config_value = {}
    result = {
        "framework": parameter_schema(model).get("framework"),
        "model_class": f"{candidate.__class__.__module__}.{candidate.__class__.__qualname__}" if candidate is not None else None,
        "model_type": config.get("model_type"),
        "model_id": config.get("hf_model_id") or config.get("model_id") or getattr(model, "model_id", None),
        "model_revision": config.get("hf_model_revision") or config.get("model_revision"),
        "task": config.get("hf_task") or config.get("task_type"),
        "model_config": config_value,
    }
    result["architecture_id"] = sha256_json(result)
    return result


def label_semantics(meta: Mapping[str, Any], *, task_family: str | None = None) -> dict[str, Any]:
    mapping = meta.get("id2label") or meta.get("label_mapping") or meta.get("label2id") or {}
    ordered: list[dict[str, Any]] = []
    if isinstance(mapping, Mapping):
        # label_mapping and label2id map label -> id; id2label maps id -> label.
        inverse = {}
        for key, value in mapping.items():
            try:
                if str(key).lstrip("-").isdigit():
                    inverse[int(key)] = str(value)
                else:
                    inverse[int(value)] = str(key)
            except (TypeError, ValueError):
                continue
        ordered = [{"index": idx, "label": inverse[idx]} for idx in sorted(inverse)]
    result = {
        "task_family": task_family or meta.get("task_type"),
        "label_format": meta.get("label_format"),
        "ignore_index": meta.get("ignore_index", meta.get("label_pad_value")),
        "labels": ordered,
        "num_labels": meta.get("num_labels", meta.get("num_classes")),
        "special_semantics": {
            key: meta.get(key) for key in ("segmentation_reduce_labels", "retrieval_positive_policy", "vqa_label_mode", "vqa_unseen_answer_policy") if meta.get(key) is not None
        },
    }
    result["label_map_hash"] = sha256_json(result)
    return result


def preprocessing_metadata(meta: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "hf_task", "loader_template", "input_shape", "x_keys", "schema", "max_length", "requested_max_length",
        "model_text_max_length", "max_length_adjusted", "dynamic_padding", "padding_mode", "channel_order",
        "training_augmentations", "eval_augmentations", "segmentation_reduce_labels", "ignore_index", "label_pad_value",
        "text_column", "tokens_column", "image_column", "label_column", "mask_column", "column_mapping",
        "mlm_probability", "masking_policy", "masking_seed",
    )
    result = {key: meta.get(key) for key in keys if meta.get(key) is not None}
    if meta.get("dynamic_padding"):
        # A selected union's padded width is an execution detail, not a family.
        result.pop("input_shape", None)
    if meta.get("shared_fitted_state") is not None:
        result["shared_fitted_state_hash"] = sha256_json(meta["shared_fitted_state"])
    result["preprocessing_contract_hash"] = sha256_json(result)
    return result


def logical_dataset_metadata(config: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    args = meta.get("dataset_args") if isinstance(meta.get("dataset_args"), Mapping) else config.get("dataset_args") or {}
    result = {
        "dataset": config.get("dataset"),
        "dataset_name": config.get("dataset_name") or meta.get("hf_id"),
        "dataset_config": config.get("dataset_config") or meta.get("hf_subset"),
        "dataset_revision": config.get("hf_dataset_revision") or meta.get("hf_dataset_revision"),
        "train_split": meta.get("train_split") or config.get("train_split"),
        "test_split": meta.get("test_split") or config.get("test_split"),
        "feature_schema": meta.get("schema"),
        "loader_args": args,
    }
    result["logical_dataset_contract_hash"] = sha256_json(result)
    return result


def dataset_metadata(config: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    result = logical_dataset_metadata(config, meta)
    result["sample_seed"] = config.get("sample_seed")
    result["dataset_contract_hash"] = sha256_json(result)
    return result


def runtime_provenance(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or Path(__file__).resolve().parents[1])
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip() or None
        except Exception:
            return None
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("torch", "transformers", "datasets", "tensorflow", "keras", "huggingface_hub"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    requirements = root / "requirements.txt"
    return {"captured_at": datetime.now(timezone.utc).isoformat(), "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "requirements_sha256": sha256_file(requirements) if requirements.exists() else None,
            "versions": versions}


def _write_cas_json_artifact(
    root: str | Path,
    artifact_type: str,
    document: Mapping[str, Any],
    *,
    role: str,
    **metadata_values: Any,
) -> dict[str, Any]:
    stage = _new_stage(root, "json")
    try:
        staged_file = stage / "document.json"
        staged_file.write_bytes(canonical_json(document).encode("utf-8"))
        path, integrity = _publish_file_to_cas(root, staged_file, suffix=".json")
        return _artifact(
            artifact_type,
            path,
            integrity,
            role=role,
            serialization="canonical-json-v1",
            **metadata_values,
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def write_json_artifact(root: str | Path, service_id: str, name: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """Persist canonical provenance JSON once, shared by all equal documents.

    ``service_id`` remains in the API because callers associate the returned
    artifact with a service in SQLite.  It intentionally does not affect the
    stored bytes or URI, allowing identical provenance to deduplicate.
    """
    del service_id
    return _write_cas_json_artifact(root, name, document, role="provenance", logical_name=name)


def _remove_model_weight_files(folder: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        if _HF_WEIGHT_FILE.fullmatch(path.name) or path.name in {
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        }:
            removed.append(path.relative_to(folder).as_posix())
            path.unlink()
    for path in sorted((item for item in folder.rglob("*") if item.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass
    return removed


def _save_support_component(
    root: str | Path,
    artifact_type: str,
    component_name: str,
    component: Any,
) -> dict[str, Any] | None:
    stage = _new_stage(root, f"support-{component_name}")
    payload = stage / "payload"
    payload.mkdir()
    try:
        saved = False
        save_pretrained = getattr(component, "save_pretrained", None)
        if callable(save_pretrained):
            try:
                save_pretrained(str(payload))
                saved = True
            except Exception:
                saved = False
        if not saved:
            to_dict = getattr(component, "to_dict", None)
            if callable(to_dict):
                try:
                    (payload / "config.json").write_bytes(canonical_json(to_dict()).encode("utf-8"))
                    saved = True
                except Exception:
                    saved = False
        removed = _remove_model_weight_files(payload)
        if not saved or not any(item.is_file() for item in payload.rglob("*")):
            return None
        path, integrity = _publish_directory_to_cas(root, payload)
        return _artifact(
            artifact_type,
            path,
            integrity,
            role="reconstruction_support",
            component=component_name,
            serialization="save_pretrained_config_bundle",
            excluded_model_weight_files=removed,
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def save_reconstruction_support_artifacts(
    root: str | Path,
    service_id: str,
    model: Any,
) -> list[dict[str, Any]]:
    """Save small model-construction assets separately from tensor state.

    Hugging Face configuration/tokenizer/processor bundles are persisted via
    their native ``save_pretrained`` methods with known model-weight payloads
    excluded.  Keras architecture JSON is captured when ``to_json`` is
    available.  Each returned value can be written directly to
    ``service_artifacts``.
    """
    del service_id
    artifacts: list[dict[str, Any]] = []
    candidates = _model_candidates(model)

    configured: set[str] = set()
    for candidate in reversed(candidates):
        for attribute, artifact_type, component_name in (
            ("config", "model_configuration", "model_config"),
            ("generation_config", "generation_configuration", "generation_config"),
        ):
            component = getattr(candidate, attribute, None)
            if component is None or component_name in configured:
                continue
            artifact = _save_support_component(root, artifact_type, component_name, component)
            if artifact:
                artifacts.append(artifact)
                configured.add(component_name)

    seen_objects: set[int] = set()
    seen_component_names: set[str] = set()
    component_specs = (
        ("tokenizer", "tokenizer_configuration", "tokenizer"),
        ("processor", "processor_configuration", "processor"),
        ("image_processor", "image_processor_configuration", "image_processor"),
        ("_image_processor", "image_processor_configuration", "image_processor"),
        ("feature_extractor", "feature_extractor_configuration", "feature_extractor"),
    )
    for owner in candidates:
        for attribute, artifact_type, component_name in component_specs:
            component = getattr(owner, attribute, None)
            if component is None or id(component) in seen_objects or component_name in seen_component_names:
                continue
            artifact = _save_support_component(root, artifact_type, component_name, component)
            seen_objects.add(id(component))
            if artifact:
                artifacts.append(artifact)
                seen_component_names.add(component_name)

    for candidate in candidates:
        to_json = getattr(candidate, "to_json", None)
        if not callable(to_json):
            continue
        try:
            raw = to_json()
            architecture = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(architecture, Mapping):
                architecture = {"keras_architecture": architecture}
            artifacts.append(
                _write_cas_json_artifact(
                    root,
                    "keras_architecture",
                    architecture,
                    role="reconstruction_support",
                    component="keras_architecture",
                )
            )
            break
        except Exception:
            continue
    return artifacts


def _legacy_initial_artifact(root: str | Path, lineage_id: str) -> dict[str, Any] | None:
    value = str(lineage_id)
    if Path(value).name != value:
        return None
    folder = Path(root).resolve() / "lineages" / value / "initial"
    if folder.is_dir() and any(folder.iterdir()):
        return _artifact("model_initial_checkpoint", folder, path_integrity(folder), role="initial", legacy_layout=True)
    return None


def _finish_model_artifact(
    root: str | Path,
    role: str,
    lineage_id: str | None,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    if role == "initial" and lineage_id:
        _write_lineage_reference(root, lineage_id, artifact)
    return artifact


def save_model_artifact(
    root: str | Path,
    service_id: str,
    role: str,
    model: Any,
    *,
    lineage_id: str | None = None,
) -> dict[str, Any] | None:
    """Persist an immutable native full-object checkpoint in the CAS.

    This remains the reconstruction fallback for models that cannot use the
    exact shared-base tensor patch store.  Tokenizers and processors are
    deliberately excluded and belong to
    :func:`save_reconstruction_support_artifacts`.
    """
    del service_id
    if role == "initial" and lineage_id:
        existing = _read_lineage_reference(root, lineage_id) or _legacy_initial_artifact(root, lineage_id)
        if existing:
            return existing

    artifact_type = f"model_{role}_checkpoint"
    for candidate in _model_candidates(model):
        save_pretrained = getattr(candidate, "save_pretrained", None)
        if not (callable(save_pretrained) and callable(getattr(candidate, "state_dict", None))):
            continue
        stage = _new_stage(root, "hf-model")
        payload = stage / "payload"
        payload.mkdir()
        try:
            try:
                save_pretrained(str(payload), safe_serialization=True)
                serialization = "transformers-save-pretrained-safetensors"
            except TypeError:
                save_pretrained(str(payload))
                serialization = "transformers-save-pretrained"
            if not any(item.is_file() for item in payload.rglob("*")):
                continue
            path, integrity = _publish_directory_to_cas(root, payload)
            artifact = _artifact(
                artifact_type,
                path,
                integrity,
                role=role,
                storage_mode="full_object",
                serialization=serialization,
                reconstruction_scope="model_state_and_model_config",
                saved_components=["model"],
            )
            return _finish_model_artifact(root, role, lineage_id, artifact)
        except Exception:
            continue
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    for candidate in _model_candidates(model):
        save = getattr(candidate, "save", None)
        if not callable(save) or not (callable(getattr(candidate, "to_json", None)) or hasattr(candidate, "weights")):
            continue
        stage = _new_stage(root, "keras-model")
        staged_file = stage / "model.keras"
        try:
            save(str(staged_file))
            if not staged_file.is_file():
                continue
            path, integrity = _publish_file_to_cas(root, staged_file, suffix=".keras")
            artifact = _artifact(
                artifact_type,
                path,
                integrity,
                role=role,
                storage_mode="full_object",
                serialization="keras-v3-native",
                reconstruction_scope="full_model",
                container_compression="framework_defined",
            )
            return _finish_model_artifact(root, role, lineage_id, artifact)
        except Exception:
            continue
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        for candidate in _model_candidates(model):
            if not isinstance(candidate, torch.nn.Module):
                continue
            stage = _new_stage(root, "torch-model")
            staged_file = stage / "model.pt"
            try:
                torch.save(candidate, staged_file)
                path, integrity = _publish_file_to_cas(root, staged_file, suffix=".pt")
                artifact = _artifact(
                    artifact_type,
                    path,
                    integrity,
                    role=role,
                    storage_mode="full_object",
                    serialization="torch-save-full-module",
                    reconstruction_scope="full_model",
                    unsafe_deserialization=True,
                )
                return _finish_model_artifact(root, role, lineage_id, artifact)
            except Exception:
                continue
            finally:
                shutil.rmtree(stage, ignore_errors=True)

    stage = _new_stage(root, "joblib-model")
    payload = stage / "payload"
    payload.mkdir()
    try:
        import joblib

        entrypoint = payload / "model.joblib"
        joblib.dump(model, entrypoint, compress=0)
        if not entrypoint.exists():
            return None
        path, integrity = _publish_directory_to_cas(root, payload)
        artifact = _artifact(
            artifact_type,
            path,
            integrity,
            role=role,
            storage_mode="full_object",
            serialization="joblib",
            reconstruction_scope="full_python_object",
            entrypoint="model.joblib",
            unsafe_deserialization=True,
            compression_codec="none",
        )
        return _finish_model_artifact(root, role, lineage_id, artifact)
    except Exception:
        return None
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _copy_loaded_model_state(model: Any, loaded: Any) -> bool:
    get_weights = getattr(loaded, "get_weights", None)
    set_weights = getattr(model, "set_weights", None)
    if callable(get_weights) and callable(set_weights):
        try:
            set_weights(get_weights())
            return True
        except Exception:
            pass
    if model.__class__ is loaded.__class__ and hasattr(model, "__dict__") and hasattr(loaded, "__dict__"):
        model.__dict__.clear()
        model.__dict__.update(loaded.__dict__)
        return True
    return False


def restore_model_artifact(artifact: Mapping[str, Any], model: Any) -> bool:
    """Restore one trusted native full-object artefact into ``model``.

    The artefact's container hash is verified before any framework loader is
    called.  This function is intentionally for generator-created artefacts;
    joblib and full-module PyTorch formats execute trusted Python code while
    loading and must not be used for untrusted input.
    """
    path = Path(str(artifact.get("artifact_uri") or ""))
    metadata_value = artifact.get("metadata") or {}
    if not isinstance(metadata_value, Mapping):
        return False
    expected = metadata_value.get("sha256")
    try:
        if not path.exists() or not expected or path_integrity(path)["sha256"] != expected:
            return False
    except Exception:
        return False
    serialization = str(metadata_value.get("serialization") or "")

    if serialization.startswith("transformers-save-pretrained") and path.is_dir():
        for candidate in _model_candidates(model):
            loader = getattr(candidate.__class__, "from_pretrained", None)
            load_state_dict = getattr(candidate, "load_state_dict", None)
            if not (callable(loader) and callable(load_state_dict)):
                continue
            try:
                loaded = loader(str(path), local_files_only=True)
                try:
                    result = load_state_dict(loaded.state_dict(), strict=True)
                except TypeError:
                    result = load_state_dict(loaded.state_dict())
                if getattr(result, "missing_keys", None) or getattr(result, "unexpected_keys", None):
                    continue
                return True
            except Exception:
                continue

    if serialization == "keras-v3-native" and path.is_file():
        try:
            try:
                from keras.models import load_model
            except Exception:
                from tensorflow.keras.models import load_model
            return _copy_loaded_model_state(model, load_model(str(path), compile=False))
        except Exception:
            return False

    if serialization == "torch-save-full-module" and path.is_file():
        try:
            import torch
            try:
                loaded = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                loaded = torch.load(path, map_location="cpu")
            return _copy_loaded_model_state(model, loaded)
        except Exception:
            return False

    if serialization == "joblib" and path.is_dir():
        try:
            import joblib
            loaded = joblib.load(path / str(metadata_value.get("entrypoint") or "model.joblib"))
            return _copy_loaded_model_state(model, loaded)
        except Exception:
            return False
    return False


def restore_lineage_initial_weights(root: str | Path, lineage_id: str, model: Any) -> bool:
    """Restore a legacy array checkpoint or a native CAS lineage checkpoint."""
    lineage_value = str(lineage_id)
    legacy_path: Path | None = (
        Path(root).resolve() / "lineages" / lineage_value / "initial" / "weights.npz"
        if Path(lineage_value).name == lineage_value
        else None
    )
    set_weights = getattr(model, "set_weights", None)
    if legacy_path is not None and legacy_path.exists() and callable(set_weights):
        try:
            with np.load(legacy_path, allow_pickle=False) as data:
                values = [data[key] for key in sorted(data.files, key=lambda key: int(key.split("_")[-1]))]
            set_weights(values)
            return True
        except Exception:
            return False

    try:
        artifact = _read_lineage_reference(root, lineage_id)
    except Exception:
        return False
    if not artifact:
        return False
    return restore_model_artifact(artifact, model)
