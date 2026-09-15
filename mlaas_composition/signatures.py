"""Read and compare already-generated update signatures during post-processing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def load_signature_vector(service_or_path: Mapping[str, Any] | str | Path) -> np.ndarray | None:
    """Load a normalised update-signature vector without generating one."""
    if isinstance(service_or_path, Mapping):
        path_value = service_or_path.get("update_signature_path") or service_or_path.get("signature_path")
    else:
        path_value = service_or_path
    if path_value is None:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as data:
                for key in ("update_signature", "signature", "vector"):
                    if key in data:
                        return _normalise_vector(data[key])
        if path.suffix.lower() == ".npy":
            return _normalise_vector(np.load(path, allow_pickle=False))
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, Mapping):
        for key in ("update_signature", "signature", "vector"):
            if key in payload:
                return _normalise_vector(payload[key])
    return _normalise_vector(payload) if isinstance(payload, list) else None


def compute_composition_mus(services: Sequence[Mapping[str, Any]]) -> float:
    """Compute mean update-signature similarity for an already selected composition."""
    vectors = [vector for service in services if (vector := load_signature_vector(service)) is not None and vector.size]
    if not vectors:
        return 0.5
    if len(vectors) == 1:
        return 1.0
    dim = max(vector.size for vector in vectors)
    matrix = np.vstack([_pad_vector(vector, dim) for vector in vectors])
    reference = matrix.mean(axis=0)
    norm = float(np.linalg.norm(reference))
    if norm <= 0.0:
        return 0.0
    return float(np.clip(np.mean(matrix @ (reference / norm)), 0.0, 1.0))


def _normalise_vector(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if not vector.size or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    return np.zeros(vector.shape, dtype=np.float32) if norm <= 0.0 else (vector / norm).astype(np.float32)


def _pad_vector(vector: np.ndarray, dim: int) -> np.ndarray:
    if vector.size == dim:
        return vector
    result = np.zeros(dim, dtype=np.float32)
    result[: vector.size] = vector
    return result
