from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_SIGNATURE_DIM = 256


def compute_and_store_update_signature(
    before: Any,
    after: Any,
    *,
    output_dir: str | Path,
    run_id: str,
    round_idx: int,
    client_id: str,
    dim: int = DEFAULT_SIGNATURE_DIM,
    seed: int = 42,
    max_source_elements: int | None = None,
) -> dict[str, Any]:
    """Persist a compressed, normalised model-update direction.

    The saved vector is a CountSketch-style random projection of
    ``after - before``. It is intended for later cosine comparisons, not for
    reconstructing model parameters.
    """
    signature = compute_update_signature(
        before,
        after,
        dim=dim,
        seed=seed,
        max_source_elements=max_source_elements,
    )
    if signature is None:
        return {}

    vector = signature["vector"]
    signature_id = _signature_id(
        vector=vector,
        run_id=run_id,
        round_idx=round_idx,
        client_id=client_id,
        source_dim=signature["source_dim"],
        source_norm=signature["source_norm"],
    )

    run_dir = Path(output_dir) / _safe_name(str(run_id)) / f"round_{int(round_idx)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{_safe_name(str(client_id))}_{signature_id}.npz"

    np.savez_compressed(
        path,
        update_signature=vector.astype(np.float32, copy=False),
        signature_norm=np.asarray(signature["source_norm"], dtype=np.float64),
        source_dim=np.asarray(signature["source_dim"], dtype=np.int64),
        layer_count=np.asarray(signature["layer_count"], dtype=np.int64),
        method=np.asarray("count_sketch_random_projection"),
    )

    return {
        "update_signature_id": signature_id,
        "signature_dim": int(vector.size),
        "signature_norm": float(signature["source_norm"]),
        "update_signature_path": str(path),
        "update_signature_method": "count_sketch_random_projection",
        "update_signature_source_dim": int(signature["source_dim"]),
        "update_signature_layer_count": int(signature["layer_count"]),
    }


def compute_update_signature(
    before: Any,
    after: Any,
    *,
    dim: int = DEFAULT_SIGNATURE_DIM,
    seed: int = 42,
    max_source_elements: int | None = None,
) -> dict[str, Any] | None:
    """Return a compressed update direction for compatible weight payloads."""
    dim = int(dim or DEFAULT_SIGNATURE_DIM)
    if dim <= 0:
        raise ValueError("update signature dim must be positive")

    left = _as_named_arrays(before)
    right = _as_named_arrays(after)
    if not right:
        return None
    if not left:
        left = {key: np.zeros_like(value, dtype=np.float64) for key, value in right.items()}

    common = [
        key
        for key in sorted(left.keys())
        if key in right and left[key].shape == right[key].shape
    ]
    if not common:
        return None

    projected = np.zeros(dim, dtype=np.float64)
    squared_sum = 0.0
    source_dim = 0
    layer_count = 0
    remaining = None if max_source_elements is None else max(0, int(max_source_elements))

    for key in common:
        if remaining is not None and remaining <= 0:
            break

        before_arr = left[key]
        after_arr = right[key]
        if before_arr.size == 0:
            continue

        diff = (after_arr - before_arr).reshape(-1)
        if remaining is not None and diff.size > remaining:
            diff = diff[:remaining]
        if diff.size == 0:
            continue

        squared_sum += float(np.dot(diff, diff))
        source_dim += int(diff.size)
        layer_count += 1
        _project_delta_into(diff, projected, key=key, seed=seed)

        if remaining is not None:
            remaining -= int(diff.size)

    if source_dim <= 0:
        return None

    source_norm = float(math.sqrt(max(squared_sum, 0.0)))
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm > 0.0:
        projected = projected / projected_norm

    return {
        "vector": projected.astype(np.float32),
        "source_norm": source_norm,
        "source_dim": int(source_dim),
        "layer_count": int(layer_count),
    }


def _as_named_arrays(weights: Any) -> dict[str, np.ndarray] | None:
    if weights is None:
        return None

    if isinstance(weights, Mapping):
        out = {}
        for key, value in weights.items():
            arr = _coerce_numeric_array(value)
            if arr is not None:
                out[str(key)] = arr
        return out or None

    if isinstance(weights, (list, tuple)):
        out = {}
        for idx, value in enumerate(weights):
            arr = _coerce_numeric_array(value)
            if arr is not None:
                out[f"layer_{idx}"] = arr
        return out or None

    arr = _coerce_numeric_array(weights)
    if arr is None:
        return None
    return {"value": arr}


def _coerce_numeric_array(value: Any) -> np.ndarray | None:
    value = _tensor_to_numpy(value)
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


def _tensor_to_numpy(value: Any) -> Any:
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


def _project_delta_into(diff: np.ndarray, projected: np.ndarray, *, key: str, seed: int) -> None:
    dim = int(projected.size)
    rng = np.random.default_rng(_stable_seed(seed, key))
    chunk_size = 131_072
    for start in range(0, int(diff.size), chunk_size):
        chunk = diff[start : start + chunk_size]
        buckets = rng.integers(0, dim, size=chunk.size, dtype=np.int64)
        signs = rng.integers(0, 2, size=chunk.size, dtype=np.int8).astype(np.float64)
        signs = signs * 2.0 - 1.0
        np.add.at(projected, buckets, signs * chunk)


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _signature_id(
    *,
    vector: np.ndarray,
    run_id: str,
    round_idx: int,
    client_id: str,
    source_dim: int,
    source_norm: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(vector, dtype=np.float32).tobytes())
    digest.update(str(run_id).encode("utf-8"))
    digest.update(str(round_idx).encode("utf-8"))
    digest.update(str(client_id).encode("utf-8"))
    digest.update(str(source_dim).encode("utf-8"))
    digest.update(f"{float(source_norm):.12g}".encode("utf-8"))
    return digest.hexdigest()[:24]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._")[:120] or "signature"
