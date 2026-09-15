"""Bounded diagnostic JSON. Never repr runtime objects or copy tensor values."""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path

MAX_CONFIG_BYTES = 128 * 1024
MAX_TEXT_CHARS = 8192


def bounded_text(value: str | None, limit: int = MAX_TEXT_CHARS) -> str | None:
    if value is None or len(value) <= limit:
        return value
    return value[:limit] + f"... [truncated {len(value) - limit} characters]"


def diagnostic_value(value, *, max_nodes=4096):
    remaining = max_nodes
    active = set()

    def clean(item, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 12:
            return {"omitted": "diagnostic size limit"}
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, str):
            if len(item) <= MAX_TEXT_CHARS and item.startswith(("{", "[")):
                try:
                    return json.dumps(clean(json.loads(item), depth + 1), allow_nan=False)
                except (ValueError, RecursionError):
                    pass
            return bounded_text(item)
        if isinstance(item, Path):
            return bounded_text(str(item))
        # Check arrays before sequences: a state dictionary must never expand
        # tensor contents, even when it is nested under an innocuous public key.
        module = type(item).__module__.split(".")[0]
        if module in {"torch", "numpy", "tensorflow"} and hasattr(item, "shape"):
            if module == "numpy" and getattr(item, "ndim", None) == 0:
                return clean(item.item(), depth + 1)
            return {"omitted": "array", "type": type(item).__name__,
                    "shape": [int(n) for n in item.shape], "dtype": str(item.dtype)}
        if id(item) in active:
            return {"omitted": "cycle"}
        if isinstance(item, (Mapping, list, tuple)):
            active.add(id(item))
            try:
                if isinstance(item, Mapping):
                    result = {}
                    for index, (key, child) in enumerate(item.items()):
                        if index >= 512 or remaining <= 0:
                            result["diagnostic_truncated"] = True
                            break
                        if isinstance(key, str) and not key.startswith("_"):
                            result[key[:256]] = clean(child, depth + 1)
                    return result
                result = [clean(child, depth + 1) for child in item[:min(512, max(0, remaining))]]
                if len(item) > len(result):
                    result.append({"omitted_items": len(item) - len(result)})
                return result
            finally:
                active.remove(id(item))
        return {"omitted": "runtime object", "type": f"{type(item).__module__}.{type(item).__name__}"}

    return clean(value)


def diagnostic_json(value, *, max_bytes=MAX_CONFIG_BYTES) -> str:
    """Return valid JSON within a byte ceiling, including pre-serialized input."""
    if isinstance(value, str):
        if len(value.encode("utf-8")) > max_bytes:
            return json.dumps({"omitted": "oversized serialized diagnostic", "characters": len(value)})
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            pass
    cleaned = diagnostic_value(value)
    result = json.dumps(cleaned, ensure_ascii=False, allow_nan=False)
    if len(result.encode("utf-8")) <= max_bytes:
        return result
    if isinstance(cleaned, dict):
        # Retain inexpensive identity/config fields rather than an invalid JSON
        # prefix. Authoritative recipes remain in the manifest and artifacts.
        kept = {"diagnostic_truncated": True}
        for key, item in cleaned.items():
            candidate = json.dumps({**kept, key: item}, ensure_ascii=False, allow_nan=False)
            if len(candidate.encode("utf-8")) <= max_bytes:
                kept[key] = item
        return json.dumps(kept, ensure_ascii=False, allow_nan=False)
    return json.dumps({"omitted": "diagnostic byte limit"})
