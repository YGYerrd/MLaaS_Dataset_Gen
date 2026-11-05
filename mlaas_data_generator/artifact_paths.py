"""Scoped relocation of artifact references without changing hashed documents."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path


_prefixes = ContextVar("artifact_path_prefixes", default=())


def load_artifact_path_map(source_db, path_map=None):
    """Load an explicit map, or the optional path_map.json beside a catalogue."""
    path = Path(path_map) if path_map is not None else Path(source_db).resolve().parent / "path_map.json"
    if path_map is None and not path.exists():
        return ()
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("version") != "composition_path_map_v1":
        raise ValueError(f"Unsupported artifact path map: {path}")
    entries = payload.get("prefixes")
    if not isinstance(entries, list):
        raise ValueError("Artifact path map requires a prefixes list")
    prefixes = {}
    for entry in entries:
        if not isinstance(entry, dict) or any(not isinstance(entry.get(k), str) or not entry[k].strip()
                                              for k in ("recorded", "resolved")):
            raise ValueError("Artifact path map prefixes require recorded and resolved paths")
        recorded = entry["recorded"].replace("\\", "/").rstrip("/")
        if not recorded or recorded in prefixes:
            raise ValueError("Artifact path map has an empty or duplicate recorded prefix")
        target = Path(entry["resolved"]).expanduser()
        if not target.is_absolute():
            target = path.parent / target
        prefixes[recorded] = str(target.resolve())
    return tuple(sorted(prefixes.items(), key=lambda item: len(item[0]), reverse=True))


def current_artifact_prefixes():
    """Return serializable mappings for spawned execution workers."""
    return _prefixes.get()


@contextmanager
def artifact_path_scope(prefixes):
    token = _prefixes.set(tuple(prefixes))
    try:
        yield
    finally:
        _prefixes.reset(token)


def resolve_artifact_path(reference):
    """Resolve only declared path prefixes; retain the original reference in evidence."""
    normalized = str(reference).replace("\\", "/").rstrip("/")
    for recorded, resolved in _prefixes.get():
        if normalized == recorded or normalized.startswith(recorded + "/"):
            root = Path(resolved)
            suffix = normalized[len(recorded):].lstrip("/")
            candidate = (root / suffix).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError(f"Artifact path escapes its mapped root: {reference}")
            return candidate
    return Path(reference)
