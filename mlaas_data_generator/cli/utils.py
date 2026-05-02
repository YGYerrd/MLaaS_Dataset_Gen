# cli/utils.py
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path

def expand_inputs(patterns: List[str]) -> List[str]:
    """Supports globs and explicit paths for file merging."""
    paths: list[str] = []
    for pat in patterns:
        matches = sorted(str(p) for p in Path().glob(pat)) if any(ch in pat for ch in "*?[]") else [pat]
        paths.extend(matches)

    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _coerce_value(raw: str):
    raw = raw.strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_dataset_args(pairs: List[str] | None):
    args = {}
    if not pairs:
        return args
    for raw in pairs:
        if "=" not in raw:
            raise SystemExit(f"Invalid dataset argument '{raw}'. Expected KEY=VALUE format.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid dataset argument '{raw}'. Key cannot be empty.")
        args[key] = _coerce_value(value)
    return args

def resolve_hidden_layers(raw: str | None, fallback: list[int]) -> list[int]:
    if not raw: return fallback
    return [int(x) for x in raw.split(",") if x.strip()]