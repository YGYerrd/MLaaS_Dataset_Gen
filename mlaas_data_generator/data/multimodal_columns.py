from __future__ import annotations

import re
from typing import Any, Iterable


_NUMBERED_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<index>\d+)$")


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _numbered_base_candidates(values: Iterable[Any]) -> list[str]:
    bases: list[str] = []
    for value in _ordered_unique(values):
        bases.append(value)
        match = _NUMBERED_SUFFIX_RE.match(value)
        if match:
            bases.append(match.group("base"))
    return _ordered_unique(bases)


def resolve_existing_column(
    preferred: Any,
    available_columns: Iterable[Any],
    *,
    aliases: Iterable[Any] = (),
    numbered_alias_bases: Iterable[Any] = (),
) -> Any:
    """Resolve a requested dataset column against exact and numbered aliases.

    This handles HF image-text datasets such as Flickr8k that expose
    ``caption_0``...``caption_4`` instead of a single ``caption`` column.
    """

    available = _ordered_unique(available_columns)
    available_set = set(available)

    for candidate in _ordered_unique([preferred, *aliases]):
        if candidate in available_set:
            return candidate

    bases = _numbered_base_candidates([preferred, *aliases, *numbered_alias_bases])
    for base in bases:
        prefix = f"{base}_"
        matches = [
            column
            for column in available
            if column.startswith(prefix) and column[len(prefix) :].isdigit()
        ]
        if matches:
            return sorted(matches, key=lambda column: (int(column[len(prefix) :]), column))[0]

    return preferred
