from __future__ import annotations

from math import ceil
from typing import Any


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _merge_max(existing: int | None, new_value: int | None) -> int | None:
    if new_value is None:
        return existing
    if existing is None:
        return new_value
    return max(existing, new_value)


def ensure_accounting_meta(meta: dict | None) -> dict:
    meta2 = dict(meta or {})
    accounting = dict(meta2.get("accounting") or {})
    accounting.setdefault("stages", [])
    for key in (
        "raw_record_count",
        "post_filter_record_count",
        "tokenized_record_count",
        "sequence_count",
        "supervised_token_count",
        "batch_count",
        "metric_instance_count",
    ):
        accounting[key] = _as_int(accounting.get(key))
    meta2["accounting"] = accounting
    return meta2


def update_accounting(meta: dict | None, **counts: Any) -> dict:
    meta2 = ensure_accounting_meta(meta)
    accounting = meta2["accounting"]
    for key, value in counts.items():
        if key not in accounting:
            continue
        accounting[key] = _as_int(value)
    return meta2


def append_accounting_stage(meta: dict | None, *, stage: str, split: str | None = None, **counts: Any) -> dict:
    meta2 = ensure_accounting_meta(meta)
    accounting = meta2["accounting"]

    stage_payload = {"stage": stage}
    if split is not None:
        stage_payload["split"] = split

    for key, value in counts.items():
        stage_payload[key] = _as_int(value)

    accounting.setdefault("stages", []).append(stage_payload)

    preferred_updates = {
        "raw_record_count": counts.get("raw_record_count") or counts.get("input_record_count"),
        "post_filter_record_count": counts.get("post_filter_record_count") or counts.get("surviving_record_count"),
        "tokenized_record_count": counts.get("tokenized_record_count") or counts.get("emitted_record_count"),
        "sequence_count": counts.get("sequence_count") or counts.get("emitted_record_count"),
        "supervised_token_count": counts.get("supervised_token_count"),
        "metric_instance_count": counts.get("metric_instance_count") or counts.get("emitted_record_count"),
    }

    if split != "test":
        for key, value in preferred_updates.items():
            accounting[key] = _merge_max(accounting.get(key), _as_int(value))

    return meta2


def finalize_accounting(meta: dict | None, *, batch_size: int | None = None) -> dict:
    meta2 = ensure_accounting_meta(meta)
    accounting = meta2["accounting"]
    if accounting.get("metric_instance_count") is None and accounting.get("sequence_count") is not None:
        accounting["metric_instance_count"] = accounting["sequence_count"]
    if accounting.get("tokenized_record_count") is None and accounting.get("post_filter_record_count") is not None:
        accounting["tokenized_record_count"] = accounting["post_filter_record_count"]
    if accounting.get("sequence_count") is None and accounting.get("tokenized_record_count") is not None:
        accounting["sequence_count"] = accounting["tokenized_record_count"]
    if accounting.get("batch_count") is None and batch_size and batch_size > 0:
        base = accounting.get("sequence_count") or accounting.get("tokenized_record_count") or accounting.get("post_filter_record_count")
        if base is not None:
            accounting["batch_count"] = int(ceil(base / int(batch_size)))
    return meta2
