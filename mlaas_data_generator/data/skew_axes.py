from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

DEFAULT_NUMERIC_BUCKETS = 5
DEFAULT_TOPK_BUCKETS = 16


@dataclass(frozen=True)
class ResolvedSkewAxis:
    requested_axis: str | None
    effective_axis: str | None
    axis_family: str | None
    bucket_ids: np.ndarray | None
    bucket_labels: dict[str, str]
    bucket_spec: dict[str, Any]
    source_fields: list[str]
    fallback_reason: str | None = None


def resolve_skew_axis(
    x,
    y,
    meta,
    *,
    split_name: str,
    task_family: str | None,
    hf_task: str | None,
    requested_axis: Any = None,
    axis_config: Any = None,
) -> ResolvedSkewAxis:
    meta = dict(meta or {})
    config = _coerce_mapping(axis_config)
    requested = _normalize_axis_name(requested_axis)
    default_axis = _default_axis(task_family, hf_task)
    axis = requested or default_axis
    if axis is None:
        return ResolvedSkewAxis(
            requested_axis=requested,
            effective_axis=None,
            axis_family=None,
            bucket_ids=None,
            bucket_labels={},
            bucket_spec={},
            source_fields=[],
            fallback_reason="no_task_aware_axis_available",
        )

    primary = _try_resolve_axis(axis, x, y, meta, split_name=split_name, task_family=task_family, hf_task=hf_task, config=config)
    if primary is not None:
        return primary

    if default_axis is not None and axis != default_axis:
        fallback = _try_resolve_axis(default_axis, x, y, meta, split_name=split_name, task_family=task_family, hf_task=hf_task, config=config)
        if fallback is not None:
            return ResolvedSkewAxis(
                requested_axis=requested,
                effective_axis=fallback.effective_axis,
                axis_family=fallback.axis_family,
                bucket_ids=fallback.bucket_ids,
                bucket_labels=fallback.bucket_labels,
                bucket_spec=fallback.bucket_spec,
                source_fields=fallback.source_fields,
                fallback_reason=f"requested axis '{axis}' unavailable; fell back to '{fallback.effective_axis}'",
            )

    return ResolvedSkewAxis(
        requested_axis=requested,
        effective_axis=None,
        axis_family=None,
        bucket_ids=None,
        bucket_labels={},
        bucket_spec={},
        source_fields=[],
        fallback_reason=f"could not resolve skew axis '{axis}'",
    )


def bucket_distribution(bucket_ids: np.ndarray | None, bucket_labels: Mapping[str, str] | None = None) -> dict[str, int]:
    if bucket_ids is None:
        return {}
    arr = np.asarray(bucket_ids, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return {}
    labels = dict(bucket_labels or {})
    counts = Counter(arr.tolist())
    return {
        labels.get(str(int(bucket_id)), str(int(bucket_id))): int(count)
        for bucket_id, count in sorted(counts.items())
    }


def axis_supports_strategy(axis: ResolvedSkewAxis, strategy: str) -> tuple[bool, str | None]:
    strategy = str(strategy or "iid").strip().lower()
    if strategy == "iid":
        return True, None
    if axis.bucket_ids is None or axis.effective_axis is None:
        return False, "resolved axis has no usable bucket ids"

    ordered = bool(axis.bucket_spec.get("ordered"))
    cardinality = int(axis.bucket_spec.get("cardinality") or len(np.unique(axis.bucket_ids)))

    if strategy == "dirichlet":
        return True, None
    if strategy == "shard":
        if not ordered:
            return False, "shard requires an ordered skew axis"
        return True, None
    if strategy == "label_per_client":
        if cardinality > int(axis.bucket_spec.get("max_supported_cardinality", 16)):
            return False, f"label_per_client requires low-cardinality buckets; got {cardinality}"
        return True, None
    if strategy == "quantity_skew":
        return True, None
    if strategy == "custom":
        return False, "custom distributions remain scalar-label only"
    return False, f"unknown strategy '{strategy}'"


def _try_resolve_axis(axis: str, x, y, meta: Mapping[str, Any], *, split_name: str, task_family: str | None, hf_task: str | None, config: Mapping[str, Any]) -> ResolvedSkewAxis | None:
    axis = _normalize_axis_name(axis)
    if axis is None:
        return None

    if axis == "class_label":
        labels = _scalar_label_array(y)
        if labels is None:
            return None
        return _categorical_axis(axis, labels, source_fields=["y"], topk=None)

    if axis == "entity_present_sentence":
        labels = _token_sentence_entity_signatures(y, ignore_index=_ignore_index(meta), background_value=_background_value(config))
        if labels is None:
            return None
        return _categorical_axis(axis, labels, source_fields=["y"], topk=int(config.get("max_buckets") or DEFAULT_TOPK_BUCKETS))

    if axis == "entity_token_label":
        labels = _dominant_token_label(y, ignore_index=_ignore_index(meta), background_value=_background_value(config))
        if labels is None:
            return None
        return _categorical_axis(axis, labels, source_fields=["y"], topk=int(config.get("max_buckets") or DEFAULT_TOPK_BUCKETS))

    if axis == "score_bin":
        values = _numeric_scalar_array(y)
        if values is None:
            return None
        return _numeric_bucket_axis(axis, values, source_fields=["y"], num_bins=int(config.get("num_bins") or DEFAULT_NUMERIC_BUCKETS))

    if axis == "supervised_token_bucket":
        values = _supervised_token_counts(y, ignore_index=_ignore_index(meta), pad_value=meta.get("pad_token_id"))
        if values is None:
            return None
        return _numeric_bucket_axis(axis, values, source_fields=["y"], num_bins=int(config.get("num_bins") or DEFAULT_NUMERIC_BUCKETS))

    if axis in {"prompt_length_bucket", "query_length_bucket"}:
        values = _sequence_lengths(x)
        if values is None:
            values = _sidecar_values(meta, split_name, config, preferred_keys=("prompt_text", "question_text", "query_text", "text"))
            if values is not None:
                values = np.asarray([len(str(v).split()) for v in values], dtype=np.float64)
        if values is None:
            return None
        return _numeric_bucket_axis(axis, values, source_fields=["x.attention_mask"], num_bins=int(config.get("num_bins") or DEFAULT_NUMERIC_BUCKETS))

    if axis == "target_length_bucket":
        values = _supervised_token_counts(y, ignore_index=_ignore_index(meta), pad_value=meta.get("pad_token_id"))
        if values is None:
            values = _sidecar_values(meta, split_name, config, preferred_keys=("target_text", "answer_text"))
            if values is not None:
                values = np.asarray([len(str(v).split()) for v in values], dtype=np.float64)
        if values is None:
            return None
        return _numeric_bucket_axis(axis, values, source_fields=["y"], num_bins=int(config.get("num_bins") or DEFAULT_NUMERIC_BUCKETS))

    if axis == "masked_token_id":
        labels = _dominant_supervised_token(y, ignore_index=_ignore_index(meta), pad_value=meta.get("pad_token_id"))
        if labels is None:
            return None
        return _categorical_axis(axis, labels, source_fields=["y"], topk=int(config.get("max_buckets") or DEFAULT_TOPK_BUCKETS))

    if axis == "answer_vocab":
        labels = _answer_vocab_values(y, meta, split_name=split_name)
        if labels is None:
            return None
        return _categorical_axis(axis, labels, source_fields=["y"], topk=int(config.get("max_buckets") or DEFAULT_TOPK_BUCKETS))

    if axis == "question_type":
        values = _sidecar_values(meta, split_name, config, preferred_keys=("question_text", "text", "query_text"))
        if values is None:
            return None
        labels = np.asarray([_question_type_bucket(v) for v in values], dtype=object)
        return _categorical_axis(axis, labels, source_fields=["meta.split_sidecars"], topk=int(config.get("max_buckets") or DEFAULT_TOPK_BUCKETS))

    if axis in {"domain_source", "text_domain", "image_domain", "source_domain", "query_domain", "document_domain", "pair_type", "relevance_label", "task_format"}:
        values = _sidecar_values(meta, split_name, config, preferred_keys=(axis, "domain", "source", "task_format", "pair_type", "relevance_label"))
        if values is None:
            return None
        labels = np.asarray([str(v).strip() if v is not None else "__missing__" for v in values], dtype=object)
        return _categorical_axis(axis, labels, source_fields=["meta.split_sidecars"], topk=int(config.get("max_buckets") or DEFAULT_TOPK_BUCKETS))

    return None


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = __import__("json").loads(text)
        except Exception:
            return {"field": text}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _default_axis(task_family: str | None, hf_task: str | None) -> str | None:
    task = _normalize_axis_name(hf_task)
    family = _normalize_axis_name(task_family)
    if task == "token_classification":
        return "entity_present_sentence"
    if task == "sentence_similarity" or family == "regression":
        return "score_bin"
    if task in {"causal_lm_generation", "seq2seq_generation"} or family == "generation":
        return "supervised_token_bucket"
    if task in {"fill_mask", "masked_lm"}:
        return "masked_token_id"
    if task == "visual_question_answering" or family == "vqa":
        return "answer_vocab"
    if task == "text_image_retrieval" or family == "retrieval":
        return "query_length_bucket"
    if family == "classification":
        return "class_label"
    return None


def _normalize_axis_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    return text or None


def _ignore_index(meta: Mapping[str, Any]) -> int:
    try:
        return int(meta.get("ignore_index", meta.get("label_pad_value", -100)))
    except Exception:
        return -100


def _background_value(config: Mapping[str, Any]) -> int:
    try:
        return int(config.get("background_label", 0))
    except Exception:
        return 0


def _scalar_label_array(y) -> np.ndarray | None:
    if y is None:
        return None
    try:
        arr = np.asarray(y, dtype=object)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    if arr.size == 0:
        return arr
    first = arr[0]
    if isinstance(first, (list, tuple, dict, np.ndarray)):
        return None
    return arr


def _numeric_scalar_array(y) -> np.ndarray | None:
    labels = _scalar_label_array(y)
    if labels is None:
        return None
    try:
        arr = np.asarray(labels, dtype=np.float64)
    except Exception:
        return None
    arr = arr.reshape(-1)
    return arr if arr.size > 0 else arr


def _token_sentence_entity_signatures(y, *, ignore_index: int, background_value: int) -> np.ndarray | None:
    arr = _token_label_matrix(y)
    if arr is None:
        return None
    labels = []
    for row in arr:
        valid = row[row != int(ignore_index)]
        valid = valid[valid != int(background_value)]
        unique = np.unique(valid)
        if unique.size == 0:
            labels.append("__none__")
        else:
            labels.append("|".join(str(int(v)) for v in unique.tolist()))
    return np.asarray(labels, dtype=object)


def _dominant_token_label(y, *, ignore_index: int, background_value: int) -> np.ndarray | None:
    arr = _token_label_matrix(y)
    if arr is None:
        return None
    labels = []
    for row in arr:
        valid = row[row != int(ignore_index)]
        valid = valid[valid != int(background_value)]
        if valid.size == 0:
            labels.append("__none__")
            continue
        values, counts = np.unique(valid.astype(np.int64, copy=False), return_counts=True)
        labels.append(str(int(values[int(np.argmax(counts))])))
    return np.asarray(labels, dtype=object)


def _token_label_matrix(y) -> np.ndarray | None:
    if y is None:
        return None
    try:
        arr = np.asarray(y)
    except Exception:
        return None
    if arr.ndim < 2:
        return None
    return arr


def _supervised_token_counts(y, *, ignore_index: int, pad_value: Any = None) -> np.ndarray | None:
    arr = _token_label_matrix(y)
    if arr is None:
        return None
    mask = arr != int(ignore_index)
    if pad_value is not None:
        try:
            mask &= arr != int(pad_value)
        except Exception:
            pass
    return np.sum(mask, axis=1).astype(np.float64, copy=False)


def _dominant_supervised_token(y, *, ignore_index: int, pad_value: Any = None) -> np.ndarray | None:
    arr = _token_label_matrix(y)
    if arr is None:
        return None
    labels = []
    for row in arr:
        valid = row[row != int(ignore_index)]
        if pad_value is not None:
            try:
                valid = valid[valid != int(pad_value)]
            except Exception:
                pass
        if valid.size == 0:
            labels.append("__none__")
            continue
        values, counts = np.unique(valid.astype(np.int64, copy=False), return_counts=True)
        labels.append(str(int(values[int(np.argmax(counts))])))
    return np.asarray(labels, dtype=object)


def _sequence_lengths(x) -> np.ndarray | None:
    if isinstance(x, Mapping):
        if x.get("caption_lengths") is not None:
            return np.asarray(x.get("caption_lengths"), dtype=np.float64).reshape(-1)
        attention_mask = x.get("attention_mask")
        if attention_mask is not None:
            arr = np.asarray(attention_mask)
            if arr.ndim >= 2:
                return np.sum(arr > 0, axis=1).astype(np.float64, copy=False)
            return arr.reshape(-1).astype(np.float64, copy=False)
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(x, dtype=object)
        except Exception:
            return None
        if arr.ndim == 1 and arr.size > 0 and isinstance(arr[0], str):
            return np.asarray([len(str(v).split()) for v in arr.tolist()], dtype=np.float64)
    return None


def _answer_vocab_values(y, meta: Mapping[str, Any], *, split_name: str) -> np.ndarray | None:
    arr = np.asarray(y, dtype=object)
    if arr.ndim == 1 and arr.size > 0 and not isinstance(arr[0], (list, tuple, dict, np.ndarray)):
        return np.asarray([str(v).strip() for v in arr.reshape(-1)], dtype=object)
    sidecar = _sidecar_values(meta, split_name, {}, preferred_keys=("answer_text", "label_text"))
    if sidecar is not None:
        return np.asarray([str(v).strip() for v in sidecar], dtype=object)
    if arr.ndim >= 2:
        return _dominant_supervised_token(arr, ignore_index=_ignore_index(meta), pad_value=meta.get("pad_token_id"))
    return None


def _sidecar_values(meta: Mapping[str, Any], split_name: str, config: Mapping[str, Any], *, preferred_keys: tuple[str, ...]) -> Any:
    sidecars = meta.get("split_sidecars")
    if not isinstance(sidecars, Mapping):
        return None
    split_payload = sidecars.get(split_name)
    if not isinstance(split_payload, Mapping):
        return None
    explicit_field = config.get("field") or config.get("column") or config.get("sidecar_field")
    if explicit_field and explicit_field in split_payload:
        return split_payload.get(explicit_field)
    for key in preferred_keys:
        if key in split_payload:
            return split_payload.get(key)
    return None


def _question_type_bucket(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "__empty__"
    first = text.split()[0]
    if first in {"what", "which", "where", "when", "why", "how", "who", "whom", "whose"}:
        return first
    if first in {"is", "are", "do", "does", "did", "can", "could", "will", "would", "has", "have"}:
        return "closed_form"
    if first in {"name", "count", "number", "color"}:
        return first
    return "__other__"


def _categorical_axis(axis: str, values: np.ndarray, *, source_fields: list[str], topk: int | None) -> ResolvedSkewAxis | None:
    arr = np.asarray(values, dtype=object).reshape(-1)
    if arr.size == 0:
        bucket_ids = np.asarray([], dtype=np.int64)
        return ResolvedSkewAxis(axis, axis, "categorical", bucket_ids, {}, {"ordered": False, "cardinality": 0}, source_fields)
    topk_value = None if topk is None else max(2, int(topk))
    labels = [str(v).strip() if v is not None and str(v).strip() else "__missing__" for v in arr.tolist()]
    counts = Counter(labels)
    if topk_value is None or len(counts) <= topk_value:
        kept = [label for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
        other_label = None
    else:
        kept = [label for label, _ in counts.most_common(topk_value - 1)]
        other_label = "__other__"
    label_to_bucket = {label: idx for idx, label in enumerate(kept)}
    bucket_labels = {str(idx): label for label, idx in label_to_bucket.items()}
    if other_label is not None:
        other_idx = len(label_to_bucket)
        bucket_labels[str(other_idx)] = other_label
    else:
        other_idx = None
    bucket_ids = np.asarray([label_to_bucket.get(label, other_idx) for label in labels], dtype=np.int64)
    if bucket_labels and len(bucket_labels) <= 1:
        return None
    return ResolvedSkewAxis(
        requested_axis=axis,
        effective_axis=axis,
        axis_family="categorical",
        bucket_ids=bucket_ids,
        bucket_labels=bucket_labels,
        bucket_spec={
            "kind": "categorical",
            "ordered": False,
            "cardinality": len(bucket_labels),
            "max_supported_cardinality": max(DEFAULT_TOPK_BUCKETS, len(bucket_labels)),
            "topk": topk_value,
        },
        source_fields=source_fields,
    )


def _numeric_bucket_axis(axis: str, values: np.ndarray, *, source_fields: list[str], num_bins: int) -> ResolvedSkewAxis | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    unique = np.unique(finite)
    if unique.size <= 1:
        return None
    bins = max(2, min(int(num_bins), int(unique.size)))
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(finite, quantiles)
    edges = np.unique(np.asarray(edges, dtype=np.float64))
    if edges.size <= 2:
        threshold = float(np.median(finite))
        edges = np.asarray([float(np.min(finite)), threshold, float(np.max(finite))], dtype=np.float64)
        edges = np.unique(edges)
        if edges.size <= 2:
            return None
    adjusted = edges.copy()
    adjusted[0] = adjusted[0] - 1e-9
    adjusted[-1] = adjusted[-1] + 1e-9
    bucket_ids = np.digitize(arr, adjusted[1:-1], right=True).astype(np.int64)
    labels = {}
    for idx in range(len(adjusted) - 1):
        lo = float(adjusted[idx] + (1e-9 if idx == 0 else 0.0))
        hi = float(adjusted[idx + 1] - (1e-9 if idx == len(adjusted) - 2 else 0.0))
        labels[str(idx)] = f"[{lo:.6g},{hi:.6g}]"
    return ResolvedSkewAxis(
        requested_axis=axis,
        effective_axis=axis,
        axis_family="ordered_numeric",
        bucket_ids=bucket_ids,
        bucket_labels=labels,
        bucket_spec={
            "kind": "quantile_bins",
            "ordered": True,
            "cardinality": len(labels),
            "num_bins": len(labels),
            "edges": [float(edge) for edge in adjusted.tolist()],
        },
        source_fields=source_fields,
    )
