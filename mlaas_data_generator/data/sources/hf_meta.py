import json
from datetime import datetime
from typing import Any, Dict


def _to_iso8601(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _extract_model_info_payload(model_info: Any) -> Dict[str, Any]:
    if model_info is None:
        return {}

    # huggingface_hub's `ModelInfo` typically has a `__dict__` and may expose
    # a `to_dict` helper depending on package version.
    if hasattr(model_info, "to_dict"):
        try:
            payload = model_info.to_dict()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    data = getattr(model_info, "__dict__", None)
    if isinstance(data, dict):
        return data

    return {}


def fetch_hf_model_meta(hf_model_id: str) -> dict:
    """Fetch and normalize Hugging Face model metadata.

    This function is intentionally resilient: it returns partial metadata when
    provider calls fail so bootstrapping can continue.
    """
    base_meta: Dict[str, Any] = {
        "hf_model_id": hf_model_id,
        "hf_pipeline_tag": None,
        "hf_downloads": None,
        "hf_likes": None,
        "hf_last_modified": None,
        "hf_author": None,
        "hf_url": f"https://huggingface.co/{hf_model_id}",
        "hf_service_meta_json": "{}",
    }

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        base_meta["hf_service_meta_json"] = json.dumps(
            {"error": f"huggingface_hub import failed: {exc}"}, default=str
        )
        return base_meta

    try:
        model_info = HfApi().model_info(hf_model_id)
        payload = _extract_model_info_payload(model_info)

        # Prefer API values when present.
        base_meta["hf_pipeline_tag"] = (
            payload.get("pipeline_tag")
            or getattr(model_info, "pipeline_tag", None)
            or base_meta["hf_pipeline_tag"]
        )
        base_meta["hf_downloads"] = (
            payload.get("downloads")
            if payload.get("downloads") is not None
            else getattr(model_info, "downloads", None)
        )
        base_meta["hf_likes"] = (
            payload.get("likes")
            if payload.get("likes") is not None
            else getattr(model_info, "likes", None)
        )

        last_modified = (
            payload.get("last_modified")
            if payload.get("last_modified") is not None
            else getattr(model_info, "last_modified", None)
        )
        base_meta["hf_last_modified"] = _to_iso8601(last_modified)

        base_meta["hf_author"] = (
            payload.get("author")
            or payload.get("created_by")
            or getattr(model_info, "author", None)
            or getattr(model_info, "created_by", None)
        )

        # Keep a directly provided model URL when available.
        direct_url = (
            payload.get("url")
            or payload.get("model_url")
            or getattr(model_info, "url", None)
            or getattr(model_info, "model_url", None)
        )
        if direct_url:
            base_meta["hf_url"] = direct_url

        trace_payload = {
            "hf_model_id": hf_model_id,
            "provider": "huggingface_hub",
            "model_info": payload,
        }
        base_meta["hf_service_meta_json"] = json.dumps(trace_payload, default=str)
        return base_meta

    except Exception as exc:
        # Graceful fallback with partial metadata for bootstrap continuity.
        base_meta["hf_service_meta_json"] = json.dumps(
            {
                "hf_model_id": hf_model_id,
                "provider": "huggingface_hub",
                "error": str(exc),
            },
            default=str,
        )
        return base_meta