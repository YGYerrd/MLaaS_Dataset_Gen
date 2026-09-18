"""Verified raw-source family records shared by catalogue and execution adapters."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path


RAW_FAMILY_VERSIONS = ("raw_source_family_v3", "raw_source_family_v4")
FAMILY_VERSIONS = ("independent_family_v1", "independent_family_v2", *RAW_FAMILY_VERSIONS)


def latency_per_example(recipe):
    """Normalise recorded task-total, HF batch-mean and tabular per-row timings."""
    metrics = recipe["observed_metrics"]
    def value(name):
        item = metrics.get(name)
        return item.get("value") if isinstance(item, dict) else item
    if value("example_count") is not None:
        count = float(value("example_count"))
        latency = float(value("inference_latency_s")) / count if count > 0 else math.nan
        source = "task_total_divided_by_examples"
    elif value("eval_batch_count") is not None:
        count = float(value("eval_sequence_count"))
        mean_ms = value("inference_latency_ms_mean")
        if mean_ms is None:
            mean_ms = value("eval_latency_ms_mean")
        latency = float(mean_ms) * float(value("eval_batch_count")) / (1000 * count) if count > 0 else math.nan
        source = "hf_batch_mean_times_batches_divided_by_examples"
    else:
        latency = float(value("latency"))
        source = "recorded_per_example_evaluation_time"
    if not math.isfinite(latency) or latency < 0:
        raise ValueError("recipe has no usable per-example inference timing")
    return latency, source


def verified_member(db_path, service_id, *, cache=None):
    """Verify durable evidence without replaying training or modifying the catalogue.

    Older raw-source publications carry composition_compatible=0 because the
    previous consumer did not understand recipes. Eligibility is derived here
    from the actual evidence, independently of that historical marker.
    """
    from mlaas_data_generator.client_reconstruction import read_recipe, replay_supported
    from mlaas_data_generator.cli.manifest.preparation import read_preparation, verify_completed
    from mlaas_data_generator.tasks.contracts import provenance_contract_values

    with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT g.*, s.status AS service_status, s.task_family, s.task_type,
                   s.model_type, s.hf_task, s.modality
            FROM service_generation_provenance g JOIN services s USING(service_id)
            WHERE g.service_id=?
        """, (service_id,)).fetchone()
    if row is None or row["service_status"] != "completed":
        raise ValueError("raw-source composition requires a completed service")
    row = provenance_contract_values(dict(row))
    if row["family_contract_version"] not in RAW_FAMILY_VERSIONS:
        raise ValueError("unsupported raw-source family version")
    recipe = read_recipe(db_path, service_id)
    doc = read_preparation(row["family_document_ref"], row["family_document_hash"], cache=cache)
    if doc["version"] != row["family_contract_version"]:
        raise ValueError("service and family versions disagree")
    for field in ("family_id", "initial_state_hash", "compatibility_contract_hash"):
        if row.get(field) != doc[field]:
            raise ValueError(f"service differs from frozen family: {field}")
    for field, expected in doc["compatibility"].items():
        if row.get(field) != expected:
            raise ValueError(f"service differs from frozen family interface: {field}")
    if row.get("parent_state_hash") != doc["initial_state_hash"]:
        raise ValueError("service does not share the family initial state")
    from mlaas_data_generator.provenance import sha256_json
    if sha256_json(doc["raw_source"]) != recipe["fingerprints"]["source"]:
        raise ValueError("recipe raw source differs from the family")
    ids = recipe["selection"]["ordered_source_row_ids"]
    if not ids or set(ids) & set(doc["reservation"]["metric_source_ids"]):
        raise ValueError("client training rows overlap evaluation or are empty")
    policy = recipe["retention"]["policy"]
    if policy == "recipe_only":
        if not replay_supported(recipe["training"]["observations"], recipe["training"].get("memory_metadata")):
            raise ValueError("client training trajectory is not replayable")
    elif policy == "checkpoint":
        verify_completed(db_path, service_id, invalidate=False, cache=cache)
    else:
        raise ValueError("unsupported weight retention policy")
    from mlaas_data_generator.task_capabilities import resolve_capabilities
    decision = resolve_capabilities({**doc["config"], **row, "weight_retention": policy}, stage="composition")["measured_composition_validation"]
    if decision["decision"] == "ineligible":
        raise ValueError(f"{decision['reason_code']}: {decision['explanation']}")
    row.update(parent_document=doc, recipe=recipe, source_ids=ids,
               sample_count=len(ids), weight_retention=policy, composition_compatible=1)
    return row


def member_identity(row):
    return {key: row.get(key) for key in (
        "service_id", "family_id", "family_contract_version", "family_document_hash",
        "client_recipe_hash", "initial_state_hash", "final_model_hash", "weight_retention")}
