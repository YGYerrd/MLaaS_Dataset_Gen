"""Read and normalise completed service-catalogue records for post-processing."""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SERVICE_COLUMNS = [
    "task_semantics_hash", "evaluation_protocol_hash", "task_tag",
    # Service identity and compatibility contract.
    "service_id", "service_status", "generation_mode", "task_family", "task_type", "modality", "input_schema",
    "output_schema", "label_format", "dataset", "dataset_name", "dataset_config",
    "train_split", "benchmark_split", "training_regime", "model_type", "model_id", "hf_task",
    # Generator provenance consumed only by downstream compatibility policy.
    "architecture_id", "parameter_schema_hash", "label_map_hash",
    "preprocessing_contract_hash", "logical_dataset_contract_hash", "lineage_id",
    "fl_run_id", "client_id", "round_id", "compatibility_contract_hash",
    "family_id", "parent_checkpoint_id", "cohort_id", "cohort_succeeded",
    "family_contract_version", "initial_state_hash", "composition_compatible",
    "family_document_ref", "family_document_hash", "client_recipe_ref", "client_recipe_hash",
    "weight_retention", "realised_sample_count", "composition_eligibility_reason",
    "composition_latency_s_per_example", "composition_latency_source",
    "parent_global_round", "parent_state_hash", "participated", "fedavg_succeeded",
    "partition_generator_version", "heterogeneity_intensity", "target_deviation_count",
    # Independently measured service quality and QoS.
    "primary_metric_name", "auxiliary_metric_name", "metric_score", "latency",
    "data_volume", "resource_cost_score", "data_distribution", "compute_time_s",
    "batch_size", "model_parameters", "training_loss", "reliability_score",
    "trust_score", "explainability_score",
    "trust_independent_inputs", "explainability_independent_inputs",
    "evaluation_protocol_version", "trust_status", "explainability_status",
    "trust_evidence_sufficient", "explainability_evidence_sufficient",
    # Optional artifacts consumed by downstream collaborative evaluation.
    "prediction_artifact_path", "prediction_artifact_type", "update_signature_id",
    "update_signature_path", "signature_dim", "signature_norm",
]


def load_services_from_db(
    db_path: str | Path,
    *,
    include_legacy_fedavg_partitions: bool = False,
) -> pd.DataFrame:
    """Load the completed-service fields used by composition post-processing.

    The normalized generator schema is the primary source.  The published
    ``flattened_services`` export remains supported for older catalogue
    snapshots and the existing dataset-compatibility workflow.
    """
    with sqlite3.connect(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "services" in tables:
            fedavg_run_columns = (
                {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(fedavg_runs)")
                }
                if "fedavg_runs" in tables
                else set()
            )
            rows = pd.read_sql_query(
                _normalized_service_query(
                    has_provenance="service_generation_provenance" in tables,
                    has_fedavg_client_rounds="fedavg_client_rounds" in tables,
                    has_fedavg_runs="fedavg_runs" in tables,
                    has_fedavg_client_partitions="fedavg_client_partitions" in tables,
                    has_partition_version="partition_generator_version"
                    in fedavg_run_columns,
                    has_heterogeneity_intensity="heterogeneity_intensity"
                    in fedavg_run_columns,
                    include_legacy_fedavg_partitions=include_legacy_fedavg_partitions,
                ),
                conn,
            )
            provenance_fields = ({r[1] for r in conn.execute("PRAGMA table_info(service_generation_provenance)")}
                                 if "service_generation_provenance" in tables else set())
            if "cohort_id" in provenance_fields and "cohort_clients" in tables:
                shared = pd.read_sql_query("""
                    SELECT g.service_id, g.family_id, g.parent_checkpoint_id, g.cohort_id,
                           CASE WHEN c.status='completed' THEN 1 ELSE 0 END AS cohort_succeeded
                    FROM service_generation_provenance g
                    LEFT JOIN cohort_clients c ON c.service_id=g.service_id AND c.cohort_id=g.cohort_id
                    WHERE g.generation_mode='shared_parent'
                """, conn)
                rows = rows.merge(shared, on="service_id", how="left", validate="one_to_one")
                rows.loc[rows["cohort_id"].notna(), "generation_mode"] = "shared_parent"
            if "family_contract_version" in provenance_fields:
                independent = pd.read_sql_query("""
                    SELECT service_id, family_id, family_contract_version, initial_state_hash,
                           composition_compatible, client_id, parent_state_hash
                    FROM service_generation_provenance WHERE family_contract_version IN ('independent_family_v1', 'independent_family_v2')
                """, conn).set_index("service_id")
                mask = rows.service_id.isin(independent.index)
                for field in independent.columns:
                    if field not in rows:
                        rows[field] = None
                    rows.loc[mask, field] = rows.loc[mask, "service_id"].map(independent[field])
                from mlaas_data_generator.cli.manifest.preparation import verify_completed
                family_reference_cache = {}
                for index in rows.index[mask & rows.composition_compatible.eq(1)]:
                    try:
                        verify_completed(db_path, rows.at[index, "service_id"], invalidate=False, cache=family_reference_cache)
                    except Exception:
                        rows.at[index, "composition_compatible"] = 0
            if "provenance_json" in provenance_fields:
                for sid, document in conn.execute("SELECT service_id, provenance_json FROM service_generation_provenance"):
                    frozen = json.loads(document or "{}").get("task_contract")
                    if frozen:
                        from mlaas_data_generator.tasks import TaskContract
                        contract = TaskContract.from_dict(frozen)
                        mask = rows.service_id.eq(sid)
                        for field, value in (("task_semantics_hash", contract.semantics_hash),
                                             ("evaluation_protocol_hash", contract.protocol_hash),
                                             ("task_tag", contract.semantics.get("task_tag"))):
                            rows.loc[mask, field] = value
            if "client_recipe_ref" in provenance_fields:
                from .raw_family import RAW_FAMILY_VERSIONS, verified_member, latency_per_example
                raw = pd.read_sql_query("SELECT * FROM service_generation_provenance", conn)
                raw = raw[raw.family_contract_version.isin(RAW_FAMILY_VERSIONS)].set_index("service_id")
                cache = {}
                for index in rows.index[rows.service_id.isin(raw.index)]:
                    sid = rows.at[index, "service_id"]
                    for field in ("family_id", "family_contract_version", "initial_state_hash",
                                  "family_document_ref", "family_document_hash", "client_recipe_ref",
                                  "client_recipe_hash", "client_id", "parent_state_hash"):
                        rows.loc[index, field] = raw.at[sid, field]
                    rows.loc[index, "composition_compatible"] = 0
                    try:
                        member = verified_member(db_path, sid, cache=cache)
                        if not finite(rows.at[index, "metric_score"]) or not 0 <= float(rows.at[index, "metric_score"]) <= 1:
                            raise ValueError("raw-source service has no valid measured quality score")
                        if not finite(rows.at[index, "latency"]) or float(rows.at[index, "latency"]) < 0:
                            raise ValueError("raw-source service has no measured inference latency")
                        if not _present_text(rows.at[index, "primary_metric_name"]):
                            raise ValueError("raw-source service has no named primary quality metric")
                        latency, latency_source = latency_per_example(member["recipe"])
                        rows.loc[index, "composition_latency_s_per_example"] = latency
                        rows.loc[index, "composition_latency_source"] = latency_source
                        rows.loc[index, "composition_compatible"] = 1
                        rows.loc[index, "weight_retention"] = member["weight_retention"]
                        rows.loc[index, "realised_sample_count"] = member["sample_count"]
                        rows.loc[index, "composition_eligibility_reason"] = "verified_raw_source_family"
                    except Exception as error:
                        rows.loc[index, "composition_eligibility_reason"] = f"{type(error).__name__}: {error}"
        elif "flattened_services" in tables:
            rows = pd.read_sql_query(_FLATTENED_SERVICE_QUERY, conn)
        else:
            raise ValueError(
                f"Unsupported catalogue database '{db_path}': expected either "
                "the normalized 'services' table or 'flattened_services'"
            )
    return normalise_service_rows(rows)


def _normalized_service_query(
    *,
    has_provenance: bool,
    has_fedavg_client_rounds: bool,
    has_fedavg_runs: bool = False,
    has_fedavg_client_partitions: bool = False,
    has_partition_version: bool = False,
    has_heterogeneity_intensity: bool = False,
    include_legacy_fedavg_partitions: bool = False,
) -> str:
    provenance_columns = """
        g.architecture_id, g.parameter_schema_hash, g.label_map_hash,
        g.preprocessing_contract_hash, g.logical_dataset_contract_hash, g.lineage_id,
        g.fl_run_id, g.client_id, g.round_id, g.compatibility_contract_hash,
        g.parent_global_round, g.parent_state_hash, g.participated,
    """ if has_provenance else """
        NULL AS architecture_id, NULL AS parameter_schema_hash, NULL AS label_map_hash,
        NULL AS preprocessing_contract_hash, NULL AS logical_dataset_contract_hash, NULL AS lineage_id,
        NULL AS fl_run_id, NULL AS client_id, NULL AS round_id, NULL AS compatibility_contract_hash,
        NULL AS parent_global_round, NULL AS parent_state_hash, NULL AS participated,
    """
    provenance_join = "LEFT JOIN service_generation_provenance g ON g.service_id = s.service_id" if has_provenance else ""
    generation_mode = "CASE WHEN g.fl_run_id IS NOT NULL THEN 'fedavg' ELSE 'independent' END" if has_provenance else "'independent'"
    succeeded_column = "f.succeeded AS fedavg_succeeded" if has_fedavg_client_rounds else "NULL AS fedavg_succeeded"
    succeeded_join = "LEFT JOIN fedavg_client_rounds f ON f.service_id = s.service_id" if has_fedavg_client_rounds else ""
    fedavg_join = (
        "LEFT JOIN fedavg_runs r ON r.fl_run_id = g.fl_run_id"
        if has_provenance and has_fedavg_runs
        else ""
    )
    version_column = (
        "r.partition_generator_version AS partition_generator_version"
        if has_provenance and has_fedavg_runs and has_partition_version
        else "NULL AS partition_generator_version"
    )
    intensity_column = (
        "r.heterogeneity_intensity AS heterogeneity_intensity"
        if has_provenance and has_fedavg_runs and has_heterogeneity_intensity
        else "NULL AS heterogeneity_intensity"
    )
    target_deviation_column = (
        "fp.target_deviation_count AS target_deviation_count"
        if has_provenance and has_fedavg_client_partitions
        else "NULL AS target_deviation_count"
    )
    partition_join = (
        "LEFT JOIN fedavg_client_partitions fp "
        "ON fp.fl_run_id = g.fl_run_id AND fp.client_id = g.client_id"
        if has_provenance and has_fedavg_client_partitions
        else ""
    )
    if include_legacy_fedavg_partitions or not has_provenance:
        fedavg_filter = ""
    elif has_fedavg_runs and has_partition_version:
        fedavg_filter = (
            "AND (g.fl_run_id IS NULL OR "
            "r.partition_generator_version IN ('joint_target_allocator_v2', 'source_pool_sampler_v3'))"
        )
    else:
        fedavg_filter = "AND g.fl_run_id IS NULL"
    return f"""
    WITH m AS (
        SELECT service_id, metric_name,
            COALESCE(value_num, CAST(value_int AS REAL), CAST(value_bool AS REAL)) AS num_value,
            COALESCE(value_text, CAST(value_num AS TEXT), CAST(value_int AS TEXT), CAST(value_bool AS TEXT), value_json) AS text_value
        FROM service_metrics
    ), p AS (
        SELECT service_id,
            MAX(CASE WHEN metric_name = 'primary_metric_name' THEN text_value END) AS primary_metric_name,
            MAX(CASE WHEN metric_name = 'auxiliary_metric_name' THEN text_value END) AS auxiliary_metric_name,
            MAX(CASE WHEN metric_name = 'metric_score' THEN num_value END) AS metric_score,
            MAX(CASE WHEN metric_name IN ('latency', 'inference_latency_s', 'inference_latency_s_mean') THEN num_value END) AS latency,
            MAX(CASE WHEN metric_name IN ('compute_time_s', 'runtime_s', 'service_runtime_s') THEN num_value END) AS compute_time_s,
            MAX(CASE WHEN metric_name IN ('dataset_size', 'train_set_size') THEN num_value END) AS data_volume,
            MAX(CASE WHEN metric_name IN ('model_params_count', 'params_count', 'model_size') THEN num_value END) AS model_parameters,
            COALESCE(
                MAX(CASE WHEN metric_name = 'train_loss' THEN num_value END),
                MAX(CASE WHEN metric_name = 'loss' THEN num_value END),
                MAX(CASE WHEN metric_name = 'cross_entropy_loss' THEN num_value END)
            ) AS training_loss,
            MAX(CASE WHEN metric_name = 'resource_cost_score' THEN num_value END) AS resource_cost_score,
            MAX(CASE WHEN metric_name IN ('data_distribution', 'split_strategy') THEN text_value END) AS data_distribution,
            MAX(CASE WHEN metric_name = 'update_signature_id' THEN text_value END) AS update_signature_id,
            MAX(CASE WHEN metric_name = 'update_signature_path' THEN text_value END) AS update_signature_path,
            MAX(CASE WHEN metric_name = 'signature_dim' THEN num_value END) AS signature_dim,
            MAX(CASE WHEN metric_name = 'signature_norm' THEN num_value END) AS signature_norm,
            MAX(CASE WHEN metric_name = 'batch_size' THEN num_value END) AS batch_size,
            MAX(CASE WHEN metric_name = 'reliability_score' THEN num_value END) AS reliability_score,
            MAX(CASE WHEN metric_name = 'trust_score' THEN num_value END) AS trust_score,
            MAX(CASE WHEN metric_name = 'explainability_score' THEN num_value END) AS explainability_score,
            MAX(CASE WHEN metric_name = 'trust_independent_inputs' THEN num_value END) AS trust_independent_inputs,
            MAX(CASE WHEN metric_name = 'explainability_independent_inputs' THEN num_value END) AS explainability_independent_inputs,
            MAX(CASE WHEN metric_name = 'evaluation_protocol_version' THEN text_value END) AS evaluation_protocol_version,
            MAX(CASE WHEN metric_name = 'trust_status' THEN text_value END) AS trust_status,
            MAX(CASE WHEN metric_name = 'explainability_status' THEN text_value END) AS explainability_status,
            MAX(CASE WHEN metric_name = 'trust_evidence_sufficient' THEN text_value END) AS trust_evidence_sufficient,
            MAX(CASE WHEN metric_name = 'explainability_evidence_sufficient' THEN text_value END) AS explainability_evidence_sufficient
        FROM m GROUP BY service_id
    ), a AS (
        SELECT service_id,
            MAX(CASE WHEN artifact_type = 'update_signature' THEN artifact_uri END) AS artifact_update_signature_path
            , MAX(CASE WHEN artifact_type IN ('prediction', 'predictions', 'benchmark_predictions') THEN artifact_uri END) AS prediction_artifact_path
            , MAX(CASE WHEN artifact_type IN ('prediction', 'predictions', 'benchmark_predictions') THEN artifact_type END) AS prediction_artifact_type
        FROM service_artifacts GROUP BY service_id
    )
    SELECT s.service_id, s.status AS service_status,
        {generation_mode} AS generation_mode,
        COALESCE(s.task_family, s.task_type, 'unknown') AS task_family,
        s.task_type, s.modality, s.input_schema, s.output_schema,
        json_extract(s.functional_attributes_json, '$.label_format') AS label_format,
        s.dataset, s.dataset_name, s.dataset_config, s.train_split, s.benchmark_split,
        s.training_regime, s.model_type, s.model_id, s.hf_task,
        {provenance_columns}
        {succeeded_column},
        {version_column}, {intensity_column}, {target_deviation_column},
        p.primary_metric_name, p.auxiliary_metric_name, p.metric_score, p.latency,
        p.data_volume, p.model_parameters, p.training_loss, p.resource_cost_score, p.data_distribution,
        p.update_signature_id, COALESCE(p.update_signature_path, a.artifact_update_signature_path) AS update_signature_path,
        p.signature_dim, p.signature_norm, p.compute_time_s, p.batch_size,
        p.reliability_score, p.trust_score, p.explainability_score,
        p.trust_independent_inputs, p.explainability_independent_inputs,
        p.evaluation_protocol_version, p.trust_status, p.explainability_status,
        p.trust_evidence_sufficient, p.explainability_evidence_sufficient,
        a.prediction_artifact_path, a.prediction_artifact_type
    FROM services s LEFT JOIN p ON p.service_id = s.service_id
    LEFT JOIN a ON a.service_id = s.service_id
    {provenance_join}
    {succeeded_join}
    {fedavg_join}
    {partition_join}
    WHERE s.status = 'completed' {fedavg_filter}
"""


_FLATTENED_SERVICE_QUERY = """
    SELECT
        "run_id" AS service_id,
        'completed' AS service_status,
        'independent' AS generation_mode,
        "task_type" AS task_family,
        "task_type" AS task_type,
        CASE
            WHEN lower("task_type") IN ('detection', 'segmentation') THEN 'image'
            WHEN lower("task_type") IN ('text_generation', 'text2text_generation') THEN 'text'
            ELSE 'unknown'
        END AS modality,
        'unknown' AS input_schema,
        'unknown' AS output_schema,
        'unknown' AS label_format,
        "dataset" AS dataset,
        "dataset" AS dataset_name,
        'unknown' AS dataset_config,
        'unknown' AS train_split,
        'unknown' AS benchmark_split,
        "type (finetune, inferance)" AS training_regime,
        'huggingface' AS model_type,
        "HF model id" AS model_id,
        "task_type" AS hf_task,
        NULL AS architecture_id,
        NULL AS parameter_schema_hash,
        NULL AS label_map_hash,
        NULL AS preprocessing_contract_hash,
        NULL AS logical_dataset_contract_hash,
        NULL AS lineage_id,
        NULL AS fl_run_id,
        NULL AS client_id,
        NULL AS round_id,
        NULL AS compatibility_contract_hash,
        NULL AS parent_global_round,
        NULL AS parent_state_hash,
        NULL AS participated,
        NULL AS fedavg_succeeded,
        NULL AS partition_generator_version,
        NULL AS heterogeneity_intensity,
        NULL AS target_deviation_count,
        "Primary metric name" AS primary_metric_name,
        "Auxiliary metric name" AS auxiliary_metric_name,
        "Primary metric" AS metric_score,
        "Latency" AS latency,
        0.0 AS data_volume,
        0.0 AS model_parameters,
        0.0 AS training_loss,
        "Resource cost score" AS resource_cost_score,
        "Split strategy" AS data_distribution,
        "Mean compute time" AS compute_time_s,
        "Batch size" AS batch_size,
        0.5 AS reliability_score,
        NULL AS trust_score,
        "Explainability score" AS explainability_score,
        '' AS prediction_artifact_path,
        '' AS prediction_artifact_type,
        NULL AS update_signature_id,
        "Update signature path" AS update_signature_path,
        NULL AS signature_dim,
        NULL AS signature_norm
    FROM flattened_services
"""


def normalise_service_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Produce a stable, composition-ready view of a catalogue snapshot."""
    services = rows.copy()
    for column in SERVICE_COLUMNS:
        if column not in services.columns:
            services[column] = None
    services["service_id"] = services["service_id"].astype(str)
    for column in (
        "service_status", "generation_mode", "task_family", "task_type", "model_type", "model_id", "hf_task", "modality",
        "input_schema", "output_schema", "label_format", "dataset", "dataset_name",
        "dataset_config", "train_split", "benchmark_split", "training_regime", "data_distribution",
        "primary_metric_name", "auxiliary_metric_name", "prediction_artifact_type",
    ):
        services[column] = services[column].fillna("unknown").astype(str).str.lower()
    services.loc[
        services["service_status"].isin({"", "unknown", "none", "null", "nan"}),
        "service_status",
    ] = "completed"
    # Artifact URIs can be case-sensitive paths or object-store keys.
    services["prediction_artifact_path"] = services["prediction_artifact_path"].fillna("").astype(str)
    services["metric_score"] = services["metric_score"].map(lambda value: bounded_number(value, 0.0))
    for column in ("latency", "compute_time_s", "data_volume", "model_parameters", "training_loss"):
        services[column] = services[column].map(lambda value: max(0.0, number(value, 0.0)))
    services["resource_cost_score"] = services["resource_cost_score"].map(lambda value: bounded_number(value, 0.5))
    services["batch_size"] = services["batch_size"].map(lambda value: max(1, int(number(value, 1.0))))
    services["reliability_score"] = services["reliability_score"].map(lambda value: bounded_number(value, 0.5))
    services["evaluation_protocol_version"] = services["evaluation_protocol_version"].map(
        lambda value: str(value).strip() if _present_text(value) else "legacy_unversioned"
    )
    for dimension in ("trust", "explainability"):
        services[f"{dimension}_independent_inputs"] = services[f"{dimension}_independent_inputs"].map(
            lambda value: int(float(value)) if finite(value) and float(value) >= 0 and float(value).is_integer() else np.nan)
        # Missing, non-finite and invalid scores remain missing; they are never
        # replaced by a plausible-looking neutral quality measurement.
        services[f"{dimension}_score"] = services[f"{dimension}_score"].map(
            lambda value: number(value) if finite(value) and 0.0 <= float(value) <= 1.0 else np.nan
        )
        services[f"{dimension}_status"] = services[f"{dimension}_status"].map(
            lambda value: str(value).strip().lower() if _present_text(value) else "insufficient_evidence"
        )
        services[f"{dimension}_evidence_sufficient"] = services[f"{dimension}_evidence_sufficient"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "1.0"}
        )
    services["signature_dim"] = services["signature_dim"].map(lambda value: int(number(value, 0.0)) if finite(value) else None)
    services["signature_norm"] = services["signature_norm"].map(lambda value: number(value, np.nan))
    for column in ("round_id", "parent_global_round", "participated", "fedavg_succeeded", "cohort_succeeded", "composition_compatible"):
        services[column] = services[column].map(
            lambda value: int(number(value, 0.0)) if finite(value) else None
        )
    for column in (
        "architecture_id", "parameter_schema_hash", "label_map_hash",
        "preprocessing_contract_hash", "logical_dataset_contract_hash", "lineage_id",
        "fl_run_id", "client_id", "compatibility_contract_hash", "parent_state_hash",
        "family_id", "parent_checkpoint_id", "cohort_id",
        "partition_generator_version",
    ):
        services[column] = services[column].map(
            lambda value: None if not _present_text(value) else str(value).strip()
        )
    services["generation_mode"] = np.where(services["cohort_id"].notna(), "shared_parent",
        np.where(services["fl_run_id"].notna(), "fedavg", "independent"))
    services["heterogeneity_intensity"] = services["heterogeneity_intensity"].map(
        lambda value: number(value, np.nan)
    )
    services["target_deviation_count"] = services["target_deviation_count"].map(
        lambda value: number(value, np.nan)
    )
    return services[SERVICE_COLUMNS]


def number(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or (isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "not available", "n/a"}):
            return float(default)
        parsed = float(value)
        return parsed if np.isfinite(parsed) else float(default)
    except Exception:
        return float(default)


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def bounded_number(value: Any, default: float) -> float:
    return float(np.clip(number(value, default), 0.0, 1.0))


def _present_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return str(value).strip().lower() not in {"", "unknown", "none", "null", "nan", "n/a", "not available"}
