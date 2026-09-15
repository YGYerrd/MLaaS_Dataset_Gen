"""Materialize one selected composition through canonical FedAvg aggregation."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mlaas_data_generator.checkpoint_store import (
    reconstruct_service_base_state,
    reconstruct_service_state,
)
from mlaas_data_generator.fedavg_aggregation import (
    FedAvgContributor,
    aggregate_fedavg_states,
)

from .regeneration import execute_tensor_state_prediction
from .validation import CohortPlan


class FedAvgValidationError(ValueError):
    """Base error carrying the validation stage that failed."""

    stage = "provenance"
    code = "provenance_integrity_failed"


class FedAvgProvenanceError(FedAvgValidationError):
    stage = "provenance"

    def __init__(self, message, *, field=None, code=None):
        self.code = code or {
            "parameter_schema_hash": "parameter_schema_mismatch", "checkpoint_parameter_schema_hash": "parameter_schema_mismatch",
            "parent_state_hash": "parent_mismatch", "initial_state_hash": "parent_mismatch",
            "parent_checkpoint_id": "parent_mismatch", "base_state_sha256": "parent_mismatch",
            "task_semantics_hash": "target_semantics_mismatch", "label_map_hash": "target_semantics_mismatch",
            "evaluation_protocol_hash": "evaluation_protocol_mismatch", "preprocessing_contract_hash": "target_semantics_mismatch",
        }.get(field, "provenance_integrity_failed")
        super().__init__(message)


class FedAvgCheckpointError(FedAvgValidationError):
    stage = "checkpoint"
    code = "checkpoint_integrity_failed"


class FedAvgAggregationExecutionError(FedAvgValidationError):
    stage = "aggregation"
    code = "parameter_aggregation_failed"


class FedAvgInferenceError(FedAvgValidationError):
    stage = "inference"
    code = "task_execution_failed"


@dataclass(frozen=True)
class SelectedFedAvgMember:
    service_id: str
    client_id: str
    sample_count: int
    row: Mapping[str, Any]


@dataclass(frozen=True)
class FedAvgCompositionExecution:
    outputs: Any
    task_family: str
    label_hash: str
    inference_latency_s: float
    aggregation_wall_s: float
    aggregation_provenance: Mapping[str, Any]

    @property
    def aggregated_state_hash(self) -> str:
        return str(self.aggregation_provenance["global_state_hash"])


_SHARED_TEXT_FIELDS = (
    "fl_run_id",
    "lineage_id",
    "compatibility_contract_hash",
    "architecture_id",
    "parameter_schema_hash",
    "label_map_hash",
    "preprocessing_contract_hash",
    "logical_dataset_contract_hash",
    "parent_state_hash",
    "task_family",
)


def resolve_selected_fedavg_members(
    db_path: str | Path, service_ids: Sequence[str]
) -> list[SelectedFedAvgMember]:
    """Resolve and validate exact FedAvg/checkpoint provenance for selected IDs."""
    ordered_ids = [str(value) for value in service_ids]
    if len(ordered_ids) < 2:
        raise FedAvgProvenanceError(
            "FedAvg composition validation requires at least two selected services"
        )
    if len(set(ordered_ids)) != len(ordered_ids):
        raise FedAvgProvenanceError("selected composition contains duplicate service IDs")

    with sqlite3.connect(str(db_path)) as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(service_generation_provenance)")}
        if "family_contract_version" in columns:
            marks = ",".join("?" for _ in ordered_ids)
            versions = [r[0] for r in conn.execute(
                f"SELECT family_contract_version FROM service_generation_provenance WHERE service_id IN ({marks})", ordered_ids)]
            if any(v in {"independent_family_v1", "independent_family_v2"} for v in versions):
                if len(versions) != len(ordered_ids) or len(set(versions)) != 1:
                    raise FedAvgProvenanceError("composition mixes independent families and historical services")
                return _resolve_independent_family_members(db_path, ordered_ids)
        if "cohort_id" in columns:
            marks = ",".join("?" for _ in ordered_ids)
            modes = [r[0] for r in conn.execute(
                f"SELECT generation_mode FROM service_generation_provenance WHERE service_id IN ({marks})", ordered_ids)]
            if "shared_parent" in modes:
                if len(modes) != len(ordered_ids) or set(modes) != {"shared_parent"}:
                    raise FedAvgProvenanceError("composition mixes shared-parent and legacy services")
                return _resolve_shared_parent_members(db_path, ordered_ids)

    placeholders = ",".join("?" for _ in ordered_ids)
    query = f"""
        SELECT
            s.service_id,
            s.status AS service_status,
            lower(COALESCE(s.task_family, s.task_type, '')) AS task_family,
            g.architecture_id,
            g.parameter_schema_hash,
            g.label_map_hash,
            g.preprocessing_contract_hash,
            g.logical_dataset_contract_hash,
            g.lineage_id,
            g.fl_run_id,
            g.client_id AS provenance_client_id,
            g.round_id AS provenance_round_id,
            g.compatibility_contract_hash,
            g.parent_global_round,
            g.parent_state_hash,
            g.participated,
            f.fl_run_id AS client_fl_run_id,
            f.client_id AS client_id,
            f.round_id AS client_round_id,
            f.succeeded,
            f.sample_count,
            f.parent_global_round AS client_parent_global_round,
            f.parent_state_hash AS client_parent_state_hash,
            f.final_state_hash,
            f.compatibility_contract_hash AS client_compatibility_contract_hash,
            c.storage_policy_version,
            c.storage_mode,
            c.parameter_schema_hash AS checkpoint_parameter_schema_hash,
            c.base_state_sha256,
            c.target_state_sha256, c.metadata_json AS checkpoint_metadata_json,
            c.reconstruction_artifact_ref,
            c.explicit_base_state_ref,
            c.round_trip_status,
            gs.state_hash AS global_parent_state_hash,
            gs.parameter_schema_hash AS global_parameter_schema_hash,
            gs.cas_ref AS global_parent_state_ref
        FROM services s
        LEFT JOIN service_generation_provenance g ON g.service_id = s.service_id
        LEFT JOIN fedavg_client_rounds f ON f.service_id = s.service_id
        LEFT JOIN service_checkpoint_storage c ON c.service_id = s.service_id
        LEFT JOIN fedavg_global_states gs
          ON gs.fl_run_id = g.fl_run_id
         AND gs.global_round = g.parent_global_round
        WHERE s.service_id IN ({placeholders})
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            raw_rows = [dict(row) for row in conn.execute(query, ordered_ids)]
    except sqlite3.Error as error:
        raise FedAvgProvenanceError(
            f"catalogue lacks required FedAvg validation provenance: {error}"
        ) from error

    by_id = {str(row["service_id"]): row for row in raw_rows}
    missing = [service_id for service_id in ordered_ids if service_id not in by_id]
    if missing:
        raise FedAvgProvenanceError(
            "selected composition services were not found: " + ", ".join(missing)
        )
    rows = [by_id[service_id] for service_id in ordered_ids]

    for field in _SHARED_TEXT_FIELDS:
        values = [_required_text(row.get(field), field, row["service_id"]) for row in rows]
        if len(set(values)) != 1:
            raise FedAvgProvenanceError(
                f"selected FedAvg services do not share {field}"
            )
    if rows[0]["task_family"] not in {"classification", "regression"}:
        raise FedAvgProvenanceError(
            "selected FedAvg services must share classification or regression"
        )

    shared_round = _shared_integer(rows, "provenance_round_id")
    parent_round = _shared_integer(rows, "parent_global_round")
    if shared_round != parent_round + 1:
        raise FedAvgProvenanceError(
            "selected client round must be exactly one greater than its parent global round"
        )

    members: list[SelectedFedAvgMember] = []
    seen_clients: set[str] = set()
    for row in rows:
        service_id = str(row["service_id"])
        client_id = _required_text(row.get("client_id"), "client_id", service_id)
        if client_id in seen_clients:
            raise FedAvgProvenanceError(
                f"selected FedAvg composition contains duplicate client_id '{client_id}'"
            )
        seen_clients.add(client_id)
        if str(row.get("service_status") or "").lower() != "completed":
            raise FedAvgProvenanceError(f"service '{service_id}' is not completed")
        if int(row.get("participated") or 0) != 1 or int(row.get("succeeded") or 0) != 1:
            raise FedAvgProvenanceError(
                f"service '{service_id}' is not a successful participating FedAvg client"
            )
        _require_equal(row, "provenance_client_id", "client_id", service_id)
        _require_equal(row, "fl_run_id", "client_fl_run_id", service_id)
        _require_equal(row, "provenance_round_id", "client_round_id", service_id)
        _require_equal(row, "parent_global_round", "client_parent_global_round", service_id)
        _require_equal(row, "parent_state_hash", "client_parent_state_hash", service_id)
        _require_equal(
            row,
            "compatibility_contract_hash",
            "client_compatibility_contract_hash",
            service_id,
        )
        _require_equal(
            row,
            "parameter_schema_hash",
            "checkpoint_parameter_schema_hash",
            service_id,
        )
        _require_equal(row, "parent_state_hash", "base_state_sha256", service_id)
        _require_equal(row, "final_state_hash", "target_state_sha256", service_id)
        _require_equal(row, "parent_state_hash", "global_parent_state_hash", service_id)
        _require_equal(
            row, "parameter_schema_hash", "global_parameter_schema_hash", service_id
        )
        if row.get("storage_policy_version") != "fedavg_round_base_exact_patch_v1" and not _is_reduced_checkpoint(row):
            raise FedAvgProvenanceError(
                f"service '{service_id}' lacks a verified FedAvg round checkpoint"
            )
        if (row.get("storage_mode") != "shared_base_exact_patch" and not _is_reduced_checkpoint(row)):
            raise FedAvgProvenanceError(
                f"service '{service_id}' checkpoint is not reconstructable as parent plus exact patch"
            )
        if row.get("round_trip_status") != "passed":
            raise FedAvgProvenanceError(
                f"service '{service_id}' checkpoint round-trip was not verified"
            )
        if not _present(row.get("reconstruction_artifact_ref")):
            raise FedAvgProvenanceError(
                f"service '{service_id}' lacks a reconstruction artifact"
            )
        if not _present(row.get("global_parent_state_ref")):
            raise FedAvgProvenanceError(
                f"service '{service_id}' lacks its persisted global parent state"
            )
        try:
            sample_count = int(row["sample_count"])
        except (TypeError, ValueError) as error:
            raise FedAvgProvenanceError(
                f"service '{service_id}' lacks a valid FedAvg sample count"
            ) from error
        if sample_count <= 0:
            raise FedAvgProvenanceError(
                f"service '{service_id}' FedAvg sample count must be positive"
            )
        members.append(SelectedFedAvgMember(service_id, client_id, sample_count, row))
    return members


def _resolve_independent_family_members(db_path, service_ids) -> list[SelectedFedAvgMember]:
    import json
    from mlaas_data_generator.cli.manifest.preparation import read_preparation
    marks = ",".join("?" for _ in service_ids)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(f"""
            SELECT g.*, s.status AS service_status, s.task_family,
                   c.storage_mode, c.round_trip_status, c.base_state_sha256, c.target_state_sha256, c.metadata_json AS checkpoint_metadata_json,
                   c.parameter_schema_hash AS checkpoint_parameter_schema_hash,
                   c.reconstruction_artifact_ref, c.explicit_base_state_ref
            FROM service_generation_provenance g JOIN services s USING(service_id)
            JOIN service_checkpoint_storage c USING(service_id)
            WHERE g.service_id IN ({marks})
        """, service_ids)]
    by_id = {r["service_id"]: r for r in rows}
    if set(by_id) != set(service_ids):
        raise FedAvgProvenanceError("selected family services lack checkpoint provenance")
    for field in ("family_id", "initial_state_hash", "compatibility_contract_hash", "task_family"):
        if len({_required_text(r.get(field), field, r["service_id"]) for r in rows}) != 1:
            raise FedAvgProvenanceError(f"selected family services disagree on {field}", field=field)
    members = []
    for sid in service_ids:
        from mlaas_data_generator.tasks.contracts import provenance_contract_values
        row = provenance_contract_values(by_id[sid])
        doc = read_preparation(row["family_document_ref"], row["family_document_hash"])
        if row["composition_compatible"] != 1 or row["service_status"] != "completed":
            raise FedAvgProvenanceError("selected service is not a published compatible family member")
        for field in ("family_id", "initial_state_hash", "compatibility_contract_hash"):
            if row[field] != doc[field]:
                raise FedAvgProvenanceError(f"service differs from its frozen family: {field}")
        for field, expected in doc["compatibility"].items():
            if row[field] != expected:
                raise FedAvgProvenanceError(f"service task interface differs: {field}", field=field)
        for left, right in (("initial_state_hash", "base_state_sha256"), ("parent_state_hash", "initial_state_hash"),
                            ("final_model_hash", "target_state_sha256"), ("parameter_schema_hash", "checkpoint_parameter_schema_hash")):
            _require_equal(row, left, right, sid)
        if row["round_trip_status"] != "passed" or not row["reconstruction_artifact_ref"]:
            raise FedAvgProvenanceError("family checkpoint is not reconstructable")
        if row["storage_mode"] != "shared_base_exact_patch" and not _is_reduced_checkpoint(row):
            raise FedAvgProvenanceError("family checkpoint lacks verified representation/performance evidence")
        evidence = json.loads(row["provenance_json"])["independent_family"]
        source_ids = evidence["source_ids"]
        if not source_ids or len(source_ids) != len(set(source_ids)) or not set(source_ids) <= set(doc["reservation"]["training_ids"]):
            raise FedAvgProvenanceError("invalid family client training membership")
        row.update(parent_document=doc, source_ids=source_ids)
        members.append(SelectedFedAvgMember(sid, sid, len(source_ids), row))
    return members


def _resolve_shared_parent_members(db_path, service_ids) -> list[SelectedFedAvgMember]:
    import json
    from mlaas_data_generator.shared_parent import verify_parent
    placeholders = ",".join("?" for _ in service_ids)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        raw = [dict(r) for r in conn.execute(f"""
            SELECT s.service_id, s.status AS service_status, s.task_family,
                   g.*, cc.client_id AS recorded_client_id, cc.sample_count,
                   cc.status AS client_status, cc.final_state_hash,
                   cc.document_json AS client_document_json,
                   p.document_json AS parent_document_json, p.status AS parent_status,
                   c.storage_policy_version, c.storage_mode, c.round_trip_status,
                   c.base_state_sha256, c.target_state_sha256, c.metadata_json AS checkpoint_metadata_json,
                   c.parameter_schema_hash AS checkpoint_parameter_schema_hash,
                   c.reconstruction_artifact_ref, c.explicit_base_state_ref
            FROM services s
            JOIN service_generation_provenance g ON g.service_id=s.service_id
            JOIN cohort_clients cc ON cc.service_id=s.service_id AND cc.cohort_id=g.cohort_id
            JOIN service_cohorts cohort ON cohort.cohort_id=g.cohort_id AND cohort.parent_checkpoint_id=g.parent_checkpoint_id
            JOIN parent_checkpoints p ON p.parent_checkpoint_id=g.parent_checkpoint_id
            JOIN service_checkpoint_storage c ON c.service_id=s.service_id
            WHERE s.service_id IN ({placeholders})
        """, service_ids)]
    by_id = {row["service_id"]: row for row in raw}
    if set(by_id) != set(service_ids):
        raise FedAvgProvenanceError("selected services lack complete shared-parent/checkpoint provenance")
    from mlaas_data_generator.tasks.contracts import provenance_contract_values
    rows = [provenance_contract_values(by_id[sid]) for sid in service_ids]
    for field in (*[f for f in _SHARED_TEXT_FIELDS if f != "fl_run_id"], "cohort_id", "parent_checkpoint_id"):
        values = [_required_text(r.get(field), field, r["service_id"]) for r in rows]
        if len(set(values)) != 1:
            raise FedAvgProvenanceError(f"selected shared-parent services disagree on {field}", field=field)
    parent = json.loads(rows[0]["parent_document_json"])
    verify_parent(parent)
    if rows[0]["task_family"] not in {"classification", "regression", "token_classification", "segmentation", "generation"}:
        raise FedAvgProvenanceError("validation_not_implemented: runtime task has no protocol")
    members = []
    seen = set()
    for row in rows:
        sid = row["service_id"]
        client = row["client_id"]
        if client in seen or client != row["recorded_client_id"]:
            raise FedAvgProvenanceError("duplicate or inconsistent client identity")
        seen.add(client)
        if any(row[k] != "completed" for k in ("service_status", "client_status", "parent_status")):
            raise FedAvgProvenanceError(f"service {sid} or its parent is incomplete")
        for field, expected in parent["compatibility"].items():
            if row.get(field) != expected:
                raise FedAvgProvenanceError(f"service {sid} differs from parent {field}", field=field)
        if row["parent_state_hash"] != parent["parent_state_hash"]:
            raise FedAvgProvenanceError("service parent state differs from parent checkpoint")
        if row["compatibility_contract_hash"] != parent["compatibility_contract_hash"]:
            raise FedAvgProvenanceError("service compatibility hash differs from parent")
        for left, right in (("parameter_schema_hash", "checkpoint_parameter_schema_hash"),
                            ("parent_state_hash", "base_state_sha256"),
                            ("final_state_hash", "target_state_sha256"), ("final_model_hash", "target_state_sha256")):
            _require_equal(row, left, right, sid)
        if (row["storage_mode"] != "shared_base_exact_patch" and not _is_reduced_checkpoint(row)) or row["round_trip_status"] != "passed":
            raise FedAvgProvenanceError(f"service {sid} lacks a verified exact checkpoint")
        if not row["reconstruction_artifact_ref"] or int(row["sample_count"] or 0) <= 0:
            raise FedAvgProvenanceError(f"service {sid} lacks reconstruction or training counts")
        partition = json.loads(row["client_document_json"])["partition"]
        if len(partition["source_ids"]) != row["sample_count"] or len(set(partition["source_ids"])) != row["sample_count"]:
            raise FedAvgProvenanceError("client sample count differs from unique partition membership")
        if not set(partition["source_ids"]) <= set(parent["reservation"]["training_ids"]):
            raise FedAvgProvenanceError("client training membership includes sources outside the frozen training partition", code="audit_overlap")
        row["parent_document"] = parent
        row["source_ids"] = partition["source_ids"]
        members.append(SelectedFedAvgMember(sid, client, int(row["sample_count"]), row))
    return members


def execute_selected_fedavg_composition(
    *, db_path: str | Path, service_ids: Sequence[str], plan: CohortPlan
) -> FedAvgCompositionExecution:
    """Aggregate selected client parameters and execute one resulting model."""
    members = resolve_selected_fedavg_members(db_path, service_ids)
    first = members[0]
    if first.row.get("generation_mode") == "shared_parent" or first.row.get("family_contract_version") in {"independent_family_v1", "independent_family_v2"}:
        reservation = first.row["parent_document"]["reservation"]
        for requested, field in ((plan.training_ids, "training_ids"),
                                 (plan.catalogue_metric_ids, "catalogue_metric_ids"), (plan.audit_ids, "audit_ids")):
            if set(requested) != set(reservation[field]):
                raise FedAvgProvenanceError(f"composition {field} differs from the pre-training reservation")
    root = _checkpoint_root(first.row, db_path)
    try:
        parent = reconstruct_service_base_state(root, first.service_id)
    except Exception as error:  # noqa: BLE001 - preserve checkpoint-stage evidence
        raise FedAvgCheckpointError(
            f"could not reconstruct parent state for '{first.service_id}': {error}"
        ) from error
    expected_parent_hash = str(first.row["parent_state_hash"])
    if parent.sha256 != expected_parent_hash:
        raise FedAvgCheckpointError(
            "reconstructed parent state does not match selected FedAvg provenance"
        )

    contributors: list[FedAvgContributor] = []
    for member in members:
        member_root = _checkpoint_root(member.row, db_path)
        try:
            state = reconstruct_service_state(member_root, member.service_id)
        except Exception as error:  # noqa: BLE001
            raise FedAvgCheckpointError(
                f"could not reconstruct client state for '{member.service_id}': {error}"
            ) from error
        if state.sha256 != str(member.row["target_state_sha256"]):
            raise FedAvgCheckpointError(
                f"client state hash does not match provenance for '{member.service_id}'"
            )
        contributors.append(
            FedAvgContributor(
                client_id=member.client_id,
                service_id=member.service_id,
                sample_count=member.sample_count,
                state=state,
                compatibility_contract_hash=str(
                    member.row["compatibility_contract_hash"]
                ),
                parent_state_hash=expected_parent_hash,
            )
        )

    aggregation_started = time.perf_counter()
    try:
        aggregation = aggregate_fedavg_states(
            parent,
            contributors,
            compatibility_contract_hash=str(
                first.row["compatibility_contract_hash"]
            ),
        )
    except Exception as error:  # noqa: BLE001
        raise FedAvgAggregationExecutionError(
            f"selected parameter aggregation failed: {error}"
        ) from error
    aggregation_wall_s = time.perf_counter() - aggregation_started

    runtime_id = f"selected_fedavg_{aggregation.state.sha256[:16]}"
    try:
        prediction = execute_tensor_state_prediction(
            db_path=db_path,
            representative_service_id=first.service_id,
            runtime_id=runtime_id,
            plan=plan,
            state=aggregation.state,
        )
    except Exception as error:  # noqa: BLE001
        raise FedAvgInferenceError(
            f"aggregated model execution failed: {error}"
        ) from error

    provenance = dict(aggregation.provenance)
    provenance.update(
        {
            "fl_run_id": first.row.get("fl_run_id"),
            "round_id": first.row.get("provenance_round_id"),
            "parent_global_round": first.row.get("parent_global_round"),
            "cohort_id": first.row.get("cohort_id"),
            "parent_checkpoint_id": first.row.get("parent_checkpoint_id"),
            "family_id": first.row.get("family_id"),
            "initial_state_hash": first.row.get("initial_state_hash"),
            "compatibility_contract_hash": str(
                first.row["compatibility_contract_hash"]
            ),
            "parameter_schema_hash": str(first.row["parameter_schema_hash"]),
            "selected_service_order": [member.service_id for member in members],
            "materialized_model_id": runtime_id,
        }
    )
    if first.row.get("generation_mode") == "shared_parent" or first.row.get("family_contract_version") in {"independent_family_v1", "independent_family_v2"}:
        unique = {sid for member in members for sid in member.row["source_ids"]}
        provenance.update(unique_training_examples=len(unique),
                          local_example_exposures=sum(m.sample_count for m in members),
                          cross_client_overlap_allowed=True,
                          parent_training_examples=(0 if first.row.get("family_contract_version") else len(first.row["parent_document"]["reservation"]["training_ids"])))
    return FedAvgCompositionExecution(
        outputs=prediction.outputs,
        task_family=str(first.row["task_family"]),
        label_hash=prediction.label_hash,
        inference_latency_s=float(prediction.latency_s),
        aggregation_wall_s=float(aggregation_wall_s),
        aggregation_provenance=provenance,
    )


def _checkpoint_root(row: Mapping[str, Any], db_path: str | Path) -> Path:
    for value in (
        row.get("explicit_base_state_ref"),
        row.get("global_parent_state_ref"),
        row.get("reconstruction_artifact_ref"),
    ):
        if not _present(value):
            continue
        path = Path(str(value)).resolve()
        for parent in path.parents:
            if parent.name in {"objects", "references"}:
                return parent.parent
    return Path(db_path).resolve().parent / "service_artifacts"


def _shared_integer(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    try:
        values = [int(row[field]) for row in rows]
    except (TypeError, ValueError, KeyError) as error:
        raise FedAvgProvenanceError(
            f"selected FedAvg services lack a valid {field}"
        ) from error
    if len(set(values)) != 1:
        raise FedAvgProvenanceError(
            f"selected FedAvg services do not share {field}"
        )
    return values[0]


def _is_reduced_checkpoint(row):
    import json
    metadata = json.loads(row.get("checkpoint_metadata_json") or "{}")
    return (row.get("storage_mode") == "full_object" and metadata.get("lossy") is True
            and metadata.get("reloaded_state_sha256") == row.get("target_state_sha256")
            and metadata.get("quality_validation", {}).get("status") == "passed"
            and bool(metadata.get("source_state_sha256")))


def _require_equal(
    row: Mapping[str, Any], left: str, right: str, service_id: str
) -> None:
    expected = row.get(right)
    if _is_reduced_checkpoint(row) and right == "target_state_sha256" and left in {"final_state_hash", "final_model_hash"}:
        import json
        metadata = json.loads(row.get("checkpoint_metadata_json") or "{}")
        if metadata.get("quality_validation", {}).get("status") != "passed":
            raise FedAvgProvenanceError("reduced checkpoint lacks a passed performance validation")
        expected = metadata.get("source_state_sha256")
    if row.get(left) is None or expected is None or row.get(left) != expected:
        raise FedAvgProvenanceError(
            f"service '{service_id}' has inconsistent {left}/{right} provenance", field=left
        )


def _required_text(value: Any, field: str, service_id: str) -> str:
    if not _present(value):
        raise FedAvgProvenanceError(
            f"service '{service_id}' lacks required {field} provenance"
        )
    return str(value)


def _present(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "none", "null", "nan", "unknown"}


__all__ = [
    "FedAvgAggregationExecutionError",
    "FedAvgCheckpointError",
    "FedAvgCompositionExecution",
    "FedAvgInferenceError",
    "FedAvgProvenanceError",
    "FedAvgValidationError",
    "SelectedFedAvgMember",
    "execute_selected_fedavg_composition",
    "resolve_selected_fedavg_members",
]
