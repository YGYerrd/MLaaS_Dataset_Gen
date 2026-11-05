"""Materialise selected recipe-backed providers and execute their parameter average."""

from __future__ import annotations

import gc
import time
from mlaas_data_generator.artifact_paths import resolve_artifact_path

from .raw_family import member_identity, verified_member


def resolve_members(db_path, service_ids, *, require_replay=False, cache=None):
    from .fedavg_validation import SelectedFedAvgMember, FedAvgProvenanceError
    cache = {} if cache is None else cache
    try:
        if len(service_ids) < 2 or len(set(service_ids)) != len(service_ids):
            raise ValueError("composition requires at least two distinct services")
        rows = [verified_member(db_path, sid, cache=cache, require_replay=require_replay)
                for sid in sorted(service_ids)]
        for key in ("family_id", "initial_state_hash", "compatibility_contract_hash", "task_family"):
            if len({r[key] for r in rows}) != 1:
                raise ValueError(f"selected services disagree on {key}")
    except Exception as error:
        raise FedAvgProvenanceError(str(error)) from error
    return [SelectedFedAvgMember(r["service_id"], r["service_id"], r["sample_count"], r) for r in rows]


def execute_composition(db_path, members, plan, *, device="cpu", replay=False, on_progress=None,
                        capture_client_losses=False):
    """Execute one average; replay=True always retrains, even retained members.

    Progress carries only small observations, never model objects or tensors.
    """
    from mlaas_data_generator.checkpoint_store import (
        capture_model_tensor_state, load_lineage_base, reconstruct_service_state)
    from mlaas_data_generator.client_reconstruction import materialize_service
    from mlaas_data_generator.fedavg_aggregation import FedAvgContributor, aggregate_fedavg_states
    from mlaas_data_generator.provenance import runtime_provenance, sha256_json
    from mlaas_data_generator.shared_parent import _clean
    from .fedavg_validation import (FedAvgCompositionExecution, FedAvgCheckpointError,
        FedAvgReconstructionError, FedAvgAggregationExecutionError, FedAvgInferenceError)
    from .regeneration import execute_raw_family_state
    from .validation import assert_cohort_isolation, CohortIsolationError

    members = sorted(members, key=lambda member: member.service_id)
    if capture_client_losses and not replay:
        raise ValueError("client loss capture requires replay of every composition member")
    if len(members) < 2 or len({m.service_id for m in members}) != len(members):
        raise ValueError("composition requires at least two distinct services")
    first = members[0].row
    doc = first["parent_document"]
    reservation = doc["reservation"]
    if plan.contract_hash != sha256_json(reservation["contract"]):
        raise CohortIsolationError("audit plan contract differs from the family reservation")
    training_ids = {sid for m in members for sid in m.row["source_ids"]}
    if set(plan.training_ids) != training_ids:
        raise CohortIsolationError("audit plan training membership differs from selected recipes")
    for key in ("catalogue_metric_ids", "audit_ids"):
        if tuple(getattr(plan, key)) != tuple(reservation[key]):
            raise CohortIsolationError(f"audit plan differs from family reservation: {key}")
    assert_cohort_isolation(training_ids, plan.catalogue_metric_ids, plan.audit_ids)
    try:
        parent = load_lineage_base(resolve_artifact_path(doc["store_root"]), doc["lineage_id"],
            expected_parameter_schema_hash=first["parameter_schema_hash"])
        if parent.sha256 != first["initial_state_hash"]:
            raise ValueError("family initial state hash changed")
    except Exception as error:
        raise FedAvgCheckpointError(str(error)) from error

    contributors, observations = [], []
    for member in members:
        row = member.row
        print(f"Materialising {member.service_id} ({row['weight_retention']})", flush=True)
        started = time.perf_counter()
        evidence = member_identity(row)
        if on_progress:
            on_progress({"stage": "training", "service_id": member.service_id})
        try:
            if replay or row["weight_retention"] == "recipe_only":
                options = {"capture_client_losses": True} if capture_client_losses else {}
                with materialize_service(db_path, member.service_id, device=device, **options) as materialized:
                    if materialized["initial"].sha256 != parent.sha256:
                        raise ValueError("replayed initial state differs from aggregate parent")
                    state = capture_model_tensor_state(materialized["model"])
                    evidence.update(replay_evidence=materialized["replay_evidence"], training=materialized["training"],
                                    materialization_method="recipe_replay", client_checkpoint_reads=0,
                                    reconstruction_fidelity="not_measured_in_this_run")
                    if capture_client_losses:
                        evidence.update(materialized["client_losses"])
                        evidence["client_loss_statistics"]["final_local_model"]["model_state_hash"] = state.sha256
                del materialized
            else:
                state = reconstruct_service_state(resolve_artifact_path(doc["store_root"]), member.service_id)
                evidence.update(materialization_method="retained_checkpoint")
            evidence.update(materialized_state_hash=state.sha256,
                            weights_equal_original=state.sha256 == row["final_model_hash"],
                            reconstruction_wall_s=time.perf_counter() - started - evidence.get("client_loss_measurement_wall_s", 0))
        except Exception as error:
            failure = FedAvgReconstructionError(f"could not materialise {member.service_id}: {error}")
            failure.service_id = member.service_id
            raise failure from error
        contributors.append(FedAvgContributor(member.client_id, member.service_id, member.sample_count,
            state, first["compatibility_contract_hash"], parent.sha256))
        observations.append(evidence)
        gc.collect()
        if replay:
            from mlaas_data_generator.cli.run_manifest import _clear_cuda_runtime_state
            _clear_cuda_runtime_state()
        if on_progress:
            on_progress({"stage": "member_completed", "service_id": member.service_id, "evidence": evidence})

    started = time.perf_counter()
    if on_progress:
        on_progress({"stage": "aggregation"})
    try:
        aggregation = aggregate_fedavg_states(parent, contributors,
            compatibility_contract_hash=first["compatibility_contract_hash"])
    except Exception as error:
        raise FedAvgAggregationExecutionError(str(error)) from error
    aggregation_wall_s = time.perf_counter() - started
    del contributors, state
    runtime_id = "selected_recipe_" + aggregation.state.sha256[:16]
    if on_progress:
        on_progress({"stage": "evaluation"})
    evaluation_started = time.perf_counter()
    try:
        prediction = execute_raw_family_state(first, runtime_id, plan, aggregation.state, device=device)
    except Exception as error:
        raise FedAvgInferenceError(str(error)) from error
    provenance = dict(aggregation.provenance)
    provenance.update(family_id=first["family_id"], family_contract_version=doc["version"],
        family_document_hash=first["family_document_hash"], initial_state_hash=parent.sha256,
        compatibility_contract_hash=first["compatibility_contract_hash"],
        parameter_schema_hash=first["parameter_schema_hash"], fl_run_id=None, round_id=None,
        parent_global_round=None, cohort_id=None, parent_checkpoint_id=None,
        selected_service_order=[m.service_id for m in members], materialized_model_id=runtime_id,
        unique_training_examples=len(training_ids), local_example_exposures=sum(m.sample_count for m in members),
        cross_client_overlap_allowed=True, parent_training_examples=0,
        materialization=observations, reconstruction_wall_s=sum(o["reconstruction_wall_s"] for o in observations),
        client_loss_measurement_wall_s=sum(o.get("client_loss_measurement_wall_s", 0) for o in observations),
        audit_reservation_hash=reservation["reservation_hash"], runtime=runtime_provenance(), device=device,
        evaluation_wall_s=time.perf_counter() - evaluation_started)
    return FedAvgCompositionExecution(prediction.outputs, first["task_family"], prediction.label_hash,
        prediction.latency_s, aggregation_wall_s, _clean(provenance))
