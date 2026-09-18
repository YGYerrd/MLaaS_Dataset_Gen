"""Materialise selected recipe-backed providers and execute their parameter average."""

from __future__ import annotations

import gc
import time

from .raw_family import member_identity, verified_member


def resolve_members(db_path, service_ids):
    from .fedavg_validation import SelectedFedAvgMember, FedAvgProvenanceError
    cache = {}
    try:
        rows = [verified_member(db_path, sid, cache=cache) for sid in sorted(service_ids)]
        for key in ("family_id", "initial_state_hash", "compatibility_contract_hash", "task_family"):
            if len({r[key] for r in rows}) != 1:
                raise ValueError(f"selected services disagree on {key}")
    except Exception as error:
        raise FedAvgProvenanceError(str(error)) from error
    return [SelectedFedAvgMember(r["service_id"], r["service_id"], r["sample_count"], r) for r in rows]


def execute_composition(db_path, members, plan, *, device="cpu"):
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
        parent = load_lineage_base(doc["store_root"], doc["lineage_id"],
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
        try:
            if row["weight_retention"] == "recipe_only":
                with materialize_service(db_path, member.service_id, device=device) as replay:
                    if replay["initial"].sha256 != parent.sha256:
                        raise ValueError("replayed initial state differs from aggregate parent")
                    state = capture_model_tensor_state(replay["model"])
                    evidence.update(replay_evidence=replay["replay_evidence"], training=replay["training"],
                                    materialization_method="recipe_replay", client_checkpoint_reads=0,
                                    reconstruction_fidelity="not_measured_in_this_run")
                del replay
            else:
                state = reconstruct_service_state(doc["store_root"], member.service_id)
                evidence.update(materialization_method="retained_checkpoint")
            evidence.update(materialized_state_hash=state.sha256,
                            weights_equal_original=state.sha256 == row["final_model_hash"],
                            reconstruction_wall_s=time.perf_counter() - started)
        except Exception as error:
            raise FedAvgReconstructionError(f"could not materialise {member.service_id}: {error}") from error
        contributors.append(FedAvgContributor(member.client_id, member.service_id, member.sample_count,
            state, first["compatibility_contract_hash"], parent.sha256))
        observations.append(evidence)
        gc.collect()

    started = time.perf_counter()
    try:
        aggregation = aggregate_fedavg_states(parent, contributors,
            compatibility_contract_hash=first["compatibility_contract_hash"])
    except Exception as error:
        raise FedAvgAggregationExecutionError(str(error)) from error
    aggregation_wall_s = time.perf_counter() - started
    del contributors, state
    runtime_id = "selected_recipe_" + aggregation.state.sha256[:16]
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
        audit_reservation_hash=reservation["reservation_hash"], runtime=runtime_provenance(), device=device)
    return FedAvgCompositionExecution(prediction.outputs, first["task_family"], prediction.label_hash,
        prediction.latency_s, aggregation_wall_s, _clean(provenance))
