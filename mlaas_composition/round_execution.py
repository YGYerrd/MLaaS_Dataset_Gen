"""Local sequential federated rounds with measured wall time and audit accuracy."""

from contextlib import closing
import csv
import gc
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from .round_manifest import VERSION


def execute_rounds(source_db, entry, members, plan, labels, selected, *, device, on_progress=None):
    from mlaas_data_generator.artifact_paths import resolve_artifact_path
    from mlaas_data_generator.checkpoint_store import load_lineage_base, capture_model_tensor_state
    from mlaas_data_generator.client_reconstruction import materialize_service
    from mlaas_data_generator.fedavg_aggregation import FedAvgContributor, aggregate_fedavg_states
    from mlaas_data_generator.cli.run_manifest import _clear_cuda_runtime_state
    from .regeneration import execute_raw_family_state
    from .validation import measure_model_outputs, measured_recipe_quality

    def progress(stage, **values):
        if on_progress:
            on_progress({"stage": stage, **values})

    started = time.perf_counter()
    members = sorted(members, key=lambda m: m.service_id)
    first, config = members[0].row, entry["round_config"]
    doc = first["parent_document"]
    global_state = load_lineage_base(resolve_artifact_path(doc["store_root"]), doc["lineage_id"],
                                     expected_parameter_schema_hash=first["parameter_schema_hash"])
    if global_state.sha256 != first["initial_state_hash"]:
        raise ValueError("family initial state hash changed")
    deadline = config["round_deadline_s"]
    total_training = total_aggregation = total_evaluation = 0.0
    history = []
    for number, availability in enumerate(config["client_availability"], 1):
        progress("round_started", communication_round=number)
        round_started = time.perf_counter()
        parent = global_state
        contributors, clients = [], []
        for member in members:
            sid = member.service_id
            client = {"client_id": sid, "client_availability": availability[sid],
                      "status": "unavailable", "accepted": False, "training_wall_s": None,
                      "aggregation_weight": 0.0, "parent_state_hash": parent.sha256,
                      "materialized_state_hash": None}
            clients.append(client)
            if not availability[sid]:
                continue
            if time.perf_counter() - round_started >= deadline:
                client["status"] = "deadline_skipped"
                continue
            progress("training", service_id=sid, communication_round=number)
            train_started = time.perf_counter()
            try:
                with materialize_service(source_db, sid, device=device, initial_state=parent) as local:
                    if local["initial"].sha256 != parent.sha256:
                        raise ValueError("client did not start from this round's global model")
                    state = capture_model_tensor_state(local["model"])
                    client["training"] = local["training"]
                    client["replay_evidence"] = local["replay_evidence"]
                del local
            except Exception as error:
                error.service_id = sid
                raise
            elapsed = time.perf_counter() - round_started
            duration = time.perf_counter() - train_started
            total_training += duration
            client.update(training_wall_s=duration, materialized_state_hash=state.sha256,
                          update_arrival_time_s=elapsed, status="completed" if elapsed <= deadline else "late",
                          accepted=elapsed <= deadline)
            if client["accepted"]:
                contributors.append(FedAvgContributor(member.client_id, sid, member.sample_count, state,
                                                       first["compatibility_contract_hash"], parent.sha256))
            del state
            gc.collect()
            _clear_cuda_runtime_state()
            progress("client_completed", service_id=sid, communication_round=number)

        progress("aggregation", communication_round=number)
        aggregation_started = time.perf_counter()
        aggregation_provenance = None
        if contributors:
            aggregation = aggregate_fedavg_states(parent, contributors,
                compatibility_contract_hash=first["compatibility_contract_hash"], minimum_contributors=1)
            global_state, aggregation_provenance = aggregation.state, aggregation.provenance
            weights = {c["service_id"]: c["normalized_weight"] for c in aggregation_provenance["contributors"]}
            for client in clients:
                client["aggregation_weight"] = weights.get(client["client_id"], 0.0)
            del aggregation
        aggregation_s = time.perf_counter() - aggregation_started
        total_aggregation += aggregation_s
        del contributors, parent
        progress("evaluation", communication_round=number)
        evaluation_started = time.perf_counter()
        prediction = execute_raw_family_state(first, f"round_{number}_{global_state.sha256[:16]}",
                                               plan, global_state, device=device)
        if prediction.task_family != "classification" or prediction.label_hash != plan.label_hash:
            raise ValueError("round model outputs disagree with the frozen audit contract")
        actual = measure_model_outputs(prediction.outputs, plan, labels, task_family="classification",
                                       inference_latency_s=prediction.latency_s)
        actual = measured_recipe_quality(actual, prediction.outputs, labels, selected)
        accuracy = actual.get("accuracy")
        if accuracy is None or not 0 <= accuracy <= 1:
            raise ValueError("round evaluation did not produce a valid measured accuracy")
        evaluation_s = time.perf_counter() - evaluation_started
        total_evaluation += evaluation_s
        completion_s = time.perf_counter() - round_started
        result = {"communication_round": number, "round_deadline_s": deadline,
                  "round_completion_time_s": completion_s, "global_accuracy_per_round": accuracy,
                  "deadline_violation": completion_s > deadline, "clients": clients,
                  "status": "updated" if any(c["accepted"] for c in clients) else "no_update",
                  "parent_state_hash": clients[0]["parent_state_hash"], "global_state_hash": global_state.sha256,
                  "aggregation_wall_s": aggregation_s, "evaluation_wall_s": evaluation_s,
                  "aggregation_provenance": aggregation_provenance, "metrics": actual,
                  "deadline_profile": config["deadline_profile"], "baseline": config["baseline"],
                  "timing_scope": config["timing_scope"], "communication": config["communication"],
                  "availability_basis": config["availability_basis"]}
        history.append(result)
        progress("round_completed", communication_round=number, result=result)
    wall_s = time.perf_counter() - started
    return {"status": "completed", "wall_s": wall_s, "rounds": history, "members": [],
            "metrics": {**actual, "total_wall_s": wall_s, "training_wall_s": total_training,
                        "aggregation_wall_s": total_aggregation, "evaluation_wall_s": total_evaluation,
                        "audit_examples": len(plan.audit_ids), "rounds_completed": len(history)},
            "provenance": {"version": VERSION, "family_id": entry["family_id"],
                           "initial_state_hash": first["initial_state_hash"],
                           "global_state_hash": global_state.sha256, "round_config": config,
                           "rounds_completed": len(history), "selected_service_order": entry["service_ids"]}}


EXPORT_COLUMNS = ("run_id", "combination_id", "family_id", "composition_size", "composition_status",
    "client_id", "communication_round", "client_availability", "round_deadline_s",
    "round_completion_time_s", "global_accuracy_per_round", "deadline_profile", "deadline_violation",
    "round_status", "client_status", "update_accepted", "aggregation_weight", "client_training_wall_s",
    "parent_state_hash", "global_state_hash", "baseline_round_time_s", "baseline_sample_count",
    "timing_scope", "communication", "availability_basis")


def export_rounds(db, output, *, overwrite=False):
    """Export every completed round, including rounds from interrupted experiments."""
    db, output = Path(db).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".csv" or output == db or (output.exists() and output.samefile(db)):
        raise ValueError("Round export must be a .csv separate from the results database")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Export exists; use --overwrite: {output}")
    temporary = None
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        for record in conn.execute("SELECT manifest_path,source_db FROM composition_runs"):
            for source in record:
                path = Path(source).resolve()
                if output == path or (output.exists() and path.exists() and output.samefile(path)):
                    raise ValueError("Round export must not replace a source input")
        has_rounds = conn.execute("SELECT 1 FROM sqlite_master WHERE name='composition_rounds'").fetchone()
        records = conn.execute("""SELECT r.*,c.family_id,c.composition_size,c.status AS composition_status
            FROM composition_rounds r JOIN compositions c USING(run_id,combination_id)
            ORDER BY r.run_id,r.combination_id,r.communication_round""") if has_rounds else ()
        output.parent.mkdir(parents=True, exist_ok=True)
        rows, rounds = 0, 0
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", delete=False,
                    dir=output.parent, prefix=f".{output.name}.", suffix=".tmp") as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS)
                writer.writeheader()
                for record in records:
                    result = json.loads(record["result_json"])
                    for client in result["clients"]:
                        row = {key: record[key] for key in EXPORT_COLUMNS if key in record.keys()}
                        row.update({key: result[key] for key in ("deadline_profile", "deadline_violation",
                            "parent_state_hash", "global_state_hash", "timing_scope", "communication", "availability_basis")})
                        row.update({key: client[key] for key in ("client_id", "client_availability", "aggregation_weight")})
                        row.update(round_status=result["status"], client_status=client["status"],
                            update_accepted=client["accepted"], client_training_wall_s=client["training_wall_s"],
                            baseline_round_time_s=result["baseline"]["baseline_round_time_s"],
                            baseline_sample_count=result["baseline"]["sample_count"])
                        writer.writerow(row)
                        rows += 1
                    rounds += 1
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, output)
            else:
                os.link(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"output": str(output), "rows": rows, "completed_rounds": rounds, "schema_version": VERSION}
