"""Replay and measure composition manifests, with one complete job per GPU."""

from __future__ import annotations

from collections import deque
from contextlib import closing
import csv
import gc
import json
import multiprocessing
import os
from pathlib import Path
import queue
import sqlite3
import time
import uuid

from filelock import FileLock
import pandas as pd

from mlaas_data_generator.diagnostic_json import bounded_text
from mlaas_data_generator.provenance import sha256_json
from mlaas_data_generator.artifact_paths import (
    artifact_path_scope, current_artifact_prefixes, load_artifact_path_map)
from .manifest import _digest
from .raw_family import member_identity, verified_member
from .recipe_execution import execute_composition
from .storage import CompositionWriter
from .validation import measure_model_outputs, measured_recipe_quality, supported_composition_reason


EXECUTOR_VERSION = "composition_batch_v1"
MIN_AUDIT_SAMPLES = 64


def load_composition_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"combination_id", "family_id", "composition_size", "service_ids"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"Composition manifest requires columns {sorted(required)}")
        entries, seen = [], set()
        for line, row in enumerate(reader, 2):
            try:
                family, cid = row["family_id"].strip(), row["combination_id"].strip()
                ids = json.loads(row["service_ids"])
                size = int(row["composition_size"])
                if not family or not isinstance(ids, list) or any(not isinstance(s, str) or not s.strip() for s in ids):
                    raise ValueError("family and service IDs must be nonempty strings")
                if size < 2 or len(ids) != size or len(set(ids)) != size:
                    raise ValueError("composition_size must match at least two distinct service IDs")
                ids = sorted(ids)
                if cid != "comb_" + _digest([family, ids]):
                    raise ValueError("combination_id does not match family and members")
                if cid in seen:
                    raise ValueError("duplicate combination_id")
                seen.add(cid)
                entries.append(dict(combination_id=cid, family_id=family, composition_size=size, service_ids=ids))
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"Invalid composition manifest line {line}: {error}") from error
    if not entries:
        raise ValueError("Composition manifest has no rows")
    return entries


def _metric(recipe, name):
    value = recipe["observed_metrics"].get(name)
    return value.get("value") if isinstance(value, dict) else value


def _snapshot(row):
    recipe, doc = row["recipe"], row["parent_document"]
    cfg = doc["config"]
    primary, secondary = (_metric(recipe, key) for key in ("primary_metric_name", "auxiliary_metric_name"))
    if not isinstance(primary, str) or not primary:
        raise ValueError("service recipe has no named primary metric")
    return {
        **member_identity(row), "sample_count": row["sample_count"], "task_family": row["task_family"],
        **{key: row.get(key) for key in ("service_status", "task_type", "model_type", "hf_task", "modality",
                                        "task_semantics_hash", "evaluation_protocol_hash")},
        "model_id": cfg.get("hf_model_id") or cfg.get("model_id") or cfg.get("model_type"),
        "dataset_name": cfg.get("dataset_name") or cfg.get("dataset_args", {}).get("dataset_name") or cfg.get("dataset"),
        "client_recipe_ref": row["client_recipe_ref"], "family_document_ref": row["family_document_ref"],
        "initial_state_ref": {"store_root": doc["store_root"], "lineage_id": doc["lineage_id"]},
        "compatibility_contract_hash": row["compatibility_contract_hash"],
        "parameter_schema_hash": row["parameter_schema_hash"],
        "audit_reservation_hash": doc["reservation"]["reservation_hash"],
        "primary_metric_name": primary, "auxiliary_metric_name": secondary,
        "metric_reference": {name: _metric(recipe, name) for name in (primary, secondary, "metric_score") if name},
    }


def _prepare_composition(source_db, entry, *, metadata_only=False, cache=None, member_cache=None):
    from .fedavg_validation import SelectedFedAvgMember
    from .cohort import read_cohort, read_cohort_labels
    rows = []
    for sid in entry["service_ids"]:
        try:
            if member_cache is not None and sid in member_cache:
                row = member_cache[sid]
                if isinstance(row, Exception):
                    raise row
            else:
                row = verified_member(source_db, sid, cache=cache, require_replay=True, verify_base=not metadata_only)
                if member_cache is not None:
                    member_cache[sid] = row
            rows.append(row)
        except Exception as error:
            if member_cache is not None:
                member_cache[sid] = error
            error.service_id = sid
            raise
    for key in ("family_id", "family_document_hash", "initial_state_hash", "compatibility_contract_hash",
                "parameter_schema_hash", "task_family"):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f"composition members disagree on {key}")
    if rows[0]["family_id"] != entry["family_id"]:
        raise ValueError("manifest family does not match the source services")
    snapshots = {}
    for row in rows:
        try:
            snapshots[row["service_id"]] = _snapshot(row)
        except Exception as error:
            error.service_id = row["service_id"]
            raise
    selected = pd.DataFrame([{**r["parent_document"]["config"], **r,
                              "primary_metric_name": snapshots[r["service_id"]]["primary_metric_name"],
                              "auxiliary_metric_name": snapshots[r["service_id"]]["auxiliary_metric_name"]} for r in rows])
    if selected.primary_metric_name.nunique() != 1:
        raise ValueError("composition members disagree on primary metric")
    path = rows[0]["parent_document"]["reservation_ref"]
    plan = read_cohort(path, seed=42, audit_samples=512, minimum=MIN_AUDIT_SAMPLES,
                        task_family=rows[0]["task_family"],
                        selected_training_ids=sorted({sid for row in rows for sid in row["source_ids"]}))
    if reason := supported_composition_reason(selected, cohort_contract_hash=plan.contract_hash):
        raise ValueError(reason)
    labels = read_cohort_labels(path, plan, task_family=rows[0]["task_family"])
    members = [SelectedFedAvgMember(r["service_id"], r["service_id"], r["sample_count"], r) for r in rows]
    return members, plan, labels, selected, snapshots


def preflight(source_db, entries):
    """Validate metadata once per family; never load initial/client tensors."""
    sources, errors = {}, {}
    # Keep full recipe/data membership caches bounded to one family.
    families = {}
    for entry in entries:
        families.setdefault(entry["family_id"], []).append(entry)
    for family_entries in families.values():
        cache, member_cache = {}, {}
        for entry in family_entries:
            try:
                *_, snapshots = _prepare_composition(source_db, entry, metadata_only=True,
                                                     cache=cache, member_cache=member_cache)
                sources.update(snapshots)
            except Exception as error:
                errors[entry["combination_id"]] = _failure(error, "preflight")
        for entry in family_entries:
            for sid in entry["service_ids"]:
                if sid not in sources:
                    row = member_cache.get(sid)
                    try:
                        sources[sid] = _snapshot(row) if isinstance(row, dict) else {"service_id": sid}
                    except Exception:
                        sources[sid] = {"service_id": sid}
    return sources, errors


def _failure(error, stage, *, wall_s=None, members=()):
    return {"status": "failed", "failure_stage": getattr(error, "stage", stage),
            "failure_service_id": getattr(error, "service_id", None),
            "error_message": bounded_text(str(error)), "wall_s": wall_s, "members": list(members)}


def execute_job(source_db, entry, expected_sources, *, device, on_progress=None):
    started, stage, active_service = time.perf_counter(), "preflight", None
    observations = []

    def progress(event):
        nonlocal stage, active_service
        stage, active_service = event["stage"], event.get("service_id")
        if stage == "member_completed":
            observations.append(event["evidence"])
        if on_progress:
            on_progress(event)

    try:
        members, plan, labels, selected, snapshots = _prepare_composition(source_db, entry, cache={})
        if {sid: sha256_json(value) for sid, value in snapshots.items()} != expected_sources:
            raise ValueError("source service evidence changed after batch preflight")
        execution = execute_composition(source_db, members, plan, device=device, replay=True, on_progress=progress)
        stage = "evaluation"
        if execution.task_family != members[0].row["task_family"] or execution.label_hash != plan.label_hash:
            raise ValueError("global model outputs disagree with the frozen audit contract")
        actual = measure_model_outputs(execution.outputs, plan, labels, task_family=execution.task_family,
                                       inference_latency_s=execution.inference_latency_s)
        actual = measured_recipe_quality(actual, execution.outputs, labels, selected)
        provenance = dict(execution.aggregation_provenance)
        metrics = {**actual, "inference_latency_s": execution.inference_latency_s,
                   "inference_latency_s_per_example": execution.inference_latency_s / len(plan.audit_ids),
                   "audit_examples": len(plan.audit_ids), "aggregation_wall_s": execution.aggregation_wall_s,
                   "training_wall_s": provenance["reconstruction_wall_s"],
                   "evaluation_wall_s": provenance["evaluation_wall_s"],
                   "total_wall_s": time.perf_counter() - started,
                   "local_example_exposures": provenance["local_example_exposures"],
                   "unique_training_examples": provenance["unique_training_examples"]}
        return {"status": "completed", "wall_s": metrics["total_wall_s"], "metrics": metrics,
                "provenance": provenance, "members": provenance["materialization"]}
    except Exception as error:
        result = _failure(error, stage, wall_s=time.perf_counter() - started, members=observations)
        if stage == "training" and not result["failure_service_id"]:
            result["failure_service_id"] = active_service
        from mlaas_data_generator.runtime_compat import is_cuda_poison_error
        result["replace_worker"] = is_cuda_poison_error(error)
        return result
    finally:
        gc.collect()
        from mlaas_data_generator.cli.run_manifest import _clear_cuda_runtime_state
        _clear_cuda_runtime_state()


def _gpu_slots(workers):
    from mlaas_data_generator.cli.run_manifest import _detect_cuda_device_ids
    slots = _detect_cuda_device_ids()
    if not slots:
        raise ValueError("No GPU is available; use --device cpu for CPU execution")
    return slots if workers is None else slots[:workers]


def _gpu_environment(slot):
    """Resolve logical slots through inherited masks before starting Python."""
    import torch
    rocm = bool(getattr(torch.version, "hip", None))
    mask = os.environ.get("HIP_VISIBLE_DEVICES") if rocm else None
    if mask is None:
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [token.strip() for token in mask.split(",")] if mask is not None else None
    if tokens is not None and (slot >= len(tokens) or not tokens[slot]):
        raise ValueError("GPU visibility changed during dispatch")
    token = tokens[slot] if tokens is not None else str(slot)
    result = {"CUDA_VISIBLE_DEVICES": token}
    if rocm:
        result["HIP_VISIBLE_DEVICES"] = token
    return result


def _worker_loop(worker_id, jobs, events, source_db, artifact_prefixes=()):
    from mlaas_data_generator.data.runtime_cache import preparation_cache_scope
    with artifact_path_scope(artifact_prefixes), preparation_cache_scope():
        while (job := jobs.get()) is not None:
            entry, expected = job
            cid = entry["combination_id"]
            def progress(event):
                events.put((worker_id, cid, "progress", event))
            result = execute_job(source_db, entry, expected, device="cuda:0", on_progress=progress)
            events.put((worker_id, cid, "result", result))
            if result.get("replace_worker"):
                return


def _start_worker(context, slot, events, source_db):
    worker_id = f"gpu{slot}_{uuid.uuid4().hex}"
    jobs = context.Queue()
    process = context.Process(target=_worker_loop,
        args=(worker_id, jobs, events, source_db, current_artifact_prefixes()))
    # Spawn imports the package before an initializer runs. Pin the inherited
    # environment at process creation, then restore the coordinator immediately.
    updates = _gpu_environment(slot)
    previous = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        process.start()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {"id": worker_id, "slot": slot, "process": process, "jobs": jobs, "entry": None}


def _stop_worker(worker):
    process = worker["process"]
    if process.is_alive():
        worker["jobs"].put(None)
        process.join(timeout=2)
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)
    worker["jobs"].close()
    worker["jobs"].cancel_join_thread()


def _run_gpu_jobs(entries, sources, source_db, slots, begin, progress, finish):
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    pending, workers = deque(entries), []

    def dispatch(worker):
        if pending:
            entry = pending.popleft()
            worker["entry"] = entry
            worker["started"] = time.perf_counter()
            worker["active_service"] = None
            begin(entry, f"gpu{worker['slot']}")
            worker["jobs"].put((entry, {sid: sha256_json(sources[sid]) for sid in entry["service_ids"]}))

    try:
        for slot in slots[:len(entries)]:
            worker = _start_worker(context, slot, events, source_db)
            workers.append(worker)
            dispatch(worker)
        while any(w["entry"] is not None for w in workers):
            try:
                wid, cid, kind, payload = events.get(timeout=0.2)
            except queue.Empty:
                pass
            else:
                worker = next((w for w in workers if w["id"] == wid), None)
                if worker and worker["entry"] and worker["entry"]["combination_id"] == cid:
                    if kind == "progress":
                        worker["active_service"] = payload.get("service_id") if payload["stage"] == "training" else None
                        progress(worker["entry"], f"gpu{worker['slot']}", payload)
                    else:
                        finish(worker["entry"], payload)
                        worker["entry"] = None
                        if payload.get("replace_worker"):
                            _stop_worker(worker)
                            if pending:
                                worker.update(_start_worker(context, worker["slot"], events, source_db))
                        dispatch(worker)
            for worker in workers:
                if worker["entry"] is not None and worker["process"].exitcode is not None:
                    error = RuntimeError(f"GPU worker exited with code {worker['process'].exitcode}")
                    error.service_id = worker.get("active_service")
                    finish(worker["entry"], _failure(error, "worker", wall_s=time.perf_counter() - worker["started"]))
                    worker["entry"] = None
                    _stop_worker(worker)
                    if pending:
                        worker.update(_start_worker(context, worker["slot"], events, source_db))
                        dispatch(worker)
    finally:
        for worker in workers:
            _stop_worker(worker)
        events.close()
        events.cancel_join_thread()


def run_composition_manifest(file, source_db, db, *, device="auto", workers=None, dry_run=False, path_map=None):
    """Apply catalogue relocation for preflight and every CPU/GPU replay job."""
    prefixes = load_artifact_path_map(source_db, path_map)
    with artifact_path_scope(prefixes):
        return _run_composition_manifest(file, source_db, db, device=device, workers=workers, dry_run=dry_run)


def _run_composition_manifest(file, source_db, db, *, device="auto", workers=None, dry_run=False):
    """Return counts; completed results resume only for identical frozen inputs."""
    manifest, source_db, db = (Path(p).resolve() for p in (file, source_db, db))
    if device not in {"auto", "cpu"} or (workers is not None and
            (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1)):
        raise ValueError("device must be auto or cpu; workers must be positive")
    for source in (source_db, manifest):
        if db == source or (db.exists() and source.exists() and db.samefile(source)):
            raise ValueError("Results database must not replace a source input")
    # mode=ro prevents a missing input from silently becoming an empty catalogue.
    with closing(sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("SELECT service_id FROM services LIMIT 1").fetchall()
    entries = load_composition_manifest(manifest)
    print(f"Preflighting {len(entries)} compositions (metadata only)", flush=True)
    sources, errors = preflight(source_db, entries)
    fingerprint = sha256_json({"version": EXECUTOR_VERSION, "entries": entries, "sources": sources,
                               "preflight_errors": errors})
    run_id = "batch_" + fingerprint
    if dry_run:
        for cid, result in errors.items():
            print(f"{cid}: {result['error_message']}", flush=True)
        return {"run_id": run_id, "dry_run": True, "ready": len(entries) - len(errors), "failed": len(errors)}
    db.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(db) + ".lock", timeout=0):
        writer = CompositionWriter(str(db))
        writer.start()
        try:
            writer.register(run_id, fingerprint=fingerprint, manifest=manifest, source_db=source_db,
                            version=EXECUTOR_VERSION, settings={"device": device, "workers": workers},
                            entries=entries, sources=sources)
            completed = writer.completed(run_id)
            pending = [entry for entry in entries if entry["combination_id"] not in completed]

            def begin(entry, worker):
                writer.begin_composition(run_id, entry["combination_id"], worker)
                print(f"[{worker}] {entry['combination_id']}: starting {entry['composition_size']} members", flush=True)

            def progress(entry, worker, event):
                writer.progress(run_id, entry["combination_id"], event)
                print(f"[{worker}] {entry['combination_id']}: {event['stage']} {event.get('service_id', '')}", flush=True)

            def finish(entry, result):
                writer.finish_composition(run_id, entry["combination_id"], result)
                print(f"{entry['combination_id']}: {result['status']} {result.get('error_message', '')}", flush=True)

            runnable = []
            for entry in pending:
                if entry["combination_id"] in errors:
                    begin(entry, "preflight")
                    finish(entry, errors[entry["combination_id"]])
                else:
                    runnable.append(entry)
            if runnable and device == "auto":
                _run_gpu_jobs(runnable, sources, str(source_db), _gpu_slots(workers), begin, progress, finish)
            elif runnable:
                from mlaas_data_generator.data.runtime_cache import preparation_cache_scope
                with preparation_cache_scope():
                    for entry in runnable:
                        begin(entry, "cpu")
                        result = execute_job(str(source_db), entry,
                            {sid: sha256_json(sources[sid]) for sid in entry["service_ids"]}, device="cpu",
                            on_progress=lambda event: progress(entry, "cpu", event))
                        finish(entry, result)
            return {**writer.summary(run_id), "skipped": len(completed)}
        finally:
            writer.finish()
