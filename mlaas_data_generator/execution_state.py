"""Persist genuine service outcomes; keep pending manifest work outside SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sqlite3
import traceback
import uuid

import psutil

from .diagnostic_json import bounded_text, diagnostic_json
from .storage.writer import make_writer

FEDAVG_OBSERVATION_STATUSES = frozenset({"not_participating", "not_executed_due_to_prior_round_failure"})


def now():
    return datetime.now(timezone.utc).isoformat()


def execution_owner():
    return {"host": socket.gethostname(), "pid": os.getpid(),
            "created": psutil.Process().create_time()}


def owner_alive(owner):
    if not owner or owner.get("host") != socket.gethostname():
        return True  # No evidence that an unowned/remote historical run died.
    try:
        process = psutil.Process(int(owner["pid"]))
        return process.is_running() and abs(process.create_time() - float(owner["created"])) < 0.01
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, KeyError, TypeError, ValueError):
        return True


def state_record(config, status, reason=None, *, record=None):
    from .config import CONFIG
    sid = str(config.get("service_id") or "")
    if not sid:
        raise ValueError("terminal service status requires a service_id")
    value = dict(record or {})
    value.update(service_id=sid, status=status)
    for key in ("case_name", "dataset", "dataset_name", "dataset_config", "model_type",
                "task_type", "hf_task", "training_regime", "modality"):
        value.setdefault(key, config.get(key))
    value.setdefault("model_id", config.get("hf_model_id") or config.get("model_id"))
    value.setdefault("service_config_json", diagnostic_json(config))
    metadata = value.get("metadata_json") or {}
    metadata = json.loads(metadata) if isinstance(metadata, str) else dict(metadata)
    metadata.update(started_at=config.get("_execution_started_at", metadata.get("started_at", now())),
                    manifest_group_id=config.get("manifest_group_id"),
                    execution_run_id=config.get("_execution_run_id"),
                    execution_owner=config.get("_execution_owner") or execution_owner())
    if status in {"completed", "failed", "interrupted"}:
        metadata["completed_at"] = now()
    if reason:
        metadata["status_reason"] = bounded_text(str(reason))
    value["metadata_json"] = metadata
    return str(config.get("db_path") or CONFIG["db_path"]), value


def _journal(db, record):
    directory = Path(str(db) + ".terminal-status")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (uuid.uuid4().hex + ".json")
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(diagnostic_json(record, max_bytes=256 * 1024))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    return target


def persist_terminal(config, *, status, reason=None, failure_stage="service_execution",
                     exc=None, record=None, metrics=None):
    """Commit status before optional diagnostics; journal if SQLite is unavailable.

    Cancellation is not an outcome and must not overwrite an earlier result.
    A disk failure affecting both stores is raised loudly.
    """
    if status == "interrupted":
        return
    if status not in {"completed", "failed"}:
        raise ValueError("invalid terminal status")
    db, record = state_record(config, status, reason, record=record)
    writer = make_writer("sqlite", db_path=db)
    committed = False
    try:
        writer.start()
        writer.write_service_progress(record)
        # Failed post-training verification must not publish a provider.
        if status != "completed":
            writer.conn.execute("UPDATE service_generation_provenance SET composition_compatible=0 WHERE service_id=?",
                                (record["service_id"],))
        writer.commit()
        committed = True
        if status != "completed":
            writer.write_service_failure(service_id=record["service_id"], row_index=config.get("row_index"),
                case_name=config.get("case_name"), manifest_group_id=config.get("manifest_group_id"),
                failure_stage=failure_stage, error_message=bounded_text(str(reason or "")),
                resolved_config_json=diagnostic_json(config),
                traceback_text=bounded_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))) if exc else None,
                fl_run_id=config.get("fl_run_id"), client_id=config.get("client_id"), round_id=config.get("round_id"))
        if metrics:
            writer.write_service_metrics(record["service_id"], metrics)
        writer.finish()
    except Exception as error:
        writer.abort()
        if committed:
            print(f"Terminal status saved for {record['service_id']}; diagnostic write failed: {bounded_text(str(error))}", flush=True)
        else:
            journal = _journal(db, record)
            print(f"SQLite status write failed for {record['service_id']}; durable status saved to {journal}: {bounded_text(str(error))}", flush=True)


def queue_services(configs):
    """Attach attempt metadata in memory, without publishing queued services."""
    for config in configs:
        config.setdefault("_execution_started_at", now())
        config["_execution_owner"] = execution_owner()


def register_manifest_roster(configs, *, run_id, manifest_path, rerun_existing=False):
    """Keep pending dispatch metadata in atomic sidecars, never catalogue rows."""
    grouped = {}
    for config in configs:
        from .config import CONFIG
        db = str(config.get("db_path") or CONFIG["db_path"])
        snapshot = {key: config.get(key) for key in (
            "service_id", "case_name", "dataset", "dataset_name", "dataset_config", "model_type",
            "hf_model_id", "hf_task", "task_type", "training_regime", "manifest_group_id", "row_index")}
        snapshot.update(db_path=db, manifest_path=str(manifest_path), started_at=now(), rerun_existing=rerun_existing)
        grouped.setdefault(db, []).append(snapshot)
    for db, rows in grouped.items():
        directory = Path(str(db) + ".manifest-roster")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (uuid.uuid4().hex + ".json")
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump({"run_id": run_id, "owner": execution_owner(), "rows": rows}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)


def reconcile_manifest_roster(db, *, run_id=None, reason="manifest process exited before a terminal result"):
    """Save finished roster outcomes; discard abandoned work without inventing failures."""
    path = Path(db)
    rows = []
    journals = {}
    for journal in Path(str(db) + ".manifest-roster").glob("*.json"):
        document = json.loads(journal.read_text(encoding="utf-8"))
        member_run = document["run_id"]
        owner = document["owner"]
        if (member_run != run_id if run_id else owner_alive(owner)):
            continue
        journals[journal] = document
        rows.extend((member_run, row["service_id"], diagnostic_json(owner), diagnostic_json(row))
                    for row in document["rows"])
    if path.is_file():
        with sqlite3.connect(path) as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='manifest_service_runs'").fetchone():
                rows.extend(conn.execute("SELECT run_id,service_id,owner_json,config_json FROM manifest_service_runs WHERE status='queued'").fetchall())
    suffix = "".join(path.suffixes)
    stem = path.name[:-len(suffix)] if suffix else path.name
    candidates = [path, *sorted(path.parent.glob(f"{stem}.gpu*{suffix}"))]
    changed = 0
    deferred_runs = set()
    for member_run, sid, owner_json, config_json in rows:
        if (member_run != run_id if run_id else owner_alive(json.loads(owner_json))):
            continue
        cfg = json.loads(config_json)
        status, destination, live, previous_completed = None, path, False, False
        for candidate in candidates:
            if not candidate.is_file():
                continue
            with sqlite3.connect(candidate) as conn:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(services)")}
                if not {"service_id", "status"}.issubset(cols):
                    continue
                row = conn.execute("SELECT status," + ("metadata_json" if "metadata_json" in cols else "NULL") +
                                   " FROM services WHERE service_id=?", (sid,)).fetchone()
            if not row:
                continue
            meta = json.loads(row[1] or "{}")
            if row[0] in {"completed", "failed", "interrupted"} | FEDAVG_OBSERVATION_STATUSES:
                if meta.get("execution_run_id") == member_run:
                    status = row[0]
                    break
                previous_completed |= row[0] == "completed"
                continue
            if meta.get("execution_run_id") != member_run:
                continue
            destination = candidate
            if not run_id and owner_alive(meta.get("execution_owner")):
                live = True
        if status is None and previous_completed and not cfg.get("rerun_existing"):
            status = "completed"
        if live and status is None:
            deferred_runs.add(member_run)
            continue
        terminal = status in {"completed", "failed"} | FEDAVG_OBSERVATION_STATUSES
        if terminal or path.is_file():
            with sqlite3.connect(path, timeout=30) as conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS manifest_service_runs (
                    run_id TEXT NOT NULL, service_id TEXT NOT NULL, status TEXT NOT NULL,
                    owner_json TEXT NOT NULL, config_json TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(run_id,service_id))""")
                if terminal:
                    conn.execute("INSERT OR REPLACE INTO manifest_service_runs VALUES (?,?,?,?,?,?)",
                                 (member_run, sid, status, owner_json, config_json, now()))
                else:
                    conn.execute("DELETE FROM manifest_service_runs WHERE run_id=? AND service_id=? AND status IN ('queued','running','interrupted')",
                                 (member_run, sid))
        changed += 1
    for journal, document in journals.items():
        # A live worker may still own an attempt even after its manifest exits.
        # Keep its sidecar until the next reconciliation.
        if document["run_id"] not in deferred_runs:
            journal.unlink()
    return changed


def recover_service_states(db, *, run_id=None, reason="previous execution process exited"):
    """Recover journals and owned pending rows without changing live/remote runs."""
    path = Path(db)
    directory = Path(str(db) + ".terminal-status")
    journals = list(directory.glob("*.json")) if directory.is_dir() else []
    if not path.exists() and not journals:
        return 0
    if path.exists() and not journals:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(services)")}
            if not {"status", "metadata_json"}.issubset(columns):
                return 0
            pending = conn.execute("SELECT metadata_json FROM services WHERE status IN ('queued','running')").fetchall()
        if not any((json.loads(raw or "{}").get("execution_run_id") == run_id if run_id else
                    not owner_alive(json.loads(raw or "{}").get("execution_owner"))) for (raw,) in pending):
            return 0
    writer = make_writer("sqlite", db_path=str(db))
    recovered = 0
    try:
        writer.start()
        for journal in journals:
            record = json.loads(journal.read_text(encoding="utf-8"))
            if record["status"] not in {"completed", "failed"}:
                journal.unlink()
                continue
            old = writer.conn.execute("SELECT status,metadata_json FROM services WHERE service_id=?", (record["service_id"],)).fetchone()
            previous = json.loads(old[1] or "{}") if old else {}
            incoming = record["metadata_json"]
            incoming = json.loads(incoming) if isinstance(incoming, str) else incoming
            same_attempt = (previous.get("started_at") == incoming.get("started_at") and
                            previous.get("execution_owner") == incoming.get("execution_owner"))
            if not old or same_attempt or (old[0] in {"queued", "running"} and
                                           previous.get("started_at", "") <= incoming.get("started_at", "")):
                writer.write_service_progress(record)
                if record["status"] != "completed":
                    writer.conn.execute("UPDATE service_generation_provenance SET composition_compatible=0 WHERE service_id=?", (record["service_id"],))
                recovered += 1
            writer.commit()
            journal.unlink()
        for sid, raw in writer.conn.execute("SELECT service_id,metadata_json FROM services WHERE status IN ('running','queued')").fetchall():
            meta = json.loads(raw or "{}")
            abandoned = (meta.get("execution_run_id") == run_id if run_id else not owner_alive(meta.get("execution_owner")))
            if not abandoned:
                continue
            writer.conn.execute("DELETE FROM services WHERE service_id=? AND status IN ('queued','running')", (sid,))
            recovered += 1
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    return recovered
