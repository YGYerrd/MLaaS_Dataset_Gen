"""Read-only, evidence-preserving export. Contract: docs/flaas-export.md."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path

from mlaas_data_generator.artifact_paths import (
    artifact_path_scope, load_artifact_path_map, resolve_artifact_path,
)

VERSION = "flaas_flat_v1"
REQUIRED_COLUMNS = (
    "client_id", "dataset", "task_type", "model_id", "data_volume", "class_distribution",
    "split_strategy", "cpu_cores", "ram_gb", "client_availability", "local_epochs",
    "batch_size", "learning_rate", "local_training_time_s", "communication_time_s",
    "communication_round", "participation_history", "training_completion_status",
    "combination_id", "global_model_accuracy", "round_deadline_s", "upload_time_s",
    "download_time_s", "model_size_mb", "upload_throughput_mbps", "round_completion_time_s",
    "selected_client_ids", "squared_sample_loss_sum", "last_selected_round",
    "global_accuracy_per_round", "observed_training_duration_s", "local_loss_current_model",
    "candidate_pool_size", "local_loss_l2_squared", "emd_score", "cumulative_model_strength",
    "training_deadline", "deadline_violation",
)
SUPPORT_COLUMNS = (
    "schema_version", "run_id", "family_id", "composition_size", "member_ordinal",
    "aggregation_weight", "composition_status", "started_at", "finished_at",
    "client_recipe_hash", "family_document_hash", "initial_state_hash", "materialized_state_hash",
    "completed_epochs", "micro_batch_size", "gradient_accumulation_steps",
    "hardware_scope", "cpu_cores_basis", "local_training_time_basis",
    "observed_training_duration_basis", "model_size_basis", "reconstruction_duration_s",
    "execution_wall_s", "aggregation_wall_s", "audit_examples", "client_loss_statistics_version",
    "loss_population", "local_loss_current_model_scope", "squared_sample_loss_scope",
    "loss_definition", "missing_attributes", "missing_attribute_reasons",
)
COLUMNS = REQUIRED_COLUMNS + SUPPORT_COLUMNS
UNAVAILABLE = dict.fromkeys((
    "client_availability", "communication_time_s", "upload_time_s", "download_time_s",
    "upload_throughput_mbps", "round_deadline_s", "training_deadline", "deadline_violation",
    "round_completion_time_s", "candidate_pool_size",
), "not_recorded") | dict.fromkeys((
    "communication_round", "participation_history", "last_selected_round",
    "global_accuracy_per_round", "cumulative_model_strength",
), "no_round_history") | {"emd_score": "definition_deferred"}
IDENTITY = ("family_id", "client_recipe_hash", "family_document_hash", "initial_state_hash")
LOSS_FIELDS = ("local_loss_current_model", "squared_sample_loss_sum", "local_loss_l2_squared")
DTYPE_BYTES = {"bool": 1, "uint8": 1, "int8": 1, "int16": 2, "uint16": 2,
               "int32": 4, "uint32": 4, "int64": 8, "uint64": 8, "float16": 2,
               "bfloat16": 2, "float32": 4, "float64": 8, "complex64": 8, "complex128": 16}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(value, label):
    try:
        return _object(json.loads(value, parse_constant=_reject_constant, object_pairs_hook=_pairs), label)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}: {error}") from error


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {label}")
    return value


def _number(value, label, *, integer=False, positive=False, optional=True):
    if value is None and optional:
        return None
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)
            or (integer and not isinstance(value, int))):
        raise ValueError(f"Invalid numeric {label}: {value!r}")
    return value


def _same(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Identity/count mismatch: {label}")


def _close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Numeric mismatch: {label}")


def _document(reference, digest):
    if not reference:
        return None
    path = resolve_artifact_path(reference)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return _parse(data, str(path))


def model_size_mb(schema):
    """Dense full-state payload, including buffers; no tensor loading or dtype assumption."""
    entries = _object(schema, "parameter schema").get("entries")
    if entries is None or entries == []:
        return None
    if not isinstance(entries, list):
        raise ValueError("Invalid parameter schema entries")
    total, names = 0, set()
    for entry in entries:
        _object(entry, "schema entry")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Missing/duplicate tensor name")
        names.add(name)
        numel = _number(entry.get("numel"), "tensor numel", integer=True, optional=False)
        shape = entry.get("shape")
        if not isinstance(shape, list):
            raise ValueError("Missing tensor shape")
        for dim in shape:
            _number(dim, "tensor dimension", integer=True, optional=False)
        _same(numel, math.prod(shape), "tensor shape/numel")
        dtype = str(entry.get("dtype", "")).removeprefix("torch.")
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported tensor dtype: {dtype}")
        total += numel * DTYPE_BYTES[dtype]
    return total / 1_000_000


class _Enrichment:
    def __init__(self, source):
        self.source = source
        self.clients = {}
        self.families = {}

    def get(self, snapshot):
        sid = snapshot["service_id"]
        key = (sid,) + tuple(snapshot.get(k) for k in IDENTITY)
        if key in self.clients:
            return self.clients[key]
        row, reasons = {}, {}
        source = self.source.execute("""SELECT g.family_id,g.client_recipe_hash,g.family_document_hash,
            g.initial_state_hash,json_extract(s.metadata_json,'$.hardware_snapshot.cpu.physical_cores') AS cores,
            json_extract(s.metadata_json,'$.hardware_snapshot.memory.total_mb') AS memory_mib
            FROM services s JOIN service_generation_provenance g USING(service_id) WHERE s.service_id=?""",
            (sid,)).fetchall()
        if len(source) > 1:
            raise ValueError(f"Duplicate source service: {sid}")
        if source:
            source = source[0]
            for field in IDENTITY:
                _same(source[field], snapshot.get(field), f"{sid}: catalogue {field}")
            row["cpu_cores"] = _number(source["cores"], "physical cores", integer=True, positive=True)
            memory = _number(source["memory_mib"], "host memory", positive=True)
            row["ram_gb"] = None if memory is None else memory * 1024 ** 2 / 1_000_000_000
        else:
            reasons.update(dict.fromkeys(("cpu_cores", "ram_gb"), "source_service_absent"))

        recipe = _document(snapshot.get("client_recipe_ref"), snapshot.get("client_recipe_hash"))
        if recipe is None:
            reasons.update(dict.fromkeys(("class_distribution", "split_strategy", "learning_rate",
                                          "local_training_time_s"), "artifact_missing"))
        else:
            digest = hashlib.sha256(_json({k: v for k, v in recipe.items() if k != "recipe_hash"}).encode()).hexdigest()
            _same(recipe.get("recipe_hash"), digest, "recipe self hash")
            _same(recipe.get("service_id"), sid, "recipe service")
            family = _object(recipe.get("family"), "recipe family")
            for field in ("family_id", "initial_state_hash"):
                _same(family.get(field), snapshot.get(field), f"recipe {field}")
            _same(family.get("document_hash"), snapshot.get("family_document_hash"), "recipe family hash")
            selection = _object(recipe.get("selection"), "recipe selection")
            row["_sample_count"] = _number(selection.get("realised_count"), "recipe sample count",
                                            integer=True, positive=True, optional=False)
            ids = selection.get("ordered_source_row_ids")
            if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError("Invalid recipe selected row IDs")
            _same(len(ids), row["_sample_count"], "recipe selected rows")
            obs = _object(recipe.get("observed_metrics", {}), "recipe observed metrics")

            def metric(name):
                item = obs.get(name)
                return _object(item, f"recipe metric {name}").get("value") if item is not None else None

            distribution = metric("train_distribution_json")
            if distribution is not None:
                _object(distribution, "class distribution")
                for count in distribution.values():
                    _number(count, "class count", integer=True, optional=False)
                _same(sum(distribution.values()), row["_sample_count"], "class count total")
                row["class_distribution"] = _json(distribution)
            strategy = next((metric(k) for k in ("split_strategy_effective", "split_strategy", "data_distribution")
                             if metric(k) is not None), None)
            if strategy is not None and (not isinstance(strategy, str) or not strategy):
                raise ValueError("Invalid split strategy")
            row["split_strategy"] = strategy
            training = _object(recipe.get("training", {}), "recipe training")
            effective = _object(training.get("effective_settings", {}), "recipe effective settings")
            row["learning_rate"] = _number(effective.get("learning_rate"), "learning rate")
            row["local_training_time_s"] = _number(metric("train_time_s"), "historical training time")

        fkey = (snapshot.get("family_document_ref"), snapshot.get("family_document_hash"))
        if fkey not in self.families:
            doc = _document(*fkey)
            if doc is None:
                self.families[fkey] = None
            else:
                contracts = _object(doc.get("contracts", {}), "family contracts")
                self.families[fkey] = (doc.get("family_id"), doc.get("initial_state_hash"),
                                      model_size_mb(contracts.get("parameter_schema", {})))
        family = self.families[fkey]
        if family is None:
            reasons["model_size_mb"] = "artifact_missing"
        else:
            _same(family[:2], (snapshot.get("family_id"), snapshot.get("initial_state_hash")), "family document")
            row["model_size_mb"] = family[2]
        self.clients[key] = row, reasons
        return row, reasons


def _losses(evidence, sample_count, snapshot):
    values = {key: _number(evidence.get(key), key) for key in LOSS_FIELDS}
    stats = evidence.get("client_loss_statistics")
    if stats is None and all(v is None for v in values.values()):
        return values
    _object(stats, "client loss statistics")
    if stats.get("version") != "client_loss_statistics_v1" or any(v is None for v in values.values()):
        raise ValueError("Incomplete or unsupported client loss telemetry")
    _same(stats.get("population"), "selected_local_training_rows", "loss population")
    initial = _object(stats.get("initial_model"), "initial loss statistics")
    final = _object(stats.get("final_local_model"), "final loss statistics")
    for item in (initial, final):
        _same(item.get("sample_count"), sample_count, "loss sample count")
        for key in ("loss_sum", "mean_loss", "squared_sample_loss_sum"):
            _number(item.get(key), key, optional=False)
        _close(item["mean_loss"], item["loss_sum"] / sample_count, "loss mean")
    _same(initial.get("model_state_hash"), snapshot.get("initial_state_hash"), "initial loss model")
    _same(final.get("model_state_hash"), evidence.get("materialized_state_hash"), "final loss model")
    _same(initial.get("loss_definition"), final.get("loss_definition"), "loss definition")
    _close(values["local_loss_current_model"], initial["mean_loss"], "initial mean loss")
    for key in LOSS_FIELDS[1:]:
        _close(values[key], final["squared_sample_loss_sum"], key)
    values.update(client_loss_statistics_version=stats["version"], loss_population=stats["population"],
                  local_loss_current_model_scope="initial_model_on_local_training_rows",
                  squared_sample_loss_scope="final_local_model_on_local_training_rows",
                  loss_definition=initial.get("loss_definition"))
    return values


def _rows(conn, enrichment):
    seen = set()
    for comp in conn.execute("SELECT * FROM compositions WHERE status='completed' ORDER BY run_id,combination_id"):
        key = comp["run_id"], comp["combination_id"]
        members = conn.execute("""SELECT m.*,ss.family_id AS source_family,ss.snapshot_json
            FROM composition_members m LEFT JOIN source_services ss USING(run_id,service_id)
            WHERE m.run_id=? AND m.combination_id=? ORDER BY m.ordinal,m.service_id""", key).fetchall()
        _same(len(members), comp["composition_size"], f"{key}: membership size")
        _number(comp["composition_size"], "composition size", integer=True, positive=True, optional=False)
        _same([m["ordinal"] for m in members], list(range(len(members))), "member ordinals")
        ids = [m["service_id"] for m in members]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate composition member: {key}")
        weights = [_number(m["aggregation_weight"], "aggregation weight", optional=False) for m in members]
        _close(math.fsum(weights), 1, "aggregation weights")
        provenance = _parse(comp["aggregation_provenance_json"], "aggregation provenance")
        _same(provenance.get("selected_service_order"), ids, "selected client order")
        # Pivot separately before joining to member rows; metrics cannot multiply membership.
        metrics = {}
        for metric in conn.execute("""SELECT * FROM composition_metrics WHERE run_id=? AND combination_id=?
                                   AND metric_name IN ('accuracy','total_wall_s','aggregation_wall_s','audit_examples')""", key):
            name = metric["metric_name"]
            if name in metrics:
                raise ValueError(f"Duplicate composition metric: {name}")
            populated = [metric[k] for k in ("value_num", "value_int", "value_bool", "value_text", "value_json")
                         if metric[k] is not None]
            if len(populated) != 1 or (metric["value_num"] is None and metric["value_int"] is None):
                raise ValueError(f"Invalid numeric composition metric: {name}")
            metrics[name] = _number(populated[0], name)
        accuracy = _number(metrics.get("accuracy"), "global accuracy", optional=False)
        if accuracy > 1:
            raise ValueError("Global accuracy outside [0,1]")
        for member in members:
            row_key = key + (member["service_id"],)
            if row_key in seen:
                raise ValueError(f"Duplicate export key: {row_key}")
            seen.add(row_key)
            snapshot = _parse(member["snapshot_json"], f"{row_key}: source snapshot")
            _same(snapshot.get("service_id"), member["service_id"], "snapshot service")
            for family in (snapshot.get("family_id"), member["source_family"]):
                _same(family, comp["family_id"], "snapshot family")
            for field in IDENTITY:
                if not isinstance(snapshot.get(field), str) or not snapshot[field]:
                    raise ValueError(f"Missing source identity: {field}")
            count = _number(member["sample_count"], "member samples", integer=True, positive=True, optional=False)
            _same(count, snapshot.get("sample_count"), "snapshot sample count")
            evidence = _parse(member["evidence_json"], f"{row_key}: member evidence")
            _same(evidence.get("service_id"), member["service_id"], "evidence service")
            for field in IDENTITY:
                _same(evidence.get(field), snapshot[field], f"evidence {field}")
            _same(member["status"], "completed", "completed composition member status")
            _same(member["state_hash"], evidence.get("materialized_state_hash"), "materialized state")
            enrichment_row, missing = enrichment.get(snapshot)
            if "_sample_count" in enrichment_row:
                _same(count, enrichment_row["_sample_count"], "recipe/member samples")
            training = _object(evidence.get("training", {}), "member training")
            row = dict.fromkeys(COLUMNS)
            row.update({k: v for k, v in enrichment_row.items() if not k.startswith("_")})
            row.update(schema_version=VERSION, run_id=key[0], combination_id=key[1],
                       client_id=member["service_id"], family_id=comp["family_id"],
                       composition_size=comp["composition_size"], member_ordinal=member["ordinal"],
                       aggregation_weight=member["aggregation_weight"], composition_status=comp["status"],
                       started_at=comp["started_at"], finished_at=comp["finished_at"],
                       data_volume=count, dataset=snapshot.get("dataset_name"), task_type=snapshot.get("task_type"),
                       model_id=snapshot.get("model_id"), training_completion_status=member["status"],
                       global_model_accuracy=accuracy, selected_client_ids=_json(ids),
                       materialized_state_hash=member["state_hash"],
                       reconstruction_duration_s=_number(member["training_wall_s"], "reconstruction duration"),
                       execution_wall_s=metrics.get("total_wall_s"), aggregation_wall_s=metrics.get("aggregation_wall_s"),
                       audit_examples=_number(metrics.get("audit_examples"), "audit examples", integer=True, positive=True))
            row.update({field: snapshot[field] for field in IDENTITY if field in COLUMNS})
            for target, field, integer in (
                ("local_epochs", "training_epochs", True), ("batch_size", "logical_batch_size", True),
                ("completed_epochs", "completed_training_epochs", False),
                ("micro_batch_size", "micro_batch_size_effective", True),
                ("gradient_accumulation_steps", "gradient_accumulation_steps_effective", True),
                ("observed_training_duration_s", "train_time_s", False),
            ):
                row[target] = _number(training.get(field), target, integer=integer,
                                      positive=target in {"local_epochs", "batch_size", "micro_batch_size", "gradient_accumulation_steps"})
            row.update(_losses(evidence, count, snapshot))
            for target, value, present in (
                ("hardware_scope", "original_generation_host", row["cpu_cores"] is not None or row["ram_gb"] is not None),
                ("cpu_cores_basis", "physical", row["cpu_cores"] is not None),
                ("local_training_time_basis", "historical_catalogue_train_loop", row["local_training_time_s"] is not None),
                ("observed_training_duration_basis", "composition_train_loop", row["observed_training_duration_s"] is not None),
                ("model_size_basis", "uncompressed_full_state_schema", row["model_size_mb"] is not None),
            ):
                if present:
                    row[target] = value
            reasons = {}
            for field in REQUIRED_COLUMNS:
                if row[field] is None:
                    reasons[field] = UNAVAILABLE.get(field, missing.get(field,
                        "loss_telemetry_absent" if field in LOSS_FIELDS else "not_recorded"))
            row["missing_attributes"] = ";".join(reasons)
            row["missing_attribute_reasons"] = ";".join(f"{field}={reason}" for field, reason in reasons.items())
            yield row


def export_flaas(db, source_db, output, *, overwrite=False, path_map=None):
    """Publish one validated CSV atomically; never mutate databases or artifacts."""
    db, source_db, output = (Path(p).resolve() for p in (db, source_db, output))
    if output in (db, source_db) or output.suffix.lower() != ".csv":
        raise ValueError("Output must be a .csv separate from the source databases")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Export exists; use --overwrite: {output}")
    prefixes = load_artifact_path_map(source_db, path_map)
    temporary = None
    with ExitStack() as stack:
        connections = []
        for path in (db, source_db):
            conn = stack.enter_context(closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("BEGIN")
            connections.append(conn)
        conn, source = connections
        # Round studies have availability-dependent membership and a distinct
        # per-client/per-round key; the one-shot export cannot represent them.
        from .round_manifest import VERSION as ROUND_VERSION
        if conn.execute("SELECT 1 FROM composition_runs WHERE executor_version=? LIMIT 1", (ROUND_VERSION,)).fetchone():
            raise ValueError("This database contains round experiments; use export-rounds for their per-round measurements")
        expected = conn.execute("""SELECT count(*) FROM composition_members m JOIN compositions c
            USING(run_id,combination_id) WHERE c.status='completed'""").fetchone()[0]
        executions = conn.execute("SELECT count(*) FROM compositions WHERE status='completed'").fetchone()[0]
        stack.enter_context(artifact_path_scope(prefixes))
        enrichment = _Enrichment(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            clients, combinations, missing = set(), set(), Counter()
            rows = 0
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", delete=False,
                                             dir=output.parent, prefix=f".{output.name}.", suffix=".tmp") as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\r\n")
                writer.writeheader()
                for row in _rows(conn, enrichment):
                    writer.writerow(row)
                    rows += 1
                    clients.add(row["client_id"])
                    combinations.add((row["run_id"], row["combination_id"]))
                    missing.update(field for field in REQUIRED_COLUMNS if row[field] is None)
                stream.flush()
                os.fsync(stream.fileno())
            _same(rows, expected, "export row count")
            _same(len(combinations), executions, "export execution count")
            with temporary.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                _same(reader.fieldnames, list(COLUMNS), "CSV header")
                count = 0
                for row in reader:
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError("CSV round-trip column mismatch")
                    count += 1
                _same(count, rows, "CSV round-trip row count")
            if overwrite:
                os.replace(temporary, output)
            else:
                # Linking publishes a complete file and fails atomically if the destination exists.
                os.link(temporary, output)
            return {"schema_version": VERSION, "output": str(output), "rows": rows,
                    "completed_executions": executions, "distinct_clients": len(clients),
                    "missing_by_attribute": dict(sorted(missing.items()))}
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
