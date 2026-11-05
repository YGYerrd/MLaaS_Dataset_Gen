"""Prepare the audited Lambda classification cohort without training models.

Source SQLite files and signed artifact documents remain unchanged. Relocation
is an explicit manifest, not a rewrite of hash-bound recipe/family contents.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlaas_data_generator.artifact_audit import ArtifactAudit, validate, canonical_hash

SOURCE_NAMES = ("validation.db", "validation.gpu0.db", "validation.gpu1.db")
SERVICE_TABLES = ("services", "service_generation_provenance", "service_metrics",
                  "service_artifacts", "service_split_provenance",
                  "service_execution_provenance", "service_checkpoint_storage")
LINEAGE_TABLES = ("model_lineages", "lineage_checkpoint_storage")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items() if k in fields})


def q(value):
    return '"' + value.replace('"', '""') + '"'


def read_only(path):
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("BEGIN")
    return conn


def file_stamp(path):
    st = path.stat()
    return {"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def selected_occurrences(connections, family_ids):
    """Keep all statuses for selected IDs, including failures without lineage."""
    records = defaultdict(list)
    for name, conn in connections.items():
        for row in conn.execute("SELECT s.service_id,s.status,g.family_id,g.client_recipe_hash,"
                                "g.family_document_hash,g.initial_state_hash,g.final_model_hash "
                                "FROM services s LEFT JOIN service_generation_provenance g USING(service_id)"):
            records[row["service_id"]].append({"source_database": name, **dict(row)})
    return {sid: rows for sid, rows in records.items()
            if any(r["family_id"] in family_ids for r in rows)}


def choose_completed(occurrences):
    """Fail closed on contradictory successful records; failed history is retained."""
    completed = [r for r in occurrences if r["status"] == "completed"]
    if not completed:
        return None
    keys = ("family_id", "client_recipe_hash", "family_document_hash", "initial_state_hash", "final_model_hash")
    identities = {tuple(row.get(k) for k in keys) for row in completed}
    if len(identities) != 1 or any(completed[0].get(k) is None for k in keys):
        raise ValueError("conflicting_or_missing_completed_identity")
    return completed[0]


def service_content(conn, sid):
    result = {}
    for table in SERVICE_TABLES:
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {q(table)} WHERE service_id=?", (sid,))]
        for row in rows:
            row.pop("metric_id", None)
            row.pop("artifact_id", None)
        result[table] = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))
    return canonical_hash(result)


class Files:
    """Content inventory with stability checks; never follows a guessed revision."""
    def __init__(self):
        self.rows = {}

    def check(self, path, role, expected=None):
        path = Path(path).absolute()
        key = str(path)
        stamp = file_stamp(path)
        if key not in self.rows:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            if file_stamp(path) != stamp:
                raise ValueError(f"file changed during verification: {path}")
            self.rows[key] = {"path": key, **stamp, "sha256": digest.hexdigest(), "roles": [role]}
        row = self.rows[key]
        if row["size_bytes"] != stamp["size_bytes"] or row["mtime_ns"] != stamp["mtime_ns"]:
            raise ValueError(f"file changed after verification: {path}")
        if role not in row["roles"]:
            row["roles"].append(role)
        if expected and expected != row["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        return row


def check_model(model_id, revision, task, hub, files):
    from transformers import AutoConfig, AutoTokenizer, AutoImageProcessor
    from safetensors import safe_open
    snapshot = hub / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    if not re.fullmatch(r"[a-f0-9]{40}", revision) or not snapshot.is_dir():
        raise ValueError(f"pinned_model_snapshot_missing: {snapshot}")
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    if task == "sequence_classification":
        processor = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    else:
        processor = AutoImageProcessor.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    required = [p for p in snapshot.iterdir() if p.is_file() and
                (p.suffix in {".json", ".txt", ".model"} and p.name not in {"README.md"})]
    weight_files = []
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = snapshot / filename
        if index.is_file():
            weight_files = [snapshot / n for n in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
            break
    if not weight_files:
        weight_files = [snapshot / n for n in ("model.safetensors", "pytorch_model.bin") if (snapshot / n).is_file()][:1]
    if not weight_files:
        raise ValueError("pinned_model_weights_missing")
    tensor_count = 0
    for path in weight_files:
        files.check(path, "pinned_model_weights", path.resolve().name if path.is_symlink()
                    and re.fullmatch(r"[a-f0-9]{64}", path.resolve().name) else None)
        if path.suffix == ".safetensors":
            with safe_open(str(path), framework="numpy") as weights:
                tensor_count += len(list(weights.keys()))
        else:
            import torch
            weights = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(weights, dict) or not weights:
                raise ValueError("invalid_model_weight_dictionary")
            tensor_count += len(weights)
            del weights
    for path in required:
        files.check(path, "pinned_model_configuration_or_processor")
    return {"model_id": model_id, "revision": revision, "task": task, "snapshot": str(snapshot),
            "status": "verified", "config_class": type(config).__name__,
            "processor_class": type(processor).__name__, "tensor_count": tensor_count,
            "weight_files": [str(p) for p in weight_files]}


def check_dataset(raw, cache, files):
    import pyarrow as pa
    from datasets.fingerprint import Hasher
    from mlaas_data_generator.data.hf_cache_paths import resolve_hf_cache_path
    name, revision, config = raw["dataset_name"], raw["revision"], raw["dataset_config"]
    root = cache / name.replace("/", "___")
    candidates = []
    for path in root.glob(f"*/*/{revision}/dataset_info.json"):
        info = json.loads(path.read_text(encoding="utf-8"))
        if config is None or info.get("config_name") == config:
            candidates.append((path, info))
    if len(candidates) != 1:
        raise ValueError(f"pinned_dataset_cache_missing_or_ambiguous: {name}/{config}@{revision}")
    info_path, info = candidates[0]
    files.check(info_path, "dataset_cache_metadata")
    result = {"dataset_name": name, "dataset_config": config, "revision": revision,
              "cache_directory": str(info_path.parent), "status": "verified", "splits": {}}
    for spec in raw["splits"].values():
        split = spec["name"]
        if info["splits"][split]["num_examples"] != spec["count"]:
            raise ValueError(f"dataset_split_size_mismatch: {split}")
        expected_shards = info["splits"][split].get("shard_lengths")
        pattern = re.compile(r".+-" + re.escape(split) + r"(?:-\d+-of-\d+)?\.arrow$")
        arrows = sorted(p for p in info_path.parent.glob("*.arrow") if pattern.fullmatch(p.name))
        if len(arrows) != (len(expected_shards) if expected_shards else 1):
            raise ValueError(f"dataset_arrow_shards_missing: {split}")
        count = 0
        for arrow in arrows:
            files.check(arrow, "dataset_arrow")
            with pa.memory_map(str(arrow), "r") as source:
                for batch in pa.ipc.open_stream(source):
                    batch.validate(full=True)
                    count += batch.num_rows
                    # Some image caches store external paths rather than bytes.
                    for column in batch.columns:
                        if pa.types.is_struct(column.type) and {"path", "bytes"} <= {f.name for f in column.type}:
                            paths, blobs = column.field("path"), column.field("bytes")
                            for idx in range(len(column)):
                                blob = blobs[idx]
                                if not blob.is_valid or len(blob.as_buffer()) == 0:
                                    image_path = paths[idx].as_py()
                                    if not image_path:
                                        raise ValueError("dataset_image_missing_bytes_and_path")
                                    files.check(resolve_hf_cache_path(image_path), "dataset_external_image")
        if count != spec["count"]:
            raise ValueError(f"dataset_arrow_row_count_mismatch: {split}")
        hasher = Hasher()
        hasher.update(info_path.parent.relative_to(cache).as_posix())
        hasher.update(split)
        fingerprint = hasher.hexdigest()
        if fingerprint != spec["fingerprint"]:
            raise ValueError(f"dataset_split_fingerprint_mismatch: {split}: {fingerprint} != {spec['fingerprint']}")
        if canonical_hash(list(range(count))) != spec["ordered_row_ids_hash"]:
            raise ValueError(f"dataset_row_identity_mismatch: {split}")
        result["splits"][split] = {"rows": count, "fingerprint": fingerprint, "arrow_files": [str(p) for p in arrows]}
    return result


def merge_database(destination, sources, selected):
    """Copy selected records exactly, apart from shard-local integer row IDs."""
    if destination.exists():
        raise FileExistsError(destination)
    out = sqlite3.connect(destination.resolve().as_uri(), uri=True)
    out.execute("PRAGMA foreign_keys=OFF")
    template = sources["validation.gpu0.db"]
    schema = template.execute("SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL "
                              "AND name NOT LIKE 'sqlite_%' ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END").fetchall()
    for kind, name, sql in schema:
        if kind == "table":
            out.execute(sql)
    out.execute("CREATE TABLE preparation_service_sources (service_id TEXT PRIMARY KEY, source_database TEXT NOT NULL)")
    out.execute("CREATE TABLE preparation_source_outcomes (service_id TEXT,source_database TEXT,status TEXT,selected INTEGER)")
    out.execute("CREATE TEMP TABLE selected_ids (service_id TEXT PRIMARY KEY)")
    counts = []
    for index, (name, source) in enumerate(sources.items()):
        filename = source.execute("PRAGMA database_list").fetchone()[2]
        alias = f"src{index}"
        out.execute(f"ATTACH DATABASE ? AS {alias}", (Path(filename).resolve().as_uri() + "?mode=ro",))
        ids = [r["service_id"] for r in selected if r["source_database"] == name]
        out.execute("DELETE FROM selected_ids")
        out.executemany("INSERT INTO selected_ids VALUES (?)", [(sid,) for sid in ids])
        out.executemany("INSERT INTO preparation_service_sources VALUES (?,?)", [(sid, name) for sid in ids])
        for table in SERVICE_TABLES:
            source_sql = source.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
            template_sql = template.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
            if source_sql != template_sql:
                raise ValueError(f"schema mismatch: {name}/{table}")
            columns = [r[1] for r in source.execute(f"PRAGMA table_info({q(table)})")
                       if r[1] not in {"metric_id", "artifact_id"}]
            column_sql = ','.join(q(x) for x in columns)
            selected_sql = ','.join('s.' + q(x) for x in columns)
            where = " WHERE s.service_id IN (SELECT service_id FROM selected_ids)"
            expected = out.execute(f"SELECT COUNT(*) FROM {alias}.{q(table)} s" + where).fetchone()[0]
            out.execute(f"INSERT INTO {q(table)} ({column_sql}) SELECT {selected_sql} FROM {alias}.{q(table)} s" + where)
            counts.append({"source": name, "table": table, "rows": expected})
            print(f"Copied {name}/{table}: {expected:,}", flush=True)
        for table in LINEAGE_TABLES:
            rows = source.execute(f"SELECT * FROM {q(table)} WHERE lineage_id IN "
                                  "(SELECT lineage_id FROM service_generation_provenance WHERE service_id IN (" +
                                  ",".join("?" for _ in ids) + "))", ids) if ids else []
            columns = [r[1] for r in source.execute(f"PRAGMA table_info({q(table)})")]
            for row in rows:
                existing = out.execute(f"SELECT * FROM {q(table)} WHERE lineage_id=?", (row["lineage_id"],)).fetchone()
                if existing:
                    left, right = dict(zip(columns, existing)), dict(row)
                    left.pop("created_at", None); right.pop("created_at", None)
                    if left != right:
                        raise ValueError(f"conflicting lineage record: {table}/{row['lineage_id']}")
                else:
                    out.execute(f"INSERT INTO {q(table)} VALUES (" + ','.join('?' for _ in columns) + ')', tuple(row))
        out.commit()
    for kind, name, sql in schema:
        if kind != "table":
            out.execute(sql)
    for table in SERVICE_TABLES:
        expected = sum(r["rows"] for r in counts if r["table"] == table)
        if out.execute(f"SELECT COUNT(*) FROM {q(table)}").fetchone()[0] != expected:
            raise ValueError(f"merged row count mismatch: {table}")
    out.commit()
    return out, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hub-cache", type=Path, default=Path(os.environ.get("HF_HUB_CACHE", Path.home()/'.cache/huggingface/hub')))
    parser.add_argument("--dataset-cache", type=Path, default=Path(os.environ.get("HF_DATASETS_CACHE", Path.home()/'.cache/huggingface/datasets')))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", ROCM_SDK_TARGET_FAMILY="custom")
    roster = [r for r in csv.DictReader(args.roster.open(encoding="utf-8")) if r["task_family"] == "classification"]
    family_ids = {r["family_id"] for r in roster}
    if len(roster) != 6975 or len(family_ids) != 70:
        raise ValueError("expected the audited 6,975-service, 70-family classification cohort")
    before = {n: file_stamp(args.root/n) for n in SOURCE_NAMES}
    sources = {n: read_only(args.root/n) for n in SOURCE_NAMES}
    occurrences = selected_occurrences(sources, family_ids)
    selected, reconciliation = [], []
    for sid, rows in sorted(occurrences.items()):
        chosen = choose_completed(rows)
        completed = [r for r in rows if r["status"] == "completed"]
        if len(completed) > 1 and len({service_content(sources[r['source_database']], sid) for r in completed}) != 1:
            raise ValueError(f"conflicting completed payload: {sid}")
        if chosen:
            selected.append(chosen)
        reconciliation.append({"service_id": sid, "selected_source": chosen["source_database"] if chosen else None,
                               "outcomes": rows, "resolution": "completed_preferred" if chosen else "no_completed_record"})
    expected = {r["service_id"] for r in roster}
    if {r["service_id"] for r in selected} != expected:
        raise ValueError("current completed membership differs from audited roster; no snapshot published")
    write_csv(args.output_dir/"reconciliation.csv", reconciliation, ["service_id", "selected_source", "resolution", "outcomes"])
    print(f"Reconciled {len(selected):,} completed services in {len(family_ids)} families", flush=True)
    audit = ArtifactAudit(args.root)
    checks, families, runtime_versions = [], {}, Counter()
    for i, record in enumerate(selected):
        sid = record["service_id"]; conn = sources[record["source_database"]]
        service = dict(conn.execute("SELECT * FROM services WHERE service_id=?", (sid,)).fetchone())
        provenance = dict(conn.execute("SELECT * FROM service_generation_provenance WHERE service_id=?", (sid,)).fetchone())
        metrics = {r[0]: r[1] for r in conn.execute("SELECT metric_name,coalesce(value_num,value_int,value_bool,value_text) "
                    "FROM service_metrics WHERE service_id=? AND metric_name IN ('primary_metric_name','accuracy','metric_score','loss','train_loss')", (sid,))}
        storage = dict(conn.execute("SELECT * FROM lineage_checkpoint_storage WHERE lineage_id=?", (provenance["lineage_id"],)).fetchone())
        row = {"service_id": sid, "family_id": provenance["family_id"], "source_database": record["source_database"],
               "recipe_ref": provenance["client_recipe_ref"], "recipe_path": str(audit.local(provenance["client_recipe_ref"])),
               "recipe_sha256": provenance["client_recipe_hash"], "family_ref": provenance["family_document_ref"],
               "family_path": str(audit.local(provenance["family_document_ref"])), "family_sha256": provenance["family_document_hash"]}
        try:
            recipe, family = validate(audit, service, provenance, metrics, storage)
            row.update(status="verified", reason=None, training_examples=recipe["selection"]["realised_count"])
            families[family["family_id"]] = family
            runtime_versions[json.dumps(recipe["runtime"]["versions"], sort_keys=True)] += 1
        except Exception as error:
            row.update(status="blocked", reason=f"{type(error).__name__}: {error}")
        checks.append(row)
        if (i+1) % 500 == 0:
            print(f"Verified recipes: {i+1:,}/{len(selected):,}", flush=True)
    write_csv(args.output_dir/"services.csv", checks, list(checks[0]))
    if any(r["status"] != "verified" for r in checks):
        raise ValueError("service integrity failures; inspect services.csv")

    from mlaas_data_generator.checkpoint_store import load_lineage_base
    files, model_results, dataset_results, pool_rows = Files(), {}, {}, []
    for index, (fid, family) in enumerate(sorted(families.items())):
        print(f"Checking replay dependencies: pool {index+1}/70", flush=True)
        cfg = family["config"]; raw = family["raw_source"]
        model_key = (cfg["hf_model_id"], cfg["hf_model_revision"], cfg["hf_task"])
        data_key = canonical_hash(raw)
        reasons = []
        base_ok = False
        try:
            state = load_lineage_base(audit.local(family["store_root"] + '/'), family["lineage_id"],
                       expected_parameter_schema_hash=family["compatibility"]["parameter_schema_hash"])
            if state.sha256 != family["initial_state_hash"]:
                raise ValueError("decoded base state hash differs")
            base_ok = True
            del state
        except Exception as error:
            reasons.append(f"base: {type(error).__name__}: {error}")
        if model_key not in model_results:
            try:
                model_results[model_key] = check_model(*model_key, args.hub_cache, files)
            except Exception as error:
                model_results[model_key] = {"model_id": model_key[0], "revision": model_key[1], "status": "blocked", "reason": f"{type(error).__name__}: {error}"}
        if data_key not in dataset_results:
            try:
                dataset_results[data_key] = check_dataset(raw, args.dataset_cache, files)
            except Exception as error:
                dataset_results[data_key] = {"dataset_name": raw["dataset_name"], "dataset_config": raw["dataset_config"], "revision": raw["revision"], "status": "blocked", "reason": f"{type(error).__name__}: {error}"}
        for label, item in (("model", model_results[model_key]), ("dataset", dataset_results[data_key])):
            if item["status"] != "verified":
                reasons.append(label + ": " + item["reason"])
        n = sum(r["family_id"] == fid for r in checks)
        pool_rows.append({"family_id": fid, "model_id": model_key[0], "model_revision": model_key[1],
            "dataset_name": raw["dataset_name"], "dataset_config": raw["dataset_config"], "dataset_revision": raw["revision"],
            "services": n, "audit_examples": len(family["reservation"]["audit_ids"]), "base_tensor_state_verified": base_ok,
            "model_dependencies": model_results[model_key]["status"], "raw_data_dependencies": dataset_results[data_key]["status"],
            "dependency_status": "blocked" if reasons else "verified", "blockers": reasons,
            "supported_composition_sizes": [k for k in (2,5,10,15) if n >= k],
            "initial_state_hash": family["initial_state_hash"], "store_root": str(audit.local(family["store_root"]+'/')),
            "full_preprocessing_replay": "not_run", "training_replay": "not_run"})
        write_json(args.output_dir/"dependencies.json", {"models": list(model_results.values()), "datasets": list(dataset_results.values())})
        write_csv(args.output_dir/"pools.csv", pool_rows, list(pool_rows[0]))
        gc.collect()

    # Record every hashed immutable dependency, including per-file directory members.
    for (path, size, mtime, _ctime), digest in audit.files.items():
        files.rows.setdefault(str(path), {"path": str(path), "size_bytes": size, "mtime_ns": mtime,
                                         "sha256": digest, "roles": ["recipe_family_base_or_support"]})
    write_csv(args.output_dir/"dependency_files.csv", list(files.rows.values()), ["path", "size_bytes", "mtime_ns", "sha256", "roles"])
    write_json(args.output_dir/"path_map.json", {"version": "composition_path_map_v1",
        "prefixes": [{"recorded": "/home/joshua/MLaaS_Dataset_Gen/outputs/", "resolved": str(args.root.resolve())}],
        "hf_hub_cache": str(args.hub_cache), "hf_datasets_cache": str(args.dataset_cache),
        "policy": "Resolve paths at read time. Never rewrite hash-bound document contents. Current validation CLI does not automatically consume this map."})

    destination = args.output_dir/"services.db"
    merged, counts = merge_database(destination, sources, selected)
    for sid, rows in occurrences.items():
        for row in rows:
            merged.execute("INSERT INTO preparation_source_outcomes VALUES (?,?,?,?)", (sid, row["source_database"], row["status"],
                int(any(r["service_id"] == sid and r["source_database"] == row["source_database"] for r in selected))))
    merged.execute("CREATE TABLE preparation_artifact_locations (recorded_path TEXT PRIMARY KEY,resolved_path TEXT,sha256 TEXT)")
    prefix = str(args.root.resolve())
    for record in files.rows.values():
        path = Path(record["path"])
        if path.is_relative_to(args.root.resolve()):
            recorded = '/home/joshua/MLaaS_Dataset_Gen/outputs/' + path.relative_to(args.root.resolve()).as_posix()
            merged.execute("INSERT OR IGNORE INTO preparation_artifact_locations VALUES (?,?,?)", (recorded,str(path),record["sha256"]))
    merged.execute("CREATE VIEW prepared_service_locations AS SELECT g.service_id,g.family_id,"
        "r.resolved_path AS recipe_path,f.resolved_path AS family_path FROM service_generation_provenance g "
        "LEFT JOIN preparation_artifact_locations r ON r.recorded_path=g.client_recipe_ref "
        "LEFT JOIN preparation_artifact_locations f ON f.recorded_path=g.family_document_ref")
    merged.commit()
    integrity = [r[0] for r in merged.execute("PRAGMA integrity_check")]
    fk = merged.execute("PRAGMA foreign_key_check").fetchall()
    if integrity != ['ok'] or fk:
        raise ValueError(f"merged database integrity failed: {integrity}/{fk}")
    merged.close()
    for name, conn in sources.items():
        conn.rollback(); conn.close()
        if file_stamp(args.root/name) != before[name]:
            raise ValueError(f"source database changed: {name}")
    installed = {}
    for name in ("torch", "transformers", "datasets", "huggingface_hub", "numpy", "safetensors", "pyarrow", "zstandard", "keras", "tensorflow"):
        try: installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError: installed[name] = None
    summary = {"version": "classification_pool_preparation_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "services": len(selected), "families": len(pool_rows), "selected_by_source": dict(Counter(r["source_database"] for r in selected)),
        "completed_duplicates": sum(sum(x['status']=='completed' for x in rr)>1 for rr in occurrences.values()),
        "selected_ids_with_failed_history": sum(any(x['status']!='completed' for x in occurrences[r['service_id']]) for r in selected),
        "families_dependencies_verified": sum(r["dependency_status"]=='verified' for r in pool_rows),
        "families_blocked": sum(r["dependency_status"]=='blocked' for r in pool_rows),
        "verified_dependency_files": len(files.rows), "unique_model_dependencies": len(model_results),
        "unique_dataset_contracts": len(dataset_results), "database": str(destination.resolve()),
        "database_integrity_check": integrity, "foreign_key_violations": len(fk), "copied_rows": counts,
        "source_databases_unchanged": True, "source_stamps": before,
        "roster_sha256": hashlib.sha256(args.roster.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "installed_runtime": installed, "original_runtimes": [{"versions": json.loads(k), "services": v} for k,v in runtime_versions.items()],
        "limits": ["No training, aggregation or global-model evaluation was performed.",
            "Raw Arrow files are readable and newly SHA-256 inventoried; split metadata fingerprints and row counts are compared with the frozen source contract. These metadata fingerprints are not historical cryptographic content hashes.",
            "Full preprocessing/selected-content hashes must still be checked by production replay before training.",
            "Local Windows/ROCm runtime differs from the recorded Linux/CUDA runtime; cloud runtime is not provisioned or certified.",
            "Database retains original references. Use prepared_service_locations/path_map.json for relocation; the existing validation CLI needs a resolver integration or the original Linux directory layout.",
            "Only selected completed services and their metrics, artifacts, split/execution/generation/checkpoint provenance and lineages are merged. Other experiment/failure tables remain empty; source outcome history is in preparation_source_outcomes and original shards."]}
    write_json(args.output_dir/"summary.json", summary)
    write_json(args.output_dir/"replay_manifest.json", {"version": "classification_replay_dependencies_v1",
        "database": str(destination.resolve()), "path_map": str((args.output_dir/'path_map.json').resolve()),
        "composition_sizes": [2,5,10,15], "pools": pool_rows,
        "service_manifest": str((args.output_dir/'services.csv').resolve())})
    report = ["Classification pool preparation", "", f"Reconciled {len(selected):,} services in {len(pool_rows)} pools.",
        f"Dependency-verified pools: {summary['families_dependencies_verified']}; blocked pools: {summary['families_blocked']}.",
        f"Selected source counts: {summary['selected_by_source']}",
        f"Selected IDs with failed historical records: {summary['selected_ids_with_failed_history']}.",
        "Source databases and signed artifacts were preserved. SQLite integrity and foreign-key checks passed.", "",
        "Files: services.db (merged snapshot), pools.csv (per-pool readiness), services.csv (resolved per-service paths),",
        "reconciliation.csv (source outcomes), path_map.json, replay_manifest.json, dependencies.json, dependency_files.csv.", "",
        "Validation boundaries:", *['- '+x for x in summary['limits']], "", "Blockers:",
        *[r['family_id']+': '+ '; '.join(r['blockers']) for r in pool_rows if r['blockers']]]
    (args.output_dir/"REPORT.txt").write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('services','families','families_dependencies_verified','families_blocked','database')}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
