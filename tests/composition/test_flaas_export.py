import csv
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from mlaas_composition.flaas_export import COLUMNS, REQUIRED_COLUMNS, export_flaas, model_size_mb


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def save_document(root, document):
    data = json.dumps(document, ensure_ascii=False, indent=2).encode()
    digest = hashlib.sha256(data).hexdigest()
    path = root / f"{digest}.json"
    path.write_bytes(data)
    return f"/old/artifacts/{path.name}", digest, path


@pytest.fixture
def dataset(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (tmp_path / "path_map.json").write_text(json.dumps({"version": "composition_path_map_v1", "prefixes": [
        {"recorded": "/old/artifacts", "resolved": str(artifacts)}]}))
    family = {"family_id": "family", "initial_state_hash": "initial", "contracts": {"parameter_schema": {
        "entries": [{"name": "weights", "dtype": "torch.float32", "numel": 10, "shape": [2, 5]},
                    {"name": "counter", "dtype": "torch.int64", "numel": 1, "shape": []}]}}}
    family_ref, family_hash, family_path = save_document(artifacts, family)
    source_db, db = tmp_path / "services.db", tmp_path / "compositions.db"
    recipes = []
    with closing(sqlite3.connect(source_db)) as source, closing(sqlite3.connect(db)) as comp, source, comp:
        source.executescript("""
            CREATE TABLE services(service_id TEXT PRIMARY KEY, metadata_json TEXT);
            CREATE TABLE service_generation_provenance(service_id TEXT PRIMARY KEY, family_id TEXT,
                client_recipe_hash TEXT, family_document_hash TEXT, initial_state_hash TEXT);
        """)
        comp.executescript((Path(__file__).parents[2] / "mlaas_composition/assets/composition_schema.sql").read_text())
        for run, status in (("r1", "completed"), ("r2", "completed"), ("r3", "failed"), ("r4", "pending")):
            comp.execute("INSERT INTO composition_runs VALUES (?,?,?,?,?,?,?,?,?)",
                         (run, run, "manifest", str(source_db), "v1", "{}", "2026-11-01", "2026-11-01", None))
            comp.execute("""INSERT INTO compositions(run_id,combination_id,family_id,composition_size,status,
                aggregation_provenance_json,started_at,finished_at) VALUES (?,?,?,?,?,?,?,?)""",
                         (run, "comb1", "family", 2, status,
                          json.dumps({"selected_service_order": ["svc_a", "svc_b"]}), "2026-11-01", "2026-11-02"))
            if status == "completed":
                for name, value, kind in (("accuracy", 0.0, "value_num"), ("total_wall_s", 23.0, "value_num"),
                                          ("aggregation_wall_s", 0.2, "value_num"), ("audit_examples", 64, "value_int")):
                    comp.execute(f"INSERT INTO composition_metrics(run_id,combination_id,metric_name,domain,direction,{kind}) VALUES (?,?,?,?,?,?)",
                                 (run, "comb1", name, "test", "neutral", value))
        for ordinal, sid in enumerate(("svc_a", "svc_b")):
            n = ordinal + 2
            recipe = {"service_id": sid, "family": {"family_id": "family", "document_hash": family_hash,
                "initial_state_hash": "initial"}, "selection": {"realised_count": n,
                "ordered_source_row_ids": [f"train:{i}" for i in range(n)]}, "training": {"effective_settings": {
                "learning_rate": 0.001}}, "observed_metrics": {"train_distribution_json": {"value": {"0": n, "1": 0}},
                "split_strategy_effective": {"value": "dirichlet"}, "train_time_s": {"value": 0.0}}}
            recipe["recipe_hash"] = hashlib.sha256(canonical(recipe).encode()).hexdigest()
            ref, digest, recipe_path = save_document(artifacts, recipe)
            recipes.append(recipe_path)
            snapshot = {"service_id": sid, "family_id": "family", "sample_count": n,
                        "client_recipe_ref": ref, "client_recipe_hash": digest,
                        "family_document_ref": family_ref, "family_document_hash": family_hash,
                        "initial_state_hash": "initial", "dataset_name": 'Data, "quoted"\nCaf\u00e9',
                        "task_type": "classification", "model_id": "org/model"}
            source.execute("INSERT INTO services VALUES (?,?)", (sid, json.dumps({"hardware_snapshot": {
                "cpu": {"physical_cores": 4, "logical_cores": 8}, "memory": {"total_mb": 1024}}})))
            source.execute("INSERT INTO service_generation_provenance VALUES (?,?,?,?,?)", (sid, "family", digest, family_hash, "initial"))
            initial = {"sample_count": n, "mean_loss": 1.0, "loss_sum": n, "squared_sample_loss_sum": n,
                       "model_state_hash": "initial", "loss_definition": "cross_entropy_per_example"}
            final = {**initial, "mean_loss": 2.0, "loss_sum": 2 * n,
                     "squared_sample_loss_sum": 5 * n, "model_state_hash": "final"}
            evidence = {**{k: snapshot[k] for k in ("service_id", "family_id", "client_recipe_hash", "family_document_hash", "initial_state_hash")},
                        "materialized_state_hash": "final", "training": {"training_epochs": 2,
                        "logical_batch_size": 8, "completed_training_epochs": 2, "micro_batch_size_effective": 4,
                        "gradient_accumulation_steps_effective": 2, "train_time_s": 5.0}}
            for run in ("r1", "r2", "r3", "r4"):
                current = dict(evidence)
                if run == "r1":
                    current.update(local_loss_current_model=1.0, squared_sample_loss_sum=5 * n,
                                   local_loss_l2_squared=5 * n, client_loss_statistics={"version": "client_loss_statistics_v1",
                                   "population": "selected_local_training_rows", "initial_model": initial, "final_local_model": final})
                comp.execute("INSERT INTO source_services VALUES (?,?,?,?,?)", (run, sid, "family", n, json.dumps(snapshot)))
                comp.execute("INSERT INTO composition_members VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (run, "comb1", sid, ordinal, n, n / 5, "completed", 11.0, "final", json.dumps(current)))
    return {"db": db, "source_db": source_db, "output": tmp_path / "flat.csv",
            "recipes": recipes, "family_path": family_path}


def export(data, **kwargs):
    return export_flaas(data["db"], data["source_db"], data["output"], **kwargs)


def read(data):
    with data["output"].open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def mutate_json(data, table, column, mutate, where="run_id='r1' AND service_id='svc_a'"):
    with closing(sqlite3.connect(data["db"])) as conn, conn:
        doc = json.loads(conn.execute(f"SELECT {column} FROM {table} WHERE {where}").fetchone()[0])
        mutate(doc)
        conn.execute(f"UPDATE {table} SET {column}=? WHERE {where}", (json.dumps(doc),))


def test_real_contract_roundtrip_and_determinism(dataset):
    before = {p: p.read_bytes() for p in [dataset["db"], dataset["source_db"], *dataset["recipes"], dataset["family_path"]]}
    summary = export(dataset)
    assert (summary["rows"], summary["completed_executions"], summary["distinct_clients"]) == (4, 2, 2)
    rows = read(dataset)
    assert len(REQUIRED_COLUMNS) == 38 and len(COLUMNS) == len(set(COLUMNS))
    assert list(rows[0]) == list(COLUMNS)
    assert dataset["output"].read_bytes().startswith(b"\xef\xbb\xbf")
    assert {(r["run_id"], r["combination_id"], r["client_id"]) for r in rows} == {
        (run, "comb1", sid) for run in ("r1", "r2") for sid in ("svc_a", "svc_b")}
    row = rows[0]
    assert row["dataset"] == 'Data, "quoted"\nCaf\u00e9'
    assert row["global_model_accuracy"] == row["local_training_time_s"] == "0.0"
    assert row["communication_round"] == row["emd_score"] == ""
    assert row["cpu_cores"] == "4" and float(row["ram_gb"]) == 1.073741824
    assert row["hardware_scope"] == "original_generation_host"
    assert float(row["model_size_mb"]) == 48 / 1_000_000
    assert row["data_volume"] == "2" and json.loads(row["class_distribution"]) == {"0": 2, "1": 0}
    assert json.loads(row["selected_client_ids"]) == ["svc_a", "svc_b"]
    assert row["observed_training_duration_s"] == "5.0" and row["reconstruction_duration_s"] == "11.0"
    assert row["squared_sample_loss_sum"] == row["local_loss_l2_squared"] == "10"
    assert rows[2]["squared_sample_loss_sum"] == ""
    assert "squared_sample_loss_sum=loss_telemetry_absent" in rows[2]["missing_attribute_reasons"]
    assert summary["missing_by_attribute"]["local_loss_current_model"] == 2
    first = dataset["output"].read_bytes()
    with pytest.raises(FileExistsError):
        export(dataset)
    export(dataset, overwrite=True)
    assert first == dataset["output"].read_bytes()
    assert before == {p: p.read_bytes() for p in before}
    assert not list(dataset["output"].parent.glob(".*.tmp"))


def test_missing_artifacts_keep_measured_results(dataset):
    dataset["recipes"][0].unlink()
    dataset["family_path"].unlink()
    export(dataset)
    row = read(dataset)[0]
    assert row["class_distribution"] == row["learning_rate"] == row["model_size_mb"] == ""
    assert "class_distribution=artifact_missing" in row["missing_attribute_reasons"]
    assert row["global_model_accuracy"] == "0.0" and row["observed_training_duration_s"] == "5.0"


@pytest.mark.parametrize("problem", ["source_identity", "artifact_hash", "snapshot_identity", "weight", "sample_count",
                                   "loss", "nan", "selected_ids", "accuracy", "duplicate_ordinal", "malformed_json"])
def test_invalid_evidence_preserves_existing_output(dataset, problem):
    dataset["output"].write_bytes(b"keep this output")
    if problem == "source_identity":
        with closing(sqlite3.connect(dataset["source_db"])) as conn, conn:
            conn.execute("UPDATE service_generation_provenance SET initial_state_hash='different'")
    elif problem == "artifact_hash":
        dataset["recipes"][0].write_text("{}")
    elif problem == "snapshot_identity":
        mutate_json(dataset, "source_services", "snapshot_json", lambda d: d.update(family_id="different"))
    elif problem in {"loss", "nan"}:
        mutate_json(dataset, "composition_members", "evidence_json", lambda d: d.update(
            squared_sample_loss_sum=4 if problem == "loss" else float("nan")))
    elif problem == "selected_ids":
        mutate_json(dataset, "compositions", "aggregation_provenance_json",
                    lambda d: d.update(selected_service_order=["svc_b", "svc_a"]), "run_id='r1'")
    else:
        sql = {"weight": "UPDATE composition_members SET aggregation_weight=0.2",
               "sample_count": "UPDATE composition_members SET sample_count=7",
               "accuracy": "UPDATE composition_metrics SET value_num=1.5 WHERE metric_name='accuracy'",
               "duplicate_ordinal": "UPDATE composition_members SET ordinal=0",
               "malformed_json": "UPDATE composition_members SET evidence_json='['"}[problem]
        with closing(sqlite3.connect(dataset["db"])) as conn, conn:
            conn.execute(sql)
    with pytest.raises(ValueError):
        export(dataset, overwrite=True)
    assert dataset["output"].read_bytes() == b"keep this output"
    assert not list(dataset["output"].parent.glob(".*.tmp"))


def test_schema_units_and_invalid_dtype():
    entries = [{"name": "half", "numel": 3, "shape": [3], "dtype": "bfloat16"},
               {"name": "flag", "numel": 1, "shape": [], "dtype": "bool"}]
    assert model_size_mb({"entries": entries}) == 7 / 1_000_000
    entries[0]["dtype"] = "unknown"
    with pytest.raises(ValueError, match="dtype"):
        model_size_mb({"entries": entries})


def test_cli_and_input_protection(dataset, capsys):
    from mlaas_composition.cli import main
    main(["export-flaas", "--db", str(dataset["db"]), "--source-db", str(dataset["source_db"]),
          "--output", str(dataset["output"])])
    assert json.loads(capsys.readouterr().out)["rows"] == 4
    with pytest.raises(ValueError, match="separate"):
        export_flaas(dataset["db"], dataset["source_db"], dataset["db"], overwrite=True)


def test_empty_completed_selection_writes_header(dataset):
    with closing(sqlite3.connect(dataset["db"])) as conn, conn:
        conn.execute("UPDATE compositions SET status='pending'")
    assert export(dataset)["rows"] == 0
    assert read(dataset) == []
