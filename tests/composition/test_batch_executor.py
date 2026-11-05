import csv
from contextlib import contextmanager
import json
from pathlib import Path
import queue
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from mlaas_composition import run_manifest as batch
from mlaas_composition.cli import main
from mlaas_composition.manifest import _digest
from mlaas_composition.recipe_execution import execute_composition
from mlaas_composition.storage import CompositionWriter
from mlaas_composition.validation import ServicePrediction
from mlaas_data_generator.checkpoint_store import capture_model_tensor_state
from mlaas_data_generator.cli.manifest import preparation
from mlaas_data_generator.client_reconstruction import persist_recipe
from mlaas_data_generator.data import master_loader
from mlaas_data_generator.provenance import sha256_json
from mlaas_data_generator.storage.writer import make_writer


class ToyModel:
    def __init__(self, value=0):
        self.values = [np.array([value], dtype=np.float32), np.array([7], dtype=np.int64)]

    def get_weights(self):
        return [value.copy() for value in self.values]

    def set_weights(self, values):
        self.values = [value.copy() for value in values]


def entry(family, *ids):
    ids = sorted(ids)
    return dict(combination_id="comb_" + _digest([family, ids]), family_id=family,
                composition_size=len(ids), service_ids=ids)


def manifest(path, entries):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows([{**e, "service_ids": json.dumps(e["service_ids"])} for e in entries])
    return path


@pytest.fixture
def source(tmp_path, monkeypatch):
    x = np.arange(1600, dtype="float32").reshape(400, 4)
    y = np.arange(400, dtype="int32") % 2
    monkeypatch.setattr(master_loader, "load_raw_dataset", lambda *a, **kw:
                        ((x, y), (x, y), {"input_shape": (4,), "task_type": "classification", "num_classes": 2}))
    db = tmp_path / "services.db"
    cfg = dict(service_id="a", dataset="digits", model_type="mlp", training_regime="generic",
               task_type="classification", modality="tabular", sample_size=8, sample_seed=42, seed=42,
               provenance_artifact_dir=str(tmp_path / "artifacts"), db_path=str(db),
               training_epochs=1, batch_size=2, learning_rate=0.01)
    doc, ref, _, _, initial, _ = preparation.prepare(cfg, build_model=lambda *a: ToyModel())
    writer = make_writer("sqlite", db_path=str(db))
    writer.start()
    for sid, count, policy in (("a", 2, "checkpoint"), ("b", 6, "recipe_only"), ("c", 4, "recipe_only")):
        config = {**cfg, "service_id": sid, "_family_document": doc, "family_document_ref": ref["document_ref"],
                  "family_document_hash": ref["document_hash"], "weight_retention": policy}
        service = SimpleNamespace(config=config, service_id=sid, _requested_configuration=config,
            _reconstruction_inputs={"selection": {"partition_source_ids": [f"train:{i}" for i in range(count)],
                                                  "effective_train_samples": count},
                                    "prepared_train_hash": "train", "prepared_evaluation_hash": "eval"},
            _reconstruction_training_observations={"train_stopping_reason": "epochs_completed",
                                                  "completed_training_epochs": 1, "optimizer_step_count": 2})
        metrics = {"primary_metric_name": {"value": "accuracy"}, "auxiliary_metric_name": {"value": "f1"},
                   "accuracy": {"value": 0.75}, "f1": {"value": 0.7}, "metric_score": {"value": 0.75}}
        artifact = persist_recipe(service, metrics, {"final_model_hash": initial.sha256})
        writer.write_service({"service_id": sid, "family_id": doc["family_id"], "status": "completed",
                              "task_family": "classification", "task_type": "classification", "model_type": "mlp", "modality": "tabular"})
        writer.write_service_generation_provenance({
            "service_id": sid, "family_id": doc["family_id"], "generation_mode": "independent",
            "provenance_version": 1, "provenance_json": {},
            "family_contract_version": doc["version"], "family_document_ref": ref["document_ref"],
            "family_document_hash": ref["document_hash"], "client_recipe_ref": artifact["artifact_uri"],
            "client_recipe_hash": artifact["metadata"]["sha256"], "initial_state_hash": initial.sha256,
            "parent_state_hash": initial.sha256, "final_model_hash": initial.sha256,
            "compatibility_contract_hash": doc["compatibility_contract_hash"], **doc["compatibility"]})
    writer.finish()
    calls, aggregate_values = [], []

    @contextmanager
    def materialize(db, sid, *, device, capture_client_losses=False):
        calls.append(sid)
        from mlaas_data_generator.client_losses import client_loss_evidence
        observations = {"sample_count": {"a": 2, "b": 6, "c": 4}[sid],
                        "mean_loss": 0.5, "squared_sample_loss_sum": 3.0,
                        "loss_definition": "test_loss", "measurement_wall_s": 0.0}
        yield {"model": ToyModel({"a": 2, "b": 6, "c": 10}[sid]), "initial": initial,
               "training": {"train_stopping_reason": "epochs_completed", "completed_training_epochs": 1},
               "client_losses": client_loss_evidence(observations, dict(observations),
                   initial_state_hash=initial.sha256, source_rows_hash="rows") if capture_client_losses else {},
               "replay_evidence": {"selected_rows_match": True, "effective_settings_match": True}}

    def predict(member, runtime_id, plan, state, *, device):
        aggregate_values.append(float(state.entries[0].to_numpy()[0]))
        assert state.entries[1].to_numpy()[0] == 7
        labels = {sid: label for sid, label in zip(doc["reservation"]["metric_source_ids"], doc["reservation"]["metric_source_labels"])}
        outputs = np.array([[0.9, 0.1] if labels[sid] == 0 else [0.1, 0.9] for sid in plan.audit_ids])
        return ServicePrediction(runtime_id, plan.audit_ids, outputs, 0.2, plan.label_hash, "classification")

    monkeypatch.setattr("mlaas_data_generator.client_reconstruction.materialize_service", materialize)
    monkeypatch.setattr("mlaas_composition.regeneration.execute_raw_family_state", predict)
    monkeypatch.setattr("mlaas_data_generator.cli.run_manifest._clear_cuda_runtime_state", lambda: None)
    return SimpleNamespace(db=db, doc=doc, initial=initial, calls=calls, aggregate_values=aggregate_values)


def test_replays_shared_and_checkpoint_members_measures_global_and_resumes(source, tmp_path):
    entries = [entry(source.doc["family_id"], "a", "b"), entry(source.doc["family_id"], "a", "c")]
    path = manifest(tmp_path / "manifest.csv", entries)
    db = tmp_path / "results.db"
    before = source.db.read_bytes()
    artifacts = {p: p.read_bytes() for p in (tmp_path / "artifacts").rglob("*") if p.is_file()}
    summary = batch.run_composition_manifest(path, source.db, db, device="cpu")
    assert summary["completed"] == 2 and summary["failed"] == 0
    assert source.calls == ["a", "b", "a", "c"]
    assert source.aggregate_values == pytest.approx([5, 22 / 3])
    assert source.db.read_bytes() == before
    assert artifacts == {p: p.read_bytes() for p in (tmp_path / "artifacts").rglob("*") if p.is_file()}
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM source_services").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM composition_members WHERE status='completed'").fetchone()[0] == 4
        assert conn.execute("SELECT value_num FROM composition_metrics WHERE metric_name='accuracy'").fetchall() == [(1.0,), (1.0,)]
        snapshot = json.loads(conn.execute("SELECT snapshot_json FROM source_services LIMIT 1").fetchone()[0])
        assert "recipe" not in snapshot and snapshot["metric_reference"]["accuracy"] == 0.75
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='services'").fetchone() is None
        evidence = json.loads(conn.execute("SELECT evidence_json FROM composition_members LIMIT 1").fetchone()[0])
        assert evidence["squared_sample_loss_sum"] == evidence["local_loss_l2_squared"] == 3.0
        assert evidence["local_loss_current_model"] == 0.5
        assert evidence["client_loss_statistics"]["final_local_model"]["model_state_hash"] == evidence["materialized_state_hash"]
        assert conn.execute("SELECT COUNT(*) FROM composition_metrics WHERE metric_name='client_loss_measurement_wall_s'").fetchone()[0] == 2
    assert summary["completed_without_client_losses"] == 0
    resumed = batch.run_composition_manifest(path, source.db, db, device="cpu")
    assert resumed["skipped"] == 2 and len(source.calls) == 4


def test_resume_preserves_older_completions_and_reports_missing_losses(source, tmp_path, capsys):
    path = manifest(tmp_path / "manifest.csv", [entry(source.doc["family_id"], "a", "b"),
                                               entry(source.doc["family_id"], "a", "c")])
    db = tmp_path / "results.db"
    first = batch.run_composition_manifest(path, source.db, db, device="cpu")
    with sqlite3.connect(db) as conn:
        cid = conn.execute("SELECT combination_id FROM compositions LIMIT 1").fetchone()[0]
        conn.execute("UPDATE composition_members SET evidence_json=json_remove(evidence_json, '$.client_loss_statistics') WHERE combination_id=?", (cid,))
        original = conn.execute("SELECT * FROM composition_metrics").fetchall()
    resumed = batch.run_composition_manifest(path, source.db, db, device="cpu")
    assert resumed["run_id"] == first["run_id"]
    assert resumed["skipped"] == 2 and resumed["completed_without_client_losses"] == 1
    assert len(source.calls) == 4
    assert "Skipping 1 completed compositions without current client loss statistics" in capsys.readouterr().out
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT * FROM composition_metrics").fetchall() == original


def test_dry_run_has_no_tensor_reads_or_output(source, tmp_path, monkeypatch):
    path = manifest(tmp_path / "manifest.csv", [entry(source.doc["family_id"], "a", "b")])
    monkeypatch.setattr(preparation, "load_lineage_base", lambda *a, **kw: pytest.fail("dry run loaded tensors"))
    db = tmp_path / "absent" / "results.db"
    summary = batch.run_composition_manifest(path, source.db, db, dry_run=True)
    assert summary["ready"] == 1 and not source.calls and not db.parent.exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_relocated_artifacts_preflight_execute_and_preserve_evidence(source, tmp_path, monkeypatch, explicit):
    from mlaas_data_generator.artifact_paths import current_artifact_prefixes
    original = Path(source.doc["store_root"])
    relocated = tmp_path / "moved artifacts"
    original.rename(relocated)
    path_map = tmp_path / ("custom_map.json" if explicit else "path_map.json")
    path_map.write_text(json.dumps({"version": "composition_path_map_v1", "prefixes": [
        {"recorded": str(original), "resolved": str(relocated)}]}))
    options = {"path_map": path_map} if explicit else {}
    path = manifest(tmp_path / "manifest.csv", [entry(source.doc["family_id"], "a", "b")])
    db = tmp_path / "results.db"
    source_bytes = source.db.read_bytes()
    artifacts = {p: p.read_bytes() for p in relocated.rglob("*") if p.is_file()}
    with monkeypatch.context() as patch:
        patch.setattr(preparation, "load_lineage_base", lambda *a, **kw: pytest.fail("dry run loaded tensors"))
        summary = batch.run_composition_manifest(path, source.db, db, dry_run=True, **options)
    assert summary["ready"] == 1 and summary["failed"] == 0
    assert not db.exists() and not source.calls
    summary = batch.run_composition_manifest(path, source.db, db, device="cpu", **options)
    assert summary["completed"] == 1 and summary["failed"] == 0
    assert source.calls == ["a", "b"] and not original.exists()
    assert current_artifact_prefixes() == ()
    assert source.db.read_bytes() == source_bytes
    assert artifacts == {p: p.read_bytes() for p in relocated.rglob("*") if p.is_file()}
    with sqlite3.connect(source.db) as conn:
        recipe = conn.execute("SELECT client_recipe_ref FROM service_generation_provenance WHERE service_id='a'").fetchone()[0]
    (relocated / Path(recipe).relative_to(original)).write_text("{}")
    summary = batch.run_composition_manifest(path, source.db, db, dry_run=True, **options)
    assert summary["ready"] == 0 and summary["failed"] == 1


def test_gpu_worker_receives_artifact_mapping(tmp_path, monkeypatch):
    from mlaas_data_generator.artifact_paths import artifact_path_scope, current_artifact_prefixes
    prefixes = (("/old/outputs", str(tmp_path)),)
    spawned = {}

    class Process:
        def __init__(self, **kwargs): spawned.update(kwargs)
        def start(self): pass

    monkeypatch.setattr(batch, "_gpu_environment", lambda slot: {})
    with artifact_path_scope(prefixes):
        worker = batch._start_worker(SimpleNamespace(Queue=queue.Queue, Process=Process), 0, queue.Queue(), "source.db")
    assert spawned["args"][-1] == prefixes
    job = entry("family", "a", "b")
    worker["jobs"].put((job, {}))
    worker["jobs"].put(None)

    def execute(*args, **kwargs):
        assert current_artifact_prefixes() == prefixes
        return {"status": "completed"}

    monkeypatch.setattr(batch, "execute_job", execute)
    spawned["target"](*spawned["args"])
    assert current_artifact_prefixes() == ()


def test_existing_execution_default_still_uses_retained_checkpoints(source, monkeypatch):
    members, plan, *_ = batch._prepare_composition(source.db, entry(source.doc["family_id"], "a", "b"))
    reads = []
    def retained(root, sid):
        reads.append(sid)
        return capture_model_tensor_state(ToyModel(2))
    monkeypatch.setattr("mlaas_data_generator.checkpoint_store.reconstruct_service_state", retained)
    result = execute_composition(source.db, members, plan)
    assert reads == ["a"] and source.calls == ["b"]
    assert result.aggregation_provenance["materialization"][0]["materialization_method"] == "retained_checkpoint"


@pytest.mark.parametrize("mutation", ["duplicate_member", "wrong_count", "wrong_identity", "duplicate_row", "bad_json", "missing_column"])
def test_invalid_manifests_are_rejected(tmp_path, mutation):
    row = entry("family", "a", "b")
    if mutation == "duplicate_member":
        row["service_ids"] = ["a", "a"]
    elif mutation == "wrong_count":
        row["composition_size"] = 3
    elif mutation == "wrong_identity":
        row["combination_id"] = "wrong"
    elif mutation == "missing_column":
        del row["composition_size"]
    path = manifest(tmp_path / "bad.csv", [row, row] if mutation == "duplicate_row" else [row])
    if mutation == "bad_json":
        path.write_text(path.read_text().replace("[", "{"))
    with pytest.raises(ValueError):
        batch.load_composition_manifest(path)


def test_member_failure_stops_aggregate_continues_and_retries(source, tmp_path, monkeypatch):
    path = manifest(tmp_path / "manifest.csv", [entry(source.doc["family_id"], "a", "b"), entry(source.doc["family_id"], "a", "c")])
    from mlaas_data_generator import client_reconstruction
    original = client_reconstruction.materialize_service

    @contextmanager
    def fail(db, sid, **kwargs):
        if sid == "b":
            raise RuntimeError("deliberate training failure")
        with original(db, sid, **kwargs) as run:
            yield run

    monkeypatch.setattr(client_reconstruction, "materialize_service", fail)
    db = tmp_path / "results.db"
    summary = batch.run_composition_manifest(path, source.db, db, device="cpu")
    assert summary["failed"] == summary["completed"] == 1
    assert len(source.aggregate_values) == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT failure_service_id FROM compositions WHERE status='failed'").fetchone()[0] == "b"
    monkeypatch.setattr(client_reconstruction, "materialize_service", original)
    summary = batch.run_composition_manifest(path, source.db, db, device="cpu")
    assert summary["completed"] == 2 and summary["skipped"] == 1
    assert source.calls[-2:] == ["a", "b"]
    with sqlite3.connect(db) as conn:
        assert sorted(r[0] for r in conn.execute("SELECT attempt_count FROM compositions")) == [1, 2]


def test_changed_recipe_creates_new_run_and_mutation_after_preflight_fails(source, tmp_path):
    job = entry(source.doc["family_id"], "a", "b")
    path = manifest(tmp_path / "manifest.csv", [job])
    db = tmp_path / "results.db"
    first = batch.run_composition_manifest(path, source.db, db, device="cpu")
    snapshots, errors = batch.preflight(source.db, [job])
    assert not errors
    with sqlite3.connect(source.db) as conn:
        ref = conn.execute("SELECT client_recipe_ref FROM service_generation_provenance WHERE service_id='a'").fetchone()[0]
        recipe = json.loads(Path(ref).read_text())
        recipe["observed_metrics"]["accuracy"]["value"] = 0.6
        recipe["recipe_hash"] = sha256_json({k: v for k, v in recipe.items() if k != "recipe_hash"})
        Path(ref).write_text(json.dumps(recipe))
        from mlaas_data_generator.provenance import sha256_file
        conn.execute("UPDATE service_generation_provenance SET client_recipe_hash=? WHERE service_id='a'", (sha256_file(ref),))
    result = batch.execute_job(source.db, job, {sid: sha256_json(s) for sid, s in snapshots.items()}, device="cpu")
    assert result["status"] == "failed" and "changed after" in result["error_message"]
    second = batch.run_composition_manifest(path, source.db, db, device="cpu")
    assert first["run_id"] != second["run_id"] and second["completed"] == 1


def test_missing_service_bad_family_and_unsupported_task_are_preflight_failures(source, tmp_path):
    jobs = [entry(source.doc["family_id"], "a", "missing"), entry("other_family", "a", "b")]
    _, errors = batch.preflight(source.db, jobs)
    assert len(errors) == 2
    with sqlite3.connect(source.db) as conn:
        conn.execute("UPDATE services SET task_family='fill_mask',task_type='fill_mask' WHERE service_id='b'")
    _, errors = batch.preflight(source.db, [entry(source.doc["family_id"], "a", "b")])
    assert len(errors) == 1


def test_new_preflight_failure_never_reuses_completed_results(source, tmp_path):
    path = manifest(tmp_path / "manifest.csv", [entry(source.doc["family_id"], "a", "b")])
    db = tmp_path / "results.db"
    first = batch.run_composition_manifest(path, source.db, db, device="cpu")
    with sqlite3.connect(source.db) as conn:
        conn.execute("UPDATE services SET modality='unsupported' WHERE service_id='b'")
    second = batch.run_composition_manifest(path, source.db, db, device="cpu")
    assert first["run_id"] != second["run_id"] and second["failed"] == 1 and second["skipped"] == 0


def test_rejects_source_destination_and_foreign_database(source, tmp_path):
    path = manifest(tmp_path / "manifest.csv", [entry(source.doc["family_id"], "a", "b")])
    with pytest.raises(ValueError, match="source input"):
        batch.run_composition_manifest(path, source.db, source.db)
    db = tmp_path / "foreign.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE unrelated (value)")
    with pytest.raises(ValueError, match="not a composition"):
        CompositionWriter(str(db)).start()


def test_cli_failure_exit_and_coordinator_lock(source, tmp_path):
    path = manifest(tmp_path / "manifest.csv", [entry(source.doc["family_id"], "a", "missing")])
    with pytest.raises(SystemExit) as error:
        main(["run-manifest", "--file", str(path), "--source-db", str(source.db), "--db", str(tmp_path / "x.db"), "--dry-run"])
    assert error.value.code == 1
    from filelock import FileLock, Timeout
    db = tmp_path / "locked.db"
    with FileLock(str(db) + ".lock"):
        with pytest.raises(Timeout):
            batch.run_composition_manifest(path, source.db, db, device="cpu")


def test_gpu_visibility_respects_inherited_noncontiguous_masks(monkeypatch):
    import torch
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assert batch._gpu_environment(1) == {"CUDA_VISIBLE_DEVICES": "5"}
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3,7")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    assert batch._gpu_environment(1) == {"CUDA_VISIBLE_DEVICES": "7", "HIP_VISIBLE_DEVICES": "7"}


def test_interrupted_composition_restarts_every_member(source, tmp_path, monkeypatch):
    path = manifest(tmp_path / "manifest.csv", [entry(source.doc["family_id"], "a", "b")])
    from mlaas_data_generator import client_reconstruction
    original = client_reconstruction.materialize_service
    @contextmanager
    def interrupted(db, sid, **kwargs):
        if sid == "b":
            raise KeyboardInterrupt()
        with original(db, sid, **kwargs) as run:
            yield run
    monkeypatch.setattr(client_reconstruction, "materialize_service", interrupted)
    db = tmp_path / "results.db"
    with pytest.raises(KeyboardInterrupt):
        batch.run_composition_manifest(path, source.db, db, device="cpu")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT status FROM compositions").fetchone()[0] == "running"
        assert conn.execute("SELECT COUNT(*) FROM composition_metrics").fetchone()[0] == 0
    monkeypatch.setattr(client_reconstruction, "materialize_service", original)
    assert batch.run_composition_manifest(path, source.db, db, device="cpu")["completed"] == 1
    assert source.calls == ["a", "a", "b"]


def test_completion_rolls_back_when_metric_write_fails(source, tmp_path, monkeypatch):
    path = manifest(tmp_path / "manifest.csv", [entry(source.doc["family_id"], "a", "b")])
    db = tmp_path / "results.db"
    def fail(*args): raise sqlite3.OperationalError("simulated full disk")
    with monkeypatch.context() as patch:
        patch.setattr(CompositionWriter, "_coerce_value_columns", fail)
        with pytest.raises(sqlite3.OperationalError, match="full disk"):
            batch.run_composition_manifest(path, source.db, db, device="cpu")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT status FROM compositions").fetchone()[0] == "running"
        assert conn.execute("SELECT COUNT(*) FROM composition_metrics").fetchone()[0] == 0
    assert batch.run_composition_manifest(path, source.db, db, device="cpu")["completed"] == 1


def test_dry_run_detects_missing_base_and_corrupted_recipe(source, tmp_path):
    from mlaas_data_generator.checkpoint_store import _identity_manifest_path
    base = _identity_manifest_path(Path(source.doc["store_root"]), "lineages", source.doc["lineage_id"])
    contents = base.read_bytes()
    base.unlink()
    job = entry(source.doc["family_id"], "a", "b")
    assert len(batch.preflight(source.db, [job])[1]) == 1
    base.write_bytes(contents)
    with sqlite3.connect(source.db) as conn:
        ref = conn.execute("SELECT client_recipe_ref FROM service_generation_provenance WHERE service_id='b'").fetchone()[0]
    Path(ref).write_text("{}")
    errors = batch.preflight(source.db, [job])[1]
    assert "integrity" in errors[job["combination_id"]]["error_message"]


def test_auto_requires_gpu_and_worker_cap(monkeypatch):
    from mlaas_data_generator.cli import run_manifest
    monkeypatch.setattr(run_manifest, "_detect_cuda_device_ids", lambda: [0, 1, 2])
    assert batch._gpu_slots(None) == [0, 1, 2] and batch._gpu_slots(2) == [0, 1]
    monkeypatch.setattr(run_manifest, "_detect_cuda_device_ids", lambda: [])
    with pytest.raises(ValueError, match="No GPU"):
        batch._gpu_slots(None)


def test_gpu_scheduler_replaces_crash_and_assigns_next_available(monkeypatch):
    jobs = [entry("family", "a", sid) for sid in ("b", "c", "d", "e")]
    events = queue.Queue()
    workers, beginnings, finishes = [], [], []

    class Process:
        exitcode = None
        def is_alive(self): return self.exitcode is None
        def join(self, timeout=None): pass
        def terminate(self): self.exitcode = -1

    class Jobs:
        def __init__(self, worker): self.worker = worker
        def put(self, job):
            if job is None:
                self.worker["process"].exitcode = 0
            else:
                if len(workers) == 1:
                    self.worker["process"].exitcode = 9
                else:
                    events.put((self.worker["id"], job[0]["combination_id"], "result", {"status": "completed"}))
        def close(self): pass
        def cancel_join_thread(self): pass

    def start(context, slot, events, source_db):
        w = {"id": str(len(workers)), "slot": slot, "process": Process(), "entry": None}
        w["jobs"] = Jobs(w)
        workers.append(w)
        return w

    events.close = lambda: None
    events.cancel_join_thread = lambda: None
    monkeypatch.setattr(batch.multiprocessing, "get_context", lambda mode: SimpleNamespace(Queue=lambda: events))
    monkeypatch.setattr(batch, "_start_worker", start)
    batch._run_gpu_jobs(jobs, {s: {} for s in "abcde"}, "source.db", [0, 1],
                        lambda e, w: beginnings.append((e["combination_id"], w)), lambda *a: None,
                        lambda e, r: finishes.append((e["combination_id"], r)))
    assert len(workers) == 3
    assert len(beginnings) == len(finishes) == 4
    assert sum(r["status"] == "failed" for _, r in finishes) == 1
    assert len({cid for cid, _ in finishes}) == 4
