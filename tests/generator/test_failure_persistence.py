import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from mlaas_data_generator.diagnostic_json import diagnostic_json, MAX_CONFIG_BYTES
from mlaas_data_generator import execution_state as state
from mlaas_data_generator.services.runner import ServiceRunner
from mlaas_data_generator.storage.writer import SQLiteWriter


def config(tmp_path):
    return {"db_path": str(tmp_path / "services.db"), "service_id": "test_failure",
            "model_type": "randomforest", "dataset": "sklearn", "task_type": "classification",
            "_execution_run_id": "this_run"}


def saved(cfg):
    if not Path(cfg["db_path"]).is_file():
        return None
    with sqlite3.connect(cfg["db_path"]) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='services'").fetchone():
            return None
        return conn.execute("SELECT status,metadata_json FROM services WHERE service_id=?", (cfg["service_id"],)).fetchone()


def test_runtime_objects_never_expand_into_failure_json(tmp_path):
    class Explosive:
        def __str__(self):
            pytest.fail("runtime object was stringified")
        __repr__ = __str__
    cfg = {**config(tmp_path), "_parent_state": Explosive(), "_prepared_model": Explosive(),
           "public_nested": {"weights": np.zeros((1000, 1000)), "model": Explosive()},
           "long_text": "x" * 1_000_000}
    payload = diagnostic_json(cfg)
    assert len(payload.encode()) <= MAX_CONFIG_BYTES
    doc = json.loads(payload)
    assert "_parent_state" not in doc and "_prepared_model" not in doc
    assert doc["public_nested"]["weights"]["shape"] == [1000, 1000]
    state.persist_terminal(cfg, status="failed", reason="failed")
    with sqlite3.connect(cfg["db_path"]) as conn:
        size = conn.execute("SELECT length(resolved_config_json) FROM service_failures").fetchone()[0]
    assert size <= MAX_CONFIG_BYTES
    assert saved(cfg)[0] == "failed"


def test_status_commits_even_if_failure_details_cannot_be_written(tmp_path, monkeypatch, capsys):
    cfg = config(tmp_path)
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("injected detail write failure")
    monkeypatch.setattr(SQLiteWriter, "write_service_failure", fail)
    state.persist_terminal(cfg, status="failed", reason="model failed")
    assert saved(cfg)[0] == "failed"
    assert "diagnostic write failed" in capsys.readouterr().out


def test_failed_status_write_has_durable_recoverable_journal(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(SQLiteWriter, "start", lambda self: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
        state.persist_terminal(cfg, status="failed", reason="model failed")
    assert state.recover_service_states(cfg["db_path"]) == 1
    assert saved(cfg)[0] == "failed"
    assert state.recover_service_states(cfg["db_path"]) == 0


@pytest.mark.parametrize("error,status", [(ValueError("bad model"), "failed"), (KeyboardInterrupt(), "interrupted")])
def test_runner_persists_terminal_status_on_exception_and_interrupt(tmp_path, monkeypatch, error, status):
    cfg = config(tmp_path)
    instance = ServiceRunner(cfg)
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(instance, "_execute_service", fail)
    if status == "interrupted":
        with pytest.raises(KeyboardInterrupt):
            instance.run()
    else:
        assert instance.run().status == "failed"
    if status == "interrupted":
        assert saved(cfg) is None
        assert not Path(cfg["db_path"]).exists()
        return
    actual, metadata = saved(cfg)
    assert actual == status
    assert json.loads(metadata)["completed_at"]


def test_queued_work_never_creates_database_or_overwrites_result(tmp_path):
    cfg = config(tmp_path)
    state.queue_services([cfg])
    assert not Path(cfg["db_path"]).exists()
    assert state.recover_service_states(cfg["db_path"]) == 0
    state.persist_terminal(cfg, status="failed", reason="real failure")
    state.queue_services([cfg])
    state.persist_terminal(cfg, status="interrupted", reason="cancelled retry")
    assert saved(cfg)[0] == "failed"


def test_legacy_recovery_removes_only_abandoned_pending_rows(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    state.persist_terminal({**cfg, "service_id": "done"}, status="completed")
    _, record = state.state_record(cfg, "queued")
    with sqlite3.connect(cfg["db_path"]) as conn:
        conn.execute("INSERT INTO services(service_id,status,metadata_json) VALUES (?,?,?)",
                     (cfg["service_id"], "queued", diagnostic_json(record["metadata_json"])))
    assert state.recover_service_states(cfg["db_path"]) == 0
    assert state.recover_service_states(cfg["db_path"], run_id="other_run") == 0
    assert state.recover_service_states(cfg["db_path"], run_id="this_run") == 1
    assert saved(cfg) is None
    assert saved({**cfg, "service_id": "done"})[0] == "completed"


@pytest.mark.parametrize("status", ["queued", "running", "interrupted"])
def test_writer_rejects_unfinished_catalogue_rows(tmp_path, status):
    writer = SQLiteWriter(str(tmp_path / "guard.db"))
    writer.start()
    try:
        for method in (writer.write_service, writer.write_service_progress):
            with pytest.raises(ValueError, match="Unfinished"):
                method({"service_id": "pending", "status": status})
        assert writer.conn.execute("SELECT count(*) FROM services").fetchone()[0] == 0
    finally:
        writer.abort()


def test_interrupted_legacy_journal_cannot_overwrite_success(tmp_path):
    cfg = config(tmp_path)
    state.persist_terminal(cfg, status="completed")
    _, record = state.state_record(cfg, "interrupted")
    state._journal(cfg["db_path"], record)
    state.recover_service_states(cfg["db_path"])
    assert saved(cfg)[0] == "completed"


def test_abandoned_roster_accounts_for_undispatched_clients(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    done = {**cfg, "service_id": "done"}
    queued = {**cfg, "service_id": "never_dispatched"}
    state.register_manifest_roster([done, queued], run_id="this_run", manifest_path="clients.csv")
    state.persist_terminal(done, status="completed")
    monkeypatch.setattr(state, "owner_alive", lambda owner: False)
    assert state.reconcile_manifest_roster(cfg["db_path"]) == 2
    assert saved(done)[0] == "completed"
    assert saved(queued) is None
    with sqlite3.connect(cfg["db_path"]) as conn:
        assert dict(conn.execute("SELECT service_id,status FROM manifest_service_runs")) == {
            "done": "completed"}


def test_roster_prefers_this_attempt_in_gpu_database_over_old_failure(tmp_path):
    cfg = config(tmp_path)
    state.persist_terminal({**cfg, "_execution_run_id": "older_run"}, status="failed", reason="old failure")
    state.register_manifest_roster([cfg], run_id="this_run", manifest_path="clients.csv")
    gpu = {**cfg, "db_path": str(tmp_path / "services.gpu0.db")}
    state.persist_terminal(gpu, status="completed")
    state.reconcile_manifest_roster(cfg["db_path"], run_id="this_run")
    with sqlite3.connect(cfg["db_path"]) as conn:
        assert conn.execute("SELECT status FROM manifest_service_runs").fetchone()[0] == "completed"


def test_family_preparation_failure_has_terminal_records_for_every_member(tmp_path, monkeypatch):
    from mlaas_data_generator.cli import run_manifest as manifest
    from mlaas_data_generator.cli.manifest import preparation
    configs = [{**config(tmp_path), "service_id": f"client_{i}", "dataset": "hf",
                "dataset_name": "glue", "dataset_config": "mrpc", "hf_task": "sentence_similarity",
                "hf_model_id": "local-test", "model_type": "hf_finetune", "training_regime": "finetune_transfer"}
               for i in range(2)]
    entries = [manifest.ManifestEntry(idx=i, ordinal=i + 1, resolved=cfg, validation=manifest.RowValidation(True))
               for i, cfg in enumerate(configs)]
    def fail(*args, **kwargs):
        raise ValueError("preparation failed before training")
    monkeypatch.setattr(preparation, "prepare", fail)
    results = manifest._execute_hf_model_group(entries)
    assert len(results) == 2 and all(row["status"] == "failed" for row in results)
    assert all(saved(cfg)[0] == "failed" for cfg in configs)


def test_manifest_shutdown_accounts_for_clients_before_dispatch(tmp_path, monkeypatch):
    import pandas as pd
    from mlaas_data_generator.cli import run_manifest as manifest
    cfg = config(tmp_path)
    source = tmp_path / "clients.csv"
    pd.DataFrame([{"service_id": "first"}, {"service_id": "second"}]).to_csv(source, index=False)
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(manifest, "_ensure_manifest_preflight", interrupt)
    with pytest.raises(KeyboardInterrupt):
        manifest.run_manifest(str(source), db_path=cfg["db_path"])
    assert not Path(cfg["db_path"]).exists()
    assert not list(Path(cfg["db_path"] + ".manifest-roster").glob("*.json"))


def test_roster_retains_intentional_fedavg_nonparticipation(tmp_path):
    from mlaas_data_generator.cli import run_manifest as manifest
    cfg = {**config(tmp_path), "fl_run_id": "fedavg_run"}
    state.register_manifest_roster([cfg], run_id="this_run", manifest_path="clients.csv")
    entry = manifest.ManifestEntry(idx=0, ordinal=1, resolved=cfg, validation=manifest.RowValidation(True))
    result = manifest._result_row(cfg, 0, "not_participating", "not scheduled", service_id=cfg["service_id"])
    manifest._finalize_manifest_results([entry], [result])
    state.reconcile_manifest_roster(cfg["db_path"], run_id="this_run")
    assert saved(cfg)[0] == "not_participating"
    with sqlite3.connect(cfg["db_path"]) as conn:
        assert conn.execute("SELECT status FROM manifest_service_runs").fetchone()[0] == "not_participating"


def test_malformed_manifest_row_is_failed_and_exported(tmp_path, monkeypatch):
    import pandas as pd
    from mlaas_data_generator.cli import run_manifest as manifest
    cfg = config(tmp_path)
    source = tmp_path / "clients.csv"
    pd.DataFrame([{"service_id": "bad_setting", "batch_size": "not-an-integer"}]).to_csv(source, index=False)
    monkeypatch.setattr(manifest, "_ensure_manifest_preflight", lambda *args: None)
    monkeypatch.setattr(manifest, "MANIFEST_RESULTS_PATH", tmp_path / "results.csv")
    monkeypatch.setattr(manifest, "FAILED_MANIFEST_PATH", tmp_path / "failed.csv")
    monkeypatch.setattr(manifest, "FAILURE_LOG_PATH", tmp_path / "failures.log")
    monkeypatch.setattr(manifest, "_write_multimodal_audit_csv", lambda **kwargs: None)
    monkeypatch.setattr(manifest, "_write_missing_multimodal_manifest_csv", lambda **kwargs: None)
    manifest.run_manifest(str(source), db_path=cfg["db_path"])
    assert saved({**cfg, "service_id": "bad_setting"})[0] == "failed"
    assert pd.read_csv(tmp_path / "failed.csv").iloc[0]["batch_size"] == "not-an-integer"
