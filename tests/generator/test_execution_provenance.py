import json
import sqlite3

import pytest

from mlaas_data_generator.execution_provenance import configuration_evidence, execution_document
from mlaas_data_generator.provenance import sha256_json
from mlaas_data_generator.storage.writer import make_writer


def test_execution_evidence_is_untruncated_and_independent(tmp_path):
    requested = configuration_evidence({"batch_size": 16, "recipe": {"values": list(range(5000))},
                                        "long_setting": "x" * 150000, "_prepared_dataset": object(),
                                        "row_index": 7, "manifest_group_id": "invocation"})
    effective = {**requested, "batch_size": 4, "learning_rate": 0.001}
    document = execution_document(requested=requested, effective=effective,
        metrics={"microbatch_retries": {"value": 2}}, record={},
        splits=[{"selected_rows": list(range(6000))}], generation={}, artifacts=[], runtime={})
    # Capturing the request must detach nested mutable settings from execution.
    effective["recipe"]["values"].append(5000)
    assert document["requested_config"]["recipe"]["values"] == list(range(5000))
    writer = make_writer("sqlite", db_path=str(tmp_path / "dataset.db"))
    writer.start()
    writer.write_service({"service_id": "svc", "status": "completed"})
    for _ in range(2):
        writer.write_service_execution_provenance(service_id="svc", run_id="run", started_at="now", document=document)
    writer.write_service({"service_id": "svc", "status": "completed"})
    writer.finish()
    with sqlite3.connect(tmp_path / "dataset.db") as conn:
        rows = conn.execute("SELECT document_json,document_sha256 FROM service_execution_provenance").fetchall()
        assert len(rows) == 2
        stored = json.loads(rows[0][0])
        assert rows[0][1] == sha256_json(stored)
        assert stored["requested_config"]["long_setting"] == "x" * 150000
        assert stored["splits"][0]["selected_rows"] == list(range(6000))
        assert stored["configuration_adjustments"]["batch_size"]["before"] == 16
        assert stored["configuration_adjustments"]["batch_size"]["after"] == 4
        assert "_prepared_dataset" not in stored["requested_config"]
        assert "row_index" not in stored["requested_config"]
        assert "manifest_group_id" not in stored["requested_config"]
        for sql in (
            "UPDATE service_execution_provenance SET document_json='{}'",
            "DELETE FROM service_execution_provenance",
            "INSERT OR REPLACE INTO service_execution_provenance SELECT * FROM service_execution_provenance",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                conn.execute(sql)


def test_training_configuration_survives_recipe_restoration(monkeypatch):
    from mlaas_data_generator.services.runner import ServiceRunner
    from mlaas_data_generator.clients import training

    runner = ServiceRunner({"service_id": "test", "batch_size": 16, "training_epochs": 8})
    monkeypatch.setattr(training, "realized_training_recipe", lambda config, count: (
        {**config, "training_epochs": 2, "gradient_accumulation_steps": 1}, {"reason": "test adjustment"}))
    monkeypatch.setattr(runner, "_train_model_effective", lambda *args, **kwargs: {})
    runner._train_model(None, [1, 2], [0, 1], task_family="classification")
    assert runner.config["training_epochs"] == 8
    assert runner._effective_training_configuration["training_epochs"] == 2


def test_existing_database_gets_no_invented_execution_evidence(tmp_path):
    writer = make_writer("sqlite", db_path=str(tmp_path / "dataset.db"))
    writer.start()
    writer.finish()
    writer.start()
    assert writer.conn.execute("SELECT count(*) FROM service_execution_provenance").fetchone()[0] == 0
    writer.finish()
