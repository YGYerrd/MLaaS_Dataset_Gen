import sqlite3

from mlaas_data_generator.storage.writer import make_writer


def test_service_writer_persists_service_metrics_artifacts_and_failures(tmp_path):
    db_path = tmp_path / "services.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.write_service(
        {
            "service_id": "svc_1",
            "status": "completed",
            "task_family": "classification",
            "task_type": "classification",
            "modality": "text",
            "dataset": "hf",
            "dataset_name": "glue",
            "model_type": "hf",
            "model_id": "distilbert-base-uncased",
            "training_regime": "inference_only",
            "dataset_variant": "0",
            "split_variant": "0",
            "knob_variant": "0",
            "service_config_json": {"batch_size": 4},
        }
    )
    writer.write_service_metrics(
        "svc_1",
        {
            "accuracy": {"value": 0.8, "domain": "quality", "direction": "higher_better"},
            "inference_latency_s_mean": {"value": 0.01, "domain": "latency", "unit": "s", "direction": "lower_better"},
            "resource_cost_score": {"value": 0.7, "domain": "cost", "direction": "higher_better"},
        },
    )
    writer.write_service_artifact("svc_1", artifact_type="report", artifact_uri="outputs/report.json", metadata={"kind": "smoke"})
    writer.write_service_split_provenance(
        "svc_1",
        split_name="benchmark",
        samples_count=8,
        data_distribution={"0": 4, "1": 4},
        split_config={"source": "validation"},
    )
    writer.write_service_failure(
        service_id="svc_1",
        row_index=0,
        case_name="case",
        manifest_group_id="group",
        failure_stage="validation",
        error_message="example",
        resolved_config_json="{}",
    )
    writer.finish()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM services").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM service_metrics").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM service_artifacts").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM service_split_provenance").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM service_failures").fetchone()[0] == 1
        old_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('runs','rounds','clients','measurements','service_client_distributions')"
            )
        }
        assert old_tables == set()
