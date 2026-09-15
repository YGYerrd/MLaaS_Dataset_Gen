import sqlite3

from mlaas_data_generator.fedavg_run import recover_abandoned_fedavg_runs
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


def test_fedavg_recovery_marks_abandoned_run_and_round_interrupted(tmp_path):
    db_path = tmp_path / "fedavg-recovery.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.write_fedavg_run(
        {
            "fl_run_id": "run_abandoned",
            "lineage_id": "lineage_abandoned",
            "status": "running",
            "client_count": 2,
            "round_count": 2,
            "participation_rate": 1.0,
            "schedule_seed": 1,
            "schedule_version": "balanced_rotating_v1",
            "started_at": "2026-01-01T00:00:00+00:00",
        }
    )
    writer.write_fedavg_round(
        {
            "fl_run_id": "run_abandoned",
            "round_id": 1,
            "status": "running",
            "scheduled_count": 2,
            "successful_count": 1,
            "failed_count": 0,
            "started_at": "2026-01-01T00:00:01+00:00",
        }
    )
    writer.finish()

    recovered = recover_abandoned_fedavg_runs(str(db_path))
    assert recovered == {"runs": 1, "rounds": 1}
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT status FROM fedavg_runs WHERE fl_run_id='run_abandoned'"
        ).fetchone()[0] == "interrupted"
        assert conn.execute(
            "SELECT status FROM fedavg_rounds WHERE fl_run_id='run_abandoned'"
        ).fetchone()[0] == "interrupted"
