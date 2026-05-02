from pathlib import Path
import sqlite3

from mlaas_data_generator.storage.writer import make_writer


def _load_sql_query(name: str) -> str:
    return (Path(__file__).resolve().parents[2] / name).read_text(encoding="utf-8")


def test_flatten_sql_prefers_model_hf_metadata_over_dataset_metadata(tmp_path):
    db_path = tmp_path / "services.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.write_service(
        {
            "service_id": "svc_flatten",
            "status": "completed",
            "task_family": "classification",
            "task_type": "classification",
            "modality": "text",
            "dataset": "hf",
            "dataset_name": "org/dataset",
            "model_type": "hf",
            "model_id": "org/model",
            "training_regime": "finetune_transfer",
        }
    )
    writer.write_service_metrics(
        "svc_flatten",
        {
            "metric_score": {"value": 0.9, "domain": "quality", "direction": "higher_better"},
            "primary_metric_name": {"value": "accuracy", "domain": "metadata"},
            "accuracy": {"value": 0.9, "domain": "quality", "direction": "higher_better"},
            "downloads": {"value": 123, "domain": "metadata"},
            "hf_model_downloads": {"value": 123, "domain": "metadata"},
            "hf_dataset_downloads": {"value": 456, "domain": "metadata"},
            "likes": {"value": 7, "domain": "metadata"},
            "hf_model_likes": {"value": 7, "domain": "metadata"},
            "hf_dataset_likes": {"value": 99, "domain": "metadata"},
            "hf_model_id": {"value": "org/model", "domain": "metadata"},
            "hf_dataset_id": {"value": "org/dataset", "domain": "metadata"},
        },
    )
    writer.finish()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(_load_sql_query("flatten.sql")).fetchone()

    assert row is not None
    assert row["HF model id"] == "org/model"
    assert row["HF dataset id"] == "org/dataset"
    assert float(row["Downloads"]) == 123.0
    assert float(row["Likes"]) == 7.0
    assert float(row["HF model downloads"]) == 123.0
    assert float(row["HF dataset downloads"]) == 456.0
    assert float(row["HF model likes"]) == 7.0
    assert float(row["HF dataset likes"]) == 99.0


def test_flatten_sql_flags_failed_rows_and_review_query_returns_them(tmp_path):
    db_path = tmp_path / "services.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.write_service(
        {
            "service_id": "svc_failed",
            "status": "failed",
            "task_family": "classification",
            "task_type": "classification",
            "modality": "text",
            "dataset": "hf",
            "dataset_name": "ag_news",
            "model_type": "hf_finetune",
            "model_id": "broken/model",
            "hf_task": "sequence_classification",
            "training_regime": "finetune_transfer",
        }
    )
    writer.write_service_failure(
        service_id="svc_failed",
        row_index=1,
        case_name="svc_failed_case",
        manifest_group_id="group-1",
        failure_stage="service_execution",
        error_message="simulated loader failure",
        resolved_config_json="{}",
        traceback_text="traceback",
    )
    writer.finish()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(_load_sql_query("flatten.sql")).fetchone()
        review_rows = conn.execute(_load_sql_query("review_bad_runs.sql")).fetchall()

    assert row is not None
    assert row["status"] == "failed"
    assert row["failure_stage"] == "service_execution"
    assert row["failure_message"] == "simulated loader failure"
    assert row["review_bucket"] == "failed"
    assert row["Primary metric"] == "Not Available"
    assert len(review_rows) == 1
    assert review_rows[0]["run_id"] == "svc_failed"


def test_flatten_sql_keeps_historical_failures_off_completed_rows(tmp_path):
    db_path = tmp_path / "services.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.write_service(
        {
            "service_id": "svc_completed",
            "status": "completed",
            "task_family": "classification",
            "task_type": "classification",
            "modality": "text",
            "dataset": "hf",
            "dataset_name": "ag_news",
            "model_type": "hf_finetune",
            "model_id": "distilbert-base-uncased",
            "hf_task": "sequence_classification",
            "training_regime": "finetune_transfer",
        }
    )
    writer.write_service_metrics(
        "svc_completed",
        {
            "metric_score": {"value": 0.82, "domain": "quality", "direction": "higher_better"},
            "primary_metric_name": {"value": "accuracy", "domain": "metadata"},
            "auxiliary_metric_name": {"value": "f1", "domain": "metadata"},
            "accuracy": {"value": 0.82, "domain": "quality", "direction": "higher_better"},
            "f1": {"value": 0.8, "domain": "quality", "direction": "higher_better"},
        },
    )
    writer.write_service_failure(
        service_id="svc_completed",
        row_index=2,
        case_name="svc_completed_case",
        manifest_group_id="group-2",
        failure_stage="service_execution",
        error_message="historical failure",
        resolved_config_json="{}",
        traceback_text="traceback",
    )
    writer.finish()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(_load_sql_query("flatten.sql")).fetchone()

    assert row is not None
    assert row["status"] == "completed"
    assert row["historical_failure_count"] == 1
    assert row["failure_stage"] is None
    assert row["failure_message"] is None
    assert row["review_bucket"] == "ok"


def test_flatten_sql_preserves_raw_lower_is_better_metric_and_metric_score(tmp_path):
    db_path = tmp_path / "services.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.write_service(
        {
            "service_id": "svc_lm",
            "status": "completed",
            "task_family": "generation",
            "task_type": "text_generation",
            "modality": "text",
            "dataset": "hf",
            "dataset_name": "roneneldan/TinyStories",
            "model_type": "hf_finetune",
            "model_id": "distilgpt2",
            "hf_task": "causal_lm_generation",
            "training_regime": "finetune_transfer",
        }
    )
    writer.write_service_metrics(
        "svc_lm",
        {
            "metric_score": {"value": 0.2, "domain": "quality", "direction": "higher_better"},
            "primary_metric_name": {"value": "loss", "domain": "metadata"},
            "auxiliary_metric_name": {"value": "perplexity", "domain": "metadata"},
            "loss": {"value": 4.0, "domain": "quality", "direction": "lower_better"},
            "perplexity": {"value": 54.6, "domain": "quality", "direction": "lower_better"},
        },
    )
    writer.finish()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(_load_sql_query("flatten.sql")).fetchone()

    assert row is not None
    assert row["Primary metric name"] == "loss"
    assert float(row["Primary metric"]) == 4.0
    assert float(row["metric_score"]) == 0.2
    assert row["review_bucket"] == "ok"


def test_flatten_sql_marks_degenerate_zero_metrics_for_review(tmp_path):
    db_path = tmp_path / "services.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.write_service(
        {
            "service_id": "svc_fillmask_zero",
            "status": "completed",
            "task_family": "classification",
            "task_type": "classification",
            "modality": "text",
            "dataset": "hf",
            "dataset_name": "wikitext",
            "model_type": "hf_finetune",
            "model_id": "suspicious/model",
            "hf_task": "fill_mask",
            "training_regime": "finetune_transfer",
        }
    )
    writer.write_service_metrics(
        "svc_fillmask_zero",
        {
            "metric_score": {"value": 0.0, "domain": "quality", "direction": "higher_better"},
            "primary_metric_name": {"value": "masked_accuracy", "domain": "metadata"},
            "auxiliary_metric_name": {"value": "perplexity_proxy", "domain": "metadata"},
            "masked_accuracy": {"value": 0.0, "domain": "quality", "direction": "higher_better"},
        },
    )
    writer.finish()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(_load_sql_query("flatten.sql")).fetchone()

    assert row is not None
    assert row["review_bucket"] == "degenerate_metric"
    assert "degenerate zero-valued primary metric" in row["review_reason"]
