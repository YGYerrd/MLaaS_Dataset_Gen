import json

import numpy as np
import pandas as pd

from mlaas_data_generator.cli.cmd_export_compositions import generate_compositions, load_services_from_db
from mlaas_data_generator.federated.update_signature import compute_and_store_update_signature, compute_update_signature
from mlaas_data_generator.storage.writer import make_writer


def _write_service(writer, service_id, signature):
    writer.write_service(
        {
            "service_id": service_id,
            "status": "completed",
            "task_family": "classification",
            "task_type": "classification",
            "modality": "text",
            "dataset": "synthetic",
            "model_type": "mlp",
            "training_regime": "generic",
        }
    )
    writer.write_service_metrics(
        service_id,
        {
            "metric_score": {"value": 0.8, "domain": "quality", "direction": "higher_better"},
            "compute_time_s": {"value": 0.1, "domain": "runtime", "direction": "lower_better"},
            "latency": {"value": 0.01, "domain": "latency", "direction": "lower_better"},
            "dataset_size": {"value": 10, "domain": "metadata"},
            "data_distribution": {"value": "iid", "domain": "metadata"},
            "resource_cost_score": {"value": 0.2, "domain": "cost", "direction": "higher_better"},
            "reliability_score": {"value": 1.0, "domain": "reliability", "direction": "higher_better"},
            "update_signature_id": {"value": signature["update_signature_id"], "domain": "metadata"},
            "signature_dim": {"value": signature["signature_dim"], "domain": "metadata"},
            "signature_norm": {"value": signature["signature_norm"], "domain": "metadata"},
            "update_signature_path": {"value": signature["update_signature_path"], "domain": "metadata"},
        },
    )
    writer.write_service_artifact(
        service_id,
        artifact_type="update_signature",
        artifact_uri=signature["update_signature_path"],
        metadata={"update_signature_id": signature["update_signature_id"]},
    )


def test_composition_export_computes_mus_from_selected_update_signatures(tmp_path):
    sig_a = compute_and_store_update_signature(
        {"w": np.asarray([0.0, 0.0])},
        {"w": np.asarray([1.0, 0.0])},
        output_dir=tmp_path / "sigs",
        run_id="svc_a",
        round_idx=1,
        client_id="service",
        dim=16,
        seed=5,
    )
    sig_b = compute_and_store_update_signature(
        {"w": np.asarray([0.0, 0.0])},
        {"w": np.asarray([2.0, 0.0])},
        output_dir=tmp_path / "sigs",
        run_id="svc_b",
        round_idx=1,
        client_id="service",
        dim=16,
        seed=5,
    )

    db_path = tmp_path / "services.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    _write_service(writer, "svc_a", sig_a)
    _write_service(writer, "svc_b", sig_b)
    writer.finish()

    services = load_services_from_db(db_path)
    requests = pd.DataFrame([
        {
            "request_id": "req_1",
            "task_family": "classification",
            "workflow_length": 2,
            "min_quality": 0.0,
            "max_latency": 1.0,
            "max_resource_cost": 1.0,
        }
    ])
    compositions = generate_compositions(services, requests, candidates_per_request=1, seed=1)

    assert len(compositions) == 1
    assert json.loads(compositions.iloc[0]["service_ids"]) == ["svc_a", "svc_b"]
    assert compositions.iloc[0]["mus"] > 0.99


def test_update_signature_supports_zero_baseline_estimator_state():
    signature = compute_update_signature(
        {},
        {"tree_0_threshold": np.asarray([0.2, -2.0, 0.8]), "feature_importances": np.asarray([0.7, 0.3])},
        dim=16,
        seed=7,
    )

    assert signature is not None
    assert signature["source_dim"] == 5
    assert signature["source_norm"] > 0
    assert signature["vector"].shape == (16,)
