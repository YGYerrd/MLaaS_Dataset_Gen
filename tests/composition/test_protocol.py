import json
import sqlite3

import pandas as pd

from mlaas_composition.cli import main
from mlaas_composition.protocol import REQUEST_SCHEMA_VERSION, read_request_file, sha256_file, validate_request_frame, write_request_file
from mlaas_composition.sampler import calculate_utility
from mlaas_data_generator.storage.writer import make_writer


def test_worked_objective_example_is_stable():
    scores = {
        "data_parameter_compatibility": 0.0,
        "model_parameter_alignment": 0.0,
        "compute_time_compatibility": 0.0,
        "latency_feasibility": 0.8,
        "accuracy_compatibility": 0.9,
        "reliability_compatibility": 0.0,
        "explainability_score": 0.0,
        "cost_efficiency": 0.0,
        "mean_accuracy": 0.82,
        "total_latency": 1.2,
        "mean_compute_time": 0.0,
        "mean_reliability": 0.0,
        "mean_explainability": 0.0,
        "mean_resource_cost": 0.0,
        "dhs": 0.0,
        "mus": 0.0,
        "ses": 0.0,
        "hsq": 0.9,
        "srs": 0.0,
        "mean_latency": 1.2,
    }
    utility = calculate_utility(
        scores,
        {"accuracy": 1.0, "latency": 1.0},
        {"max_latency": 1.0},
        {"max_latency": 1.0},
    )
    assert utility == 0.81


def test_v1_requests_remain_valid_and_implicitly_functional():
    requests = pd.DataFrame([{
        "request_id": "legacy",
        "request_schema_version": "1.0.0",
        "composition_type": "collaborative",
        "functional_requirements": json.dumps({"task_family": "classification"}),
        "qos_requirements": json.dumps({"max_resource_cost": 0.5}),
        "hard_constraints": json.dumps({"max_resource_cost_score": 0.5}),
        "composition_size": 2,
        "preference_weights": json.dumps({"resource_cost": 1.0}),
        "constraint_penalties": json.dumps({"max_resource_cost": 1.0}),
    }])

    validate_request_frame(requests)


def test_cli_uses_immutable_requests_and_writes_hashed_run_manifest(tmp_path):
    db_path = tmp_path / "services.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    for service_id, metric_score in (("svc_a", 0.9), ("svc_b", 0.8)):
        writer.write_service(
            {
                "service_id": service_id,
                "status": "completed",
                "task_family": "classification",
                "task_type": "classification",
                "modality": "text",
                "dataset": "synthetic",
                "dataset_name": "synthetic",
                "benchmark_split": "test",
                "model_type": "generic",
                "training_regime": "generic",
                "input_schema": "feature_vector",
                "output_schema": "class_probability_vector",
                "functional_attributes_json": {"label_format": "single_index"},
            }
        )
        writer.write_service_metrics(
            service_id,
            {
                "metric_score": metric_score,
                "latency": 0.1,
                "compute_time_s": 0.2,
                "resource_cost_score": 0.1,
                "reliability_score": 1.0,
                "explainability_score": 0.7,
                "data_distribution": "iid",
            },
        )
    writer.finish()

    request_path = tmp_path / "requests.json"
    main([
        "generate-requests", "--db", str(db_path), "--output", str(request_path),
        "--requests", "1", "--seed", "17",
    ])
    requests = read_request_file(request_path)
    assert requests.loc[0, "request_schema_version"] == REQUEST_SCHEMA_VERSION
    assert requests.loc[0, "generation_seed"] == 17
    assert requests.loc[0, "request_profile_id"] == "controlled_collaborative_v3"
    assert requests.loc[0, "compatibility_mode"] == "functional"
    assert requests.loc[0, "principle_id"] == "feasible_reference_bounds_v1"

    output_dir = tmp_path / "experiment"
    main([
        "evaluate", "--db", str(db_path), "--requests-file", str(request_path),
        "--output-dir", str(output_dir), "--search-budget", "1", "--seed", "23",
    ])
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 23
    assert manifest["request_schema_version"] == "3.0.0"
    assert manifest["objective_version"] == "collaborative_objective.v2"
    assert manifest["inputs"]["service_catalogue"]["sha256"] == sha256_file(db_path)
    assert manifest["inputs"]["requests"]["sha256"] == sha256_file(request_path)
    assert set(manifest["inputs"]["protocol_assets"]) == {
        "request_profiles.v3.json",
        "request_principles.v1.json",
        "request.schema.v3.json",
        "objective.v2.json",
    }
    assert set(manifest["outputs"]) == {
        "services.csv", "requests.csv", "compositions.csv", "references.csv", "search_results.csv"
    }
    assert all((output_dir / name).exists() for name in manifest["outputs"])
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM services").fetchone()[0] == 2

    try:
        write_request_file(requests, request_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable request export was unexpectedly overwritten")
