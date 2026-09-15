import json
import pandas as pd
import pytest

from mlaas_composition.requirements import generate_service_requests
from mlaas_composition.protocol import validate_request_frame, read_request_file, write_request_file
from mlaas_composition.pool import build_composition_instance, ProvenanceIntegrityError


def services():
    return pd.DataFrame([{
        "service_id": f"client_{i}", "service_status": "completed", "generation_mode": "independent",
        "family_contract_version": "independent_family_v2", "family_id": "family", "client_id": str(i),
        "initial_state_hash": "a" * 64, "parent_state_hash": "a" * 64,
        "compatibility_contract_hash": "b" * 64, "composition_compatible": 1,
        "task_semantics_hash": "c" * 64, "evaluation_protocol_hash": "d" * 64,
        "task_family": "token_classification", "hf_task": "token_classification", "task_type": "classification",
        "modality": "text", "model_type": "hf_finetune", "input_schema": "tokens", "output_schema": "token_labels",
        "label_format": "token_labels", "architecture_id": "bert", "parameter_schema_hash": "schema",
        "label_map_hash": "labels", "preprocessing_contract_hash": "preprocess", "logical_dataset_contract_hash": "data",
        "metric_score": 0.7, "latency": 0.1, "compute_time_s": 1, "data_volume": 12,
        "resource_cost_score": 0.8, "reliability_score": 1, "explainability_score": 0.5,
        "dataset_name": "fixture", "dataset_config": "default", "benchmark_split": "validation",
    } for i in range(4)])


def test_v5_request_roundtrip_binds_semantics_and_eligible_membership(tmp_path):
    frame = services()
    requests = generate_service_requests(frame, request_count=1, profile_id="measured_tasks_v5", compatibility_mode="strict")
    assert requests.iloc[0].request_schema_version == "5.0.0"
    validate_request_frame(requests)
    file = write_request_file(requests, tmp_path / "requests.json")
    request = read_request_file(file).iloc[0].to_dict()
    assert len(build_composition_instance(frame, request).C_R) >= 2
    requirements = request["compatibility_requirements"]
    assert requirements["evaluation_protocol_hash"] == "d" * 64
    broken = frame.copy()
    broken.loc[0, "evaluation_protocol_hash"] = "e" * 64
    with pytest.raises((ValueError, ProvenanceIntegrityError)):
        build_composition_instance(broken, request)
    request["required_capabilities"] = "{}"
    with pytest.raises(ValueError, match="must require"):
        build_composition_instance(frame, request)


def test_v5_requires_strict_and_does_not_promote_missing_contracts():
    with pytest.raises(ValueError, match="strict"):
        generate_service_requests(services(), profile_id="measured_tasks_v5")
    frame = services().drop(columns="task_semantics_hash")
    with pytest.raises(ValueError, match="No viable"):
        generate_service_requests(frame, profile_id="measured_tasks_v5", compatibility_mode="strict")
