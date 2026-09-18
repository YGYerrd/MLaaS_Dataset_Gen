import copy
import json

import numpy as np
import pytest

from mlaas_composition.catalogue import load_services_from_db
from mlaas_data_generator.cli.manifest import hf_manifest_builder as builder
from mlaas_data_generator.services import runner
from mlaas_data_generator.storage.writer import make_writer


def test_flat_null_scores_clear_stale_catalogue_measurements(tmp_path):
    path = tmp_path / "catalogue.db"
    writer = make_writer("sqlite", db_path=str(path))
    writer.start()
    writer.write_service({"service_id": "service", "status": "completed", "task_family": "classification"})
    writer.write_service_metrics("service", {"trust_score": .9, "explainability_score": .8})
    writer.write_service_metrics("service", {"trust_score": None, "explainability_score": None})
    writer.finish()
    rows = load_services_from_db(path)
    assert rows[["trust_score", "explainability_score"]].isna().all().all()


def test_disabled_evaluation_is_explicit_and_clears_prior_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_perturbation_stage", lambda *a, **k: pytest.fail("disabled stage invoked"))
    metrics, artifact = runner._service_perturbation_metrics(object(), np.zeros((2, 1)), np.zeros(2),
        config={"enable_perturbation_metrics": False, "explainability_enabled": False}, meta={}, task_family="classification")
    values = {key: value["value"] for key, value in metrics.items()}
    assert values["evaluation_protocol_version"] == "benchmark_evaluation_v1"
    assert values["metric_evaluation_disabled"] is True
    assert artifact is None
    for dimension in ("trust", "explainability"):
        assert values[f"{dimension}_status"] == "unsupported"
        assert values[f"{dimension}_evidence_sufficient"] is False
        assert values[f"{dimension}_score"] is None


def test_hf_explanation_metadata_does_not_disable_trust(monkeypatch):
    registry = copy.deepcopy(builder.MODEL_REGISTRY)
    for model in registry.values():
        model["explainability"] = {"supported": False}
    monkeypatch.setattr(builder, "MODEL_REGISTRY", registry)
    frame = builder.build_hf_manifest(task_keys=["text_classification"], models_per_task=1,
        datasets_per_model=1, training_regimes=["finetune_transfer"], knob_variants_per_pair=1,
        total_services=1, seed=123, manifest_profile="test")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert bool(row.enable_perturbation_metrics) and not bool(row.explainability_enabled)
    payload = json.loads(row.service_config)
    assert payload["enable_perturbation_metrics"] is True
    assert payload["explainability_enabled"] is False


def test_fedavg_early_return_preserves_selected_evaluation_profile():
    frame = builder.build_hf_manifest(task_keys=["text_classification"], models_per_task=1,
        datasets_per_model=1, training_regimes=["finetune_transfer"], generation_mode="fedavg",
        fedavg_runs=1, fedavg_clients_per_run=4, fedavg_rounds=2, metric_evaluation_profile="pilot", seed=3)
    assert not frame.empty
    assert set(frame.metric_evaluation_profile) == {"pilot"}
    assert {json.loads(value)["metric_evaluation_profile"] for value in frame.service_config} == {"pilot"}


@pytest.mark.parametrize("metadata_groups", [False, True])
def test_runner_aligns_source_and_group_ids_to_frozen_evaluation_subset(monkeypatch, tmp_path, metadata_groups):
    seen = {}
    def evaluate(*args, **kwargs):
        seen.update(kwargs["config"])
        return {"evaluation_protocol_version": "production_evaluation_v1", "trust_score": .5}
    monkeypatch.setattr(runner, "run_perturbation_stage", evaluate)
    groups = ["document_a", "document_b", "document_a", "document_c"]
    config = {"enable_perturbation_metrics": True, "provenance_artifact_dir": str(tmp_path),
              "perturbation_stage_logging": False,
              "_prepared_benchmark_sample_info": {"source_ids": ["test:1", "test:3"]},
              "_family_document": {"reservation": {"metric_source_ids": [f"test:{i}" for i in range(4)],
                                                    "catalogue_positions": [1, 3]}}}
    meta = {"test_group_ids": groups} if metadata_groups else {}
    if not metadata_groups:
        config["task_source_groups"] = {"test": groups}
    runner._service_perturbation_metrics(object(), np.zeros((2, 1)), np.zeros(2),
        config=config, meta=meta, task_family="classification")
    assert seen["metric_evaluation_source_ids"] == ["test:1", "test:3"]
    assert seen["metric_evaluation_group_ids"] == ["document_b", "document_c"]
    assert "metric_evaluation_group_ids" not in config


def test_group_alignment_never_silently_treats_documents_as_independent_inputs():
    with pytest.raises(ValueError, match="cannot be aligned"):
        runner._evaluation_evidence_config({"task_source_groups": {"test": ["a", "b", "a"]}}, {}, 2)


def test_group_positions_must_match_frozen_source_order():
    config = {"task_source_groups": {"test": ["a", "b", "a"]},
              "_prepared_benchmark_sample_info": {"source_ids": ["test:2", "test:0"]},
              "_family_document": {"reservation": {"metric_source_ids": ["test:0", "test:1", "test:2"],
                                                    "catalogue_positions": [0, 2]}}}
    with pytest.raises(ValueError, match="disagree with group positions"):
        runner._evaluation_evidence_config(config, {}, 2)
