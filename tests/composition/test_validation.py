import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from mlaas_composition import fedavg_validation as fedavg_module
from mlaas_composition.fedavg_validation import (
    FedAvgCompositionExecution,
    FedAvgProvenanceError,
    execute_selected_fedavg_composition,
    resolve_selected_fedavg_members,
)
from mlaas_composition.validation import (
    CohortIsolationError,
    ServicePrediction,
    aggregate_and_measure,
    average_probabilities,
    build_cohort_plan,
    measure_model_outputs,
)
from mlaas_composition.validation_cli import (
    _read_selected_composition,
    validate_from_files,
)
from mlaas_data_generator.checkpoint_store import (
    CompressionSpec,
    ContentAddressedTensorStore,
    TENSOR_STATE_FORMAT,
    capture_model_tensor_state,
    persist_service_patch_from_base_state,
)
from mlaas_data_generator.storage.writer import make_writer


def _plan(task_family="classification"):
    labels = [float(i) if task_family == "regression" else i % 2 for i in range(12)]
    return build_cohort_plan(
        training_ids=["train:0", "train:1"],
        metric_source_ids=[f"test:{i}" for i in range(12)],
        metric_source_labels=labels,
        contract={"dataset": "fixture"},
        audit_samples=6,
        minimum_audit_samples=4,
        task_family=task_family,
    )


def test_cohort_plan_is_deterministic_and_disjoint():
    first = _plan()
    second = _plan()
    assert first == second
    assert not set(first.training_ids) & set(first.catalogue_metric_ids)
    assert not set(first.training_ids) & set(first.audit_ids)
    assert not set(first.catalogue_metric_ids) & set(first.audit_ids)


def test_legacy_probability_averaging_still_rejects_different_audit_ids():
    plan = _plan()
    good = ServicePrediction(
        "a",
        plan.audit_ids,
        np.tile([[0.8, 0.2]], (len(plan.audit_ids), 1)),
        0.1,
        plan.label_hash,
    )
    bad = ServicePrediction(
        "b",
        tuple(reversed(plan.audit_ids)),
        np.tile([[0.7, 0.3]], (len(plan.audit_ids), 1)),
        0.1,
        plan.label_hash,
    )
    with pytest.raises(CohortIsolationError):
        average_probabilities([good, bad], plan)


def test_single_materialized_model_metrics_cover_classification_and_regression():
    classification = _plan()
    classification_labels = np.asarray(
        [int(item.split(":")[1]) % 2 for item in classification.audit_ids]
    )
    probabilities = np.asarray(
        [[0.9, 0.1] if value == 0 else [0.1, 0.9] for value in classification_labels]
    )
    measured = measure_model_outputs(
        probabilities,
        classification,
        classification_labels,
        task_family="classification",
        inference_latency_s=0.25,
    )
    assert measured["accuracy"] == pytest.approx(1.0)
    assert measured["f1_macro"] == pytest.approx(1.0)
    assert measured["inference_latency_s"] == pytest.approx(0.25)

    regression = _plan("regression")
    labels = np.asarray([float(item.split(":")[1]) for item in regression.audit_ids])
    measured = measure_model_outputs(
        labels + 2.0,
        regression,
        labels,
        task_family="regression",
        inference_latency_s=0.5,
    )
    assert measured["rmse"] == pytest.approx(2.0)
    assert measured["mae"] == pytest.approx(2.0)
    assert measured["quality_score"] == pytest.approx(1.0 / 3.0)


def test_legacy_prediction_ensemble_primitive_is_not_used_by_new_measurement():
    plan = _plan("regression")
    labels = np.asarray([float(item.split(":")[1]) for item in plan.audit_ids])
    left = ServicePrediction(
        "left", plan.audit_ids, labels + 1.0, 0.2, plan.label_hash, "regression"
    )
    right = ServicePrediction(
        "right", plan.audit_ids, labels - 1.0, 0.3, plan.label_hash, "regression"
    )
    _, legacy = aggregate_and_measure(
        [left, right], plan, labels, task_family="regression"
    )
    actual = measure_model_outputs(
        labels + 2.0,
        plan,
        labels,
        task_family="regression",
        inference_latency_s=0.1,
    )
    assert legacy["rmse"] == pytest.approx(0.0)
    assert actual["rmse"] == pytest.approx(2.0)


class _ToyModel:
    def __init__(self, float_value=0.0, integer_value=7):
        self.weights = [
            np.asarray([float_value], dtype=np.float32),
            np.asarray([integer_value], dtype=np.int64),
        ]

    def get_weights(self):
        return [np.array(value, copy=True) for value in self.weights]

    def set_weights(self, values):
        self.weights = [np.array(value, copy=True) for value in values]


def _create_fedavg_fixture(tmp_path, *, task_family="classification"):
    db = tmp_path / "services.db"
    root = tmp_path / "service_artifacts"
    parent_model = _ToyModel()
    parent = capture_model_tensor_state(parent_model)
    store = ContentAddressedTensorStore(root)
    parent_ref = store.put_state(parent, compression=CompressionSpec("zlib", 1))
    schema_hash = "a" * 64
    contract_hash = "b" * 64
    shared = {
        "fl_run_id": "run-1",
        "lineage_id": "lineage-1",
        "round_id": 1,
        "parent_global_round": 0,
        "parent_state_hash": parent.sha256,
        "compatibility_contract_hash": contract_hash,
        "parameter_schema_hash": schema_hash,
    }
    specifications = [
        ("service-a", "client-a", 1, 2.0, 10),
        ("service-b", "client-b", 3, 6.0, 20),
        ("service-c", "client-c", 10, 100.0, 30),
    ]
    patches = {}
    for service_id, _client_id, _samples, float_value, integer_value in specifications:
        model = _ToyModel(float_value, integer_value)
        patches[service_id] = persist_service_patch_from_base_state(
            root,
            service_id,
            shared["lineage_id"],
            model,
            base_state=parent,
            base_ref=parent_ref,
            base_round_id=0,
            parameter_schema_hash=schema_hash,
            compression=CompressionSpec("zlib", 1),
        )

    writer = make_writer("sqlite", db_path=str(db))
    writer.start()
    writer.write_fedavg_run(
        {
            "fl_run_id": shared["fl_run_id"],
            "lineage_id": shared["lineage_id"],
            "status": "complete",
            "client_count": 3,
            "round_count": 2,
            "participation_rate": 1.0,
            "schedule_seed": 1,
            "schedule_version": "test",
            "partition_generator_version": "joint_target_allocator_v2",
            "heterogeneity_intensity": 0.5,
            "compatibility_contract_hash": contract_hash,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    writer.write_model_lineage(
        {
            "lineage_id": shared["lineage_id"],
            "fl_run_id": shared["fl_run_id"],
            "provenance_json": {},
        }
    )
    writer.write_fedavg_global_state(
        {
            "fl_run_id": shared["fl_run_id"],
            "global_round": 0,
            "state_hash": parent.sha256,
            "cas_ref": str(parent_ref.manifest_path),
            "parameter_schema_hash": schema_hash,
            "parent_global_round": None,
            "parent_state_hash": None,
            "storage_format": "full_immutable_tensor_state_v1",
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    for service_id, client_id, samples, _float_value, _integer_value in specifications:
        patch = patches[service_id]
        writer.write_service(
            {
                "service_id": service_id,
                "status": "completed",
                "task_family": task_family,
                "task_type": task_family,
                "modality": "tabular",
                "dataset": "synthetic",
                "dataset_name": "fixture",
                "benchmark_split": "test",
                "model_type": "mlp",
                "training_regime": "generic",
                "functional_attributes_json": {
                    "label_format": (
                        "continuous" if task_family == "regression" else "single_index"
                    )
                },
            }
        )
        writer.write_service_metrics(
            service_id,
            {
                "metric_score": {
                    "value": 0.7,
                    "domain": "quality",
                    "direction": "higher_better",
                },
                "latency": {
                    "value": 0.1,
                    "domain": "latency",
                    "direction": "lower_better",
                },
            },
        )
        writer.write_service_generation_provenance(
            {
                "service_id": service_id,
                "provenance_version": 1,
                "architecture_id": "architecture-1",
                "parameter_schema_hash": schema_hash,
                "label_map_hash": "label-map-1",
                "preprocessing_contract_hash": "preprocessing-1",
                "logical_dataset_contract_hash": "dataset-contract-1",
                "lineage_id": shared["lineage_id"],
                "fl_run_id": shared["fl_run_id"],
                "client_id": client_id,
                "round_id": 1,
                "compatibility_contract_hash": contract_hash,
                "parent_global_round": 0,
                "parent_state_hash": parent.sha256,
                "participated": 1,
                "provenance_json": {},
            }
        )
        writer.write_fedavg_client_round(
            {
                "fl_run_id": shared["fl_run_id"],
                "client_id": client_id,
                "round_id": 1,
                "service_id": service_id,
                "scheduled": 1,
                "participated": 1,
                "succeeded": 1,
                "status": "success",
                "sample_count": samples,
                "parent_global_round": 0,
                "parent_state_hash": parent.sha256,
                "final_state_hash": patch.target_state_sha256,
                "compatibility_contract_hash": contract_hash,
            }
        )
        writer.write_service_checkpoint_storage(
            {
                "service_id": service_id,
                "lineage_id": shared["lineage_id"],
                "storage_policy_version": "fedavg_round_base_exact_patch_v1",
                "storage_mode": "shared_base_exact_patch",
                "state_hash_spec_version": TENSOR_STATE_FORMAT,
                "parameter_schema_hash": schema_hash,
                "base_state_sha256": parent.sha256,
                "target_state_sha256": patch.target_state_sha256,
                "reconstruction_artifact_ref": str(patch.manifest_path),
                "reconstruction_artifact_sha256": "c" * 64,
                "round_trip_status": "passed",
                "parent_global_round": 0,
                "explicit_base_state_ref": str(parent_ref.manifest_path),
            }
        )
    writer.finish()
    return db, root, parent


def _requests(task_family="classification"):
    return pd.DataFrame(
        [
            {
                "request_id": "r",
                "request_schema_version": "1.0.0",
                "composition_type": "collaborative",
                "functional_requirements": json.dumps({"task_family": task_family}),
                "qos_requirements": json.dumps({"min_accuracy": 0.0}),
                "hard_constraints": json.dumps({}),
                "composition_size": 2,
                "preference_weights": json.dumps({"accuracy": 1.0}),
                "constraint_penalties": json.dumps({"min_accuracy": 1.0}),
            }
        ]
    )


def _evaluation(tmp_path):
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    pd.DataFrame(
        [
            {
                "request_id": "r",
                "candidate_id": "selected",
                "service_ids": json.dumps(["service-a", "service-b"]),
                "selected_flag": True,
                "utility_score": 0.8,
                "mean_accuracy": 0.7,
            },
            {
                "request_id": "r",
                "candidate_id": "other",
                "service_ids": json.dumps(["service-b", "service-c"]),
                "selected_flag": False,
                "utility_score": 0.6,
                "mean_accuracy": 0.6,
            },
        ]
    ).to_csv(evaluation / "compositions.csv", index=False)
    pd.DataFrame(
        [
            {
                "request_id": "r",
                "method": "greedy",
                "service_ids": json.dumps(["service-b", "service-c"]),
                "utility_score": 0.9,
                "mean_accuracy": 0.75,
            }
        ]
    ).to_csv(evaluation / "search_results.csv", index=False)
    return evaluation


def _cohort_file(tmp_path, task_family="classification"):
    labels = [float(i) if task_family == "regression" else i % 2 for i in range(10)]
    payload = {
        "training_ids": ["train:0"],
        "metric_source_ids": [f"test:{i}" for i in range(10)],
        "metric_source_labels": labels,
        "contract": {"dataset": "fixture"},
    }
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_selected_fedavg_executor_reweights_subset_and_runs_one_model(tmp_path, monkeypatch):
    db, _root, _parent = _create_fedavg_fixture(tmp_path)
    plan = _plan()
    calls = []

    def fake_execute(*, db_path, representative_service_id, runtime_id, plan, state):
        del db_path
        calls.append((representative_service_id, runtime_id, state))
        np.testing.assert_allclose(state.entries[0].to_numpy(), [5.0])
        np.testing.assert_array_equal(state.entries[1].to_numpy(), [7])
        labels = [int(item.split(":")[1]) % 2 for item in plan.audit_ids]
        outputs = np.asarray(
            [[0.9, 0.1] if value == 0 else [0.1, 0.9] for value in labels]
        )
        return ServicePrediction(runtime_id, plan.audit_ids, outputs, 0.12, plan.label_hash)

    monkeypatch.setattr(fedavg_module, "execute_tensor_state_prediction", fake_execute)
    result = execute_selected_fedavg_composition(
        db_path=db, service_ids=["service-a", "service-b"], plan=plan
    )
    assert len(calls) == 1
    assert calls[0][0] == "service-a"
    assert result.aggregation_provenance["total_samples"] == 4
    weights = {
        item["service_id"]: item["normalized_weight"]
        for item in result.aggregation_provenance["contributors"]
    }
    assert weights == pytest.approx({"service-a": 0.25, "service-b": 0.75})
    assert "service-c" not in weights


def test_resolver_fails_closed_for_bad_weight_and_missing_checkpoint(tmp_path):
    db, _root, _parent = _create_fedavg_fixture(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE fedavg_client_rounds SET sample_count = 0 WHERE service_id = 'service-b'"
        )
    with pytest.raises(FedAvgProvenanceError, match="positive"):
        resolve_selected_fedavg_members(db, ["service-a", "service-b"])

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE fedavg_client_rounds SET sample_count = 3 WHERE service_id = 'service-b'"
        )
        conn.execute(
            "DELETE FROM service_checkpoint_storage WHERE service_id = 'service-b'"
        )
    with pytest.raises(FedAvgProvenanceError, match="checkpoint"):
        resolve_selected_fedavg_members(db, ["service-a", "service-b"])


def test_validate_materializes_only_selected_composition_and_persists_results(tmp_path, monkeypatch):
    db, _root, _parent = _create_fedavg_fixture(tmp_path)
    evaluation = _evaluation(tmp_path)
    cohort_path, payload = _cohort_file(tmp_path)
    plan = build_cohort_plan(
        training_ids=payload["training_ids"],
        metric_source_ids=payload["metric_source_ids"],
        metric_source_labels=payload["metric_source_labels"],
        contract=payload["contract"],
        audit_samples=4,
        minimum_audit_samples=4,
    )
    restored = []
    original_reconstruct = fedavg_module.reconstruct_service_state

    def track_reconstruct(root, service_id):
        restored.append(service_id)
        return original_reconstruct(root, service_id)

    def fake_execute(*, db_path, representative_service_id, runtime_id, plan, state):
        del db_path, state
        labels = [int(item.split(":")[1]) % 2 for item in plan.audit_ids]
        outputs = np.asarray(
            [[0.9, 0.1] if value == 0 else [0.1, 0.9] for value in labels]
        )
        return ServicePrediction(runtime_id, plan.audit_ids, outputs, 0.02, plan.label_hash)

    monkeypatch.setattr(fedavg_module, "reconstruct_service_state", track_reconstruct)
    monkeypatch.setattr(fedavg_module, "execute_tensor_state_prediction", fake_execute)
    output = tmp_path / "validation"
    frame, summary = validate_from_files(
        db_path=db,
        requests=_requests(),
        evaluation_dir=evaluation,
        request_id="r",
        output_dir=output,
        cohort_file=cohort_path,
        audit_seed=42,
        audit_samples=4,
        min_audit_samples=4,
    )
    row = frame.iloc[0]
    assert restored == ["service-a", "service-b"]
    assert row["validation_status"] == "completed"
    assert row["actual_accuracy"] == pytest.approx(1.0)
    assert row["selected_total_samples"] == 4
    assert pd.isna(row["ensemble_accuracy"])
    assert summary["scientific_validity"] == "mechanical_validation_only"
    assert summary["predicted_surrogate_utility"] == pytest.approx(0.8)
    assert (output / "aggregation_provenance.json").exists()
    with sqlite3.connect(db) as conn:
        stored = conn.execute(
            "SELECT validation_status, selected_total_samples, actual_accuracy, "
            "ensemble_accuracy FROM composition_validation_candidates"
        ).fetchone()
    assert stored == ("completed", 4, 1.0, None)


def test_search_and_evaluator_selection_resolution_is_unchanged(tmp_path):
    evaluation = _evaluation(tmp_path)
    evaluator = _read_selected_composition(
        evaluation,
        "r",
        selection_source="evaluator",
        search_method=None,
        candidate_id=None,
    )
    search = _read_selected_composition(
        evaluation,
        "r",
        selection_source="search",
        search_method="greedy",
        candidate_id=None,
    )
    assert evaluator["service_ids"] == ["service-a", "service-b"]
    assert search["service_ids"] == ["service-b", "service-c"]
    assert search["recorded_surrogate_utility"] == pytest.approx(0.9)


def test_retired_prediction_runtime_options_fail_with_migration_message(tmp_path):
    with pytest.raises(ValueError, match="retired"):
        validate_from_files(
            db_path=tmp_path / "unused.db",
            requests=_requests(),
            evaluation_dir=tmp_path,
            request_id="r",
            output_dir=tmp_path / "out",
            cohort_file=tmp_path / "cohort.json",
            audit_seed=42,
            audit_samples=4,
            min_audit_samples=4,
            predictions_file=tmp_path / "predictions.json",
        )


def test_execution_result_contract_carries_parameter_provenance():
    result = FedAvgCompositionExecution(
        outputs=np.asarray([1.0]),
        task_family="regression",
        label_hash="labels",
        inference_latency_s=0.1,
        aggregation_wall_s=0.2,
        aggregation_provenance={"global_state_hash": "state"},
    )
    assert result.aggregated_state_hash == "state"


def test_validation_status_migration_preserves_legacy_rows(tmp_path):
    db = tmp_path / "legacy-validation.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE composition_validation_runs (
              validation_run_id TEXT PRIMARY KEY,
              request_id TEXT NOT NULL,
              cohort_id TEXT,
              cohort_manifest_json TEXT,
              inputs_json TEXT,
              provenance_json TEXT,
              summary_json TEXT,
              created_at TEXT
            );
            CREATE TABLE composition_validation_candidates (
              validation_run_id TEXT NOT NULL,
              candidate_id TEXT NOT NULL,
              service_ids_json TEXT NOT NULL,
              validation_status TEXT NOT NULL,
              status_reason TEXT,
              recorded_surrogate_utility REAL,
              regenerated_surrogate_utility REAL,
              measured_utility REAL,
              ensemble_accuracy REAL,
              parallel_latency_s REAL,
              serial_latency_s REAL,
              qos_violations_json TEXT,
              selected_flag INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY (validation_run_id, candidate_id),
              CHECK (validation_status IN (
                'completed','unsupported','regeneration_failed','cohort_isolation_failed'
              ))
            );
            INSERT INTO composition_validation_runs (
              validation_run_id, request_id
            ) VALUES ('old-run', 'old-request');
            INSERT INTO composition_validation_candidates (
              validation_run_id, candidate_id, service_ids_json,
              validation_status, selected_flag
            ) VALUES ('old-run', 'old-candidate', '["old-service"]', 'completed', 1);
            """
        )

    writer = make_writer("sqlite", db_path=str(db))
    writer.start()
    writer.write_composition_validation_candidate(
        {
            "validation_run_id": "old-run",
            "candidate_id": "new-candidate",
            "service_ids_json": ["service-a", "service-b"],
            "validation_status": "checkpoint_failed",
            "selected_flag": True,
            "selected_total_samples": 4,
        }
    )
    writer.finish()

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT candidate_id, validation_status, selected_total_samples "
            "FROM composition_validation_candidates ORDER BY candidate_id"
        ).fetchall()
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='composition_validation_candidates'"
        ).fetchone()[0]
    assert rows == [
        ("new-candidate", "checkpoint_failed", 4),
        ("old-candidate", "completed", None),
    ]
    assert "checkpoint_failed" in schema
