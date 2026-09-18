import json
import numpy as np
import pytest

from mlaas_data_generator.task_capabilities import resolve_capabilities, capability_matrix, CAPABILITIES
from mlaas_data_generator.tasks import TaskContract, TaskContractError, resolve_task_contract
from mlaas_data_generator.tasks.contracts import target_fingerprints, validate_partition
from mlaas_data_generator.tasks.runtime import ConfusionAccumulator, TaskEvaluation
from mlaas_data_generator.shared_parent import reserve_evaluation
from mlaas_composition.validation_cli import _read_cohort, _read_cohort_labels
from mlaas_composition.validation import measure_model_outputs


def config(task="token_classification"):
    return {"model_type": "hf_finetune", "hf_task": task, "task_type": "classification",
            "hf_model_id": "fixture/model", "num_labels": 3, "max_length": 32}


def test_capability_matrix_has_independent_answers_and_pending_runtime():
    rows = list(capability_matrix())
    assert rows
    assert all(set(row["capabilities"]) == set(CAPABILITIES) for row in rows)
    decisions = resolve_capabilities(config())
    assert decisions["training"]["support"] == "supported"
    assert decisions["measured_composition_validation"]["support"] == "experimental"
    assert decisions["checkpoint_restoration"]["decision"] == "pending"
    bad = resolve_capabilities(config(), checks={"checkpoint_present": False})
    assert bad["checkpoint_restoration"]["support"] == "supported"
    assert bad["checkpoint_restoration"]["reason_code"] == "checkpoint_missing"
    assert bad["parameter_aggregation"]["reason_code"] == "checkpoint_missing"
    assert resolve_capabilities(config(), checks={"parent": False})["parameter_aggregation"]["reason_code"] == "parent_mismatch"


def test_unknown_and_backend_mismatch_are_explicit():
    assert resolve_capabilities(config("unregistered"))["training"]["reason_code"] == "unknown_task_contract"
    assert resolve_capabilities({**config(), "model_type": "mlp"})["measured_composition_validation"]["reason_code"] == "backend_not_implemented"
    assert resolve_capabilities({**config(), "training_regime": "inference_only"})["training"]["reason_code"] == "inference_only"


def test_contract_hashes_bind_semantics_not_client_quantity():
    first = resolve_task_contract(config())
    assert TaskContract.from_dict(json.loads(json.dumps(first.as_dict()))) == first
    assert resolve_task_contract({**config(), "sample_size": 99, "training_seed": 123}).semantics_hash == first.semantics_hash
    assert resolve_task_contract({**config(), "max_length": 12}).semantics_hash != first.semantics_hash
    with pytest.raises(TaskContractError, match="hash"):
        TaskContract.from_dict({**first.as_dict(), "task_semantics_hash": "tampered"})
    with pytest.raises(TaskContractError, match="loaded dataset"):
        resolve_task_contract(config(), {"task_type": "regression"})
    with pytest.raises(TaskContractError, match="declared HF task"):
        resolve_task_contract(config(), {"hf_task": "image_classification", "task_type": "classification"})
    with pytest.raises(TaskContractError, match="label mappings"):
        resolve_task_contract({**config(), "label_mapping": {"O": 0, "B": 1}}, {"label_mapping": {"B": 0, "O": 1}})


def test_contract_resolution_does_not_materialize_runtime_payloads():
    class RuntimeArray(np.ndarray):
        def tolist(self):
            raise AssertionError("runtime data must not be serialized into a task contract")
    payload = np.zeros((2, 3)).view(RuntimeArray)
    original = config("image_segmentation")
    expected = resolve_task_contract(original).as_dict()
    actual = resolve_task_contract({**original, "_prepared_dataset": ({"pixel_values": payload}, payload),
                                    "_parent_state": payload, "unused_runtime_statistic": payload},
                                   {"unused_source_rows": payload}).as_dict()
    assert actual == expected


def test_contract_resolution_does_not_mutate_sentence_pair_dataset_args():
    args = {"label_mode": "old"}
    value = {"model_type": "hf_finetune", "hf_task": "sentence_similarity",
             "dataset_name": "glue", "dataset_config": "mrpc", "dataset_args": args}
    result = resolve_task_contract(value)
    assert result.family == "classification"
    assert args == {"label_mode": "old"}


@pytest.mark.parametrize("missing", [float("nan"), np.float32("nan")])
def test_contract_ignores_missing_scalar_fields_without_changing_identity(missing):
    cfg = {"model_type": "hf_finetune", "hf_task": "image_classification",
           "task_type": "classification", "num_labels": 3}
    expected = resolve_task_contract(cfg).as_dict()
    actual = resolve_task_contract({**cfg, "max_length": missing, "label_pad_value": missing,
                                    "task_tag": missing, "source_max_length": missing},
                                   {"num_labels": missing, "task_type": missing}).as_dict()
    assert actual == expected


def test_mixed_text_image_manifest_exports_preflight_and_roundtrips(tmp_path):
    from mlaas_data_generator.cli.manifest.hf_manifest_builder import build_hf_manifest, save_manifest
    from mlaas_data_generator.cli.run_manifest import load_manifest, _resolve_row, _validate_row
    frame = build_hf_manifest(task_keys=["text_classification", "image_classification"],
                              models_per_task=1, datasets_per_model=1, clients_per_family=1,
                              manifest_profile="test", seed=42)
    image_rows = frame[frame.hf_task == "image_classification"]
    assert image_rows.max_length.isna().all()
    path = tmp_path / "mixed.csv"
    save_manifest(frame, path)
    report = json.loads(path.with_suffix(".preflight.json").read_text())
    assert {r["hf_model_id"] for r in report["task_capabilities"]} == set(frame.hf_model_id)
    assert all(r["capabilities"]["training"]["support"] == "supported" for r in report["task_capabilities"])
    restored, defaults = load_manifest(path)
    assert set(restored.service_id) == set(frame.service_id)
    for _, row in restored.iterrows():
        assert _validate_row(_resolve_row(row, defaults)).ok


def test_structured_evaluation_retains_existing_prediction_counts(monkeypatch):
    from types import SimpleNamespace
    from mlaas_data_generator.services.runner import ServiceRunner
    from mlaas_data_generator.tasks import runtime
    contract = resolve_task_contract(config("image_segmentation"))
    metrics = {"loss": .5, "iou": .4, "dice": .6,
               "per_class": {"0": {"predicted": 7, "support": 5}, "1": {"predicted": 3, "support": 5}}}
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(metrics=metrics)
    monkeypatch.setattr(runtime, "evaluate_task_model", evaluate)
    model = SimpleNamespace(core=SimpleNamespace(evaluation_memory_metadata={}))
    result = ServiceRunner({"service_id": "fixture", "task_contract": contract.as_dict()})._evaluate_model(
        model, np.zeros((2, 1)), np.zeros((2, 1)), inference_only=False, task_family="segmentation")
    assert result[:3] == (.5, .4, .6)
    assert result[3]["metric_stat_class_0_pred_total"] == 7
    assert result[3]["metric_stat_class_1_target_total"] == 5
    assert len(calls) == 1


@pytest.mark.parametrize("task", ["image_segmentation", "seq2seq_generation"])
def test_structured_profiles_reject_unimplemented_sharding(task):
    contract = resolve_task_contract({**config(task), "task_tag": "summarization"})
    with pytest.raises(TaskContractError, match="partition_axis_unavailable"):
        validate_partition(contract, {"split_strategy": "shard"})
    validate_partition(contract, {"split_strategy": "iid"})


def test_supervision_and_alignment_checks():
    contract = resolve_task_contract(config())
    assert contract.supervision_count([[-100, 0, 2], [1, -100]]) == 3
    for targets in ([[-100, -100]], [[-1]], [[3]], [[1.5]]):
        with pytest.raises(TaskContractError):
            contract.supervision_count(targets)
    acc = ConfusionAccumulator()
    with pytest.raises(TaskContractError, match="alignment"):
        acc.update([0, 1], [0])
    with pytest.raises(TaskContractError, match="insufficient_audit_support"):
        acc.finalise()


def test_corpus_metrics_are_independent_of_batch_boundaries():
    truth, prediction = np.array([0, 1, 1, -100]), np.array([0, 0, 1, 2])
    whole, split = ConfusionAccumulator(), ConfusionAccumulator()
    whole.update(truth, prediction)
    split.update(truth[:1], prediction[:1])
    split.update(truth[1:], prediction[1:])
    assert whole.finalise() == split.finalise()
    metrics = whole.finalise()
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["iou"] == pytest.approx(1 / 2)
    assert metrics["dice"] == pytest.approx(2 / 3)
    assert metrics["supervision_count"] == 3
    assert "2" not in metrics["per_class"]


def test_structured_reservation_hashes_and_groups_roundtrip(tmp_path):
    targets = [np.array([i % 2, -100]) for i in range(6)]
    contract = resolve_task_contract(config())
    frozen = reserve_evaluation(["train:0"], [f"test:{i}" for i in range(6)], targets,
        task_family="token_classification", seed=42,
        contract={"task_semantics_hash": contract.semantics_hash, "evaluation_protocol_hash": contract.protocol_hash},
        source_groups={"train": ["train-doc"], "test": ["a", "a", "b", "b", "c", "c"]})
    assert frozen["version"] == "reserved_evaluation_v2"
    assert frozen["metric_source_labels"] == target_fingerprints(targets)
    for pair in ((0, 1), (2, 3), (4, 5)):
        assert (pair[0] in frozen["audit_positions"]) == (pair[1] in frozen["audit_positions"])
    path = tmp_path / "reservation.json"
    path.write_text(json.dumps(frozen))
    plan = _read_cohort(path, seed=1, audit_samples=1, minimum=1, task_family="token_classification")
    labels = _read_cohort_labels(path, plan, task_family="token_classification")
    result = TaskEvaluation(plan.audit_ids, tuple(labels), contract.as_dict(),
        {"f1": 0.5, "accuracy": 0.75, "quality_score": 0.5, "supervision_count": len(labels)}, 0.1)
    assert measure_model_outputs(result, plan, labels, task_family="token_classification", inference_latency_s=0.1)["quality_score"] == 0.5
    with pytest.raises(TaskContractError, match="audit"):
        result.validated_metrics(source_ids=plan.audit_ids, target_labels=[{"target_sha256": "changed"}] * len(labels), task_family="token_classification")
    frozen["metric_source_labels"][0] = {"target_sha256": "changed"}
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="hash mismatch"):
        _read_cohort(path, seed=1, audit_samples=1, minimum=1, task_family="token_classification")


def test_reservation_rejects_source_group_leakage():
    with pytest.raises(TaskContractError, match="audit_group_overlap"):
        reserve_evaluation(["train:0"], ["test:0", "test:1"], [[0], [1]], task_family="token_classification", seed=1,
            contract={}, source_groups={"train": ["a"], "test": ["a", "b"]})


def test_recipe_profiles_come_from_the_frozen_task_contract():
    from mlaas_data_generator.clients.training import build_training_policy
    from mlaas_data_generator.tasks.contracts import uses_task_protocol
    cfg = {**config("seq2seq_generation"), "task_type": "generation", "task_tag": "summarization", "training_regime": "finetune_transfer"}
    cfg["task_contract"] = resolve_task_contract(cfg).as_dict()
    policy = build_training_policy({"config": cfg})
    assert policy["partition_profiles"] == ["iid"]
    assert not uses_task_protocol({**cfg, "task_tag": "translation"})


def test_noncontiguous_segmentation_labels_and_absent_classes():
    contract = resolve_task_contract({**config("image_segmentation"), "num_labels": 10, "label_pad_value": 255})
    assert contract.supervision_count([np.array([[2, 7], [255, 7]])]) == 3
    acc = ConfusionAccumulator()
    acc.update([[2, 7], [255, 7]], [[2, 2], [9, 7]], ignore_index=255)
    assert set(acc.finalise()["per_class"]) == {"2", "7"}
    assert acc.finalise()["iou"] == pytest.approx(0.5)
    with pytest.raises(TaskContractError, match="insufficient_supervision"):
        contract.supervision_count([np.full((2, 2), 255)])


def test_preprocessing_identity_roundtrips_through_csv_parser():
    from mlaas_data_generator.cli.run_manifest import _coerce_by_column
    identity = {"max_length": 32, "input_shape": [32], "schema": None}
    assert _coerce_by_column("preprocessing_identity", json.dumps(identity)) == identity


def test_new_task_protocol_never_borrows_legacy_calibration():
    from mlaas_data_generator.clients.calibration import attach_calibration
    policy = {"task": "token_classification"}
    attach_calibration(policy, {**config(), "task_contract_version": "task_contract_v1"}, evidence={})
    assert policy["calibration"]["status"] == "uncalibrated"
    assert "evaluation protocol" in policy["calibration"]["reason"]
