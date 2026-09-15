"""Reference exposure, class coverage and measured updates are separate checks."""
from copy import deepcopy

import numpy as np
import pytest

from mlaas_data_generator.clients import recipes, training
from mlaas_data_generator.clients.partitioning import client_partition
from mlaas_data_generator.clients.reviewed_opportunity import runtime_evidence, VIT_SVHN_REFERENCE


def policy():
    return training.build_training_policy({"config": {
        "hf_model_id": "google/vit-base-patch16-224", "dataset_name": "ufldl-stanford/svhn",
        "dataset_config": "cropped_digits", "hf_task": "image_classification",
        "task": "image_classification", "training_regime": "finetune_transfer"}}, task="image_classification")


def sampled(average=600):
    return recipes.sample_client_recipes(policy(), count=100, average_samples=average, seed="vit-svhn-test")


def supported(average=600):
    return next(row for row in sampled(average) if row.get("calibration_band") == "supported")


@pytest.mark.parametrize("average", [256, 600, 1200])
def test_quantities_vary_and_epochs_scale_to_reference_without_fixed_730_or_six(average):
    rows = sampled(average)
    assert sum(row["sample_size"] for row in rows) == average * 100
    assert len({row["sample_size"] for row in rows}) == 100
    good = [row for row in rows if row.get("calibration_band") == "supported"]
    assert len(good) == 70
    assert len({row["training_epochs"] for row in good}) > 1
    assert any(row["training_epochs"] != 6 for row in good)
    assert any(row["sample_size"] < 730 for row in good)
    for row in rows:
        training.validate_training_recipe(row)
        if row.get("calibration_band") == "supported":
            assert row["sample_size"] * row["training_epochs"] >= 4380
            assert training.planned_optimizer_steps(row, row["sample_size"]) >= 552
    assert {r["calibration_band"] for r in rows} == {"supported", "challenging", "stress"}
    assert any(r["training_epochs"] < 6 for r in rows if r["calibration_band"] == "stress")


@pytest.mark.parametrize("strategy", ["dirichlet", "shard"])
def test_supported_partitions_represent_all_source_classes(strategy):
    row = {**supported(), "sample_size": 200, "requested_quota": 200,
           "max_client_quantity": 200, "split_strategy": strategy, "heterogeneity_intensity": 1.,
           "shard_support_quantile": .99, "hf_task": "image_classification", "skew_axis": "class_label"}
    labels = np.repeat(np.arange(10), np.arange(1, 11) * 100)
    ids = list(map(str, range(len(labels))))
    partition = client_partition(np.arange(len(labels)), labels, row,
                                 source_ids=ids, meta={}, task_family="classification")
    assert partition["metadata"]["represented_bucket_count"] == 10
    assert len(partition["indices"]) == 200
    with pytest.raises(ValueError, match="source is missing|axis is unavailable"):
        client_partition(np.arange(100), np.zeros(100, dtype=int), row,
                         source_ids=ids[:100], meta={}, task_family="classification")


def test_realized_quantity_changes_epochs_and_old_policy_keeps_its_budget():
    row = supported()
    count = 100
    effective, change = training.realized_training_recipe(row, count)
    assert change and effective["sample_size"] == count
    assert count * effective["training_epochs"] >= 4380
    assert training.planned_optimizer_steps(effective, count) >= 552
    assert effective["max_train_time_s"] == row["max_train_time_s"]
    old = deepcopy(row)
    old["training_policy"].pop("reviewed_training_opportunity")
    old["training_policy"].pop("optimizer_step_accounting")
    old["sample_size"] = 256
    old["training_epochs"] = 3
    old.update(batch_size=8, gradient_accumulation_steps=1)
    assert training.planned_optimizer_steps(old, 256) == 96
    training.validate_training_opportunity(old)
    assert runtime_evidence(old, {}, 256) == {}


def test_real_hf_loop_counts_partial_accumulation_windows_and_successful_updates(monkeypatch, request):
    import torch
    from transformers import ViTConfig, ViTForImageClassification
    from mlaas_data_generator.models.adapters.hf_core import HFCore
    from mlaas_data_generator.models.adapters.hf_task import ImageClassificationSpec
    previous_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    torch.set_num_threads(1)
    cfg = ViTConfig(image_size=8, patch_size=4, hidden_size=8, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=16, num_labels=10)
    monkeypatch.setattr(ImageClassificationSpec, "build_model",
                        lambda self, transformers, model_id, labels: ViTForImageClassification(cfg))
    core = HFCore("local-test", num_labels=10, task_spec=ImageClassificationSpec(), batch_size=8,
                  device="cpu", mixed_precision=False)
    row = dict(training_epochs=2, batch_size=8, gradient_accumulation_steps=2,
               training_policy={"optimizer_step_accounting": "flush_partial_accumulation_v1"})
    # 17 samples => 3 batches => 2 update windows per epoch, including the tail.
    result = core.finetune({"pixel_values": np.ones((17, 3, 8, 8), dtype=np.float32)},
                          np.arange(17) % 10, epochs=2, gradient_accumulation_steps=2,
                          lr=1e-4, max_train_time_s=None)
    assert result["planned_optimizer_step_count"] == training.planned_optimizer_steps(row, 17) == 4
    assert result["optimizer_step_count"] == result["successful_optimizer_step_count"] == 4
    assert result["train_samples"] == 34


@pytest.mark.parametrize("successful,presentations,stopped,status", [
    (550, 4380, False, "met"), (549, 4380, False, "shortfall"),
    (550, 4379, False, "shortfall"), (100, 800, True, "resource_limited"),
    (None, None, False, "unmeasured"),
])
def test_runner_reports_actual_budget_shortfall_and_checks_realized_coverage(monkeypatch, successful, presentations, stopped, status):
    from mlaas_data_generator.services.runner import ServiceRunner
    row = supported()
    row.update(split_strategy="dirichlet", skew_axis="class_label")
    runner = ServiceRunner({**row, "hf_model_id": "google/vit-base-patch16-224",
                            "dataset_name": "ufldl-stanford/svhn"})
    metrics = {"successful_optimizer_step_count": successful, "train_samples": presentations,
               "train_stopped_early": stopped}
    monkeypatch.setattr(runner, "_train_model_effective", lambda *a, **kw: dict(metrics))
    n = row["sample_size"]
    result = runner._train_model(None, np.zeros(n), np.arange(n) % 10, task_family="classification")
    assert result["training_opportunity_status"] == status
    assert result["training_opportunity_met"] is (status == "met")
    with pytest.raises(ValueError, match="all ten source classes"):
        runner._train_model(None, np.zeros(n), np.zeros(n), task_family="classification")


def test_reference_values_match_preserved_review_evidence():
    import hashlib
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "outputs/family_review_20260913/service_results.json"
    if not path.is_file():
        pytest.skip("local review evidence is not distributed with the package")
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == VIT_SVHN_REFERENCE["source_results_sha256"]
    rows = [r for r in json.loads(payload) if r["service_id"] in VIT_SVHN_REFERENCE["source_services"]]
    assert len(rows) == 3
    for row in rows:
        m = row["metrics"]
        assert row["quality"] == "healthy"
        assert sum(json.loads(m["train_distribution_json"]).values()) == 730
        assert m["training_epochs"] == 6
        assert m["optimizer_step_count"] == 552 and m["successful_optimizer_step_count"] == 550


def test_final_generated_roster_keeps_budgets_after_quantity_reassignment():
    from mlaas_data_generator.cli.manifest.hf_manifest_builder import build_hf_manifest
    from mlaas_data_generator.cli.run_manifest import _resolve_row
    frame = build_hf_manifest(task_keys=["image_classification"], models_per_task=20,
        datasets_per_model=20, clients_per_family=12, average_samples=600, seed=42)
    rows = [_resolve_row(row, {}) for _, row in frame[(frame.hf_model_id == "google/vit-base-patch16-224") &
                 (frame.dataset_name == "ufldl-stanford/svhn")].iterrows()]
    assert len(rows) == 12
    assert sum(row["sample_size"] for row in rows) == 7200
    for row in rows:
        training.validate_training_recipe(row)
        if row["calibration_band"] == "supported":
            assert row["sample_size"] * row["training_epochs"] >= 4380
            assert training.planned_optimizer_steps(row, row["sample_size"]) >= 552
    others = frame[(frame.hf_model_id == "google/vit-base-patch16-224") &
                   (frame.dataset_name != "ufldl-stanford/svhn")]
    assert len(others)
    assert all("reviewed_training_opportunity" not in p for p in others.training_policy)


def test_review_reference_does_not_apply_to_trained_parent_clients():
    parent = {"config": {"hf_model_id": "google/vit-base-patch16-224", "dataset_name": "ufldl-stanford/svhn",
        "dataset_config": "cropped_digits", "task": "image_classification"}, "parent_state_hash": "trained"}
    assert "reviewed_training_opportunity" not in training.build_training_policy(parent)
