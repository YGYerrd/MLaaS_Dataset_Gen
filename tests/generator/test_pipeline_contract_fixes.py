from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from mlaas_data_generator.task_contract import resolve_sentence_pair_contract
from mlaas_data_generator.services.taxonomy import canonical_task_family, canonical_metric_names
from mlaas_data_generator.models.adapters.hf_adapter import build_task_spec
from mlaas_data_generator.models.adapters.hf_core import HFCore
from mlaas_data_generator.clients import recipes, training
from mlaas_data_generator.clients.capacity import CAPACITY_POLICY, known_training_capacity
from mlaas_data_generator.clients.partitioning import client_partition, PARTITION_VERSION
from mlaas_data_generator.cli.manifest.hf_manifest_builder import build_hf_manifest
from mlaas_data_generator.precision_policy import FP32_REASON


@pytest.mark.parametrize("subset", ["mrpc", "wnli", "stsb"])
def test_sentence_pair_contract_agrees_with_head_loss_and_metrics(subset):
    import torch
    regression = subset == "stsb"
    cfg = {"dataset": "hf", "dataset_name": "glue", "dataset_config": subset,
           "hf_task": "sentence_similarity", "task": "sentence_similarity", "task_type": "regression"}
    resolve_sentence_pair_contract(cfg)
    assert cfg["task_type"] == ("regression" if regression else "classification")
    family = canonical_task_family(cfg["task_type"], cfg["hf_task"])
    assert family == cfg["task_type"]
    assert canonical_metric_names(family, hf_task=cfg["hf_task"]) == (("pearson", "spearman") if regression else ("accuracy", "f1"))
    _, spec = build_task_spec(cfg["hf_task"], num_labels=cfg["num_labels"], label_format=cfg["label_format"])
    captured = {}
    class Model:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured.update(kwargs)
            return object()
    spec.build_model(SimpleNamespace(AutoModelForSequenceClassification=Model), "local-test", cfg["num_labels"])
    assert captured["num_labels"] == (1 if regression else 2)
    assert captured["problem_type"] == ("regression" if regression else "single_label_classification")
    logits = torch.tensor([[0.2], [0.8]], requires_grad=True) if regression else torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
    targets = torch.tensor([0., 1.]) if regression else torch.tensor([0, 1])
    loss = spec.loss_fn(torch, logits, targets, {})
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    metrics = spec.metrics(targets.numpy(), spec.preds_from_logits(torch, logits.detach(), {}).numpy())
    if regression:
        assert set(metrics["named_metrics"]) == {"pearson", "spearman"}
    else:
        assert metrics["primary"] == 1.0 and metrics["secondary"] == 1.0


def test_disagreement_with_loaded_label_schema_is_rejected_before_model_build():
    cfg = {"dataset_name": "glue", "dataset_config": "mrpc", "hf_task": "sentence_similarity"}
    with pytest.raises(ValueError, match="preprocessing produced regression"):
        resolve_sentence_pair_contract(cfg, {"task_type": "regression"})


def test_pair_classlabel_indices_preserve_dataset_meaning_and_missing_train_classes():
    from datasets import ClassLabel
    from mlaas_data_generator.data.preprocessors.hf_text_similarity import _encode_categorical_targets
    feature = ClassLabel(names=["not_equivalent", "equivalent"])
    train, test, count, mapping = _encode_categorical_targets([1, 1], [0, 1], feature)
    assert train.tolist() == [1, 1] and test.tolist() == [0, 1]
    assert count == 2 and mapping == {"not_equivalent": 0, "equivalent": 1}
    with pytest.raises(ValueError, match="outside the dataset"):
        _encode_categorical_targets([0, -1], [1], feature)


@pytest.mark.parametrize("model_id,family", [
    ("google/mobilenet_v2_1.0_224", "mobilenet_v2"), ("google/efficientnet-b0", "efficientnet"),
    ("local-checkpoint", "mobilenet_v2"),
    ("apple/mobilevit-small", "mobilevit"), ("google/mobilebert-uncased", "mobilebert"),
    ("local-checkpoint", "mobilevit"), ("local-checkpoint", "mobilebert"),
])
def test_fp32_guard_overrides_old_requests_and_restores_float_weights(model_id, family):
    import torch
    core = HFCore.__new__(HFCore)
    core.model_id = model_id
    core.model = torch.nn.Linear(4, 2).half()
    core.model.config = SimpleNamespace(model_type=family)
    core.device = torch.device("cuda")  # The guard must run before any CUDA/AMP call.
    core.mixed_precision = True
    core.precision_type = "fp16"
    core._configure_precision_mode()
    assert not core.autocast_enabled and core.grad_scaler is None
    assert core.effective_precision_type == "fp32" and core.precision_fallback_reason == FP32_REASON
    assert all(p.dtype == torch.float32 for p in core.model.parameters())
    core.model(torch.ones(2, 4)).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in core.model.parameters())
    core.mixed_precision = True
    core._configure_precision_mode()
    assert not core.autocast_enabled


def test_new_image_manifests_freeze_fp32_for_affected_families():
    df = build_hf_manifest(task_keys=["image_classification"], models_per_task=20,
                          clients_per_family=4, average_samples=1000, seed=42)
    selected = df[df.hf_model_id.str.contains("mobilenet|efficientnet|mobilevit")]
    assert selected.hf_model_id.nunique() == 3
    assert not selected.mixed_precision.any()
    assert set(selected.precision_type) == {"fp32"}


def test_synthetic_manifest_scales_distribution_instead_of_clipping_to_full_pool():
    df = build_hf_manifest(task_keys=["tabular_regression"], clients_per_family=100,
                          average_samples=1200, seed=42)
    rows = df[df.dataset_name == "synthetic"]
    assert known_training_capacity({"dataset": "synthetic"}) == 1600
    assert rows.sample_size.max() < 1600
    for _, group in rows.groupby("model_type"):
        assert group.sample_size.nunique() == 100
        assert set(group.max_client_quantity) == {1600}
        assert (group.unconstrained_sample_size > group.sample_size).all()
        for row in group.to_dict("records"):
            import math
            row = {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()}
            training.validate_training_recipe(row)


def test_shard_quantity_relaxes_exhausted_support_to_honor_requested_count():
    y = np.array([0] * 800 + [1] * 800)
    sizes = []
    for i, quota in enumerate((900, 1200, 1500)):
        cfg = {"partition_generator_version": PARTITION_VERSION, "split_strategy": "shard",
               "sample_size": quota, "requested_quota": quota, "max_client_quantity": 1750,
               "quantity_capacity_policy": CAPACITY_POLICY, "heterogeneity_intensity": 1.0,
               "sample_seed": i, "client_partition_index": i, "shard_support_quantile": 0.5, "skew_axis": "class_label"}
        part = client_partition(np.arange(len(y))[:, None], y, cfg, meta={}, task_family="classification", source_ids=list(map(str, range(len(y)))))
        assert len(part["indices"]) == len(set(part["indices"])) == quota
        assert part["metadata"]["support_relaxation_count"] == quota - 800
        sizes.append(len(part["indices"]))
    assert len(set(sizes)) == 3


def test_realized_training_adjusts_work_without_mutating_frozen_recipe():
    policy = training.build_training_policy({"config": {"training_regime": "generic", "model_type": "mlp", "task_type": "regression"}})
    row = recipes.sample_client_recipes(policy, count=1, seed="capacity-test", average_samples=1500)[0]
    original = deepcopy(row)
    effective, evidence = training.realized_training_recipe(row, 900)
    assert row == original
    assert evidence["effective_planned_optimizer_steps"] >= evidence["target_optimizer_steps"]
    assert effective["training_epochs"] > row["training_epochs"]
    assert effective["sample_size"] == 900
    assert effective["max_train_time_s"] == row["max_train_time_s"]
    assert (effective, evidence) == training.realized_training_recipe(row, 900)
    training.validate_training_recipe(row)


def test_runner_uses_adjusted_training_budget_and_retains_requested_config(tmp_path, monkeypatch):
    from mlaas_data_generator.services.runner import ServiceRunner
    policy = training.build_training_policy({"config": {"training_regime": "generic", "model_type": "mlp", "task_type": "regression"}})
    row = recipes.sample_client_recipes(policy, count=1, seed="runtime-capacity", average_samples=1500)[0]
    instance = ServiceRunner({**row, "model_type": "mlp", "service_id": "adjusted", "db_path": str(tmp_path / "services.db")})
    frozen = deepcopy(instance.config)
    observed = {}
    def fit(model, x, y, **kwargs):
        observed.update(instance.config)
        return {"optimizer_step_count": 10}
    monkeypatch.setattr(instance, "_train_model_effective", fit)
    metrics = instance._train_model(None, np.ones((900, 2)), np.arange(900), task_family="regression")
    assert observed["sample_size"] == 900
    assert observed["training_epochs"] > frozen["training_epochs"]
    assert instance.config == frozen
    assert metrics["training_budget_adjustment"]["effective"]["sample_size"] == 900
