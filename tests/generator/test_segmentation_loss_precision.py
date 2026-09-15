import math

import pytest
import torch

from mlaas_data_generator.models.adapters.hf_task import ImageSegmentationSpec


def test_dense_half_logits_use_finite_fp32_loss_outside_autocast():
    # More valid pixels than FP16's finite count range, without loading a model.
    logits = torch.zeros((2, 3, 256, 256), dtype=torch.float16, requires_grad=True)
    labels = torch.zeros((2, 256, 256), dtype=torch.long)
    loss = ImageSegmentationSpec().loss_fn(torch, logits, labels, {"ignore_index": 255})
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(math.log(3))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_all_ignored_resized_targets_have_explicit_supervision_error():
    logits = torch.zeros((1, 3, 2, 2), dtype=torch.float16)
    labels = torch.full((1, 4, 4), 255, dtype=torch.long)
    with pytest.raises(ValueError, match="insufficient_supervision"):
        ImageSegmentationSpec().loss_fn(torch, logits, labels, {"ignore_index": 255})


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_memory_metadata_survives_recipe_config_restoration(monkeypatch, fail):
    from mlaas_data_generator.services.runner import ServiceRunner
    from mlaas_data_generator.clients import training
    runner = ServiceRunner({"service_id": "metadata-only", "batch_size": 4})
    requested = runner.config
    monkeypatch.setattr(training, "realized_training_recipe", lambda cfg, n: (dict(cfg), None))
    def train(*args, **kwargs):
        runner.config["training_memory_metadata"] = {"oom_retry_count": 2}
        if fail:
            raise RuntimeError("synthetic failure")
        return {}
    monkeypatch.setattr(runner, "_train_model_effective", train)
    if fail:
        with pytest.raises(RuntimeError):
            runner._train_model(None, [1], [1], task_family="segmentation")
    else:
        runner._train_model(None, [1], [1], task_family="segmentation")
    assert runner.config is requested
    assert runner.config["training_memory_metadata"]["oom_retry_count"] == 2
    assert runner.config["batch_size"] == 4
