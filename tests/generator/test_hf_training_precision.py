"""Exercise AMP optimizer updates on tiny CPU tensors, without HF downloads."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlaas_data_generator.models.adapters.hf_core import HFCore


class TinyLossModel(torch.nn.Module):
    def __init__(self, bad_call=None, always_bad=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1., dtype=torch.float16))
        self.calls = 0
        self.bad_call = bad_call
        self.always_bad = always_bad

    def forward(self, x, labels):
        self.calls += 1
        logits = self.weight * x
        loss = ((logits - labels) ** 2).mean()
        if self.always_bad or self.calls == self.bad_call:
            loss = loss * float("inf")
        return SimpleNamespace(logits=logits, loss=loss)


def tiny_core(bad_call=None, always_bad=False):
    core = HFCore.__new__(HFCore)
    core.torch = torch
    core.device = torch.device("cpu")
    core.model = TinyLossModel(bad_call, always_bad)
    core.task_spec = SimpleNamespace(
        name="sequence_classification",
        encode_batch=lambda tokenizer, xs, ys, *args, **kw: (
            {"x": torch.tensor(xs, dtype=torch.float32)},
            torch.tensor(ys, dtype=torch.float32), {}),
        build_forward_inputs=lambda enc, labels_t, **kw: {**enc, "labels": labels_t},
        extract_loss=lambda torch, outputs, *args: outputs.loss,
    )
    core.model_id = "tiny-test"
    core.tokenizer = None
    core.batch_size = 2
    core.max_length = 8
    core.label_pad_value = -100
    core.weight_format = None
    core.tokenizer_load_s = core.model_load_s = 0.
    core.tokenizer_cache_hit = core.model_cache_hit = False
    core.autocast_enabled = True
    core.grad_scaler = torch.amp.GradScaler("cpu", init_scale=128.)
    core._make_autocast_context = lambda: torch.autocast("cpu", dtype=torch.float16)
    return core


@pytest.mark.parametrize("bad_call", [None, 2])
def test_half_loaded_model_updates_with_scaler_and_accumulation(bad_call):
    core = tiny_core(bad_call)
    result = core.finetune(np.ones(4), np.zeros(4), epochs=1, lr=.1,
                          optimizer="sgd", gradient_accumulation_steps=2,
                          max_train_time_s=None)
    assert core.model.weight.dtype == torch.float32
    assert core.model.weight.item() == pytest.approx(.8)
    assert result["successful_optimizer_step_count"] == 1
    assert result["skipped_optimizer_step_count"] == 0
    assert result["train_fp32_loss_retry_count"] == int(bad_call is not None)
    assert result["train_loss"] == pytest.approx(1.)


def test_nonfinite_fp32_retry_still_fails_without_update():
    core = tiny_core(always_bad=True)
    with pytest.raises(ValueError, match="nonfinite_training_loss"):
        core.finetune(np.ones(2), np.zeros(2), optimizer="sgd", lr=.1)
    assert core.model.calls == 2
    assert core.model.weight.item() == 1.


@pytest.mark.parametrize("failure", ["loss", "exception"])
def test_precision_retry_restores_batchnorm_statistics(failure):
    core = tiny_core()
    core.requested_precision_type = "fp16"
    core.model.bn = torch.nn.BatchNorm1d(1)
    original = core.model.forward

    def forward(x, labels):
        bad = core.autocast_enabled
        core.model.bn(x.reshape(-1, 1) * (float("inf") if bad else 1.))
        if bad and failure == "exception":
            raise RuntimeError("unsupported float16 operation")
        result = original(x, labels)
        if bad:
            result.loss = result.loss * float("inf")
        return result

    core.model.forward = forward
    core.finetune(np.ones(2), np.zeros(2), optimizer="sgd", lr=.1)
    bn = core.model.bn
    assert bn.num_batches_tracked.item() == 1
    assert bn.running_mean.item() == pytest.approx(.1)
    assert bn.running_var.item() == pytest.approx(.9)
    assert torch.isfinite(bn(torch.ones(2, 1))).all()


def test_nonfinite_amp_latches_fp32_without_losing_accumulated_gradients():
    core = tiny_core()
    original = core.model.forward
    def forward(*args, **kwargs):
        result = original(*args, **kwargs)
        # Fail after the first batch has already accumulated scaled gradients.
        if core.autocast_enabled and core.model.calls >= 2:
            result.loss = result.loss * float("inf")
        return result
    core.model.forward = forward
    result = core.finetune(np.ones(8), np.zeros(8), epochs=1, lr=.1,
                          optimizer="sgd", gradient_accumulation_steps=2,
                          max_train_time_s=None)
    assert result["train_fp32_loss_retry_count"] == 1
    assert result["successful_optimizer_step_count"] == 2
    assert core.model.calls == 5  # Four batches, only one retried forward.
    assert core.model.weight.item() == pytest.approx(.64)
    assert not core.autocast_enabled
    assert core.effective_precision_type == "fp32"
    assert core._run_with_precision_context(lambda: core.autocast_enabled) is False


@pytest.mark.parametrize("model_id,family", [
    ("google/flan-t5-small", None), ("google/byt5-small", None),
    ("local-checkpoint", "t5"), ("google/mt5-small", "mt5"),
])
def test_t5_policy_applies_before_cuda_and_to_manifest_config(model_id, family):
    from mlaas_data_generator.precision_policy import apply_precision_policy, T5_FP32_REASON
    core = tiny_core()
    core.model_id = model_id
    core.model.config = SimpleNamespace(model_type=family)
    core.device = torch.device("cuda")  # No GPU call should be needed for this policy.
    core.mixed_precision, core.precision_type = True, "fp16"
    core._configure_precision_mode()
    assert core.grad_scaler is None and not core.autocast_enabled
    assert core.precision_fallback_reason == T5_FP32_REASON
    config = apply_precision_policy({"hf_model_id": model_id, "mixed_precision": True,
                                     "precision_type": "fp16"}, model_family=family)
    assert config["precision_type"] == "fp32" and not config["mixed_precision"]


@pytest.mark.parametrize("mixed,precision", [(False, "fp16"), (True, "fp32"), (True, "bf16")])
def test_precision_configuration_normalizes_loaded_weights(mixed, precision):
    core = tiny_core()
    core.mixed_precision = mixed
    core.precision_type = precision
    core._configure_precision_mode()
    assert core.model.weight.dtype == torch.float32
    assert core.effective_precision_type == "fp32"  # CPU policy
    assert core.grad_scaler is None


@pytest.mark.parametrize("fail", [False, True])
def test_bart_generation_preserves_attention_and_restores_cache(monkeypatch, fail):
    from mlaas_data_generator.models.adapters import hf_core
    monkeypatch.setattr(hf_core.os, "name", "nt")
    core = tiny_core()
    core.device = torch.device("cuda")
    cfg = SimpleNamespace(model_type="bart", _attn_implementation="sdpa", use_cache=False)
    core.model.config = cfg
    changes = []
    def change(value):
        changes.append(value)
        cfg._attn_implementation = value
    core.model.set_attn_implementation = change
    try:
        with core._generation_inference_mode():
            assert cfg._attn_implementation == "sdpa"
            assert cfg.use_cache
            if fail:
                raise ValueError("probe failed")
    except ValueError:
        assert fail
    assert changes == []
    assert cfg._attn_implementation == "sdpa" and not cfg.use_cache
