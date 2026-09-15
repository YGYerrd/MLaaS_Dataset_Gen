"""DPT layout guard: scope, numerical behavior, and reused model lifecycle."""
import copy
import io
from types import SimpleNamespace

import pytest
import torch

from mlaas_data_generator.models.adapters import rocm_dpt_layout as layout


def model():
    result = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1),
                                 torch.nn.ConvTranspose2d(4, 3, 2, stride=2))
    result.config = SimpleNamespace(model_type="dpt")
    return result


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(layout.sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    monkeypatch.delenv("MLAAS_ROCM_DPT_CONTIGUOUS_CONV", raising=False)


@pytest.mark.parametrize("change", ["linux", "nvidia", "cpu", "other_model", "disabled"])
def test_scopes_workaround_to_windows_rocm_dpt(rocm, monkeypatch, change):
    net = model()
    device = "cuda:0"
    if change == "linux":
        monkeypatch.setattr(layout.sys, "platform", "linux")
    elif change == "nvidia":
        monkeypatch.setattr(torch.version, "hip", None)
    elif change == "cpu":
        device = "cpu"
    elif change == "other_model":
        net.config.model_type = "segformer"
    else:
        monkeypatch.setenv("MLAAS_ROCM_DPT_CONTIGUOUS_CONV", "0")
    assert layout.configure_dpt_convolution_layout(net, torch, device) == 0
    assert not any(m._forward_pre_hooks for m in net.modules())


@pytest.mark.parametrize("keyword_input", [False, True])
def test_layout_preserves_outputs_gradients_and_parameter_identity(rocm, keyword_input):
    net = model().double()
    original = copy.deepcopy(net)
    params = list(net.parameters())
    keys = list(net.state_dict())
    assert layout.configure_dpt_convolution_layout(net, torch, "cuda") == 2
    seen = []
    for layer in net:
        layer.register_forward_pre_hook(lambda m, a, kw: seen.append(
            (a[0] if a else kw['input']).is_contiguous()), with_kwargs=True)
    x = torch.randn(2, 3, 5, 5, dtype=torch.float64).to(memory_format=torch.channels_last).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    if keyword_input:
        actual = net[1](input=net[0](input=x))
    else:
        actual = net(x)
    expected = original(ref_x)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for a, b in zip(net.parameters(), original.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    assert all(seen)
    assert all(a is b for a, b in zip(params, net.parameters()))
    assert keys == list(net.state_dict())


def test_model_reuse_copy_and_serialization_do_not_duplicate_hooks(rocm):
    net = model()
    assert layout.configure_dpt_convolution_layout(net, torch, "cuda") == 2
    assert layout.configure_dpt_convolution_layout(net, torch, "cuda") == 0
    stream = io.BytesIO()
    torch.save(net, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)  # Trusted in-memory fixture.
    for candidate in (net, copy.deepcopy(net), restored):
        assert layout.configure_dpt_convolution_layout(candidate, torch, "cuda") == 0
        assert sum(len(m._forward_pre_hooks) for m in candidate.modules()) == 2
        candidate.load_state_dict(net.state_dict())
        assert candidate(torch.randn(2, 3, 5, 5)).shape == (2, 3, 10, 10)


@pytest.mark.parametrize("family", ["dpt", "upernet"])
def test_hfcore_installs_layout_during_model_configuration(rocm, monkeypatch, family):
    from mlaas_data_generator.models.adapters.hf_core import HFCore
    monkeypatch.delenv("MLAAS_ROCM_UPERNET_CONTIGUOUS_CONV", raising=False)
    core = HFCore.__new__(HFCore)
    core.model, core.torch, core.device = model(), torch, torch.device("cuda")
    core.model.config.model_type = family
    core.model.config.backbone_config = SimpleNamespace(model_type="convnext")
    core._should_enable_gradient_checkpointing = lambda: False
    core._configure_precision_mode = lambda: None
    core._configure_memory_optimizations()
    assert sum(len(m._forward_pre_hooks) for m in core.model.modules()) == 2


@pytest.mark.parametrize("backbone", ["convnext", "swin", None])
@pytest.mark.parametrize("as_dict", [False, True])
def test_upernet_requires_convnext_backbone(rocm, monkeypatch, backbone, as_dict):
    monkeypatch.delenv("MLAAS_ROCM_UPERNET_CONTIGUOUS_CONV", raising=False)
    net = model()
    net.config.model_type = "upernet"
    net.config.backbone_config = ({"model_type": backbone} if as_dict
                                  else SimpleNamespace(model_type=backbone))
    assert layout.configure_dpt_convolution_layout(net, torch, "cuda") == 0
    count = layout.configure_rocm_convolution_layout(net, torch, "cuda")
    assert count == (2 if backbone == "convnext" else 0)
    assert layout.configure_rocm_convolution_layout(net, torch, "cuda") == 0


def test_upernet_optout_is_independent_of_dpt(rocm, monkeypatch):
    monkeypatch.setenv("MLAAS_ROCM_UPERNET_CONTIGUOUS_CONV", "0")
    net = model()
    net.config.model_type = "upernet"
    net.config.backbone_config = SimpleNamespace(model_type="convnext")
    assert layout.configure_rocm_convolution_layout(net, torch, "cuda") == 0
    assert layout.configure_rocm_convolution_layout(model(), torch, "cuda") == 2
