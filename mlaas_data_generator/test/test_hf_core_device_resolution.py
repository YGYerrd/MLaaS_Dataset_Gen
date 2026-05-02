import types

from mlaas_data_generator.models.adapters.hf_core import HFCore


class _TorchStub:
    def __init__(self, available):
        self.cuda = types.SimpleNamespace(is_available=lambda: available)


def test_hf_core_resolves_cpu_when_cuda_is_unavailable():
    core = HFCore.__new__(HFCore)
    core.torch = _TorchStub(False)

    assert core._resolve_device(None) == "cpu"


def test_hf_core_resolves_cuda_when_available():
    core = HFCore.__new__(HFCore)
    core.torch = _TorchStub(True)

    assert core._resolve_device(None) == "cuda"
