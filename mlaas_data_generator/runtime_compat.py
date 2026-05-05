from __future__ import annotations

from typing import Any


_ROCM_RUNTIME_MARKERS = (
    "miopen",
    "hiprtc_error_compilation",
    "code object build failed",
    "miopenstatusunknownerror",
    "offload-arch.exe",
)

_CUDA_DEVICE_ASSERT_MARKERS = (
    "device-side assert triggered",
    "cuda error: device-side assert",
    "torch_use_cuda_dsa",
)

_CUDA_OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "out of memory",
)


def is_rocm_miopen_runtime_error(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in _ROCM_RUNTIME_MARKERS)


def is_cuda_device_assert_error(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in _CUDA_DEVICE_ASSERT_MARKERS)


def is_cuda_oom_error(value: Any) -> bool:
    text = str(value or "").lower()
    return "cuda" in text and any(marker in text for marker in _CUDA_OOM_MARKERS)


def is_cuda_poison_error(value: Any) -> bool:
    return is_cuda_device_assert_error(value) or is_cuda_oom_error(value)
