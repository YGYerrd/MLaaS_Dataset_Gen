from __future__ import annotations

from typing import Any


_ROCM_RUNTIME_MARKERS = (
    "miopen",
    "hiprtc_error_compilation",
    "code object build failed",
    "miopenstatusunknownerror",
    "offload-arch.exe",
)


def is_rocm_miopen_runtime_error(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in _ROCM_RUNTIME_MARKERS)
