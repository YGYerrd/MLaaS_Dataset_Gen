"""Choose a volume before writing large immutable checkpoint objects."""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

_LOCAL_POLICY = Path(__file__).resolve().parent.parent / "storage.local.json"
_LOG = logging.getLogger(__name__)


def choose_storage_root(root: str | Path, required_bytes: int = 0) -> Path:
    primary = Path(root).resolve()
    policy = json.loads(_LOCAL_POLICY.read_text(encoding="utf-8")) if _LOCAL_POLICY.exists() else {}
    destination = os.getenv("MLAAS_OVERFLOW_DIR") or policy.get("overflow_dir")
    if not destination:
        return primary
    reserve = int(policy.get("reserve_bytes", 20 * 1024**3))
    if reserve < 0 or required_bytes < 0:
        raise ValueError("Storage reserve and required bytes must be non-negative")
    ancestor = primary
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free >= reserve + required_bytes:
        return primary
    overflow = Path(destination).resolve()
    # Require a provisioned destination; never silently create a mount point
    # on the primary disk when an external volume is absent.
    if not overflow.is_dir():
        raise OSError(f"Checkpoint overflow drive is unavailable: {overflow}")
    if shutil.disk_usage(overflow).free < reserve + required_bytes:
        raise OSError(f"Insufficient checkpoint space on overflow drive: {overflow}")
    _LOG.warning("Primary checkpoint storage is low; writing new objects to %s", overflow)
    return overflow
