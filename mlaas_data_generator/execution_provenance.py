"""Dataset-owned execution evidence, independent of input manifest files."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import math

import numpy as np

from .provenance import canonical_json


def evidence_value(value):
    """Copy JSON evidence without diagnostic length/node limits.

    Runtime objects are described explicitly, never repr'd or expanded into
    model weights. Arrays in evidence (for example selected row IDs) are data.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, np.generic):
        return evidence_value(value.item())
    if isinstance(value, np.ndarray):
        return evidence_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): evidence_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [evidence_value(v) for v in value]
    return {"unserialized_runtime_type": f"{type(value).__module__}.{type(value).__name__}"}


def configuration_evidence(config):
    # Private entries include live models, datasets and worker caches. Public
    # configuration contains the resolved training and evaluation settings.
    return evidence_value({k: v for k, v in config.items() if not k.startswith("_")})


def execution_document(*, requested, effective, metrics, record, splits,
                       generation, artifacts, runtime, training_adjustment=None,
                       training_config=None, fit_arguments=None):
    effective = configuration_evidence(effective)
    changes = {}
    for key in sorted(requested.keys() | effective.keys()):
        if key not in requested or key not in effective or canonical_json(requested[key]) != canonical_json(effective[key]):
            changes[key] = {"before_present": key in requested, "after_present": key in effective,
                            "before": requested.get(key), "after": effective.get(key)}
    return evidence_value({
        "version": 1,
        "evidence_scope": "observed_execution; original_request_match_not_asserted",
        "requested_config": requested,
        "effective_config": effective,
        "configuration_adjustments": changes,
        "training_budget_adjustment": training_adjustment,
        "training_config": training_config,
        "adapter_fit_arguments": fit_arguments,
        "metrics": metrics,
        "service_record": record,
        "splits": splits,
        "generation": generation,
        "artifacts": artifacts,
        "runtime": runtime,
    })
