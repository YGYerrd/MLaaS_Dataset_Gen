"""Content-addressed compressed evidence for large evaluation plans."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tempfile


def persist_evaluation_evidence(root, values, samples):
    from .production_evaluation import _jsonable
    payload = _jsonable({"evaluation": values, "perturbation_samples": samples})
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    digest = hashlib.sha256(compressed).hexdigest()
    directory = Path(root).resolve() / "metric_evaluation_evidence"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json.gz"
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("existing evaluation evidence failed integrity verification")
    else:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(compressed)
        temporary.replace(path)
    return {"artifact_type": "perturbation_samples", "artifact_uri": str(path), "metadata": {
        "sha256": digest, "size_bytes": len(compressed), "format": "json.gz", "role": "metric_evaluation_evidence",
        "evaluation_protocol_version": values["evaluation_protocol_version"],
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(), "uncompressed_size_bytes": len(raw),
        "trust_status": values.get("trust_status"), "explainability_status": values.get("explainability_status"),
        "sample_count": len(samples or [])}}
