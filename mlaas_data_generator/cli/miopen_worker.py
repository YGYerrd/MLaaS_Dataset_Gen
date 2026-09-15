"""Run selected vision services with a process-isolated MIOpen environment."""
from __future__ import annotations

import os
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile


MIOPEN_FAST_FAMILIES = {
    # RX 9070 XT: observed with DETR-DC5 on Windows ROCm 7.2.
    # The other ResNet-50 DETR variants are precautionary family matches.
    ("win32", "gfx1201"): frozenset({"detr", "conditional-detr", "deformable-detr"}),
}


def registered_family(config):
    family = config.get("model_family")
    if not family:
        metadata = config.get("hf_service_meta_json") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (ValueError, TypeError):
                metadata = {}
        if isinstance(metadata, dict):
            family = metadata.get("model_family")
    if not family:
        from ..registry.models import MODEL_REGISTRY
        model_id = config.get("hf_model_id") or config.get("model_id")
        family = next((spec.get("family") for spec in MODEL_REGISTRY.values()
                       if spec.get("hf_model_id") == model_id and model_id), None)
    return str(family or "").strip().lower().replace("_", "-")


def should_isolate(config):
    if config.get("_miopen_fast_isolated") or sys.platform != "win32":
        return False
    family = registered_family(config)
    if not any(family in families for families in MIOPEN_FAST_FAMILIES.values()):
        return False
    device = str(config.get("device") or "auto").strip().lower()
    if device not in {"auto", "gpu", "auto_gpu", "cuda_auto", "cuda"} and not device.startswith("cuda:"):
        return False
    import torch
    if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        return False
    try:
        index = int(device.split(":", 1)[1]) if device.startswith("cuda:") else torch.cuda.current_device()
        architecture = str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", "")).split(":", 1)[0]
    except (RuntimeError, ValueError, IndexError, AssertionError):
        return False
    return family in MIOPEN_FAST_FAMILIES.get((sys.platform, architecture), ())


def run_isolated(payload, *, grouped):
    # Environment is supplied at process creation, before torch/MIOpen imports.
    # Never modify os.environ in the parent: native libraries may cache it.
    env = dict(os.environ)
    env["MIOPEN_FIND_MODE"] = "FAST"
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
    with tempfile.TemporaryDirectory(prefix="mlaas-miopen-") as folder:
        request, response = Path(folder) / "request.pkl", Path(folder) / "response.pkl"
        # Trusted, locally generated payloads only; preserve arrays/Path values in
        # resolved family contracts without changing their provenance.
        with request.open("wb") as handle:
            pickle.dump({"payload": payload, "grouped": grouped}, handle)
        command = [sys.executable, "-m", "mlaas_data_generator.cli.miopen_worker", str(request), str(response)]
        child = subprocess.Popen(command, env=env)
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
        if code or not response.exists():
            raise RuntimeError(f"isolated MIOpen worker failed (exit code {code})")
        with response.open("rb") as handle:
            return pickle.load(handle)


def main():
    request, response = map(Path, sys.argv[1:])
    with request.open("rb") as handle:
        job = pickle.load(handle)
    from . import run_manifest as runner
    from ..data.runtime_cache import preparation_cache_scope
    payload = job["payload"]
    for item in payload:
        item["resolved"].update(_miopen_fast_isolated=True,
                                miopen_process_isolated=True, miopen_find_mode="FAST")
    with preparation_cache_scope():
        result = (runner._execute_hf_group_worker(payload) if job["grouped"]
                  else [runner._execute_entry_worker(payload[0])])
    with response.open("wb") as handle:
        pickle.dump(result, handle)


if __name__ == "__main__":
    main()
