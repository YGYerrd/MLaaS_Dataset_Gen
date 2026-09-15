"""Real cached HF models/data through public generation/composition entry points.

Only dataset transport is substituted: immutable cached Arrow files are read and
an explicitly recorded first-N subset becomes the smoke source. Preprocessing,
models, training, storage, search, inference and metrics use production code.
No downloads, model mocks or synthetic success records are used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import platform
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd


PROFILES = {
    "token": ("huawei-noah/TinyBERT_General_4L_312D", "wikiann_en", "token_classification"),
    "segmentation": ("nvidia/segformer-b0-finetuned-ade-512-512", "ade20k_mini_segmentation", "image_segmentation"),
    "summarization": ("google/flan-t5-small", "xsum", "seq2seq_generation"),
}


def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    torch.set_num_threads(4)
    from datasets import Dataset
    from huggingface_hub.constants import HF_HUB_CACHE
    from datasets.config import HF_DATASETS_CACHE
    from mlaas_data_generator.registry.datasets import DATASET_REGISTRY
    from mlaas_data_generator.config import CONFIG
    from mlaas_data_generator.provenance import sha256_file, sha256_json
    from mlaas_data_generator.cli.run_manifest import run_manifest
    from mlaas_data_generator.cli.manifest.hf_manifest_builder import save_manifest
    from mlaas_data_generator.behavioural_diagnostics import family_diagnostics
    from mlaas_composition.catalogue import load_services_from_db
    from mlaas_composition.requirements import generate_service_requests
    from mlaas_composition.protocol import write_request_file
    from mlaas_composition.cli import build_parser
    from mlaas_composition.validation_cli import validate_from_files

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_id, dataset_key, task = PROFILES[args.profile]
    spec = dict(DATASET_REGISTRY[dataset_key])
    model_root = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--"))
    model_revision = (model_root / "refs/main").read_text().strip()
    model_path = model_root / "snapshots" / model_revision
    dataset_root = Path(HF_DATASETS_CACHE) / spec["dataset_name"].replace("/", "___") / (spec.get("dataset_config") or "default")
    sources, source_evidence = {}, {}
    for split, limit in ((spec["train_split"], 48), (spec["test_split"], 16)):
        candidates = sorted(dataset_root.rglob(f"*-{split}.arrow"))
        if len(candidates) != 1:
            raise ValueError(f"expected one cached Arrow source for {dataset_key}/{split}, found {len(candidates)}")
        path = candidates[0]
        dataset = Dataset.from_file(str(path))
        chosen = list(range(min(limit, len(dataset))))
        sources[split] = dataset.select(chosen, keep_in_memory=True)
        source_evidence[split] = {"path": str(path), "sha256": sha256_file(path), "positions": chosen,
                                  "upstream_revision": path.parent.name}
    revision = next(iter(source_evidence.values()))["upstream_revision"]
    evidence = {"version": "real_backend_task_smoke_v1", "profile": args.profile, "mode": "independent",
        "scope": "pinned cached model and declared Arrow first-N smoke source", "model_id": model_id,
        "model_revision": model_revision, "sources": source_evidence, "seed": 42,
        "status": "running", "production_dataset_promotion": False}
    evidence["source_contract_hash"] = sha256_json(source_evidence)
    repository = Path(__file__).resolve().parents[2]
    evidence["code_files"] = {str(p.relative_to(repository)): sha256_file(p)
        for package in ("mlaas_data_generator", "mlaas_composition", "tests/smoke")
        for p in sorted((repository / package).rglob("*.py"))}
    evidence["code_hash"] = sha256_json(evidence["code_files"])
    evidence["hardware"] = {"platform": platform.platform(), "processor": platform.processor(),
        "device": args.device, "accelerator": torch.cuda.get_device_name(0) if args.device == "cuda" else None}
    evidence_path = root / "smoke_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2))
    db = root / "services.db"
    base = {**CONFIG, **{k: v for k, v in spec.items() if k != "explainability"},
        "dataset": "hf", "task": spec["task_key"], "hf_task": task,
        "task_type": "segmentation" if args.profile == "segmentation" else "generation" if args.profile == "summarization" else "classification",
        "modality": "image" if args.profile == "segmentation" else "text",
        "model_type": "hf_finetune", "hf_model_id": model_id, "hf_model_revision": model_revision,
        "hf_dataset_revision": revision, "generation_mode": "independent", "training_regime": "finetune_transfer",
        "training_epochs": 1, "batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5, "optimizer": "adamw", "weight_decay": 0.0, "warmup_ratio": 0.0,
        "max_length": 32, "source_max_length": 32, "target_max_length": 16,
        "generation_config": {"max_new_tokens": 12, "num_beams": 1, "do_sample": False},
        "device": args.device, "precision_type": "fp32", "mixed_precision": False,
        "checkpoint_weight_format": "native", "checkpoint_measure_full": False,
        "enable_perturbation_metrics": False, "explainability_enabled": False,
        "provenance_artifact_dir": str(root / "artifacts"), "db_path": str(db),
        "max_train_time_s": 120, "max_eval_time_s": 120, "seed": 42, "enabled": True,
        "split_strategy": "iid", "task_smoke_source_hash": evidence["source_contract_hash"]}
    rows = [{**base, "sample_size": 12 + i * 4, "max_samples": None,
             "sample_seed": 42 + i, "training_seed": 42 + i,
             "service_id": f"smoke_{args.profile}_{evidence['source_contract_hash'][:12]}_{i}"} for i in range(4)]
    if args.profile == "segmentation" and args.processor_size:
        for row in rows:
            row["processor_size"] = {"height": args.processor_size, "width": args.processor_size}
    manifest = root / "clients.csv"
    if not manifest.exists():
        save_manifest(pd.DataFrame(rows), manifest, manifest_format="expanded")

    def cached_load(name, subset=None, *, split, **kwargs):
        if name != spec["dataset_name"] or (subset or None) != (spec.get("dataset_config") or None) or split not in sources:
            raise ValueError("smoke transport requested an undeclared dataset/subset/split")
        if kwargs.get("revision") != revision:
            raise ValueError("smoke transport revision differs from the frozen cache revision")
        return sources[split]

    with ExitStack() as stack:
        stack.enter_context(patch("datasets.load_dataset", cached_load))
        from mlaas_data_generator import config as config_module
        from mlaas_data_generator.cli import run_manifest as manifest_module
        stack.enter_context(patch.dict(config_module.CONFIG, {"checkpoint_weight_format": "native",
                                                             "checkpoint_measure_full": False, "checkpoint_patch_operation_policy": "fixed_xor"}))
        for name, filename in (("MANIFEST_RESULTS_PATH", "results.csv"), ("FAILED_MANIFEST_PATH", "failed.csv"), ("FAILURE_LOG_PATH", "failures.log")):
            for module in (config_module, manifest_module):
                stack.enter_context(patch.object(module, name, root / filename))
        run_manifest(file=str(manifest), dry_run=True, db_path=str(db))
        run_manifest(file=str(manifest), db_path=str(db), log_mode="concise")
        services = load_services_from_db(db)
        if len(services) != 4 or not services.composition_compatible.eq(1).all():
            raise RuntimeError("all four frozen clients must finish and verify; inspect manifest results")
        with sqlite3.connect(db) as connection:
            reference = connection.execute("SELECT family_document_ref FROM service_generation_provenance LIMIT 1").fetchone()[0]
        family = json.loads(Path(reference).read_text())
        diagnostics = family_diagnostics(db=db, family_id=family["family_id"], output_dir=root / "diagnostics", batch_size=2)
        if diagnostics["status"] != "completed" or diagnostics["diagnosed_clients"] != 4:
            raise RuntimeError("all four clients must pass catalogue diagnostics")
        requests = generate_service_requests(services, request_count=1, profile_id="measured_tasks_v5", compatibility_mode="strict", seed=42)
        requests_path = root / "requests.json"
        if not requests_path.exists():
            write_request_file(requests, requests_path)
        argv = ["evaluate", "--db", str(db), "--requests-file", str(requests_path), "--output-dir", str(root / "search"), "--search-budget", "10", "--seed", "42"]
        parsed = build_parser().parse_args(argv)
        parsed.handler(parsed, argv=argv)
        _, summary = validate_from_files(db_path=db, requests=requests, evaluation_dir=root / "search",
            request_id=str(requests.iloc[0].request_id), output_dir=root / "validation", cohort_file=family["reservation_ref"],
            audit_seed=42, audit_samples=8, min_audit_samples=1)
        if summary.get("validation_status", summary.get("status")) != "completed":
            raise RuntimeError(f"selected validation did not complete: {summary}")
        # Resume must verify existing artifacts without performing additional training.
        before = services.service_id.tolist()
        artifacts_before = immutable_artifacts(root)
        run_manifest(file=str(manifest), db_path=str(db), log_mode="concise")
        if load_services_from_db(db).service_id.tolist() != before:
            raise RuntimeError("resume changed the completed service roster")
        if immutable_artifacts(root) != artifacts_before:
            raise RuntimeError("resume rewrote immutable preparation or checkpoint artifacts")
        import torch, transformers
        evidence.update(status="completed", torch_version=torch.__version__, transformers_version=transformers.__version__,
                        device=args.device, family_id=family["family_id"], task_contract=family["config"]["task_contract"], validation=summary)
        evidence["verification"] = verify_composition(root, db, services, family["reservation_ref"], summary)
        evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False))
        if args.shared_parent:
            evidence["shared_parent"] = run_shared_parent(root, db)
        import psutil
        memory = psutil.Process().memory_info()
        evidence["process_memory"] = {"rss_bytes": memory.rss, "peak_rss_bytes": getattr(memory, "peak_wset", None)}
        evidence["resume"] = "verified roster and immutable artifact bytes/modification times unchanged"
        evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False))
    return evidence


def run_shared_parent(root, history_db):
    from mlaas_data_generator.parent_selection import select_parent_configs
    from mlaas_data_generator.shared_parent import train_parents
    from mlaas_data_generator.cohort_manifest import build_cohort_manifest
    from mlaas_data_generator.cli.run_manifest import run_manifest
    from mlaas_data_generator.behavioural_diagnostics import cohort_diagnostics
    from mlaas_composition.catalogue import load_services_from_db
    from mlaas_composition.requirements import generate_service_requests
    from mlaas_composition.protocol import write_request_file
    from mlaas_composition.cli import main as compose
    from mlaas_composition.validation_cli import validate_from_files
    folder = root / "shared_parent"
    folder.mkdir(exist_ok=True)
    selection, parents_path, db = folder / "selection.json", folder / "parents.json", folder / "services.db"
    if not selection.exists():
        select_parent_configs(history_db, selection, limit=1)
    parents = train_parents(selection, db=str(db), output=parents_path)
    if len(parents["parents"]) != 1 or parents.get("failures"):
        raise RuntimeError("shared smoke requires exactly one completed parent")
    parent = parents["parents"][0]
    manifest = folder / "clients.csv"
    frame = pd.read_csv(manifest) if manifest.exists() else build_cohort_manifest(parents_path, file=manifest,
        clients_per_parent=4, seed=42, average_samples=48, db=str(db), reuse_existing=True)
    run_manifest(file=str(manifest), dry_run=True, db_path=str(db))
    run_manifest(file=str(manifest), db_path=str(db), log_mode="concise")
    services = load_services_from_db(db)
    if len(services) != 4 or not services.cohort_succeeded.eq(1).all():
        raise RuntimeError("shared smoke requires all four planned clients to complete")
    cohort_id = frame.iloc[0].cohort_id
    diagnostics = cohort_diagnostics(db=db, cohort_id=cohort_id, output_dir=folder / "diagnostics", batch_size=2)
    requests = generate_service_requests(services, request_count=1, profile_id="measured_tasks_v5", compatibility_mode="strict", seed=42)
    requests_path = folder / "requests.json"
    if not requests_path.exists():
        write_request_file(requests, requests_path)
    compose(["evaluate", "--db", str(db), "--requests-file", str(requests_path), "--output-dir", str(folder / "search"), "--search-budget", "10", "--seed", "42"])
    _, summary = validate_from_files(db_path=db, requests=requests, evaluation_dir=folder / "search",
        request_id=str(requests.iloc[0].request_id), output_dir=folder / "validation", cohort_file=parent["reservation_ref"],
        audit_seed=42, audit_samples=8, min_audit_samples=1)
    if summary["validation_status"] != "completed" or diagnostics["status"] != "completed":
        raise RuntimeError(f"shared diagnostics/validation failed: {summary}")
    artifacts_before = immutable_artifacts(root)
    run_manifest(file=str(manifest), db_path=str(db), log_mode="concise")
    if immutable_artifacts(root) != artifacts_before:
        raise RuntimeError("shared-parent resume rewrote immutable checkpoint artifacts")
    return {"status": "completed", "parent_checkpoint_id": parent["parent_checkpoint_id"], "cohort_id": cohort_id,
            "validation": summary, "verification": verify_composition(folder, db, services, parent["reservation_ref"], summary)}


def verify_composition(folder, db, services, reservation_ref, summary):
    """Audit baselines never influence catalogue selection or the frozen roster."""
    from mlaas_composition.fedavg_validation import resolve_selected_fedavg_members, _checkpoint_root
    from mlaas_composition.regeneration import execute_tensor_state_prediction
    from mlaas_composition.validation_cli import _read_cohort
    from mlaas_data_generator.checkpoint_store import reconstruct_service_base_state, reconstruct_service_state
    from mlaas_data_generator.fedavg_aggregation import FedAvgContributor, FedAvgAggregationError, aggregate_fedavg_states
    from dataclasses import replace
    from mlaas_data_generator.provenance import sha256_file
    members = resolve_selected_fedavg_members(db, summary["service_ids"])
    first = members[0]
    root = _checkpoint_root(first.row, db)
    parent = reconstruct_service_base_state(root, first.service_id)
    contributors = [FedAvgContributor(m.client_id, m.service_id, m.sample_count,
        reconstruct_service_state(_checkpoint_root(m.row, db), m.service_id),
        m.row["compatibility_contract_hash"], parent.sha256) for m in members]
    contract_hash = first.row["compatibility_contract_hash"]
    aggregate = aggregate_fedavg_states(parent, contributors, compatibility_contract_hash=contract_hash)
    replay = aggregate_fedavg_states(parent, reversed(contributors), compatibility_contract_hash=contract_hash)
    if aggregate.state.sha256 != replay.state.sha256 or aggregate.state.sha256 != summary["aggregated_state_hash"]:
        raise RuntimeError("stored-state aggregation replay differs from selected validation")
    for index in aggregate.provenance["preserved_tensor_indices"]:
        if aggregate.state.entries[index] != parent.entries[index]:
            raise RuntimeError("aggregation changed a preserved tensor")
    if sum(c.sample_count for c in contributors) != summary["selected_total_samples"]:
        raise RuntimeError("aggregation weights differ from persisted training-example counts")
    negative_checks = {}
    for name, changed in {
        "parent_mismatch": replace(contributors[0], parent_state_hash="0" * 64),
        "contract_mismatch": replace(contributors[0], compatibility_contract_hash="0" * 64),
        "parameter_schema_mismatch": replace(contributors[0], state=replace(contributors[0].state, entries=contributors[0].state.entries[:-1])),
    }.items():
        try:
            aggregate_fedavg_states(parent, [changed, *contributors[1:]], compatibility_contract_hash=contract_hash)
        except FedAvgAggregationError:
            negative_checks[name] = "rejected_before_model_execution"
        else:
            raise RuntimeError(f"invalid {name} was accepted")
    # One baseline chosen solely on catalogue quality, with a stable tie break.
    best = services.sort_values(["metric_score", "service_id"], ascending=[False, True]).iloc[0].service_id
    best_member = next(m for m in resolve_selected_fedavg_members(db, services.service_id.tolist()) if m.service_id == best)
    best_state = reconstruct_service_state(_checkpoint_root(best_member.row, db), best)
    plan = _read_cohort(reservation_ref, seed=42, audit_samples=8, minimum=1, task_family=summary["task_family"])
    baselines = {}
    for name, sid, state in (("parent", first.service_id, parent), ("catalogue_best_individual", best, best_state)):
        result = execute_tensor_state_prediction(db_path=db, representative_service_id=sid,
            runtime_id=f"smoke_baseline_{name}", plan=plan, state=state)
        baselines[name] = {"service_id": sid, "state_hash": state.sha256, "metrics": result.outputs.metrics}
    # Prediction equality is checked on exact restored weights using the same audit protocol.
    replay_prediction = execute_tensor_state_prediction(db_path=db, representative_service_id=first.service_id,
        runtime_id="smoke_aggregate_replay", plan=plan, state=aggregate.state)
    for key in ("prediction_hash", "quality_score", summary["actual_primary_metric_name"], summary["actual_secondary_metric_name"]):
        if replay_prediction.outputs.metrics[key] != summary["task_metrics"][key]:
            raise RuntimeError(f"restored aggregate prediction replay changed {key}")
    report = {"status": "passed", "aggregate_replay_hash": replay.state.sha256,
        "negative_checks": negative_checks,
        "restored_prediction_replay": "passed", "aggregation": aggregate.provenance, "baselines": baselines,
        "audit_reservation_sha256": sha256_file(reservation_ref), "baseline_selection": "catalogue metric_score descending, service_id tie break"}
    (folder / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def immutable_artifacts(root):
    from mlaas_data_generator.provenance import sha256_file
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns, sha256_file(path))
            for path in (root / "artifacts").rglob("*") if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--shared-parent", action="store_true", help="Also train a task parent and exercise four parent-derived clients")
    parser.add_argument("--processor-size", type=int, default=None, help="Declared spatial protocol for a segmentation smoke run")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        path = Path(args.output_dir) / "smoke_evidence.json"
        if path.exists():
            evidence = json.loads(path.read_text())
            evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            path.write_text(json.dumps(evidence, indent=2))
        raise


if __name__ == "__main__":
    main()
