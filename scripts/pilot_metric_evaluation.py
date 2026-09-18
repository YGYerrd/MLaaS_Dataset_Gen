"""Pilot frozen large evaluation budgets on checkpoint-free reconstructed services.

Source catalogues and recipes are read only. Each worker reconstructs one model,
prepares a larger held-out pool with the original fitted preprocessing, and writes
its evaluation evidence to a separate output directory. Partial stages resume.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def metric_snapshot(db):
    from mlaas_data_generator.provenance import sha256_json
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        return sha256_json(conn.execute("SELECT * FROM service_metrics ORDER BY service_id,metric_name").fetchall())


def select_services(databases):
    selected, seen = [], set()
    for db in databases:
        with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
            rows = conn.execute("SELECT s.service_id,g.family_id FROM services s JOIN service_generation_provenance g "
                "USING(service_id) WHERE s.status='completed' AND g.client_recipe_ref IS NOT NULL "
                "ORDER BY g.family_id,s.service_id").fetchall()
        for sid, family in rows:
            if family not in seen:
                selected.append({"db": str(Path(db).resolve()), "service_id": sid, "family_id": family})
                seen.add(family)
    return selected


def expanded_benchmark(run, cap):
    """Expand only the evaluation slice; retain the recipe's training/fitted state."""
    import numpy as np
    from mlaas_data_generator.shared_parent import _dataset_args, _source_ids
    from mlaas_data_generator.data.master_loader import load_dataset
    from mlaas_data_generator.data.tokenization_selection import training_row_selection
    from mlaas_data_generator.data.fedavg_partition import _take_rows
    from mlaas_data_generator.data.image_store import close_image_stores
    cfg = {**run["family"]["config"], "service_id": run["recipe"]["service_id"],
        "benchmark_sample_size": cap, "_recorded_source_ids": run["recipe"]["selection"]["ordered_source_row_ids"]}
    cfg = {**(cfg.get("dataset_args") or {}), **cfg}
    for key in ("test_split", "benchmark_split"):
        if cfg.get(key):
            cfg[key] = re.sub(r"\[.*\]$", "", cfg[key])
    cfg.pop("dataset_args", None)
    args = _dataset_args(cfg)
    args.update(benchmark_sample_size=cap, _shared_fitted_state=run["family"]["shared_fitted_state"]["transforms"])
    args["_defer_evaluation_sampling"] = False
    if cfg.get("dataset") == "hf":
        args["require_official_train_split"] = True
    # The source is still pinned, but its larger test slice has a new fingerprint.
    args.pop("_expected_raw_source", None)
    with training_row_selection([cfg]):
        train, test, meta = load_dataset(cfg.get("dataset", "hf"), **args)
    try:
        count = len(test[1])
        split = str(meta.get("test_split") or cfg.get("test_split") or "test")
        ids = _source_ids(meta, count, "test", split)
        # Keep the original downstream audit reservation out of this pilot.
        def canonical_id(value):
            name, position = str(value).rsplit(":", 1)
            return re.sub(r"\[.*\]$", "", name) + ":" + position
        audit = {canonical_id(v) for v in run["family"]["reservation"].get("audit_ids", [])}
        training = {canonical_id(v) for v in run["recipe"]["selection"]["ordered_source_row_ids"]}
        training.update(canonical_id(v) for v in run["family"]["reservation"].get("training_ids", []))
        positions = [i for i, sid in enumerate(ids) if canonical_id(sid) not in audit]
        if training & {canonical_id(ids[i]) for i in positions}:
            raise ValueError("pilot evaluation overlaps reconstructed training rows")
        selected = (_take_rows(test[0], positions), _take_rows(test[1], positions))
        meta = {**meta, "evaluation_source_ids": [ids[i] for i in positions]}
        if meta.get("test_group_ids") is not None:
            meta["metric_evaluation_group_ids"] = [meta["test_group_ids"][i] for i in positions]
        # Numeric perturbation calibration is derived from training features, not a single test row.
        if isinstance(train[0], np.ndarray) and train[0].ndim == 2:
            values = np.asarray(train[0], dtype=float)
            meta["pilot_training_feature_calibration"] = {
                "scale": np.std(values, axis=0).tolist(), "minimum": np.min(values, axis=0).tolist(),
                "maximum": np.max(values, axis=0).tolist(), "mean": np.mean(values, axis=0).tolist(),
                "source": "frozen_preprocessed_training_pool"}
        return selected, meta, (train, test)
    except BaseException:
        close_image_stores((train, test))
        raise


def worker(spec):
    from mlaas_data_generator.client_reconstruction import materialize_service
    from mlaas_data_generator.data.image_store import close_image_stores
    from mlaas_data_generator.evaluation_protocol import apply_evaluation_profile, resolve_evaluation_plan
    from mlaas_data_generator.services.production_evaluation import run_production_evaluation
    from mlaas_data_generator.provenance import sha256_json
    from unittest.mock import patch
    root = Path(spec["output_dir"])
    before = metric_snapshot(spec["db"])
    start = time.perf_counter()
    result = {"source": spec, "version": "metric_evaluation_pilot_v1"}
    from mlaas_data_generator.provenance import sha256_file, runtime_provenance
    repository = Path(__file__).resolve().parents[1]
    result["implementation_hashes"] = {name: sha256_file(repository / name) for name in (
        "mlaas_data_generator/evaluation_protocol.py", "mlaas_data_generator/services/evaluation_outputs.py",
        "mlaas_data_generator/services/production_evaluation.py", "scripts/pilot_metric_evaluation.py")}
    result["runtime"] = runtime_provenance(repository)
    resources = None
    try:
        with patch("mlaas_data_generator.checkpoint_store.reconstruct_service_state", side_effect=RuntimeError("client checkpoint read during pilot")), \
             patch("mlaas_data_generator.cli.manifest.preparation.reconstruct_service_state", side_effect=RuntimeError("client checkpoint read during pilot")), \
             materialize_service(spec["db"], spec["service_id"], device=spec["device"]) as run:
            benchmark, meta, resources = expanded_benchmark(run, spec["benchmark_sample_size"])
            cfg = {**apply_evaluation_profile(run["service"].config, spec.get("evaluation_profile", "pilot")),
                "trust_max_duration_s": spec["trust_seconds"], "explainability_max_duration_s": spec["explanation_seconds"],
                "metric_evaluation_progress_path": str(root / "progress.json"),
                "evaluation_model_identity": run["recipe"]["fingerprints"]["original_model"],
                "metric_evaluation_source_ids": meta["evaluation_source_ids"],
                "metric_evaluation_dataset_split": meta.get("test_split", "test"),
                "metric_evaluation_dataset_revision": run["service"].config.get("hf_dataset_revision"),
                "perturbation_progress_logging": True}
            calibration = meta.get("pilot_training_feature_calibration")
            if calibration:
                cfg.update(feature_scales=calibration["scale"], feature_min=calibration["minimum"],
                    feature_max=calibration["maximum"], input_calibration_provenance=calibration["source"])
            if run["task"] == "regression":
                cfg["regression_target_scale"] = 1.0 if meta.get("target_scaler") else 5.0
                cfg["regression_target_scale_provenance"] = "frozen_standardized_target_unit" if meta.get("target_scaler") else "STSB_score_range_0_5"
            processor = getattr(getattr(run["model"], "core", None), "tokenizer", None)
            if getattr(processor, "image_mean", None) is not None and getattr(processor, "do_normalize", True):
                cfg.update(image_mean=processor.image_mean, image_std=processor.image_std, image_value_range=[0, 1])
            result.update(task=run["task"], hf_task=cfg.get("hf_task"), dataset=cfg.get("dataset_name") or cfg.get("dataset"),
                model=cfg.get("hf_model_id") or cfg.get("model_type"), recipe_hash=run["recipe"]["recipe_hash"],
                recipe_reconstruction=run["replay_evidence"], client_checkpoint_reads=0,
                available_inputs=len(benchmark[1]), source_ids_hash=sha256_json(meta["evaluation_source_ids"]),
                plan=resolve_evaluation_plan(cfg))
            result["evaluation"] = run_production_evaluation(run["model"], *benchmark, task_family=run["task"],
                hf_task=cfg.get("hf_task"), config=cfg, meta=meta, service_id=spec["service_id"])
            result["status"] = "evaluated"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if resources is not None:
            close_image_stores(resources)
    result.update(duration_s=time.perf_counter() - start, original_metrics_unchanged=before == metric_snapshot(spec["db"]))
    (root / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("status", "task", "dataset", "model", "error", "duration_s", "original_metrics_unchanged")}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", action="append")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    from mlaas_data_generator.evaluation_protocol import PROFILES, resolve_evaluation_plan
    parser.add_argument("--evaluation-profile", choices=tuple(PROFILES), default="benchmark")
    parser.add_argument("--trust-seconds", type=float, default=None)
    parser.add_argument("--explanation-seconds", type=float, default=None)
    parser.add_argument("--benchmark-sample-size", type=int, default=2048)
    parser.add_argument("--service-id", action="append")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        result = worker(json.loads(Path(args.worker).read_text(encoding="utf-8")))
        # Resources and the report are finalized by worker(). Avoid native ML
        # runtime shutdown deadlocks in these isolated disposable processes.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if result["status"] == "evaluated" and result["original_metrics_unchanged"] else 1)
    if not args.source_db or not args.output_dir:
        parser.error("--source-db and --output-dir are required")
    plan = resolve_evaluation_plan({"metric_evaluation_profile": args.evaluation_profile})
    if args.trust_seconds is None:
        args.trust_seconds = plan["trust_max_duration_s"]
    if args.explanation_seconds is None:
        args.explanation_seconds = plan["explainability_max_duration_s"]
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = select_services(args.source_db)
    if args.service_id:
        selected = [s for s in selected if s["service_id"] in args.service_id]
    if not selected:
        parser.error("no reconstructable completed families selected")
    if args.workers < 1:
        parser.error("--workers must be positive")
    results = []
    def evaluate(index, source):
        folder = root / f"family_{index:02d}_{source['service_id']}"
        folder.mkdir(exist_ok=True)
        spec = {**source, "output_dir": str(folder), "device": args.device, "trust_seconds": args.trust_seconds,
            "evaluation_profile": args.evaluation_profile,
            "explanation_seconds": args.explanation_seconds, "benchmark_sample_size": args.benchmark_sample_size}
        path = folder / "spec.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous.setdefault("evaluation_profile", "pilot")
            operational = {"trust_seconds", "explanation_seconds"}
            if {k: v for k, v in previous.items() if k not in operational} != {k: v for k, v in spec.items() if k not in operational}:
                raise ValueError("pilot configuration changed; use a new output directory")
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"Pilot {index + 1}/{len(selected)}: {source['service_id']}", flush=True)
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, __file__, "--worker", str(path)], stdout=log, stderr=subprocess.STDOUT)
        result_path = folder / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
            "status": "failed", "source": spec, "error": f"worker exit {process.returncode}"}
        if result.get("source") != spec or (process.returncode and result.get("status") == "evaluated"):
            result = {"status": "failed", "source": spec, "error": f"worker exit {process.returncode}; no matching completed report"}
        print(f"  {result['status']}: {result.get('error', '')}", flush=True)
        return {**{k: v for k, v in result.items() if k != "evaluation"}, "result_path": str(result_path),
            "metrics": {k: v for k, v in result.get("evaluation", {}).items() if k != "perturbation_samples"}}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate, i, source) for i, source in enumerate(selected)]
        for future in as_completed(futures):
            results.append(future.result())
            summary = {"version": "metric_evaluation_pilot_v1", "production_enabled": False,
                "evaluation_profile": args.evaluation_profile, "evaluation_protocol_version": plan["protocol_version"],
                "families_requested": len(selected), "families_attempted": len(results), "worker_count": args.workers,
                "reports": sorted(results, key=lambda row: row["source"]["service_id"])}
            temporary = root / "summary.tmp"
            temporary.write_text(json.dumps(summary, indent=2, allow_nan=False, default=str), encoding="utf-8")
            temporary.replace(root / "summary.json")
    if any(row["status"] != "evaluated" or not row.get("original_metrics_unchanged") for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
