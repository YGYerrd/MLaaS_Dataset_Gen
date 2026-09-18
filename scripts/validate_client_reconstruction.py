"""Stage F: fresh-process replay comparisons over representative services."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASK_PROFILES = {
    "pair_classification": ("huawei-noah/TinyBERT_General_4L_312D", "glue_mrpc", "sentence_similarity"),
    "pair_regression": ("huawei-noah/TinyBERT_General_4L_312D", "glue_stsb", "sentence_similarity"),
    "tokens": ("huawei-noah/TinyBERT_General_4L_312D", "wikiann_en", "token_classification"),
    "summarization": ("google/flan-t5-small", "xsum", "seq2seq_generation"),
    "segmentation": ("nvidia/segformer-b0-finetuned-ade-512-512", "ade20k_mini_segmentation", "image_segmentation"),
    "masked_lm": ("huawei-noah/TinyBERT_General_4L_312D", "ag_news", "fill_mask"),
}


def task_configurations(root, device, profiles=None):
    """Small real-backend cohorts using the normal raw-source loader."""
    from huggingface_hub.constants import HF_HUB_CACHE
    from mlaas_data_generator.config import CONFIG
    from mlaas_data_generator.registry.datasets import DATASET_REGISTRY
    from mlaas_data_generator.services.taxonomy import canonical_task_family
    rows = []
    def revision(kind, name):
        return (Path(HF_HUB_CACHE) / (kind + "--" + name.replace("/", "--")) / "refs/main").read_text().strip()
    for profile in profiles or TASK_PROFILES:
        model, dataset_key, task = TASK_PROFILES[profile]
        spec = {k: v for k, v in DATASET_REGISTRY[dataset_key].items() if k != "explainability"}
        if task == "fill_mask":
            spec.update(loader_template="hf_masked_lm", label_column=None, task_key="fill_mask")
        base = {**CONFIG, **spec, "dataset": "hf", "task": spec["task_key"], "hf_task": task,
            "model_type": "hf_finetune", "hf_model_id": model,
            "hf_model_revision": revision("models", model),
            "hf_dataset_revision": revision("datasets", spec["dataset_name"]),
            "task_type": canonical_task_family(spec.get("task_type", "classification"), task),
            "modality": "image" if profile == "segmentation" else "text",
            "test_split": spec["test_split"] + "[:16]", "training_regime": "finetune_transfer",
            "generation_mode": "independent", "training_epochs": 1, "batch_size": 4,
            "learning_rate": 5e-5, "optimizer": "adamw", "weight_decay": 0.01,
            "warmup_ratio": 0.0, "gradient_accumulation_steps": 1,
            "max_length": 32, "source_max_length": 32, "target_max_length": 16,
            "max_new_tokens": 12, "num_beams": 1, "do_sample": False,
            "generation_config": {"max_new_tokens": 12, "num_beams": 1, "do_sample": False},
            "mlm_probability": 0.3, "dynamic_padding": profile != "segmentation",
            "device": device, "mixed_precision": False, "precision_type": "fp32",
            "checkpoint_weight_format": "native", "weight_retention": "recipe_only",
            "retain_validation_checkpoint": True, "checkpoint_measure_full": False,
            "max_train_time_s": None, "max_eval_time_s": None,
            "measure_system_metrics": False, "update_signature_enabled": False,
            "enable_perturbation_metrics": False, "explainability_enabled": False,
            "db_path": str(root / "services.db"), "provenance_artifact_dir": str(root / "artifacts"),
            "initialization_seed": 42, "evaluation_seed": 42, "seed": 42}
        if profile == "segmentation":
            base["processor_size"] = {"height": 128, "width": 128}
        for index in range(2):
            rows.append({**base, "service_id": f"replay_{profile}_{index}", "sample_size": 12 if index == 0 else 8,
                         "sample_seed": 51 + index, "training_seed": 101 + index,
                         "split_strategy": "iid"})
    return rows


def configurations(root, suite, device):
    from mlaas_data_generator.config import CONFIG
    families = [
        {"dataset": "digits", "model_type": "mlp", "hidden_layers": [32], "dropout": 0.1, "sample_size": 240},
        {"dataset": "wine", "model_type": "logreg", "sample_size": 96},
        {"dataset": "diabetes", "model_type": "mlp", "hidden_layers": [32], "sample_size": 240, "task_type": "regression"},
    ]
    if suite == "representative":
        hub = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
        def revision(kind, name):
            return (hub / (kind + "--" + name.replace("/", "--")) / "refs/main").read_text().strip()
        for model, dataset, extra in (
            ("huawei-noah/TinyBERT_General_4L_312D", "ag_news", {"hf_task": "sequence_classification",
                "text_column": "text", "label_column": "label", "modality": "text", "max_length": 64,
                "test_split": "test[:256]"}),
            ("facebook/deit-tiny-patch16-224", "beans", {"hf_task": "image_classification",
                "image_column": "image", "label_column": "labels", "modality": "image"}),
            ("roneneldan/TinyStories-1M", "roneneldan/TinyStories", {"hf_task": "text_generation",
                "task_type": "generation", "text_column": "text", "modality": "text", "max_length": 64,
                "max_new_tokens": 16, "do_sample": False, "test_split": "validation[:64]"}),
        ):
            families.append({"dataset": "hf", "dataset_name": dataset, "hf_model_id": model,
                "model_type": "hf_finetune", "hf_model_revision": revision("models", model),
                "hf_dataset_revision": revision("datasets", dataset), "training_regime": "finetune_transfer",
                "sample_size": 32, "batch_size": 8, "training_epochs": 2, "learning_rate": 5e-5,
                "optimizer": "adamw", "train_split": "train", "test_split": "test", "device": device, **extra})
    else:
        families.extend([
            {"dataset": "iris", "model_type": "mlp", "hidden_layers": [16], "dropout": 0.2, "sample_size": 96},
            {"dataset": "digits", "model_type": "logreg", "sample_size": 240},
            {"dataset": "synthetic", "task_type": "regression", "model_type": "mlp", "hidden_layers": [32], "sample_size": 240},
        ])
    rows = []
    for f, family in enumerate(families):
        for client, strategy in enumerate(("iid", "iid", "dirichlet", "shard")):
            config = {**CONFIG, "service_id": f"replay_f{f + 1}_c{client + 1}", "task_type": "classification",
                "training_regime": "generic", "batch_size": 32, "training_epochs": 8, "learning_rate": 0.003,
                "device": "cpu", "optimizer": "adam", "db_path": str(root / "services.db"),
                "provenance_artifact_dir": str(root / "artifacts"), "max_train_time_s": None,
                "measure_system_metrics": False, "update_signature_enabled": False,
                "checkpoint_weight_format": "native", "weight_retention": "recipe_only",
                "retain_validation_checkpoint": True, "mixed_precision": False, "precision_type": "fp32",
                "sample_seed": 51 + client, "training_seed": 101 + client,
                "initialization_seed": 42, "evaluation_seed": 42, **family}
            config["sample_size"] = int(config["sample_size"] * (0.5 if client == 1 else 1))
            config["training_epochs"] += client % 2
            config["batch_size"] = max(4, config["batch_size"] // (1 + client % 2))
            if config["dataset"] == "hf":
                config.update(gradient_accumulation_steps=1 + client % 2, warmup_ratio=0.1 if client % 2 else 0,
                              weight_decay=0.01 if client % 2 else 0)
            if config["task_type"] == "generation":
                strategy = "iid"
            config["split_strategy"] = strategy
            if strategy in {"dirichlet", "shard"}:
                config["skew_axis"] = "score_bin" if config["task_type"] == "regression" else "class_label"
                config["distribution_param"] = 0.5
            rows.append(config)
    return rows


def metrics_snapshot(db):
    if not Path(db).is_file():
        return []
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        return conn.execute("SELECT * FROM service_metrics ORDER BY service_id,metric_name").fetchall()


def reusable_comparison(summary, recipe, *, reference_id, device, tolerances):
    """Reuse only an intact, passing report for this exact recipe and gate."""
    from mlaas_data_generator.provenance import sha256_file
    from mlaas_data_generator.client_reconstruction import DEFAULT_TOLERANCES
    path = Path(summary["report_ref"])
    if sha256_file(path) != path.stem:
        raise ValueError("cached comparison failed content-address verification")
    report = json.loads(path.read_text(encoding="utf-8"))
    return (report.get("version") == "client_reconstruction_comparison_v3"
            and report.get("replay_evidence", {}).get("fresh_process") is True
            and report["service_id"] == recipe["service_id"] and report["recipe_hash"] == recipe["recipe_hash"]
            and report["reference_service_id"] == reference_id and report["gate"]["passed"]
            and report["gate"]["tolerances"] == {**DEFAULT_TOLERANCES, **tolerances}
            and report["training"].get("device", "cpu") == device
            and all(summary[k] == report[k] for k in summary if k != "report_ref"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite", choices=("offline", "representative", "tasks"), default="offline")
    parser.add_argument("--task-profile", choices=tuple(TASK_PROFILES), action="append",
                        help="Select one or more profiles for --suite tasks; defaults to all six")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--db", help="Validate an existing database without generating services")
    parser.add_argument("--resume-generation", action="store_true", help="Verify and reuse completed validation services in this output directory")
    parser.add_argument("--reuse-comparisons", action="store_true", help="With --resume-generation, reuse intact passing reports when recipe/device/tolerances match")
    parser.add_argument("--service-id", action="append", dest="service_ids")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--tolerances", help="JSON overrides of the fidelity thresholds")
    parser.add_argument("--worker-service", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    parser.add_argument("--reference-service", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.task_profile and (args.suite != "tasks" or args.db):
        parser.error("--task-profile requires --suite tasks without --db")
    if args.reuse_comparisons and not args.resume_generation:
        parser.error("--reuse-comparisons requires --resume-generation")
    if args.worker_service:
        from mlaas_data_generator.client_reconstruction import compare_reconstruction
        report, artifact = compare_reconstruction(args.db, args.worker_service, output_dir=args.output_dir,
            device=args.device, reference_service_id=args.reference_service,
            fresh_process=True,
            tolerances=json.loads(args.tolerances) if args.tolerances else None)
        Path(args.worker_result).write_text(json.dumps({"report": report, "artifact": artifact}, indent=2), encoding="utf-8")
        return
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prior_reports = {}
    if args.reuse_comparisons and (root / "summary.json").exists():
        prior = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if not prior["original_metrics_unchanged"]:
            raise ValueError("cannot reuse a run that changed original metrics")
        prior_reports = {r["service_id"]: r for r in prior["reports"]}
    db = Path(args.db).resolve() if args.db else root / "services.db"
    failures, services = [], []
    if args.db:
        with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
            ids = args.service_ids or [r[0] for r in conn.execute(
                "SELECT g.service_id FROM service_generation_provenance g JOIN services s USING(service_id) "
                "WHERE s.status='completed' AND g.client_recipe_ref IS NOT NULL ORDER BY g.family_id,g.service_id")]
        services = [(sid, None) for sid in ids[:args.limit]]
    else:
        if db.exists() and not args.resume_generation:
            raise ValueError("use a new output directory to keep references immutable")
        from mlaas_data_generator.cli.manifest.preparation import execute_entries, group_key, verify_completed
        from mlaas_data_generator.cli.run_manifest import ManifestEntry, RowValidation
        rows = (task_configurations(root, args.device, args.task_profile) if args.suite == "tasks"
                else configurations(root, args.suite, args.device))[:args.limit]
        config_path = root / "configurations.json"
        if args.resume_generation and config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != rows:
            raise ValueError("validation configurations changed; use a new output directory")
        config_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        groups = defaultdict(list)
        for row in rows:
            groups[group_key(row)].append(row)
        for configs in groups.values():
            reference = configs[0]["service_id"]
            # A further client per family exercises truly absent client weights.
            twin = {**configs[0], "service_id": reference + "_unretained", "retain_validation_checkpoint": False}
            configs = configs + [twin]
            if args.resume_generation and db.exists():
                remaining = []
                with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
                    for row in configs:
                        status = conn.execute("SELECT status FROM services WHERE service_id=?", (row["service_id"],)).fetchone()
                        if status and status[0] == "completed":
                            verify_completed(db, row["service_id"], invalidate=False)
                            services.append((row["service_id"], reference if row is twin else None))
                        else:
                            remaining.append(row)
                configs = remaining
            entries = [ManifestEntry(i, i + 1, c, RowValidation(True)) for i, c in enumerate(configs)]
            try:
                results = execute_entries(entries)
            except Exception as exc:
                failures.extend({"service_id": c["service_id"], "stage": "generation", "error": str(exc)} for c in configs)
                continue
            for row, result in zip(configs, results):
                if result["status"] == "success":
                    services.append((row["service_id"], reference if row is twin else None))
                else:
                    failures.append({"service_id": row["service_id"], "stage": "generation", "error": result["error_message"]})
    before = metrics_snapshot(db)
    reports, reused = [], []
    def summary():
        unchanged = before == metrics_snapshot(db)
        result = {"suite": args.suite if not args.db else "existing", "services_compared": len(reports),
            "family_count": len({r["family_id"] for r in reports}), "original_metrics_unchanged": unchanged,
            "comparisons_reused": reused,
            "gate_passed": not failures and unchanged and bool(reports) and len(reports) == len(services),
            "failures": failures, "reports": reports}
        (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    summary()
    for index, (sid, reference) in enumerate(services):
        if sid in prior_reports:
            from mlaas_data_generator.client_reconstruction import read_recipe
            if reusable_comparison(prior_reports[sid], read_recipe(db, sid), reference_id=reference or sid,
                                   device=args.device, tolerances=json.loads(args.tolerances) if args.tolerances else {}):
                reports.append(prior_reports[sid])
                reused.append(sid)
                summary()
                print(f"Reused verified comparison: {sid}", flush=True)
                continue
        print(f"Reconstruction comparison {index + 1}/{len(services)}: {sid}", flush=True)
        result_path = root / f"worker_{index}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--db", str(db), "--output-dir", str(root / "comparisons"),
                   "--worker-service", sid, "--worker-result", str(result_path), "--device", args.device]
        if reference:
            command.extend(["--reference-service", reference])
        if args.tolerances:
            command.extend(["--tolerances", args.tolerances])
        with (root / f"worker_{index}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            failures.append({"service_id": sid, "stage": "reconstruction", "log": str(log.name), "exit_code": process.returncode})
        else:
            value = json.loads(result_path.read_text(encoding="utf-8"))
            report = value["report"]
            reports.append({k: report[k] for k in ("service_id", "family_id", "task", "realised_samples", "retention",
                "reference_checkpoint_metrics", "reconstructed_metrics", "metric_deltas", "training_loss", "predictions",
                "parameters", "bitwise_equal", "training_changed_initial_state", "replay_evidence", "gate")} | {"report_ref": value["artifact"]["artifact_uri"]})
            if not report["gate"]["passed"]:
                failures.append({"service_id": sid, "stage": "fidelity", "failures": report["gate"]["failures"]})
        summary()
    result = summary()
    print(json.dumps({k: v for k, v in result.items() if k != "reports"}, indent=2))
    if not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
