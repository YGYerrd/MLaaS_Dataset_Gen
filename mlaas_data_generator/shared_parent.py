"""Task-trained immutable parents and independently executed client cohorts.

There is intentionally no aggregation dependency in this execution path.
"""
from __future__ import annotations

import json
import hashlib
import math
import sqlite3
import gc
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .checkpoint_store import (capture_model_tensor_state,
                               ensure_lineage_base, load_lineage_base, restore_model_tensor_state,
                               reconstruct_service_state)
from .config import CONFIG
from .reduced_checkpoint import source_state_hash
from .family_preflight import MIN_EVALUATION_EXAMPLES, evaluation_size_error
from .clients.partitioning import client_partition, materialize_partition
from .data.fedavg_partition import _take_rows
from .data.image_store import DiskImageArray, DiskArraySequence, close_image_stores
from .parent_selection import selected_recipes, quality_metric
from .provenance import (canonical_json, sha256_json, sha256_file, path_integrity, parameter_schema,
                         model_identity, label_semantics, preprocessing_metadata, dataset_metadata,
                         save_reconstruction_support_artifacts, write_json_artifact, runtime_provenance)
from .services import runner as service_runner
from .services.taxonomy import canonical_task_family
from .storage.writer import make_writer

DEFAULT_COHORT_DB = "outputs/cohort_services.db"
PARENT_VERSION = "shared_parent_v1"
PARENTS_VERSION = "parent_checkpoints_v1"


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def prepared_data_hash(value: Any) -> str:
    """Hash prepared features and labels without copying whole numeric pools."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, Mapping):
            digest.update(b"mapping:")
            for key in sorted(item):
                visit(str(key)); visit(item[key])
        elif hasattr(item, "to_numpy"):
            visit(list(map(str, getattr(item, "columns", ()))))
            visit(item.to_numpy())
        elif isinstance(item, DiskImageArray):
            # Identical to the dense-array hash, including split-wide padding.
            digest.update(canonical_json(["array", str(item.dtype), list(item.shape)]).encode())
            progress_at = time.monotonic()
            for index in range(len(item)):
                digest.update(memoryview(item[index]).cast("B"))
                if time.monotonic() - progress_at >= 10:
                    print(f"Prepared-data hash: images={index + 1}/{len(item)}", flush=True)
                    progress_at = time.monotonic()
        elif isinstance(item, np.ndarray):
            digest.update(canonical_json(["array", str(item.dtype), list(item.shape)]).encode())
            if item.dtype.hasobject:
                for entry in item.flat:
                    visit(entry)
            elif item.flags.c_contiguous:
                digest.update(memoryview(item).cast("B"))
            else:
                for chunk in np.nditer(item, flags=["external_loop", "buffered", "zerosize_ok"], order="C", buffersize=65536):
                    digest.update(chunk.tobytes())
        elif isinstance(item, (list, tuple, DiskArraySequence)):
            digest.update(f"sequence:{len(item)}:".encode())
            for entry in item:
                visit(entry)
        else:
            encoded = canonical_json(_clean(item)).encode()
            digest.update(f"scalar:{len(encoded)}:".encode()); digest.update(encoded)

    visit(value)
    return digest.hexdigest()


def _write(db: str, table: str, row: Mapping[str, Any]) -> None:
    writer = make_writer("sqlite", db_path=db)
    writer.start()
    try:
        writer.write_generation_record(table, _clean(row))
        writer.commit()
    finally:
        writer.finish()


def _read(db: str, table: str, key: str, value: str) -> dict | None:
    if not Path(db).is_file():
        return None
    if (table, key) not in {("parent_checkpoints", "parent_checkpoint_id"),
                            ("service_cohorts", "cohort_id"), ("cohort_clients", "service_id")}:
        raise ValueError("invalid generation identity")
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            return None
        row = conn.execute(f"SELECT * FROM {table} WHERE {key}=?", (value,)).fetchone()
        return dict(row) if row else None


def parent_config(recipe: Mapping[str, Any], db: str) -> dict:
    config = {**CONFIG, **recipe["config"]}
    for key, legacy in (("checkpoint_patch_operation_policy", "adaptive"), ("checkpoint_measure_full", True), ("checkpoint_weight_format", "native")):
        config[key] = recipe["config"].get(key, legacy)
    for key in ("service_id", "fl_run_id", "round_id", "client_id", "lineage_id", "dataset_args",
                "max_samples", "fedavg_source_pool_cap", "train_examples", "benchmark_examples"):
        config.pop(key, None)
    config.update(db_path=str(Path(db).resolve()), generation_mode="shared_parent",
                  sample_size=None, max_train_time_s=None, max_eval_time_s=None,
                  timeout_s=None, benchmark_sample_size=None, split_strategy="iid",
                  sample_seed=int(config.get("seed", 42)), training_seed=int(config.get("seed", 42)))
    from .tasks.contracts import uses_task_protocol, VERSION
    if uses_task_protocol(config):
        config["task_contract_version"] = VERSION
    return config


def _dataset_args(config: Mapping[str, Any]) -> dict:
    # Import at call time: the manifest runner dispatches to this module.
    from .cli.run_manifest import _build_dataset_args
    args = _build_dataset_args(dict(config))
    args.pop("max_samples", None)
    args.update(benchmark_sample_size=None, seed=int(config.get("seed", 42)))
    args.pop("train_examples", None)
    args.pop("benchmark_examples", None)
    return args


def _source_ids(meta: Mapping[str, Any], count: int, split: str, name: str) -> list[str]:
    values = meta.get(f"selected_{split}_rows")
    if values is None:
        values = range(count)
    values = list(values)
    if len(values) != count:
        raise ValueError(f"{split} source IDs do not align with prepared examples")
    ids = [f"{name}:{v}" for v in values]
    if len(set(ids)) != count:
        raise ValueError(f"{split} source IDs are not unique")
    return ids


def reserve_evaluation(train_ids: list[str], test_ids: list[str], labels: Any, *,
                       task_family: str, seed: int, contract: Mapping[str, Any], source_groups=None) -> dict:
    if set(train_ids) & set(test_ids):
        raise ValueError("training and evaluation source IDs overlap")
    error = evaluation_size_error(len(test_ids))
    if error:
        raise ValueError(error)
    count = min(512, len(test_ids) // 2)
    rng = np.random.default_rng(seed)
    if task_family == "classification":
        from .data.fedavg_partition import _largest_remainder_counts
        target = np.asarray(labels)
        if target.ndim > 1:
            target = target.argmax(axis=-1)
        classes, sizes = np.unique(target, return_counts=True)
        quotas = _largest_remainder_counts(sizes / sizes.sum(), count, tie_keys=list(map(str, classes)))
        audit = sorted(int(i) for cls, n in zip(classes, quotas)
                       for i in rng.choice(np.flatnonzero(target == cls), int(n), replace=False))
    else:
        audit = sorted(map(int, rng.choice(len(test_ids), count, replace=False)))
    audit_set = set(audit)
    if source_groups:
        from .tasks.reservation import grouped_audit_positions
        audit = grouped_audit_positions(train_ids, test_ids, source_groups, count=count, seed=seed)
        audit_set = set(audit)
    metric = [i for i in range(len(test_ids)) if i not in audit_set]
    if contract.get("task_semantics_hash"):
        from .tasks.contracts import target_fingerprints
        target_labels = target_fingerprints(labels)
    else:
        target_labels = _clean(labels)
    result = {"version": "reserved_evaluation_v1", "training_ids": train_ids,
              "metric_source_ids": test_ids, "metric_source_labels": target_labels,
              "catalogue_metric_ids": [test_ids[i] for i in metric],
              "audit_ids": [test_ids[i] for i in audit],
              "catalogue_positions": metric, "audit_positions": audit,
              "seed": seed, "task_family": task_family, "contract": dict(contract),
              "historical_evaluation_exposure": "unknown"}
    if contract.get("task_semantics_hash"):
        result.update(version="reserved_evaluation_v2",
                      target_encoding="source_row_sha256_v1", source_groups=source_groups)
    result["reservation_hash"] = sha256_json(result)
    return result


def _contracts(model: Any, config: dict, meta: dict, task_family: str) -> dict:
    result = {"identity": model_identity(model, config), "parameter_schema": parameter_schema(model),
            "labels": label_semantics(meta, task_family=task_family),
            "preprocessing": preprocessing_metadata(meta), "dataset": dataset_metadata(config, meta)}
    if config.get("task_contract_version"):
        from .tasks.contracts import bind_task_contract
        result["task"] = bind_task_contract(config, meta, model).as_dict()
    return result


def _compatibility(contracts: dict) -> dict:
    result = {"architecture_id": contracts["identity"]["architecture_id"],
            "parameter_schema_hash": sha256_json(contracts["parameter_schema"]),
            "label_map_hash": contracts["labels"]["label_map_hash"],
            "preprocessing_contract_hash": contracts["preprocessing"]["preprocessing_contract_hash"],
            "logical_dataset_contract_hash": contracts["dataset"]["logical_dataset_contract_hash"]}
    if contracts.get("task"):
        result.update({key: contracts["task"][key] for key in ("task_semantics_hash", "evaluation_protocol_hash")})
    return result


def _prepare(config: dict, *, load_dataset=None, build_model=None):
    config["dataset_args"] = _dataset_args(config)
    load = load_dataset or service_runner._load_dataset
    from .data.tokenization_selection import training_row_selection
    with training_row_selection([config]):
        train, test, meta = load(config.get("dataset", "hf"), **config["dataset_args"],
                                 min_evaluation_examples=MIN_EVALUATION_EXAMPLES)
    error = evaluation_size_error(service_runner._sample_count(*test))
    if error:
        close_image_stores((train, test))
        raise ValueError(error)
    try:
        return _prepare_loaded(config, train, test, meta, build_model=build_model)
    except BaseException:
        close_image_stores((train, test))
        raise


def _prepare_loaded(config, train, test, meta, *, build_model=None):
    meta = dict(meta)
    from .segmentation_policy import apply_segmentation_label_policy, validate_ade20k_mini_metadata
    from .precision_policy import apply_precision_policy
    apply_segmentation_label_policy(config)
    apply_precision_policy(config)
    validate_ade20k_mini_metadata(meta, config)
    meta.setdefault("hf_task", config.get("hf_task"))
    meta.setdefault("hf_model_id", config.get("hf_model_id"))
    from .task_contract import resolve_sentence_pair_contract
    resolve_sentence_pair_contract(config, meta)
    task_type = service_runner._resolved_task_type(config, meta)
    task_family = canonical_task_family(task_type, meta.get("hf_task") or config.get("hf_task"))
    if task_family == "unknown":
        raise ValueError("cannot freeze an unknown task family; resolve the dataset task contract before preparation")
    config["task_type"] = meta["task_type"] = task_type
    if config.get("training_regime") == "generic" and build_model is None:
        import keras
        # Keras auto-generated layer paths are part of the parameter schema.
        # Reconstructing a parent must not depend on earlier models in this process.
        keras.backend.clear_session()
    service_runner.prepare_model_for_local_update_seed(config)
    runner = service_runner.ServiceRunner(config)
    if config.get("training_regime") == "generic":
        # Regularisation is an optimizer setting shared across fixed architecture.
        runner.config["weight_decay"] = 0.0
    model = build_model(config, meta, task_family) if build_model else runner._build_model(meta=meta, task_family=task_family)
    if config.get("task_contract_version"):
        from .tasks.contracts import bind_task_contract, validate_partition
        contract = bind_task_contract(config, meta, model)
        validate_partition(contract, config)
        contract.supervision_count(train[1])
        contract.supervision_count(test[1])
        for split in ("train", "test"):
            if meta.get(f"{split}_group_ids") is not None:
                config.setdefault("task_source_groups", {})[split] = list(map(str, meta[f"{split}_group_ids"]))
    if task_family in {"classification", "regression"}:
        core = getattr(model, "core", None)
        model_config = getattr(getattr(core, "model", None), "config", None)
        if model_config is not None and getattr(model_config, "problem_type", None) is None:
            # Transformers normally infers this on the first labelled forward.
            # Freeze that output/loss semantic before hashing the parent contract.
            model_config.problem_type = (
                "regression" if getattr(model_config, "num_labels", 2) == 1 else
                "multi_label_classification" if meta.get("label_format") in {"multilabel", "multihot"} else
                "single_label_classification")
    train_name = str(meta.get("train_split") or config.get("train_split") or "train")
    test_name = str(meta.get("test_split") or config.get("test_split") or "test")
    if test_name == "train_test_split":
        # These rows retain their original source indices in the HF loader.
        test_name = train_name
    train_ids = _source_ids(meta, service_runner._sample_count(*train), "train", train_name)
    test_ids = _source_ids(meta, service_runner._sample_count(*test), "test", test_name)
    return train, test, meta, task_family, model, train_ids, test_ids


def train_parents(file: str | Path, *, db: str = DEFAULT_COHORT_DB, output: str | Path | None = None,
                  dry_run: bool = False, load_dataset=None, build_model=None) -> dict:
    recipes = selected_recipes(file)
    result: dict = {"version": PARENTS_VERSION, "parents": [], "failures": [],
                    "family_count": len({r["family_id"] for r in recipes}), "parent_count": len(recipes)}
    for parent_index, recipe in enumerate(recipes, start=1):
        config = parent_config(recipe, db)
        requested_id = "parent_" + sha256_json({"parent_id": recipe["parent_id"],
                                               "selection": recipe["selection_hash"], "config": config})[:24]
        if dry_run:
            result["parents"].append({"parent_checkpoint_id": requested_id, "config": config,
                                      "parent_id": recipe["parent_id"], "family_id": recipe["family_id"],
                                      "parent_rank": recipe["parent_rank"], "recipe_id": recipe["recipe_id"],
                                      "status": "planned", "selection_hash": recipe["selection_hash"]})
            continue
        existing = _read(db, "parent_checkpoints", "parent_checkpoint_id", requested_id)
        if existing and existing["status"] == "completed":
            document = json.loads(existing["document_json"])
            verify_parent(document)
            result["parents"].append(document)
            continue
        row = {"parent_checkpoint_id": requested_id, "family_id": recipe["family_id"],
               "selection_hash": recipe["selection_hash"], "status": "preparing", "document_json": {}}
        _write(db, "parent_checkpoints", row)
        print(f"Parent {parent_index}/{len(recipes)} {requested_id}: preparing "
              f"model={config.get('hf_model_id') or config.get('model_type')} "
              f"dataset={config.get('dataset_name') or config.get('dataset')}", flush=True)
        train = test = model = runner = state = base = support = None
        try:
            service_runner._resolve_hf_revisions(config)
            if config.get("dataset") == "hf" and any(not service_runner._is_hf_commit_sha(config.get(k)) for k in ("hf_model_revision", "hf_dataset_revision")):
                raise ValueError("parent training requires resolvable model and dataset revisions")
            train, test, meta, task_family, model, train_ids, test_ids = _prepare(
                config, load_dataset=load_dataset, build_model=build_model)
            contracts = _contracts(model, config, meta, task_family)
            compatibility = _compatibility(contracts)
            root = service_runner._provenance_artifact_root(config)
            reservation = reserve_evaluation(train_ids, test_ids, test[1], task_family=task_family,
                                             seed=int(config["seed"]), contract=compatibility,
                                             source_groups=config.get("task_source_groups"))
            reservation_artifact = write_json_artifact(root, requested_id, "cohort_reservation", reservation)
            support = save_reconstruction_support_artifacts(root, requested_id, model)
            print(f"Parent {requested_id}: hashing prepared data", flush=True)
            data_hash = prepared_data_hash((train, test))
            frozen = {"config": _clean(config), "contracts": contracts, "reservation": reservation,
                      "prepared_data_hash": data_hash,
                      "reconstruction_support": support, "selection": recipe}
            preparation = write_json_artifact(root, requested_id, "parent_preparation", frozen)
            row["document_json"] = {"preparation_ref": preparation["artifact_uri"],
                                    "preparation_hash": preparation["metadata"]["sha256"]}
            _write(db, "parent_checkpoints", {**row, "status": "training"})
            print(f"Parent {requested_id}: training examples={len(train_ids)} "
                  f"epochs={config.get('training_epochs', 1)} batch_size={config.get('batch_size')}", flush=True)
            service_runner.prepare_model_for_local_update(model, config)
            runner = service_runner.ServiceRunner(config)
            metrics = runner._train_model(model, *train, task_family=task_family)
            if metrics.get("train_stopped_early") or metrics.get("optimizer_step_count", 1) < 1:
                raise ValueError("parent training did not complete the selected epoch schedule")
            metric_positions = np.asarray(reservation["catalogue_positions"])
            print(f"Parent {requested_id}: evaluating examples={len(metric_positions)}", flush=True)
            loss, primary, secondary, eval_metrics = runner._evaluate_model(
                model, _take_rows(test[0], metric_positions), _take_rows(test[1], metric_positions),
                inference_only=False, task_family=task_family)
            if not math.isfinite(float(primary)):
                raise ValueError("parent evaluation did not produce finite quality")
            if _compatibility(_contracts(model, config, meta, task_family)) != compatibility:
                raise ValueError("parent training changed the compatibility contract")
            lineage = "lineage_" + requested_id
            trained_hash = capture_model_tensor_state(model).sha256
            base = ensure_lineage_base(root, lineage, model, parameter_schema_hash=compatibility["parameter_schema_hash"],
                                       compression=service_runner._checkpoint_compression(config))
            if base.tensor_state_sha256 != trained_hash:
                raise ValueError("retraining differs from an already persisted parent; retain the original parent or select a new seed")
            state = load_lineage_base(root, lineage, expected_parameter_schema_hash=compatibility["parameter_schema_hash"])
            restore_model_tensor_state(model, state, verify=True)
            metric_name, _ = quality_metric(config)
            document = _clean({"version": PARENT_VERSION, "parent_checkpoint_id": requested_id,
                "parent_id": recipe["parent_id"], "selection": recipe,
                "family_id": recipe["family_id"], "selection_hash": recipe["selection_hash"],
                "status": "completed", "lineage_id": lineage, "store_root": str(Path(root).resolve()),
                "parent_state_hash": state.sha256, "base_ref": str(base.base.manifest_path),
                "config": config, "contracts": contracts, "compatibility": compatibility,
                "compatibility_contract_hash": sha256_json(compatibility), "reservation": reservation,
                "reservation_ref": reservation_artifact["artifact_uri"],
                "reservation_file_hash": reservation_artifact["metadata"]["sha256"],
                "preparation_ref": preparation["artifact_uri"], "preparation_hash": preparation["metadata"]["sha256"],
                "prepared_data_hash": data_hash, "reconstruction_support": support,
                "training_metrics": metrics, "evaluation_metrics": eval_metrics,
                "quality": {"metric": metric_name, "value": primary, "secondary": secondary, "loss": loss,
                            "historical_median": recipe["median_quality"], "quality_gate": "report_only",
                            "comparison": "new_full_pool_baseline; historical evaluation may differ"},
                "historical_service_ids": recipe["historical_service_ids"], "runtime": runtime_provenance()})
            artifact = write_json_artifact(root, requested_id, "parent_checkpoint", document)
            document.update(document_ref=artifact["artifact_uri"], document_hash=artifact["metadata"]["sha256"])
            _write(db, "parent_checkpoints", {**row, "status": "completed", "parent_state_hash": state.sha256,
                                               "document_json": document, "error_message": None})
            result["parents"].append(document)
            print(f"Parent {requested_id}: {metric_name}={primary:.6g}; training examples={len(train_ids)}")
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            _write(db, "parent_checkpoints", {**row, "status": status, "error_message": str(exc)})
            if status == "interrupted":
                raise
            result["failures"].append({"parent_checkpoint_id": requested_id, "error": str(exc)})
            print(f"Parent {requested_id} failed: {exc}")
        finally:
            close_image_stores((train, test))
            train = test = model = runner = state = base = support = None
            # Release cycles before asking the already-loaded backends to return
            # their cached allocations. Do not import another framework here.
            gc.collect()
            keras = sys.modules.get("keras")
            if keras is not None and config.get("training_regime") == "generic":
                try:
                    keras.backend.clear_session()
                except Exception:
                    pass
            torch = sys.modules.get("torch")
            if torch is not None:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    if not dry_run:
        target = Path(output) if output else Path(file).with_suffix(".parents.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def verify_parent(document: Mapping[str, Any]) -> None:
    if document.get("version") != PARENT_VERSION or document.get("status") != "completed":
        raise ValueError("a completed shared_parent_v1 checkpoint is required")
    stored = json.loads(Path(document["document_ref"]).read_text(encoding="utf-8"))
    if sha256_file(document["document_ref"]) != document["document_hash"]:
        raise ValueError("parent document failed integrity verification")
    if stored != {k: v for k, v in document.items() if k not in {"document_ref", "document_hash"}}:
        raise ValueError("parent document was changed after publication")
    if sha256_file(document["reservation_ref"]) != document["reservation_file_hash"]:
        raise ValueError("cohort reservation failed integrity verification")
    if document.get("preparation_hash") and sha256_file(document["preparation_ref"]) != document["preparation_hash"]:
        raise ValueError("parent preparation evidence failed integrity verification")
    for artifact in document.get("reconstruction_support", ()):
        if path_integrity(artifact["artifact_uri"])["sha256"] != artifact["metadata"]["sha256"]:
            raise ValueError("parent preprocessing/model assets failed integrity verification")
    state = load_lineage_base(document["store_root"], document["lineage_id"],
                              expected_parameter_schema_hash=document["compatibility"]["parameter_schema_hash"])
    if state.sha256 != document["parent_state_hash"]:
        raise ValueError("parent checkpoint state hash mismatch")


def load_parent_reference(path: str, digest: str) -> dict:
    if sha256_file(path) != digest:
        raise ValueError("parent reference hash mismatch")
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    document.update(document_ref=path, document_hash=digest)
    verify_parent(document)
    return document


def execute_cohort(entries: list[Any], *, load_dataset=None, build_model=None, execute_service=None) -> list[dict]:
    from .cohort_manifest import validate_cohort_entries
    if not entries:
        return []
    ordered = sorted(entries, key=lambda e: e.resolved["client_id"])
    validate_cohort_entries(ordered)
    first = ordered[0].resolved
    db = str(first.get("db_path") or DEFAULT_COHORT_DB)
    document = load_parent_reference(first["parent_document_ref"], first["parent_document_hash"])
    cohort_id = first["cohort_id"]
    keys = ("cohort_id", "parent_document_ref", "parent_document_hash", "cohort_manifest_hash", "cohort_size")
    if any(any(e.resolved.get(k) != first.get(k) for k in keys) for e in ordered):
        raise ValueError("cohort rows disagree on frozen identity")
    expected = int(first["cohort_size"])
    if len(ordered) != expected or len({e.resolved["service_id"] for e in ordered}) != expected:
        raise ValueError("execute the complete frozen cohort manifest; completed clients are resumed automatically")
    if sha256_json(sorted(e.resolved["client_recipe_hash"] for e in ordered)) != first["cohort_manifest_hash"]:
        raise ValueError("cohort membership differs from the frozen manifest")
    _write(db, "parent_checkpoints", {"parent_checkpoint_id": document["parent_checkpoint_id"],
        "family_id": document["family_id"], "selection_hash": document["selection_hash"], "status": "completed",
        "parent_state_hash": document["parent_state_hash"], "document_json": document})
    config = dict(document["config"])
    config["provenance_artifact_dir"] = document["store_root"]
    train, test, meta, task_family, model, train_ids, test_ids = _prepare(config, load_dataset=load_dataset, build_model=build_model)
    reservation = document["reservation"]
    from .tasks.contracts import target_fingerprints
    loaded_targets = target_fingerprints(test[1]) if reservation.get("version") == "reserved_evaluation_v2" else _clean(test[1])
    if train_ids != reservation["training_ids"] or test_ids != reservation["metric_source_ids"] or loaded_targets != reservation["metric_source_labels"]:
        raise ValueError("reloaded dataset differs from the frozen parent reservation")
    if document.get("prepared_data_hash") and prepared_data_hash((train, test)) != document["prepared_data_hash"]:
        raise ValueError("reloaded features or labels differ from the frozen parent data")
    if _compatibility(_contracts(model, config, meta, task_family)) != document["compatibility"]:
        raise ValueError("reloaded model/data compatibility differs from parent")
    state = load_lineage_base(document["store_root"], document["lineage_id"])
    base = ensure_lineage_base(document["store_root"], document["lineage_id"], model,
                               parameter_schema_hash=document["compatibility"]["parameter_schema_hash"])
    positions = np.asarray(reservation["catalogue_positions"])
    benchmark = (_take_rows(test[0], positions), _take_rows(test[1], positions))
    partitions = {e.resolved["client_id"]: client_partition(*train, e.resolved, source_ids=train_ids,
                    meta=meta, task_family=task_family) for e in ordered}
    used: set[str] = set()
    exposures = 0
    for partition in partitions.values():
        used.update(partition["metadata"]["source_ids"])
        exposures += partition["metadata"]["effective_quota"]
    cohort_row = {"cohort_id": cohort_id, "parent_checkpoint_id": document["parent_checkpoint_id"],
        "manifest_hash": first["cohort_manifest_hash"], "scheduled_clients": expected, "status": "running",
        "document_json": {"unique_training_examples": len(used), "local_example_exposures": exposures,
                          "reused_example_exposures": exposures - len(used), "parent_training_examples": len(train_ids)}}
    _write(db, "service_cohorts", cohort_row)
    for entry in ordered:
        row = entry.resolved
        partition = partitions[row["client_id"]]
        if _read(db, "cohort_clients", "service_id", row["service_id"]) is None:
            _write(db, "cohort_clients", {"service_id": row["service_id"], "cohort_id": cohort_id,
                "client_id": row["client_id"], "recipe_hash": row["client_recipe_hash"], "status": "pending",
                "sample_count": len(partition["indices"]), "partition_hash": partition["metadata"]["partition_hash"],
                "document_json": {"partition": partition["metadata"]}})
    run = execute_service or service_runner.execute_service
    results = []
    try:
        for entry in ordered:
            row = dict(entry.resolved)
            sid, client = row["service_id"], row["client_id"]
            partition = partitions[client]
            previous = _read(db, "cohort_clients", "service_id", sid)
            if previous and previous.get("final_state_hash"):
                try:
                    restored = reconstruct_service_state(document["store_root"], sid)
                    if source_state_hash(document["store_root"], sid, restored) != previous["final_state_hash"]:
                        raise ValueError("completed client checkpoint state hash mismatch")
                except Exception as exc:
                    error = f"checkpoint_integrity: {exc}; restore the published artifact before retrying"
                    _write(db, "cohort_clients", {**previous, "status": "failed", "error_message": error})
                    results.append({"service_id": sid, "status": "failed", "error_message": error})
                else:
                    _write(db, "cohort_clients", {**previous, "status": "completed", "error_message": None})
                    results.append({"service_id": sid, "status": "skipped", "error_message": "already completed"})
                continue
            client_row = {"service_id": sid, "cohort_id": cohort_id, "client_id": client,
                "recipe_hash": row["client_recipe_hash"], "status": "running", "error_message": None,
                "sample_count": len(partition["indices"]), "partition_hash": partition["metadata"]["partition_hash"],
                "document_json": {"partition": partition["metadata"]}}
            _write(db, "cohort_clients", client_row)
            try:
                reset_timer = service_runner._StageTimer()
                restore_model_tensor_state(model, state, verify=True)
                reset_s = reset_timer.elapsed_s()
                row.update(db_path=db, lineage_id=document["lineage_id"], max_eval_time_s=None,
                    benchmark_sample_size=None, provenance_artifact_dir=document["store_root"],
                    dataset_args=config["dataset_args"], _prepared_model=model,
                    _prepared_dataset=(train, benchmark, meta),
                    _prepared_train_partition=materialize_partition(*train, partition),
                    _prepared_benchmark_sample_info={"source_ids": reservation["catalogue_metric_ids"], "frozen": True},
                    _parent_state=state, _parent_ref=base.base, _parent_lineage_base=base,
                    _restored_parent=state, _model_reset_s=reset_s,
                    _expected_compatibility=document["compatibility"], _retain_client_checkpoint=True,
                    training_universe_hash=sha256_json(train_ids),
                    benchmark_cohort_hash=sha256_json(reservation["catalogue_metric_ids"]))
                summary = run(row)
                if summary.status != "success":
                    raise ValueError(summary.error or "client execution failed")
                final = reconstruct_service_state(document["store_root"], sid)
                if source_state_hash(document["store_root"], sid, final) != capture_model_tensor_state(model).sha256:
                    raise ValueError("published client checkpoint differs from trained model")
                client_row.update(status="completed", final_state_hash=source_state_hash(document["store_root"], sid, final),
                                  document_json={"partition": partition["metadata"], "metrics": _clean(summary.metrics or {})})
                _write(db, "cohort_clients", client_row)
                results.append({"service_id": sid, "status": "success", "error_message": ""})
            except Exception as exc:
                _write(db, "cohort_clients", {**client_row, "status": "failed", "error_message": str(exc)})
                results.append({"service_id": sid, "status": "failed", "error_message": str(exc)})
            except BaseException as exc:
                _write(db, "cohort_clients", {**client_row, "status": "interrupted", "error_message": str(exc)})
                raise
            print(f"Cohort {cohort_id}: {len(results)}/{expected} client={client} status={results[-1]['status']}")
    except BaseException:
        _write(db, "service_cohorts", {**cohort_row, "status": "interrupted"})
        raise
    successes = sum(r["status"] in {"success", "skipped"} for r in results)
    _write(db, "service_cohorts", {**cohort_row, "status": "completed" if successes == expected else "partial" if successes else "failed",
        "document_json": {**cohort_row["document_json"], "successful_clients": successes, "failed_clients": expected - successes}})
    return results
