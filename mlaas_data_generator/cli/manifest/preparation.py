"""Durable preparation for the existing independent manifest executor.

This is checkpoint/data infrastructure, not another generation mode. Both grouped
and row-local execution use the same preparation and ServiceRunner.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from ...terminal import debug_enabled, print_service_summary

import numpy as np
from filelock import FileLock

from ... import shared_parent as shared
from ...reduced_checkpoint import source_state_hash
from ...checkpoint_store import (capture_model_tensor_state, ensure_lineage_base,
    load_lineage_base, restore_model_tensor_state, reconstruct_service_state)
from ...data.fedavg_partition import _take_rows
from ...data.image_store import close_image_stores
from ...family_preflight import (MIN_EVALUATION_EXAMPLES, evaluation_size_error,
    hf_split_error, independent_family_applies, manifest_family_error)
from ...provenance import sha256_json, sha256_file, write_json_artifact, save_reconstruction_support_artifacts, path_integrity
from ...services import runner
from ...execution_state import persist_terminal

VERSION = "independent_family_v1"
TASK_VERSION = "independent_family_v2"
SUPPORTED_VERSIONS = (VERSION, TASK_VERSION)


def applies(config):
    from ...debug_mode import weights_disabled
    return independent_family_applies(config) and not weights_disabled(config)


def contract_config(config):
    from ...task_contract import resolve_sentence_pair_contract
    config = resolve_sentence_pair_contract(dict(config))
    from ...segmentation_policy import apply_segmentation_label_policy
    from ...precision_policy import apply_precision_policy, SEGFORMER_MINI_FP32_REASON
    apply_segmentation_label_policy(config)
    apply_precision_policy(config)
    from ..run_manifest import _build_dataset_args
    args = _build_dataset_args(dict(config))
    for key in ("max_samples", "sample_size", "sample_seed", "train_examples", "benchmark_examples", "benchmark_sample_size", "seed"):
        args.pop(key, None)
    keys = ("dataset", "dataset_name", "dataset_config", "model_type", "hf_model_id", "hf_model_revision",
            "hf_dataset_revision", "hf_task", "task", "task_type", "task_tag", "training_regime", "modality",
            "input_schema", "output_schema", "hidden_layers", "activation", "dropout", "num_classes", "num_labels",
            "mobilebert_head_normalization")
    result = {key: config[key] for key in keys if config.get(key) is not None}
    result.update({key: value for key, value in args.items() if value is not None})
    result.update(initialization_seed=int(config.get("initialization_seed", config.get("seed", 42))),
                  evaluation_seed=int(config.get("evaluation_seed", config.get("seed", 42))))
    if (config.get("precision_policy") == SEGFORMER_MINI_FP32_REASON
            or config.get("mobilebert_head_normalization") == "layer_norm_v1"):
        # Preparation builds the initial evaluator before client runtime knobs
        # are applied. Freeze the reviewed precision for that stage as well.
        result.update({key: config[key] for key in ("precision_type", "mixed_precision", "precision_policy") if key in config})
    from ...tasks.contracts import uses_task_protocol, VERSION as TASK_CONTRACT_VERSION
    if uses_task_protocol(config):
        result["task_contract_version"] = TASK_CONTRACT_VERSION
        for key in ("task_source_groups", "generation_config", "max_new_tokens", "num_beams", "do_sample"):
            if key in config:
                result[key] = config[key]
    return shared._clean(result)


def group_key(config):
    contract = contract_config(config)
    from ...segmentation_policy import is_ade20k_mini, ADE20K_MINI_CONTRACT_VERSION
    if is_ade20k_mini(contract):
        return sha256_json({"segmentation_label_contract": ADE20K_MINI_CONTRACT_VERSION, "config": contract})
    from ...hf_tasks import normalize_hf_task
    if normalize_hf_task(contract.get("hf_task"), default="unknown", unknown="unknown") == "sequence_classification":
        # Old text preparations hashed an unknown label task. Do not rediscover
        # them through the implicit cache after correcting task normalization.
        return sha256_json({"text_task_semantics_version": 2, "config": contract})
    return sha256_json(contract)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(shared._clean(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def bind_manifest(config, manifest_path, *, cache=None):
    """Bind a recipe before revision resolution, and reuse a previous binding."""
    if not applies(config):
        return
    ignored = {"db_path", "manifest_group_id", "device", "provenance_artifact_dir", "no_save_weights"}
    recipe = {k: v for k, v in config.items() if k not in ignored and not k.startswith("_")}
    config["_manifest_recipe_hash"] = sha256_json(shared._clean(recipe))
    directory = Path(manifest_path).resolve().with_suffix(".resolved")
    reference = directory / (sha256_json(str(config["service_id"])) + ".json")
    config["_manifest_reference"] = str(reference)
    if config.get("family_document_ref") and config.get("family_document_hash"):
        config["_family_reference"] = {"document_ref": config["family_document_ref"], "document_hash": config["family_document_hash"]}
    if reference.exists():
        saved = json.loads(reference.read_text(encoding="utf-8"))
        if saved["recipe_hash"] != config["_manifest_recipe_hash"]:
            raise ValueError("manifest recipe changed after preparation; use a new service ID and manifest")
        if saved.get("document_ref"):
            config["_family_reference"] = saved
        else:
            config.update(saved.get("resolved_revisions", {}))
            config["_frozen_revisions"] = True
    if config.get("_family_reference"):
        saved = config["_family_reference"]
        document = read_preparation(saved["document_ref"], saved["document_hash"], cache=cache)
        from ...segmentation_policy import validate_ade20k_mini_preparation
        validate_ade20k_mini_preparation(document)
        for key in ("hf_model_revision", "hf_dataset_revision"):
            if document["config"].get(key):
                config[key] = document["config"][key]


def freeze_manifest_revisions(config):
    """Persist source pins for every row before any execution is dispatched."""
    if not config.get("_manifest_reference") or config.get("_family_reference") or config.get("_frozen_revisions"):
        return
    revisions = {k: config[k] for k in ("hf_model_revision", "hf_dataset_revision") if config.get(k)}
    if config.get("dataset") == "hf" and not runner._is_hf_commit_sha(revisions.get("hf_dataset_revision")):
        return
    _atomic_json(config["_manifest_reference"], {"recipe_hash": config["_manifest_recipe_hash"], "resolved_revisions": revisions})


def load_prepared(config, *, load_dataset=None, build_model=None):
    from ..run_manifest import _build_dataset_args
    cfg = dict(config)
    from ...precision_policy import apply_precision_policy
    from ...segmentation_policy import apply_segmentation_label_policy
    apply_segmentation_label_policy(cfg)
    apply_precision_policy(cfg)
    error = manifest_family_error(cfg)
    if error:
        raise ValueError(error)
    cfg.update(seed=int(cfg.get("evaluation_seed", cfg.get("seed", 42))),
               training_seed=int(cfg.get("initialization_seed", cfg.get("seed", 42))),
               sample_size=None, weight_decay=0.0)
    cfg.pop("max_samples", None)
    cfg["dataset_args"] = _build_dataset_args(cfg)
    cfg["dataset_args"].update(seed=cfg["seed"], benchmark_sample_size=None)
    for key in ("train_examples", "benchmark_examples"):
        cfg["dataset_args"].pop(key, None)
    if cfg.get("dataset") == "hf":
        split = str(cfg.get("train_split") or "train")
        error = hf_split_error(split, cfg["dataset_args"].get("test_split"))
        if error:
            raise ValueError(error)
        cfg["dataset_args"]["train_split"] = split
    train = test = None
    try:
        from ...data.tokenization_selection import training_row_selection, active_selection
        active = active_selection()
        if active is not None and cfg.get("client_selection_configs"):
            active.configs = cfg["client_selection_configs"]
        scope = nullcontext() if active is not None else training_row_selection(cfg.get("client_selection_configs", [cfg]))
        with scope:
            train, test, meta = (load_dataset or runner._load_dataset)(cfg.get("dataset", "hf"), **cfg["dataset_args"],
                min_evaluation_examples=MIN_EVALUATION_EXAMPLES, require_official_train_split=cfg.get("dataset") == "hf")
        error = evaluation_size_error(runner._sample_count(*test), split=str(meta.get("test_split", "test")), stage="after preprocessing")
        if error:
            raise ValueError(error)
        if cfg.get("dataset") == "hf" and str(meta.get("train_split", split)) != split:
            raise ValueError("loader substituted the official training split")
        if cfg.get("dataset") == "hf":
            error = hf_split_error(split, meta.get("test_split", cfg["dataset_args"].get("test_split")))
            if error:
                raise ValueError(error)
        prepared = shared._prepare_loaded(cfg, train, test, meta, build_model=build_model)
        return cfg, prepared
    except BaseException:
        close_image_stores((train, test))
        raise


def read_preparation(path, digest, *, cache=None):
    key = (str(path), str(digest))
    if cache is not None and key in cache:
        return cache[key]
    if sha256_file(path) != digest:
        raise ValueError("family preparation failed integrity verification")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("contracts", {}).get("labels", {}).get("task_family") == "unknown":
        raise ValueError("frozen preparation has an unknown task family; regenerate with a new service ID and manifest")
    if doc.get("version") not in SUPPORTED_VERSIONS or doc["family_id"] != "family_" + sha256_json(doc["family_contract"]):
        raise ValueError("invalid frozen family identity")
    contract = doc["family_contract"]
    if (doc["compatibility_contract_hash"] != sha256_json(contract)
            or any(doc[key] != contract[key] for key in ("initial_state_hash", "compatibility", "prepared_data_hash"))
            or doc["reservation"]["reservation_hash"] != contract["reservation_hash"]):
        raise ValueError("family document disagrees with its identity contract")
    if sha256_file(doc["reservation_ref"]) != doc["reservation_file_hash"]:
        raise ValueError("family evaluation reservation changed")
    if json.loads(Path(doc["reservation_ref"]).read_text(encoding="utf-8")) != doc["reservation"]:
        raise ValueError("family evaluation reservation disagrees with preparation")
    for artifact in doc.get("reconstruction_support", ()):
        if path_integrity(artifact["artifact_uri"])["sha256"] != artifact["metadata"]["sha256"]:
            raise ValueError("family reconstruction support changed")
    state = load_lineage_base(doc["store_root"], doc["lineage_id"], expected_parameter_schema_hash=doc["compatibility"]["parameter_schema_hash"])
    if state.sha256 != doc["initial_state_hash"]:
        raise ValueError("family initial checkpoint changed")
    if cache is not None:
        cache[key] = doc
    return doc


def verify_loaded(doc, prepared, cfg):
    train, test, meta, task, model, train_ids, test_ids = prepared
    if (train_ids != doc["reservation"]["training_ids"] or test_ids != doc["reservation"]["metric_source_ids"]
            or shared.prepared_data_hash((train, test)) != doc["prepared_data_hash"]
            or shared._compatibility(shared._contracts(model, cfg, meta, task)) != doc["compatibility"]):
        raise ValueError("reloaded model, data, or task interface differs from frozen family")


def prepare(config, *, load_dataset=None, build_model=None, selection_configs=None):
    error = manifest_family_error(config)
    if error:
        raise ValueError(error)
    reference = config.get("_family_reference")
    root = Path(runner._provenance_artifact_root(config)).resolve()
    # Stage A/B still uses the existing prepared-data family format. Its cache
    # must distinguish selected client unions until Stage C replaces that format.
    selection_keys = ("service_id", "client_id", "client_partition_index", "sample_size", "sample_frac",
        "sample_seed", "seed", "split_strategy", "distribution_type", "distribution_param", "skew_axis",
        "skew_axis_config", "heterogeneity_intensity", "shard_support_quantile", "hf_task", "task_type",
        "partition_generator_version", "training_regime", "regression_support_policy", "min_target_bin_fraction",
        "training_policy", "recipe_sampler_version", "calibration_band", "task_contract")
    selection_configs = [{k: c[k] for k in selection_keys if c.get(k) is not None}
                         for c in (selection_configs or [config])]
    index_key = sha256_json({"contract": group_key(config), "raw_selection_v1": selection_configs})
    index = root / "manifest_preparations" / (index_key + ".json")
    index.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(index) + ".lock"):
        if reference is None and index.exists():
            reference = json.loads(index.read_text(encoding="utf-8"))
        doc = read_preparation(reference["document_ref"], reference["document_hash"]) if reference else None
        if doc:
            from ...segmentation_policy import validate_ade20k_mini_preparation
            validate_ade20k_mini_preparation(doc)
        if doc and group_key(config) != group_key(doc["config"]):
            raise ValueError("manifest contract differs from its frozen preparation")
        cfg = dict(doc["config"] if doc else contract_config(config))
        if doc and not cfg.get("client_selection_configs"):
            raise ValueError("full-pool family predates raw client selection; use a new manifest and service IDs")
        cfg.setdefault("client_selection_configs", selection_configs)
        cfg.update(device=config.get("device", "auto"), db_path=config.get("db_path"),
                   provenance_artifact_dir=str(root))
        if cfg.get("dataset") == "hf" and not doc:
            runner._resolve_hf_revisions(cfg)
            for key in ("hf_model_revision", "hf_dataset_revision"):
                local_model = key == "hf_model_revision" and Path(str(cfg.get("hf_model_id", ""))).is_dir()
                if not local_model and not runner._is_hf_commit_sha(cfg.get(key)):
                    raise ValueError("family preparation requires resolvable model and dataset revisions")
        cfg, prepared = load_prepared(cfg, load_dataset=load_dataset, build_model=build_model)
        try:
            train, test, meta, task, model, train_ids, test_ids = prepared
            if doc:
                verify_loaded(doc, prepared, cfg)
            else:
                contracts = shared._contracts(model, cfg, meta, task)
                compatibility = shared._compatibility(contracts)
                reservation = shared.reserve_evaluation(train_ids, test_ids, test[1], task_family=task,
                    seed=cfg["evaluation_seed"], contract=compatibility, source_groups=cfg.get("task_source_groups"))
                initial = capture_model_tensor_state(model)
                data_hash = shared.prepared_data_hash((train, test))
                version = TASK_VERSION if cfg.get("task_contract_version") else VERSION
                contract = {"version": version, "initial_state_hash": initial.sha256,
                    "compatibility": compatibility, "reservation_hash": reservation["reservation_hash"],
                    "prepared_data_hash": data_hash}
                family = "family_" + sha256_json(contract)
                lineage = "lineage_" + family
                ensure_lineage_base(root, lineage, model, parameter_schema_hash=compatibility["parameter_schema_hash"])
                reservation_artifact = write_json_artifact(root, family, "family_reservation", reservation)
                doc = {"version": version, "family_id": family, "family_contract": contract,
                    "initial_state_hash": initial.sha256, "lineage_id": lineage, "store_root": str(root),
                    "config": shared._clean(cfg), "contracts": contracts, "compatibility": compatibility,
                    "compatibility_contract_hash": sha256_json(contract), "reservation": reservation,
                    "reservation_ref": reservation_artifact["artifact_uri"],
                    "reservation_file_hash": reservation_artifact["metadata"]["sha256"],
                    "prepared_data_hash": data_hash,
                    "reconstruction_support": save_reconstruction_support_artifacts(root, family, model)}
                artifact = write_json_artifact(root, family, "family_preparation", shared._clean(doc))
                reference = {"document_ref": artifact["artifact_uri"], "document_hash": artifact["metadata"]["sha256"]}
                _atomic_json(index, reference)
            state = load_lineage_base(doc["store_root"], doc["lineage_id"])
            restore_model_tensor_state(model, state, verify=True)
            base = ensure_lineage_base(doc["store_root"], doc["lineage_id"], model,
                parameter_schema_hash=doc["compatibility"]["parameter_schema_hash"])
            return doc, reference, cfg, prepared, state, base
        except BaseException:
            close_image_stores(prepared[:2])
            raise


def execute_entries(entries, *, load_dataset=None, build_model=None, execute_service=None,
                    on_start=None, on_result=None):
    """Execute one contract group using the ordinary service runner."""
    if not entries:
        return []
    from ...data.tokenization_selection import training_row_selection
    with training_row_selection([e.resolved for e in entries]) as selection:
        doc, reference, cfg, prepared, state, base = prepare(entries[0].resolved,
            load_dataset=load_dataset, build_model=build_model,
            selection_configs=[e.resolved for e in entries])
    train, test, meta, task, model, _, _ = prepared
    reservation = doc["reservation"]
    positions = np.asarray(reservation["catalogue_positions"], dtype=int)
    benchmark = (_take_rows(test[0], positions), _take_rows(test[1], positions))
    results = []
    try:
        # Bind the entire group before the first client can train or interrupt.
        bound_cache = {}
        for entry in entries:
            row = entry.resolved
            existing = row.get("_family_reference")
            if existing and read_preparation(existing["document_ref"], existing["document_hash"], cache=bound_cache)["family_id"] != doc["family_id"]:
                raise ValueError("group contains conflicting frozen family preparations")
            if row.get("_manifest_reference"):
                _atomic_json(row["_manifest_reference"], {**reference, "recipe_hash": row.get("_manifest_recipe_hash"), "family_id": doc["family_id"]})
        for entry in entries:
            if on_start is not None:
                on_start(entry)
            row = dict(entry.resolved)
            sid = row["service_id"]
            concise = not debug_enabled(row)
            row.update(_family_ordinal=len(results) + 1, _family_total=len(entries),
                       _defer_terminal_results=concise, family_id=doc["family_id"])
            summary = None
            restore_verify_s = None
            try:
                if model is None:
                    runner._clear_cuda_cache_if_available()
                    model = shared._prepare_loaded(cfg, train, test, meta, build_model=build_model)[4]
                    if shared._compatibility(shared._contracts(model, cfg, meta, task)) != doc["compatibility"]:
                        raise ValueError("rebuilt model differs from frozen family interface")
                reset_timer = runner._StageTimer()
                restore_model_tensor_state(model, state, verify=True)
                reset_s = reset_timer.elapsed_s()
                row.update({k: cfg[k] for k in ("hf_model_revision", "hf_dataset_revision") if cfg.get(k)})
                row.update({k: cfg[k] for k in ("task_contract_version", "task_contract", "task_source_groups") if k in cfg})
                row.update(family_id=doc["family_id"], family_contract_version=doc["version"],
                    family_document_ref=reference["document_ref"], family_document_hash=reference["document_hash"],
                    manifest_reference=row.get("_manifest_reference"), initial_state_hash=state.sha256,
                    lineage_id=doc["lineage_id"], parent_state_hash=state.sha256,
                    compatibility_contract_hash=doc["compatibility_contract_hash"],
                    training_seed=int(next((row[k] for k in ("training_seed", "sample_seed", "seed") if row.get(k) is not None), 42)),
                    _family_document=doc, _prepared_model=model, _prepared_dataset=(train, benchmark, meta),
                    _prepared_benchmark_sample_info={"source_ids": reservation["catalogue_metric_ids"], "frozen": True},
                    _parent_state=state, _parent_ref=base.base, _parent_lineage_base=base,
                    _restored_parent=state, _model_reset_s=reset_s,
                    _expected_compatibility=doc["compatibility"], _retain_client_checkpoint=True,
                    provenance_artifact_dir=doc["store_root"], dataset_args=cfg["dataset_args"],
                    max_eval_time_s=None, benchmark_sample_size=None,
                    training_universe_hash=sha256_json(reservation["training_ids"]),
                    benchmark_cohort_hash=sha256_json(reservation["catalogue_metric_ids"]))
                entry.resolved.update({k: row[k] for k in ("family_id", "family_contract_version", "initial_state_hash", "family_document_ref", "family_document_hash")})
                entry.resolved.update({k: row[k] for k in ("hf_model_revision", "hf_dataset_revision") if row.get(k)})
                selected_split = selection.materialize(sid, train)
                if selected_split is not None:
                    row["_pretokenization_split_info"] = selected_split
                summary = (execute_service or runner.execute_service)(row)
                if summary.status != "success":
                    raise ValueError(summary.error or "service training failed")
                verify_timer = runner._StageTimer()
                final = reconstruct_service_state(doc["store_root"], sid)
                if source_state_hash(doc["store_root"], sid, final) != capture_model_tensor_state(model).sha256:
                    raise ValueError("saved checkpoint differs from trained service")
                verify_loaded_contract = shared._compatibility(shared._contracts(model, cfg, meta, task))
                if verify_loaded_contract != doc["compatibility"]:
                    raise ValueError("training changed the frozen task interface")
                restore_verify_s = verify_timer.elapsed_s()
                with sqlite3.connect(row["db_path"]) as conn:
                    conn.execute("UPDATE service_generation_provenance SET composition_compatible=1 WHERE service_id=?", (sid,))
                results.append({"service_id": sid, "status": "success", "error_message": ""})
            except BaseException as exc:
                model = None
                status = "failed" if isinstance(exc, Exception) else "interrupted"
                if summary is None or summary.status == "success" or status == "interrupted":
                    persist_terminal(row, status=status, reason=str(exc) or type(exc).__name__,
                                     failure_stage="family_client_verification", exc=exc)
                if status == "interrupted":
                    raise
                if Path(str(row.get("db_path", ""))).is_file():
                    with sqlite3.connect(row["db_path"]) as conn:
                        conn.execute("UPDATE service_generation_provenance SET composition_compatible=0 WHERE service_id=?", (sid,))
                results.append({"service_id": sid, "status": "failed", "error_message": str(exc)})
            if concise:
                print_service_summary(row, getattr(summary, "metrics", {}) or {},
                                      getattr(summary, "checkpoint_storage", None),
                                      status=results[-1]["status"], error=results[-1]["error_message"],
                                      restore_verify_s=restore_verify_s)
            else:
                print(f"Family {doc['family_id'][:23]}: {len(results)}/{len(entries)} service={sid} status={results[-1]['status']}", flush=True)
            if on_result is not None:
                on_result(entry, results[-1])
    finally:
        close_image_stores((train, test))
    return results


def verify_completed(db, service_id, *, invalidate=True, cache=None):
    """Historical records remain readable; new family records require evidence."""
    with sqlite3.connect(str(db)) as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(service_generation_provenance)")}
        if "family_document_ref" not in columns:
            return
        conn.row_factory = sqlite3.Row
        raw = conn.execute("SELECT * FROM service_generation_provenance WHERE service_id=?", (service_id,)).fetchone()
        if raw is None or not raw["family_document_ref"]:
            return
        from ...tasks.contracts import provenance_contract_values
        row = provenance_contract_values(raw)
        try:
            doc = read_preparation(row["family_document_ref"], row["family_document_hash"], cache=cache)
            if (row["initial_state_hash"] != doc["initial_state_hash"]
                    or row["family_id"] != doc["family_id"] or row["compatibility_contract_hash"] != doc["compatibility_contract_hash"]):
                raise ValueError("completed service disagrees with family/checkpoint provenance")
            if any(row.get(key) != value for key, value in doc["compatibility"].items()):
                raise ValueError("completed service task interface differs from frozen family")
            from ...checkpoint_verification import checkpoint_bytes_verified
            if checkpoint_bytes_verified(conn, service_id, row, doc, cache=cache):
                return
            print(f"Checkpoint verification: {service_id} requires full reconstruction", flush=True)
            state = reconstruct_service_state(doc["store_root"], service_id)
            if source_state_hash(doc["store_root"], service_id, state) != row["final_model_hash"]:
                raise ValueError("completed service disagrees with family/checkpoint provenance")
        except Exception:
            if invalidate:
                conn.execute("UPDATE service_generation_provenance SET composition_compatible=0 WHERE service_id=?", (service_id,))
                conn.commit()
            raise
