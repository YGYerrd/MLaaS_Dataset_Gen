"""Durable preparation for the existing independent manifest executor.

This is checkpoint/data infrastructure, not another generation mode. Both grouped
and row-local execution use the same preparation and ServiceRunner.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from copy import deepcopy
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
from ...data.source_contract import prepared_fingerprint, unpadded_targets
from ...evaluation_sampling import evaluation_settings

VERSION = "raw_source_family_v4"
TASK_VERSION = VERSION
SUPPORTED_VERSIONS = (VERSION, "raw_source_family_v3")


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
    from ...services.taxonomy import canonical_task_family
    task_family = canonical_task_family(config.get("task_type"), config.get("hf_task"))
    if task_family != "unknown":
        # Raw HF metadata may use the generic classification type while its
        # declared HF task resolves to generation/token labels/etc. Family
        # lookup must use that same semantic task before and after loading.
        result["task_type"] = task_family
    result.update(initialization_seed=int(config.get("initialization_seed", config.get("seed", 42))),
                  **evaluation_settings(config))
    from ...evaluation_sampling import _uses_quarter_benchmark_cap
    if _uses_quarter_benchmark_cap(config) and not config.get("task_source_groups"):
        # Part of the family identity: old full-pool preparations must never be
        # silently reused for the new bounded evaluation pool.
        result["evaluation_preparation_policy"] = "bounded_raw_pool_v1"
    if (config.get("precision_policy") == SEGFORMER_MINI_FP32_REASON
            or config.get("mobilebert_head_normalization") == "layer_norm_v1"):
        # Preparation builds the initial evaluator before client runtime knobs
        # are applied. Freeze the reviewed precision for that stage as well.
        result.update({key: config[key] for key in ("precision_type", "mixed_precision", "precision_policy") if key in config})
    from ...tasks.contracts import uses_task_protocol, VERSION as TASK_CONTRACT_VERSION
    from ...hf_tasks import normalize_hf_task
    if uses_task_protocol(config) or normalize_hf_task(config.get("hf_task")) == "fill_mask":
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


def bind_manifest(config, manifest_path=None, *, cache=None):
    """Accept a manifest row as disposable execution input.

    Family documents are durable artifacts and may be supplied explicitly by a
    retry/export row.  The manifest file itself is never hashed, mutated, or
    used as persistent state.
    """
    if not applies(config):
        return
    if config.get("family_document_ref") and config.get("family_document_hash"):
        config["_family_reference"] = {"document_ref": config["family_document_ref"], "document_hash": config["family_document_hash"]}
    if config.get("_family_reference"):
        saved = config["_family_reference"]
        document = read_preparation(saved["document_ref"], saved["document_hash"], cache=cache)
        from ...segmentation_policy import validate_ade20k_mini_preparation
        validate_ade20k_mini_preparation(document)
        for key in ("hf_model_revision", "hf_dataset_revision"):
            if document["config"].get(key):
                config[key] = document["config"][key]


def freeze_manifest_revisions(config):
    """Mark in-memory source pins as resolved for this execution.

    The resolved configuration is persisted with the completed service.  It is
    deliberately not written beside, or linked back to, the input manifest.
    """
    config["_frozen_revisions"] = True


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
    cfg.update(seed=evaluation_settings(cfg)["evaluation_seed"],
               training_seed=int(cfg.get("initialization_seed", cfg.get("seed", 42))),
               sample_size=None, weight_decay=0.0)
    cfg.pop("max_samples", None)
    cfg["dataset_args"] = _build_dataset_args(cfg)
    cfg["dataset_args"].update(seed=cfg["seed"], _defer_evaluation_sampling=True)
    if cfg.get("evaluation_preparation_policy") == "bounded_raw_pool_v1":
        cfg["dataset_args"].update(
            _evaluation_preparation_policy="bounded_raw_pool_v1",
            **evaluation_settings(cfg))
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
                _shared_fitted_state=cfg.get("_shared_fitted_state"), _expected_raw_source=cfg.get("_expected_raw_source"),
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
            or any(doc[key] != contract[key] for key in ("config", "initial_state_hash", "compatibility", "raw_source", "shared_fitted_state", "evaluation_data_hash"))
            or doc["reservation"]["reservation_hash"] != contract["reservation_hash"]):
        raise ValueError("family document disagrees with its identity contract")
    if sha256_file(doc["reservation_ref"]) != doc["reservation_file_hash"]:
        raise ValueError("family evaluation reservation changed")
    if json.loads(Path(doc["reservation_ref"]).read_text(encoding="utf-8")) != doc["reservation"]:
        raise ValueError("family evaluation reservation disagrees with preparation")
    for artifact in doc.get("reconstruction_support", ()):
        if path_integrity(artifact["artifact_uri"])["sha256"] != artifact["metadata"]["sha256"]:
            raise ValueError("family reconstruction support changed")
    if _support_hashes(doc["reconstruction_support"]) != contract["reconstruction_support_hashes"]:
        raise ValueError("family reconstruction assets disagree with its identity")
    state = load_lineage_base(doc["store_root"], doc["lineage_id"], expected_parameter_schema_hash=doc["compatibility"]["parameter_schema_hash"])
    if state.sha256 != doc["initial_state_hash"]:
        raise ValueError("family initial checkpoint changed")
    if cache is not None:
        cache[key] = doc
    return doc


def verify_loaded(doc, prepared, cfg):
    train, test, meta, task, model, train_ids, test_ids = prepared
    if set(train_ids) & set(test_ids):
        raise ValueError("training and evaluation source IDs overlap")
    groups = cfg.get("task_source_groups")
    if groups and set(groups.get("train", ())) & set(groups.get("test", ())):
        raise ValueError("training and evaluation source groups overlap")
    contracts = shared._contracts(model, cfg, meta, task)
    compatibility = shared._compatibility(contracts)
    if (test_ids != doc["reservation"]["metric_source_ids"]
            or prepared_fingerprint(test, meta) != doc["evaluation_data_hash"]
            or _raw_contract(cfg, meta) != doc["raw_source"]
            or _fitted_state(meta) != doc["shared_fitted_state"]
            or (compatibility != doc["compatibility"]
                and not _legacy_sampling_metadata_matches(doc, contracts, cfg))):
        raise ValueError("reloaded model, data, or task interface differs from frozen family")


def _legacy_sampling_metadata_matches(doc, contracts, cfg):
    """Verify v3 identity across the addition of operational sampling metadata.

    v3 loaded the full benchmark with a null cap. v4 defers sampling explicitly
    and records its seed. Reconstruct only that older metadata representation,
    then recompute its logical hash from the actual loaded dataset interface.
    Source bytes, selected IDs, fitted transforms and all other compatibility
    dimensions must still match in ``verify_loaded``.
    """
    if doc.get("version") != "raw_source_family_v3":
        return False
    expected_args = doc.get("contracts", {}).get("dataset", {}).get("loader_args")
    actual_dataset = contracts.get("dataset", {})
    actual_args = actual_dataset.get("loader_args")
    if not isinstance(expected_args, dict) or not isinstance(actual_args, dict):
        return False
    if "benchmark_sample_size" not in expected_args or expected_args["benchmark_sample_size"] is not None:
        return False
    if actual_args.get("benchmark_sample_size") is not None:
        return False
    if "_defer_evaluation_sampling" in expected_args or actual_args.get("_defer_evaluation_sampling") is not True:
        return False
    recorded_seed = doc.get("config", {}).get("evaluation_seed", doc.get("config", {}).get("seed", 42))
    for seed in (recorded_seed, cfg.get("evaluation_seed", recorded_seed), actual_args.get("evaluation_seed", recorded_seed)):
        if isinstance(seed, bool) or seed != recorded_seed:
            return False
    equivalent_args = dict(actual_args)
    equivalent_args["benchmark_sample_size"] = None
    equivalent_args.pop("_defer_evaluation_sampling")
    if "evaluation_seed" not in expected_args:
        equivalent_args.pop("evaluation_seed", None)
    if equivalent_args != expected_args:
        return False
    equivalent_dataset = {key: value for key, value in actual_dataset.items()
                          if key not in {"logical_dataset_contract_hash", "dataset_contract_hash", "sample_seed"}}
    equivalent_dataset["loader_args"] = equivalent_args
    equivalent_compatibility = shared._compatibility(contracts)
    equivalent_compatibility["logical_dataset_contract_hash"] = sha256_json(equivalent_dataset)
    return equivalent_compatibility == doc["compatibility"]


def _raw_contract(cfg, meta):
    if meta.get("raw_source"):
        return meta["raw_source"]
    # Injected loaders must at least declare an immutable source revision.
    if cfg.get("hf_dataset_revision"):
        return {"version": "pinned_source_v1", "dataset": cfg.get("dataset"),
                "dataset_name": cfg.get("dataset_name"), "dataset_config": cfg.get("dataset_config"),
                "revision": cfg["hf_dataset_revision"], "train_split": meta.get("train_split", "train"),
                "test_split": meta.get("test_split", "test")}
    raise ValueError("family loader must report raw_source fingerprints before preparation")


def _fitted_state(meta):
    return shared._clean({"transforms": meta.get("shared_fitted_state", {}),
        "labels": {k: meta[k] for k in ("label_mapping", "id2label", "label2id", "num_classes", "num_labels",
            "target_scaler", "label_normalization", "label_min", "label_max", "vqa_answer_vocab") if k in meta}})


def _support_hashes(artifacts):
    return sorted(({"type": a["artifact_type"], "sha256": a["metadata"]["sha256"]} for a in artifacts),
                  key=lambda a: (a["type"], a["sha256"]))


def prepare(config, *, load_dataset=None, build_model=None, selection_configs=None,
            allow_legacy_evaluation=False):
    error = manifest_family_error(config)
    if error:
        raise ValueError(error)
    reference = config.get("_family_reference")
    root = Path(runner._provenance_artifact_root(config)).resolve()
    selection_keys = ("service_id", "client_id", "client_partition_index", "sample_size", "sample_frac",
        "sample_seed", "seed", "split_strategy", "distribution_type", "distribution_param", "skew_axis",
        "skew_axis_config", "heterogeneity_intensity", "shard_support_quantile", "hf_task", "task_type",
        "partition_generator_version", "training_regime", "regression_support_policy", "min_target_bin_fraction",
        "training_policy", "recipe_sampler_version", "calibration_band", "task_contract", "_recorded_source_ids")
    selection_configs = [{k: c[k] for k in selection_keys if c.get(k) is not None}
                         for c in (selection_configs or [config])]
    index_key = sha256_json({"version": VERSION, "contract": group_key(config)})
    index = root / "manifest_preparations" / (index_key + ".json")
    index.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(index) + ".lock"):
        if reference is None and index.exists():
            reference = json.loads(index.read_text(encoding="utf-8"))
        doc = read_preparation(reference["document_ref"], reference["document_hash"]) if reference else None
        if doc:
            from ...segmentation_policy import validate_ade20k_mini_preparation
            validate_ade20k_mini_preparation(doc)
            if doc["version"] != VERSION and not allow_legacy_evaluation:
                raise ValueError("frozen family predates the evaluation cap; use a new service ID and manifest")
        if doc and group_key(config) != group_key(doc["config"]):
            raise ValueError("manifest contract differs from its frozen preparation")
        cfg = deepcopy(doc["config"] if doc else contract_config(config))
        cfg["client_selection_configs"] = selection_configs
        if doc:
            cfg.update(_shared_fitted_state=doc["shared_fitted_state"]["transforms"],
                       _expected_raw_source=doc["raw_source"])
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
                if set(train_ids) & set(test_ids):
                    raise ValueError("training and evaluation source IDs overlap")
                groups = cfg.get("task_source_groups")
                if groups:
                    if set(groups.get("train", ())) & set(groups.get("test", ())):
                        raise ValueError("training and evaluation source groups overlap")
                    groups = {**groups, "train": []}
                reservation = shared.reserve_evaluation([], test_ids, unpadded_targets(test[1], meta), task_family=task,
                    seed=cfg["evaluation_seed"], contract=compatibility, source_groups=groups,
                    benchmark_sample_size=cfg["benchmark_sample_size"])
                initial = capture_model_tensor_state(model)
                data_hash = prepared_fingerprint(test, meta)
                version = VERSION
                raw_source, fitted_state = _raw_contract(cfg, meta), _fitted_state(meta)
                semantic_config = {k: v for k, v in cfg.items() if k not in {
                    "client_selection_configs", "device", "db_path", "provenance_artifact_dir"} and not k.startswith("_")}
                if groups:
                    semantic_config["task_source_groups"] = groups
                support = save_reconstruction_support_artifacts(root, index_key, model)
                contract = {"version": version, "initial_state_hash": initial.sha256,
                    "compatibility": compatibility, "reservation_hash": reservation["reservation_hash"],
                    "config": shared._clean(semantic_config), "raw_source": raw_source,
                    "shared_fitted_state": fitted_state, "evaluation_data_hash": data_hash,
                    "reconstruction_support_hashes": _support_hashes(support)}
                family = "family_" + sha256_json(contract)
                lineage = "lineage_" + family
                ensure_lineage_base(root, lineage, model, parameter_schema_hash=compatibility["parameter_schema_hash"])
                reservation_artifact = write_json_artifact(root, family, "family_reservation", reservation)
                doc = {"version": version, "family_id": family, "family_contract": contract,
                    "initial_state_hash": initial.sha256, "lineage_id": lineage, "store_root": str(root),
                    "config": shared._clean(semantic_config), "contracts": contracts, "compatibility": compatibility,
                    "compatibility_contract_hash": sha256_json(contract), "reservation": reservation,
                    "reservation_ref": reservation_artifact["artifact_uri"],
                    "reservation_file_hash": reservation_artifact["metadata"]["sha256"],
                    "raw_source": raw_source, "shared_fitted_state": fitted_state, "evaluation_data_hash": data_hash,
                    "reconstruction_support": support}
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
    from ...client_reconstruction import apply_retention_defaults, retention_error, retention_settings, read_recipe
    for entry in entries:
        apply_retention_defaults(entry.resolved)
        if error := retention_error(entry.resolved):
            raise ValueError(error)
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
        # Validate explicit durable family references before the first client
        # can train or interrupt.  No state is written beside the manifest.
        bound_cache = {}
        completed_cache = {}
        for entry in entries:
            row = entry.resolved
            existing = row.get("_family_reference")
            if existing and read_preparation(existing["document_ref"], existing["document_hash"], cache=bound_cache)["family_id"] != doc["family_id"]:
                raise ValueError("group contains conflicting frozen family preparations")
        for entry in entries:
            if on_start is not None:
                on_start(entry)
            row = dict(entry.resolved)
            sid = row["service_id"]
            concise = not debug_enabled(row)
            row.update(_family_ordinal=len(results) + 1, _family_total=len(entries),
                       _defer_terminal_results=concise, family_id=doc["family_id"])
            entry.resolved.update(family_id=doc["family_id"], family_contract_version=doc["version"],
                initial_state_hash=state.sha256, family_document_ref=reference["document_ref"],
                family_document_hash=reference["document_hash"])
            if not row.get("_rerun_existing"):
                from ..run_manifest import _find_existing_service_database
                existing_db = _find_existing_service_database(
                    row.get("db_path"), sid, completed_cache, doc["family_id"]
                )
                if existing_db is not None:
                    outcome = {"service_id": sid, "status": "skipped",
                               "error_message": f"Service family already completed in SQLite database: {existing_db}"}
                    results.append(outcome)
                    if on_result is not None:
                        on_result(entry, outcome)
                    continue
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
                    initial_state_hash=state.sha256,
                    lineage_id=doc["lineage_id"], parent_state_hash=state.sha256,
                    compatibility_contract_hash=doc["compatibility_contract_hash"],
                    training_seed=int(next((row[k] for k in ("training_seed", "sample_seed", "seed") if row.get(k) is not None), 42)),
                    _family_document=doc, _prepared_model=model, _prepared_dataset=(train, benchmark, meta),
                    _prepared_benchmark_sample_info={**reservation.get("evaluation_sampling", {}),
                        "source_ids": reservation["catalogue_metric_ids"], "frozen": True},
                    _parent_state=state, _parent_ref=base.base, _parent_lineage_base=base,
                    _restored_parent=state, _model_reset_s=reset_s,
                    _expected_compatibility=doc["compatibility"], _retain_client_checkpoint=True,
                    provenance_artifact_dir=doc["store_root"], dataset_args=cfg["dataset_args"],
                    max_eval_time_s=None, **evaluation_settings(cfg),
                    training_universe_hash=sha256_json(doc["raw_source"]),
                    benchmark_cohort_hash=sha256_json(reservation["catalogue_metric_ids"]))
                entry.resolved.update({k: row[k] for k in ("hf_model_revision", "hf_dataset_revision") if row.get(k)})
                selected_split = selection.materialize(sid, train)
                if selected_split is not None:
                    row["_pretokenization_split_info"] = selected_split
                summary = (execute_service or runner.execute_service)(row)
                if summary.status != "success":
                    raise ValueError(summary.error or "service training failed")
                verify_timer = runner._StageTimer()
                if retention_settings(row)["client_checkpoint_retained"]:
                    final = reconstruct_service_state(doc["store_root"], sid)
                    if source_state_hash(doc["store_root"], sid, final) != capture_model_tensor_state(model).sha256:
                        raise ValueError("saved checkpoint differs from trained service")
                    del final
                else:
                    recipe = read_recipe(row["db_path"], sid)
                    if recipe["fingerprints"]["original_model"] != capture_model_tensor_state(model).sha256:
                        raise ValueError("recipe differs from trained service")
                verify_loaded_contract = shared._compatibility(shared._contracts(model, cfg, meta, task))
                if verify_loaded_contract != doc["compatibility"]:
                    raise ValueError("training changed the frozen task interface")
                restore_verify_s = verify_timer.elapsed_s()
                if row.get("weight_retention") == "recipe_only":
                    # Release trained client parameters and optimizer buffers;
                    # the next grouped client starts from the common state.
                    restore_model_tensor_state(model, state, verify=True)
                    runner.prepare_model_for_local_update(model, row)
                # Composition consumers verify raw-source family and recipe
                # evidence independently of the publication eligibility marker.
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
            from ...client_reconstruction import read_recipe, replay_supported
            recipe = read_recipe(db, service_id)
            if not recipe["retention"].get("client_checkpoint_retained", True):
                if recipe["retention"]["policy"] != "recipe_only" or not replay_supported(
                        recipe["training"]["observations"], recipe["training"].get("memory_metadata")):
                    raise ValueError("completed recipe-only service has no supported reconstruction")
                return
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
