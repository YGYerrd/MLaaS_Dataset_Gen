"""Client recipes, weight retention, materialisation and fidelity validation."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .provenance import sha256_json, sha256_file, write_json_artifact, runtime_provenance
from .shared_parent import _clean
from .artifact_paths import resolve_artifact_path

VERSION = "client_reconstruction_recipe_v1"
MAX_SKIPPED_UPDATE_FRACTION = 0.20
REPLAY_REQUIREMENTS = (
    f"completed epochs, at most {MAX_SKIPPED_UPDATE_FRACTION:.0%} skipped optimizer updates, "
    "and no memory recovery or loss-precision retries"
)
TRAINING_KEYS = (
    "training_regime", "training_epochs", "batch_size", "learning_rate", "optimizer",
    "weight_decay", "momentum", "warmup_ratio", "gradient_accumulation_steps",
    "mixed_precision", "precision_type", "precision_policy", "max_grad_norm",
    "max_train_time_s", "training_seed", "seed", "max_length",
    "mobilebert_head_normalization", "generation_config", "max_new_tokens", "num_beams", "do_sample",
)


def retention_policy(config):
    """Resolve an omitted policy using the existing durable-family support contract."""
    if config.get("weight_retention") is not None:
        return config["weight_retention"]
    from .debug_mode import weights_disabled
    from .family_preflight import independent_family_applies
    if independent_family_applies(config) and not config.get("fl_run_id") and not weights_disabled(config):
        return "recipe_only"
    return "checkpoint"


def apply_retention_defaults(config):
    config["weight_retention"] = retention_policy(config)
    config.setdefault("retain_validation_checkpoint", False)
    return config


def retention_error(config):
    policy = retention_policy(config)
    if policy not in {"checkpoint", "recipe_only"}:
        return "weight_retention must be checkpoint or recipe_only"
    if str(config.get("retain_validation_checkpoint", False)).lower() not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
        return "retain_validation_checkpoint must be a boolean"
    if policy == "recipe_only":
        from .debug_mode import weights_disabled
        from .family_preflight import independent_family_applies
        if weights_disabled(config):
            return "recipe_only cannot be combined with diagnostic no_save_weights"
        if config.get("fl_run_id") or not independent_family_applies(config):
            return "recipe_only requires a trained independent durable family; this model or generation mode is unsupported"
    return None


def retention_settings(config):
    policy = retention_policy(config)
    reference = str(config.get("retain_validation_checkpoint", False)).lower() in {"true", "1", "yes", "on"}
    retained = policy == "checkpoint" or reference
    return {"policy": policy, "client_checkpoint_retained": retained,
            "role": "reconstruction_validation_reference" if retained else "recipe_and_observations"}


def replay_supported(observations, memory=None):
    """Accept representative retraining, including a small fraction of skipped updates."""
    memory = memory or {}
    attempted = observations.get("optimizer_step_count", memory.get("optimizer_step_count"))
    skipped = observations.get("skipped_optimizer_step_count", memory.get("skipped_optimizer_step_count", 0))
    successful = observations.get("successful_optimizer_step_count", memory.get("successful_optimizer_step_count"))
    if attempted is not None:
        if not (attempted > 0 and 0 <= skipped <= MAX_SKIPPED_UPDATE_FRACTION * attempted):
            return False
        if successful is not None and successful != attempted - skipped:
            return False
    elif skipped:
        return False  # A nonzero skip count needs an observed denominator.
    if successful is not None and not successful > 0:
        return False
    return (observations.get("train_stopping_reason") == "epochs_completed"
            and not observations.get("train_stopped_early")
            and not observations.get("oom_retry_count", memory.get("oom_retry_count", 0))
            and not observations.get("train_fp32_loss_retry_count", memory.get("train_fp32_loss_retry_count", 0)))


def effective_training_settings(service):
    """Use the same observed-setting resolution for persistence and replay."""
    cfg = service.config
    effective = getattr(service, "_effective_training_configuration", cfg)
    observations = getattr(service, "_reconstruction_training_observations", {})
    settings = {k: effective[k] for k in TRAINING_KEYS if k in effective}
    evaluation_settings = getattr(service, "_reconstruction_evaluation_settings", {})
    if evaluation_settings.get("generation_config"):
        settings["generation_config"] = evaluation_settings["generation_config"]
        for key in ("max_new_tokens", "num_beams", "do_sample"):
            if key in settings["generation_config"]:
                settings[key] = settings["generation_config"][key]
    fit_arguments = getattr(service, "_adapter_fit_arguments", None) or {}
    for argument, key in (("epochs", "training_epochs"), ("lr", "learning_rate"), ("batch_size", "batch_size")):
        if argument in fit_arguments:
            settings[key] = fit_arguments[argument]
    for key in ("training_epochs", "batch_size", "optimizer", "weight_decay", "warmup_ratio", "max_length"):
        if key in observations:
            settings[key] = observations[key]
    for observed, key in (("logical_batch_size", "batch_size"),
            ("gradient_accumulation_steps_requested", "gradient_accumulation_steps"),
            ("precision_type_effective", "precision_type"), ("mixed_precision_effective", "mixed_precision")):
        if observed in observations:
            settings[key] = observations[observed]
    optimizer_configuration = getattr(service, "_observed_optimizer", None)
    if optimizer_configuration:
        settings["optimizer"] = optimizer_configuration["class_name"].lower()
        for key in ("learning_rate", "weight_decay", "momentum"):
            value = optimizer_configuration["config"].get(key)
            if isinstance(value, (int, float)):
                settings[key] = value
    return _clean(settings)


def persist_recipe(service, metrics, generation):
    cfg = service.config
    family = cfg["_family_document"]
    inputs = service._reconstruction_inputs
    selection = dict(inputs["selection"])
    source_ids = selection.get("partition_source_ids")
    count = int(selection["effective_train_samples"])
    if source_ids is None or len(source_ids) != count or len(set(source_ids)) != count:
        raise ValueError("client recipe requires ordered, unique accepted source-row IDs")
    effective = getattr(service, "_effective_training_configuration", cfg)
    observations = getattr(service, "_reconstruction_training_observations", {})
    settings = effective_training_settings(service)
    evaluation_settings = getattr(service, "_reconstruction_evaluation_settings", {})
    optimizer_configuration = getattr(service, "_observed_optimizer", None)
    # Runtime memory recovery keeps logical batches intact. Record both the
    # optimizer recipe and actual physical microbatch/accumulation separately.
    memory = effective.get("training_memory_metadata") or {}
    execution = {k: observations.get(k, memory.get(k)) for k in (
        "batch_size_requested", "micro_batch_size_effective", "logical_batch_size",
        "gradient_accumulation_steps_requested", "gradient_accumulation_steps_effective",
        "effective_optimizer_batch_size", "oom_retry_count", "memory_fallback_applied", "train_fp32_loss_retry_count")
        if k in observations or k in memory}
    stopping = {k: observations[k] for k in (
        "training_epochs", "completed_training_epochs", "optimizer_step_count", "planned_optimizer_step_count",
        "successful_optimizer_step_count", "skipped_optimizer_step_count", "train_stopped_early", "train_stopping_reason")
        if k in observations}
    stopping["status"] = "completed"
    replayable = replay_supported(observations, memory)
    document = _clean({
        "version": VERSION, "service_id": service.service_id,
        "family": {"family_id": family["family_id"], "document_ref": cfg["family_document_ref"],
                   "document_hash": cfg["family_document_hash"], "initial_state_hash": family["initial_state_hash"]},
        "selection": {"ordered_source_row_ids": source_ids,
            "requested_count": selection.get("requested_sample_size_total"), "realised_count": count,
            "resolved": {k: v for k, v in selection.items() if k not in {
                "partition_source_ids", "accepted_source_row_ids", "partition_indices"}}},
        "training": {"effective_settings": settings,
            "requested_settings": {k: service._requested_configuration[k] for k in TRAINING_KEYS if k in service._requested_configuration},
            "adapter_fit_arguments": getattr(service, "_adapter_fit_arguments", None),
            "optimizer_configuration": optimizer_configuration,
            "budget_adjustment": getattr(service, "_training_budget_adjustment", None),
            "runtime_execution": execution, "memory_metadata": memory, "stopping": stopping,
            "observations": observations,
            "validation_replay_supported": replayable},
        "seeds": {"sampling": selection.get("sample_seed"), "training": settings.get("training_seed", settings.get("seed")),
                  "initialization": family["config"]["initialization_seed"], "evaluation": family["config"]["evaluation_seed"]},
        "fingerprints": {"source": sha256_json(family["raw_source"]),
            "ordered_source_rows": sha256_json(source_ids), "prepared_train": inputs["prepared_train_hash"],
            "prepared_evaluation": inputs["prepared_evaluation_hash"], "original_model": generation["final_model_hash"]},
        "retention": retention_settings(cfg),
        "evaluation": {"effective_settings": evaluation_settings},
        "observed_metrics": metrics, "runtime": runtime_provenance(),
    })
    document["recipe_hash"] = sha256_json(document)
    return write_json_artifact(family["store_root"], service.service_id, "client_reconstruction_recipe", document)


def read_recipe(db, service_id):
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM service_generation_provenance WHERE service_id=?", (service_id,)).fetchone()
    if row is None or "client_recipe_ref" not in row.keys() or not row["client_recipe_ref"]:
        raise ValueError("service has no authoritative client reconstruction recipe")
    path = resolve_artifact_path(row["client_recipe_ref"])
    if sha256_file(path) != row["client_recipe_hash"]:
        raise ValueError("client recipe failed integrity verification")
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if recipe.get("version") != VERSION or recipe.get("recipe_hash") != sha256_json({k: v for k, v in recipe.items() if k != "recipe_hash"}):
        raise ValueError("invalid client reconstruction recipe")
    family = recipe["family"]
    if (recipe["service_id"] != service_id or family["family_id"] != row["family_id"]
            or family["document_hash"] != row["family_document_hash"]
            or family["initial_state_hash"] != row["initial_state_hash"]
            or recipe["fingerprints"]["original_model"] != row["final_model_hash"]):
        raise ValueError("recipe disagrees with service provenance")
    ids = recipe["selection"]["ordered_source_row_ids"]
    if len(ids) != recipe["selection"]["realised_count"] or len(set(ids)) != len(ids) or sha256_json(ids) != recipe["fingerprints"]["ordered_source_rows"]:
        raise ValueError("invalid recorded client rows")
    return recipe


@contextmanager
def materialize_service(db, service_id, *, device="cpu"):
    """Yield a representative trained model using only family + recipe.

    No client checkpoint is read or written. The caller owns the yielded model;
    prepared dataset resources are closed on leaving the context. Interrupted
    or recovery trajectories fail explicitly until their replay is supported.
    Original and reconstructed runs may skip different updates within the 20% limit.
    """
    from .cli.manifest import preparation
    from .data.tokenization_selection import training_row_selection
    from .data.source_contract import prepared_fingerprint
    from .data.fedavg_partition import _take_rows
    from .data.image_store import close_image_stores
    from .checkpoint_store import restore_model_tensor_state
    from .services.runner import ServiceRunner, prepare_model_for_local_update
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        status = conn.execute("SELECT status FROM services WHERE service_id=?", (service_id,)).fetchone()
    if status is None or status[0] != "completed":
        raise ValueError("materialisation requires a completed service")
    recipe = read_recipe(db, service_id)
    # Re-evaluate older recipes under the current policy, rather than trusting
    # the eligibility flag recorded when zero skips were required.
    if not replay_supported(recipe["training"]["observations"], recipe["training"].get("memory_metadata")):
        raise ValueError(f"recipe reconstruction requires {REPLAY_REQUIREMENTS}")
    family = recipe["family"]
    doc = preparation.read_preparation(family["document_ref"], family["document_hash"])
    if doc["family_id"] != family["family_id"] or doc["initial_state_hash"] != family["initial_state_hash"]:
        raise ValueError("recipe initial state disagrees with family")
    if sha256_json(doc["raw_source"]) != recipe["fingerprints"]["source"]:
        raise ValueError("recipe source disagrees with family")
    config = {**doc["config"], "service_id": service_id, "db_path": str(db), "device": device,
        "provenance_artifact_dir": doc["store_root"], "_family_reference": family,
        "_recorded_source_ids": recipe["selection"]["ordered_source_row_ids"]}
    with training_row_selection([config]) as selection:
        _, _, cfg, prepared, initial, _ = preparation.prepare(config, allow_legacy_evaluation=True)
    train, test, meta, task, model, _, _ = prepared
    try:
        split = selection.materialize(service_id, train)
        source_ids = split["resolved"]["partition_source_ids"]
        if source_ids != recipe["selection"]["ordered_source_row_ids"]:
            raise ValueError("reconstructed selected rows differ from recipe")
        selected = (split["x_train"], split["y_train"])
        positions = np.asarray(doc["reservation"]["catalogue_positions"], dtype=int)
        benchmark = (_take_rows(test[0], positions), _take_rows(test[1], positions))
        for name, pair in (("prepared_train", selected), ("prepared_evaluation", benchmark)):
            if prepared_fingerprint(pair, meta) != recipe["fingerprints"][name]:
                raise ValueError(f"reconstructed {name} differs from recorded content")
        effective = {**cfg, **recipe["training"]["effective_settings"], "device": device,
                     "_family_document": doc, "max_train_time_s": None}
        service = ServiceRunner(effective)
        # Optimizer, scheduler and RNG are reset exactly as in the generator.
        restore_model_tensor_state(model, initial, verify=True)
        prepare_model_for_local_update(model, effective)
        if recipe["training"].get("optimizer_configuration"):
            import keras
            model.optimizer = keras.optimizers.deserialize(recipe["training"]["optimizer_configuration"])
        training = service._train_model_effective(model, *selected, task_family=task)
        if not replay_supported(training, service.config.get("training_memory_metadata")):
            raise ValueError(f"reconstructed training requires {REPLAY_REQUIREMENTS}")
        service._reconstruction_training_observations = training
        service._reconstruction_evaluation_settings = recipe.get("evaluation", {}).get("effective_settings", {})
        optimizer = getattr(model, "optimizer", None)
        if callable(getattr(optimizer, "get_config", None)):
            service._observed_optimizer = {"class_name": type(optimizer).__name__, "config": optimizer.get_config()}
        actual_settings = effective_training_settings(service)
        # Completed epochs are replayed without the original wall-clock cutoff.
        expected_settings = {k: v for k, v in recipe["training"]["effective_settings"].items() if k != "max_train_time_s"}
        actual_settings = {k: actual_settings.get(k) for k in expected_settings}
        if actual_settings != expected_settings:
            changed = [k for k in expected_settings if actual_settings[k] != expected_settings[k]]
            raise ValueError(f"replay effective training settings differ: {changed}")
        stopping = recipe["training"]["stopping"]
        # Preserve the training budget; numerical differences may change which
        # updates AMP skips without changing the family, data or recipe.
        for key in ("optimizer_step_count", "completed_training_epochs"):
            if key in stopping and training.get(key) != stopping[key]:
                raise ValueError(f"replay completion differs from recorded {key}")
        core = getattr(model, "core", None)
        for key, value in recipe.get("evaluation", {}).get("effective_settings", {}).items():
            if core is not None:
                setattr(core, key, value)
        yield {"model": model, "service": service, "recipe": recipe, "family": doc,
               "initial": initial, "benchmark": benchmark, "meta": meta, "task": task, "training": training,
               "replay_evidence": {"ordered_source_rows_hash": sha256_json(source_ids),
                   "effective_settings_hash": sha256_json(actual_settings),
                   "selected_rows_match": True, "effective_settings_match": True}}
    finally:
        close_image_stores((train, test))


def _evaluate_materialized(run, *, collect_predictions=True):
    from .services.taxonomy import canonical_metric_names
    from .services.runner import prepare_model_for_local_update_seed
    from .behavioural_diagnostics import predict_decisions
    service, task = run["service"], run["task"]
    cfg = service.config
    prepare_model_for_local_update_seed({**cfg, "training_seed": run["recipe"]["seeds"]["evaluation"]})
    if cfg.get("task_contract") and task in {"token_classification", "segmentation", "generation", "fill_mask"}:
        from .tasks.contracts import TaskContract
        from .tasks.runtime import evaluate_task_model
        contract = TaskContract.from_dict(cfg["task_contract"])
        result = evaluate_task_model(run["model"], *run["benchmark"], contract,
            run["family"]["reservation"]["catalogue_metric_ids"], collect=collect_predictions)
        primary, secondary = contract.evaluation["primary"], contract.evaluation["secondary"]
        predictions = result.predictions
        if task != "generation":
            predictions = tuple(np.asarray(p)[np.asarray(t) != contract.semantics["ignore_index"]]
                                for p, t in zip(predictions, result.targets))
        return {k: result.metrics[k] for k in ("loss", primary, secondary)}, predictions
    if collect_predictions and task not in {"classification", "regression", "generation"}:
        raise ValueError(f"reference prediction comparison is not implemented for {task}")
    primary, secondary = canonical_metric_names(task, hf_task=cfg.get("hf_task"), task_tag=cfg.get("task_tag"))
    loss, first, second, _ = service._evaluate_model(run["model"], *run["benchmark"], inference_only=False,
                                                   task_family=task, enforce_time_limit=False)
    metrics = {"loss": loss, primary: first}
    if secondary:
        metrics[secondary] = second
    if not collect_predictions:
        return metrics, ()
    if task == "generation":
        core = getattr(run["model"], "core", None)
        if core is None or not core.task_spec.supports_generation:
            raise ValueError("generation comparison requires a generation-capable adapter")
        predictions = []
        core.model.eval()
        with core.torch.no_grad():
            for xb, yb in core._batch_iter(*run["benchmark"]):
                enc, _, _ = core.task_spec.encode_batch(core.tokenizer, xb, yb, core.max_length,
                    core.torch, core.device, ignore_index=core.label_pad_value, inference_only=True)
                inputs = core.task_spec.build_forward_inputs(enc, labels_t=None, inference_only=True)
                generated = core._run_with_precision_context(lambda: core.task_spec.generate_predictions(
                    core.model, inputs, core.tokenizer, core.torch, core.generation_config))
                predictions.extend(core.tokenizer.batch_decode(generated, skip_special_tokens=True))
        if len(predictions) != len(run["family"]["reservation"]["catalogue_metric_ids"]):
            raise ValueError("generated outputs do not align to reference rows")
    else:
        predictions = predict_decisions(run["model"], *run["benchmark"], task_family=task,
            batch_size=int(cfg.get("batch_size", 32)), ignore_index=int(run["meta"].get("ignore_index", -100)))
    return metrics, tuple(predictions)


def _prediction_comparison(reference, replay, task):
    if not reference or len(reference) != len(replay):
        raise ValueError("reference predictions are empty or misaligned")
    equal_examples = sum(np.array_equal(a, b) for a, b in zip(reference, replay))
    result = {"reference_hash": sha256_json(_clean(reference)), "reconstructed_hash": sha256_json(_clean(replay)),
              "example_count": len(reference), "exact_example_agreement": equal_examples / len(reference)}
    if task == "generation":
        result["agreement"] = result["exact_example_agreement"]
    else:
        if any(np.shape(a) != np.shape(b) for a, b in zip(reference, replay)):
            raise ValueError("reference prediction shapes differ")
        a = np.concatenate([np.asarray(v).reshape(-1) for v in reference]).astype(np.float64)
        b = np.concatenate([np.asarray(v).reshape(-1) for v in replay]).astype(np.float64)
        if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("empty or nonfinite predictions")
        result.update(agreement=float(np.mean(a == b)), count=int(a.size),
                      rmse=float(np.sqrt(np.mean((a - b) ** 2))), max_abs_difference=float(np.max(np.abs(a - b))),
                      reference_rms=float(np.sqrt(np.mean(a ** 2))))
    return result


def _parameter_comparison(initial, reference, replay):
    if [e.descriptor() for e in initial.entries] != [e.descriptor() for e in reference.entries] or [e.descriptor() for e in reference.entries] != [e.descriptor() for e in replay.entries]:
        raise ValueError("parameter schemas differ")
    squared_error = reference_norm = update_dot = original_update_norm = replay_update_norm = 0.0
    for start, original, reconstructed in zip(initial.entries, reference.entries, replay.entries):
        if original.dtype == "bfloat16":
            return {"available": False, "reason": "bfloat16 numeric comparison is unsupported"}
        a, b, c = (entry.to_numpy().astype(np.float64) for entry in (start, original, reconstructed))
        if not all(np.isfinite(v).all() for v in (a, b, c)):
            raise ValueError("nonfinite parameters")
        squared_error += float(np.sum((b - c) ** 2))
        reference_norm += float(np.sum(b ** 2))
        original_update_norm += float(np.sum((b - a) ** 2))
        replay_update_norm += float(np.sum((c - a) ** 2))
        update_dot += float(np.sum((b - a) * (c - a)))
    denominator = np.sqrt(original_update_norm * replay_update_norm)
    return {"available": True, "relative_l2_error": float(np.sqrt(squared_error / max(reference_norm, 1e-30))),
            "update_cosine_similarity": float(update_dot / denominator) if denominator else None,
            "original_update_l2": float(np.sqrt(original_update_norm)),
            "reconstructed_update_l2": float(np.sqrt(replay_update_norm))}


DEFAULT_TOLERANCES = {"quality_absolute": 0.02, "loss_absolute": 1e-4, "loss_relative": 0.01,
                      "prediction_agreement": 0.98, "regression_prediction_relative_rmse": 0.01}


def fidelity_gate(report, tolerances=None):
    """Gate functional reconstruction; weight hashes and parameter similarity are diagnostic."""
    limits = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    if set(limits) != set(DEFAULT_TOLERANCES) or any(not np.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError("invalid fidelity tolerances")
    if limits["prediction_agreement"] > 1:
        raise ValueError("prediction_agreement must be in [0,1]")
    failures = []
    if report.get("version") == "client_reconstruction_comparison_v3":
        evidence = report.get("replay_evidence", {})
        if not evidence.get("selected_rows_match") or not evidence.get("effective_settings_match"):
            failures.append("replay_recipe")
        if evidence.get("client_checkpoint_reads") != 0:
            failures.append("client_checkpoint_read")
    def close(name, value, reference, absolute, relative=0):
        if (value is None or reference is None or not np.isfinite(value) or not np.isfinite(reference)
                or abs(value - reference) > absolute + relative * abs(reference)):
            failures.append(name)
    for name, value in report["reconstructed_metrics"].items():
        reference = report["reference_checkpoint_metrics"][name]
        is_loss = name in {"loss", "perplexity", "perplexity_proxy", "rmse", "mae"}
        close(name, value, reference, limits["loss_absolute"] if is_loss else limits["quality_absolute"],
              limits["loss_relative"] if is_loss else 0)
    close("training_loss", report["training_loss"]["reconstructed"], report["training_loss"]["original"],
          limits["loss_absolute"], limits["loss_relative"])
    p = report["predictions"]
    if report["task"] == "regression":
        close("predictions", p.get("rmse"), 0, limits["loss_absolute"] +
              limits["regression_prediction_relative_rmse"] * p.get("reference_rms", 0))
    elif not np.isfinite(p["agreement"]) or p["agreement"] < limits["prediction_agreement"]:
        failures.append("predictions")
    if not report["training_changed_initial_state"]:
        failures.append("no_training_update")
    return {"passed": not failures, "failures": failures, "tolerances": limits}


def compare_reconstruction(db, service_id, *, output_dir, device="cpu", tolerances=None,
                           reference_service_id=None, reference_store=None, fresh_process=False):
    """Compare an independent replay with retained weights; never write original metrics."""
    from .checkpoint_store import capture_model_tensor_state, restore_model_tensor_state, reconstruct_service_state
    from .reduced_checkpoint import source_state_hash
    from unittest.mock import patch
    from contextlib import ExitStack
    # Keep the guard active through replay/evaluation. Only the comparison below
    # may restore client weights; loading the common family base is permitted.
    with ExitStack() as stack:
        with patch("mlaas_data_generator.checkpoint_store.reconstruct_service_state",
                   side_effect=RuntimeError("client checkpoint read during reconstruction")), \
             patch("mlaas_data_generator.cli.manifest.preparation.reconstruct_service_state",
                   side_effect=RuntimeError("client checkpoint read during reconstruction")):
            run = stack.enter_context(materialize_service(db, service_id, device=device))
            recipe, doc = run["recipe"], run["family"]
            reconstructed = capture_model_tensor_state(run["model"])
            replay_metrics, replay_predictions = _evaluate_materialized(run)
        # Only the comparison reads client weights, after materialisation finishes.
        root, reference_id = reference_store or doc["store_root"], reference_service_id or service_id
        reference = reconstruct_service_state(root, reference_id)
        if source_state_hash(root, reference_id, reference) != recipe["fingerprints"]["original_model"]:
            raise ValueError("retained reference checkpoint disagrees with recipe")
        restore_model_tensor_state(run["model"], reference, verify=True)
        reference_metrics, reference_predictions = _evaluate_materialized(run)
        original_loss = recipe["training"].get("observations", {}).get("train_loss")
        if original_loss is None:
            original_loss = recipe["observed_metrics"].get("train_loss", {}).get("value")
        replay_loss = run["training"].get("train_loss")
        report = {"version": "client_reconstruction_comparison_v3", "service_id": service_id,
            "recipe_hash": recipe["recipe_hash"], "family_id": doc["family_id"], "task": run["task"],
            "realised_samples": recipe["selection"]["realised_count"], "training": run["training"],
            "retention": recipe["retention"], "reference_service_id": reference_id,
            "reference_source_ids": doc["reservation"]["catalogue_metric_ids"],
            "reference_prepared_fingerprint": recipe["fingerprints"]["prepared_evaluation"],
            "original_observed_metrics": recipe["observed_metrics"],
            "reference_checkpoint_metrics": reference_metrics, "reconstructed_metrics": replay_metrics,
            "metric_deltas": {k: replay_metrics[k] - reference_metrics[k] for k in reference_metrics},
            "training_loss": {"original": original_loss, "reconstructed": replay_loss,
                              "delta": replay_loss - original_loss if original_loss is not None and replay_loss is not None else None},
            "predictions": _prediction_comparison(reference_predictions, replay_predictions, run["task"]),
            "parameters": _parameter_comparison(run["initial"], reference, reconstructed),
            "reconstructed_model_hash": reconstructed.sha256,
            "bitwise_equal": reconstructed.sha256 == recipe["fingerprints"]["original_model"],
            "training_changed_initial_state": recipe["fingerprints"]["original_model"] != run["initial"].sha256,
            "runtime": runtime_provenance(),
            "scope": {"config": doc["config"], "family_document_hash": recipe["family"]["document_hash"],
                      "source_hash": recipe["fingerprints"]["source"], "device": device,
                      "task_contract": run["service"].config.get("task_contract")},
            "replay_evidence": {**run["replay_evidence"], "client_checkpoint_reads": 0,
                                "fresh_process": fresh_process}}
        report["gate"] = fidelity_gate(report, tolerances)
        report = _clean(report)
        artifact = write_json_artifact(output_dir, service_id, "client_reconstruction_comparison", report)
        return report, artifact


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--materialize-only", action="store_true", help="Export a model from family + recipe without reading a client checkpoint")
    args = parser.parse_args()
    if args.materialize_only:
        from .provenance import save_model_artifact
        from .checkpoint_store import capture_model_tensor_state
        with materialize_service(args.db, args.service_id, device=args.device) as run:
            metrics, _ = _evaluate_materialized(run, collect_predictions=False)
            artifact = save_model_artifact(args.output_dir, args.service_id, "materialized", run["model"])
            if artifact is None:
                raise ValueError("model cannot be exported")
            report = {"recipe_hash": run["recipe"]["recipe_hash"], "training": run["training"],
                      "reconstructed_metrics": metrics, "model_hash": capture_model_tensor_state(run["model"]).sha256,
                      "model_artifact": artifact, "runtime": runtime_provenance()}
            observation = write_json_artifact(args.output_dir, args.service_id, "client_materialization", _clean(report))
            print(json.dumps(observation, indent=2))
        return
    report, artifact = compare_reconstruction(args.db, args.service_id, output_dir=args.output_dir, device=args.device)
    print(json.dumps({"service_id": args.service_id, "metric_deltas": report["metric_deltas"],
                      "report": artifact["artifact_uri"]}, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
