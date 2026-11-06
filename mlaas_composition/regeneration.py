"""Regenerate legacy services as executable representatives under a CohortPlan."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .validation import (
    CohortIsolationError,
    CohortPlan,
    ServicePrediction,
    assert_cohort_isolation,
    cohort_label_hash,
)


def execute_stored_service_prediction(
    *, db_path: str | Path, service_id: str, plan: CohortPlan
) -> ServicePrediction:
    """Reconstruct one recorded service checkpoint and run it on the audit cohort."""
    from mlaas_data_generator.checkpoint_store import reconstruct_service_state

    row = _service_row(db_path, service_id)
    state = reconstruct_service_state(_checkpoint_root(row, db_path), service_id)
    return execute_tensor_state_prediction(
        db_path=db_path,
        representative_service_id=service_id,
        runtime_id=service_id,
        plan=plan,
        state=state,
    )


def execute_tensor_state_prediction(
    *,
    db_path: str | Path,
    representative_service_id: str,
    runtime_id: str,
    plan: CohortPlan,
    state: Any,
) -> ServicePrediction:
    """Restore one supplied state into a compatible model and execute it once."""
    service_id = str(representative_service_id)
    row = _service_row(db_path, service_id)
    generation = _json_object(row.get("generation_provenance_json"))
    if generation.get("independent_family"):
        return _execute_independent_family_state(generation["independent_family"], runtime_id, plan, state)
    if generation.get("shared_parent"):
        return _execute_shared_parent_state(db_path, generation["shared_parent"]["parent_checkpoint_id"], runtime_id, plan, state)
    config = _rehydration_config(row)
    config["db_path"] = str(db_path)
    task_family = str(row.get("task_family") or row.get("task_type") or "").strip().lower()
    if task_family not in {"classification", "regression"}:
        raise ValueError(
            "stored runtime validation supports classification or regression services"
        )
    _seed_everything(int(config.get("seed", plan.seed)))
    from mlaas_data_generator.checkpoint_store import restore_model_tensor_state
    from mlaas_data_generator.services.runner import ServiceRunner, _load_dataset

    train, test, meta = _load_dataset(
        config.get("dataset", "hf"), **dict(config["dataset_args"])
    )
    (x_train, y_train), (x_test, y_test) = train, test
    train_ids = tuple(f"train:{index}" for index in range(_size(y_train)))
    test_ids = tuple(f"test:{index}" for index in range(_size(y_test)))
    assert_cohort_isolation(plan.training_ids, plan.catalogue_metric_ids, plan.audit_ids)
    if set(plan.training_ids) - set(train_ids):
        raise CohortIsolationError(
            "loaded training dataset no longer matches the immutable cohort plan IDs"
        )
    if set(plan.catalogue_metric_ids).union(plan.audit_ids) - set(test_ids):
        raise CohortIsolationError(
            "loaded test dataset no longer matches the immutable cohort plan IDs"
        )
    source_ids = tuple(plan.catalogue_metric_ids) + tuple(plan.audit_ids)
    source_labels = _select(y_test, test_ids, source_ids)
    if cohort_label_hash(source_ids, np.asarray(source_labels).tolist()) != plan.label_hash:
        raise CohortIsolationError(
            "loaded test labels do not match the immutable cohort label contract"
        )

    if str(config.get("model_type") or "").lower() in {"cnn", "mobilenetv2", "mlp", "logreg"}:
        try:
            import keras

            keras.backend.clear_session()
        except Exception:
            pass
    model = ServiceRunner(config)._build_model(meta=dict(meta or {}), task_family=task_family)
    restore_model_tensor_state(model, state, verify=True)
    outputs, latency = _predict_outputs(
        model, _select(x_test, test_ids, plan.audit_ids), task_family=task_family
    )
    return ServicePrediction(
        str(runtime_id),
        plan.audit_ids,
        outputs,
        latency,
        plan.label_hash,
        task_family=task_family,
    )


def regenerate_service_prediction(
    *, db_path: str | Path, service_id: str, plan: CohortPlan, artifact_dir: str | Path, seed: int | None = None
) -> ServicePrediction:
    """Train/evaluate a representative using only plan-approved data partitions.

    The source service's original metrics are read, never changed.  This is
    intentionally a regeneration, not an assertion that old weights can be
    recreated bit-for-bit.
    """
    row = _service_row(db_path, service_id)
    config = _rehydration_config(row)
    if seed is not None:
        config["seed"] = int(seed)
    _seed_everything(int(config.get("seed", plan.seed)))
    from mlaas_data_generator.services.runner import ServiceRunner, _load_dataset

    train, test, meta = _load_dataset(config.get("dataset", "hf"), **dict(config["dataset_args"]))
    (x_train, y_train), (x_test, y_test) = train, test
    train_ids = tuple(f"train:{index}" for index in range(_size(y_train)))
    test_ids = tuple(f"test:{index}" for index in range(_size(y_test)))
    assert_cohort_isolation(plan.training_ids, plan.catalogue_metric_ids, plan.audit_ids)
    if set(plan.training_ids) - set(train_ids) or set(plan.catalogue_metric_ids).union(plan.audit_ids) - set(test_ids):
        raise CohortIsolationError("loaded dataset no longer matches the immutable cohort plan IDs")
    audit_labels = _select(y_test, test_ids, plan.audit_ids)
    source_labels = _select(y_test, test_ids, plan.catalogue_metric_ids)
    source_ids = tuple(plan.catalogue_metric_ids) + tuple(plan.audit_ids)
    all_source_labels = _select(y_test, test_ids, source_ids)
    if cohort_label_hash(source_ids, np.asarray(all_source_labels).tolist()) != plan.label_hash:
        raise CohortIsolationError(
            "loaded test labels do not match the immutable cohort label contract"
        )

    runner = ServiceRunner(config)
    task_family = "classification"
    model = runner._build_model(meta=dict(meta or {}), task_family=task_family)
    training_regime = str(config.get("training_regime") or "finetune_transfer").lower()
    if training_regime not in {"inference", "inference_only"}:
        runner._train_model(model, _select(x_train, train_ids, plan.training_ids), _select(y_train, train_ids, plan.training_ids), task_family=task_family)
    _, primary, _, _ = runner._evaluate_model(model, _select(x_test, test_ids, plan.catalogue_metric_ids), source_labels, inference_only=training_regime in {"inference", "inference_only"}, task_family=task_family)
    probabilities, latency = _predict_probabilities(model, _select(x_test, test_ids, plan.audit_ids))
    artifact = _save_checkpoint(model, artifact_dir, service_id, plan.cohort_id)
    regenerated_accuracy = float(primary)
    original_accuracy = _metric_score(db_path, service_id)
    _persist_regeneration(db_path, service_id, plan, config, artifact, original_accuracy, regenerated_accuracy)
    return ServicePrediction(service_id, plan.audit_ids, probabilities, latency, plan.label_hash)


def _execute_independent_family_state(evidence, runtime_id, plan, state):
    from mlaas_data_generator.cli.manifest.preparation import read_preparation, load_prepared, verify_loaded
    from mlaas_data_generator.checkpoint_store import restore_model_tensor_state
    from mlaas_data_generator.data.image_store import close_image_stores
    doc = read_preparation(evidence["family_document_ref"], evidence["family_document_hash"])
    for requested, key in ((plan.training_ids, "training_ids"), (plan.catalogue_metric_ids, "catalogue_metric_ids"), (plan.audit_ids, "audit_ids")):
        if set(requested) != set(doc["reservation"][key]):
            raise CohortIsolationError("audit plan differs from the family's frozen reservation")
    config, prepared = load_prepared(doc["config"])
    train, test, meta, task_family, model, train_ids, test_ids = prepared
    try:
        verify_loaded(doc, prepared, config)
        source_ids = tuple(plan.catalogue_metric_ids) + tuple(plan.audit_ids)
        labels = _select(test[1], tuple(test_ids), source_ids)
        label_values = _frozen_target_values(labels, doc["reservation"])
        if cohort_label_hash(source_ids, label_values) != plan.label_hash:
            raise CohortIsolationError("reloaded family evaluation labels changed")
        restore_model_tensor_state(model, state, verify=True)
        outputs, latency = _predict_frozen_task(model, test, test_ids, plan, config, task_family)
        return ServicePrediction(str(runtime_id), plan.audit_ids, outputs, latency, plan.label_hash, task_family)
    finally:
        close_image_stores((train, test))


def execute_raw_family_state(member, runtime_id, plan, state, *, device="cpu"):
    """Build the frozen raw-source interface without training another provider."""
    from mlaas_data_generator.cli.manifest.preparation import prepare
    from mlaas_data_generator.client_reconstruction import recorded_model_source
    from mlaas_data_generator.checkpoint_store import restore_model_tensor_state
    from mlaas_data_generator.data.image_store import close_image_stores
    doc, recipe = member["parent_document"], member["recipe"]
    config = {**doc["config"], "service_id": member["service_id"], "device": device,
              "provenance_artifact_dir": doc["store_root"], "_family_reference": recipe["family"],
              "_recorded_model_source": recorded_model_source(doc, recipe, member.get("provenance_json")),
              "_recorded_source_ids": recipe["selection"]["ordered_source_row_ids"]}
    _, _, cfg, prepared, _, _ = prepare(config, allow_legacy_evaluation=True)
    train, test, meta, task, model, _, test_ids = prepared
    try:
        source_ids = tuple(doc["reservation"]["metric_source_ids"])
        labels = _select(test[1], tuple(test_ids), source_ids)
        # Dynamic token padding is not part of target identity.
        from mlaas_data_generator.data.source_contract import unpadded_targets
        label_values = _frozen_target_values(unpadded_targets(labels, meta), doc["reservation"])
        if cohort_label_hash(source_ids, label_values) != plan.label_hash:
            raise CohortIsolationError("raw-source audit labels changed")
        restore_model_tensor_state(model, state, verify=True)
        core = getattr(model, "core", None)
        for key, value in recipe.get("evaluation", {}).get("effective_settings", {}).items():
            if core is not None:
                setattr(core, key, value)
        outputs, latency = _predict_frozen_task(model, test, test_ids, plan, cfg, task)
        from mlaas_data_generator.tasks.runtime import TaskEvaluation
        if isinstance(outputs, TaskEvaluation):
            from dataclasses import replace
            from mlaas_data_generator.tasks.contracts import target_fingerprints
            audit_targets = _select(test[1], tuple(test_ids), plan.audit_ids)
            outputs = replace(outputs, target_fingerprints=tuple(target_fingerprints(unpadded_targets(audit_targets, meta))))
        return ServicePrediction(runtime_id, plan.audit_ids, outputs, latency, plan.label_hash, task)
    finally:
        close_image_stores((train, test))


def _execute_shared_parent_state(db_path, parent_id, runtime_id, plan, state):
    from mlaas_data_generator.shared_parent import verify_parent, _prepare, _compatibility, _contracts, prepared_data_hash
    from mlaas_data_generator.checkpoint_store import restore_model_tensor_state
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT document_json FROM parent_checkpoints WHERE parent_checkpoint_id=? AND status='completed'", (parent_id,)).fetchone()
    if row is None:
        raise CohortIsolationError("shared parent is missing")
    parent = json.loads(row[0])
    verify_parent(parent)
    reservation = parent["reservation"]
    for requested, key in ((plan.training_ids, "training_ids"), (plan.catalogue_metric_ids, "catalogue_metric_ids"), (plan.audit_ids, "audit_ids")):
        if set(requested) != set(reservation[key]):
            raise CohortIsolationError("audit plan differs from the parent's frozen reservation")
    config = dict(parent["config"])
    train, test, meta, task_family, model, train_ids, test_ids = _prepare(config)
    if parent.get("prepared_data_hash") and prepared_data_hash((train, test)) != parent["prepared_data_hash"]:
        raise CohortIsolationError("reloaded features or labels differ from frozen parent data")
    if _compatibility(_contracts(model, config, meta, task_family)) != parent["compatibility"]:
        raise CohortIsolationError("reconstructed model/data contract changed")
    if train_ids != reservation["training_ids"] or test_ids != reservation["metric_source_ids"]:
        raise CohortIsolationError("reloaded source identities changed")
    source_ids = tuple(plan.catalogue_metric_ids) + tuple(plan.audit_ids)
    labels = _select(test[1], tuple(test_ids), source_ids)
    if cohort_label_hash(source_ids, _frozen_target_values(labels, reservation)) != plan.label_hash:
        raise CohortIsolationError("reloaded audit labels changed")
    restore_model_tensor_state(model, state, verify=True)
    from mlaas_data_generator.data.image_store import close_image_stores
    try:
        outputs, latency = _predict_frozen_task(model, test, test_ids, plan, config, task_family)
        return ServicePrediction(str(runtime_id), plan.audit_ids, outputs, latency, plan.label_hash, task_family)
    finally:
        close_image_stores((train, test))


def _frozen_target_values(labels, reservation):
    if reservation.get("version") == "reserved_evaluation_v2":
        from mlaas_data_generator.tasks.contracts import target_fingerprints
        return target_fingerprints(labels)
    return np.asarray(labels).tolist()


def _predict_frozen_task(model, test, test_ids, plan, config, task_family):
    x = _select(test[0], tuple(test_ids), plan.audit_ids)
    if config.get("task_contract") and task_family in {"token_classification", "segmentation", "generation", "fill_mask"}:
        from mlaas_data_generator.tasks import TaskContract
        from mlaas_data_generator.tasks.runtime import evaluate_task_model
        y = _select(test[1], tuple(test_ids), plan.audit_ids)
        result = evaluate_task_model(model, x, y, TaskContract.from_dict(config["task_contract"]), plan.audit_ids)
        return result, result.inference_latency_s
    return _predict_outputs(model, x, task_family=task_family)


def _service_row(db_path: str | Path, service_id: str) -> Mapping[str, Any]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT s.*,
                   g.provenance_json AS generation_provenance_json,
                   c.reconstruction_artifact_ref,
                   c.target_state_sha256
            FROM services s
            LEFT JOIN service_generation_provenance g ON g.service_id = s.service_id
            LEFT JOIN service_checkpoint_storage c ON c.service_id = s.service_id
            WHERE s.service_id = ?
            """,
            (service_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"service '{service_id}' does not exist")
    return dict(row)


def _rehydration_config(row: Mapping[str, Any]) -> dict[str, Any]:
    service_config = _json_object(row.get("service_config_json"))
    metadata = _json_object(row.get("metadata_json"))
    loader_meta = _json_object(metadata.get("loader_meta"))
    dataset_args = _json_object(loader_meta.get("dataset_args"))
    generation = _json_object(row.get("generation_provenance_json"))
    if not dataset_args:
        dataset_args = _json_object(_json_object(generation.get("dataset")).get("loader_args"))
    config = dict(service_config)
    config.update({
        "service_id": str(row["service_id"]), "dataset": row.get("dataset"), "dataset_name": row.get("dataset_name"),
        "dataset_config": row.get("dataset_config"), "benchmark_split": row.get("benchmark_split"),
        "model_type": row.get("model_type"), "hf_model_id": row.get("model_id"), "hf_task": row.get("hf_task"),
        "training_regime": row.get("training_regime"), "task_type": row.get("task_type") or "classification",
        "modality": row.get("modality"), "dataset_args": dataset_args,
    })
    for key in (
        "dataset_name", "dataset_config", "hf_model_id", "hf_task", "modality",
        "max_samples", "max_length", "train_split", "benchmark_split",
        "training_regime", "input_schema", "service_source", "model_role",
    ):
        value = config.get(key)
        if value is not None:
            dataset_args.setdefault(key, value)
    config["dataset_args"].setdefault("task", config["task_type"])
    return config


def _checkpoint_root(row: Mapping[str, Any], db_path: str | Path) -> Path:
    reference = row.get("reconstruction_artifact_ref")
    if reference:
        path = Path(str(reference)).resolve()
        for parent in path.parents:
            if parent.name in {"objects", "references"}:
                return parent.parent
    return Path(db_path).resolve().parent / "service_artifacts"


def _predict_outputs(model: Any, x: Any, *, task_family: str) -> tuple[np.ndarray, float]:
    core = getattr(model, "core", None)
    if core is not None:
        return _predict_hf_outputs(model, x, task_family=task_family)
    started = __import__("time").perf_counter()
    try:
        raw = model.predict(x, verbose=0)
    except TypeError:
        raw = model.predict(x)
    elapsed = __import__("time").perf_counter() - started
    values = np.asarray(raw, dtype=float)
    if task_family == "regression":
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError("regression model returned non-scalar predictions")
        return values, float(elapsed)
    return _as_probabilities(values), float(elapsed)


def _predict_hf_outputs(model: Any, x: Any, *, task_family: str) -> tuple[np.ndarray, float]:
    core = model.core
    torch = core.torch
    batches: list[np.ndarray] = []
    started = __import__("time").perf_counter()
    core.model.eval()
    with torch.no_grad():
        for xb, _ in core._batch_iter(x, None):
            enc, _, _ = core.task_spec.encode_batch(
                core.tokenizer,
                xb,
                None,
                core.max_length,
                torch,
                core.device,
                ignore_index=core.label_pad_value,
                inference_only=True,
            )
            inputs = core.task_spec.build_forward_inputs(
                enc, labels_t=None, inference_only=True
            )
            logits = core._extract_logits(core.model(**inputs))
            if task_family == "classification":
                logits = torch.softmax(logits, dim=-1)
            batches.append(logits.detach().cpu().numpy())
    elapsed = __import__("time").perf_counter() - started
    values = np.concatenate(batches, axis=0)
    if task_family == "regression":
        values = values.reshape(values.shape[0], -1)
        if values.shape[1] != 1:
            raise ValueError("regression model returned non-scalar predictions")
        values = values[:, 0]
    return values, float(elapsed)


def _as_probabilities(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        if np.issubdtype(values.dtype, np.integer):
            classes = int(np.max(values)) + 1
            return np.eye(classes, dtype=float)[values.astype(int)]
        values = np.column_stack((1.0 - values, values))
    if values.ndim != 2:
        raise ValueError("classification model returned invalid probability dimensions")
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(values < 0.0) or np.any(values > 1.0) or not np.allclose(row_sums, 1.0, atol=1e-5):
        shifted = values - np.max(values, axis=1, keepdims=True)
        exponent = np.exp(shifted)
        values = exponent / exponent.sum(axis=1, keepdims=True)
    return values


def _predict_probabilities(model: Any, x: Any) -> tuple[np.ndarray, float]:
    """Run a classifier through the same task adapter used by service evaluation."""
    core = getattr(model, "core", None)
    if core is None:
        raise ValueError("strict validator currently requires an HF adapter with a core")
    torch = core.torch
    batches: list[np.ndarray] = []
    started = __import__("time").perf_counter()
    core.model.eval()
    with torch.no_grad():
        for xb, _ in core._batch_iter(x, None):
            enc, _, _ = core.task_spec.encode_batch(core.tokenizer, xb, None, core.max_length, torch, core.device, ignore_index=core.label_pad_value, inference_only=True)
            inputs = core.task_spec.build_forward_inputs(enc, labels_t=None, inference_only=True)
            logits = core._extract_logits(core.model(**inputs))
            batches.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    elapsed = __import__("time").perf_counter() - started
    return np.concatenate(batches, axis=0), float(elapsed)


def _save_checkpoint(model: Any, artifact_dir: str | Path, service_id: str, cohort_id: str) -> str:
    target = Path(artifact_dir) / "regenerations" / cohort_id / _safe_name(service_id)
    target.mkdir(parents=True, exist_ok=True)
    core = getattr(model, "core", None)
    if core is None or not callable(getattr(core.model, "save_pretrained", None)):
        raise ValueError("model cannot be checkpointed as a Hugging Face artifact")
    core.model.save_pretrained(target, safe_serialization=True)
    if callable(getattr(core.tokenizer, "save_pretrained", None)):
        core.tokenizer.save_pretrained(target)
    return str(target)


def _persist_regeneration(db_path: str | Path, service_id: str, plan: CohortPlan, config: Mapping[str, Any], artifact: str, original: float | None, regenerated: float) -> None:
    from mlaas_data_generator.storage.writer import make_writer
    config_json = json.dumps(config, sort_keys=True, default=str)
    writer = make_writer("sqlite", db_path=str(db_path)); writer.start()
    try:
        writer.write_service_regeneration({
            "regeneration_id": str(uuid.uuid4()), "service_id": service_id, "cohort_id": plan.cohort_id,
            "cohort_plan_hash": plan.audit_hash,
            "seed": int(config.get("seed", plan.seed)), "resolved_config_json": config_json,
            "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_versions_json": {"python": sys.version.split()[0]},
            "provenance_json": {"representative_not_original_weights": True, "cohort_hash": plan.audit_hash},
            "original_accuracy": original, "regenerated_accuracy": regenerated,
            "accuracy_difference": None if original is None else regenerated - original, "artifact_uri": artifact,
        }); writer.finish()
    except Exception:
        writer.abort(); raise


def _metric_score(db_path: str | Path, service_id: str) -> float | None:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT COALESCE(value_num, CAST(value_int AS REAL)) FROM service_metrics WHERE service_id = ? AND metric_name = 'metric_score'", (service_id,)).fetchone()
    return None if row is None or row[0] is None else float(row[0])


def _select(value: Any, ids: tuple[str, ...], wanted: tuple[str, ...]) -> Any:
    positions = [ids.index(item) for item in wanted]
    if isinstance(value, Mapping):
        return {key: _select(item, ids, wanted) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[positions]
    return [value[index] for index in positions]


def _size(value: Any) -> int:
    return len(next(iter(value.values()))) if isinstance(value, Mapping) else len(value)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping): return dict(value)
    if isinstance(value, str):
        try: return dict(json.loads(value))
        except (ValueError, TypeError): return {}
    return {}


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)
