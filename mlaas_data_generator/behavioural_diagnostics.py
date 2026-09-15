"""Post-generation behaviour on the frozen catalogue benchmark, never the audit.

Diagnostics are descriptive artifacts; they neither train models nor alter the
catalogue. Pair comparisons use one common complete set of evaluation units.
"""
from __future__ import annotations

import csv
import inspect
import hashlib
import json
import sqlite3
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .checkpoint_store import reconstruct_service_state, restore_model_tensor_state
from .data.fedavg_partition import _take_rows
from .provenance import sha256_json
from .shared_parent import (DEFAULT_COHORT_DB, _prepare, _compatibility, _contracts, _clean,
                            prepared_data_hash, verify_parent)

VERSION = "behavioural_diversity_v1"


def _stats(values) -> dict:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return dict(count=0, mean=None, median=None, minimum=None, maximum=None, std=None)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()), std=float(values.std()))


def diagnostic_targets(labels, task_family: str, *, ignore_index: int = -100) -> np.ndarray:
    values = np.asarray(labels)
    if task_family == "regression":
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("regression diagnostics require finite scalar targets")
        return values.astype(float)
    if task_family == "token_classification":
        if values.ndim != 2:
            raise ValueError("token diagnostics require aligned token-index labels")
        values = values[values != ignore_index]
    elif task_family == "classification":
        if values.ndim == 2:
            if values.shape[1] == 1:
                values = values[:, 0]
            elif np.isin(values, [0, 1]).all() and np.all(values.sum(axis=1) == 1):
                values = values.argmax(axis=1)
            else:
                raise ValueError("class diagnostics require single-label targets, not multilabel targets")
    else:
        raise ValueError(f"unsupported behavioural diagnostic task: {task_family}")
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("classification diagnostics require nonnegative integer class IDs")
    return values.astype(np.int64)


def analyse_predictions(targets, predictions: Mapping[str, Any], *, task_family="classification",
                        class_labels: Mapping[int, str] | None = None) -> dict:
    """Calculate all unordered pairs without allocating a clients² × examples tensor.

    Predictions are aligned hard class IDs (or scalar regression predictions).
    Token padding must already be removed with the shared target mask.
    """
    regression = task_family == "regression"
    truth = diagnostic_targets(targets, "regression" if regression else "classification")
    if not len(truth):
        raise ValueError("the catalogue benchmark has no scored evaluation units")
    values = {}
    for sid in sorted(predictions):
        pred = np.asarray(predictions[sid])
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(f"service {sid} has incomplete or non-finite predictions")
        if not regression and (np.any(pred < 0) or np.any(pred != np.floor(pred))):
            raise ValueError(f"service {sid} has invalid predicted class IDs")
        values[sid] = pred.astype(float if regression else np.int64)
    services, pairs, per_class, spreads = [], [], [], []
    errors = {sid: pred - truth if regression else pred != truth for sid, pred in values.items()}
    for sid, error in errors.items():
        quality = ({"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}
                   if regression else {"accuracy": float(1 - error.mean()), "error_count": int(error.sum())})
        services.append({"service_id": sid, "unit_count": len(truth), **quality})
    for left, right in combinations(values, 2):
        row = {"service_id_a": left, "service_id_b": right, "unit_count": len(truth)}
        a, b = errors[left], errors[right]
        if regression:
            delta = values[left] - values[right]
            correlation = None
            if len(truth) > 1 and a.std() > 0 and b.std() > 0:
                correlation = float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1))
            row.update(prediction_mae=float(np.abs(delta).mean()),
                       prediction_rmse=float(np.sqrt(np.mean(delta ** 2))), residual_correlation=correlation)
        else:
            shared, union = int((a & b).sum()), int((a | b).sum())
            disagreement = int((values[left] != values[right]).sum())
            row.update(disagreement_count=disagreement, pairwise_disagreement=disagreement / len(truth),
                       shared_error_count=shared, error_union_count=union,
                       error_overlap=shared / len(truth), error_jaccard=shared / union if union else None)
        pairs.append(row)
    if not regression:
        names = {int(k): str(v) for k, v in (class_labels or {}).items()}
        classes = sorted(set(truth.tolist()) | set(names))
        for cls in classes:
            mask = truth == cls
            count = int(mask.sum())
            scores = []
            for sid, pred in values.items():
                score = float(np.mean(pred[mask] == cls)) if count else None
                scores.append(score)
                per_class.append({"service_id": sid, "class_id": cls, "class_label": names.get(cls, str(cls)),
                                  "support": count, "accuracy": score})
            stats = _stats(scores)
            spreads.append({"class_id": cls, "class_label": names.get(cls, str(cls)), "support": count,
                            **stats, "accuracy_spread": stats["maximum"] - stats["minimum"] if stats["count"] else None})
    metrics = (("prediction_mae", "prediction_rmse", "residual_correlation") if regression else
               ("pairwise_disagreement", "error_overlap", "error_jaccard"))
    prediction_groups, profiles = {}, []
    for sid, pred in values.items():
        digest = hashlib.sha256(pred.dtype.str.encode() + pred.tobytes()).hexdigest()
        prediction_groups.setdefault(digest, []).append(sid)
        profile = {"service_id": sid, "prediction_hash": digest,
                   "constant_prediction": bool(np.all(pred == pred[0]))}
        if regression:
            baseline_rmse = float(np.sqrt(np.mean((truth - truth.mean()) ** 2)))
            scale = max(float(truth.std()), np.finfo(float).eps)
            profile.update(prediction_std=float(pred.std()), prediction_std_over_target_std=float(pred.std()/scale),
                           benchmark_mean_baseline_rmse=baseline_rmse)
        else:
            from sklearn.metrics import f1_score
            labels, counts = np.unique(pred, return_counts=True)
            profile.update(predicted_class_counts=dict(zip(map(str, labels), map(int, counts))),
                           macro_f1=float(f1_score(truth, pred, average="macro", zero_division=0)),
                           weighted_f1=float(f1_score(truth, pred, average="weighted", zero_division=0)))
        profiles.append(profile)
    return {"service_count": len(values), "pair_count": len(pairs), "unit_count": len(truth),
            "prediction_diversity": {"unique_prediction_vectors": len(prediction_groups),
                "constant_predictors": sum(p["constant_prediction"] for p in profiles),
                "identical_prediction_groups": [ids for ids in prediction_groups.values() if len(ids) > 1]},
            "prediction_profiles": profiles,
            "pairwise_summary": {name: _stats(row[name] for row in pairs) for name in metrics},
            "services": services, "pairs": pairs, "per_class": per_class, "per_class_spread": spreads}


def predict_decisions(model, x, y, *, task_family: str, batch_size: int, ignore_index: int) -> np.ndarray:
    """Keep only hard decisions, avoiding per-client full probability tensors."""
    core = getattr(model, "core", None)
    if core is None:
        parameters = inspect.signature(model.predict).parameters
        kwargs = {"verbose": 0} if "verbose" in parameters else {}
        if "batch_size" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            kwargs["batch_size"] = batch_size
        outputs = np.asarray(model.predict(x, **kwargs))
        if not np.isfinite(outputs).all():
            raise ValueError("non-finite model outputs")
        if task_family == "regression":
            return outputs.reshape(-1) if outputs.ndim == 2 and outputs.shape[1] == 1 else outputs
        if outputs.ndim == 2 and outputs.shape[1] > 1:
            return outputs.argmax(axis=1)
        if outputs.ndim == 2 and outputs.shape[1] == 1:
            return (outputs[:, 0] >= 0.5).astype(int)
        return outputs
    core.batch_size = batch_size
    core.model.eval()
    decisions, observed_targets = [], []
    with core.torch.no_grad():
        for xb, yb in core._batch_iter(x, y):
            enc, _, _ = core.task_spec.encode_batch(core.tokenizer, xb, None, core.max_length,
                core.torch, core.device, ignore_index=ignore_index, inference_only=True)
            inputs = core.task_spec.build_forward_inputs(enc, labels_t=None, inference_only=True)
            logits = core._extract_logits(core._run_with_precision_context(lambda: core.model(**inputs)))
            if not bool(core.torch.isfinite(logits).all()):
                raise ValueError("non-finite model outputs")
            pred = logits.squeeze(-1) if task_family == "regression" else logits.argmax(dim=-1)
            pred = pred.detach().cpu().numpy()
            if task_family == "token_classification":
                mask = np.asarray(yb) != ignore_index
                if pred.shape != mask.shape:
                    raise ValueError("token outputs do not align with the frozen label mask")
                pred = pred[mask]
            decisions.append(pred)
            observed_targets.append(diagnostic_targets(yb, task_family, ignore_index=ignore_index))
    expected = diagnostic_targets(y, task_family, ignore_index=ignore_index)
    if not observed_targets or not np.array_equal(np.concatenate(observed_targets), expected):
        raise ValueError("inference changed the evaluation membership or order")
    return np.concatenate(decisions)


def _snapshot(db: str | Path, cohort_id: str):
    source = Path(db).resolve(strict=True)
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        cohort = conn.execute("SELECT * FROM service_cohorts WHERE cohort_id=?", (cohort_id,)).fetchone()
        if cohort is None:
            raise ValueError(f"shared-parent cohort not found: {cohort_id}")
        parent = conn.execute("SELECT document_json FROM parent_checkpoints WHERE parent_checkpoint_id=? AND status='completed'",
                              (cohort["parent_checkpoint_id"],)).fetchone()
        if parent is None:
            raise ValueError("cohort has no completed parent")
        clients = [dict(row) for row in conn.execute("""
            SELECT cc.*, s.status AS service_status, g.parent_checkpoint_id, g.parent_state_hash,
                   g.compatibility_contract_hash, g.final_model_hash, g.client_id AS provenance_client_id,
                   g.cohort_id AS provenance_cohort_id, g.parameter_schema_hash
            FROM cohort_clients cc LEFT JOIN services s ON s.service_id=cc.service_id
            LEFT JOIN service_generation_provenance g ON g.service_id=cc.service_id
            WHERE cc.cohort_id=? ORDER BY cc.client_id,cc.service_id
        """, (cohort_id,))]
    return dict(cohort), json.loads(parent[0]), clients


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def family_diagnostics(*, db: str | Path, family_id: str, output_dir: str | Path,
                       batch_size: int | None = None, load_dataset=None, build_model=None) -> dict:
    """Compare independent checkpoints on their frozen catalogue subset only.

    This does not train, aggregate, reject providers or access the audit subset
    for prediction. Reloading verifies the original preparation before scoring.
    """
    from .cli.manifest import preparation
    from .clients.recipes import recipe_summary
    from .data.image_store import close_image_stores
    from .reduced_checkpoint import source_state_hash

    if batch_size is not None and batch_size < 1:
        raise ValueError("diagnostic batch size must be positive")
    source = Path(db).resolve(strict=True)
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("BEGIN")
        clients = [dict(r) for r in con.execute("""
            SELECT s.service_id,s.status,s.service_config_json,g.family_id,g.family_document_ref,
                   g.family_document_hash,g.final_model_hash,g.initial_state_hash,
                   g.parameter_schema_hash,g.compatibility_contract_hash,g.composition_compatible
            FROM services s JOIN service_generation_provenance g USING(service_id)
            WHERE g.family_id=? AND g.generation_mode='independent' ORDER BY s.service_id
        """, (family_id,))]
    if not clients:
        raise ValueError("no independent services found for this family")
    first = clients[0]
    doc = preparation.read_preparation(first["family_document_ref"], first["family_document_hash"])
    if doc["family_id"] != family_id:
        raise ValueError("preparation belongs to a different family")
    config = {**doc["config"], "provenance_artifact_dir": doc["store_root"]}
    if not config.get("task_contract") and config.get("hf_task") not in {None, "sequence_classification", "token_classification", "image_classification", "sentence_similarity"}:
        raise ValueError("family task does not support aligned prediction diagnostics")
    cfg, prepared = preparation.load_prepared(config, load_dataset=load_dataset, build_model=build_model)
    try:
        preparation.verify_loaded(doc, prepared, cfg)
        train, test, meta, task, model, train_ids, test_ids = prepared
        if cfg.get("task_contract"):
            from .tasks import TaskContract
            from .tasks.diagnostics import diagnose_tasks
            reservation = doc["reservation"]
            ids = reservation["catalogue_metric_ids"]
            positions = np.asarray(reservation["catalogue_positions"], dtype=int)
            if [test_ids[i] for i in positions] != ids or set(ids) & (set(train_ids) | set(reservation["audit_ids"])):
                raise ValueError("catalogue diagnostic membership differs from the frozen reservation")
            model.core.batch_size = int(batch_size or cfg.get("batch_size") or 32)
            def restore(client):
                if client["status"] != "completed" or client["composition_compatible"] != 1:
                    raise ValueError("client is not a completed compatible member")
                expected = {"family_document_ref": first["family_document_ref"], "family_document_hash": first["family_document_hash"],
                            "initial_state_hash": doc["initial_state_hash"], "compatibility_contract_hash": doc["compatibility_contract_hash"],
                            "parameter_schema_hash": doc["compatibility"]["parameter_schema_hash"]}
                if any(client[k] != v for k, v in expected.items()):
                    raise ValueError("client provenance differs from frozen family")
                state = reconstruct_service_state(doc["store_root"], client["service_id"])
                if source_state_hash(doc["store_root"], client["service_id"], state) != client["final_model_hash"]:
                    raise ValueError("client checkpoint hash mismatch")
                restore_model_tensor_state(model, state, verify=True)
                return state.sha256
            return diagnose_tasks(model, TaskContract.from_dict(cfg["task_contract"]),
                _take_rows(test[0], positions), _take_rows(test[1], positions), ids, clients, restore, output_dir,
                {"family_id": family_id, "reservation_hash": reservation["reservation_hash"], "historical_evaluation_exposure": "unknown"})
        if task not in {"classification", "token_classification", "regression"} or meta.get("label_format") in {"multilabel", "multihot"}:
            raise ValueError("diagnostics require single-label classification, aligned tokens or scalar regression")
        reservation = doc["reservation"]
        positions = np.asarray(reservation["catalogue_positions"], dtype=int)
        ids = reservation["catalogue_metric_ids"]
        if (not ids or len(set(ids)) != len(ids) or [test_ids[i] for i in positions] != ids
                or set(ids) & (set(train_ids) | set(reservation["audit_ids"]))):
            raise ValueError("catalogue prediction membership is invalid or overlaps training/audit data")
        x, y = _take_rows(test[0], positions), _take_rows(test[1], positions)
        ignore = meta.get("ignore_index")
        if ignore is None:
            ignore = meta.get("label_pad_value")
        ignore = int(-100 if ignore is None else ignore)
        targets = diagnostic_targets(y, task, ignore_index=ignore)
        labels = doc["contracts"]["labels"]
        names = {int(row["index"]): row["label"] for row in labels.get("labels", [])}
        for index in range(int(labels.get("num_labels") or 0)):
            names.setdefault(index, str(index))
        predictions, members, excluded = {}, [], []
        for client in clients:
            sid = client["service_id"]
            try:
                if client["status"] != "completed" or client["composition_compatible"] != 1:
                    raise ValueError("service is not a completed composition-compatible family member")
                expected = {"family_document_ref": first["family_document_ref"], "family_document_hash": first["family_document_hash"],
                            "initial_state_hash": doc["initial_state_hash"], "compatibility_contract_hash": doc["compatibility_contract_hash"],
                            "parameter_schema_hash": doc["compatibility"]["parameter_schema_hash"]}
                if any(client[k] != v for k, v in expected.items()):
                    raise ValueError("client provenance differs from its frozen family")
                state = reconstruct_service_state(doc["store_root"], sid)
                if source_state_hash(doc["store_root"], sid, state) != client["final_model_hash"]:
                    raise ValueError("client checkpoint hash mismatch")
                restore_model_tensor_state(model, state, verify=True)
                del state
                pred = predict_decisions(model, x, y, task_family=task, batch_size=int(batch_size or cfg.get("batch_size") or 32), ignore_index=ignore)
                if pred.shape != targets.shape or not np.isfinite(pred).all():
                    raise ValueError("predictions do not cover all scored units")
                if task != "regression" and (np.any(pred < 0) or np.any(pred != np.floor(pred)) or (names and not np.isin(pred, list(names)).all())):
                    raise ValueError("predictions violate the frozen class mapping")
                predictions[sid] = pred
                members.append({"service_id": sid, "final_model_hash": client["final_model_hash"]})
            except Exception as exc:
                excluded.append({"service_id": sid, "reason": str(exc)})
            print(f"Family diagnostics: {len(members)+len(excluded)}/{len(clients)} checked", flush=True)
        analysis = analyse_predictions(targets, predictions, task_family=task, class_labels=names)
        recipes = [json.loads(r["service_config_json"]) for r in clients]
        report = {"version": "independent_family_diversity_v1", "family_id": family_id,
            "status": "insufficient_clients" if len(members) < 2 else "partial" if excluded else "completed",
            "evaluation_subset": "catalogue_metric", "evaluation_ids": ids, "evaluation_ids_hash": sha256_json(ids),
            "target_hash": prepared_data_hash(targets), "task_family": task,
            "historical_evaluation_exposure": reservation.get("historical_evaluation_exposure", "unknown"),
            "diagnosed_clients": len(members), "recorded_clients": len(clients), "members": members, "excluded": excluded,
            "recipe_diversity": recipe_summary(recipes), **analysis}
        report["report_hash"] = sha256_json(report)
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        # Numeric arrays and Unicode IDs only: readers do not need pickle.
        np.savez_compressed(out / "predictions.npz", targets=targets, service_ids=np.asarray(list(predictions), dtype=str),
                            predictions=np.stack(list(predictions.values())) if predictions else np.empty((0, len(targets))))
        for key in ("services", "pairs", "per_class", "prediction_profiles", "excluded"):
            columns = list(dict.fromkeys(k for row in report[key] for k in row))
            _write_csv(out / f"{key}.csv", report[key], columns)
        return report
    finally:
        close_image_stores(prepared[:2])


def cohort_diagnostics(*, db: str | Path = DEFAULT_COHORT_DB, cohort_id: str,
                       output_dir: str | Path, batch_size: int | None = None,
                       load_dataset=None, build_model=None) -> dict:
    """Reconstruct completed clients sequentially and publish reviewable JSON/CSV."""
    if batch_size is not None and batch_size < 1:
        raise ValueError("diagnostic batch size must be positive")
    cohort, parent, clients = _snapshot(db, cohort_id)
    verify_parent(parent)
    config = dict(parent["config"])
    train, test, meta, task, model, train_ids, test_ids = _prepare(config, load_dataset=load_dataset, build_model=build_model)
    reservation = parent["reservation"]
    if train_ids != reservation["training_ids"] or test_ids != reservation["metric_source_ids"]:
        raise ValueError("reloaded source IDs differ from the parent reservation")
    from .tasks.contracts import target_fingerprints
    loaded_targets = target_fingerprints(test[1]) if reservation.get("version") == "reserved_evaluation_v2" else _clean(test[1])
    if loaded_targets != reservation["metric_source_labels"]:
        raise ValueError("reloaded labels differ from the parent reservation")
    if parent.get("prepared_data_hash") and prepared_data_hash((train, test)) != parent["prepared_data_hash"]:
        raise ValueError("reloaded data differ from the frozen parent data")
    if _compatibility(_contracts(model, config, meta, task)) != parent["compatibility"]:
        raise ValueError("diagnostic model/data contract differs from the parent")
    if meta.get("label_format") in {"multilabel", "multihot"}:
        raise ValueError("class diversity diagnostics currently require single-label outputs")
    metric_ids = reservation["catalogue_metric_ids"]
    if set(metric_ids) & (set(reservation["audit_ids"]) | set(train_ids)):
        raise ValueError("catalogue diagnostic examples overlap training or audit data")
    positions = np.asarray(reservation["catalogue_positions"], dtype=int)
    if [test_ids[i] for i in positions] != metric_ids or len(set(metric_ids)) != len(metric_ids):
        raise ValueError("catalogue diagnostic positions do not match frozen IDs")
    x, y = _take_rows(test[0], positions), _take_rows(test[1], positions)
    if config.get("task_contract"):
        from .tasks import TaskContract
        from .tasks.diagnostics import diagnose_tasks
        from .reduced_checkpoint import source_state_hash
        from .data.image_store import close_image_stores
        model.core.batch_size = int(batch_size or config.get("batch_size") or 32)
        def restore(client):
            if client["status"] != "completed" or client["service_status"] != "completed":
                raise ValueError("client has not completed")
            expected = {"parent_checkpoint_id": parent["parent_checkpoint_id"], "parent_state_hash": parent["parent_state_hash"],
                        "compatibility_contract_hash": parent["compatibility_contract_hash"], "provenance_cohort_id": cohort_id,
                        "parameter_schema_hash": parent["compatibility"]["parameter_schema_hash"], "provenance_client_id": client["client_id"]}
            if any(client[k] != v for k, v in expected.items()):
                raise ValueError("client provenance differs from frozen cohort")
            state = reconstruct_service_state(parent["store_root"], client["service_id"])
            source_hash = source_state_hash(parent["store_root"], client["service_id"], state)
            if source_hash != client["final_state_hash"] or source_hash != client["final_model_hash"]:
                raise ValueError("client checkpoint hash mismatch")
            restore_model_tensor_state(model, state, verify=True)
            return state.sha256
        try:
            return diagnose_tasks(model, TaskContract.from_dict(config["task_contract"]), x, y, metric_ids, clients,
                restore, output_dir, {"cohort_id": cohort_id, "reservation_hash": reservation["reservation_hash"], "historical_evaluation_exposure": "unknown"})
        finally:
            close_image_stores((train, test))
    ignore_index = meta.get("ignore_index")
    if ignore_index is None:
        ignore_index = meta.get("label_pad_value")
    ignore_index = int(-100 if ignore_index is None else ignore_index)
    targets = diagnostic_targets(y, task, ignore_index=ignore_index)
    if not len(targets):
        raise ValueError("catalogue benchmark has no scored units")
    size = int(batch_size or config.get("batch_size") or 32)
    labels = parent["contracts"]["labels"]
    class_names = {int(row["index"]): row["label"] for row in labels["labels"]}
    if task != "regression" and labels.get("num_labels"):
        for index in range(int(labels["num_labels"])):
            class_names.setdefault(index, str(index))
    predictions, members, excluded = {}, [], []
    for client in clients:
        sid = client["service_id"]
        if client["status"] != "completed" or client["service_status"] != "completed":
            excluded.append({"service_id": sid, "stage": "eligibility", "reason": f"client={client['status']}; service={client['service_status']}"})
            continue
        try:
            expected = {"parent_checkpoint_id": parent["parent_checkpoint_id"], "parent_state_hash": parent["parent_state_hash"],
                "compatibility_contract_hash": parent["compatibility_contract_hash"], "provenance_cohort_id": cohort_id,
                "parameter_schema_hash": parent["compatibility"]["parameter_schema_hash"], "provenance_client_id": client["client_id"]}
            if any(client[k] != v for k, v in expected.items()):
                raise ValueError("client provenance differs from its cohort/parent")
            state = reconstruct_service_state(parent["store_root"], sid)
            from .reduced_checkpoint import source_state_hash
            source_hash = source_state_hash(parent["store_root"], sid, state)
            if source_hash != client["final_state_hash"] or source_hash != client["final_model_hash"]:
                raise ValueError("client checkpoint state hash mismatch")
            restore_model_tensor_state(model, state, verify=True)
            pred = predict_decisions(model, x, y, task_family=task, batch_size=size, ignore_index=ignore_index)
            if pred.shape != targets.shape or not np.isfinite(pred).all():
                raise ValueError("client predictions do not cover the complete catalogue benchmark")
            if task != "regression" and (np.any(pred < 0) or np.any(pred != np.floor(pred)) or (class_names and not np.isin(pred, list(class_names)).all())):
                raise ValueError("client predictions violate the frozen class mapping")
            predictions[sid] = pred
            members.append({"service_id": sid, "client_id": client["client_id"], "state_hash": state.sha256,
                            "recipe_hash": client["recipe_hash"], "prediction_hash": prepared_data_hash(pred)})
        except Exception as exc:
            excluded.append({"service_id": sid, "stage": "checkpoint_or_inference", "reason": str(exc)})
        print(f"Diagnostics {cohort_id}: {len(members) + len(excluded)}/{len(clients)} clients checked")
    analysis = analyse_predictions(targets, predictions, task_family=task, class_labels=class_names)
    report = {"version": VERSION, "cohort_id": cohort_id, "family_id": parent["family_id"],
        "parent_checkpoint_id": parent["parent_checkpoint_id"], "parent_state_hash": parent["parent_state_hash"],
        "compatibility_contract_hash": parent["compatibility_contract_hash"], "cohort_manifest_hash": cohort["manifest_hash"],
        "reservation_hash": reservation["reservation_hash"], "evaluation_subset": "catalogue_metric",
        "evaluation_ids": metric_ids, "evaluation_ids_hash": sha256_json(metric_ids),
        "target_hash": prepared_data_hash(targets), "task_family": task,
        "evaluation_unit": "supervised_token" if task == "token_classification" else "example",
        "evaluation_example_count": len(metric_ids), "ignore_index": ignore_index if task == "token_classification" else None,
        "batch_size": size, "historical_evaluation_exposure": reservation.get("historical_evaluation_exposure", "unknown"),
        "scheduled_clients": cohort["scheduled_clients"], "recorded_clients": len(clients),
        "diagnosed_clients": len(members), "excluded_clients": len(excluded), "members": members, "excluded": excluded,
        "status": "insufficient_clients" if len(members) < 2 else "partial" if excluded or len(clients) != cohort["scheduled_clients"] else "completed",
        "member_snapshot_hash": sha256_json(members),
        "definitions": {"pairwise_disagreement": "different hard predictions / scored units",
            "error_overlap": "both clients wrong / scored units",
            "error_jaccard": "both wrong / either wrong; null when neither client makes errors",
            "per_class_accuracy": "correct predictions among true examples/tokens of that class (recall)",
            "accuracy_spread": "maximum minus minimum client per-class accuracy; null for absent classes",
            "std": "population standard deviation (ddof=0)",
            "regression": "prediction MAE/RMSE between clients and Pearson correlation of residuals; null correlation for constant residuals"},
        **analysis}
    report["report_hash"] = sha256_json(report)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    tables = {"pairs": ["service_id_a", "service_id_b", "unit_count"] + (
        ["prediction_mae", "prediction_rmse", "residual_correlation"] if task == "regression" else
        ["disagreement_count", "pairwise_disagreement", "shared_error_count", "error_union_count", "error_overlap", "error_jaccard"]),
        "services": ["service_id", "unit_count"] + (["rmse", "mae"] if task == "regression" else ["accuracy", "error_count"]),
        "per_class": ["service_id", "class_id", "class_label", "support", "accuracy"],
        "per_class_spread": ["class_id", "class_label", "support", "count", "mean", "median", "minimum", "maximum", "std", "accuracy_spread"],
        "excluded": ["service_id", "stage", "reason"]}
    for name, columns in tables.items():
        _write_csv(output / f"{name}.csv", report[name], columns)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
