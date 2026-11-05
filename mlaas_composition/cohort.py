"""Frozen audit-cohort readers shared by selection validation and batch replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mlaas_data_generator.artifact_paths import resolve_artifact_path

from .validation import CohortPlan, CohortIsolationError, build_cohort_plan


def read_cohort(
    path: str | Path,
    *,
    seed: int,
    audit_samples: int,
    minimum: int,
    task_family: str,
    selected_training_ids: list[str] | None = None,
) -> CohortPlan:
    payload = json.loads(resolve_artifact_path(path).read_text(encoding="utf-8"))
    if payload.get("version") in {"reserved_evaluation_v1", "reserved_evaluation_v2"}:
        from .validation import CohortPlan, assert_cohort_isolation, cohort_label_hash
        from mlaas_data_generator.provenance import sha256_json
        digest = sha256_json({k: v for k, v in payload.items() if k != "reservation_hash"})
        if digest != payload.get("reservation_hash"):
            raise CohortIsolationError("frozen cohort reservation hash mismatch")
        if payload["task_family"] != task_family:
            raise CohortIsolationError("frozen audit task does not match the composition")
        train, metric, audit = (tuple(payload[k]) for k in ("training_ids", "catalogue_metric_ids", "audit_ids"))
        if selected_training_ids is not None:
            if train and set(train) != set(selected_training_ids):
                raise CohortIsolationError("reservation training IDs differ from selected recipes")
            train = tuple(selected_training_ids)
        assert_cohort_isolation(train, metric, audit)
        if len(audit) < minimum:
            raise CohortIsolationError("frozen audit cohort is below minimum_audit_samples")
        source = set(payload["metric_source_ids"])
        if selected_training_ids is not None:
            # v4 caps catalogue measurements; unused evaluation rows stay reserved.
            if not (set(metric) | set(audit)) <= source:
                raise CohortIsolationError("frozen evaluation partitions exceed their source")
            if set(train) & source:
                raise CohortIsolationError("selected training rows overlap reserved evaluation source")
        elif set(metric) | set(audit) != source:
            raise CohortIsolationError("frozen evaluation partitions do not cover their source")
        return CohortPlan("audit_" + digest[:24], sha256_json(payload["contract"]), int(payload["seed"]),
                          train, metric, audit, cohort_label_hash(payload["metric_source_ids"], payload["metric_source_labels"]))
    required = {"training_ids", "metric_source_ids", "metric_source_labels", "contract"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"cohort file is missing: {sorted(missing)}")
    plan = build_cohort_plan(
        training_ids=payload["training_ids"],
        metric_source_ids=payload["metric_source_ids"],
        metric_source_labels=payload["metric_source_labels"],
        contract=payload["contract"],
        seed=seed,
        audit_samples=audit_samples,
        minimum_audit_samples=minimum,
        task_family=task_family,
    )
    labels_by_id = dict(
        zip(map(str, payload["metric_source_ids"]), payload["metric_source_labels"])
    )
    audit_labels = [labels_by_id[item] for item in plan.audit_ids]
    supplied = payload.get("audit_labels")
    if supplied and list(supplied) != audit_labels:
        raise ValueError("cohort file audit_labels do not match the immutable cohort plan")
    return plan


def read_cohort_labels(
    path: str | Path, plan: CohortPlan, *, task_family: str
) -> np.ndarray:
    payload = json.loads(resolve_artifact_path(path).read_text(encoding="utf-8"))
    labels = dict(
        zip(map(str, payload["metric_source_ids"]), payload["metric_source_labels"])
    )
    if payload.get("version") == "reserved_evaluation_v2":
        return [labels[item] for item in plan.audit_ids]
    dtype = float if str(task_family).lower() == "regression" else int
    return np.asarray([labels[item] for item in plan.audit_ids], dtype=dtype)


