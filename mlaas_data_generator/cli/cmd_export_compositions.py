from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import CONFIG
from ..federated.update_signature import compute_composition_mus


SERVICE_COLUMNS = [
    "service_id",
    "task_family",
    "model_type",
    "modality",
    "metric_score",
    "latency",
    "data_volume",
    "resource_cost_score",
    "data_distribution",
    "update_signature_id",
    "update_signature_path",
    "signature_dim",
    "signature_norm",
    "compute_time_s",
    "batch_size",
    "reliability_score",
    "trust_score",
    "explainability_score",
]


def load_services_from_db(db_path: str | Path) -> pd.DataFrame:
    query = """
    WITH m AS (
        SELECT
            service_id,
            metric_name,
            COALESCE(value_num, CAST(value_int AS REAL), CAST(value_bool AS REAL)) AS num_value,
            COALESCE(value_text, CAST(value_num AS TEXT), CAST(value_int AS TEXT), CAST(value_bool AS TEXT), value_json) AS text_value
        FROM service_metrics
    ),
    p AS (
        SELECT
            service_id,
            MAX(CASE WHEN metric_name = 'metric_score' THEN num_value END) AS metric_score,
            MAX(CASE WHEN metric_name IN ('latency', 'inference_latency_s', 'inference_latency_s_mean') THEN num_value END) AS latency,
            MAX(CASE WHEN metric_name IN ('compute_time_s', 'runtime_s', 'service_runtime_s') THEN num_value END) AS compute_time_s,
            MAX(CASE WHEN metric_name IN ('dataset_size', 'train_set_size') THEN num_value END) AS data_volume,
            MAX(CASE WHEN metric_name = 'resource_cost_score' THEN num_value END) AS resource_cost_score,
            MAX(CASE WHEN metric_name IN ('data_distribution', 'split_strategy') THEN text_value END) AS data_distribution,
            MAX(CASE WHEN metric_name = 'update_signature_id' THEN text_value END) AS update_signature_id,
            MAX(CASE WHEN metric_name = 'update_signature_path' THEN text_value END) AS update_signature_path,
            MAX(CASE WHEN metric_name = 'signature_dim' THEN num_value END) AS signature_dim,
            MAX(CASE WHEN metric_name = 'signature_norm' THEN num_value END) AS signature_norm,
            MAX(CASE WHEN metric_name = 'batch_size' THEN num_value END) AS batch_size,
            MAX(CASE WHEN metric_name = 'reliability_score' THEN num_value END) AS reliability_score,
            MAX(CASE WHEN metric_name = 'trust_score' THEN num_value END) AS trust_score,
            MAX(CASE WHEN metric_name = 'explainability_score' THEN num_value END) AS explainability_score
        FROM m
        GROUP BY service_id
    ),
    a AS (
        SELECT
            service_id,
            MAX(CASE WHEN artifact_type = 'update_signature' THEN artifact_uri END) AS artifact_update_signature_path
        FROM service_artifacts
        GROUP BY service_id
    )
    SELECT
        s.service_id,
        COALESCE(s.task_family, s.task_type, 'unknown') AS task_family,
        COALESCE(s.model_type, s.model_id, 'unknown') AS model_type,
        COALESCE(s.modality, 'unknown') AS modality,
        p.metric_score,
        p.latency,
        p.data_volume,
        p.resource_cost_score,
        p.data_distribution,
        p.update_signature_id,
        COALESCE(p.update_signature_path, a.artifact_update_signature_path) AS update_signature_path,
        p.signature_dim,
        p.signature_norm,
        p.compute_time_s,
        p.batch_size,
        p.reliability_score,
        p.trust_score,
        p.explainability_score
    FROM services s
    LEFT JOIN p ON p.service_id = s.service_id
    LEFT JOIN a ON a.service_id = s.service_id
    WHERE s.status = 'completed'
    """
    with sqlite3.connect(str(db_path)) as conn:
        rows = pd.read_sql_query(query, conn)
    return normalise_service_rows(rows)


def normalise_service_rows(rows: pd.DataFrame) -> pd.DataFrame:
    services = rows.copy()
    for column in SERVICE_COLUMNS:
        if column not in services.columns:
            services[column] = None

    services["service_id"] = services["service_id"].astype(str)
    services["task_family"] = services["task_family"].fillna("unknown").astype(str).str.lower()
    services["model_type"] = services["model_type"].fillna("unknown").astype(str).str.lower()
    services["modality"] = services["modality"].fillna("unknown").astype(str).str.lower()
    services["data_distribution"] = services["data_distribution"].fillna("unknown").astype(str).str.lower()

    services["metric_score"] = services["metric_score"].map(lambda value: _bounded(value, 0.0))
    services["latency"] = services["latency"].map(lambda value: max(0.0, _number(value, 0.0)))
    services["compute_time_s"] = services["compute_time_s"].map(lambda value: max(0.0, _number(value, 0.0)))
    services["data_volume"] = services["data_volume"].map(lambda value: max(0.0, _number(value, 0.0)))
    services["resource_cost_score"] = services["resource_cost_score"].map(lambda value: _bounded(value, 0.5))
    services["batch_size"] = services["batch_size"].map(lambda value: max(1, int(_number(value, 1.0))))
    services["reliability_score"] = services["reliability_score"].map(lambda value: _bounded(value, 1.0))
    services["trust_score"] = services["trust_score"].map(lambda value: _bounded(value, 0.5))
    services["explainability_score"] = services["explainability_score"].map(lambda value: _bounded(value, 0.5))
    services["signature_dim"] = services["signature_dim"].map(lambda value: int(_number(value, 0.0)) if _finite(value) else None)
    services["signature_norm"] = services["signature_norm"].map(lambda value: _number(value, np.nan))
    return services[SERVICE_COLUMNS]


def generate_service_requests(services: pd.DataFrame, *, request_count: int = 25, seed: int = 42) -> pd.DataFrame:
    services = normalise_service_rows(services)
    rng = np.random.default_rng(seed)
    task_families = sorted(value for value in services["task_family"].dropna().unique())
    if not task_families:
        task_families = ["unknown"]

    requests = []
    for idx in range(int(request_count)):
        task = str(rng.choice(task_families))
        pool = services[services["task_family"] == task]
        if pool.empty:
            pool = services
        workflow_length = int(min(max(1, len(pool)), rng.integers(2, min(4, len(pool)) + 1) if len(pool) > 1 else 1))
        requests.append(
            {
                "request_id": f"req_{idx:06d}",
                "task_family": task,
                "workflow_length": workflow_length,
                "min_quality": float(rng.uniform(0.4, 0.8)),
                "max_latency": float(max(0.001, np.percentile(pool["latency"], 75) * rng.uniform(1.0, 2.0))),
                "max_resource_cost": float(rng.uniform(0.5, 1.0)),
            }
        )
    return pd.DataFrame(requests)


def generate_compositions(
    services: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    candidates_per_request: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    services = normalise_service_rows(services)
    rng = np.random.default_rng(seed)
    rows = []

    for _, request_row in requests.iterrows():
        request = request_row.to_dict()
        task = str(request.get("task_family") or "").lower()
        pool = services[services["task_family"] == task]
        if pool.empty:
            pool = services

        workflow_length = max(1, min(int(request.get("workflow_length") or 1), len(pool)))
        combinations = list(itertools.combinations(range(len(pool)), workflow_length))
        rng.shuffle(combinations)
        combinations = combinations[: int(candidates_per_request)]

        for candidate_idx, combo in enumerate(combinations):
            selected = pool.iloc[list(combo)].reset_index(drop=True)
            scores = _composition_scores(selected)
            score = _weighted_score(scores)
            adjusted = _apply_request_penalties(score, selected, request)
            service_ids = [str(value) for value in selected["service_id"].tolist()]
            rows.append(
                {
                    "request_id": str(request.get("request_id")),
                    "candidate_id": f"{request.get('request_id')}_cand_{candidate_idx:03d}",
                    "service_ids": json.dumps(service_ids),
                    "workflow_length": len(service_ids),
                    **scores,
                    "composability_score": score,
                    "penalty_adjusted_score": adjusted,
                    "selected_flag": False,
                }
            )

    compositions = pd.DataFrame(rows)
    if compositions.empty:
        return compositions
    for request_id, group in compositions.groupby("request_id"):
        compositions.loc[group["penalty_adjusted_score"].idxmax(), "selected_flag"] = True
    for column in ("dhs", "mus", "shs", "ses", "hsq", "srs", "composability_score", "penalty_adjusted_score"):
        compositions[column] = compositions[column].astype(float).clip(0.0, 1.0)
    return compositions


def _composition_scores(selected: pd.DataFrame) -> dict[str, float]:
    records = selected.to_dict(orient="records")
    return {
        "dhs": _dominant_fraction(selected["data_distribution"]),
        "mus": compute_composition_mus(records),
        "shs": float(np.mean([
            _dominant_fraction(selected["task_family"]),
            _dominant_fraction(selected["modality"]),
            _dominant_fraction(selected["model_type"]),
        ])),
        "ses": _service_efficiency_score(selected),
        "hsq": float(np.mean(selected["metric_score"].astype(float))),
        "srs": float(np.mean([
            selected["reliability_score"].astype(float).mean(),
            selected["trust_score"].astype(float).mean(),
            selected["explainability_score"].astype(float).mean(),
        ])),
    }


def _weighted_score(scores: Mapping[str, float]) -> float:
    weights = {"dhs": 0.15, "mus": 0.25, "shs": 0.15, "ses": 0.15, "hsq": 0.20, "srs": 0.10}
    total = sum(weights.values())
    value = sum(float(scores[key]) * weight for key, weight in weights.items()) / total
    return float(np.clip(value, 0.0, 1.0))


def _apply_request_penalties(score: float, selected: pd.DataFrame, request: Mapping[str, Any]) -> float:
    penalty = 0.0
    quality = float(selected["metric_score"].astype(float).mean())
    latency = float(selected["latency"].astype(float).mean())
    cost = float(selected["resource_cost_score"].astype(float).mean())
    min_quality = _number(request.get("min_quality"), np.nan)
    max_latency = _number(request.get("max_latency"), np.nan)
    max_cost = _number(request.get("max_resource_cost"), np.nan)
    if _finite(min_quality) and quality < min_quality:
        penalty += min_quality - quality
    if _finite(max_latency) and latency > max_latency:
        penalty += min(0.5, (latency - max_latency) / max(max_latency, 1e-9))
    if _finite(max_cost) and cost > max_cost:
        penalty += cost - max_cost
    return float(np.clip(score - penalty, 0.0, 1.0))


def _service_efficiency_score(selected: pd.DataFrame) -> float:
    latency_score = 1.0 / (1.0 + float(selected["latency"].astype(float).mean()))
    compute_score = 1.0 / (1.0 + float(selected["compute_time_s"].astype(float).mean()))
    cost_score = 1.0 - float(selected["resource_cost_score"].astype(float).mean())
    return float(np.clip(np.mean([latency_score, compute_score, cost_score]), 0.0, 1.0))


def _dominant_fraction(values) -> float:
    items = [str(value) for value in values if value is not None]
    if not items:
        return 0.5
    return float(max(items.count(item) for item in set(items)) / len(items))


def _number(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "not available", "n/a"}:
            return float(default)
        parsed = float(value)
        return parsed if np.isfinite(parsed) else float(default)
    except Exception:
        return float(default)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _bounded(value: Any, default: float) -> float:
    return float(np.clip(_number(value, default), 0.0, 1.0))


def _handle(args: argparse.Namespace) -> None:
    services = load_services_from_db(args.db)
    requests = generate_service_requests(services, request_count=args.requests, seed=args.seed)
    compositions = generate_compositions(
        services,
        requests,
        candidates_per_request=args.candidates_per_request,
        seed=args.seed,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    services.to_csv(out_dir / "services.csv", index=False)
    requests.to_csv(out_dir / "requests.csv", index=False)
    compositions.to_csv(out_dir / "compositions.csv", index=False)
    print(f"Wrote composition export files to {out_dir}")


def register_export_compositions(subparsers) -> None:
    parser = subparsers.add_parser("export-compositions", help="Generate request and composition datasets from service DB")
    parser.add_argument("--db", default=str(CONFIG["db_path"]), help="SQLite service database")
    parser.add_argument("--output-dir", default="outputs/compositions", help="Output directory")
    parser.add_argument("--requests", type=int, default=25, help="Number of generated service requests")
    parser.add_argument("--candidates-per-request", type=int, default=10, help="Candidate compositions per request")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(_handler=_handle)
