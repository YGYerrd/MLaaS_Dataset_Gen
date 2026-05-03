from __future__ import annotations

import json
import os
import sqlite3
from importlib import resources
from typing import Any, Mapping

import numpy as np

ALLOWED_METRIC_DOMAINS = {
    "quality",
    "qos",
    "performance",
    "latency",
    "runtime",
    "resource",
    "cost",
    "reliability",
    "explainability",
    "metadata",
}
SQLITE_INT_MIN = -(2**63)
SQLITE_INT_MAX = 2**63 - 1


def make_writer(kind: str, **kwargs):
    if kind == "sqlite":
        return SQLiteWriter(**kwargs)
    raise ValueError(f"Unknown writer kind: {kind}")


class SQLiteWriter:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.conn: sqlite3.Connection | None = None

    def start(self) -> None:
        folder = os.path.dirname(self.db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        sql = resources.files(__package__).joinpath("schemaV2.sql").read_text(encoding="utf-8")
        self.conn.executescript(sql)
        # Each GPU worker writes to its own DB file, so WAL adds portability risk
        # without providing useful concurrency benefits here.
        self.conn.execute("PRAGMA journal_mode = DELETE;")
        self.conn.execute("PRAGMA synchronous = FULL;")
        self.conn.commit()

    def finish(self) -> None:
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def abort(self) -> None:
        if self.conn is not None:
            self.conn.rollback()
            self.conn.close()
            self.conn = None

    def _ins(self, table: str, row: Mapping[str, Any]) -> None:
        if self.conn is None:
            raise RuntimeError("SQLiteWriter.start() must be called before writing")
        keys = list(row.keys())
        placeholders = ",".join(["?"] * len(keys))
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(keys)}) VALUES ({placeholders})"
        self.conn.execute(sql, [row[k] for k in keys])

    def write_service(self, row: Mapping[str, Any]) -> None:
        service_row = dict(row)
        for key in ("service_config_json", "registry_metadata_json", "functional_attributes_json", "metadata_json"):
            if key in service_row and service_row[key] is not None and not isinstance(service_row[key], str):
                service_row[key] = json.dumps(service_row[key], ensure_ascii=False, default=str)
        self._ins("services", service_row)

    def write_service_metrics(self, service_id: str, values: Mapping[str, Any]) -> None:
        for metric_name, spec in (values or {}).items():
            if spec is None:
                continue
            self.write_service_metric(service_id, metric_name, spec)

    def write_service_metric(self, service_id: str, metric_name: str, spec: Any) -> None:
        if isinstance(spec, Mapping) and "value" in spec:
            value = spec.get("value")
            domain = str(spec.get("domain") or "metadata").strip().lower()
            unit = spec.get("unit")
            direction = str(spec.get("direction") or "neutral").strip().lower()
        else:
            value = spec
            domain = _infer_metric_domain(metric_name)
            unit = None
            direction = _infer_metric_direction(metric_name)

        if value is None:
            return
        if domain not in ALLOWED_METRIC_DOMAINS:
            domain = "metadata"
        if direction not in {"higher_better", "lower_better", "neutral"}:
            direction = "neutral"

        row = {
            "service_id": service_id,
            "metric_name": str(metric_name).strip().lower(),
            "domain": domain,
            "unit": unit,
            "direction": direction,
        }
        row.update(self._coerce_value_columns(value))
        self._ins("service_metrics", row)

    def write_service_artifact(
        self,
        service_id: str,
        *,
        artifact_type: str,
        artifact_uri: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._ins(
            "service_artifacts",
            {
                "service_id": service_id,
                "artifact_type": artifact_type,
                "artifact_uri": artifact_uri,
                "metadata_json": json.dumps(metadata, ensure_ascii=False, default=str) if metadata else None,
            },
        )

    def write_service_split_provenance(
        self,
        service_id: str,
        *,
        split_name: str,
        samples_count: int | None = None,
        data_distribution: Mapping[str, Any] | None = None,
        split_config: Mapping[str, Any] | None = None,
    ) -> None:
        self._ins(
            "service_split_provenance",
            {
                "service_id": service_id,
                "split_name": str(split_name),
                "samples_count": samples_count,
                "data_distribution_json": json.dumps(data_distribution, ensure_ascii=False, default=str)
                if data_distribution is not None
                else None,
                "split_config_json": json.dumps(split_config, ensure_ascii=False, default=str)
                if split_config is not None
                else None,
            },
        )

    def write_service_failure(
        self,
        *,
        service_id: str | None,
        row_index: int | None,
        case_name: str | None,
        manifest_group_id: str | None,
        failure_stage: str,
        error_message: str | None,
        resolved_config_json: str | None,
        traceback_text: str | None = None,
    ) -> None:
        self._ins(
            "service_failures",
            {
                "service_id": service_id,
                "row_index": row_index,
                "case_name": case_name,
                "manifest_group_id": manifest_group_id,
                "failure_stage": failure_stage,
                "error_message": error_message,
                "resolved_config_json": resolved_config_json,
                "traceback_text": traceback_text,
            },
        )

    def _coerce_value_columns(self, value: Any) -> dict[str, Any]:
        out = {
            "value_num": None,
            "value_int": None,
            "value_bool": None,
            "value_text": None,
            "value_json": None,
        }

        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)

        if isinstance(value, bool):
            out["value_bool"] = 1 if value else 0
            return out

        if isinstance(value, int):
            if SQLITE_INT_MIN <= int(value) <= SQLITE_INT_MAX:
                out["value_int"] = value
            else:
                try:
                    as_float = float(value)
                except OverflowError:
                    out["value_text"] = str(value)
                else:
                    out["value_num"] = as_float if np.isfinite(as_float) else None
                    if out["value_num"] is None:
                        out["value_text"] = str(value)
            return out

        if isinstance(value, float):
            if np.isnan(value):
                out["value_json"] = json.dumps(None)
            else:
                out["value_num"] = value
            return out

        if isinstance(value, str):
            out["value_text"] = value
            return out

        try:
            out["value_json"] = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            out["value_text"] = str(value)
        return out


def _infer_metric_domain(metric_name: str) -> str:
    name = str(metric_name or "").lower()
    if any(key in name for key in ("accuracy", "f1", "loss", "rmse", "mae", "perplexity", "rouge", "bleu", "cider", "map", "iou", "dice", "silhouette", "metric_score")):
        return "quality"
    if "latency" in name:
        return "latency"
    if any(key in name for key in ("runtime", "duration", "time_s", "seconds", "throughput")):
        return "runtime"
    if any(key in name for key in ("memory", "vram", "ram", "cpu", "gpu", "tokens", "params", "model_size")):
        return "resource"
    if "cost" in name or "efficiency" in name:
        return "cost"
    if any(key in name for key in ("reliability", "trust", "failure", "error", "retry", "oom", "nan")):
        return "reliability"
    if "explain" in name:
        return "explainability"
    return "metadata"


def _infer_metric_direction(metric_name: str) -> str:
    name = str(metric_name or "").lower()
    if any(key in name for key in ("loss", "rmse", "mae", "latency", "runtime", "duration", "memory", "cost", "error", "retry", "oom", "nan")):
        return "lower_better"
    if any(key in name for key in ("accuracy", "f1", "score", "throughput", "recall", "precision", "map", "iou", "dice", "silhouette", "trust", "reliability")):
        return "higher_better"
    return "neutral"
