from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from importlib import resources
from typing import Any, Mapping

import numpy as np

from ..diagnostic_json import diagnostic_json, bounded_text

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
        sql = resources.files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")
        self.conn.executescript(sql)
        self._migrate_schema()
        self.conn.commit()
        # Each GPU worker writes to its own DB file, so WAL adds portability risk
        # without providing useful concurrency benefits here.
        self.conn.execute("PRAGMA journal_mode = DELETE;")
        self.conn.execute("PRAGMA synchronous = FULL;")
        self.conn.commit()

    def _migrate_schema(self) -> None:
        """Apply additive migrations for databases created before provenance.

        New checkpoint-storage tables are created by ``schema.sql``.  Historical
        services deliberately receive no synthetic row in those tables: absence
        is the provenance-incomplete marker.
        """
        if self.conn is None:
            return
        self._migrate_fedavg_status_contract()
        fedavg_run_columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(fedavg_runs)")
        }
        if "heterogeneity_intensity" not in fedavg_run_columns:
            self.conn.execute(
                "ALTER TABLE fedavg_runs ADD COLUMN heterogeneity_intensity REAL"
            )
        if "partition_generator_version" not in fedavg_run_columns:
            self.conn.execute(
                "ALTER TABLE fedavg_runs ADD COLUMN partition_generator_version "
                "TEXT NOT NULL DEFAULT 'sequential_depletion_v1'"
            )
        self.conn.execute(
            "UPDATE fedavg_runs SET partition_generator_version="
            "'sequential_depletion_v1' WHERE partition_generator_version IS NULL "
            "OR trim(partition_generator_version)=''"
        )
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(service_artifacts)")}
        for name, sql_type in (
            ("artifact_sha256", "TEXT"),
            ("size_bytes", "INTEGER"),
            ("artifact_format", "TEXT"),
            ("artifact_role", "TEXT"),
        ):
            if name not in columns:
                self.conn.execute(f"ALTER TABLE service_artifacts ADD COLUMN {name} {sql_type}")

        provenance_columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(service_generation_provenance)")
        }
        if "logical_dataset_contract_hash" not in provenance_columns:
            self.conn.execute(
                "ALTER TABLE service_generation_provenance "
                "ADD COLUMN logical_dataset_contract_hash TEXT"
            )
        for name, sql_type in (
            ("generation_mode", "TEXT"),
            ("family_id", "TEXT"),
            ("family_contract_version", "TEXT"),
            ("initial_state_hash", "TEXT"),
            ("family_document_ref", "TEXT"),
            ("family_document_hash", "TEXT"),
            ("client_recipe_ref", "TEXT"),
            ("client_recipe_hash", "TEXT"),
            ("composition_compatible", "INTEGER"),
            ("parent_checkpoint_id", "TEXT"),
            ("cohort_id", "TEXT"),
            ("compatibility_contract_hash", "TEXT"),
            ("parent_global_round", "INTEGER"),
            ("parent_state_hash", "TEXT"),
            ("training_universe_hash", "TEXT"),
            ("benchmark_cohort_hash", "TEXT"),
            ("scheduled_to_participate", "INTEGER"),
            ("participated", "INTEGER"),
        ):
            if name not in provenance_columns:
                self.conn.execute(
                    f"ALTER TABLE service_generation_provenance ADD COLUMN {name} {sql_type}"
                )

        service_columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(services)")
        }
        if "family_id" not in service_columns:
            self.conn.execute("ALTER TABLE services ADD COLUMN family_id TEXT")
            self.conn.execute(
                "UPDATE services SET family_id=("
                "SELECT family_id FROM service_generation_provenance "
                "WHERE service_generation_provenance.service_id=services.service_id"
                ") WHERE family_id IS NULL"
            )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_services_family_service "
            "ON services(family_id,service_id,status)"
        )

        checkpoint_columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(service_checkpoint_storage)")
        }
        for name, sql_type in (
            ("parent_global_round", "INTEGER"),
            ("explicit_base_state_ref", "TEXT"),
        ):
            if name not in checkpoint_columns:
                self.conn.execute(
                    f"ALTER TABLE service_checkpoint_storage ADD COLUMN {name} {sql_type}"
                )

        failure_columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(service_failures)")
        }
        for name, sql_type in (
            ("fl_run_id", "TEXT"),
            ("client_id", "TEXT"),
            ("round_id", "INTEGER"),
        ):
            if name not in failure_columns:
                self.conn.execute(f"ALTER TABLE service_failures ADD COLUMN {name} {sql_type}")

        validation_columns = {
            row[1]
            for row in self.conn.execute(
                "PRAGMA table_info(composition_validation_candidates)"
            )
        }
        for name, sql_type in (
            ("selection_source", "TEXT"),
            ("cohort_id", "TEXT"),
            ("parent_checkpoint_id", "TEXT"),
            ("local_example_exposures", "INTEGER"),
            ("unique_training_examples", "INTEGER"),
            ("search_method", "TEXT"),
            ("recorded_surrogate_quality", "REAL"),
            ("measured_quality_score", "REAL"),
            ("utility_prediction_error", "REAL"),
            ("quality_prediction_error", "REAL"),
            ("ensemble_f1_macro", "REAL"),
            ("ensemble_rmse", "REAL"),
            ("ensemble_mae", "REAL"),
            ("actual_primary_metric_name", "TEXT"),
            ("actual_secondary_metric_name", "TEXT"),
            ("prediction_wall_s", "REAL"),
            ("aggregation_method", "TEXT"),
            ("aggregation_version", "TEXT"),
            ("fl_run_id", "TEXT"),
            ("round_id", "INTEGER"),
            ("parent_global_round", "INTEGER"),
            ("parent_state_hash", "TEXT"),
            ("compatibility_contract_hash", "TEXT"),
            ("parameter_schema_hash", "TEXT"),
            ("aggregated_state_hash", "TEXT"),
            ("selected_total_samples", "INTEGER"),
            ("contributor_provenance_json", "TEXT"),
            ("aggregation_provenance_json", "TEXT"),
            ("aggregation_wall_s", "REAL"),
            ("inference_latency_s", "REAL"),
            ("actual_accuracy", "REAL"),
            ("actual_f1_macro", "REAL"),
            ("actual_rmse", "REAL"),
            ("actual_mae", "REAL"),
        ):
            if name not in validation_columns:
                self.conn.execute(
                    f"ALTER TABLE composition_validation_candidates ADD COLUMN {name} {sql_type}"
                )
        self._migrate_composition_validation_status_contract()

    def _migrate_fedavg_status_contract(self) -> None:
        """Allow explicit interruption recovery in databases made by older builds."""
        if self.conn is None:
            return
        definitions = {
            "fedavg_runs": """
                CREATE TABLE fedavg_runs_status_new (
                  fl_run_id TEXT PRIMARY KEY, lineage_id TEXT NOT NULL,
                  status TEXT NOT NULL, client_count INTEGER NOT NULL,
                  round_count INTEGER NOT NULL, participation_rate REAL NOT NULL,
                  schedule_seed INTEGER NOT NULL, schedule_version TEXT NOT NULL,
                  heterogeneity_intensity REAL,
                  partition_generator_version TEXT NOT NULL DEFAULT 'sequential_depletion_v1',
                  compatibility_contract_hash TEXT, compatibility_contract_json TEXT,
                  g0_state_hash TEXT, partition_plan_hash TEXT, partition_plan_ref TEXT,
                  training_universe_hash TEXT, benchmark_cohort_hash TEXT,
                  benchmark_cohort_ref TEXT, status_reason TEXT,
                  started_at TEXT NOT NULL, completed_at TEXT, metadata_json TEXT,
                  CHECK (client_count >= 2), CHECK (round_count >= 2),
                  CHECK (participation_rate > 0 AND participation_rate <= 1),
                  CHECK (status IN ('running','complete','incomplete','failed','interrupted'))
                )
            """,
            "fedavg_rounds": """
                CREATE TABLE fedavg_rounds_status_new (
                  fl_run_id TEXT NOT NULL, round_id INTEGER NOT NULL,
                  status TEXT NOT NULL, scheduled_count INTEGER NOT NULL,
                  successful_count INTEGER NOT NULL, failed_count INTEGER NOT NULL,
                  aggregation_method TEXT, aggregation_version TEXT,
                  contributor_metadata_json TEXT, excluded_metadata_json TEXT,
                  parent_state_hash TEXT, output_state_hash TEXT,
                  started_at TEXT NOT NULL, completed_at TEXT, status_reason TEXT,
                  metadata_json TEXT, PRIMARY KEY (fl_run_id, round_id),
                  FOREIGN KEY (fl_run_id) REFERENCES fedavg_runs(fl_run_id) ON DELETE CASCADE,
                  CHECK (round_id >= 1),
                  CHECK (status IN ('running','complete','incomplete','failed','interrupted'))
                )
            """,
        }
        needs_migration = []
        for table in definitions:
            row = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if row is not None and "interrupted" not in str(row[0]).lower():
                needs_migration.append(table)
        if not needs_migration:
            return

        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys = OFF")
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            for table in needs_migration:
                replacement = f"{table}_status_new"
                self.conn.execute(f"DROP TABLE IF EXISTS {replacement}")
                self.conn.execute(definitions[table])
                columns = [
                    row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")
                ]
                column_list = ",".join(columns)
                self.conn.execute(
                    f"INSERT INTO {replacement} ({column_list}) "
                    f"SELECT {column_list} FROM {table}"
                )
                self.conn.execute(f"DROP TABLE {table}")
                self.conn.execute(f"ALTER TABLE {replacement} RENAME TO {table}")
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self.conn.execute("PRAGMA foreign_keys = ON")
        violations = self.conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(
                "FedAvg status migration introduced foreign-key violations: "
                f"{violations[:3]}"
            )

    def _migrate_composition_validation_status_contract(self) -> None:
        """Expand validation statuses while retaining every existing result row."""
        if self.conn is None:
            return
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'composition_validation_candidates'"
        ).fetchone()
        if row is None or "checkpoint_failed" in str(row[0]):
            return
        legacy = "composition_validation_candidates_status_legacy"
        self.conn.execute(
            f"ALTER TABLE composition_validation_candidates RENAME TO {legacy}"
        )
        self.conn.execute(
            """
            CREATE TABLE composition_validation_candidates (
              validation_run_id             TEXT NOT NULL,
              candidate_id                  TEXT NOT NULL,
              service_ids_json              TEXT NOT NULL,
              validation_status             TEXT NOT NULL,
              status_reason                 TEXT,
              selection_source              TEXT,
              search_method                 TEXT,
              recorded_surrogate_utility    REAL,
              recorded_surrogate_quality    REAL,
              regenerated_surrogate_utility REAL,
              measured_utility              REAL,
              measured_quality_score        REAL,
              utility_prediction_error      REAL,
              quality_prediction_error      REAL,
              ensemble_accuracy             REAL,
              ensemble_f1_macro             REAL,
              ensemble_rmse                 REAL,
              ensemble_mae                  REAL,
              actual_primary_metric_name    TEXT,
              actual_secondary_metric_name  TEXT,
              parallel_latency_s            REAL,
              serial_latency_s              REAL,
              prediction_wall_s             REAL,
              aggregation_method            TEXT,
              aggregation_version           TEXT,
              fl_run_id                     TEXT,
              round_id                      INTEGER,
              parent_global_round           INTEGER,
              parent_state_hash             TEXT,
              compatibility_contract_hash  TEXT,
              parameter_schema_hash         TEXT,
              aggregated_state_hash         TEXT,
              selected_total_samples        INTEGER,
              contributor_provenance_json   TEXT,
              aggregation_provenance_json   TEXT,
              aggregation_wall_s            REAL,
              inference_latency_s           REAL,
              actual_accuracy               REAL,
              actual_f1_macro               REAL,
              actual_rmse                   REAL,
              actual_mae                    REAL,
              qos_violations_json           TEXT,
              selected_flag                 INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY (validation_run_id, candidate_id),
              FOREIGN KEY (validation_run_id)
                REFERENCES composition_validation_runs(validation_run_id)
                ON DELETE CASCADE,
              CHECK (validation_status IN (
                'completed','unsupported','regeneration_failed',
                'cohort_isolation_failed','checkpoint_failed',
                'aggregation_failed','inference_failed'
              )),
              CHECK (selected_flag IN (0,1))
            )
            """
        )
        old_columns = {
            item[1] for item in self.conn.execute(f"PRAGMA table_info({legacy})")
        }
        new_columns = [
            item[1]
            for item in self.conn.execute(
                "PRAGMA table_info(composition_validation_candidates)"
            )
        ]
        copied = [name for name in new_columns if name in old_columns]
        names = ",".join(copied)
        self.conn.execute(
            f"INSERT INTO composition_validation_candidates ({names}) "
            f"SELECT {names} FROM {legacy}"
        )
        self.conn.execute(f"DROP TABLE {legacy}")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_composition_validation_candidates_status "
            "ON composition_validation_candidates(validation_status)"
        )

    def commit(self) -> None:
        """Make primary results durable before recording completion timings."""
        if self.conn is not None:
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
        if row.get("status") in {"queued", "running", "interrupted"}:
            raise ValueError("Unfinished attempts must not be written to the service catalogue")
        service_row = dict(row)
        for key in ("service_config_json", "registry_metadata_json", "functional_attributes_json", "metadata_json"):
            if key in service_row and service_row[key] is not None:
                service_row[key] = diagnostic_json(service_row[key])
        self._ins("services", service_row)

    def write_service_progress(self, row: Mapping[str, Any]) -> None:
        """Update a finished service outcome without replacing child rows."""
        if row.get("status") in {"queued", "running", "interrupted"}:
            raise ValueError("Unfinished attempts must not be written to the service catalogue")
        if self.conn is None:
            raise RuntimeError("SQLiteWriter.start() must be called before writing")
        service_row = dict(row)
        for key in (
            "service_config_json",
            "registry_metadata_json",
            "functional_attributes_json",
            "metadata_json",
        ):
            if key in service_row and service_row[key] is not None:
                service_row[key] = diagnostic_json(service_row[key])
        keys = list(service_row)
        placeholders = ",".join(["?"] * len(keys))
        self.conn.execute(
            f"INSERT INTO services ({','.join(keys)}) VALUES ({placeholders}) "
            "ON CONFLICT(service_id) DO UPDATE SET "
            "status=excluded.status, metadata_json=excluded.metadata_json, "
            "updated_at=datetime('now')",
            [service_row[key] for key in keys],
        )

    def write_service_metrics(self, service_id: str, values: Mapping[str, Any]) -> None:
        for metric_name, spec in (values or {}).items():
            if spec is None and metric_name not in {"trust_score", "explainability_score"}:
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

        if value is None and metric_name not in {"trust_score", "explainability_score"}:
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
        details = dict(metadata or {})
        self._ins(
            "service_artifacts",
            {
                "service_id": service_id,
                "artifact_type": artifact_type,
                "artifact_uri": artifact_uri,
                "metadata_json": json.dumps(details, ensure_ascii=False, default=str) if details else None,
                "artifact_sha256": details.get("sha256"),
                "size_bytes": details.get("size_bytes"),
                "artifact_format": details.get("format"),
                "artifact_role": details.get("role"),
            },
        )

    def write_generation_record(self, table: str, row: Mapping[str, Any]) -> None:
        """Upsert coordination status while protecting frozen generation identity."""
        contracts = {
            "parent_checkpoints": ("parent_checkpoint_id", ("family_id", "selection_hash")),
            "service_cohorts": ("cohort_id", ("parent_checkpoint_id", "manifest_hash", "scheduled_clients")),
            "cohort_clients": ("service_id", ("cohort_id", "client_id", "recipe_hash", "partition_hash")),
        }
        if table not in contracts or self.conn is None:
            raise ValueError("invalid generation table or writer not started")
        key, immutable = contracts[table]
        value = dict(row)
        if not isinstance(value.get("document_json"), str):
            value["document_json"] = json.dumps(value.get("document_json", {}), sort_keys=True, allow_nan=False)
        cursor = self.conn.execute(f"SELECT * FROM {table} WHERE {key}=?", (value[key],))
        existing = cursor.fetchone()
        if existing:
            old = dict(zip([d[0] for d in cursor.description], existing))
            for field in immutable:
                if old.get(field) is not None and field in value and old[field] != value[field]:
                    raise ValueError(f"immutable {table}.{field} changed for {value[key]}")
            if table == "cohort_clients" and old.get("final_state_hash"):
                # Published weights and evidence remain immutable. Availability
                # may change when verification discovers/restores damaged files.
                if value.get("status") not in {"completed", "failed"} or any(
                    old.get(field) != value.get(field, old.get(field))
                    for field in ("final_state_hash", "document_json", "sample_count")
                ):
                    raise ValueError(f"published client checkpoint is immutable: {value[key]}")
            elif old["status"] == "completed" and table == "parent_checkpoints":
                if any(old.get(field) != value.get(field, old.get(field))
                       for field in ("parent_state_hash", "final_state_hash", "status", "document_json")):
                    raise ValueError(f"completed {table} record is immutable: {value[key]}")
                return
        keys = list(value)
        updates = ",".join(f"{field}=excluded.{field}" for field in keys if field != key)
        self.conn.execute(
            f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)}) "
            f"ON CONFLICT({key}) DO UPDATE SET {updates}, updated_at=datetime('now')",
            [value[field] for field in keys],
        )

    def write_model_lineage(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        if value.get("provenance_json") is not None and not isinstance(value["provenance_json"], str):
            value["provenance_json"] = json.dumps(value["provenance_json"], ensure_ascii=False, default=str)
        if self.conn is None:
            raise RuntimeError("SQLiteWriter.start() must be called before writing")
        keys = list(value.keys())
        placeholders = ",".join(["?"] * len(keys))
        self.conn.execute(
            f"INSERT OR IGNORE INTO model_lineages ({','.join(keys)}) VALUES ({placeholders})",
            [value[key] for key in keys],
        )

    def write_lineage_checkpoint_storage(self, row: Mapping[str, Any]) -> None:
        """Persist the immutable, shared base-state storage record for a lineage.

        A lineage may be encountered once per client service.  Repeating the
        same record is idempotent, while changing any base identity field is an
        integrity error rather than an update.
        """
        if self.conn is None:
            raise RuntimeError("SQLiteWriter.start() must be called before writing")
        value = dict(row)
        for key in ("base_compression_settings_json", "metadata_json"):
            if value.get(key) is not None and not isinstance(value[key], str):
                value[key] = json.dumps(value[key], ensure_ascii=False, sort_keys=True, default=str)

        lineage_id = value.get("lineage_id")
        if not lineage_id:
            raise ValueError("lineage_checkpoint_storage requires lineage_id")

        existing = self.conn.execute(
            """
            SELECT storage_policy_version, state_hash_spec_version,
                   parameter_schema_hash, base_state_sha256,
                   base_artifact_ref, base_artifact_sha256,
                   base_compression_codec, base_compression_settings_json,
                   base_logical_bytes, base_physical_bytes
            FROM lineage_checkpoint_storage
            WHERE lineage_id = ?
            """,
            (lineage_id,),
        ).fetchone()
        identity_fields = (
            "storage_policy_version",
            "state_hash_spec_version",
            "parameter_schema_hash",
            "base_state_sha256",
            "base_artifact_ref",
            "base_artifact_sha256",
            "base_compression_codec",
            "base_compression_settings_json",
            "base_logical_bytes",
            "base_physical_bytes",
        )
        if existing is not None:
            mismatches = [
                name
                for name, stored in zip(identity_fields, existing)
                if name in value and value[name] != stored
            ]
            if mismatches:
                joined = ", ".join(mismatches)
                raise ValueError(
                    f"Immutable checkpoint storage for lineage {lineage_id!r} conflicts in: {joined}"
                )
            return

        keys = list(value.keys())
        placeholders = ",".join(["?"] * len(keys))
        self.conn.execute(
            f"INSERT INTO lineage_checkpoint_storage ({','.join(keys)}) VALUES ({placeholders})",
            [value[key] for key in keys],
        )

    def write_service_checkpoint_storage(self, row: Mapping[str, Any]) -> None:
        """Persist one service's checkpoint choice and measured storage result.

        ``selected_physical_bytes`` is the incremental service-specific storage
        retained by the selected mode.  The shared lineage base is measured in
        ``lineage_checkpoint_storage`` and intentionally excluded from this
        per-service comparison.  Savings compare that selected byte count with
        an actually serialized full-checkpoint candidate.
        """
        value = dict(row)
        for key in ("patch_compression_settings_json", "metadata_json"):
            if value.get(key) is not None and not isinstance(value[key], str):
                value[key] = json.dumps(value[key], ensure_ascii=False, sort_keys=True, default=str)

        full_bytes = value.get("full_checkpoint_physical_bytes")
        selected_bytes = value.get("selected_physical_bytes")
        if full_bytes is not None and selected_bytes is not None:
            expected_savings = int(full_bytes) - int(selected_bytes)
            supplied_savings = value.get("savings_bytes")
            if supplied_savings is None:
                value["savings_bytes"] = expected_savings
            elif int(supplied_savings) != expected_savings:
                raise ValueError(
                    "savings_bytes must equal full_checkpoint_physical_bytes "
                    "minus selected_physical_bytes"
                )

            expected_ratio = expected_savings / int(full_bytes) if int(full_bytes) > 0 else 0.0
            supplied_ratio = value.get("savings_ratio")
            if supplied_ratio is None:
                value["savings_ratio"] = expected_ratio
            elif not np.isclose(float(supplied_ratio), expected_ratio, rtol=1e-12, atol=1e-12):
                raise ValueError(
                    "savings_ratio must equal savings_bytes divided by "
                    "full_checkpoint_physical_bytes"
                )

        self._ins("service_checkpoint_storage", value)

    def write_service_execution_provenance(self, *, service_id, run_id, started_at, document):
        """Append complete execution evidence without diagnostic truncation."""
        import uuid
        from ..provenance import canonical_json, sha256_json
        if self.conn is None:
            raise RuntimeError("SQLiteWriter.start() must be called before writing")
        execution_id = uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO service_execution_provenance "
            "(execution_id,service_id,run_id,started_at,document_sha256,document_json) VALUES (?,?,?,?,?,?)",
            (execution_id, service_id, run_id, started_at, sha256_json(document), canonical_json(document)),
        )
        return execution_id

    def write_service_generation_provenance(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        if value.get("provenance_json") is not None and not isinstance(value["provenance_json"], str):
            value["provenance_json"] = json.dumps(value["provenance_json"], ensure_ascii=False, default=str)
        self._ins("service_generation_provenance", value)

    def write_fedavg_run(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        for key in ("compatibility_contract_json", "metadata_json"):
            if value.get(key) is not None and not isinstance(value[key], str):
                value[key] = json.dumps(value[key], ensure_ascii=False, sort_keys=True, default=str)
        if self.conn is None:
            raise RuntimeError("SQLiteWriter.start() must be called before writing")
        keys = list(value.keys())
        placeholders = ",".join(["?"] * len(keys))
        updates = ",".join(
            f"{key}=excluded.{key}" for key in keys if key != "fl_run_id"
        )
        self.conn.execute(
            f"INSERT INTO fedavg_runs ({','.join(keys)}) VALUES ({placeholders}) "
            f"ON CONFLICT(fl_run_id) DO UPDATE SET {updates}",
            [value[key] for key in keys],
        )

    def write_fedavg_global_state(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        for key in ("compression_json", "metadata_json"):
            if value.get(key) is not None and not isinstance(value[key], str):
                value[key] = json.dumps(value[key], ensure_ascii=False, sort_keys=True, default=str)
        if self.conn is None:
            raise RuntimeError("SQLiteWriter.start() must be called before writing")
        identity_fields = (
            "state_hash",
            "cas_ref",
            "parameter_schema_hash",
            "parent_global_round",
            "parent_state_hash",
            "storage_format",
            "compression_json",
            "logical_bytes",
            "physical_bytes",
        )
        existing = self.conn.execute(
            """
            SELECT state_hash, cas_ref, parameter_schema_hash,
                   parent_global_round, parent_state_hash, storage_format,
                   compression_json, logical_bytes, physical_bytes
            FROM fedavg_global_states
            WHERE fl_run_id = ? AND global_round = ?
            """,
            (value.get("fl_run_id"), value.get("global_round")),
        ).fetchone()
        if existing is not None:
            mismatches = [
                name
                for name, stored in zip(identity_fields, existing)
                if value.get(name) != stored
            ]
            if mismatches:
                raise ValueError(
                    "Immutable FedAvg global state conflicts in: " + ", ".join(mismatches)
                )
            return
        keys = list(value.keys())
        placeholders = ",".join(["?"] * len(keys))
        self.conn.execute(
            f"INSERT INTO fedavg_global_states ({','.join(keys)}) VALUES ({placeholders})",
            [value[key] for key in keys],
        )

    def write_fedavg_client_partition(self, row: Mapping[str, Any]) -> None:
        """Persist one immutable requested-versus-realised client partition."""
        value = dict(row)
        json_fields = (
            "requested_partition_params_json",
            "target_bucket_counts_json",
            "target_bucket_proportions_json",
            "realised_bucket_counts_json",
            "realised_bucket_proportions_json",
            "metadata_json",
        )
        for key in json_fields:
            if value.get(key) is not None and not isinstance(value[key], str):
                value[key] = json.dumps(
                    value[key], ensure_ascii=False, sort_keys=True, default=str
                )
        if self.conn is None:
            raise RuntimeError("SQLiteWriter.start() must be called before writing")
        columns = [
            key for key in value if key not in {"created_at"}
        ]
        existing = self.conn.execute(
            f"SELECT {','.join(columns)} FROM fedavg_client_partitions "
            "WHERE fl_run_id=? AND client_id=?",
            (value.get("fl_run_id"), value.get("client_id")),
        ).fetchone()
        if existing is not None:
            mismatches = [
                key
                for key, stored in zip(columns, existing)
                if value.get(key) != stored
            ]
            if mismatches:
                raise ValueError(
                    "Immutable FedAvg client partition conflicts in: "
                    + ", ".join(mismatches)
                )
            return
        self._ins("fedavg_client_partitions", value)

    def write_fedavg_round(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        for key in ("contributor_metadata_json", "excluded_metadata_json", "metadata_json"):
            if value.get(key) is not None and not isinstance(value[key], str):
                value[key] = json.dumps(value[key], ensure_ascii=False, sort_keys=True, default=str)
        if self.conn is None:
            raise RuntimeError("SQLiteWriter.start() must be called before writing")
        keys = list(value)
        placeholders = ",".join(["?"] * len(keys))
        updates = ",".join(
            f"{key}=excluded.{key}"
            for key in keys
            if key not in {"fl_run_id", "round_id"}
        )
        self.conn.execute(
            f"INSERT INTO fedavg_rounds ({','.join(keys)}) VALUES ({placeholders}) "
            f"ON CONFLICT(fl_run_id,round_id) DO UPDATE SET {updates}",
            [value[key] for key in keys],
        )

    def write_fedavg_round_metrics(
        self,
        fl_run_id: str,
        round_id: int,
        values: Mapping[str, Any],
    ) -> None:
        """Upsert task-aware metrics for one aggregated global model."""
        for metric_name, spec in (values or {}).items():
            if spec is None:
                continue
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
                continue
            if domain not in ALLOWED_METRIC_DOMAINS:
                domain = "metadata"
            if direction not in {"higher_better", "lower_better", "neutral"}:
                direction = "neutral"
            row = {
                "fl_run_id": str(fl_run_id),
                "round_id": int(round_id),
                "metric_name": str(metric_name).strip().lower(),
                "domain": domain,
                "unit": unit,
                "direction": direction,
            }
            row.update(self._coerce_value_columns(value))
            self._ins("fedavg_round_metrics", row)

    def interrupt_running_fedavg(
        self,
        *,
        reason: str,
        fl_run_id: str | None = None,
    ) -> dict[str, int]:
        """Mark unfinished FedAvg coordination rows after a stopped process."""
        if self.conn is None:
            raise RuntimeError("SQLiteWriter.start() must be called before writing")
        now = datetime.now(timezone.utc).isoformat()
        params: list[Any] = [str(reason), now]
        run_filter = ""
        if fl_run_id is not None:
            run_filter = " AND fl_run_id = ?"
            params.append(str(fl_run_id))
        rounds = self.conn.execute(
            "UPDATE fedavg_rounds SET status='interrupted', status_reason=?, "
            "completed_at=COALESCE(completed_at, ?) WHERE status='running'" + run_filter,
            params,
        ).rowcount
        runs = self.conn.execute(
            "UPDATE fedavg_runs SET status='interrupted', status_reason=?, "
            "completed_at=COALESCE(completed_at, ?) WHERE status='running'" + run_filter,
            params,
        ).rowcount
        return {"runs": int(runs), "rounds": int(rounds)}

    def write_fedavg_client_round(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        if value.get("metadata_json") is not None and not isinstance(value["metadata_json"], str):
            value["metadata_json"] = json.dumps(
                value["metadata_json"], ensure_ascii=False, sort_keys=True, default=str
            )
        for key in ("scheduled", "participated", "succeeded"):
            if key in value:
                value[key] = int(bool(value[key]))
        self._ins("fedavg_client_rounds", value)

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
        fl_run_id: str | None = None,
        client_id: str | None = None,
        round_id: int | None = None,
    ) -> None:
        self._ins(
            "service_failures",
            {
                "service_id": service_id,
                "row_index": row_index,
                "case_name": case_name,
                "manifest_group_id": manifest_group_id,
                "failure_stage": failure_stage,
                "error_message": bounded_text(error_message),
                "resolved_config_json": diagnostic_json(resolved_config_json),
                "traceback_text": bounded_text(traceback_text),
                "fl_run_id": fl_run_id,
                "client_id": client_id,
                "round_id": round_id,
            },
        )

    def write_service_regeneration(self, row: Mapping[str, Any]) -> None:
        """Persist a regenerated representative without touching source metrics."""
        value = dict(row)
        for key in ("resolved_config_json", "dependency_versions_json", "provenance_json"):
            if key in value and value[key] is not None and not isinstance(value[key], str):
                value[key] = json.dumps(value[key], ensure_ascii=False, default=str)
        self._ins("service_regenerations", value)

    def write_composition_validation_run(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        for key in ("cohort_manifest_json", "inputs_json", "provenance_json", "summary_json"):
            if key in value and value[key] is not None and not isinstance(value[key], str):
                value[key] = json.dumps(value[key], ensure_ascii=False, default=str)
        self._ins("composition_validation_runs", value)

    def write_composition_validation_candidate(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        for key in (
            "service_ids_json",
            "qos_violations_json",
            "contributor_provenance_json",
            "aggregation_provenance_json",
        ):
            if key in value and value[key] is not None and not isinstance(value[key], str):
                value[key] = json.dumps(value[key], ensure_ascii=False, default=str)
        self._ins("composition_validation_candidates", value)

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
            if not np.isfinite(value):
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
