"""Coordinator-owned SQLite results; source services are small immutable references."""

from datetime import datetime, timezone
from importlib import resources
import json
from pathlib import Path
import sqlite3

from mlaas_data_generator.diagnostic_json import diagnostic_json
from mlaas_data_generator.shared_parent import _clean
from mlaas_data_generator.storage.writer import SQLiteWriter, _infer_metric_domain, _infer_metric_direction


def now():
    return datetime.now(timezone.utc).isoformat()


class CompositionWriter(SQLiteWriter):
    """Reuse the catalogue's codecs/transactions with only composition tables."""

    def start(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        try:
            tables = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            expected = {"composition_runs", "source_services", "compositions", "composition_members", "composition_metrics",
                        "composition_rounds"}
            if tables - expected:
                raise ValueError("Destination is not a composition results database")
            version = self.conn.execute("PRAGMA user_version").fetchone()[0]
            if tables and version != 1:
                raise ValueError(f"Unsupported composition database version: {version}")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA journal_mode=DELETE")
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.executescript(resources.files("mlaas_composition").joinpath("assets/composition_schema.sql").read_text())
            self.conn.execute("PRAGMA user_version=1")
            self.conn.commit()
        except BaseException:
            self.abort()
            raise

    def register(self, run_id, *, fingerprint, manifest, source_db, version, settings, entries, sources):
        with self.conn:
            self.conn.execute("""INSERT INTO composition_runs
                (run_id,input_fingerprint,manifest_path,source_db,executor_version,settings_json,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET
                updated_at=excluded.updated_at, settings_json=excluded.settings_json,
                manifest_path=excluded.manifest_path, source_db=excluded.source_db""",
                (run_id, fingerprint, str(manifest), str(source_db), version, diagnostic_json(settings), now(), now()))
            for sid, snapshot in sources.items():
                self.conn.execute("INSERT OR IGNORE INTO source_services VALUES (?,?,?,?,?)",
                                  (run_id, sid, snapshot.get("family_id"), snapshot.get("sample_count"), diagnostic_json(snapshot)))
            for entry in entries:
                cid = entry["combination_id"]
                self.conn.execute("""INSERT OR IGNORE INTO compositions
                    (run_id,combination_id,family_id,composition_size,status) VALUES (?,?,?,?,'pending')""",
                    (run_id, cid, entry["family_id"], entry["composition_size"]))
                counts = [sources[sid].get("sample_count") for sid in entry["service_ids"]]
                total = sum(counts) if all(count and count > 0 for count in counts) else 0
                for ordinal, sid in enumerate(entry["service_ids"]):
                    count = sources[sid].get("sample_count")
                    self.conn.execute("""INSERT OR IGNORE INTO composition_members
                        (run_id,combination_id,service_id,ordinal,sample_count,aggregation_weight)
                        VALUES (?,?,?,?,?,?)""", (run_id, cid, sid, ordinal, count, count / total if count and total else None))

    def completed(self, run_id):
        return {r[0] for r in self.conn.execute(
            "SELECT combination_id FROM compositions WHERE run_id=? AND status='completed'", (run_id,))}

    def completed_without_client_losses(self, run_id, version):
        return self.conn.execute("""SELECT COUNT(*) FROM compositions c
            WHERE c.run_id=? AND c.status='completed' AND EXISTS (
                SELECT 1 FROM composition_members m
                WHERE m.run_id=c.run_id AND m.combination_id=c.combination_id
                AND COALESCE(json_extract(m.evidence_json, '$.client_loss_statistics.version'), '') != ?)
            """, (run_id, version)).fetchone()[0]

    def begin_composition(self, run_id, cid, worker):
        with self.conn:
            self.conn.execute("""UPDATE compositions SET status='running', attempt_count=attempt_count+1,
                worker=?, started_at=?, finished_at=NULL, wall_s=NULL, aggregation_provenance_json=NULL,
                failure_stage=NULL, failure_service_id=NULL, error_message=NULL WHERE run_id=? AND combination_id=?""",
                (worker, now(), run_id, cid))
            self.conn.execute("DELETE FROM composition_metrics WHERE run_id=? AND combination_id=?", (run_id, cid))
            self.conn.execute("DELETE FROM composition_rounds WHERE run_id=? AND combination_id=?", (run_id, cid))
            self.conn.execute("""UPDATE composition_members SET status='pending', training_wall_s=NULL,
                state_hash=NULL,evidence_json=NULL WHERE run_id=? AND combination_id=?""", (run_id, cid))

    def progress(self, run_id, cid, event):
        if event["stage"] == "round_completed":
            with self.conn:
                self._round_result(run_id, cid, event["result"])
            return
        sid = event.get("service_id")
        if not sid:
            return
        with self.conn:
            if event["stage"] == "training":
                self.conn.execute("""UPDATE composition_members SET status='running'
                    WHERE run_id=? AND combination_id=? AND service_id=?""", (run_id, cid, sid))
            elif event["stage"] == "member_completed":
                self._member_evidence(run_id, cid, event["evidence"])

    def _member_evidence(self, run_id, cid, evidence):
        self.conn.execute("""UPDATE composition_members SET status='completed',training_wall_s=?,state_hash=?,evidence_json=?
            WHERE run_id=? AND combination_id=? AND service_id=?""",
            (evidence.get("reconstruction_wall_s"), evidence.get("materialized_state_hash"),
             json.dumps(_clean(evidence), allow_nan=False), run_id, cid, evidence["service_id"]))

    def _round_result(self, run_id, cid, result):
        self.conn.execute("""INSERT OR REPLACE INTO composition_rounds VALUES (?,?,?,?,?,?,?)""",
            (run_id, cid, result["communication_round"], result["round_deadline_s"],
             result["round_completion_time_s"], result["global_accuracy_per_round"],
             json.dumps(_clean(result), allow_nan=False)))
        # Membership reflects the latest completed round; complete history lives
        # in composition_rounds, including unavailable and late clients.
        for client in result["clients"]:
            evidence = {**client, "service_id": client["client_id"],
                        "communication_round": result["communication_round"]}
            self.conn.execute("""UPDATE composition_members SET status=?,training_wall_s=?,state_hash=?,
                aggregation_weight=?,evidence_json=? WHERE run_id=? AND combination_id=? AND service_id=?""",
                (client["status"], client["training_wall_s"], client["materialized_state_hash"],
                 client["aggregation_weight"], json.dumps(_clean(evidence), allow_nan=False),
                 run_id, cid, client["client_id"]))

    def finish_composition(self, run_id, cid, result):
        # Metrics, membership evidence and completion become visible together.
        with self.conn:
            for result_round in result.get("rounds", []):
                self._round_result(run_id, cid, result_round)
            for evidence in result.get("members", []):
                self._member_evidence(run_id, cid, evidence)
            provenance = result.get("provenance")
            self.conn.execute("""UPDATE compositions SET status=?,finished_at=?,wall_s=?,
                aggregation_provenance_json=?,failure_stage=?,failure_service_id=?,error_message=?
                WHERE run_id=? AND combination_id=?""",
                (result["status"], now(), result.get("wall_s"), json.dumps(_clean(provenance), allow_nan=False) if provenance else None,
                 result.get("failure_stage"), result.get("failure_service_id"), result.get("error_message"), run_id, cid))
            if result["status"] == "completed":
                for name, value in result["metrics"].items():
                    domain = "runtime" if name.endswith("wall_s") else _infer_metric_domain(name)
                    if name == "quality_score":
                        domain = "quality"
                    direction = "lower_better" if name.endswith("wall_s") else _infer_metric_direction(name)
                    self._ins("composition_metrics", {
                        "run_id": run_id, "combination_id": cid, "metric_name": name,
                        "domain": domain, "unit": "seconds" if name.endswith(("_s", "_s_per_example")) else None,
                        "direction": direction, **self._coerce_value_columns(value)})
            else:
                self.conn.execute("""UPDATE composition_members SET status='failed'
                    WHERE run_id=? AND combination_id=? AND (status='running' OR service_id=?)""",
                    (run_id, cid, result.get("failure_service_id")))

    def summary(self, run_id):
        counts = {"completed": 0, "failed": 0, "pending": 0, "running": 0}
        counts.update(dict(self.conn.execute("SELECT status,COUNT(*) FROM compositions WHERE run_id=? GROUP BY status", (run_id,))))
        with self.conn:
            self.conn.execute("UPDATE composition_runs SET summary_json=?,updated_at=? WHERE run_id=?",
                              (json.dumps(counts), now(), run_id))
        return {"run_id": run_id, **counts}
