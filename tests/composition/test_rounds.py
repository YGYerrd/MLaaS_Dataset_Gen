import csv
from contextlib import contextmanager
import json
import sqlite3
from types import SimpleNamespace

import pytest

from mlaas_composition.cli import main
from mlaas_composition.manifest import generate_composition_manifest
from mlaas_composition.round_manifest import extend_manifest, load_baselines, VERSION
from mlaas_composition.round_execution import export_rounds
from mlaas_composition import run_manifest as batch, round_execution
from tests.composition.test_manifest import prepare, read_rows
from tests.composition.test_batch_executor import source, entry, manifest, ToyModel


def timing_database(path, families=("family_0", "family_1")):
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE composition_runs(run_id,executor_version);
            CREATE TABLE compositions(run_id,combination_id,family_id,composition_size,status);
            CREATE TABLE composition_metrics(run_id,combination_id,metric_name,value_num,value_int);
            INSERT INTO composition_runs VALUES('run','composition_batch_v1');
        """)
        for family in families:
            for size in (2, 3, 5):
                for i, total in enumerate((100, 120, 500)):
                    cid = f"{family}_{size}_{i}"
                    conn.execute("INSERT INTO compositions VALUES('run',?,?,?,'completed')", (cid, family, size))
                    conn.executemany("INSERT INTO composition_metrics VALUES('run',?,?,?,NULL)",
                        [(cid, "total_wall_s", total), (cid, "client_loss_measurement_wall_s", 20)])
    return path


def round_entry(family, ids=("a", "b"), rounds=3, baseline_s=1000, availability=1.0):
    row = entry(family, *ids)
    row["service_ids"] = json.dumps(row["service_ids"])
    baseline = {"baseline_round_time_s": baseline_s, "sample_count": 3,
                "observations_sha256": "a" * 64, "basis": "test"}
    extend_manifest([row], {}, {(family, len(ids)): baseline}, rounds=rounds, seed=42,
                    deadline_profile="moderate", availability_probability=availability)
    row["service_ids"] = json.loads(row["service_ids"])
    return row


@pytest.mark.parametrize("profile,multiplier", [("tight", .75), ("moderate", 1), ("relaxed", 1.5)])
def test_round_manifest_uses_frozen_local_evidence_and_small_sizes(tmp_path, profile, multiplier):
    prepared = prepare(tmp_path / "prepared")
    timing = timing_database(tmp_path / "timings.db")
    before = timing.read_bytes()
    output = tmp_path / "rounds.csv"
    main(["generate-manifest", "--prepared-dir", str(prepared), "--compositions", "12", "--rounds", "4",
          "--timing-db", str(timing), "--deadline-profile", profile, "--availability-probability", ".6",
          "--output", str(output)])
    rows = batch.load_composition_manifest(output)
    assert len(rows) == 12
    assert {row["composition_size"] for row in rows} == {2, 3, 5}
    for row in rows:
        config = row["round_config"]
        assert row["rounds"] == len(config["client_availability"]) == 4
        assert config["baseline"]["baseline_round_time_s"] == 100
        assert config["baseline"]["sample_count"] == 3
        assert config["round_deadline_s"] == 100 * multiplier
        assert config["communication"] == "in_process_no_network"
    assert timing.read_bytes() == before
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["composition_sizes"] == [2, 3, 5]
    assert metadata["rounds"] == 4
    other = tmp_path / "repeat.csv"
    generate_composition_manifest(prepared, other, compositions=12, rounds=4, timing_db=timing,
                                  deadline_profile=profile, availability_probability=.6)
    assert output.read_bytes() == other.read_bytes()


def test_baseline_skips_unmeasured_families_and_supports_small_pilots(tmp_path):
    prepared = prepare(tmp_path / "prepared")
    timing = timing_database(tmp_path / "timings.db", families=("family_0",))
    output = tmp_path / "rounds.csv"
    meta = generate_composition_manifest(prepared, output, compositions=3, rounds=2, timing_db=timing)
    assert len(read_rows(output)) == 3
    assert len(meta["skipped_pairs"]) == 3
    assert {r["reason"] for r in meta["skipped_pairs"]} == {"no_round_timing_baseline"}
    other = tmp_path / "pilot.csv"
    meta = generate_composition_manifest(prepared, other, compositions=3, rounds=2, timing_db=timing,
                                         family_ids=["family_0"])
    assert meta["skipped_pairs"] == []
    with pytest.raises(ValueError, match="Unknown or unverified"):
        generate_composition_manifest(prepared, tmp_path / "bad.csv", compositions=3, family_ids=["wrong"])


@pytest.mark.parametrize("options", [{"rounds": 0}, {"rounds": True}, {"rounds": 1.2},
    {"rounds": 2}, {"rounds": 2, "timing_db": "unused", "availability_probability": float("nan")},
    {"rounds": 2, "timing_db": "unused", "availability_probability": 1.1},
    {"timing_db": "unused"}])
def test_invalid_round_options_do_not_write(tmp_path, options):
    with pytest.raises(ValueError):
        generate_composition_manifest(tmp_path, tmp_path / "bad.csv", compositions=1, **options)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("change", ["size", "rounds", "schedule", "boolean", "deadline", "nonfinite", "profile"])
def test_loader_rejects_invalid_round_contract(tmp_path, change):
    row = round_entry("family", ids=tuple("abcdefg") if change == "size" else ("a", "b"))
    config = json.loads(row["round_config"])
    if change == "rounds": row["rounds"] = 0
    if change == "schedule": config["client_availability"].pop()
    if change == "boolean": config["client_availability"][0]["a"] = 1
    if change == "deadline": config["round_deadline_s"] += 1
    if change == "nonfinite": config["round_deadline_s"] = float("inf")
    if change == "profile": config["deadline_profile"] = "unknown"
    row["round_config"] = json.dumps(config)
    path = manifest(tmp_path / "bad.csv", [row])
    with pytest.raises(ValueError, match="Invalid composition manifest"):
        batch.load_composition_manifest(path)


@pytest.fixture
def evolving(source, monkeypatch):
    starts = []
    @contextmanager
    def materialize(db, sid, *, device, initial_state):
        value = float(initial_state.entries[0].to_numpy()[0])
        starts.append((sid, value))
        source.calls.append(sid)
        yield {"model": ToyModel(value + {"a": 2, "b": 6, "c": 10}[sid]), "initial": initial_state,
               "training": {"train_time_s": 1}, "replay_evidence": {"selected_rows_match": True}}
    monkeypatch.setattr("mlaas_data_generator.client_reconstruction.materialize_service", materialize)
    source.starts = starts
    return source


def saved_rounds(db):
    with sqlite3.connect(db) as conn:
        return [json.loads(row[0]) for row in conn.execute(
            "SELECT result_json FROM composition_rounds ORDER BY communication_round")]


def test_rounds_continue_global_state_persist_export_and_resume(evolving, tmp_path):
    path = manifest(tmp_path / "rounds.csv", [round_entry(evolving.doc["family_id"])])
    db = tmp_path / "results.db"
    before = evolving.db.read_bytes()
    result = batch.run_composition_manifest(path, evolving.db, db, device="cpu")
    assert result["completed"] == 1 and result["failed"] == 0
    assert evolving.starts == [("a", 0), ("b", 0), ("a", 5), ("b", 5), ("a", 10), ("b", 10)]
    assert evolving.aggregate_values == [5, 10, 15]
    rounds = saved_rounds(db)
    assert [r["communication_round"] for r in rounds] == [1, 2, 3]
    assert all(r["global_accuracy_per_round"] == 1 for r in rounds)
    assert rounds[1]["parent_state_hash"] == rounds[0]["global_state_hash"]
    assert rounds[2]["parent_state_hash"] == rounds[1]["global_state_hash"]
    assert all(r["round_completion_time_s"] > 0 for r in rounds)
    output = tmp_path / "rounds_export.csv"
    assert export_rounds(db, output)["rows"] == 6
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["communication_round"] for r in rows] == ["1", "1", "2", "2", "3", "3"]
    assert {r["client_availability"] for r in rows} == {"True"}
    assert {r["global_accuracy_per_round"] for r in rows} == {"1.0"}
    assert evolving.db.read_bytes() == before
    assert batch.run_composition_manifest(path, evolving.db, db, device="cpu")["skipped"] == 1
    assert len(evolving.starts) == 6
    with pytest.raises(FileExistsError): export_rounds(db, output)


def test_unavailable_clients_and_single_contributor_preserve_history(evolving, tmp_path):
    row = round_entry(evolving.doc["family_id"])
    config = json.loads(row["round_config"])
    config["client_availability"] = [{"a": False, "b": False}, {"a": False, "b": True}, {"a": True, "b": False}]
    row["round_config"] = json.dumps(config)
    path = manifest(tmp_path / "rounds.csv", [row])
    db = tmp_path / "results.db"
    assert batch.run_composition_manifest(path, evolving.db, db, device="cpu")["completed"] == 1
    assert evolving.starts == [("b", 0), ("a", 6)]
    assert evolving.aggregate_values == [0, 6, 8]
    rounds = saved_rounds(db)
    assert rounds[0]["status"] == "no_update"
    assert rounds[0]["parent_state_hash"] == rounds[0]["global_state_hash"]
    assert [c["aggregation_weight"] for c in rounds[1]["clients"]] == [0, 1]
    assert rounds[1]["clients"][0]["status"] == "unavailable"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT service_id,status,aggregation_weight FROM composition_members ORDER BY service_id").fetchall() == [
            ("a", "completed", 1.0), ("b", "unavailable", 0.0)]


def test_deadlines_exclude_late_updates_and_stop_starting_clients(evolving, tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(round_execution, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    from mlaas_data_generator import client_reconstruction
    original = client_reconstruction.materialize_service
    @contextmanager
    def slow(*args, **kwargs):
        with original(*args, **kwargs) as value:
            clock[0] += 1.25
            yield value
    monkeypatch.setattr(client_reconstruction, "materialize_service", slow)
    row = round_entry(evolving.doc["family_id"], ids=("a", "b", "c"), rounds=1, baseline_s=2)
    path = manifest(tmp_path / "rounds.csv", [row])
    db = tmp_path / "results.db"
    assert batch.run_composition_manifest(path, evolving.db, db, device="cpu")["completed"] == 1
    result = saved_rounds(db)[0]
    assert [c["status"] for c in result["clients"]] == ["completed", "late", "deadline_skipped"]
    assert [c["client_availability"] for c in result["clients"]] == [True, True, True]
    assert [c["accepted"] for c in result["clients"]] == [True, False, False]
    assert evolving.aggregate_values == [2]
    assert result["round_completion_time_s"] == 2.5 and result["deadline_violation"] is True


def test_round_failure_keeps_completed_rounds_and_retry_restarts_series(evolving, tmp_path, monkeypatch):
    from mlaas_data_generator import client_reconstruction
    original = client_reconstruction.materialize_service
    @contextmanager
    def fail_second_round(*args, **kwargs):
        if kwargs["initial_state"].sha256 != evolving.initial.sha256:
            raise RuntimeError("round two failure")
        with original(*args, **kwargs) as value:
            yield value
    path = manifest(tmp_path / "rounds.csv", [round_entry(evolving.doc["family_id"], rounds=2)])
    db = tmp_path / "results.db"
    with monkeypatch.context() as patch:
        patch.setattr(client_reconstruction, "materialize_service", fail_second_round)
        assert batch.run_composition_manifest(path, evolving.db, db, device="cpu")["failed"] == 1
    assert len(saved_rounds(db)) == 1
    assert export_rounds(db, tmp_path / "partial.csv")["rows"] == 2
    assert batch.run_composition_manifest(path, evolving.db, db, device="cpu")["completed"] == 1
    assert len(saved_rounds(db)) == 2


def test_round_dry_run_is_metadata_only_and_deadline_changes_run_identity(evolving, tmp_path, monkeypatch):
    row = round_entry(evolving.doc["family_id"])
    path = manifest(tmp_path / "rounds.csv", [row])
    db = tmp_path / "absent" / "results.db"
    monkeypatch.setattr("mlaas_data_generator.cli.manifest.preparation.load_lineage_base",
                        lambda *a, **kw: pytest.fail("dry run loaded tensors"))
    first = batch.run_composition_manifest(path, evolving.db, db, dry_run=True)
    assert first["ready"] == 1 and not db.parent.exists() and not evolving.starts
    config = json.loads(row["round_config"])
    config.update(deadline_profile="tight", deadline_multiplier=.75, round_deadline_s=750)
    row["round_config"] = json.dumps(config)
    manifest(path, [row])
    second = batch.run_composition_manifest(path, evolving.db, db, dry_run=True)
    assert first["run_id"] != second["run_id"]
