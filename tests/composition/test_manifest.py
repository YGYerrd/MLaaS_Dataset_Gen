import csv
import hashlib
import json
import random
import sqlite3
from collections import Counter
from contextlib import closing

import pytest

from mlaas_composition.cli import main
from mlaas_composition import manifest


def prepare(root, sizes=(12, 13), reverse=False):
    root.mkdir()
    pools, services, metrics = [], [], []
    for index, size in enumerate(sizes):
        family = f"family_{index}"
        pools.append({"family_id": family, "dependency_status": "verified", "supported_composition_sizes": "[2,5,10,15]"})
        for member in range(size):
            sid = f"{family}_svc_{member:03d}"
            services.append({"service_id": sid, "family_id": family, "status": "verified"})
            metrics.append((sid, "training_runtime_s", float(member + 1), None))
    if reverse:
        pools.reverse()
        services.reverse()
        metrics.reverse()
    for name, rows in (("pools.csv", pools), ("services.csv", services)):
        with (root / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with closing(sqlite3.connect(root / "services.db")) as conn, conn:
        conn.executescript("""
            CREATE TABLE services (service_id TEXT PRIMARY KEY, family_id TEXT, status TEXT);
            CREATE TABLE service_generation_provenance (service_id TEXT PRIMARY KEY, family_id TEXT);
            CREATE TABLE service_metrics (service_id TEXT, metric_name TEXT, value_num REAL, value_int INTEGER,
                                          UNIQUE(service_id, metric_name));
        """)
        conn.executemany("INSERT INTO services VALUES (?, ?, 'completed')", [(r["service_id"], r["family_id"]) for r in services])
        conn.executemany("INSERT INTO service_generation_provenance VALUES (?, ?)", [(r["service_id"], r["family_id"]) for r in services])
        conn.executemany("INSERT INTO service_metrics VALUES (?, ?, ?, ?)", metrics)
    return root


def read_rows(output):
    with output.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_reproducibility_input_order_and_source_preservation(tmp_path):
    prepared = prepare(tmp_path / "prepared")
    shuffled = prepare(tmp_path / "shuffled", reverse=True)
    before = {path: path.read_bytes() for path in prepared.iterdir()}
    random.seed(913)
    global_state = random.getstate()
    outputs = [tmp_path / f"run_{i}.csv" for i in range(3)]
    for source, output in zip((prepared, prepared, shuffled), outputs):
        manifest.generate_composition_manifest(source, output, compositions=20, seed=42)
    assert random.getstate() == global_state
    assert len({output.read_bytes() for output in outputs}) == 1
    assert len({output.with_suffix(".metadata.json").read_bytes() for output in outputs}) == 1
    assert before == {path: path.read_bytes() for path in prepared.iterdir()}
    other = tmp_path / "other_seed.csv"
    manifest.generate_composition_manifest(prepared, other, compositions=20, seed=43)
    assert other.read_bytes() != outputs[0].read_bytes()


def test_cli_manifest_contract_and_combination_ids(tmp_path, capsys):
    prepared = prepare(tmp_path / "prepared")
    output = tmp_path / "nested" / "compositions.csv"
    main(["generate-manifest", "--prepared-dir", str(prepared), "--compositions", "30", "--output", str(output)])
    rows = read_rows(output)
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert list(rows[0]) == ["combination_id", "family_id", "composition_size", "service_ids"]
    assert metadata["seed"] == 42
    assert metadata["requested_count"] == metadata["actual_count"] == len(rows) == 30
    assert metadata["composition_sizes"] == [2, 3, 5, 7, 10]
    assert metadata["eligible_pair_count"] == 10
    assert metadata["per_pair_quota"] == 3
    assert len(metadata["selection_inputs_sha256"]) == 64
    assert not metadata["skipped_pairs"] and not metadata["shortfalls"]
    assert set(Counter((r["family_id"], int(r["composition_size"])) for r in rows).values()) == {3}
    assert len({r["combination_id"] for r in rows}) == len(rows)
    for row in rows:
        members = json.loads(row["service_ids"])
        assert len(members) == len(set(members)) == int(row["composition_size"])
        assert members == sorted(members)
        assert all(sid.startswith(row["family_id"] + "_svc_") for sid in members)
        canonical = json.dumps([row["family_id"], members], separators=(",", ":"))
        assert row["combination_id"] == "comb_" + hashlib.sha256(canonical.encode()).hexdigest()
    # Independently seeded full selections share the same membership-based ID.
    small = prepare(tmp_path / "small", sizes=(2,))
    ids = []
    for seed in (1, 99):
        path = tmp_path / f"full_{seed}.csv"
        manifest.generate_composition_manifest(small, path, compositions=1, seed=seed)
        ids.append(read_rows(path)[0]["combination_id"])
    assert ids[0] == ids[1]
    assert "Wrote 30 compositions" in capsys.readouterr().out


@pytest.mark.parametrize(("target", "quota", "actual"), [(1, 1, 10), (14, 1, 10), (15, 2, 20), (26, 3, 30)])
def test_full_batch_rounding(tmp_path, target, quota, actual):
    prepared = prepare(tmp_path / "prepared")
    result = manifest.generate_composition_manifest(prepared, tmp_path / "out.csv", compositions=target)
    assert result["per_pair_quota"] == quota
    assert result["actual_count"] == actual


def test_small_pool_capacity_and_skipped_sizes(tmp_path):
    prepared = prepare(tmp_path / "prepared", sizes=(3,))
    result = manifest.generate_composition_manifest(prepared, tmp_path / "out.csv", compositions=100)
    assert result["eligible_pair_count"] == 2
    assert result["per_pair_quota"] == 50
    assert result["actual_count"] == 4
    assert {row["composition_size"] for row in result["skipped_pairs"]} == {5, 7, 10}
    assert [row["actual_count"] for row in result["shortfalls"]] == [3, 1]
    assert {row["reason"] for row in result["shortfalls"]} == {"combination_space_exhausted"}


def test_attempt_limit_reports_shortfall_without_duplicates(tmp_path, monkeypatch):
    prepared = prepare(tmp_path / "prepared", sizes=(12,))
    calls = Counter()

    def repeated_draw(rng, ids, weights, size):
        calls[size] += 1
        return tuple(ids[:size])

    monkeypatch.setattr(manifest, "_draw_members", repeated_draw)
    result = manifest.generate_composition_manifest(prepared, tmp_path / "out.csv", compositions=10)
    assert result["actual_count"] == 5
    assert set(calls.values()) == {100}
    assert len(result["shortfalls"]) == 5
    assert {row["reason"] for row in result["shortfalls"]} == {"sampling_attempt_limit"}


def test_only_verified_completed_members_and_pools_are_used(tmp_path):
    prepared = prepare(tmp_path / "prepared", sizes=(12, 12))
    path = prepared / "pools.csv"
    path.write_text(path.read_text().replace("family_1,verified", "family_1,blocked"))
    path = prepared / "services.csv"
    path.write_text(path.read_text().replace("family_0_svc_000,family_0,verified", "family_0_svc_000,family_0,blocked"))
    with closing(sqlite3.connect(prepared / "services.db")) as conn, conn:
        conn.execute("UPDATE services SET status='failed' WHERE service_id='family_0_svc_001'")
    output = tmp_path / "out.csv"
    result = manifest.generate_composition_manifest(prepared, output, compositions=1)
    assert result["actual_count"] == 5
    assert len(result["skipped_pairs"]) == 5
    for row in read_rows(output):
        assert row["family_id"] == "family_0"
        assert not {"family_0_svc_000", "family_0_svc_001"} & set(json.loads(row["service_ids"]))


@pytest.mark.parametrize("problem", ["duplicate_service", "duplicate_family", "wrong_family", "wrong_service_family", "missing_service", "unknown_family"])
def test_rejects_inconsistent_prepared_inputs(tmp_path, problem):
    prepared = prepare(tmp_path / "prepared")
    if problem.startswith("duplicate"):
        path = prepared / ("services.csv" if problem == "duplicate_service" else "pools.csv")
        with path.open("a") as stream:
            stream.write(path.read_text().splitlines()[1] + "\n")
    else:
        with closing(sqlite3.connect(prepared / "services.db")) as conn, conn:
            if problem == "missing_service":
                conn.execute("DELETE FROM services WHERE service_id='family_0_svc_000'")
            elif problem == "wrong_service_family":
                conn.execute("UPDATE services SET family_id='wrong' WHERE service_id='family_0_svc_000'")
            else:
                conn.execute("UPDATE service_generation_provenance SET family_id='wrong' WHERE service_id='family_0_svc_000'")
        if problem == "unknown_family":
            path = prepared / "services.csv"
            path.write_text(path.read_text().replace("family_0_svc_000,family_0", "family_0_svc_000,wrong"))
    output = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="Duplicate|membership mismatch|missing from database"):
        manifest.generate_composition_manifest(prepared, output, compositions=1)
    assert not output.exists() and not output.with_suffix(".metadata.json").exists()


def test_timing_fallback_missing_values_and_input_fingerprint(tmp_path):
    prepared = prepare(tmp_path / "prepared", sizes=(12,))
    before = manifest.generate_composition_manifest(prepared, tmp_path / "before.csv", compositions=5)
    with closing(sqlite3.connect(prepared / "services.db")) as conn, conn:
        conn.execute("UPDATE service_metrics SET value_num=-1 WHERE service_id='family_0_svc_000'")
        conn.execute("INSERT INTO service_metrics VALUES ('family_0_svc_000','train_runtime_s',NULL,7)")
        conn.execute("UPDATE service_metrics SET value_num=? WHERE service_id='family_0_svc_001'", (float("inf"),))
        conn.execute("DELETE FROM service_metrics WHERE service_id='family_0_svc_002'")
        conn.execute("INSERT INTO service_metrics VALUES ('family_0_svc_003','train_runtime_s',999,NULL)")
    pools, _ = manifest._load_pools(prepared)
    times = dict(pools["family_0"])
    assert times["family_0_svc_000"] == 7
    assert times["family_0_svc_001"] is None
    assert times["family_0_svc_002"] is None
    assert times["family_0_svc_003"] == 4
    after = manifest.generate_composition_manifest(prepared, tmp_path / "after.csv", compositions=5)
    assert before["selection_inputs_sha256"] != after["selection_inputs_sha256"]


def test_sampling_has_a_mild_training_preference():
    ids = [f"s{i}" for i in range(20)]
    weights = manifest._sampling_weights([1.0] * 10 + [10.0] * 10)
    rng = random.Random(42)
    fast = set(ids[:10])
    count = sum(len(fast.intersection(manifest._draw_members(rng, ids, weights, 2))) for _ in range(10000))
    assert 0.52 < count / 20000 < 0.57
    assert manifest._sampling_weights([None, 2.0, 2.0]) == [1.0, 1.0, 1.0]
    assert manifest._sampling_weights([None, None]) == [1.0, 1.0]
    assert manifest._sampling_weights([0.0, None, 10.0]) == [1.2, 1.0, 1.0]


@pytest.mark.parametrize("target", [0, -1, 1.5, True])
def test_invalid_counts_do_not_write_files(tmp_path, target):
    with pytest.raises(ValueError, match="positive integer"):
        manifest.generate_composition_manifest(tmp_path, tmp_path / "out.csv", compositions=target)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("target", ["0", "-1", "1.5"])
def test_cli_rejects_invalid_counts(tmp_path, target, capsys):
    with pytest.raises(SystemExit) as error:
        main(["generate-manifest", "--prepared-dir", str(tmp_path), "--compositions", target, "--output", str(tmp_path / "out.csv")])
    assert error.value.code == 2
    assert "positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("existing", ["csv", "metadata", "both"])
def test_overwrite_checks_both_files_before_writing(tmp_path, existing):
    prepared = prepare(tmp_path / "prepared")
    output = tmp_path / "out.csv"
    metadata_path = output.with_suffix(".metadata.json")
    paths = [output] if existing == "csv" else [metadata_path] if existing == "metadata" else [output, metadata_path]
    for path in paths:
        path.write_text("original")
    with pytest.raises(FileExistsError, match="--overwrite"):
        manifest.generate_composition_manifest(prepared, output, compositions=10)
    assert all(path.read_text() == "original" for path in paths)
    assert output.exists() == (existing in {"csv", "both"})
    assert metadata_path.exists() == (existing in {"metadata", "both"})
    manifest.generate_composition_manifest(prepared, output, compositions=10, overwrite=True)
    assert len(read_rows(output)) == 10
    assert json.loads(metadata_path.read_text())["actual_count"] == 10


def test_empty_eligibility_missing_inputs_and_source_output_protection(tmp_path):
    prepared = prepare(tmp_path / "prepared", sizes=(1,))
    with pytest.raises(ValueError, match="No eligible"):
        manifest.generate_composition_manifest(prepared, tmp_path / "out.csv", compositions=1)
    for name in ("pools.csv", "services.csv"):
        before = (prepared / name).read_bytes()
        with pytest.raises(ValueError, match="prepared input"):
            manifest.generate_composition_manifest(prepared, prepared / name, compositions=1, overwrite=True)
        assert (prepared / name).read_bytes() == before
    with pytest.raises(ValueError, match="must be .csv"):
        manifest.generate_composition_manifest(prepared, tmp_path / "out.json", compositions=1)
    (prepared / "services.db").unlink()
    with pytest.raises(sqlite3.OperationalError):
        manifest.generate_composition_manifest(prepared, tmp_path / "out.csv", compositions=1)
    assert not (prepared / "services.db").exists()
