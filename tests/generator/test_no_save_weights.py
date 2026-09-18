import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlaas_data_generator.services import runner
from mlaas_data_generator.cli import main as cli, run_manifest as manifest
from mlaas_data_generator.cli.manifest import preparation
from tests.generator.test_service_runner import DummyModel


@pytest.fixture
def debug_training(monkeypatch, tmp_path):
    model = DummyModel()
    def load(*args, **kwargs):
        x, y = np.array([[0.], [1.]]), np.array([0, 1])
        return (x, y), (x, y[::-1]), {"input_shape": (1,), "num_classes": 2,
            "task_type": "classification", "input_schema": "tabular_features"}
    monkeypatch.setattr(runner, "load_dataset", load)
    monkeypatch.setattr(runner, "create_model", lambda **kwargs: model)
    monkeypatch.setattr(runner, "capture_hardware_snapshot", lambda: {"platform": "test"})
    def forbidden(*args, **kwargs):
        pytest.fail("debug execution attempted checkpoint persistence or verification")
    for name in ("ensure_lineage_base", "save_model_artifact", "restore_model_artifact",
                 "persist_service_patch", "persist_service_patch_from_base_state",
                 "save_reconstruction_support_artifacts"):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(preparation, "prepare", forbidden)
    monkeypatch.setattr(preparation, "verify_completed", forbidden)
    cfg = {"service_id": "client", "db_path": str(tmp_path / "services.db"),
        "dataset": "synthetic", "dataset_name": "synthetic", "model_type": "mlp",
        "task_type": "classification", "training_regime": "generic", "training_epochs": 1,
        "batch_size": 2, "no_save_weights": True, "save_weights": True,
        "provenance_artifact_dir": str(tmp_path / "artifacts"), "update_signature_enabled": False,
        "partition_generator_version": "independent_source_pool_v3",
        "split_strategy": "iid", "sample_size": 2, "sample_seed": 42,
        "training_seed": 42, "client_partition_index": 0, "heterogeneity_intensity": 0.5}
    return cfg, model


def assert_debug_record(cfg):
    with sqlite3.connect(cfg["db_path"]) as conn:
        row = conn.execute("SELECT status,metadata_json FROM services WHERE service_id='client__debug_no_weights'").fetchone()
        assert row[0] == "completed"
        assert json.loads(row[1])["checkpoint_persistence"]["weights_saved"] is False
        assert conn.execute("SELECT count(*) FROM service_metrics WHERE service_id='client__debug_no_weights'").fetchone()[0] > 0
        assert conn.execute("SELECT composition_compatible FROM service_generation_provenance WHERE service_id='client__debug_no_weights'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM service_checkpoint_storage").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM lineage_checkpoint_storage").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM service_split_provenance WHERE service_id='client__debug_no_weights'").fetchone()[0] == 2
        partition = json.loads(conn.execute(
            "SELECT split_config_json FROM service_split_provenance "
            "WHERE service_id='client__debug_no_weights' AND split_name='train'"
        ).fetchone()[0])
        assert partition["partition_generator_version"] == "independent_source_pool_v4"
        assert partition["source_pool_sample_count"] == 2
        assert partition["effective_quota"] == 2


def test_debug_trains_and_evaluates_without_checkpoints(debug_training, tmp_path):
    cfg, model = debug_training
    result = runner.execute_service(cfg)
    assert result.status == "success", result.error
    assert model.fit_calls == 1 and not result.weights_saved
    assert result.checkpoint_storage is None
    assert_debug_record(cfg)
    with sqlite3.connect(cfg["db_path"]) as conn:
        document = json.loads(conn.execute(
            "SELECT document_json FROM service_execution_provenance"
        ).fetchone()[0])
        assert document["requested_config"]["training_epochs"] == 1
        assert document["effective_config"]["batch_size"] == 2
        assert document["training_config"]["batch_size"] == 2
        assert document["metrics"]["runtime_total_s"]["value"] == result.metrics["runtime_total_s"]["value"]
        assert len(document["splits"]) == 2
    # Weight-free debugging still persists the default benchmark evidence.
    assert result.metrics["evaluation_protocol_version"]["value"] == "benchmark_evaluation_v1"
    assert Path(result.metrics["metric_evaluation_evidence_ref"]["value"]).is_file()


def test_normal_versioned_client_can_select_without_frozen_pool(debug_training):
    cfg, model = debug_training
    client = runner.ServiceRunner({**cfg, "no_save_weights": False})
    result = client._resolve_service_split(np.arange(10).reshape(-1, 1), np.arange(10) % 2, {"task_type": "classification"})
    assert len(result["y_train"]) == 2
    assert result["resolved"]["source_pool_sample_count"] == 10


def test_manifest_flag_bypasses_family_persistence_and_preserves_provider(debug_training, tmp_path):
    cfg, model = debug_training
    from mlaas_data_generator.storage.writer import SQLiteWriter
    writer = SQLiteWriter(cfg["db_path"])
    writer.start()
    writer.write_service({"service_id": "client", "status": "completed"})
    writer.finish()
    source = tmp_path / "clients.csv"
    pd.DataFrame([{k: v for k, v in cfg.items() if k != "no_save_weights"}]).to_csv(source, index=False)
    result_path = manifest.run_manifest(str(source), no_save_weights=True)
    assert model.fit_calls == 1
    results = pd.read_csv(result_path)
    assert results.iloc[0].service_id == "client__debug_no_weights"
    assert results.iloc[0].status == "success"
    assert_debug_record(cfg)
    with sqlite3.connect(cfg["db_path"]) as conn:
        assert conn.execute("SELECT status FROM services WHERE service_id='client'").fetchone()[0] == "completed"
        assert conn.execute("SELECT status FROM manifest_service_runs").fetchone()[0] == "completed"


@pytest.mark.parametrize("extra", [{"fl_run_id": "rounds"}, {"generation_mode": "shared_parent"}])
def test_checkpoint_dependent_modes_reject_debug_before_training(debug_training, extra):
    cfg, model = debug_training
    result = runner.execute_service({**cfg, **extra})
    assert result.status == "failed" and "require checkpoints" in result.error
    assert model.fit_calls == 0


def test_cli_parsers_and_manifest_boolean():
    assert cli.build_parser().parse_args(["run-manifest", "--file", "clients.csv", "--no-save-weights"]).no_save_weights
    assert manifest.build_parser().parse_args(["--file", "clients.csv", "--no-save-weights"]).no_save_weights
    normal = manifest._resolve_row(pd.Series({"service_id": "client", "no_save_weights": "false"}), {})
    debug = manifest._resolve_row(pd.Series({"service_id": "client", "no_save_weights": "true"}), {})
    assert normal["service_id"] == "client"
    assert debug["service_id"] == "client__debug_no_weights" and debug["save_weights"] is False
    assert not preparation.applies(debug)


def test_smoke_flag_skips_optional_work_and_both_signature_snapshots(debug_training, monkeypatch, tmp_path):
    cfg, model = debug_training
    cfg.update(update_signature_enabled=True, enable_perturbation_metrics=True,
               explainability_enabled=True)
    def forbidden(*args, **kwargs):
        pytest.fail("smoke test executed optional perturbation/signature work")
    monkeypatch.setattr(runner, "_snapshot_model_weights", forbidden)
    monkeypatch.setattr(runner, "run_perturbation_stage", forbidden)
    monkeypatch.setattr(runner, "compute_and_store_update_signature", forbidden)
    source = tmp_path / "smoke.csv"
    pd.DataFrame([cfg]).to_csv(source, index=False)
    path = manifest.run_manifest(str(source), smoke_test=True, no_save_weights=True)
    result = pd.read_csv(path).iloc[0]
    assert result.status == "success", result.get("error_message")
    assert result.service_id == "client__debug_no_weights__smoke_test"
    assert model.fit_calls == 1
    with sqlite3.connect(cfg["db_path"]) as conn:
        stored = json.loads(conn.execute("select service_config_json from services where service_id=?",
                                        (result.service_id,)).fetchone()[0])
        assert stored["smoke_test"] and not stored["update_signature_enabled"]
        assert not stored["enable_perturbation_metrics"] and not stored["explainability_enabled"]


@pytest.mark.parametrize("profile", [None, "benchmark"])
def test_manifest_can_disable_scoring_without_disabling_signatures(debug_training, monkeypatch, tmp_path, profile):
    cfg, model = debug_training
    cfg.update(enable_perturbation_metrics=True, explainability_enabled=True,
               update_signature_enabled=True, update_signature_dir=str(tmp_path / "signatures"))
    monkeypatch.setattr(runner, "run_perturbation_stage",
                        lambda *a, **k: pytest.fail("disabled scoring stage invoked"))
    source = tmp_path / "clients.csv"
    pd.DataFrame([cfg]).to_csv(source, index=False)
    results = manifest.run_manifest(str(source), no_trust_explainability=True,
                                    metric_evaluation_profile=profile)
    result = pd.read_csv(results).iloc[0]
    assert result.status == "success", result.get("error_message")
    assert result.service_id == "client__debug_no_weights"
    assert model.fit_calls == 1
    with sqlite3.connect(cfg["db_path"]) as conn:
        document = json.loads(conn.execute(
            "SELECT document_json FROM service_execution_provenance"
        ).fetchone()[0])
    metrics = document["metrics"]
    assert metrics["metric_evaluation_disabled"]["value"] is True
    assert metrics["trust_score"]["value"] is None
    assert metrics["explainability_score"]["value"] is None
    assert metrics["accuracy"]["value"] == 0.75
    assert metrics["update_signature_available"]["value"] is True
    assert "metric_evaluation_evidence_ref" not in metrics


def test_smoke_cli_and_identity_are_idempotent():
    from mlaas_data_generator.debug_mode import apply_debug_mode
    assert cli.build_parser().parse_args(["run-manifest", "--file", "x.csv", "--smoke-test"]).smoke_test
    assert manifest.build_parser().parse_args(["--file", "x.csv", "--smoke-test"]).smoke_test
    cfg = {"service_id": "x", "smoke_test": True, "no_save_weights": True}
    for _ in range(3):
        apply_debug_mode(cfg)
    assert cfg["service_id"] == "x__debug_no_weights__smoke_test"


def test_nonfinite_generation_loss_is_persisted_as_failure(debug_training, monkeypatch):
    cfg, _ = debug_training
    original = runner._validate_service_metrics
    def validate(**kwargs):
        kwargs.update(task_family="generation", loss_value=float("nan"))
        return original(**kwargs)
    monkeypatch.setattr(runner, "_validate_service_metrics", validate)
    result = runner.execute_service(cfg)
    assert result.status == "failed" and "nonfinite_evaluation_loss" in result.error
    with sqlite3.connect(cfg["db_path"]) as conn:
        assert conn.execute("select status from services where service_id=?", (result.service_id,)).fetchone()[0] == "failed"


def test_grouped_debug_execution_reuses_model_without_durable_preparation(debug_training, monkeypatch):
    cfg, model = debug_training
    builds, resets = [], []
    monkeypatch.setattr(manifest, "_build_prepared_hf_model", lambda *args: builds.append(True) or model)
    monkeypatch.setattr(model, "set_weights", lambda weights: resets.append(True), raising=False)
    rows = [manifest._resolve_row(pd.Series({**cfg, "service_id": f"group_{i}"}), {}) for i in range(2)]
    entries = [manifest.ManifestEntry(idx=i, ordinal=i + 1, resolved=row, validation=manifest.RowValidation(True))
               for i, row in enumerate(rows)]
    results = manifest._execute_hf_model_group(entries)
    assert len(results) == 2 and all(r["status"] == "success" for r in results)
    assert len(builds) == 1 and len(resets) == 2 and model.fit_calls == 2


def test_grouped_execution_releases_arrays_before_loading_next_dataset(debug_training, monkeypatch):
    import gc
    import weakref
    from types import SimpleNamespace

    cfg, model = debug_training
    arrays = []

    def load(*args, **kwargs):
        gc.collect()
        assert all(ref() is None for ref in arrays), "Previous dataset still retained"
        train = np.zeros((8, 4), dtype="int32")
        test = np.zeros((2, 4), dtype="int32")
        arrays.extend([weakref.ref(train), weakref.ref(test)])
        return (train, train), (test, test), {"input_shape": (4,), "num_classes": 2}

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run(self):
            return SimpleNamespace(status="success", error="", service_id=self.config["service_id"])

    monkeypatch.setattr(runner, "_load_dataset", load)
    monkeypatch.setattr(runner, "ServiceRunner", FakeRunner)
    monkeypatch.setattr(manifest, "_build_prepared_hf_model", lambda *args: model)
    monkeypatch.setattr(model, "set_weights", lambda weights: None, raising=False)
    rows = [manifest._resolve_row(pd.Series({**cfg, "service_id": f"release_{i}",
                                           "dataset_name": f"dataset_{i}"}), {}) for i in range(2)]
    entries = [manifest.ManifestEntry(idx=i, ordinal=i + 1, resolved=row,
                                     validation=manifest.RowValidation(True)) for i, row in enumerate(rows)]
    results = manifest._execute_hf_model_group(entries)
    assert len(results) == 2 and all(r["status"] == "success" for r in results)
    assert len(arrays) == 4
    assert all(ref() is None for ref in arrays)


@pytest.mark.parametrize("grouped", [False, True])
def test_client_quantity_survives_hf_loading_and_stale_source_caps(debug_training, monkeypatch, tmp_path, grouped):
    import datasets
    from mlaas_data_generator.data import master_loader
    cfg, model = debug_training
    monkeypatch.setattr(model, "predict", lambda x, **kw: np.tile([[0.8, 0.2]], (len(x), 1)))
    raw = {split: datasets.Dataset.from_dict({"text": [str(i) for i in range(n)],
               "label": [i % 2 for i in range(n)]}) for split, n in (("train", 3000), ("test", 800))}
    monkeypatch.setattr(datasets, "load_dataset", lambda name, config, *, split, **kw: raw[split])
    seen = []
    def preprocess(train, test, meta, **kwargs):
        seen.append((len(train[0]), len(test[0])))
        converted = [(np.asarray(list(map(float, ds["text"])))[:, None], np.asarray(ds["label"]))
                     for ds, _ in (train, test)]
        return *converted, {**meta, "input_shape": (1,), "num_classes": 2, "task_type": "classification"}
    monkeypatch.setitem(master_loader.PREPROCESSOR_REGISTRY, "hf", preprocess)
    monkeypatch.setattr(runner, "load_dataset", master_loader.load_dataset)
    monkeypatch.setattr(manifest, "_build_prepared_hf_model", lambda *args: model)
    monkeypatch.setattr(model, "set_weights", lambda weights: None, raising=False)
    configs = []
    for quantity in (512, 2048):
        row = {**cfg, "service_id": f"quantity_{quantity}", "dataset": "hf", "dataset_name": "fixture/text",
               "hf_task": "sequence_classification", "hf_model_id": "fixture/model", "modality": "text",
               "sample_size": quantity, "max_samples": 200, "benchmark_sample_size": 384,
               "requested_quota": 200, "max_client_quantity": 200}
        resolved = manifest._resolve_row(pd.Series(row), {})
        assert "max_samples" not in resolved and "max_samples" not in resolved["dataset_args"]
        configs.append(resolved)
    if grouped:
        entries = [manifest.ManifestEntry(idx=i, ordinal=i + 1, resolved=c, validation=manifest.RowValidation(True))
                   for i, c in enumerate(configs)]
        results = manifest._execute_hf_model_group(entries)
        assert all(r["status"] == "success" for r in results), results
    else:
        for c in configs:
            result = runner.execute_service(c)
            assert result.status == "success", result.error
    assert set(seen) == {(3000, 384)}
    assert model.fit_calls == 2
    with sqlite3.connect(cfg["db_path"]) as db:
        for quantity in (512, 2048):
            metrics = dict(db.execute("SELECT metric_name, COALESCE(value_int,value_num) FROM service_metrics "
                                     "WHERE service_id=? AND metric_name IN ('train_set_size','benchmark_set_size')",
                                     (f"quantity_{quantity}__debug_no_weights",)))
            assert metrics == {"train_set_size": quantity, "benchmark_set_size": 384}


def test_benchmark_does_not_inherit_client_training_quantity():
    x, y = np.arange(800)[:, None], np.arange(800) % 2
    xb, yb, info = runner._cap_benchmark_split(x, y, config={"sample_size": 2048, "max_samples": 200})
    assert len(xb) == len(yb) == info["effective_sample_size_total"] == 800
