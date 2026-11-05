import copy
import json
import sqlite3

import pandas as pd
import pytest

from mlaas_data_generator import client_reconstruction as replay
from mlaas_data_generator.cli import main, run_manifest
from mlaas_data_generator.cli.manifest import preparation
from mlaas_data_generator.config import CONFIG


@pytest.mark.parametrize("attempted, skipped, allowed", [
    (512, 0, True), (512, 3, True), (512, 25, True), (512, 26, True),
    (66, 4, True), (20, 2, True), (15, 2, True),
    (512, 102, True), (512, 103, False), (100, 20, True), (100, 21, False),
    (5, 1, True), (4, 1, False), (1, 0, True), (1, 1, False),
    (0, 0, False), (512, -1, False), (512, 513, False),
])
def test_replay_allows_at_most_twenty_percent_skipped_attempts(attempted, skipped, allowed):
    observed = {"train_stopping_reason": "epochs_completed", "optimizer_step_count": attempted,
                "successful_optimizer_step_count": attempted - skipped, "skipped_optimizer_step_count": skipped}
    assert replay.replay_supported(observed) is allowed
    assert replay.replay_supported({"train_stopping_reason": "epochs_completed"}, observed) is allowed


@pytest.mark.parametrize("extra", [
    {"train_stopped_early": True}, {"train_stopping_reason": "resource_limit"},
    {"oom_retry_count": 1}, {"train_fp32_loss_retry_count": 1},
    {"successful_optimizer_step_count": 0}, {"successful_optimizer_step_count": 512},
    {"optimizer_step_count": None},
])
def test_skip_allowance_keeps_completion_and_recovery_checks(extra):
    observed = {"train_stopping_reason": "epochs_completed", "optimizer_step_count": 512,
                "successful_optimizer_step_count": 444, "skipped_optimizer_step_count": 68}
    assert not replay.replay_supported({**observed, **extra})


def test_skip_allowance_preserves_adapters_without_amp_counts():
    assert replay.replay_supported({"train_stopping_reason": "epochs_completed", "optimizer_step_count": 20})
    assert not replay.replay_supported({"train_stopping_reason": "epochs_completed", "skipped_optimizer_step_count": 1})


@pytest.mark.parametrize("extra", [
    {"weight_retention": "invalid"}, {"no_save_weights": True},
    {"generation_mode": "shared_parent"}, {"fl_run_id": "x"},
    {"training_regime": "inference_only"}, {"model_type": "randomforest"},
    {"retain_validation_checkpoint": "typo"},
])
def test_invalid_retention_rejected(extra):
    cfg = {"weight_retention": "recipe_only", "model_type": "mlp", "training_regime": "generic", **extra}
    assert replay.retention_error(cfg)
    assert not run_manifest._validate_row(cfg).ok


@pytest.mark.parametrize("policy", ["recipe_only", "checkpoint"])
def test_retention_cli_and_row_resolution(policy):
    for parser, prefix in ((main.build_parser(), ["run-manifest"]), (run_manifest.build_parser(), [])):
        args = parser.parse_args(prefix + ["--file", "x.csv", "--weight-retention", policy,
                                          "--no-retain-validation-checkpoint"])
        assert args.weight_retention == policy and args.retain_validation_checkpoint is False
    cfg = run_manifest._resolve_row(pd.Series({"service_id": "normal", "weight_retention": "recipe_only",
                                               "retain_validation_checkpoint": "false"}), {})
    assert cfg["service_id"] == "normal" and not replay.retention_settings(cfg)["client_checkpoint_retained"]
    assert not replay.retention_settings({"weight_retention": "recipe_only"})["client_checkpoint_retained"]


@pytest.mark.parametrize("model_type", ["mlp", "cnn", "logreg", "hf_finetune", "hf_train", "transformers_finetune"])
def test_supported_independent_defaults_to_recipe_only(model_type):
    from mlaas_data_generator.services.runner import ServiceRunner
    row = {"service_id": "normal", "model_type": model_type}
    cfg = run_manifest._resolve_row(pd.Series(row), {})
    assert cfg["weight_retention"] == ServiceRunner(row).config["weight_retention"] == "recipe_only"
    assert not cfg["retain_validation_checkpoint"]
    assert replay.retention_error(cfg) is None
    assert not replay.retention_settings(cfg)["client_checkpoint_retained"]


@pytest.mark.parametrize("extra", [{"model_type": "randomforest"}, {"training_regime": "inference_only"},
    {"training_regime": "inference"}, {"generation_mode": "shared_parent"},
    {"generation_mode": "fedavg"}, {"fl_run_id": "federated"}, {"no_save_weights": True}])
def test_other_modes_keep_checkpoint_default(extra):
    cfg = run_manifest._resolve_row(pd.Series({"service_id": "other", "model_type": "mlp", **extra}), {})
    assert cfg["weight_retention"] == "checkpoint"
    assert replay.retention_error(cfg) is None


def test_explicit_retention_overrides_and_family_identity():
    row = pd.Series({"service_id": "normal", "model_type": "mlp"})
    normal = run_manifest._resolve_row(row, {})
    checkpoint = run_manifest._resolve_row(row, {"weight_retention": "checkpoint"})
    validation = run_manifest._resolve_row(row, {"retain_validation_checkpoint": True})
    assert checkpoint["weight_retention"] == "checkpoint"
    assert validation["weight_retention"] == "recipe_only"
    assert replay.retention_settings(checkpoint)["client_checkpoint_retained"]
    assert replay.retention_settings(validation)["client_checkpoint_retained"]
    assert preparation.group_key(normal) == preparation.group_key(checkpoint) == preparation.group_key(validation)


def test_unprepared_recipe_execution_fails_before_training(tmp_path, monkeypatch):
    from mlaas_data_generator.services.runner import ServiceRunner
    service = ServiceRunner({"model_type": "mlp", "service_id": "unprepared", "db_path": str(tmp_path / "services.db")})
    monkeypatch.setattr(service, "_execute_service", lambda **kw: pytest.fail("unprepared client trained"))
    result = service.run()
    assert result.status == "failed" and "requires durable preparation" in result.error


@pytest.fixture(scope="module")
def trained_recipe(tmp_path_factory):
    root = tmp_path_factory.mktemp("true-recipe-only")
    config = {**CONFIG, "service_id": "recipe", "dataset": "iris", "model_type": "mlp",
        "training_regime": "generic", "task_type": "classification", "hidden_layers": [8],
        "dropout": 0.1, "training_epochs": 3, "batch_size": 16, "sample_size": 64,
        "sample_seed": 47, "training_seed": 53, "device": "cpu", "max_train_time_s": None,
        "measure_system_metrics": False, "update_signature_enabled": False,
        # This fixture exercises weight retention/replay, not metric inference.
        "enable_perturbation_metrics": False, "explainability_enabled": False,
        "checkpoint_weight_format": "native", "db_path": str(root / "services.db"),
        "provenance_artifact_dir": str(root / "artifacts")}
    entry = run_manifest.ManifestEntry(0, 1, config, run_manifest.RowValidation(True))
    from unittest.mock import patch
    from mlaas_data_generator.services.runner import execute_service
    from mlaas_data_generator.checkpoint_store import capture_model_tensor_state
    seen = []
    def execute(row):
        result = execute_service(row)
        seen.append((row, result))
        return result
    with patch("mlaas_data_generator.services.runner.persist_service_patch", side_effect=AssertionError("client weight write")):
        results = preparation.execute_entries([entry], execute_service=execute)
    assert results[0]["status"] == "success", results
    assert seen[0][1].final_tensor_state is None and not seen[0][1].weights_saved
    assert capture_model_tensor_state(seen[0][0]["_prepared_model"]).sha256 == seen[0][0]["initial_state_hash"]
    return config


def test_recipe_only_persists_normal_completed_service_without_client_weights(trained_recipe):
    cfg = trained_recipe
    with sqlite3.connect(cfg["db_path"]) as conn:
        assert conn.execute("SELECT service_id,status FROM services").fetchone() == ("recipe", "completed")
        assert conn.execute("SELECT count(*) FROM service_checkpoint_storage").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM lineage_checkpoint_storage").fetchone()[0] == 1
        # Replayable classification recipes are eligible without client checkpoints.
        assert conn.execute("SELECT composition_compatible FROM service_generation_provenance").fetchone()[0] == 1
        stored = json.loads(conn.execute("SELECT service_config_json FROM services").fetchone()[0])
        assert stored["weight_retention"] == "recipe_only" and not stored["retain_validation_checkpoint"]
    recipe = replay.read_recipe(cfg["db_path"], "recipe")
    assert recipe["retention"] == {"policy": "recipe_only", "client_checkpoint_retained": False, "role": "recipe_and_observations"}
    assert recipe["training"]["observations"]["train_loss"] > 0
    preparation.verify_completed(cfg["db_path"], "recipe")


def test_materialization_never_reads_client_checkpoint_or_writes_originals(trained_recipe, monkeypatch):
    from mlaas_data_generator.checkpoint_store import capture_model_tensor_state
    cfg = trained_recipe
    def forbidden(*a, **k):
        pytest.fail("materialization read a client checkpoint")
    monkeypatch.setattr("mlaas_data_generator.checkpoint_store.reconstruct_service_state", forbidden)
    monkeypatch.setattr(preparation, "reconstruct_service_state", forbidden)
    with sqlite3.connect(cfg["db_path"]) as conn:
        before = conn.execute("SELECT * FROM service_metrics").fetchall()
    preparation.verify_completed(cfg["db_path"], "recipe")
    with replay.materialize_service(cfg["db_path"], "recipe") as run:
        assert capture_model_tensor_state(run["model"]).sha256 != run["family"]["initial_state_hash"]
        assert run["training"]["train_loss"] == pytest.approx(
            run["recipe"]["training"]["observations"]["train_loss"],
            rel=replay.DEFAULT_TOLERANCES["loss_relative"], abs=replay.DEFAULT_TOLERANCES["loss_absolute"])
    with sqlite3.connect(cfg["db_path"]) as conn:
        assert conn.execute("SELECT * FROM service_metrics").fetchall() == before


@pytest.mark.parametrize("original_skips, reconstructed_skips", [
    (3, 0), (3, 4), (25, 26), (68, 70), (102, 103), (103, 0),
])
def test_recipe_skip_allowance_through_generation_and_materialization(
        trained_recipe, tmp_path, monkeypatch, original_skips, reconstructed_skips):
    from mlaas_data_generator.services.runner import ServiceRunner, execute_service
    cfg = {**trained_recipe, "service_id": "skip_recipe", "db_path": str(tmp_path / "services.db"),
           "provenance_artifact_dir": str(tmp_path / "artifacts")}
    for key in ("family_id", "family_contract_version", "initial_state_hash", "family_document_ref", "family_document_hash"):
        cfg.pop(key, None)
    train = ServiceRunner._train_model_effective
    skipped = original_skips

    def with_amp_observations(self, *args, **kwargs):
        observed = train(self, *args, **kwargs)
        # Exercise real persistence and retraining with simulated AMP counters;
        # the small CPU fixture does not need a GPU or 512 physical updates.
        observed.update(optimizer_step_count=512, successful_optimizer_step_count=512 - skipped,
                        skipped_optimizer_step_count=skipped)
        return observed

    monkeypatch.setattr(ServiceRunner, "_train_model_effective", with_amp_observations)
    entry = run_manifest.ManifestEntry(0, 1, cfg, run_manifest.RowValidation(True))
    result = preparation.execute_entries([entry], execute_service=execute_service)[0]
    if original_skips > 102:
        assert result["status"] == "failed", result
        with sqlite3.connect(cfg["db_path"]) as conn:
            failure = json.loads(conn.execute("SELECT metadata_json FROM services").fetchone()[0])
        assert failure["failure_stage"] == "recipe_replay_support"
        assert "at most 20%" in failure["error_message"]
        return

    assert result["status"] == "success", result
    recipe = replay.read_recipe(cfg["db_path"], cfg["service_id"])
    assert recipe["training"]["validation_replay_supported"]
    assert recipe["training"]["stopping"]["skipped_optimizer_step_count"] == original_skips
    assert not recipe["retention"]["client_checkpoint_retained"]
    # An older saved eligibility flag must not override the current policy.
    recipe["training"]["validation_replay_supported"] = False
    monkeypatch.setattr(replay, "read_recipe", lambda *a, **kw: recipe)
    preparation.verify_completed(cfg["db_path"], cfg["service_id"])
    skipped = reconstructed_skips
    if reconstructed_skips > 102:
        with pytest.raises(ValueError, match="reconstructed training requires.*at most 20%"):
            with replay.materialize_service(cfg["db_path"], cfg["service_id"]):
                pytest.fail("excessive reconstructed skips accepted")
    else:
        with replay.materialize_service(cfg["db_path"], cfg["service_id"]) as run:
            assert run["training"]["successful_optimizer_step_count"] == 512 - reconstructed_skips
            assert run["replay_evidence"]["effective_settings_match"]


@pytest.mark.parametrize("overrides, retained", [({}, False), ({"weight_retention": "checkpoint"}, True),
    ({"retain_validation_checkpoint": True}, True)])
def test_manifest_retention_execution_and_resume(trained_recipe, tmp_path, monkeypatch, overrides, retained):
    cfg = {**trained_recipe, "service_id": "manifest_recipe", "db_path": str(tmp_path / "manifest.db"),
           "provenance_artifact_dir": str(tmp_path / "artifacts")}
    for key in ("weight_retention", "retain_validation_checkpoint", "family_id", "family_contract_version",
                "initial_state_hash", "family_document_ref", "family_document_hash"):
        cfg.pop(key, None)
    manifest = tmp_path / "clients.csv"
    pd.DataFrame([cfg]).to_csv(manifest, index=False)
    result = run_manifest.run_manifest(str(manifest), **overrides)
    assert pd.read_csv(result).iloc[0].status == "success"
    recipe = replay.read_recipe(cfg["db_path"], "manifest_recipe")
    assert recipe["retention"]["client_checkpoint_retained"] is retained
    assert recipe["retention"]["policy"] == overrides.get("weight_retention", "recipe_only")
    with sqlite3.connect(cfg["db_path"]) as conn:
        assert conn.execute("SELECT count(*) FROM service_checkpoint_storage").fetchone()[0] == int(retained)
        assert conn.execute("SELECT count(*) FROM lineage_checkpoint_storage").fetchone()[0] == 1
    monkeypatch.setattr(preparation, "execute_entries", lambda *a, **k: pytest.fail("resume retrained a completed recipe-only service"))
    run_manifest.run_manifest(str(manifest), **overrides)
    assert replay.read_recipe(cfg["db_path"], "manifest_recipe")["recipe_hash"] == recipe["recipe_hash"]
    if overrides.get("retain_validation_checkpoint"):
        # Simulate small numerical replay differences through the full comparison
        # path. Integrity still verifies the retained original against its recipe.
        from contextlib import contextmanager
        materialize = replay.materialize_service
        @contextmanager
        def perturbed(*args, **kwargs):
            with materialize(*args, **kwargs) as run:
                weights = run["model"].get_weights()
                weights[0].flat[0] += 1e-5
                run["model"].set_weights(weights)
                yield run
        monkeypatch.setattr(replay, "materialize_service", perturbed)
        with sqlite3.connect(cfg["db_path"]) as conn:
            before = conn.execute("SELECT * FROM service_metrics").fetchall()
        report, _ = replay.compare_reconstruction(cfg["db_path"], "manifest_recipe", output_dir=tmp_path / "comparison")
        assert not report["bitwise_equal"]
        assert report["parameters"]["relative_l2_error"] > 0
        assert report["gate"]["passed"], report["gate"]
        with sqlite3.connect(cfg["db_path"]) as conn:
            assert conn.execute("SELECT * FROM service_metrics").fetchall() == before


def test_fidelity_gate_detects_bad_quality_predictions_loss_and_no_update():
    report = {"task": "classification", "reconstructed_metrics": {"accuracy": 0.8, "loss": 0.5},
              "reference_checkpoint_metrics": {"accuracy": 0.8, "loss": 0.5},
              "training_loss": {"original": 0.4, "reconstructed": 0.4},
              "predictions": {"agreement": 1.0}, "training_changed_initial_state": True}
    assert replay.fidelity_gate(report)["passed"]
    for section, key, value, failure in (("reconstructed_metrics", "accuracy", 0.7, "accuracy"),
            ("reconstructed_metrics", "loss", float("nan"), "loss"),
            ("training_loss", "reconstructed", 0.8, "training_loss"),
            ("training_loss", "original", None, "training_loss"),
            ("predictions", "agreement", 0.9, "predictions")):
        bad = copy.deepcopy(report)
        bad[section][key] = value
        assert failure in replay.fidelity_gate(bad)["failures"]
    report["training_changed_initial_state"] = False
    assert not replay.fidelity_gate(report)["passed"]


@pytest.mark.parametrize("task", ["classification", "generation", "regression"])
def test_functional_gate_accepts_non_bitwise_reconstruction(task):
    report = {"task": task, "reconstructed_metrics": {"loss": 0.503},
        "reference_checkpoint_metrics": {"loss": 0.5},
        "training_loss": {"original": 0.4, "reconstructed": 0.402},
        "predictions": {"agreement": 0.99, "rmse": 0.005, "reference_rms": 1.0},
        "training_changed_initial_state": True, "bitwise_equal": False,
        "parameters": {"relative_l2_error": 0.1, "update_cosine_similarity": 0.95}}
    if task != "regression":
        report["reconstructed_metrics"]["accuracy" if task == "classification" else "rougeL"] = 0.79
        report["reference_checkpoint_metrics"]["accuracy" if task == "classification" else "rougeL"] = 0.8
    assert replay.fidelity_gate(report)["passed"]


@pytest.mark.parametrize("task", ["causal_lm_generation", "seq2seq_generation", "fill_mask"])
def test_reconstruction_source_accepts_official_text_supervision(monkeypatch, task):
    import datasets
    from mlaas_data_generator.data.sources.huggingface import load_huggingface_source
    source = datasets.Dataset.from_dict({"text": ["one sentence", "another sentence"],
                                        "summary": ["one", "another"]})
    calls = []
    def load(name, config, *, split, **kwargs):
        calls.append(split)
        return source
    monkeypatch.setattr(datasets, "load_dataset", load)
    _, _, meta = load_huggingface_source(dataset_name="local", hf_task=task, task="generation",
        text_column="text", test_split="validation", require_official_train_split=True)
    assert calls == ["train", "validation"] and meta["test_split"] == "validation"


def test_comparison_resume_rejects_changed_recipe_results_or_bytes(tmp_path):
    from pathlib import Path
    from scripts.validate_client_reconstruction import reusable_comparison
    from mlaas_data_generator.provenance import write_json_artifact
    report = {"version": "client_reconstruction_comparison_v3", "replay_evidence": {"fresh_process": True},
        "service_id": "s", "recipe_hash": "recipe", "reference_service_id": "reference",
        "training": {"device": "cpu"}, "gate": {"passed": True, "tolerances": replay.DEFAULT_TOLERANCES},
        "metric_deltas": {"accuracy": 0.0}}
    artifact = write_json_artifact(tmp_path, "s", "comparison", report)
    summary = {"service_id": "s", "metric_deltas": report["metric_deltas"], "report_ref": artifact["artifact_uri"]}
    recipe = {"service_id": "s", "recipe_hash": "recipe"}
    args = {"reference_id": "reference", "device": "cpu", "tolerances": {}}
    assert reusable_comparison(summary, recipe, **args)
    assert not reusable_comparison(summary, {**recipe, "recipe_hash": "changed"}, **args)
    assert not reusable_comparison(summary, recipe, **{**args, "tolerances": {"quality_absolute": 0.001}})
    assert not reusable_comparison({**summary, "metric_deltas": {"accuracy": 1.0}}, recipe, **args)
    Path(artifact["artifact_uri"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="content-address"):
        reusable_comparison(summary, recipe, **args)
