import json
import sqlite3
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from mlaas_data_generator.cli.manifest import preparation
from mlaas_data_generator.data import master_loader
from mlaas_data_generator.data.tokenization_selection import training_row_selection
from mlaas_data_generator.client_reconstruction import persist_recipe, read_recipe
from tests.generator.test_service_runner import DummyModel


@pytest.fixture
def raw_family(tmp_path, monkeypatch):
    x = np.arange(800, dtype="float32").reshape(200, 4)
    y = np.arange(200, dtype="int32") % 2
    meta = {"input_shape": (4,), "task_type": "classification", "num_classes": 2}
    monkeypatch.setattr(master_loader, "load_raw_dataset", lambda *a, **k: ((x, y), (x[:20], y[:20]), meta))
    cfg = {"service_id": "a", "dataset": "digits", "model_type": "mlp", "training_regime": "generic",
           "task_type": "classification", "sample_size": 8, "sample_seed": 42, "seed": 42,
           "provenance_artifact_dir": str(tmp_path), "db_path": str(tmp_path / "services.db")}
    model = DummyModel()
    model.set_weights = lambda weights: None
    return cfg, model, x


def test_family_reuses_fitted_state_and_identity_for_different_clients(raw_family, monkeypatch):
    cfg, model, x = raw_family
    sizes = []
    original = master_loader.PREPROCESSOR_REGISTRY["tabular_scaling"]

    def prepare(tr, te, meta, **kwargs):
        sizes.append(len(tr[1]))
        return original(tr, te, meta, **kwargs)

    monkeypatch.setitem(master_loader.PREPROCESSOR_REGISTRY, "tabular_scaling", prepare)
    doc, ref, _, first, _, _ = preparation.prepare(cfg, build_model=lambda *a: model)
    second_cfg = {**cfg, "service_id": "b", "sample_size": 11, "sample_seed": 78}
    second, ref2, _, reloaded, _, _ = preparation.prepare(second_cfg, build_model=lambda *a: model)
    assert sizes == [8, 11]
    assert ref == ref2 and doc["family_id"] == second["family_id"]
    assert doc["shared_fitted_state"] == second["shared_fitted_state"]
    assert "prepared_data_hash" not in doc and "client_selection_configs" not in doc["config"]
    assert doc["reservation"]["training_ids"] == []
    np.testing.assert_array_equal(first[1][0], reloaded[1][0])  # Frozen scaler/evaluation.
    x[0, 0] += 1
    with pytest.raises(ValueError, match="raw source differs"):
        preparation.prepare(second_cfg, build_model=lambda *a: model)
    assert sizes == [8, 11]  # Reject source changes before preparation.


def test_prepare_reuses_relocated_frozen_family(raw_family, tmp_path):
    from mlaas_data_generator.artifact_paths import artifact_path_scope
    cfg, model, _ = raw_family
    original, relocated = tmp_path / "original", tmp_path / "relocated"
    cfg = {**cfg, "provenance_artifact_dir": str(original)}
    doc, ref, _, _, initial, _ = preparation.prepare(cfg, build_model=lambda *a: model)
    saved = json.loads(Path(ref["document_ref"]).read_text(encoding="utf-8"))
    before = {p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()}
    original.rename(relocated)
    with artifact_path_scope(((str(original).replace("\\", "/"), str(relocated)),)):
        loaded, ref2, _, _, restored, _ = preparation.prepare(
            {**cfg, "_family_reference": ref}, build_model=lambda *a: model)
    assert loaded == saved and ref2 == ref and restored.sha256 == initial.sha256
    assert not original.exists()
    assert before == {p.relative_to(relocated): p.read_bytes() for p in relocated.rglob("*") if p.is_file()}


def test_recorded_rows_bypass_sampler_and_preserve_order(raw_family, monkeypatch):
    cfg, _, x = raw_family
    monkeypatch.setattr("mlaas_data_generator.clients.partitioning.client_partition",
                        lambda *a, **k: pytest.fail("reconstruction must not resample"))
    config = {**cfg, "_recorded_source_ids": ["train:43", "train:9", "train:100"]}
    with training_row_selection([config]) as selection:
        train, _, _ = master_loader.load_dataset("digits", scaler=None)
    split = selection.materialize("a", train)
    np.testing.assert_array_equal(split["x_train"], x[[43, 9, 100]])
    assert split["resolved"]["partition_source_ids"] == config["_recorded_source_ids"]
    config["_recorded_source_ids"] = ["train:999"]
    with pytest.raises(ValueError, match="missing or duplicated"), training_row_selection([config]):
        master_loader.load_dataset("digits")


@pytest.mark.parametrize("cap", [1024, 19])
def test_family_freezes_capped_evaluation_for_all_clients(raw_family, monkeypatch, cap):
    cfg, model, _ = raw_family
    rows = np.arange(10000, dtype="float32").reshape(2500, 4)
    labels = np.arange(2500, dtype="int32") % 2
    def load(*args, **kwargs):
        assert kwargs["_defer_evaluation_sampling"] is True
        return (rows, labels), (rows, labels), {"input_shape": (4,), "task_type": "classification", "num_classes": 2}
    monkeypatch.setattr(master_loader, "load_raw_dataset", load)
    cfg = {**cfg, "benchmark_sample_size": cap, "evaluation_seed": 7}
    first, _, _, prepared, _, _ = preparation.prepare(cfg, build_model=lambda *a: model)
    second, *_ = preparation.prepare({**cfg, "service_id": "b", "sample_seed": 78, "sample_size": 13},
                                      build_model=lambda *a: model)
    assert len(prepared[1][1]) == 2500  # Complete source retained for replay/audit.
    assert len(first["reservation"]["catalogue_metric_ids"]) == cap
    assert len(first["reservation"]["audit_ids"]) == 512
    assert first["reservation"] == second["reservation"]
    assert first["family_id"] == second["family_id"]


def test_legacy_evaluation_is_only_reopened_for_reconstruction(raw_family, monkeypatch):
    cfg, model, _ = raw_family
    doc, ref, *_ = preparation.prepare(cfg, build_model=lambda *a: model)
    doc["version"] = "raw_source_family_v3"
    monkeypatch.setattr(preparation, "read_preparation", lambda *a, **kw: doc)
    bound = {**cfg, "_family_reference": ref}
    with pytest.raises(ValueError, match="predates the evaluation cap"):
        preparation.prepare(bound, build_model=lambda *a: model)
    loaded, *_ = preparation.prepare(bound, build_model=lambda *a: model, allow_legacy_evaluation=True)
    assert loaded["reservation"] == doc["reservation"]


def test_recipe_records_effective_settings_memory_and_integrity(raw_family):
    cfg, model, _ = raw_family
    doc, ref, *_ = preparation.prepare(cfg, build_model=lambda *a: model)
    cfg.update(_family_document=doc, family_document_ref=ref["document_ref"], family_document_hash=ref["document_hash"])
    selected = {"partition_source_ids": ["train:7", "train:4"], "effective_train_samples": 2,
                "requested_sample_size_total": 8, "sample_seed": 42, "effective_strategy": "iid"}
    memory = {"oom_retry_count": 1, "micro_batch_size_effective": 1, "gradient_accumulation_steps_effective": 4}
    service = SimpleNamespace(config=cfg, service_id="a", _requested_configuration={"training_epochs": 1, "max_new_tokens": 16},
        _reconstruction_evaluation_settings={"generation_config": {"max_new_tokens": 64, "do_sample": False}},
        _effective_training_configuration={"training_epochs": 7, "batch_size": 4, "training_memory_metadata": memory},
        _reconstruction_training_observations={**memory, "completed_training_epochs": 3, "optimizer_step_count": 6,
            "logical_batch_size": 2, "precision_type_effective": "fp32", "mixed_precision_effective": False,
            "train_stopped_early": True, "train_stopping_reason": "resource_limit"},
        _adapter_fit_arguments={"epochs": 7},
        _reconstruction_inputs={"selection": selected, "prepared_train_hash": "train", "prepared_evaluation_hash": "eval"})
    metrics = {"accuracy": {"value": 0.8, "domain": "quality"}}
    artifact = persist_recipe(service, metrics, {"final_model_hash": "final"})
    recipe = json.loads(Path(artifact["artifact_uri"]).read_text(encoding="utf-8"))
    assert recipe["training"]["effective_settings"]["training_epochs"] == 7
    assert recipe["training"]["effective_settings"]["max_new_tokens"] == 64
    assert recipe["training"]["requested_settings"]["max_new_tokens"] == 16
    assert recipe["evaluation"]["effective_settings"]["generation_config"]["max_new_tokens"] == 64
    assert recipe["training"]["effective_settings"]["batch_size"] == 2
    assert recipe["training"]["effective_settings"]["precision_type"] == "fp32"
    assert recipe["training"]["requested_settings"]["training_epochs"] == 1
    assert recipe["training"]["runtime_execution"]["micro_batch_size_effective"] == 1
    assert recipe["training"]["stopping"]["optimizer_step_count"] == 6
    assert not recipe["training"]["validation_replay_supported"]
    assert recipe["observed_metrics"] == metrics
    assert recipe["selection"]["ordered_source_row_ids"] == ["train:7", "train:4"]
    with sqlite3.connect(cfg["db_path"]) as conn:
        conn.execute("CREATE TABLE service_generation_provenance(service_id, client_recipe_ref, client_recipe_hash, family_id, family_document_hash, initial_state_hash, final_model_hash)")
        conn.execute("INSERT INTO service_generation_provenance VALUES(?,?,?,?,?,?,?)", (
            "a", artifact["artifact_uri"], artifact["metadata"]["sha256"], doc["family_id"], ref["document_hash"], doc["initial_state_hash"], "final"))
    assert read_recipe(cfg["db_path"], "a") == recipe
    with sqlite3.connect(cfg["db_path"]) as conn:
        conn.execute("UPDATE service_generation_provenance SET client_recipe_hash='changed'")
    with pytest.raises(ValueError, match="integrity"):
        read_recipe(cfg["db_path"], "a")


def test_dynamic_padding_width_is_not_family_preprocessing_identity():
    from mlaas_data_generator.provenance import preprocessing_metadata
    from mlaas_data_generator.data.source_contract import prepared_fingerprint
    meta = {"dynamic_padding": True, "input_shape": (2,), "ignore_index": -100}
    assert preprocessing_metadata(meta) == preprocessing_metadata({**meta, "input_shape": (8,)})
    short = ({"input_ids": np.array([[3, 4]]), "attention_mask": np.array([[1, 1]])}, np.array([[9, 8]]))
    padded = ({"input_ids": np.array([[3, 4, 0]]), "attention_mask": np.array([[1, 1, 0]])}, np.array([[9, 8, -100]]))
    assert prepared_fingerprint(short, meta) == prepared_fingerprint(padded, meta)
    assert prepared_fingerprint((short[0], np.array([[-100, 8]])), meta) != prepared_fingerprint((short[0], np.array([[8, -100]])), meta)


def test_family_lookup_uses_resolved_hf_task_semantics():
    cfg = {"dataset": "hf", "model_type": "hf_finetune", "hf_task": "text_generation", "task_type": "generation"}
    loaded = {**cfg, "task_type": "classification"}
    assert preparation.group_key(cfg) == preparation.group_key(loaded)
    assert preparation.contract_config(loaded)["task_type"] == "generation"


def test_masking_survives_regrouping_and_dynamic_padding():
    from mlaas_data_generator.data.preprocessors.hf_text_fill_mask import _source_mlm_labels
    tokenizer = SimpleNamespace(mask_token_id=99, vocab_size=100)
    enc = {"input_ids": np.array([[5, 6, 7, 0], [10, 11, 0, 0]]),
           "attention_mask": np.array([[1, 1, 1, 0], [1, 1, 0, 0]])}
    together = _source_mlm_labels(enc, tokenizer, 0.9, -100, seed=42, source_ids=["train:7", "train:4"])
    alone = _source_mlm_labels({k: v[1:, :2] for k, v in enc.items()}, tokenizer, 0.9, -100,
                              seed=42, source_ids=["train:4"])
    np.testing.assert_array_equal(together[0]["input_ids"][1, :2], alone[0]["input_ids"][0])
    np.testing.assert_array_equal(together[1][1, :2], alone[1][0])
