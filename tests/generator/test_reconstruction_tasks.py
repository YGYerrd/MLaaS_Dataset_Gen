"""The small shared reconstruction gate and task-specific semantics."""
import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlaas_data_generator import client_reconstruction as replay
from mlaas_data_generator.provenance import sha256_file, sha256_json, write_json_artifact
from mlaas_data_generator.task_capabilities import resolve_capabilities, reconstruction_report_capabilities
from mlaas_data_generator.tasks import resolve_task_contract
from mlaas_data_generator.tasks.runtime import evaluate_task_model
from tests.generator.test_task_runtime import Tokenizer
from tests.generator.test_weight_retention import trained_recipe


@pytest.mark.parametrize("task", ["sentence_similarity", "token_classification", "seq2seq_generation",
                                  "image_segmentation", "fill_mask", "causal_lm_generation"])
def test_reconstruction_capabilities_do_not_require_client_weights(task):
    cfg = {"hf_task": task, "model_type": "hf_finetune", "task_type": "classification",
           "task_tag": "summarization", "training_regime": "finetune_transfer"}
    checks = {"checkpoint_present": False, "family_integrity": True, "recipe_integrity": True,
              "replay_supported": True}
    decisions = resolve_capabilities(cfg, checks=checks)
    for key in ("recipe_persistence", "materialization"):
        assert decisions[key]["support"] == "supported"
        assert decisions[key]["decision"] == "eligible"
    assert decisions["reconstruction_validation"]["decision"] == "pending"
    assert resolve_capabilities(cfg, checks={**checks, "recipe_integrity": False})["materialization"]["decision"] == "ineligible"
    assert resolve_capabilities({**cfg, "generation_mode": "shared_parent"})["materialization"]["support"] == "unsupported"
    assert resolve_capabilities({**cfg, "no_save_weights": True})["recipe_persistence"]["support"] == "unsupported"
    if task == "fill_mask":
        assert decisions["measured_composition_validation"]["support"] == "unsupported"


def test_token_family_names_roundtrip():
    from mlaas_data_generator.cli.manifest.preparation import contract_config, group_key
    from mlaas_data_generator.services.taxonomy import canonical_task_family
    for task in ("token_classification", "fill_mask"):
        cfg = {"hf_task": task, "task_type": "classification", "model_type": "hf_finetune"}
        frozen = contract_config(cfg)
        assert canonical_task_family(frozen["task_type"], task) == task
        assert group_key(cfg) == group_key(frozen)


def masked_fixture(batch_size):
    from mlaas_data_generator.models.adapters.hf_task import FillMaskSpec
    from mlaas_data_generator.models.adapters.hf_core import HFCore
    class MaskedModel(torch.nn.Module):
        config = SimpleNamespace()
        def forward(self, input_ids, attention_mask):
            # Always predicts token 1. Unsupervised positions must not affect metrics.
            return SimpleNamespace(logits=torch.nn.functional.one_hot(torch.ones_like(input_ids), 10).float())
    core = SimpleNamespace(model=MaskedModel(), tokenizer=Tokenizer(), torch=torch, task_spec=FillMaskSpec(),
        max_length=4, label_pad_value=-100, batch_size=batch_size, device="cpu",
        _run_with_precision_context=lambda fn: fn(), _extract_logits=lambda out: out.logits,
        _device_type=lambda: "cpu", _batch_item_count=HFCore._batch_item_count)
    core._batch_iter = lambda x, y: (({k: v[i:i + batch_size] for k, v in x.items()}, y[i:i + batch_size])
                                     for i in range(0, len(y), batch_size))
    model = SimpleNamespace(core=core)
    cfg = {"model_type": "hf_finetune", "hf_task": "fill_mask", "task_type": "classification", "mlm_probability": .3}
    contract = resolve_task_contract(cfg, model=model)
    x = {"input_ids": np.array([[2, 3, 0, 0]] * 3), "attention_mask": np.array([[1, 1, 0, 0]] * 3)}
    y = np.array([[1, -100, -100, -100], [1, 2, -100, -100], [-100, 2, -100, -100]])
    return model, contract, x, y


def test_masked_metrics_compare_only_supervised_tokens_across_batches():
    results = []
    for batch_size in (1, 2, 3):
        model, contract, x, y = masked_fixture(batch_size)
        result = evaluate_task_model(model, x, y, contract, ["a", "b", "c"], collect=True)
        assert result.metrics["masked_accuracy"] == .5
        assert result.metrics["supervision_count"] == 4
        assert result.metrics["perplexity_proxy"] == pytest.approx(np.exp(result.metrics["loss"]))
        results.append(result.metrics)
        run = {"model": model, "service": SimpleNamespace(config={"task_contract": contract.as_dict()}),
               "task": "fill_mask", "benchmark": (x, y), "recipe": {"seeds": {"evaluation": 42}},
               "family": {"reservation": {"catalogue_metric_ids": ["a", "b", "c"]}}}
        metrics, predictions = replay._evaluate_materialized(run)
        assert [len(row) for row in predictions] == [1, 2, 1]
        assert metrics["masked_accuracy"] == .5
        assert replay._prediction_comparison(predictions, predictions, "fill_mask")["count"] == 4
    assert len({r["prediction_hash"] for r in results}) == 1
    assert max(r["loss"] for r in results) == pytest.approx(min(r["loss"] for r in results))


def test_mask_contract_freezes_probability_seed_and_special_tokens():
    model, contract, _, _ = masked_fixture(2)
    cfg = {"model_type": "hf_finetune", "hf_task": "fill_mask", "task_type": "classification", "mlm_probability": .3}
    assert resolve_task_contract({**cfg, "mlm_probability": .4}, model=model).semantics_hash != contract.semantics_hash
    assert resolve_task_contract(cfg, {"masking_seed": 73}, model).semantics_hash != contract.semantics_hash
    model.core.tokenizer.special_tokens_map = {"mask_token": "new-mask"}
    assert resolve_task_contract(cfg, model=model).semantics_hash != contract.semantics_hash


def test_sequence_filter_preserves_source_positions_for_replacement():
    from mlaas_data_generator.data.tokenization_selection import filter_token_supervision
    x = {"input_ids": np.arange(12).reshape(4, 3)}
    y = np.array([[-100] * 3, [1, -100, -100], [-100] * 3, [2, 1, -100]])
    train, test, meta = filter_token_supervision((x, y), (x, y), {
        "selected_train_rows": [9, 8, 7, 6], "selected_test_rows": [19, 18, 17, 16]})
    assert meta["accepted_train_positions"] == [1, 3]
    assert meta["selected_train_rows"] == [8, 6] and meta["selected_test_rows"] == [18, 16]
    np.testing.assert_array_equal(train[0]["input_ids"], x["input_ids"][[1, 3]])
    with pytest.raises(ValueError, match="no samples survived preprocessing for split='train'"):
        filter_token_supervision((x, np.full_like(y, -100)), (x, y), {})


def test_masking_excludes_special_and_padding_positions():
    from mlaas_data_generator.data.preprocessors.hf_text_fill_mask import _source_mlm_labels
    tokenizer = SimpleNamespace(mask_token_id=9, vocab_size=10)
    enc = {"input_ids": np.array([[8, 1, 2, 0]]), "attention_mask": np.array([[1, 1, 1, 0]]),
           "special_tokens_mask": np.array([[1, 0, 0, 1]])}
    x, y = _source_mlm_labels(enc, tokenizer, .99, -100, seed=42, source_ids=["train:7"])
    assert y[0, 0] == y[0, 3] == -100
    assert x["input_ids"][0, 0] == 8 and x["input_ids"][0, 3] == 0


def test_replay_detects_adapter_effective_setting_drift(trained_recipe, monkeypatch):
    from mlaas_data_generator.services.runner import ServiceRunner
    original = ServiceRunner._train_model_effective
    def changed(self, *args, **kwargs):
        observed = original(self, *args, **kwargs)
        observed["training_epochs"] += 1
        return observed
    monkeypatch.setattr(ServiceRunner, "_train_model_effective", changed)
    with pytest.raises(ValueError, match="effective training settings differ"):
        with replay.materialize_service(trained_recipe["db_path"], "recipe"):
            pytest.fail("changed effective recipe was accepted")


@pytest.mark.parametrize("corruption", ["recipe_bytes", "source_evidence"])
def test_corrupt_recipe_or_source_fails_before_replay(trained_recipe, tmp_path, monkeypatch, corruption):
    db = tmp_path / "services.db"
    shutil.copyfile(trained_recipe["db_path"], db)
    recipe = replay.read_recipe(db, "recipe")
    path = tmp_path / "recipe.json"
    recipe["fingerprints"]["source"] = "corrupt"
    recipe["recipe_hash"] = sha256_json({k: v for k, v in recipe.items() if k != "recipe_hash"})
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE service_generation_provenance SET client_recipe_ref=?,client_recipe_hash=?",
                     (str(path), sha256_file(path) if corruption == "source_evidence" else "corrupt"))
    monkeypatch.setattr("mlaas_data_generator.cli.manifest.preparation.prepare", lambda *a, **k: pytest.fail("prepared corrupt input"))
    with pytest.raises(ValueError, match="integrity|source disagrees"):
        with replay.materialize_service(db, "recipe"):
            pytest.fail("corrupt evidence accepted")


def test_scoped_report_does_not_promote_an_unmeasured_task_or_changed_bytes(tmp_path):
    report = {"version": "client_reconstruction_comparison_v3", "service_id": "s", "family_id": "family-a",
        "recipe_hash": "recipe-a", "task": "fill_mask", "runtime": {"versions": {"torch": "test"}},
        "scope": {"config": {"model_type": "hf_finetune", "hf_task": "fill_mask", "task_type": "classification"}, "device": "cpu"},
        "reconstructed_metrics": {"masked_accuracy": .5, "loss": .5},
        "reference_checkpoint_metrics": {"masked_accuracy": .5, "loss": .5},
        "training_loss": {"original": .4, "reconstructed": .4}, "predictions": {"agreement": 1},
        "training_changed_initial_state": True,
        "replay_evidence": {"fresh_process": True, "client_checkpoint_reads": 0, "selected_rows_match": True, "effective_settings_match": True}}
    report["gate"] = replay.fidelity_gate(report)
    def save(value):
        return write_json_artifact(tmp_path, "s", "client_reconstruction_comparison", value)["artifact_uri"]
    path = save(report)
    result = reconstruction_report_capabilities(path)
    assert result["capabilities"]["reconstruction_validation"]["decision"] == "eligible"
    assert result["family_id"] == "family-a" and result["scope"]["device"] == "cpu"
    report["replay_evidence"]["fresh_process"] = False
    assert reconstruction_report_capabilities(save(report))["capabilities"]["reconstruction_validation"]["decision"] == "ineligible"
    Path(path).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="content-address"):
        reconstruction_report_capabilities(path)
