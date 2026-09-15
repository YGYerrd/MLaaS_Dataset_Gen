"""New recipes activate the adaptation without reinterpreting old checkpoints."""

import pandas as pd
import pytest
import torch

from mlaas_data_generator.clients import recipes, training
from mlaas_data_generator.cli.manifest.hf_manifest_builder import build_hf_manifest
from mlaas_data_generator.cli.manifest.preparation import contract_config, group_key
from mlaas_data_generator.cli.run_manifest import _resolve_row
from mlaas_data_generator.mobilebert_policy import apply_mobilebert_generation_defaults
from mlaas_data_generator.models.adapters.mobilebert_stability import NormalizedLinear


def test_generation_and_csv_resolution_preserve_normalization_clipping_and_fp32(tmp_path):
    frame = build_hf_manifest(task_keys=["sentence_similarity"], models_per_task=20,
                              clients_per_family=8, average_samples=600, seed=42)
    rows = frame[frame.hf_model_id == "google/mobilebert-uncased"]
    assert len(rows) == 8
    assert set(rows.mobilebert_head_normalization) == {"layer_norm_v1"}
    assert set(rows.max_grad_norm) == {1.0}
    assert not rows.mixed_precision.any() and set(rows.precision_type) == {"fp32"}
    path = tmp_path / "manifest.csv"
    from mlaas_data_generator.cli.manifest.review import save_review_manifest, expand_review_manifest
    save_review_manifest(rows, path)
    expanded, defaults = expand_review_manifest(pd.read_csv(path, dtype=str), path)
    for _, raw in expanded.iterrows():
        row = _resolve_row(raw, defaults)
        training.validate_training_recipe(row)
        assert row["max_grad_norm"] == 1.0
        assert row["training_policy"]["runtime_settings"]["mobilebert_head_normalization"] == "layer_norm_v1"
        assert row["training_policy"]["calibration"]["status"] == "uncalibrated"
        service = row["service_config"]
        assert service["mobilebert_head_normalization"] == "layer_norm_v1"
        assert service["max_grad_norm"] == 1.0
        frozen = contract_config(row)
        assert frozen["mobilebert_head_normalization"] == "layer_norm_v1"
        assert frozen["precision_type"] == "fp32" and not frozen["mixed_precision"]
        assert group_key(row) != group_key({**row, "mobilebert_head_normalization": "none"})
        for key, value in (("max_grad_norm", None), ("mobilebert_head_normalization", "none"),
                           ("precision_type", "fp16"), ("mixed_precision", True)):
            with pytest.raises(ValueError, match="frozen runtime"):
                training.validate_training_recipe({**row, key: value})


def test_tagged_parent_recipes_keep_runtime_settings_and_untagged_replays_keep_identity():
    base = {"hf_model_id": "google/mobilebert-uncased", "dataset_name": "glue", "dataset_config": "mrpc",
            "task": "text_classification", "hf_task": "sentence_similarity", "task_type": "classification",
            "training_regime": "finetune_transfer"}
    original = dict(base)
    apply_mobilebert_generation_defaults(base)
    policy = training.build_training_policy({"config": base, "parent_state_hash": "trained-parent"})
    rows = recipes.sample_client_recipes(policy, count=8, seed="tagged-parent", average_samples=300)
    for row in rows:
        assert row["mobilebert_head_normalization"] == "layer_norm_v1" and row["max_grad_norm"] == 1.0
        training.validate_training_recipe(row)
    old = _resolve_row(pd.Series(original), {})
    assert old.get("mobilebert_head_normalization") is None
    assert group_key(original) != group_key(base)


def test_mobilebert_training_and_inference_reload_preserve_logits_and_identity(tmp_path, monkeypatch, request):
    from transformers import MobileBertConfig, MobileBertForSequenceClassification
    from mlaas_data_generator.models.adapters import hf_core
    from mlaas_data_generator.models.adapters.hf_adapter import (
        TransformersTextFineTuneAdapter, TransformersTextClassifierAdapter,
    )
    from mlaas_data_generator.provenance import model_identity
    from mlaas_data_generator.checkpoint_store import capture_model_tensor_state, restore_model_tensor_state
    from mlaas_data_generator.services.runner import _apply_runtime_knobs_to_model
    previous_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    torch.set_num_threads(1)
    cfg = MobileBertConfig(vocab_size=32, hidden_size=32, embedding_size=8, intra_bottleneck_size=8,
        intermediate_size=16, num_attention_heads=2, num_hidden_layers=1, num_feedforward_networks=1,
        max_position_embeddings=16, classifier_activation=False, num_labels=2)
    base = tmp_path / "base"
    MobileBertForSequenceClassification(cfg).save_pretrained(base)
    monkeypatch.setattr(hf_core, "get_cached_tokenizer", lambda **kw: (None, 0., False))
    model = TransformersTextFineTuneAdapter(str(base), 2, device="cpu", max_length=8, batch_size=2,
        mobilebert_head_normalization="layer_norm_v1", max_grad_norm=1., mixed_precision=True, precision_type="fp16")
    assert isinstance(model.core.model.classifier, NormalizedLinear)
    assert model.core.precision_fallback_reason == "mobile_models_fp32_stability_policy_v2"
    x = {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]).numpy()}
    result = model.fit(x, [0, 1], epochs=1, lr=1e-4, max_train_time_s=None)
    assert result["successful_optimizer_step_count"] == 1
    assert "gradient_norm_max" in result
    state = capture_model_tensor_state(model)
    saved = tmp_path / "trained"
    model.core.model.save_pretrained(saved)
    fine = TransformersTextFineTuneAdapter(str(saved), 2, device="cpu", max_length=8)
    infer = TransformersTextClassifierAdapter(str(saved), device="cpu", max_length=8)
    old = TransformersTextClassifierAdapter(str(base), device="cpu", max_length=8)
    assert not isinstance(old.core.model.classifier, NormalizedLinear)
    identity_config = {"hf_model_id": "google/mobilebert-uncased", "model_type": "hf_finetune"}
    assert model_identity(old, identity_config)["architecture_id"] != model_identity(infer, identity_config)["architecture_id"]
    for restored in (fine, infer):
        assert isinstance(restored.core.model.classifier, NormalizedLinear)
        restore_model_tensor_state(restored, state, verify=True)
        assert capture_model_tensor_state(restored).sha256 == state.sha256
        _apply_runtime_knobs_to_model(restored, {"max_grad_norm": 1., "mixed_precision": True, "precision_type": "fp16"})
        assert restored.core.max_grad_norm == 1. and not restored.core.autocast_enabled
        with torch.no_grad():
            torch.testing.assert_close(restored.core.model(torch.tensor(x["input_ids"])).logits,
                                       model.core.model(torch.tensor(x["input_ids"])).logits)
    with pytest.raises(ValueError, match="saved checkpoint contract"):
        TransformersTextFineTuneAdapter(str(saved), 2, device="cpu", mobilebert_head_normalization="none")
