"""Production propagation of the reviewed ADE20K-mini labels and precision."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from mlaas_data_generator.segmentation_policy import (
    ADE20K_MINI, ADE20K_MINI_LABELS, validate_ade20k_mini_preparation,
)
from mlaas_data_generator.precision_policy import (
    apply_precision_policy, SEGFORMER_MINI_FP32_REASON,
)
from mlaas_data_generator.cli.manifest import preparation
from mlaas_data_generator.cli.run_manifest import _resolve_row
from mlaas_data_generator.provenance import label_semantics, sha256_json

MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"


def config():
    return dict(dataset="hf", dataset_name=ADE20K_MINI, hf_model_id=MODEL,
                model_type="hf_finetune", hf_task="image_segmentation", task_type="segmentation",
                task="image_segmentation", training_regime="finetune_transfer",
                train_split="train", test_split="validation", image_column="image",
                mask_column="annotation", label_column="annotation", mixed_precision=True,
                precision_type="fp16", seed=42, sample_seed=43, batch_size=2)


def metadata():
    return dict(hf_id=ADE20K_MINI, hf_model_id=MODEL, hf_task="image_segmentation",
                task_type="segmentation", input_shape=(3, 32, 32), num_labels=3, num_classes=3,
                label_mapping=dict(ADE20K_MINI_LABELS), segmentation_reduce_labels=False,
                ignore_index=255, label_pad_value=255)


def test_generated_manifest_and_resolved_legacy_rows_freeze_reviewed_settings():
    from mlaas_data_generator.cli.manifest.hf_manifest_builder import build_hf_manifest
    frame = build_hf_manifest(task_keys=["image_segmentation"], models_per_task=20,
                              clients_per_family=4, average_samples=600, seed=42)
    selected = frame[(frame.hf_model_id == MODEL) & (frame.dataset_name == ADE20K_MINI)]
    assert len(selected) == 4
    assert set(selected.num_labels) == {3} and set(selected.num_classes) == {3}
    assert set(selected.precision_type) == {"fp32"} and not selected.mixed_precision.any()
    assert set(selected.precision_policy) == {SEGFORMER_MINI_FP32_REASON}
    row = _resolve_row(pd.Series({**config(), "num_labels": 150, "num_classes": 150}), {})
    assert row["num_labels"] == row["num_classes"] == 3
    assert row["precision_type"] == "fp32" and row["mixed_precision"] is False


def test_preparation_keeps_fp32_and_has_a_new_cache_namespace():
    old = config()
    frozen = preparation.contract_config(old)
    assert frozen["num_labels"] == frozen["num_classes"] == 3
    assert frozen["precision_type"] == "fp32" and frozen["mixed_precision"] is False
    assert frozen["precision_policy"] == SEGFORMER_MINI_FP32_REASON
    legacy = {k: v for k, v in frozen.items()
              if k not in {"num_labels", "num_classes", "precision_type", "mixed_precision", "precision_policy"}}
    assert preparation.group_key(old) != sha256_json(legacy)
    assert preparation.group_key(old) == preparation.group_key(frozen)
    assert old["precision_type"] == "fp16"  # Inspecting a contract does not alter its source.


@pytest.mark.parametrize("dataset,model", [("zhoubolei/scene_parse_150", MODEL),
                                          (ADE20K_MINI, "Intel/dpt-large-ade")])
def test_fp32_policy_is_scoped_to_segformer_on_mini(dataset, model):
    cfg = {**config(), "dataset_name": dataset, "hf_model_id": model}
    assert apply_precision_policy(cfg)["precision_type"] == "fp16"


@pytest.mark.parametrize("change", ["count", "mapping", "reduction"])
def test_legacy_frozen_labels_are_rejected_before_loading(monkeypatch, tmp_path, change):
    from mlaas_data_generator.services import runner
    doc = {"config": config(), "contracts": {"labels": label_semantics(metadata())}}
    validate_ade20k_mini_preparation(doc)
    labels = doc["contracts"]["labels"]
    if change == "count":
        labels["num_labels"] = 150
    elif change == "mapping":
        labels["labels"][0]["label"] = "wall"
    else:
        labels["special_semantics"]["segmentation_reduce_labels"] = True
    monkeypatch.setattr(preparation, "read_preparation", lambda *a, **k: doc)
    monkeypatch.setattr(preparation, "manifest_family_error", lambda cfg: None)
    monkeypatch.setattr(runner, "_provenance_artifact_root", lambda cfg: str(tmp_path))
    def no_load(*args, **kwargs):
        pytest.fail("legacy preparation reached dataset/model loading")
    monkeypatch.setattr(preparation, "load_prepared", no_load)
    with pytest.raises(ValueError, match="legacy ADE20K-mini family"):
        preparation.prepare({**config(), "_family_reference": {"document_ref": "old", "document_hash": "hash"}})
    # Reading historical artifacts remains independent of the production gate.
    other = deepcopy(doc)
    other["config"]["dataset_name"] = "zhoubolei/scene_parse_150"
    validate_ade20k_mini_preparation(other)


def test_legacy_loaded_metadata_cannot_build_a_150_label_production_model(monkeypatch):
    from mlaas_data_generator.models.builders import create_model
    from mlaas_data_generator.models.adapters import hf_adapter
    def no_model(**kwargs):
        pytest.fail("legacy metadata reached model construction")
    monkeypatch.setattr(hf_adapter, "TransformersTextFineTuneAdapter", no_model)
    meta = {**metadata(), "num_labels": 150, "num_classes": 150}
    with pytest.raises(ValueError, match="unreduced three-label"):
        create_model((3, 32, 32), 150, model_type="hf_finetune", meta=meta)


def test_local_segformer_initial_eval_train_and_checkpoint_reload(tmp_path, request):
    """Use real Transformers processors/model/serialization, with tiny random weights."""
    import torch
    import transformers
    from mlaas_data_generator.data.preprocessors.hf_image import preprocess_hf_image
    from mlaas_data_generator.models.builders import create_model
    from mlaas_data_generator.tasks.contracts import resolve_task_contract
    from mlaas_data_generator.tasks.runtime import evaluate_task_model
    from mlaas_data_generator.checkpoint_store import capture_model_tensor_state, restore_model_tensor_state
    from tests.generator.test_segmentation_declared_labels import Split

    previous_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    torch.set_num_threads(1)
    base = tmp_path / "base"
    model_config = transformers.SegformerConfig(
        num_labels=150, depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 2, 4, 8], sr_ratios=[4, 2, 1, 1], decoder_hidden_size=16)
    transformers.SegformerForSemanticSegmentation(model_config).save_pretrained(base)
    processor = transformers.SegformerImageProcessor(size={"height": 32, "width": 32}, do_reduce_labels=True)
    processor.save_pretrained(base)
    split = Split([ADE20K_MINI_LABELS, ADE20K_MINI_LABELS])
    train, test, meta = preprocess_hf_image(
        (split, None), (split, None), {"hf_id": ADE20K_MINI}, hf_model_id=str(base),
        task_type="segmentation", label_column="annotation", training_augmentations=False)
    meta.update(hf_model_id=str(base), hf_task="image_segmentation")
    assert meta["num_labels"] == 3 and not meta["segmentation_reduce_labels"]
    assert set(np.unique(train[1])) == {0, 1, 2}
    adapter = create_model(meta["input_shape"], meta["num_classes"], meta=meta,
                           model_type="hf_finetune", mixed_precision=True, precision_type="fp16",
                           device="cpu", batch_size=2)
    core = adapter.core
    assert core.model.config.num_labels == 3
    assert core.precision_fallback_reason == SEGFORMER_MINI_FP32_REASON
    assert not core.autocast_enabled and core.effective_precision_type == "fp32"
    # Exercise the CUDA policy branch without requiring a GPU allocation.
    core.device = "cuda"
    core.model.half()
    core._configure_precision_mode()
    assert not core.autocast_enabled and core.grad_scaler is None
    assert all(p.dtype == torch.float32 for p in core.model.parameters())
    core.device = "cpu"
    contract = resolve_task_contract(config(), meta, adapter)
    baseline = evaluate_task_model(adapter, *test, contract, ["test:0", "test:1"])
    assert np.isfinite(baseline.metrics["iou"])
    initial = capture_model_tensor_state(adapter)
    trained = adapter.fit(*train, epochs=1, lr=1e-4, max_train_time_s=None)
    assert trained["successful_optimizer_step_count"] == 1
    state = capture_model_tensor_state(adapter)
    assert state.sha256 != initial.sha256
    saved = tmp_path / "trained"
    core.model.save_pretrained(saved)
    restored_meta = {**meta, "hf_model_id": str(saved)}
    restored = create_model(meta["input_shape"], 3, meta=restored_meta, model_type="hf_finetune",
                            mixed_precision=True, precision_type="fp16", device="cpu", batch_size=2)
    restore_model_tensor_state(restored, state, verify=True)
    assert capture_model_tensor_state(restored).sha256 == state.sha256
    assert restored.core.precision_fallback_reason == SEGFORMER_MINI_FP32_REASON
    result = evaluate_task_model(restored, *test, contract, ["test:0", "test:1"])
    assert np.isfinite(result.metrics["iou"])


def test_initial_preparation_receives_reviewed_precision(monkeypatch):
    from mlaas_data_generator import shared_parent
    from mlaas_data_generator.services import runner
    cfg = config()
    monkeypatch.setattr(preparation, "manifest_family_error", lambda cfg: None)
    monkeypatch.setattr(runner, "_sample_count", lambda *a: 500)
    def inspect(cfg, train, test, meta, **kwargs):
        assert cfg["precision_type"] == "fp32" and cfg["mixed_precision"] is False
        assert cfg["num_labels"] == cfg["num_classes"] == 3
        return train, test, meta
    monkeypatch.setattr(shared_parent, "_prepare_loaded", inspect)
    preparation.load_prepared(cfg, load_dataset=lambda *a, **k:
                             (([0], [0]), ([1], [1]), {"train_split": "train", "test_split": "validation"}))
