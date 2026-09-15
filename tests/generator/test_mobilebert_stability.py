from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlaas_data_generator.models.adapters.mobilebert_stability import (
    NormalizedLinear, apply_mobilebert_head_normalization,
)
from tests.generator.test_hf_training_precision import tiny_core


def model():
    m = torch.nn.Module()
    m.config = SimpleNamespace(model_type="mobilebert")
    m.classifier = torch.nn.Linear(8, 2)
    return m


def test_large_features_are_stable_without_changing_checkpoint_or_rng():
    m = model()
    state = {k: v.clone() for k, v in m.state_dict().items()}
    rng = torch.get_rng_state().clone()
    apply_mobilebert_head_normalization(m, "layer_norm_v1")
    assert torch.equal(rng, torch.get_rng_state())
    assert set(state) == set(m.state_dict())
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in state.items())
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8) * 1e7
    logits = m.classifier(x)
    assert logits.abs().max() < 10
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    restored = model()
    apply_mobilebert_head_normalization(restored, m.config.mobilebert_head_normalization)
    restored.load_state_dict(m.state_dict(), strict=True)
    torch.testing.assert_close(restored.classifier(x), logits)


def test_policy_is_explicit_and_idempotent():
    m = model()
    apply_mobilebert_head_normalization(m, None)
    assert not isinstance(m.classifier, NormalizedLinear)
    apply_mobilebert_head_normalization(m, "layer_norm_v1")
    head = m.classifier
    apply_mobilebert_head_normalization(m, "layer_norm_v1")
    assert m.classifier is head
    m.config.model_type = "bert"
    with pytest.raises(ValueError, match="requires a MobileBERT"):
        apply_mobilebert_head_normalization(m, "layer_norm_v1")


def test_clipping_unscales_gradients_before_the_update():
    core = tiny_core()
    core.max_grad_norm = .5
    result = core.finetune(np.ones(2), np.zeros(2), epochs=1, lr=.1, optimizer="sgd")
    assert core.model.weight.item() == pytest.approx(.95, abs=1e-5)
    assert result["gradient_clipped_step_count"] == 1
    assert result["gradient_norm_max"] == pytest.approx(2.)


def test_nonfinite_gradient_does_not_mutate_parameters():
    core = tiny_core()
    core.grad_scaler = None
    core.autocast_enabled = False
    core.max_grad_norm = 1.
    core.model.weight.register_hook(lambda grad: grad * float("inf"))
    with pytest.raises(RuntimeError, match="non-finite"):
        core.finetune(np.ones(2), np.zeros(2), epochs=1, lr=.1, optimizer="sgd")
    assert core.model.weight.item() == 1.


def test_normalization_changes_preparation_identity():
    from mlaas_data_generator.cli.manifest.preparation import group_key
    config = {"dataset": "hf", "dataset_name": "glue", "dataset_config": "mrpc",
              "model_type": "hf_finetune", "hf_model_id": "google/mobilebert-uncased",
              "hf_task": "sentence_similarity"}
    assert group_key(config) != group_key({**config, "mobilebert_head_normalization": "layer_norm_v1"})


def test_grouped_model_receives_and_clears_per_client_clipping():
    from mlaas_data_generator.services.runner import _apply_runtime_knobs_to_model
    core = SimpleNamespace(max_grad_norm=None)
    adapter = SimpleNamespace(core=core)
    _apply_runtime_knobs_to_model(adapter, {"max_grad_norm": 1.})
    assert core.max_grad_norm == 1.
    _apply_runtime_knobs_to_model(adapter, {})
    assert core.max_grad_norm is None
    with pytest.raises(ValueError, match="finite and positive"):
        _apply_runtime_knobs_to_model(adapter, {"max_grad_norm": float('nan')})


def test_saved_model_config_reinstates_normalization(tmp_path):
    from transformers import MobileBertConfig, MobileBertForSequenceClassification
    from mlaas_data_generator.models.adapters.hf_core import HFCore
    cfg = MobileBertConfig(vocab_size=32, hidden_size=32, embedding_size=8,
                          intra_bottleneck_size=8, intermediate_size=16,
                          num_attention_heads=2, num_hidden_layers=1,
                          num_feedforward_networks=1, max_position_embeddings=16,
                          classifier_activation=False, num_labels=2)
    original = MobileBertForSequenceClassification(cfg).eval()
    apply_mobilebert_head_normalization(original, "layer_norm_v1")
    original.save_pretrained(tmp_path)
    spec = SimpleNamespace(name="sentence_similarity", requires_tokenizer=False,
        build_model=lambda transformers, path, labels: MobileBertForSequenceClassification.from_pretrained(
            path, local_files_only=True))
    restored = HFCore(str(tmp_path), num_labels=2, max_length=8, device="cpu", task_spec=spec)
    assert isinstance(restored.model.classifier, NormalizedLinear)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(restored.model(tokens).logits, original(tokens).logits)
