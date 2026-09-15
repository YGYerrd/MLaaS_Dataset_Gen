"""Retrieval must learn from negatives, including after memory recovery."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlaas_data_generator.contrastive_batch import contrastive_ranges
from mlaas_data_generator.clients.resources import _batch_sizes_for
from mlaas_data_generator.models.adapters.hf_task import TextImageRetrievalSpec
from tests.generator.test_hf_training_precision import tiny_core


class RetrievalModel(torch.nn.Module):
    def __init__(self, limit=100):
        super().__init__()
        self.projection = torch.nn.Linear(3, 3, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.7))
        self.limit = limit
        self.calls = []

    def forward(self, input_ids, attention_mask, pixel_values):
        self.calls.append(len(input_ids))
        if len(input_ids) > self.limit:
            raise torch.OutOfMemoryError("HIP out of memory (synthetic retrieval)")
        text = torch.nn.functional.normalize(self.projection(input_ids.float()), dim=-1)
        image = torch.nn.functional.normalize(pixel_values, dim=-1)
        logits = text @ image.T * self.logit_scale.exp()
        return SimpleNamespace(text_embeds=text, image_embeds=image, logits_per_text=logits)


def retrieval_core(batch=1, limit=100):
    torch.manual_seed(42)
    core = tiny_core()
    core.task_spec = TextImageRetrievalSpec()
    core.model = RetrievalModel(limit)
    core.batch_size = batch
    core.autocast_enabled = False
    core.grad_scaler = None
    core.generation_config = {}
    return core


def data(n):
    rng = np.random.default_rng(42)
    return dict(input_ids=rng.integers(1, 10, (n, 3)), attention_mask=np.ones((n, 3)),
                pixel_values=rng.normal(size=(n, 3)).astype(np.float32))


def train(core, n):
    return core.finetune(data(n), None, epochs=1, lr=.01, optimizer="adamw",
                         gradient_accumulation_steps=2, max_train_time_s=None)


@pytest.mark.parametrize("model,expected", [
    ("openai/clip-vit-large-patch14", 8),
    ("openai/clip-vit-base-patch32", 16),
    ("openai/clip-vit-large-patch14-336", 2),
    ("google/siglip-so400m-patch14-384", 2),
])
def test_16gb_recipes(model, expected):
    assert _batch_sizes_for("text_image_retrieval", {"hf_model_id": model}) == (expected,)


def test_partition_preserves_every_pair_without_singletons():
    for capacity in range(2, 18):
        for count in range(2, 70):
            if capacity == 2 and count % 2:
                with pytest.raises(ValueError, match="insufficient_contrastive_batch"):
                    contrastive_ranges(count, capacity)
                continue
            ranges = contrastive_ranges(count, capacity)
            assert all(2 <= stop - start <= capacity for start, stop in ranges)
            assert [i for a, b in ranges for i in range(a, b)] == list(range(count))


@pytest.mark.parametrize("samples", [2, 3, 17, 33])
def test_old_singleton_recipe_updates_weights_with_all_samples(samples):
    core = retrieval_core()
    before = core.model.projection.weight.detach().clone()
    result = train(core, samples)
    assert not torch.equal(before, core.model.projection.weight)
    assert result["train_loss"] > 0
    assert result["train_sequence_count"] == samples
    assert min(core.model.calls) >= 2
    assert result["batch_size_requested"] == core.batch_size == 1
    assert result["contrastive_candidates_min"] >= 2


def test_single_pair_rejected_before_model_forward():
    core = retrieval_core()
    with pytest.raises(ValueError, match="insufficient_contrastive_batch"):
        train(core, 1)
    with pytest.raises(ValueError, match="insufficient_contrastive_batch"):
        core._backward_microbatch(data(1), None, loss_weight=1.)
    assert core.model.calls == []
    with pytest.raises(ValueError, match="at least two pairs"):
        core.task_spec.extract_loss(torch, SimpleNamespace(loss=torch.tensor(0.)), torch.ones(1, 1), None, {})


def test_memory_fallback_stops_at_two_and_preserves_samples():
    core = retrieval_core(limit=2)
    result = train(core, 34)
    assert result["train_sequence_count"] == 34
    assert result["micro_batch_size_effective"] == 2
    assert result["contrastive_candidates_min"] == 2
    assert min(core.model.calls) == 2
    assert result["successful_optimizer_step_count"] == 2
    assert result["effective_optimizer_batch_size"] == 32


def test_oom_at_two_fails_instead_of_noop_success():
    core = retrieval_core(limit=1)
    before = core.model.projection.weight.detach().clone()
    with pytest.raises(RuntimeError, match="insufficient_contrastive_batch_memory"):
        train(core, 16)
    assert min(core.model.calls) == 2
    assert torch.equal(before, core.model.projection.weight)
    assert all(p.grad is None for p in core.model.parameters())


def test_odd_tail_that_cannot_fit_fails_without_dropping_or_repeating():
    core = retrieval_core(limit=2)
    with pytest.raises(ValueError, match="cannot partition 3 pairs with capacity 2"):
        train(core, 3)
    assert min(core.model.calls) == 3  # No singleton was attempted.


@pytest.mark.parametrize("batch,limit", [(1, 100), (2, 100), (4, 100), (5, 100), (5, 1)])
def test_evaluation_loss_uses_all_candidates_even_after_oom(batch, limit):
    core = retrieval_core(batch=batch, limit=limit)
    xs = data(5)
    unlimited = retrieval_core(batch=5)
    enc, _, _ = core.task_spec.encode_batch(None, xs, None, 8, torch, "cpu")
    outputs = unlimited.model(**enc)
    expected = core.task_spec.loss_fn(torch, outputs.logits_per_text, None, {}).item()
    loss, _, _, qos = core.eval(xs, np.arange(5))
    assert loss == pytest.approx(expected, rel=1e-6)
    assert loss > 0
    assert qos["eval_loss_candidate_count"] == 5
    assert qos["evaluation_loss_protocol"] == "pooled_symmetric_contrastive_v1"
    assert qos["metric_stat_candidate_count"] == 5


def test_singleton_evaluation_pool_rejected():
    with pytest.raises(ValueError, match="insufficient_contrastive_candidates"):
        retrieval_core().eval(data(1), np.zeros(1))


def test_all_amp_updates_skipped_is_not_training_success():
    core = retrieval_core()
    # Finite forward loss, deliberately overflowing scaled backward gradients.
    core.grad_scaler = torch.amp.GradScaler("cpu", init_scale=1e38)
    core.model.projection.weight.register_hook(lambda grad: grad * float("inf"))
    with pytest.raises(ValueError, match="no_contrastive_optimizer_updates"):
        train(core, 16)
    assert core.training_memory_metadata["skipped_optimizer_step_count"] == 1


def test_pooled_loss_multiple_blocks_matches_dense_loss():
    core = retrieval_core(batch=270, limit=300)
    xs = data(270)
    enc, _, _ = core.task_spec.encode_batch(None, xs, None, 8, torch, "cpu")
    outputs = core.model(**enc)
    stats = core.task_spec.accumulate_metric_statistics(None,
        core.task_spec.batch_metric_statistics_from_outputs(torch, outputs, None, {}))
    actual, count = core.task_spec.pooled_evaluation_loss(torch, stats, core.model)
    expected = core.task_spec.loss_fn(torch, outputs.logits_per_text, None, {}).item()
    assert count == 270
    assert actual == pytest.approx(expected, rel=1e-6)
