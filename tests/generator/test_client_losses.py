"""Measured local loss statistics and isolation from the replayed training run."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlaas_data_generator.client_losses import measure_client_losses, _sample_losses, client_loss_evidence
from mlaas_data_generator.models.adapters.hf_core import HFCore
from mlaas_data_generator.models.adapters.hf_task import (
    SequenceClassificationSpec, SentenceSimilaritySpec, TokenClassificationSpec, ImageSegmentationSpec)


class LogitModel(torch.nn.Module):
    def __init__(self, *, oom=False, invalid=False):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[np.log(.9), np.log(.1)],
            [np.log(.5), np.log(.5)], [np.log(.1), np.log(.9)]], dtype=torch.float32))
        self.register_buffer("observations", torch.zeros(()))
        self.dropout = torch.nn.Dropout(.5)
        self.oom, self.invalid = oom, invalid

    def forward(self, input_ids, **kwargs):
        assert not self.training and not self.dropout.training and not torch.is_grad_enabled()
        assert "labels" not in kwargs
        self.observations.add_(1)
        torch.rand(1)
        random.random()
        np.random.random()
        if self.oom and len(input_ids) > 1:
            raise RuntimeError("CUDA out of memory: injected evaluation failure")
        logits = self.logits[input_ids[:, 0]]
        return SimpleNamespace(logits=logits * float("nan") if self.invalid else logits)


def classifier(**kwargs):
    core = HFCore.__new__(HFCore)
    core.model = LogitModel(**kwargs)
    core.torch, core.device = torch, torch.device("cpu")
    core.tokenizer, core.task_spec = None, SequenceClassificationSpec()
    core.max_length, core.batch_size, core.label_pad_value = 8, 2, -100
    core.autocast_enabled = True  # Observation must not change the training policy.
    return SimpleNamespace(core=core), {"input_ids": np.arange(3).reshape(3, 1)}, np.zeros(3, dtype=np.int64)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_exact_per_example_squares_and_rng_mode_buffer_preservation(batch_size):
    model, x, y = classifier()
    core = model.core
    core.model.train()
    core.model.dropout.eval()  # Preserve mixed module modes, not just root mode.
    before = {k: v.clone() for k, v in core.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    stats = measure_client_losses(model, x, y, batch_size=batch_size)
    expected = -np.log([.9, .5, .1])
    assert stats["sample_count"] == 3
    assert stats["loss_sum"] == pytest.approx(expected.sum())
    assert stats["squared_sample_loss_sum"] == pytest.approx((expected ** 2).sum())
    assert stats["squared_sample_loss_sum"] != pytest.approx(3 * stats["mean_loss"] ** 2)
    assert core.model.training and not core.model.dropout.training and core.autocast_enabled
    assert all(torch.equal(before[k], v) for k, v in core.model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())
    assert random.getstate() == py_rng
    assert np.random.get_state()[0] == np_rng[0]
    np.testing.assert_array_equal(np.random.get_state()[1], np_rng[1])
    assert np.random.get_state()[2:] == np_rng[2:]
    assert all(p.grad is None for p in core.model.parameters())


def test_oom_retry_does_not_double_count_examples():
    model, x, y = classifier(oom=True)
    result = measure_client_losses(model, x, y, batch_size=3)
    assert result["sample_count"] == 3
    assert result["eval_oom_retry_count"] == 1
    assert result["eval_batch_size_effective"] == 1
    assert result["squared_sample_loss_sum"] == pytest.approx(np.square(-np.log([.9, .5, .1])).sum())


def test_initial_loss_uses_recipe_input_length_without_changing_adapter_settings():
    model, x, y = classifier()
    spec = model.core.task_spec
    encode = spec.encode_batch
    seen = []

    def record(tokenizer, xb, yb, max_length, *args, **kwargs):
        seen.append(max_length)
        return encode(tokenizer, xb, yb, max_length, *args, **kwargs)

    spec.encode_batch = record
    result = measure_client_losses(model, x, y, batch_size=2, max_length=4)
    assert seen == [4, 4] and result["max_length"] == 4
    assert model.core.max_length == 8


def test_hf_training_updates_are_identical_with_loss_measurements():
    import copy
    from tests.generator.test_hf_training_precision import tiny_core

    class TrainClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 4)
            self.dropout = torch.nn.Dropout(.3)
            self.output = torch.nn.Linear(4, 2)

        def forward(self, input_ids, labels=None):
            logits = self.output(self.dropout(self.embedding(input_ids).mean(1)))
            loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return SimpleNamespace(logits=logits, loss=loss)

    torch.manual_seed(11)
    initial = TrainClassifier()
    x = {"input_ids": np.array([[1, 2], [3, 4], [5, 6]])}
    y = np.array([0, 1, 0])
    states, losses = [], []
    for capture in (False, True):
        core = tiny_core()
        core.model, core.task_spec = copy.deepcopy(initial), SequenceClassificationSpec()
        core.autocast_enabled, core.grad_scaler = False, None
        torch.manual_seed(97)
        if capture:
            measure_client_losses(SimpleNamespace(core=core), x, y, batch_size=2)
        metrics = core.finetune(x, y, epochs=2, lr=.01, optimizer="sgd", max_train_time_s=None)
        if capture:
            measure_client_losses(SimpleNamespace(core=core), x, y, batch_size=2)
        states.append({k: v.clone() for k, v in core.model.state_dict().items()})
        losses.append(metrics["train_loss"])
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert losses[0] == losses[1]


def test_bad_loss_fails_and_restores_state():
    model, x, y = classifier(invalid=True)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="finite logits"):
        measure_client_losses(model, x, y, batch_size=2)
    assert model.core.model.training
    assert model.core.model.observations.item() == 0
    assert torch.equal(before, torch.get_rng_state())


def test_regression_losses_are_squared_before_summing_again():
    values, name = _sample_losses(torch, SentenceSimilaritySpec(is_regression=True),
        torch.tensor([[1.], [2.], [3.]]), torch.zeros(3), {}, -100)
    np.testing.assert_allclose(values, [1, 4, 9])
    assert np.square(values).sum() == 98
    assert name == "mean_squared_error_per_example"


def test_multilabel_and_onehot_reductions():
    logits = torch.tensor([[1., -1.], [0., 2.]])
    labels = torch.tensor([[1., 0.], [0., 1.]])
    values, _ = _sample_losses(torch, SequenceClassificationSpec(multilabel=True), logits, labels,
                               {"multilabel": True}, -100)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1)
    np.testing.assert_allclose(values, expected.numpy())
    values, _ = _sample_losses(torch, SequenceClassificationSpec(), logits, labels,
                               {"label_mode": "single_onehot"}, -100)
    np.testing.assert_allclose(values, torch.nn.functional.cross_entropy(logits, labels.argmax(1), reduction="none"))


def test_structured_losses_average_within_each_example_before_squaring():
    logits = torch.tensor([[[3., 0.], [1., 2.], [0., 0.]], [[0., 3.], [2., 1.], [0., 3.]]])
    labels = torch.tensor([[0, -100, -100], [0, 1, 1]])
    values, _ = _sample_losses(torch, TokenClassificationSpec(), logits, labels, {}, -100)
    ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    np.testing.assert_allclose(values, [ce[0, 0], ce[1].mean()])
    with pytest.raises(ValueError, match="supervision in every"):
        _sample_losses(torch, TokenClassificationSpec(), logits, torch.full_like(labels, -100), {}, -100)


def test_segmentation_uses_logits_resolution_and_ignores_void_pixels():
    logits = torch.tensor([[[[2., 0.]], [[0., 2.]]]])
    labels = torch.tensor([[[0, 0, -100, -100], [0, 0, -100, -100]]])
    values, definition = _sample_losses(torch, ImageSegmentationSpec(), logits, labels, {}, -100)
    assert values.tolist() == pytest.approx([np.log1p(np.exp(-2))])
    assert "logits_resolution" in definition


def test_seq2seq_uses_teacher_forcing_and_counts_source_examples():
    from tests.generator.test_task_runtime import fixture
    model, _, x, y = fixture()
    batches = model.core._batch_iter
    model.core._batch_iter = lambda x, y, **kwargs: batches(x, y)
    stats = measure_client_losses(model, x, y, batch_size=2)
    assert stats["sample_count"] == 3  # Seven supervised tokens, three samples.
    assert stats["squared_sample_loss_sum"] == pytest.approx(3 * np.log(10) ** 2)
    assert model.core.model.generate_inputs == []


@pytest.mark.parametrize("loss,outputs,targets,expected", [
    ("mse", [[1.], [2.], [3.]], [0., 0., 0.], [1., 4., 9.]),
    ("sparse_categorical_crossentropy", [[.9, .1], [.5, .5], [.1, .9]], [0, 0, 0], -np.log([.9, .5, .1])),
])
def test_keras_unreduced_losses(loss, outputs, targets, expected):
    import keras
    inputs = keras.Input(shape=(len(outputs[0]),))
    model = keras.Model(inputs, keras.layers.Activation("linear")(inputs))
    model.compile(loss=loss)
    stats = measure_client_losses(model, np.asarray(outputs, dtype="float32"), np.asarray(targets), batch_size=2)
    assert stats["sample_count"] == 3
    assert stats["loss_sum"] == pytest.approx(np.sum(expected))
    assert stats["squared_sample_loss_sum"] == pytest.approx(np.square(expected).sum())


def test_metric_aliases_keep_initial_and_final_models_separate():
    base = {"sample_count": 3, "loss_definition": "ce", "mean_loss": 2.,
            "squared_sample_loss_sum": 15., "measurement_wall_s": 1.5}
    final = {**base, "mean_loss": 1., "squared_sample_loss_sum": 4., "measurement_wall_s": 2.5}
    result = client_loss_evidence(base, final, initial_state_hash="initial", source_rows_hash="rows")
    assert result["local_loss_current_model"] == 2.
    assert result["local_loss_l2_squared"] == result["squared_sample_loss_sum"] == 4.
    assert result["client_loss_measurement_wall_s"] == 4.
    assert result["client_loss_statistics"]["initial_model"]["model_state_hash"] == "initial"


# The existing fixture trains a tiny local Iris model without model downloads.
from tests.generator.test_weight_retention import trained_recipe


def test_recipe_loss_capture_preserves_training_and_source_artifacts(trained_recipe):
    from pathlib import Path
    from mlaas_data_generator.client_reconstruction import materialize_service
    from mlaas_data_generator.checkpoint_store import capture_model_tensor_state
    db = trained_recipe["db_path"]
    original = Path(db).read_bytes()
    with materialize_service(db, "recipe") as run:
        expected_hash = capture_model_tensor_state(run["model"]).sha256
        expected_loss = run["training"].get("train_loss")
    with materialize_service(db, "recipe", capture_client_losses=True) as run:
        assert capture_model_tensor_state(run["model"]).sha256 == expected_hash
        assert run["training"].get("train_loss") == expected_loss
        evidence = run["client_losses"]["client_loss_statistics"]
        assert evidence["initial_model"]["sample_count"] == run["recipe"]["selection"]["realised_count"]
        assert evidence["final_local_model"]["sample_count"] == evidence["initial_model"]["sample_count"]
        assert evidence["source_rows_hash"] == run["recipe"]["fingerprints"]["ordered_source_rows"]
        assert run["client_losses"]["local_loss_current_model"] > 0
    assert Path(db).read_bytes() == original
