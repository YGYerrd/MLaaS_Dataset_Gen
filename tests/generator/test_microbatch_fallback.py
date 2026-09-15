"""Synthetic OOMs on tiny CPU tensors; no model downloads or GPU allocation."""
from types import SimpleNamespace
import json
import sqlite3

import numpy as np
import pytest
import torch

from tests.generator.test_hf_training_precision import tiny_core
from tests.generator.test_hf_eval_protocol import make_core
from tests.generator.test_task_runtime import fixture


def train(core, *, accumulation=1, samples=14, epochs=1):
    np.random.seed(42)
    return core.finetune(np.linspace(.2, 1, samples), np.zeros(samples), epochs=epochs,
                         lr=.1, optimizer="sgd", gradient_accumulation_steps=accumulation,
                         max_train_time_s=None)


def failing_core(limit, *, backward=False, late=False, batch_size=4):
    core = tiny_core()
    core.batch_size = batch_size
    core.autocast_enabled = False
    core.grad_scaler = None
    core.model.register_buffer("examples_seen", torch.tensor(0))
    original = core.model.forward

    class BackwardOOM(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss):
            return loss.clone()

        @staticmethod
        def backward(ctx, grad):
            # Simulate gradients written by an earlier branch of backward.
            core.model.weight.grad = torch.tensor(123.)
            raise torch.OutOfMemoryError("HIP out of memory (synthetic backward)")

    def forward(x, labels):
        result = original(x, labels)
        core.model.examples_seen.add_(len(x))
        if len(x) > limit and (not late or core.model.calls >= 4):
            if backward:
                result.loss = BackwardOOM.apply(result.loss)
                return result
            raise torch.OutOfMemoryError("HIP out of memory (synthetic forward)")
        return result

    core.model.forward = forward
    return core


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("late", [False, True])
def test_retry_preserves_updates_samples_and_buffers(backward, late):
    baseline = failing_core(99)
    expected = train(baseline, accumulation=2, epochs=2)
    core = failing_core(1, backward=backward, late=late)
    actual = train(core, accumulation=2, epochs=2)
    assert core.model.weight.item() == pytest.approx(baseline.model.weight.item(), abs=1e-7)
    assert actual["train_loss"] == pytest.approx(expected["train_loss"], abs=1e-7)
    assert actual["successful_optimizer_step_count"] == expected["successful_optimizer_step_count"] == 4
    assert actual["train_sequence_count"] == 28
    assert core.model.examples_seen.item() == 28
    assert actual["batch_size_requested"] == core.batch_size == 4
    assert actual["micro_batch_size_effective"] == 1
    assert actual["gradient_accumulation_steps_requested"] == 2
    assert actual["gradient_accumulation_steps_effective"] == 8
    assert actual["effective_optimizer_batch_size"] == 8
    assert actual["oom_retry_count"] == 2 and actual["memory_fallback_applied"]


def test_odd_requested_batch_keeps_integral_accumulation():
    core = failing_core(1, batch_size=3)
    result = train(core, accumulation=2)
    assert result["gradient_accumulation_steps_effective"] == 6
    assert result["effective_optimizer_batch_size"] == 6
    assert result["oom_retry_count"] == 1


def test_amp_scaler_survives_backward_retry():
    core = failing_core(1, backward=True)
    core.grad_scaler = torch.amp.GradScaler("cpu", init_scale=128.)
    baseline = failing_core(99)
    train(baseline)
    result = train(core)
    assert core.model.weight.item() == pytest.approx(baseline.model.weight.item(), abs=1e-7)
    assert result["skipped_optimizer_step_count"] == 0


def test_microbatch_one_exhaustion_fails_and_clears_gradients():
    core = failing_core(0, backward=True)
    with pytest.raises(torch.OutOfMemoryError):
        train(core)
    assert core.model.weight.item() == 1.
    assert core.model.weight.grad is None
    assert core.training_memory_metadata["oom_retry_count"] == 2
    assert core.training_memory_metadata["successful_optimizer_step_count"] == 0


def test_optimizer_oom_never_replays_partially_committed_step():
    core = failing_core(99)
    original = core._build_optimizer
    calls = []
    def build(**kwargs):
        opt = original(**kwargs)
        def fail(*args, **kw):
            calls.append(1)
            raise torch.OutOfMemoryError("HIP out of memory in optimizer state allocation")
        opt.step = fail
        return opt
    core._build_optimizer = build
    with pytest.raises(torch.OutOfMemoryError):
        train(core)
    assert len(calls) == 1
    assert core.training_memory_metadata["oom_retry_count"] == 0


def test_non_oom_errors_do_not_change_batch_recipe():
    core = failing_core(99)
    def fail(*args, **kwargs):
        raise RuntimeError("HIP illegal memory access")
    core.model.forward = fail
    with pytest.raises(RuntimeError, match="illegal memory access"):
        train(core)
    assert core.training_memory_metadata["oom_retry_count"] == 0


def test_timeout_discards_partial_window_and_does_not_reset_on_retry():
    core = failing_core(2)
    checks = []
    def timed_out(start, limit):
        checks.append((start, limit))
        return len(checks) >= 3  # OOM, one successful microbatch, then budget ends.
    core._training_timed_out = timed_out
    result = train(core)
    assert result["train_stopped_early"]
    assert result["train_sequence_count"] == 0
    assert result["successful_optimizer_step_count"] == 0
    assert core.model.weight.item() == 1.
    assert core.model.examples_seen.item() == 0
    assert len(set(checks)) == 1


def test_fallback_resets_between_clients():
    core = failing_core(1)
    assert train(core)["oom_retry_count"] == 2
    # The next client starts from its own requested batch again.
    result = train(core)
    assert result["oom_retry_count"] == 2 and core.batch_size == 4


def test_evaluation_restart_discards_failed_attempt_statistics():
    _, _, x, y = fixture()
    baseline = make_core().eval(x, y)
    core = make_core()
    core.batch_size = 4
    original = core.model.forward
    def forward(**kwargs):
        if len(kwargs["labels"]) > 1:
            raise torch.OutOfMemoryError("HIP out of memory")
        return original(**kwargs)
    core.model.forward = forward
    loss, primary, secondary, qos = core.eval(x, y)
    assert (loss, primary, secondary) == pytest.approx(baseline[:3])
    assert qos["eval_sequence_count"] == 3
    assert qos["eval_supervised_token_count"] == baseline[3]["eval_supervised_token_count"]
    assert qos["eval_batch_count"] == 3
    assert qos["eval_batch_size_effective"] == 1 and qos["eval_oom_retry_count"] == 2
    assert core.batch_size == 4


def test_structured_evaluation_retries_without_duplicating_streamed_predictions():
    from mlaas_data_generator.tasks.runtime import evaluate_task_model
    model, contract, x, y = fixture(batch_size=4)
    baseline = evaluate_task_model(model, x, y, contract, ["a", "b", "c"])
    original = model.core.model.forward
    def forward(**kwargs):
        if len(kwargs["labels"]) > 1:
            raise torch.OutOfMemoryError("HIP out of memory after generation")
        return original(**kwargs)
    model.core.model.forward = forward
    streamed = []
    result = evaluate_task_model(model, x, y, contract, ["a", "b", "c"], collect=True,
                                 prediction_sink=lambda i, p, t: streamed.append(i))
    assert streamed == [0, 1, 2] and len(result.predictions) == 3
    for metric in ("loss", "rouge1", "rouge2", "supervision_count", "prediction_hash"):
        expected = baseline.metrics[metric]
        assert result.metrics[metric] == (expected if isinstance(expected, str) else pytest.approx(expected))
    assert result.metrics["eval_oom_retry_count"] == 2
    assert model.core.batch_size == 4


def test_failure_persists_memory_policy_and_update_counts(tmp_path, monkeypatch):
    from mlaas_data_generator.services.runner import ServiceRunner, _fit_transformers_with_memory_metadata
    cfg = {"service_id": "oom", "db_path": str(tmp_path / "test.db"), "batch_size": 4}
    instance = ServiceRunner(cfg)
    core = failing_core(0)
    model = SimpleNamespace(core=core, fit=lambda *args, **kwargs: train(core))
    def execute(**kwargs):
        _fit_transformers_with_memory_metadata(model, None, None, {}, instance.config)
    monkeypatch.setattr(instance, "_execute_service", execute)
    assert instance.run().status == "failed"
    with sqlite3.connect(cfg["db_path"]) as db:
        metrics = dict(db.execute("select metric_name,coalesce(value_num,value_int,value_bool) from service_metrics"))
        stored = json.loads(db.execute("select service_config_json from services").fetchone()[0])
        metadata = json.loads(db.execute("select metadata_json from services").fetchone()[0])
        trace = db.execute("select traceback_text from service_failures").fetchone()[0]
    assert metrics["oom_retry_count"] == 2 and metrics["successful_optimizer_step_count"] == 0
    assert stored["batch_size"] == 4
    assert metadata["training_memory_metadata"]["micro_batch_size_effective"] == 1
    assert "OutOfMemoryError" in trace
