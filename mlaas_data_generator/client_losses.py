"""Per-example losses on a fixed model and the complete local training partition.

These are evaluation measurements, not losses accumulated across optimizer steps.
Structured examples contribute their mean loss over supervised tokens/pixels.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np

VERSION = "client_loss_statistics_v1"


def _sample_losses(torch, spec, logits, labels, extra, ignore_index):
    """Return one unregularized loss per source row, with explicit reductions."""
    if logits is None or labels is None or not bool(torch.isfinite(logits).all()):
        raise ValueError("client loss requires finite logits and supervised targets")
    logits = logits.float()
    name = spec.name
    if name == "sentence_similarity" and spec.is_regression:
        if logits.numel() != labels.numel() or logits.shape[0] != labels.shape[0]:
            raise ValueError("client regression predictions and targets do not align")
        values = (logits.reshape(len(labels), -1) - labels.float().reshape(len(labels), -1)).square().mean(1)
        definition = "mean_squared_error_per_example"
    elif name in {"sequence_classification", "sentence_similarity", "image_classification"}:
        if logits.ndim == 3:
            logits = logits[:, 0, :]
        if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
            raise ValueError("client classification predictions and targets do not align")
        if extra.get("multilabel", False):
            if labels.shape != logits.shape or not bool(((labels >= 0) & (labels <= 1)).all()):
                raise ValueError("invalid multilabel client targets")
            values = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float(), reduction="none").mean(1)
            definition = "binary_cross_entropy_mean_over_labels_per_example"
        else:
            if extra.get("label_mode") == "single_onehot":
                labels = labels.argmax(-1)
            labels = labels.reshape(-1)
            if len(labels) != len(logits) or not bool(((labels >= 0) & (labels < logits.shape[1])).all()):
                raise ValueError("invalid classification client targets")
            values = torch.nn.functional.cross_entropy(logits, labels.long(), reduction="none")
            definition = "cross_entropy_per_example"
    elif name in {"token_classification", "image_segmentation", "seq2seq_generation", "fill_mask", "causal_lm_generation"}:
        if name == "image_segmentation":
            labels = spec._resize_labels_to_logits(torch, labels, logits.shape[-2:])
            scores = logits
            definition = "cross_entropy_mean_over_supervised_pixels_per_example_at_logits_resolution"
        else:
            if name == "causal_lm_generation":
                logits, labels = logits[:, :-1, :], labels[:, 1:]
            scores = logits.movedim(-1, 1)
            definition = "cross_entropy_mean_over_supervised_tokens_per_example"
        if tuple(scores.shape[:1] + scores.shape[2:]) != tuple(labels.shape):
            raise ValueError("client structured predictions and targets do not align")
        valid = labels != ignore_index
        counts = valid.reshape(len(labels), -1).sum(1)
        if not bool((counts > 0).all()):
            raise ValueError("client loss requires supervision in every selected example")
        positions = torch.nn.functional.cross_entropy(scores, labels.long(),
            ignore_index=ignore_index, reduction="none")
        values = positions.reshape(len(labels), -1).sum(1) / counts
    else:
        raise ValueError(f"client loss measurement is unsupported for {name}")
    return values.detach().double().cpu().numpy(), definition


class _Totals:
    def __init__(self):
        self.count = 0
        self.loss_sum = self.squared_sum = 0.0
        self.definition = None

    def add(self, values, definition):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("client per-example losses must be finite, nonnegative and nonempty")
        if self.definition is not None and self.definition != definition:
            raise ValueError("client loss definition changed between batches")
        self.definition = definition
        self.count += len(values)
        self.loss_sum += math.fsum(map(float, values))
        self.squared_sum += math.fsum(float(v) ** 2 for v in values)

    def result(self, expected):
        if self.count != expected or not self.count:
            raise ValueError("client loss measurement did not cover the complete selected partition")
        if not math.isfinite(self.loss_sum) or not math.isfinite(self.squared_sum):
            raise ValueError("client loss statistics overflowed")
        return {"sample_count": self.count, "loss_sum": self.loss_sum,
                "mean_loss": self.loss_sum / self.count, "squared_sample_loss_sum": self.squared_sum,
                "loss_definition": self.definition, "reduction": "one_loss_per_source_example"}


def _measure_hf(core, x, y, batch_size, totals, max_length):
    from .models.adapters.microbatch import capture_window_state, restore_window_state, evaluation_batches
    torch = core.torch
    modes = [(module, module.training) for module in core.model.modules()]
    saved = capture_window_state(core)
    original_max_length = core.max_length
    state = {"eval_batch_size_requested": batch_size, "eval_batch_size_effective": batch_size,
             "eval_oom_retry_count": 0, "eval_memory_fallback_applied": False}

    def evaluate(xb, yb):
        enc, labels, extra = core.task_spec.encode_batch(core.tokenizer, xb, yb, core.max_length,
            torch, core.device, ignore_index=core.label_pad_value, inference_only=False)
        # Only encoder-decoder models need targets in the forward inputs. Other
        # models must not choose/mutate their loss configuration during this pass.
        teacher_forcing = core.task_spec.name == "seq2seq_generation"
        inputs = core.task_spec.build_forward_inputs(enc,
            labels_t=labels if teacher_forcing else None, inference_only=not teacher_forcing)
        outputs = core.model(**inputs)
        values, definition = _sample_losses(torch, core.task_spec, core._extract_logits(outputs),
                                            labels, extra, core.label_pad_value)
        if len(values) != core._batch_item_count(xb):
            raise ValueError("client loss batch did not return one loss per source example")
        return values, definition

    try:
        if max_length is not None:
            core.max_length = int(max_length)
        core.model.eval()
        # Use native model weights, without changing the trainer's AMP policy or
        # scaler. Compute the loss in float32 and its sums/squares in float64.
        with torch.no_grad(), torch.autocast(device_type=core._device_type(), enabled=False):
            for values, definition in evaluation_batches(core,
                    core._batch_iter(x, y, batch_size=batch_size), state, evaluate):
                totals.add(values, definition)
        state["max_length"] = core.max_length
    finally:
        restore_window_state(core, saved)
        core.max_length = original_max_length
        for module, training in modes:
            module.training = training
    return {**state, "forward_precision": "native_weights_without_autocast", "loss_precision": "float32"}


def _measure_keras(model, x, y, batch_size, totals):
    import keras
    from .models.adapters.microbatch import slice_batch
    loss = keras.losses.get(model.loss)
    if isinstance(loss, keras.losses.Loss):
        loss = type(loss).from_config({**loss.get_config(), "reduction": "none"})
    definition = "keras_" + str(getattr(loss, "name", getattr(loss, "__name__", type(loss).__name__)))
    # Inference must not change BatchNorm statistics or layer RNG counters.
    saved = [(v, np.array(keras.ops.convert_to_numpy(v), copy=True)) for v in model.non_trainable_variables]
    try:
        for start in range(0, len(y), batch_size):
            stop = min(start + batch_size, len(y))
            predictions = model(slice_batch(x, start, stop), training=False)
            targets = y[start:stop]
            if definition in {"keras_mean_squared_error", "keras_mse"} and np.ndim(targets) == 1:
                targets = np.asarray(targets).reshape(-1, 1)
            values = np.asarray(keras.ops.convert_to_numpy(loss(targets, predictions)), dtype=np.float64)
            if values.ndim == 0 or values.shape[0] != stop - start:
                raise ValueError("Keras client loss must return one unreduced loss per example")
            totals.add(values.reshape(stop - start, -1).mean(1), definition)
    finally:
        for variable, value in saved:
            variable.assign(value)
    return {"eval_batch_size_requested": batch_size, "eval_batch_size_effective": batch_size,
            "forward_precision": "model_compute_dtype", "loss_precision": "keras_loss_dtype"}


def measure_client_losses(model, x, y, *, batch_size, max_length=None):
    """Measure without updating weights, optimizer, trainer settings or RNG state."""
    batch_size = int(batch_size)
    if batch_size < 1 or y is None or len(y) < 1:
        raise ValueError("client loss requires positive batch size and a nonempty labelled partition")
    before = time.perf_counter()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    totals = _Totals()
    try:
        core = getattr(model, "core", None)
        if core is not None:
            metadata = _measure_hf(core, x, y, batch_size, totals, max_length)
        elif hasattr(model, "loss") and callable(model):
            metadata = _measure_keras(model, x, y, batch_size, totals)
        else:
            raise ValueError("client loss measurement requires an HF or Keras model")
        return {**totals.result(len(y)), **metadata, "measurement_wall_s": time.perf_counter() - before}
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def client_loss_evidence(initial, final, *, initial_state_hash, source_rows_hash):
    """Give the three requested attributes unambiguous model/population scope."""
    if initial["sample_count"] != final["sample_count"] or initial["loss_definition"] != final["loss_definition"]:
        raise ValueError("initial and final client loss populations/definitions differ")
    return {
        "squared_sample_loss_sum": final["squared_sample_loss_sum"],
        "local_loss_l2_squared": final["squared_sample_loss_sum"],
        "local_loss_current_model": initial["mean_loss"],
        "client_loss_measurement_wall_s": initial["measurement_wall_s"] + final["measurement_wall_s"],
        "client_loss_statistics": {"version": VERSION, "population": "selected_local_training_rows",
            "source_rows_hash": source_rows_hash, "evaluation_mode": True,
            "initial_model": {**initial, "model_state_hash": initial_state_hash},
            "final_local_model": final,
            "local_loss_l2_squared_definition": "sum_of_squared_per_example_final_local_model_losses"},
    }
