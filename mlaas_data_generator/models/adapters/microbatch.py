"""OOM recovery without restarting an already partially trained service."""
import gc
import random
import traceback

import numpy as np


class TrainingBudgetExpired(Exception):
    pass


def is_device_oom(exc):
    message = str(exc).lower()
    return type(exc).__name__ == "OutOfMemoryError" or any(
        text in message for text in ("cuda out of memory", "hip out of memory")
    )


def smaller_microbatch(current, requested):
    # Divisors keep accumulation integral, including odd requested batches.
    return next((n for n in range(current // 2, 0, -1) if requested % n == 0), 1)


def release_failed_attempt(core, exc):
    # Tracebacks otherwise retain failed forward/backward tensors during retry.
    traceback.clear_frames(exc.__traceback__)
    gc.collect()
    if core._device_type() == "cuda":
        with core.torch.cuda.device(core.device):
            core.torch.cuda.empty_cache()


def slice_batch(value, start, stop):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: slice_batch(item, start, stop) for key, item in value.items()}
    return value[start:stop]


def evaluation_batches(core, batches, state, evaluate):
    """Retry an uncommitted evaluation microbatch; never duplicate sink output."""
    for xb, yb in batches:
        count = core._batch_item_count(xb)
        offset = 0
        while offset < count:
            stop = min(count, offset + state["eval_batch_size_effective"])
            try:
                result = evaluate(slice_batch(xb, offset, stop), slice_batch(yb, offset, stop))
            except Exception as exc:
                if not is_device_oom(exc) or state["eval_batch_size_effective"] <= 1:
                    raise
                release_failed_attempt(core, exc)
                old = state["eval_batch_size_effective"]
                state["eval_batch_size_effective"] = max(1, old // 2)
                state["eval_oom_retry_count"] += 1
                state["eval_memory_fallback_applied"] = True
                print(f"[task evaluation] OOM retry | batch_size={old}->{state['eval_batch_size_effective']}", flush=True)
                continue
            yield result
            offset = stop


def capture_window_state(core):
    torch = core.torch
    return {
        "buffers": [(buffer, buffer.detach().cpu().clone()) for buffer in core.model.buffers()],
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(core.device) if core._device_type() == "cuda" else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def restore_window_state(core, state):
    torch = core.torch
    with torch.no_grad():
        for buffer, original in state["buffers"]:
            buffer.copy_(original)
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], core.device)
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def backward_window(core, window, optimizer, state, start, timeout):
    """Retry only forward/backward; optimizer.step is deliberately outside.

    No parameter or optimizer state has changed at this boundary. On failure,
    replay all samples in this accumulation window, discarding partial grads,
    statistics and buffer updates. Previous successful updates remain intact.
    """
    original_state = capture_window_state(core)
    minimum = state.get("contrastive_min_pairs", 1)
    while True:
        try:
            totals = [0.0, 0, 0, 0, 0]  # loss sum, denominator, tokens, sequences, FP32 retries
            candidate_counts = []
            window_count = sum(core._batch_item_count(xb) for xb, _ in window)
            for xb, yb in window:
                count = core._batch_item_count(xb)
                capacity = state["micro_batch_size_effective"]
                if minimum == 2:
                    from ...contrastive_batch import contrastive_ranges
                    ranges = contrastive_ranges(count, capacity)
                else:
                    ranges = ((i, min(count, i + capacity)) for i in range(0, count, capacity))
                for offset, stop in ranges:
                    if core._training_timed_out(start, timeout):
                        raise TrainingBudgetExpired()
                    result = core._backward_microbatch(
                        slice_batch(xb, offset, stop), slice_batch(yb, offset, stop),
                        loss_weight=((stop - offset) / window_count if minimum == 2
                                     else (stop - offset) / count / len(window)),
                    )
                    candidate_counts.append(stop - offset)
                    totals = [a + b for a, b in zip(totals, result)]
            if minimum == 2:
                state["contrastive_candidates_min"] = min(
                    state.get("contrastive_candidates_min", min(candidate_counts)), min(candidate_counts))
                state["contrastive_candidates_max"] = max(
                    state.get("contrastive_candidates_max", max(candidate_counts)), max(candidate_counts))
            return totals
        except TrainingBudgetExpired:
            optimizer.zero_grad(set_to_none=True)
            restore_window_state(core, original_state)
            raise
        except Exception as exc:
            if not is_device_oom(exc):
                raise
            optimizer.zero_grad(set_to_none=True)
            release_failed_attempt(core, exc)
            restore_window_state(core, original_state)
            old = state["micro_batch_size_effective"]
            if old <= minimum:
                if minimum == 2:
                    raise RuntimeError(
                        "insufficient_contrastive_batch_memory: two pairs do not fit; "
                        "refusing a singleton contrastive forward. Reduce memory use or model size."
                    ) from exc
                raise
            logical_batch = state.get("logical_batch_size", state["batch_size_requested"])
            state["micro_batch_size_effective"] = (
                max(2, old // 2) if minimum == 2 else smaller_microbatch(old, logical_batch))
            state["gradient_accumulation_steps_effective"] = (
                state["gradient_accumulation_steps_requested"]
                * ((logical_batch + state["micro_batch_size_effective"] - 1)
                   // state["micro_batch_size_effective"])
            )
            state["oom_retry_count"] += 1
            state["memory_fallback_applied"] = True
            print("[HFCore.finetune] OOM microbatch retry | "
                  f"microbatch={old}->{state['micro_batch_size_effective']} | "
                  f"accumulation={state['gradient_accumulation_steps_effective']} | "
                  f"effective_batch={state['effective_optimizer_batch_size']}", flush=True)
