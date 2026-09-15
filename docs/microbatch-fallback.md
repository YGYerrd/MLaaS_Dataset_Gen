# Automatic HF microbatch fallback

Training preserves the manifest's batch size and requested accumulation. On a
forward/backward CUDA or HIP OOM it reduces the microbatch and increases actual
accumulation: requested batch 4 and accumulation 1 becomes 2×2, then 1×4.
An already accumulated recipe of 4×2 becomes 2×4, then 1×8. Odd batch sizes use
smaller divisors of the requested batch, so accumulation stays integral.

Recovery discards the entire uncommitted optimizer window, clears partial
gradients, restores buffers and random-generator states, and replays the same
examples. Previously committed updates are retained. Scheduler and optimizer
steps happen only after a complete successful window. The final incomplete
window uses its actual number of batches. One wall-clock training budget covers
all retries. A timeout discards the incomplete window.

Microbatch loss is weighted by its fraction of the original batch. This preserves
the optimizer example count, not bit-for-bit computation: dropout, BatchNorm and
losses normalized over different numbers of valid tokens/pixels may differ.
Forward/backward recovery applies to HFCore adapters. OOM during optimizer.step
fails explicitly because it may have partially modified parameters/optimizer
state; restarting that step without a snapshot would be unsafe. OOM at microbatch
1 and non-OOM device errors also fail explicitly. No service-wide fit retry or
permanent batch-size mutation remains.

Stored training metrics include `batch_size_requested`,
`micro_batch_size_effective`, `gradient_accumulation_steps_requested`,
`gradient_accumulation_steps_effective`, `gradient_accumulation`,
`effective_optimizer_batch_size`, `oom_retry_count`, and `memory_fallback_applied`.
`gradient_accumulation_steps` reports actual accumulation; the manifest/config
continues to record the original recipe. Failure metrics retain fallback and
optimizer-update counts; failure metadata includes `training_memory_metadata`.

Evaluation independently tries smaller batches down to 1 without changing
training settings. Structured evaluation retries only uncommitted microbatches,
so metric counts and prediction sinks cannot receive duplicate examples. The
legacy HF evaluation path discards a failed evaluation attempt and restarts its
evaluation statistics. Both paths share one evaluation time budget and record
`eval_batch_size_requested`, `eval_batch_size_effective`, `eval_oom_retry_count`,
and `eval_memory_fallback_applied`. The policy resets for every service invocation.

Validation uses tiny CPU models with injected forward, backward and optimizer
OOMs. It does not establish that any particular production workload fits in VRAM
or benchmark GPU speed.
