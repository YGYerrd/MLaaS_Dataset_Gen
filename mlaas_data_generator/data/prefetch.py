"""Bounded, ordered CPU batch reads with deterministic worker cleanup."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def prefetch_batches(iterable, *, depth=2):
    """Overlap source reads with consumption, retaining at most depth ahead.

    The source must only perform CPU reads; tensor encoding and GPU operations
    belong to the consumer. Exiting cancels queued reads and joins the active
    reader before closing the source, including on timeout or consumer failure.
    """
    if depth < 0:
        raise ValueError("prefetch depth must be nonnegative")
    source = iter(iterable)
    executor = None
    pending = deque()
    sentinel = object()

    def batches():
        for _ in range(depth):
            pending.append(executor.submit(next, source, sentinel))
        while pending:
            item = pending.popleft().result()
            if item is sentinel:
                return
            pending.append(executor.submit(next, source, sentinel))
            yield item

    iterator = source
    try:
        if depth:
            # A single worker preserves source order and never advances an
            # iterator concurrently. Futures bound submitted and completed work.
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlaas-prefetch")
            iterator = batches()
        yield iterator
    finally:
        if executor is not None:
            iterator.close()
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            pending.clear()
        close = getattr(source, "close", None)
        if close is not None:
            close()
