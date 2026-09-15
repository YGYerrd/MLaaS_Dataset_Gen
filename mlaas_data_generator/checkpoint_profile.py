"""Scoped checkpoint timings, kept outside immutable checkpoint identities."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
from typing import Any


_active_profile: ContextVar[CheckpointProfile | None] = ContextVar("checkpoint_profile", default=None)


class CheckpointProfile:
    """Collect synchronous nested spans for one checkpoint operation.

    Inclusive times overlap. Exclusive times subtract instrumented child spans
    and can be summed. No tensors or payloads are retained by the collector.
    Use a separate instance per concurrent service.
    """

    def __init__(self):
        self._steps: dict[str, dict[str, Any]] = {}
        self._stack: list[dict[str, float]] = []

    @contextmanager
    def span(self, name: str):
        frame = {"children_s": 0.0}
        self._stack.append(frame)
        started = perf_counter()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = max(0.0, perf_counter() - started)
            self._stack.pop()
            if self._stack:
                self._stack[-1]["children_s"] += elapsed
            step = self._steps.setdefault(name, {
                "calls": 0, "failed_calls": 0, "inclusive_s": 0.0, "exclusive_s": 0.0,
            })
            step["calls"] += 1
            step["failed_calls"] += int(failed)
            step["inclusive_s"] += elapsed
            step["exclusive_s"] += max(0.0, elapsed - frame["children_s"])

    def call(self, name, function, /, *args, **kwargs):
        token = _active_profile.set(self)
        try:
            with self.span(name):
                return function(*args, **kwargs)
        finally:
            _active_profile.reset(token)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": "mlaas-checkpoint-profile-v1",
            "clock": "perf_counter",
            "scope": "instrumented checkpoint calls; inclusive times overlap",
            "profiled_s": sum(step["exclusive_s"] for step in self._steps.values()),
            "steps": {name: dict(step) for name, step in sorted(self._steps.items())},
        }


def profiled(name: str):
    """Instrument a substep only while a caller has activated a collector."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            profile = _active_profile.get()
            if profile is None:
                return function(*args, **kwargs)
            with profile.span(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def profile_call(name, function, /, *args, **kwargs):
    """Time a call site separately, for example XOR versus REPLACE encoding."""
    profile = _active_profile.get()
    if profile is None:
        return function(*args, **kwargs)
    with profile.span(name):
        return function(*args, **kwargs)
