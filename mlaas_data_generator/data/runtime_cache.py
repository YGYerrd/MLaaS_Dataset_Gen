"""Bounded, process-local preparation reuse for a single manifest execution.

Cache raw HF splits (Arrow buffers remain shared) and tokenizer templates only.
Sampling and preprocessing still run with each recipe's own settings. Returning
copies prevents tokenizer padding changes or Dataset formatting leaking between
clients. No cache files or model weights are written.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy


_CACHE = ContextVar("manifest_preparation_cache", default=None)
_LIMITS = {"dataset": 64, "tokenizer": 4}


def start_preparation_cache():
    """Start the cache in a manifest worker (which exits with that run)."""
    return _CACHE.set({name: OrderedDict() for name in _LIMITS})


@contextmanager
def preparation_cache_scope():
    token = start_preparation_cache()
    try:
        yield
    finally:
        _CACHE.reset(token)


def cached_preparation(kind, key, load):
    caches = _CACHE.get()
    if caches is None:
        return load(), False
    cache = caches[kind]
    hit = key in cache
    if hit:
        value = cache.pop(key)
    else:
        value = load()  # Failed loads must never poison the cache.
    cache[key] = value
    while len(cache) > _LIMITS[kind]:
        cache.popitem(last=False)
    return deepcopy(value), hit
