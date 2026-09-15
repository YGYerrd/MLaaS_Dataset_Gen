import numpy as np
from ..family_preflight import evaluation_size_error

from .sources.generic import (
    KERAS_DATASETS,
    SKLEARN_DATASETS,
    SKLEARN_DEFAULT_TASK,
    load_keras_source,
    load_sklearn_source,
    load_csv_source,
)
from .sources.huggingface import load_huggingface_source

from .preprocessors.generic_scaling import preprocess_tabular_scaling, preprocess_image_float01
from .preprocessors.hf import preprocess_hf


PREPROCESSOR_REGISTRY = {
    "tabular_scaling": preprocess_tabular_scaling,
    "image_float01": preprocess_image_float01,
    "hf": preprocess_hf,
    "hf_text": preprocess_hf,
    "hf_text_classification": preprocess_hf,
    "hf_token_classification": preprocess_hf,
    "hf_sentence_pair_classification": preprocess_hf,
    "hf_masked_lm": preprocess_hf,
    "hf_causal_lm": preprocess_hf,
    "hf_seq2seq": preprocess_hf,
}


def _apply_preprocessors(train, test, meta, preprocessors, *, min_evaluation_examples=None):
    if min_evaluation_examples is not None:
        features, labels = test
        count = len(labels) if labels is not None else len(features)
        error = evaluation_size_error(count, minimum=int(min_evaluation_examples),
            split=meta.get("test_split", "test"), stage="before preprocessing")
        if error:
            raise ValueError(error)
    for p in preprocessors or []:
        name = p["name"]
        fn = PREPROCESSOR_REGISTRY.get(name)
        if fn is None:
            raise KeyError(f"Unknown preprocessor '{name}'. Available: {list(PREPROCESSOR_REGISTRY)}")
        args = p.get("args", {}) or {}
        train, test, meta = fn(train, test, meta, **args)
    return train, test, meta


def _select_generic_benchmark(train, test, meta, args):
    if args.get("benchmark_sample_size") is None:
        return train, test, meta
    meta = dict(meta)
    splits = []
    for name, (x, y), cap, seed in (
        ("train", train, None, args.get("seed", 42)),
        ("test", test, args["benchmark_sample_size"], args.get("benchmark_sample_seed") or 42),
    ):
        indices = np.arange(len(x))
        if cap is not None and len(indices) > int(cap):
            indices = np.random.default_rng(int(seed)).permutation(indices)[:int(cap)]
            x, y = x[indices], y[indices] if y is not None else None
        meta[f"selected_{name}_rows"] = indices.tolist()
        splits.append((x, y))
    return splits[0], splits[1], meta


def load_raw_dataset(name, **kwargs):
    """Load source splits without fitting, decoding images or tokenizing rows."""
    key = name.lower()
    # This runtime preflight is not a dataset transformation or identity field.
    kwargs.pop("min_evaluation_examples", None)
    require_official_train_split = kwargs.pop("require_official_train_split", False)

    # ---- KERAS ----
    if key in KERAS_DATASETS:
        train, test, meta = load_keras_source(key)
        train, test, meta = _select_generic_benchmark(train, test, meta, kwargs)

        return train, test, meta

    # ---- SKLEARN ----
    if key in SKLEARN_DATASETS:
        task = kwargs.get("task", SKLEARN_DEFAULT_TASK.get(key, "classification"))
        train, test, meta = load_sklearn_source(
            key,
            task=task,
            test_size=kwargs.get("test_size", 0.2),
            seed=kwargs.get("seed", 42),
        )
        train, test, meta = _select_generic_benchmark(train, test, meta, kwargs)

        return train, test, meta

    # ---- CSV ----
    if key == "csv":
        train, test, meta = load_csv_source(
            csv_path=kwargs["csv_path"],
            target=kwargs["target"],
            task=kwargs.get("task", "regression"),
            test_size=kwargs.get("test_size", 0.2),
            seed=kwargs.get("seed", 42),
        )
        train, test, meta = _select_generic_benchmark(train, test, meta, kwargs)

        return train, test, meta

    # ---- HUGGINGFACE ----
    if key in ("hf", "huggingface"):
        # Source: raw HF datasets
        train, test, meta = load_huggingface_source(**kwargs, require_official_train_split=require_official_train_split)

        return train, test, meta

    raise KeyError(
        f"Unknown dataset '{name}'. Choices: {list(KERAS_DATASETS) + list(SKLEARN_DATASETS) + ['csv', 'hf']}"
    )


def prepare_dataset(name, train, test, meta, **kwargs):
    """Prepare loaded splits within the caller's client-selection scope."""
    preprocessors = kwargs.get("preprocessors")
    if preprocessors is None:
        key = name.lower()
        if key in KERAS_DATASETS:
            preprocessors = [{"name": "image_float01"}]
        elif key in SKLEARN_DATASETS or key == "csv":
            preprocessors = [{"name": "tabular_scaling", "args": {
                "scaler": kwargs.get("scaler", "standard"),
                "y_standardize": kwargs.get("y_standardize", True)}}]
        else:
            preprocessors = [{"name": "hf", "args": {
                k: v for k, v in kwargs.items() if k != "preprocessors"}}]
    # HF owns the shared label vocabulary and dispatches selection immediately
    # after that cheap fit. Generic arrays can be selected before any scaling.
    from .tokenization_selection import active_selection
    selection = active_selection()
    if selection is not None:
        prepare = lambda tr, te, m: _apply_preprocessors(tr, te, m, preprocessors,
            min_evaluation_examples=kwargs.get("min_evaluation_examples"))
        if name.lower() not in {"hf", "huggingface"}:
            return selection.prepare(train, test, meta, prepare)
        from .preprocessors.hf import _resolve_text_hf_task, prepare_hf_candidates
        hf_args = next((p.get("args", {}) for p in preprocessors if p["name"].startswith("hf")), kwargs)
        task = _resolve_text_hf_task(meta, hf_args)
        if task not in {"sequence_classification", "sentence_similarity"}:
            return prepare_hf_candidates(selection, train, test, meta, hf_args, prepare)
        if not any(p["name"].startswith("hf") for p in preprocessors):
            return selection.prepare(train, test, meta, prepare)
        if len(preprocessors) > 1:
            raise ValueError("client selection with fitted text labels requires a single HF preprocessor")
    return _apply_preprocessors(train, test, meta, preprocessors,
                               min_evaluation_examples=kwargs.get("min_evaluation_examples"))


def load_dataset(name, **kwargs):
    return prepare_dataset(name, *load_raw_dataset(name, **kwargs), **kwargs)
