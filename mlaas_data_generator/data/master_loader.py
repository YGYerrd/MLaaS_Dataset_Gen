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


def _apply_preprocessors(train, test, meta, preprocessors):
    for p in preprocessors or []:
        name = p["name"]
        fn = PREPROCESSOR_REGISTRY.get(name)
        if fn is None:
            raise KeyError(f"Unknown preprocessor '{name}'. Available: {list(PREPROCESSOR_REGISTRY)}")
        args = p.get("args", {}) or {}
        train, test, meta = fn(train, test, meta, **args)
    return train, test, meta


def load_dataset(name, **kwargs):
    key = name.lower()

    preprocessors = kwargs.get("preprocessors", None)

    # ---- KERAS ----
    if key in KERAS_DATASETS:
        train, test, meta = load_keras_source(key)

        if preprocessors is None:
            preprocessors = [{"name": "image_float01", "args": {}}]

        return _apply_preprocessors(train, test, meta, preprocessors)

    # ---- SKLEARN ----
    if key in SKLEARN_DATASETS:
        task = kwargs.get("task", SKLEARN_DEFAULT_TASK.get(key, "classification"))
        train, test, meta = load_sklearn_source(
            key,
            task=task,
            test_size=kwargs.get("test_size", 0.2),
            seed=kwargs.get("seed", 42),
        )

        if preprocessors is None:
            preprocessors = [{
                "name": "tabular_scaling",
                "args": {
                    "scaler": kwargs.get("scaler", "standard"),
                    "y_standardize": kwargs.get("y_standardize", True),
                }
            }]

        return _apply_preprocessors(train, test, meta, preprocessors)

    # ---- CSV ----
    if key == "csv":
        train, test, meta = load_csv_source(
            csv_path=kwargs["csv_path"],
            target=kwargs["target"],
            task=kwargs.get("task", "regression"),
            test_size=kwargs.get("test_size", 0.2),
            seed=kwargs.get("seed", 42),
        )

        if preprocessors is None:
            preprocessors = [{
                "name": "tabular_scaling",
                "args": {
                    "scaler": kwargs.get("scaler", "standard"),
                    "y_standardize": kwargs.get("y_standardize", True),
                }
            }]

        return _apply_preprocessors(train, test, meta, preprocessors)

    # ---- HUGGINGFACE ----
    if key in ("hf", "huggingface"):
        # Source: raw HF datasets
        train, test, meta = load_huggingface_source(**kwargs)

        # Default: HF text preprocessor always tokenises
        if preprocessors is None:
            preprocessors = [{
                "name": "hf",
                "args": {k: v for k, v in kwargs.items() if k != "preprocessors"},
            }]

        return _apply_preprocessors(train, test, meta, preprocessors)

    raise KeyError(
        f"Unknown dataset '{name}'. Choices: {list(KERAS_DATASETS) + list(SKLEARN_DATASETS) + ['csv', 'hf']}"
    )