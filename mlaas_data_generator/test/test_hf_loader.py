import numpy as np

def validate_loaded(train, test, meta, name):
    x_train, y_train = train
    x_test, y_test = test

    assert isinstance(x_train, (list, tuple)), f"{name}: x_train should be list/tuple, got {type(x_train)}"
    assert isinstance(x_test, (list, tuple)), f"{name}: x_test should be list/tuple, got {type(x_test)}"

    assert isinstance(y_train, np.ndarray), f"{name}: y_train should be np.ndarray"
    assert isinstance(y_test, np.ndarray), f"{name}: y_test should be np.ndarray"

    assert y_train.ndim == 1, f"{name}: y_train should be 1D"
    assert y_test.ndim == 1, f"{name}: y_test should be 1D"

    assert len(x_train) == len(y_train), f"{name}: len mismatch train"
    assert len(x_test) == len(y_test), f"{name}: len mismatch test"

    assert isinstance(x_train[0], str), f"{name}: x_train[0] not str"

    assert "task_type" in meta
    assert "num_classes" in meta

    print(f"OK: {name} | train={len(x_train)} test={len(x_test)} classes={meta['num_classes']}")

def run_suite(load_dataset_fn):
    # Curated set: covers common split/column edge cases
    cases = [
        # Has train/test, text column is 'text'
        dict(title="imdb", dataset_name="imdb", text_column="text", label_column="label"),

        # Often train/validation (depending on dataset)
        dict(title="sst2", dataset_name="glue", dataset_config="sst2", text_column="sentence", label_column="label"),

        # Different text field
        dict(title="ag_news", dataset_name="ag_news", text_column="text", label_column="label"),

        # Another variant field name
        dict(title="yelp_polarity", dataset_name="yelp_polarity", text_column="text", label_column="label"),
    ]

    for c in cases:
        (train, test, meta) = load_dataset_fn(
            "hf",
            dataset_name=c["dataset_name"],
            dataset_config=c.get("dataset_config"),
            text_column=c["text_column"],
            label_column=c["label_column"],
            max_samples=2000,
            seed=42,
        )
        validate_loaded(train, test, meta, c["title"])

if __name__ == "__main__":
    # Import your project's loader
    from ..data.loaders.master_loader import load_dataset
    run_suite(load_dataset)